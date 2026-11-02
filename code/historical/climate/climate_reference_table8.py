r"""Independently verify Climate reference rows for Table 8 from saved predictions.

Run in PowerShell from C:\Research\Beyond_Accuracy_Reproduction:
    python .\climate_reference_table8.py .\completed_prediction_audit.zip --csv .\climate_table8_check.csv

Reads a ZIP archive or an extracted directory. No model is trained or queried.
Five sampling seeds (42-46), 500 examples per stance class, and the canonical
P3 prompt for frozen models are required. The optional archived metrics are
compared against recomputation for each model and seed.
"""

import argparse
import csv
import io
import math
import statistics
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


SEEDS = (42, 43, 44, 45, 46)
LABELS = ("BELIEVER", "DENIER")
MODELS = ("distilbert", "tfidf_lr", "phi3.5:latest", "qwen2.5:3b-instruct")
FIELDS = ("accuracy", "macro_f1", "ece", "latency_mean_sec")


def csv_files(source):
    if not source.exists():
        raise FileNotFoundError(f"No such file or directory: {source}")
    if source.is_dir():
        prefix = "predictions/" if source.name.lower() == "predictions" else ""
        for path in source.rglob("*.csv"):
            yield prefix + path.relative_to(source).as_posix(), path.read_bytes()
    elif zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            for item in archive.infolist():
                if item.filename.lower().endswith(".csv"):
                    yield item.filename.replace("\\", "/"), archive.read(item)
    else:
        raise ValueError("Input must be a ZIP archive or an extracted directory")


def condition(path):
    parts = path.replace("\\", "/").split("/")
    try:
        base = parts.index("predictions")
    except ValueError:
        return None
    sub = parts[base + 1:]
    if len(sub) < 5:
        return None
    if sub[-3] != "climate" or sub[-1] != "within_n500_per_label.csv":
        return None
    if not sub[-2].startswith("seed_"):
        return None
    try:
        seed = int(sub[-2][5:])
    except ValueError:
        return None
    if seed not in SEEDS:
        return None
    if sub[0] == "adapted" and len(sub) == 5:
        return sub[0], seed
    if sub[0] == "frozen" and len(sub) == 6 and sub[2] == "p3_json_strict":
        return sub[0], seed
    return None


def metrics(rows):
    correct = [r["true_label"] == r["predicted_label"] for r in rows]
    accuracy = sum(correct) / len(rows)
    f1 = []
    for label in LABELS:
        tp = sum(c and r["true_label"] == label for r, c in zip(rows, correct))
        fp = sum(r["true_label"] != label and r["predicted_label"] == label for r in rows)
        fn = sum(r["true_label"] == label and r["predicted_label"] != label for r in rows)
        f1.append(2 * tp / (2 * tp + fp + fn))
    confidence = [float(r["confidence"]) for r in rows]
    latency = [float(r["latency_sec"]) for r in rows]
    if any(not math.isfinite(x) or not 0 <= x <= 1 for x in confidence):
        raise ValueError("Missing or out-of-range confidence")
    if any(not math.isfinite(x) or x < 0 for x in latency):
        raise ValueError("Missing or invalid latency")
    # Ten equal-width bins. Multiplication reproduces numpy.linspace(0,1,11)
    # at floating-point boundaries, including 0.30000000000000004 and
    # 0.7000000000000001; the last bin includes 1.0.
    edges = [i * 0.1 for i in range(11)]
    ece = 0.0
    for b in range(10):
        selected = [i for i, x in enumerate(confidence)
                    if edges[b] <= x < edges[b+1] or (b == 9 and x == 1.0)]
        if selected:
            ece += len(selected) / len(rows) * abs(
                sum(correct[i] for i in selected) / len(selected)
                - sum(confidence[i] for i in selected) / len(selected)
            )
    return dict(accuracy=accuracy, macro_f1=statistics.mean(f1), ece=ece,
                latency_mean_sec=statistics.mean(latency))


def calculate(source):
    results = defaultdict(dict)
    archived = {}
    paired = {}
    for name, raw in csv_files(source):
        rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
        if name.rsplit("/", 1)[-1] == "metrics_per_condition.csv":
            for r in rows:
                if (r.get("dataset"), r.get("period"), r.get("n_per_label")) != ("climate", "within", "500"):
                    continue
                if r.get("model") not in MODELS or r.get("prompt_id") not in ("", "p3_json_strict"):
                    continue
                key = (r["model"], int(r["seed"]))
                if key in archived:
                    raise ValueError(f"Duplicate archived metric row for {key}")
                archived[key] = r
        selected = condition(name)
        if selected is None:
            continue
        family, seed = selected
        if len(rows) != 1000:
            raise ValueError(f"Expected 1000 rows in {name}, got {len(rows)}")
        model = rows[0]["model"]
        if model not in MODELS:
            continue
        key = (model, seed)
        if seed in results[model]:
            raise ValueError(f"Duplicate prediction file for {key}")
        prompt = "p3_json_strict" if family == "frozen" else ""
        expected_source = "llm_self_reported" if family == "frozen" else "predicted_probability"
        for r in rows:
            if (r["dataset"], r["task_type"], r["period"], r["n_per_label"],
                r["family"], r["model"], r["prompt_id"], r["seed"],
                r["status"], r["abstain"].lower(), r["confidence_source"]) != (
                "climate", "climate_stance", "within", "500", family, model,
                prompt, str(seed), "ok", "false", expected_source):
                raise ValueError(f"Wrong condition, status, or confidence source in {name}")
        if Counter(r["true_label"] for r in rows) != Counter({"BELIEVER": 500, "DENIER": 500}):
            raise ValueError(f"Unexpected stance balance in {name}")
        if any(r["predicted_label"] not in LABELS for r in rows):
            raise ValueError(f"Unexpected predicted stance in {name}")
        identities = tuple((r["example_id"], r["text_sha256"], r["true_label"]) for r in rows)
        if len(set(identities)) != 1000:
            raise ValueError(f"Duplicate example in {name}")
        if seed in paired and paired[seed] != identities:
            raise ValueError(f"Models do not share ordered examples for seed {seed}")
        paired[seed] = identities
        results[model][seed] = metrics(rows)
    expected = {(m, s) for m in MODELS for s in SEEDS}
    found = {(m, s) for m, seeds in results.items() for s in seeds}
    if expected != found:
        raise ValueError(f"Missing prediction files: {sorted(expected - found)}")
    if archived:
        missing = expected - set(archived)
        if missing:
            raise ValueError(f"Missing archived metric rows: {sorted(missing)}")
        for model, seed in sorted(expected):
            for field in FIELDS:
                recomputed = results[model][seed][field]
                recorded = float(archived[model, seed][field])
                if not math.isclose(recomputed, recorded, rel_tol=0, abs_tol=1e-9):
                    raise ValueError(f"Archive mismatch {model} seed {seed} {field}: "
                                     f"recomputed={recomputed}, archived={recorded}")
    return results, bool(archived)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Completed prediction ZIP or extracted directory")
    parser.add_argument("--csv", type=Path, help="Optional CSV with means, sample SDs, and seed values")
    args = parser.parse_args()
    try:
        results, compared = calculate(args.input)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        parser.exit(2, f"Verification failed: {exc}\n")
    summary = []
    print("Climate stance | within/reference | 500 per class | frozen canonical P3")
    print("Five-seed mean (sample SD):")
    for model in MODELS:
        out = {"model": model}
        for field in FIELDS:
            x = [results[model][seed][field] for seed in SEEDS]
            out[f"mean_{field}"] = statistics.mean(x)
            out[f"sd_{field}"] = statistics.stdev(x)
            out.update({f"seed_{seed}_{field}": results[model][seed][field] for seed in SEEDS})
        summary.append(out)
        print(f"{model:24} accuracy {out['mean_accuracy']:.4f} ± {out['sd_accuracy']:.4f} | "
              f"Macro-F1 {out['mean_macro_f1']:.4f} ± {out['sd_macro_f1']:.4f} | "
              f"ECE {out['mean_ece']:.4f} ± {out['sd_ece']:.4f} | "
              f"recorded latency {out['mean_latency_mean_sec']:.6f} ± "
              f"{out['sd_latency_mean_sec']:.6f} s")
    print("Verified: 4 models × 5 seeds; 1000 matched examples per seed.")
    print("Per-seed archived metrics: " + ("all matched" if compared else "not present in input"))
    print("ECE uses predicted probabilities for supervised models and self-reported confidence for LLMs.")
    print("Latency reproduces saved rows; it does not verify a controlled timing protocol.")
    if args.csv:
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
        print(f"Saved summary: {args.csv}")


if __name__ == "__main__":
    main()
