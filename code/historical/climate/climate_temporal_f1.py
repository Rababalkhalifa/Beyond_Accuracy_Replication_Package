r"""Check Climate stance Macro-F1 across periods using saved prediction rows.

PowerShell (after placing this script beside the ZIP):
    python .\climate_temporal_f1.py .\completed_prediction_audit.zip --csv .\climate_temporal_f1.csv

Reads saved results only. Delta is within-period Macro-F1 minus target-period
Macro-F1, paired by sampling seed. Positive delta indicates lower target F1.
"""

import argparse
import csv
import io
import statistics
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

SEEDS = (42, 43, 44, 45, 46)
PERIODS = ("within", "short", "long")
LABELS = ("BELIEVER", "DENIER")
MODELS = ("distilbert", "tfidf_lr", "phi3.5:latest", "qwen2.5:3b-instruct")


def files(source):
    if not source.exists():
        raise FileNotFoundError(f"No such file or directory: {source}")
    if source.is_dir():
        prefix = "predictions/" if source.name.lower() == "predictions" else ""
        for p in source.rglob("*.csv"):
            yield prefix + p.relative_to(source).as_posix(), p.read_bytes()
    elif zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as z:
            for info in z.infolist():
                if info.filename.lower().endswith(".csv"):
                    yield info.filename.replace("\\", "/"), z.read(info)
    else:
        raise ValueError("Input must be a ZIP archive or extracted directory")


def selected(name):
    parts = name.replace("\\", "/").split("/")
    try:
        sub = parts[parts.index("predictions") + 1:]
    except ValueError:
        return None
    if len(sub) < 5 or sub[-3] != "climate" or not sub[-2].startswith("seed_"):
        return None
    period = sub[-1].removesuffix("_n500_per_label.csv")
    if sub[-1] != f"{period}_n500_per_label.csv" or period not in PERIODS:
        return None
    try:
        seed = int(sub[-2][5:])
    except ValueError:
        return None
    if seed not in SEEDS:
        return None
    if sub[0] == "adapted" and len(sub) == 5:
        return "adapted", period, seed
    if sub[0] == "frozen" and len(sub) == 6 and sub[2] == "p3_json_strict":
        return "frozen", period, seed
    return None


def macro_f1(rows):
    scores = []
    for label in LABELS:
        tp = sum(r["true_label"] == label and r["predicted_label"] == label for r in rows)
        fp = sum(r["true_label"] != label and r["predicted_label"] == label for r in rows)
        fn = sum(r["true_label"] == label and r["predicted_label"] != label for r in rows)
        scores.append(2 * tp / (2 * tp + fp + fn))
    return statistics.mean(scores)


def calculate(source):
    results = defaultdict(dict)
    identities = {}
    archived = {}
    for name, raw in files(source):
        rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
        if name.rsplit("/", 1)[-1] == "metrics_per_condition.csv":
            for r in rows:
                if (r.get("dataset"), r.get("n_per_label")) != ("climate", "500"):
                    continue
                if r.get("model") not in MODELS or r.get("period") not in PERIODS:
                    continue
                if r.get("prompt_id") not in ("", "p3_json_strict"):
                    continue
                key = (r["model"], r["period"], int(r["seed"]))
                if key in archived:
                    raise ValueError(f"Duplicate archive row: {key}")
                archived[key] = r
        condition = selected(name)
        if condition is None:
            continue
        family, period, seed = condition
        if len(rows) != 1000:
            raise ValueError(f"Expected 1000 rows in {name}; found {len(rows)}")
        model = rows[0]["model"]
        if model not in MODELS:
            continue
        key = (model, period, seed)
        if key in results:
            raise ValueError(f"Duplicate prediction CSV: {key}")
        prompt = "p3_json_strict" if family == "frozen" else ""
        if any((r["dataset"], r["task_type"], r["period"], r["seed"],
                r["n_per_label"], r["family"], r["model"], r["prompt_id"],
                r["status"], r["abstain"].lower()) !=
               ("climate", "climate_stance", period, str(seed), "500",
                family, model, prompt, "ok", "false") for r in rows):
            raise ValueError(f"Wrong condition, invalid response, or abstention: {name}")
        if Counter(r["true_label"] for r in rows) != Counter({"BELIEVER": 500, "DENIER": 500}):
            raise ValueError(f"Unexpected class balance: {name}")
        if any(r["predicted_label"] not in LABELS for r in rows):
            raise ValueError(f"Unexpected predicted label: {name}")
        ids = tuple((r["example_id"], r["text_sha256"], r["true_label"]) for r in rows)
        if len(set(ids)) != 1000:
            raise ValueError(f"Repeated example: {name}")
        pair = (period, seed)
        if pair in identities and ids != identities[pair]:
            raise ValueError(f"Models have different examples/order within {pair}")
        identities[pair] = ids
        results[key] = macro_f1(rows)
    expected = {(m, p, s) for m in MODELS for p in PERIODS for s in SEEDS}
    if expected != set(results):
        raise ValueError(f"Missing prediction conditions: {sorted(expected - set(results))}")
    if archived:
        if expected - set(archived):
            raise ValueError(f"Missing archive metrics: {sorted(expected - set(archived))}")
        for key in expected:
            if abs(results[key] - float(archived[key]["macro_f1"])) > 1e-10:
                raise ValueError(f"Archived F1 does not match predictions: {key}")
    return results, bool(archived)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Completed prediction ZIP or extracted directory")
    parser.add_argument("--csv", type=Path, help="Optional detailed summary CSV")
    args = parser.parse_args()
    try:
        results, compared = calculate(args.input)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        parser.exit(2, f"Verification failed: {exc}\n")
    print("Climate stance | 500 per class | canonical P3 for frozen models")
    print("Model                       Within F1  Short F1   Long F1   Delta short Delta long")
    summary = []
    for model in MODELS:
        by_period = {p: [results[model, p, s] for s in SEEDS] for p in PERIODS}
        short_delta = [results[model, "within", s] - results[model, "short", s] for s in SEEDS]
        long_delta = [results[model, "within", s] - results[model, "long", s] for s in SEEDS]
        out = {"model": model}
        for p in PERIODS:
            out[f"mean_{p}_f1"] = statistics.mean(by_period[p])
            out[f"sd_{p}_f1"] = statistics.stdev(by_period[p])
            out.update({f"seed_{s}_{p}_f1": results[model, p, s] for s in SEEDS})
        for p, values in (("short", short_delta), ("long", long_delta)):
            out[f"mean_delta_{p}"] = statistics.mean(values)
            out[f"sd_delta_{p}"] = statistics.stdev(values)
            out.update({f"seed_{s}_delta_{p}": v for s, v in zip(SEEDS, values)})
        summary.append(out)
        print(f"{model:27} {out['mean_within_f1']:9.4f} {out['mean_short_f1']:9.4f} "
              f"{out['mean_long_f1']:9.4f} {out['mean_delta_short']:11.4f} "
              f"{out['mean_delta_long']:10.4f}")
    print("Verified: 4 models × 3 periods × 5 seeds; matched examples within each period/seed.")
    print("Per-condition archived Macro-F1: " + ("all matched" if compared else "not present"))
    print("These data cannot independently verify that supervised weights were fixed across periods.")
    if args.csv:
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
        print(f"Saved summary: {args.csv}")


if __name__ == "__main__":
    main()
