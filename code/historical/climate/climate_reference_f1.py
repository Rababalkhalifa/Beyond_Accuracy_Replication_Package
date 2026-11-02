"""Recompute Climate reference-period Macro-F1 from saved prediction CSVs.

Usage:
    python climate_reference_f1.py completed_prediction_audit.zip
    python climate_reference_f1.py path/to/extracted_archive --csv climate_f1.csv

Reads existing predictions only. Frozen models use the canonical P3 prompt.
"""

import argparse
import csv
import io
import statistics
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


SEEDS = (42, 43, 44, 45, 46)
LABELS = ("BELIEVER", "DENIER")
MODELS = (
    "distilbert",
    "tfidf_lr",
    "phi3.5:latest",
    "qwen2.5:3b-instruct",
)


def prediction_files(source):
    """Yield (normalized relative name, CSV bytes) from a ZIP or directory."""
    if not source.exists():
        raise FileNotFoundError(
            f"{source} does not exist. Replace the example path with the real ZIP "
            "or predictions-folder path on your computer."
        )
    if source.is_dir():
        prefix = "predictions/" if source.name.lower() == "predictions" else ""
        for path in source.rglob("*.csv"):
            yield prefix + path.relative_to(source).as_posix(), path.read_bytes()
    elif zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            for info in archive.infolist():
                name = info.filename.replace("\\", "/")  # Windows ZIP paths
                if name.lower().endswith(".csv"):
                    yield name, archive.read(info)
    else:
        raise ValueError("Input must be an extracted directory or a ZIP archive")


def selected_condition(name):
    parts = name.replace("\\", "/").split("/")
    if len(parts) < 6 or parts[-3] != "climate":
        return None
    if parts[-1] != "within_n500_per_label.csv" or not parts[-2].startswith("seed_"):
        return None
    try:
        seed = int(parts[-2][5:])
        base = parts.index("predictions")
    except (ValueError, IndexError):
        return None
    if seed not in SEEDS:
        return None
    family = parts[base + 1]
    if family == "adapted":
        return (family, seed)
    if family == "frozen" and parts[-4] == "p3_json_strict":
        return (family, seed)
    return None


def macro_f1(rows):
    scores = []
    for label in LABELS:
        tp = sum(r["true_label"] == label and r["predicted_label"] == label for r in rows)
        fp = sum(r["true_label"] != label and r["predicted_label"] == label for r in rows)
        fn = sum(r["true_label"] == label and r["predicted_label"] != label for r in rows)
        scores.append(2 * tp / (2 * tp + fp + fn))
    return sum(scores) / len(scores)


def calculate(source):
    results = defaultdict(dict)
    paired_examples = {}
    for name, raw in prediction_files(source):
        condition = selected_condition(name)
        if condition is None:
            continue
        family, seed = condition
        rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
        if len(rows) != 1000:
            raise ValueError(f"Expected 1000 examples in {name}; found {len(rows)}")
        model = rows[0]["model"]
        if model not in MODELS:
            continue
        key = (model, seed)
        if seed in results[model]:
            raise ValueError(f"Duplicate result for {key}: {name}")
        expected_prompt = "p3_json_strict" if family == "frozen" else ""
        if any(
            r["dataset"] != "climate"
            or r["task_type"] != "climate_stance"
            or r["period"] != "within"
            or r["n_per_label"] != "500"
            or r["family"] != family
            or r["model"] != model
            or r["seed"] != str(seed)
            or r["prompt_id"] != expected_prompt
            or r["status"] != "ok"
            or r["abstain"].lower() != "false"
            for r in rows
        ):
            raise ValueError(f"Wrong condition, invalid response, or abstention in {name}")
        if Counter(r["true_label"] for r in rows) != Counter({"BELIEVER": 500, "DENIER": 500}):
            raise ValueError(f"Gold labels are not balanced in {name}")
        if any(r["predicted_label"] not in LABELS for r in rows):
            raise ValueError(f"Prediction outside stance labels in {name}")
        identities = tuple(
            (r["example_id"], r["text_sha256"], r["true_label"]) for r in rows
        )
        if len(set(identities)) != len(identities):
            raise ValueError(f"Repeated example identities in {name}")
        if seed in paired_examples and identities != paired_examples[seed]:
            raise ValueError(f"Models use different examples or order for seed {seed}: {name}")
        paired_examples[seed] = identities
        results[model][seed] = macro_f1(rows)

    expected = {(model, seed) for model in MODELS for seed in SEEDS}
    observed = {(model, seed) for model, values in results.items() for seed in values}
    if missing := expected - observed:
        raise ValueError(f"Missing model/seed results: {sorted(missing)}")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Prediction ZIP or extracted directory")
    parser.add_argument("--csv", type=Path, help="Optional summary CSV output")
    args = parser.parse_args()
    try:
        results = calculate(args.input)
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(2, f"Input error: {exc}\n")
    summary = []
    print("Climate stance | within/reference | 500 per class | canonical P3")
    print("Macro-F1 across paired seeds 42-46 (sample SD)")
    for model in MODELS:
        values = [results[model][seed] for seed in SEEDS]
        row = {"model": model, "mean_macro_f1": statistics.mean(values),
               "seed_sd": statistics.stdev(values)}
        row.update({f"seed_{seed}": results[model][seed] for seed in SEEDS})
        summary.append(row)
        print(f"{model:24s} {row['mean_macro_f1']:.4f} ± {row['seed_sd']:.4f}")
    print("Verified: 4 models × 5 seeds; 1000 matched examples per seed.")
    if args.csv:
        with args.csv.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=summary[0].keys())
            writer.writeheader()
            writer.writerows(summary)
        print(f"Saved summary: {args.csv}")


if __name__ == "__main__":
    main()
