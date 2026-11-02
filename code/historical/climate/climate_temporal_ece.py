"""Audit canonical P3 Climate ECE across periods from archived predictions.

Read-only. Uses the runner's 10-bin ECE definition and checks each seed's
result against metrics_per_condition.csv. Different confidence sources for
supervised models and frozen LLMs must be interpreted separately.
"""

import argparse
import csv
import io
import math
import statistics
import zipfile
from pathlib import Path

import numpy as np


MODELS = (
    ("distilbert", "adapted", None),
    ("tfidf_lr", "adapted", None),
    ("phi3.5:latest", "frozen", "p3_json_strict"),
    ("qwen2.5:3b-instruct", "frozen", "p3_json_strict"),
)
PERIODS = ("within", "short", "long")
SEEDS = range(42, 47)


def read_csv(archive, ending):
    names = [name for name in archive.namelist()
             if name.replace("\\", "/").endswith(ending)]
    if len(names) != 1:
        raise ValueError(f"Expected one file ending in {ending}, found {len(names)}")
    with archive.open(names[0]) as stream:
        return list(csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8-sig", newline="")))


def predictions(archive, model, family, prompt, seed, period):
    model_slug = {
        "distilbert": "distilbert",
        "tfidf_lr": "tfidf_lr",
        "phi3.5:latest": "phi3.5_latest",
        "qwen2.5:3b-instruct": "qwen2.5_3b-instruct",
    }[model]
    components = ["predictions", family, model_slug]
    if prompt:
        components.append(prompt)
    components += ["climate", f"seed_{seed}", f"{period}_n500_per_label.csv"]
    return read_csv(archive, "/".join(components))


def ece(rows):
    valid = [r for r in rows if r["status"] == "ok" and r["predicted_label"]]
    correct = np.asarray([r["true_label"] == r["predicted_label"] for r in valid], dtype=float)
    confidence = np.asarray([float(r["confidence"]) if r["confidence"] else math.nan
                             for r in valid], dtype=float)
    finite = np.isfinite(confidence)
    if not finite.any():
        return math.nan, len(valid)
    confidence = np.clip(confidence[finite], 0.0, 1.0)
    correct = correct[finite]
    edges = np.linspace(0.0, 1.0, 11)
    result = 0.0
    for i in range(10):
        mask = ((confidence >= edges[i]) &
                (confidence <= edges[i + 1] if i == 9 else confidence < edges[i + 1]))
        if mask.any():
            result += float(mask.sum() / len(confidence)) * abs(
                float(correct[mask].mean()) - float(confidence[mask].mean()))
    return result, len(valid)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="completed_prediction_audit.zip")
    parser.add_argument("--csv", type=Path, help="Optional per-seed output CSV")
    args = parser.parse_args()
    if not args.archive.is_file():
        parser.error(f"Archive not found: {args.archive}")

    with zipfile.ZipFile(args.archive) as archive:
        saved = read_csv(archive, "metrics_per_condition.csv")
        results = []
        for seed in SEEDS:
            for period in PERIODS:
                reference = None
                for model, family, prompt in MODELS:
                    rows = predictions(archive, model, family, prompt, seed, period)
                    if len(rows) != 1000:
                        raise ValueError(f"{model} seed {seed} {period}: expected 1000 rows")
                    identity = {(r["example_id"], r["text_sha256"], r["true_label"])
                                for r in rows}
                    if len(identity) != 1000:
                        raise ValueError(f"{model} seed {seed} {period}: duplicate identities")
                    if reference is None:
                        reference = identity
                    elif reference != identity:
                        raise ValueError(f"Unmatched evaluation examples: seed {seed} {period} {model}")
                    source = "predicted_probability" if family == "adapted" else "llm_self_reported"
                    if any(r["confidence_source"] != source for r in rows):
                        raise ValueError(f"Unexpected confidence source: {model} seed {seed} {period}")
                    calculated, valid = ece(rows)
                    matches = [r for r in saved if r["dataset"] == "climate"
                               and r["model"] == model and r["family"] == family
                               and r["prompt_id"] == (prompt or "")
                               and int(r["seed"]) == seed and r["period"] == period
                               and int(r["n_per_label"]) == 500]
                    if len(matches) != 1:
                        raise ValueError(f"Expected one saved metric: {model} seed {seed} {period}")
                    archived = matches[0]
                    if (valid != int(archived["valid_rows"]) or
                            not math.isclose(calculated, float(archived["ece"]), abs_tol=1e-10)):
                        raise ValueError(f"ECE differs from saved metric: {model} seed {seed} {period}")
                    results.append(dict(model=model, family=family, seed=seed,
                                        period=period, valid_rows=valid,
                                        confidence_source=source, ece=calculated))

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
    print("Climate stance | canonical P3 | n500 per class | five-seed mean ECE (sample SD)")
    print(f"{'Model':25} {'Within':>14} {'Short':>14} {'Long':>14} {'Short−within':>14} {'Long−within':>14}")
    for model, family, prompt in MODELS:
        series = {period: [r["ece"] for r in results if r["model"] == model and r["period"] == period]
                  for period in PERIODS}
        means = {period: statistics.mean(series[period]) for period in PERIODS}
        parts = [f"{means[p]:.4f}±{statistics.stdev(series[p]):.4f}" for p in PERIODS]
        print(f"{model:25} {parts[0]:>14} {parts[1]:>14} {parts[2]:>14} "
              f"{means['short'] - means['within']:>14.4f} {means['long'] - means['within']:>14.4f}")
    print("Verified 4 models × 3 periods × 5 seeds: matched examples and saved per-condition ECE.")
    print("Supervised ECE uses predicted probabilities; frozen LLM ECE uses self-reported confidence.")
    if args.csv:
        print(f"Saved per-seed ECE: {args.csv}")


if __name__ == "__main__":
    main()
