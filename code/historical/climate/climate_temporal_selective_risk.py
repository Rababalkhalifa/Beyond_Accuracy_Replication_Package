"""Derive matched-coverage selective risk for canonical Climate P3 predictions.

Read-only, reviewer-motivated analysis. Per seed and period, retain the 80% most
confident predictions. At the cutoff, tied scores are fractionally included,
giving the expected risk under random tie-breaking without arbitrary row order.
This confidence-threshold analysis is separate from explicit P5 abstention.
"""

import argparse
import csv
import io
import math
import statistics
import zipfile
from pathlib import Path


MODELS = (
    ("distilbert", "adapted", None, "distilbert"),
    ("tfidf_lr", "adapted", None, "tfidf_lr"),
    ("phi3.5:latest", "frozen", "p3_json_strict", "phi3.5_latest"),
    ("qwen2.5:3b-instruct", "frozen", "p3_json_strict", "qwen2.5_3b-instruct"),
)
PERIODS = ("within", "short", "long")
SEEDS = range(42, 47)


def read_csv(archive, ending):
    paths = [name for name in archive.namelist()
             if name.replace("\\", "/").endswith(ending)]
    if len(paths) != 1:
        raise ValueError(f"Expected one {ending}; found {len(paths)}")
    with archive.open(paths[0]) as stream:
        return list(csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8-sig", newline="")))


def risk_at_coverage(rows, kept):
    scores = [float(r["confidence"]) for r in rows]
    errors = [float(r["true_label"] != r["predicted_label"]) for r in rows]
    if not all(math.isfinite(score) and 0 <= score <= 1 for score in scores):
        raise ValueError("Nonfinite or out-of-range confidence")
    threshold = sorted(scores, reverse=True)[kept - 1]
    above = [i for i, score in enumerate(scores) if score > threshold]
    tied = [i for i, score in enumerate(scores) if score == threshold]
    proportion = (kept - len(above)) / len(tied)
    error_sum = (sum(errors[i] for i in above) +
                 proportion * sum(errors[i] for i in tied))
    return error_sum / kept, threshold, len(tied), proportion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="completed_prediction_audit.zip")
    parser.add_argument("--csv", type=Path, help="Optional per-seed audit CSV")
    args = parser.parse_args()
    if not args.archive.is_file():
        parser.error(f"Archive not found: {args.archive}")

    output = []
    with zipfile.ZipFile(args.archive) as archive:
        metrics = read_csv(archive, "metrics_per_condition.csv")
        for seed in SEEDS:
            for period in PERIODS:
                reference = None
                for model, family, prompt, slug in MODELS:
                    components = ["predictions", family, slug]
                    if prompt:
                        components.append(prompt)
                    components += ["climate", f"seed_{seed}",
                                   f"{period}_n500_per_label.csv"]
                    rows = read_csv(archive, "/".join(components))
                    identities = {(r["example_id"], r["text_sha256"], r["true_label"])
                                  for r in rows}
                    if len(rows) != 1000 or len(identities) != 1000:
                        raise ValueError(f"Expected 1000 unique examples: {model} {seed} {period}")
                    if reference is None:
                        reference = identities
                    elif reference != identities:
                        raise ValueError(f"Unmatched examples: {model} {seed} {period}")
                    if any(r["status"] != "ok" or not r["predicted_label"] for r in rows):
                        raise ValueError(f"Invalid canonical row: {model} {seed} {period}")
                    source = "predicted_probability" if family == "adapted" else "llm_self_reported"
                    if any(r["confidence_source"] != source for r in rows):
                        raise ValueError(f"Unexpected confidence source: {model} {seed} {period}")
                    saved = [r for r in metrics if r["dataset"] == "climate"
                             and r["model"] == model and r["family"] == family
                             and r["prompt_id"] == (prompt or "")
                             and int(r["seed"]) == seed and r["period"] == period
                             and int(r["n_per_label"]) == 500]
                    if len(saved) != 1:
                        raise ValueError(f"Missing or duplicate archived metric: {model} {seed} {period}")
                    risk_100 = sum(r["true_label"] != r["predicted_label"] for r in rows) / 1000
                    if not math.isclose(risk_100, float(saved[0]["error_rate_valid"]), abs_tol=1e-12):
                        raise ValueError(f"Full-coverage risk mismatch: {model} {seed} {period}")
                    risk_80, threshold, n_tied, included = risk_at_coverage(rows, 800)
                    output.append(dict(model=model, family=family, period=period, seed=seed,
                                       confidence_source=source, risk_at_80=risk_80,
                                       risk_at_100=risk_100, threshold_80=threshold,
                                       tied_at_threshold=n_tied, tie_inclusion_fraction=included))

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(output[0]))
            writer.writeheader()
            writer.writerows(output)
    print("Climate stance | canonical P3 | n500 per class | five-seed mean risk at matched coverage")
    print(f"{'Model':25} {'80% within':>11} {'80% short':>11} {'80% long':>11} {'100% within':>12}")
    for model, _, _, _ in MODELS:
        at80 = {period: [r["risk_at_80"] for r in output
                         if r["model"] == model and r["period"] == period]
                for period in PERIODS}
        at100 = [r["risk_at_100"] for r in output
                 if r["model"] == model and r["period"] == "within"]
        values = [statistics.mean(at80[p]) for p in PERIODS] + [statistics.mean(at100)]
        print(f"{model:25} {values[0]:11.4f} {values[1]:11.4f} {values[2]:11.4f} {values[3]:12.4f}")
    print("Verified 4 models × 3 periods × 5 seeds; matched examples and archived full-coverage error rates.")
    print("80% selection is derived from P3 confidence, not a saved run metric; ties use fractional inclusion.")
    print("Supervised probabilities and LLM self-reported confidence are distinct score sources.")
    if args.csv:
        print(f"Saved per-seed analysis: {args.csv}")


if __name__ == "__main__":
    main()
