r"""Derive canonical P3 Yelp selective risk at matched 80% coverage.

Requires yelp_temporal_ece.py beside this file. Reads the same four archived
ZIPs; compares each full-coverage risk with archived accuracy. At a cutoff tie,
fractionally include tied scores to estimate risk under random tie-breaking.
This reviewer-motivated analysis is separate from explicit P5 abstention.
"""

import argparse
import csv
import math
import statistics
from collections import Counter
from pathlib import Path

from yelp_temporal_ece import archive, rows_at, MODELS, FOLDERS, STEMS, PERIODS, SEEDS


def saved_accuracy(files, suffix, models):
    index = {}
    for row in rows_at(files, suffix):
        if row["model"] not in models or row["split"] not in PERIODS or row["prompt"] != "p3_json_strict":
            continue
        key = row["model"], int(row["eval_n_per_label"]), row["split"], int(row["seed"])
        if key in index:
            raise ValueError(f"Duplicate saved metric: {key}")
        index[key] = float(row["accuracy"])
    return index


def risk_at_coverage(rows, kept):
    scores = [float(row["confidence"]) for row in rows]
    errors = [float(row["gold_label"] != row["pred_label"]) for row in rows]
    if not all(math.isfinite(s) and 0 <= s <= 1 for s in scores):
        raise ValueError("Nonfinite or out-of-range confidence")
    threshold = sorted(scores, reverse=True)[kept - 1]
    higher = [i for i, value in enumerate(scores) if value > threshold]
    tied = [i for i, value in enumerate(scores) if value == threshold]
    fraction = (kept - len(higher)) / len(tied)
    risk = (sum(errors[i] for i in higher) +
            fraction * sum(errors[i] for i in tied)) / kept
    return risk, threshold, len(tied), fraction


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", type=Path, help="Find four Yelp ZIPs recursively")
    for arg in STEMS:
        p.add_argument("--" + arg.replace("_", "-"), type=Path)
    p.add_argument("--csv", type=Path, help="Optional per-seed output")
    args = p.parse_args()
    for arg, stem in STEMS.items():
        if getattr(args, arg) is None and args.source_dir:
            found = list(args.source_dir.rglob(stem + "*.zip"))
            if len(found) != 1:
                p.error(f"Expected one {stem}*.zip beneath {args.source_dir}; found {len(found)}")
            setattr(args, arg, found[0])
        if getattr(args, arg) is None:
            p.error(f"Supply --{arg.replace('_', '-')} or --source-dir")
    frozen, aligned, later, audit = (archive(getattr(args, arg)) for arg in STEMS)
    sup_accuracy = saved_accuracy(later, "later_run/metrics_per_seed.csv", MODELS[:2])
    llm_accuracy = saved_accuracy(audit, "metrics_frozen_partial.csv", MODELS[2:])
    results = []
    labeled_abstentions = 0
    for n in (500, 1000):
        for period in PERIODS:
            for seed in SEEDS:
                reference = None
                for model in MODELS:
                    if model in FOLDERS:
                        rows = rows_at(frozen, f"predictions/frozen/{FOLDERS[model]}/seed{seed}/"
                                               f"n{n}_{period}_p3_json_strict.csv")
                        if any(r["status"] != "ok" for r in rows):
                            raise ValueError(f"Non-OK frozen row: {model} n{n} {period} seed{seed}")
                        labeled_abstentions += sum(
                            r["abstain"].lower() == "true" and r["pred_label"] in ("NEG", "POS")
                            for r in rows)
                        saved = llm_accuracy
                    else:
                        folder = "later_distilbert" if model == "distilbert" else "aligned_tfidf"
                        rows = rows_at(aligned, f"{folder}/seed{seed}/n{n}_{period}.csv")
                        saved = sup_accuracy
                        if any(not math.isclose(float(r["confidence"]),
                                                max(float(r["prob_pos"]), 1 - float(r["prob_pos"])),
                                                abs_tol=1e-5) for r in rows):
                            raise ValueError(f"Supervised confidence mismatch: {model} n{n} {period} seed{seed}")
                    if (len(rows) != 2*n or
                            Counter(r["gold_label"] for r in rows) != {"NEG": n, "POS": n} or
                            any(r["pred_label"] not in ("NEG", "POS") for r in rows)):
                        raise ValueError(f"Invalid labels/counts: {model} n{n} {period} seed{seed}")
                    identity = {(r["example_id"], r["text_sha1"], r["gold_label"]) for r in rows}
                    if len(identity) != len(rows) or (reference is not None and reference != identity):
                        raise ValueError(f"Unmatched examples: {model} n{n} {period} seed{seed}")
                    reference = identity
                    risk100 = sum(r["gold_label"] != r["pred_label"] for r in rows) / len(rows)
                    key = model, n, period, seed
                    if key not in saved or not math.isclose(risk100, 1 - saved[key], abs_tol=1e-8):
                        raise ValueError(f"Full-coverage risk differs from archived accuracy: {key}")
                    risk80, cutoff, ties, fraction = risk_at_coverage(rows, 8*n//5)
                    results.append(dict(setting=f"Yelp n{n}", model=model, period=period,
                                        seed=seed, confidence_source=("predicted_probability" if model in MODELS[:2]
                                                                 else "llm_self_reported"),
                                        risk_at_80=risk80, risk_at_100=risk100,
                                        threshold_80=cutoff, tied_at_threshold=ties,
                                        tie_inclusion_fraction=fraction))

    print("Yelp sentiment | archived aligned cohort | canonical P3 | five-seed mean risk")
    print(f"{'Setting':11} {'Model':25} {'80% within':>11} {'80% short':>11} {'80% long':>11} {'100% within':>12}")
    for n in (500, 1000):
        for model in MODELS:
            means = [statistics.mean(r["risk_at_80"] for r in results
                                     if r["setting"] == f"Yelp n{n}" and r["model"] == model
                                     and r["period"] == period) for period in PERIODS]
            full = statistics.mean(r["risk_at_100"] for r in results
                                   if r["setting"] == f"Yelp n{n}" and r["model"] == model
                                   and r["period"] == "within")
            print(f"Yelp n{n:<4}  {model:25} {means[0]:11.4f} {means[1]:11.4f} {means[2]:11.4f} {full:12.4f}")
    print("Verified 2 settings × 4 models × 3 periods × 5 seeds: matched reviews and archived full-coverage accuracy.")
    if labeled_abstentions:
        print(f"Included {labeled_abstentions} P3 row(s) with abstain=true and a valid label, as in canonical scoring.")
    print("80% risk is derived from P3 confidence, separate from P5 abstention; ties use fractional inclusion.")
    print("Supervised probabilities and LLM self-reported confidence are distinct score sources.")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved per-seed analysis: {args.csv}")


if __name__ == "__main__":
    main()
