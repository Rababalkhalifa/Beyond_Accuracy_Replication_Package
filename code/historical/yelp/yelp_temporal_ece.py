r"""Verify canonical P3 Yelp ECE across periods on the archived aligned cohort.

PowerShell:
  python .\yelp_temporal_ece.py --source-dir C:\Research --csv .\yelp_temporal_ece.csv

Reads predictions only. Supervised rows come from yelp_prediction_alignment.zip;
frozen rows come from original_yelp_frozen_predictions.zip. Refuses mismatched
examples and checks every seed-level ECE against archived metrics.
"""

import argparse
import csv
import io
import math
import statistics
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np


SEEDS = range(42, 47)
PERIODS = ("within", "short", "long")
MODELS = ("distilbert", "tfidf_lr", "phi3.5:latest", "qwen2.5:3b-instruct")
FOLDERS = {"phi3.5:latest": "phi3.5_latest",
           "qwen2.5:3b-instruct": "qwen2.5_3b-instruct"}
STEMS = {"frozen": "original_yelp_frozen_predictions",
         "aligned": "yelp_prediction_alignment",
         "later": "code_and_remaining_results",
         "frozen_audit": "yelp_results_audit"}


def archive(path):
    if not zipfile.is_zipfile(path):
        raise ValueError(f"Missing or invalid ZIP: {path}")
    with zipfile.ZipFile(path) as z:
        return {name.replace("\\", "/"): z.read(name) for name in z.namelist()
                if name.lower().endswith(".csv")}


def rows_at(files, suffix):
    hits = [name for name in files if name.endswith(suffix)]
    if len(hits) != 1:
        raise ValueError(f"Expected one {suffix}; found {len(hits)}")
    return list(csv.DictReader(io.StringIO(files[hits[0]].decode("utf-8-sig"))))


def ece(rows):
    correct = np.asarray([r["gold_label"] == r["pred_label"] for r in rows], dtype=float)
    confidence = np.asarray([float(r["confidence"]) for r in rows], dtype=float)
    if not np.isfinite(confidence).all() or not ((confidence >= 0) & (confidence <= 1)).all():
        raise ValueError("Nonfinite or out-of-range confidence")
    edges = np.linspace(0.0, 1.0, 11)
    result = 0.0
    for i in range(10):
        lo, hi = edges[i], edges[i + 1]
        mask = (confidence >= lo) & (confidence <= hi if i == 9 else confidence < hi)
        if mask.any():
            result += float(mask.mean() * abs(correct[mask].mean() - confidence[mask].mean()))
    return result


def metrics(files, suffix, models):
    result = {}
    for r in rows_at(files, suffix):
        if r["model"] not in models or r["split"] not in PERIODS or r["prompt"] != "p3_json_strict":
            continue
        key = (r["model"], int(r["eval_n_per_label"]), r["split"], int(r["seed"]))
        if key in result:
            raise ValueError(f"Duplicate archived ECE metric: {key}")
        result[key] = float(r["ece"])
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", type=Path, help="Find the four Yelp ZIPs recursively")
    for arg in STEMS:
        p.add_argument("--" + arg.replace("_", "-"), type=Path)
    p.add_argument("--csv", type=Path, help="Optional per-seed ECE output")
    args = p.parse_args()
    for arg, stem in STEMS.items():
        if getattr(args, arg) is None and args.source_dir:
            hits = list(args.source_dir.rglob(stem + "*.zip"))
            if len(hits) != 1:
                p.error(f"Expected one {stem}*.zip beneath {args.source_dir}; found {len(hits)}")
            setattr(args, arg, hits[0])
        if getattr(args, arg) is None:
            p.error(f"Supply --{arg.replace('_', '-')} or --source-dir")
    frozen, aligned, later, audit = (archive(getattr(args, arg)) for arg in STEMS)
    saved_supervised = metrics(later, "later_run/metrics_per_seed.csv", MODELS[:2])
    saved_frozen = metrics(audit, "metrics_frozen_partial.csv", MODELS[2:])

    result = []
    labeled_abstentions = 0
    for n in (500, 1000):
        for period in PERIODS:
            for seed in SEEDS:
                reference = None
                for model in MODELS:
                    if model in FOLDERS:
                        suffix = (f"predictions/frozen/{FOLDERS[model]}/seed{seed}/"
                                  f"n{n}_{period}_p3_json_strict.csv")
                        rows = rows_at(frozen, suffix)
                        if any(r["status"] != "ok" for r in rows):
                            raise ValueError(f"Non-OK frozen response: {model} n{n} {period} {seed}")
                        labeled_abstentions += sum(
                            r["abstain"].lower() == "true" and r["pred_label"] in ("NEG", "POS")
                            for r in rows)
                        saved = saved_frozen
                    else:
                        folder = "later_distilbert" if model == "distilbert" else "aligned_tfidf"
                        rows = rows_at(aligned, f"{folder}/seed{seed}/n{n}_{period}.csv")
                        saved = saved_supervised
                        if any(not math.isclose(float(r["confidence"]),
                                                max(float(r["prob_pos"]), 1 - float(r["prob_pos"])),
                                                abs_tol=1e-5) for r in rows):
                            raise ValueError(f"Supervised confidence mismatch: {model} n{n} {period} {seed}")
                    if len(rows) != 2*n or Counter(r["gold_label"] for r in rows) != {"NEG": n, "POS": n}:
                        raise ValueError(f"Wrong rows or balance: {model} n{n} {period} {seed}")
                    if any(r["pred_label"] not in ("NEG", "POS") for r in rows):
                        raise ValueError(f"Missing label: {model} n{n} {period} {seed}")
                    identity = {(r["example_id"], r["text_sha1"], r["gold_label"]) for r in rows}
                    if len(identity) != len(rows) or (reference is not None and reference != identity):
                        raise ValueError(f"Unmatched examples: {model} n{n} {period} {seed}")
                    reference = identity
                    value = ece(rows)
                    key = (model, n, period, seed)
                    if key not in saved or not math.isclose(value, saved[key], abs_tol=1e-8):
                        raise ValueError(f"ECE differs from saved metric: {key}: {value}")
                    result.append(dict(setting=f"Yelp n{n}", model=model, period=period,
                                       seed=seed, confidence_source=("predicted_probability" if model in MODELS[:2]
                                                                    else "llm_self_reported"), ece=value))

    print("Yelp sentiment | archived aligned cohort | canonical P3 | five-seed mean ECE (sample SD)")
    print(f"{'Setting':11} {'Model':25} {'Within':>14} {'Short':>14} {'Long':>14}")
    for n in (500, 1000):
        for model in MODELS:
            summaries = []
            for period in PERIODS:
                values = [r["ece"] for r in result if r["setting"] == f"Yelp n{n}"
                          and r["model"] == model and r["period"] == period]
                summaries.append(f"{statistics.mean(values):.4f}±{statistics.stdev(values):.4f}")
            print(f"Yelp n{n:<4}  {model:25} {summaries[0]:>14} {summaries[1]:>14} {summaries[2]:>14}")
    print("Verified 2 settings × 4 models × 3 periods × 5 seeds: matched examples and archived ECE.")
    if labeled_abstentions:
        print(f"Included {labeled_abstentions} P3 row(s) with abstain=true and a valid label, as in canonical scoring.")
    print("Supervised ECE uses predicted probabilities; frozen LLM ECE uses self-reported confidence.")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(result[0]))
            writer.writeheader()
            writer.writerows(result)
        print(f"Saved per-seed ECE: {args.csv}")


if __name__ == "__main__":
    main()
