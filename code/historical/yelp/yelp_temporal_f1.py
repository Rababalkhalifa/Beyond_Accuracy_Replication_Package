r"""Recompute Yelp Table 9 Macro-F1 from the archived, aligned prediction cohort.

PowerShell:
  python .\yelp_temporal_f1.py --source-dir C:\Research --csv .\yelp_temporal_f1.csv

This reads saved predictions only. It refuses to mix the newer supervised
Yelp cohort in completed_prediction_audit.zip with archived frozen predictions.
Positive delta means lower target-period Macro-F1 than the within reference.
"""

import argparse
import csv
import io
import statistics
import zipfile
from collections import Counter
from pathlib import Path

SEEDS = range(42, 47)
PERIODS = ("within", "short", "long")
MODELS = ("distilbert", "tfidf_lr", "phi3.5:latest", "qwen2.5:3b-instruct")
FOLDERS = {"phi3.5:latest": "phi3.5_latest",
           "qwen2.5:3b-instruct": "qwen2.5_3b-instruct"}


def archive(path):
    if not path.is_file() or not zipfile.is_zipfile(path):
        raise ValueError(f"Missing or invalid ZIP: {path}")
    with zipfile.ZipFile(path) as z:
        return {name.replace("\\", "/"): z.read(name) for name in z.namelist()
                if name.lower().endswith(".csv")}


def rows_at(files, suffix):
    hits = [name for name in files if name.endswith(suffix)]
    if len(hits) != 1:
        raise ValueError(f"Expected one {suffix}; found {len(hits)}")
    return list(csv.DictReader(io.StringIO(files[hits[0]].decode("utf-8-sig"))))


def score(rows, n, model, period, seed):
    if len(rows) != 2 * n or Counter(r["gold_label"] for r in rows) != {"NEG": n, "POS": n}:
        raise ValueError(f"Wrong row count or class balance: {model}, n{n}, {period}, seed {seed}")
    if any(r["pred_label"] not in ("NEG", "POS") for r in rows):
        raise ValueError(f"Missing/invalid predicted label: {model}, n{n}, {period}, seed {seed}")
    if model in FOLDERS and any(r["status"] != "ok" for r in rows):
        raise ValueError(f"Non-OK frozen response: {model}, n{n}, {period}, seed {seed}")
    ids = Counter((r["example_id"], r["text_sha1"], r["gold_label"]) for r in rows)
    if len(ids) != len(rows):
        raise ValueError(f"Repeated Yelp example: {model}, n{n}, {period}, seed {seed}")
    f1 = []
    for label in ("NEG", "POS"):
        tp = sum(r["gold_label"] == r["pred_label"] == label for r in rows)
        fp = sum(r["gold_label"] != label and r["pred_label"] == label for r in rows)
        fn = sum(r["gold_label"] == label and r["pred_label"] != label for r in rows)
        f1.append(2 * tp / (2 * tp + fp + fn))
    return statistics.mean(f1), ids


def metric_index(files, suffix, models):
    records = rows_at(files, suffix)
    index = {}
    for r in records:
        if r["model"] not in models or r["split"] not in PERIODS:
            continue
        if r.get("prompt") != "p3_json_strict":
            continue
        key = (r["model"], int(r["eval_n_per_label"]), r["split"], int(r["seed"]))
        if key in index:
            raise ValueError(f"Duplicate archived metric: {key}")
        index[key] = float(r["macro_f1"])
    return index


def calculate(frozen, aligned, later, frozen_audit):
    supervised_saved = metric_index(later, "later_run/metrics_per_seed.csv", MODELS[:2])
    frozen_saved = metric_index(frozen_audit, "metrics_frozen_partial.csv", MODELS[2:])
    result = {}
    p3_abstain_with_label = 0
    for n in (500, 1000):
        for period in PERIODS:
            for seed in SEEDS:
                matched = None
                for model in MODELS:
                    if model in FOLDERS:
                        suffix = (f"predictions/frozen/{FOLDERS[model]}/seed{seed}/"
                                  f"n{n}_{period}_p3_json_strict.csv")
                        rows = rows_at(frozen, suffix)
                        saved = frozen_saved
                        p3_abstain_with_label += sum(
                            r.get("abstain", "").lower() == "true" and
                            r.get("pred_label") in ("NEG", "POS") for r in rows)
                    else:
                        folder = "later_distilbert" if model == "distilbert" else "aligned_tfidf"
                        rows = rows_at(aligned, f"{folder}/seed{seed}/n{n}_{period}.csv")
                        saved = supervised_saved
                    value, ids = score(rows, n, model, period, seed)
                    if matched is not None and matched != ids:
                        raise ValueError(f"Unmatched Yelp evaluation examples: n{n}, {period}, seed {seed}, {model}")
                    matched = ids
                    key = (model, n, period, seed)
                    if key not in saved or abs(value - saved[key]) > 1e-8:
                        raise ValueError(f"Saved per-seed Macro-F1 mismatch or missing: {key}")
                    result[key] = value
    return result, p3_abstain_with_label


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", type=Path, help="Find the four Yelp ZIPs recursively")
    for arg in ("frozen", "aligned", "later", "frozen_audit"):
        p.add_argument("--" + arg.replace("_", "-"), type=Path)
    p.add_argument("--csv", type=Path, help="Write means, sample SDs, and per-seed values")
    args = p.parse_args()
    stems = {"frozen": "original_yelp_frozen_predictions",
             "aligned": "yelp_prediction_alignment",
             "later": "code_and_remaining_results",
             "frozen_audit": "yelp_results_audit"}
    for arg, stem in stems.items():
        if getattr(args, arg) is None and args.source_dir:
            hits = list(args.source_dir.rglob(stem + "*.zip"))
            if len(hits) != 1:
                p.error(f"Expected one {stem}*.zip beneath {args.source_dir}; found {len(hits)}")
            setattr(args, arg, hits[0])
        if getattr(args, arg) is None:
            p.error(f"Supply --{arg.replace('_', '-')} or --source-dir")
    try:
        values, abstain_count = calculate(*(archive(getattr(args, arg)) for arg in stems))
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        p.exit(2, f"Verification failed: {exc}\n")
    print("Yelp sentiment | archived aligned cohort | canonical P3 for frozen models")
    print("Setting     Model                       Within F1  Short F1   Long F1   Delta short Delta long")
    summary = []
    for n in (500, 1000):
        for model in MODELS:
            periods = {period: [values[model, n, period, seed] for seed in SEEDS]
                       for period in PERIODS}
            out = {"setting": f"Yelp n{n}", "model": model}
            for period in PERIODS:
                out[f"mean_{period}_f1"] = statistics.mean(periods[period])
                out[f"sd_{period}_f1"] = statistics.stdev(periods[period])
                out.update({f"seed_{seed}_{period}_f1": value
                            for seed, value in zip(SEEDS, periods[period])})
            for period in ("short", "long"):
                changes = [values[model, n, "within", seed] - values[model, n, period, seed]
                           for seed in SEEDS]
                out[f"mean_delta_{period}"] = statistics.mean(changes)
                out[f"sd_delta_{period}"] = statistics.stdev(changes)
                out.update({f"seed_{seed}_delta_{period}": value
                            for seed, value in zip(SEEDS, changes)})
            summary.append(out)
            print(f"Yelp n{n:<4}  {model:27} {out['mean_within_f1']:9.4f} "
                  f"{out['mean_short_f1']:9.4f} {out['mean_long_f1']:9.4f} "
                  f"{out['mean_delta_short']:11.4f} {out['mean_delta_long']:10.4f}")
    print("Verified: 2 sample settings × 4 models × 3 periods × 5 seeds; matched reviews per setting/period/seed.")
    print("All per-seed archived Macro-F1 values matched predictions.")
    if abstain_count:
        print(f"Note: {abstain_count} P3 row(s) had abstain=true and a valid label; label included in canonical scoring.")
    print("Saved predictions alone cannot verify fixed supervised weights across periods.")
    if args.csv:
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0]))
            w.writeheader()
            w.writerows(summary)
        print(f"Saved summary: {args.csv}")


if __name__ == "__main__":
    main()
