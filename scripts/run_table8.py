"""Script to reproduce Table 8 (RQ1: Canonical Within-Period Performance).

Computes five-seed means for Accuracy, Macro-F1, 10-bin ECE, and Latency
across Climate n500, Yelp n500, and Yelp n1000.
"""

import argparse
import csv
import statistics
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.loader import ArchiveManager
from src.metrics import calc_condition_metrics
from src.tables import build_table8_latex

SEEDS = range(42, 47)
MODELS = ("distilbert", "tfidf_lr", "phi3.5:latest", "qwen2.5:3b-instruct")
SETTINGS = ("Climate n500", "Yelp n500", "Yelp n1000")


def generate_table8(archives_dir: Path, out_dir: Path) -> list:
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with ArchiveManager(archives_dir) as mgr:
        # Load Yelp supervised latency lookup
        yelp_metrics = mgr.load_yelp_metrics_indices()
        
        # Collect per-seed records
        records = {}
        for setting in SETTINGS:
            n = 1000 if setting.endswith("1000") else 500
            labels = ("BELIEVER", "DENIER") if setting.startswith("Climate") else ("NEG", "POS")
            gold_col = "true_label" if setting.startswith("Climate") else "gold_label"
            pred_col = "predicted_label" if setting.startswith("Climate") else "pred_label"
            conf_col = "confidence"
            lat_col = "latency_sec"

            for seed in SEEDS:
                for model in MODELS:
                    if setting.startswith("Climate"):
                        rows = mgr.load_climate_predictions(model, seed, "within")
                        m = calc_condition_metrics(rows, gold_col, pred_col, labels, conf_col, lat_col)
                    else:
                        rows = mgr.load_yelp_predictions(model, n, seed, "within")
                        m = calc_condition_metrics(rows, gold_col, pred_col, labels, conf_col,
                                                   lat_col if model not in ("distilbert", "tfidf_lr") else None)
                        if model in ("distilbert", "tfidf_lr"):
                            key = (f"Yelp n{n}", model, seed, "within")
                            m["latency_mean_sec"] = float(yelp_metrics[key]["latency_mean_sec"])

                    records[setting, model, seed] = m

        # Aggregate across five seeds
        summary_rows = []
        for setting in SETTINGS:
            for model in MODELS:
                seed_vals = [records[setting, model, seed] for seed in SEEDS]
                row = {
                    "dataset_setting": setting,
                    "model": model,
                    "accuracy": statistics.mean(v["accuracy"] for v in seed_vals),
                    "macro_f1": statistics.mean(v["macro_f1"] for v in seed_vals),
                    "ece": statistics.mean(v["ece"] for v in seed_vals),
                    "latency_mean_sec": statistics.mean(v["latency_mean_sec"] for v in seed_vals)
                }
                summary_rows.append(row)

    # Print to console
    print("\n" + "=" * 80)
    print("RQ1 Table 8: Canonical Within-Period Performance (5-Seed Means)")
    print("=" * 80)
    print(f"{'Dataset Setting':15} {'Model':22} {'Accuracy':>10} {'Macro-F1':>10} {'ECE':>10} {'Latency (s)':>14}")
    print("-" * 85)
    for r in summary_rows:
        print(f"{r['dataset_setting']:15} {r['model']:22} {r['accuracy']:10.4f} {r['macro_f1']:10.4f} "
              f"{r['ece']:10.4f} {r['latency_mean_sec']:14.6f}")
    print("=" * 80 + "\n")

    # Save LaTeX table
    tex_path = out_dir / "RQ1_Table8.tex"
    tex_path.write_text(build_table8_latex(summary_rows), encoding="utf-8")
    print(f"Saved LaTeX table: {tex_path}")

    # Save CSV summary
    csv_path = out_dir / "RQ1_Table8_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved CSV summary: {csv_path}")

    return summary_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce RQ1 Table 8.")
    parser.add_argument("--archives-dir", type=Path, default=Path("archives"),
                        help="Path to directory containing zip archives")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/tables"),
                        help="Path to output directory for tables")
    args = parser.parse_args()
    generate_table8(args.archives_dir, args.out_dir)
