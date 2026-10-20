"""Script to reproduce Table 9 (RQ2: Temporal Robustness and Shift Degradation).

Computes five-seed means for Within, Short, and Long Macro-F1, along with
seed-paired deltas across Climate n500, Yelp n500, and Yelp n1000.
"""

import argparse
import csv
import statistics
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.loader import ArchiveManager
from src.metrics import macro_f1
from src.tables import build_table9_latex

SEEDS = range(42, 47)
MODELS = ("distilbert", "tfidf_lr", "phi3.5:latest", "qwen2.5:3b-instruct")
SETTINGS = ("Climate n500", "Yelp n500", "Yelp n1000")
PERIODS = ("within", "short", "long")


def generate_table9(archives_dir: Path, out_dir: Path) -> list:
    out_dir.mkdir(parents=True, exist_ok=True)

    with ArchiveManager(archives_dir) as mgr:
        # Collect per-seed Macro-F1 across periods
        records = {}
        for setting in SETTINGS:
            n = 1000 if setting.endswith("1000") else 500
            labels = ("BELIEVER", "DENIER") if setting.startswith("Climate") else ("NEG", "POS")
            gold_col = "true_label" if setting.startswith("Climate") else "gold_label"
            pred_col = "predicted_label" if setting.startswith("Climate") else "pred_label"

            for seed in SEEDS:
                for model in MODELS:
                    f1_by_period = {}
                    for period in PERIODS:
                        if setting.startswith("Climate"):
                            rows = mgr.load_climate_predictions(model, seed, period)
                        else:
                            rows = mgr.load_yelp_predictions(model, n, seed, period)
                        f1_by_period[period] = macro_f1(rows, labels, gold_col, pred_col)

                    delta_s = f1_by_period["within"] - f1_by_period["short"]
                    delta_l = f1_by_period["within"] - f1_by_period["long"]

                    records[setting, model, seed] = {
                        "within": f1_by_period["within"],
                        "short": f1_by_period["short"],
                        "long": f1_by_period["long"],
                        "delta_short": delta_s,
                        "delta_long": delta_l
                    }

        # Aggregate across seeds
        summary_rows = []
        for setting in SETTINGS:
            for model in MODELS:
                seed_vals = [records[setting, model, seed] for seed in SEEDS]
                row = {
                    "dataset_setting": setting,
                    "model": model,
                    "within_f1": statistics.mean(v["within"] for v in seed_vals),
                    "short_f1": statistics.mean(v["short"] for v in seed_vals),
                    "long_f1": statistics.mean(v["long"] for v in seed_vals),
                    "delta_short": statistics.mean(v["delta_short"] for v in seed_vals),
                    "delta_long": statistics.mean(v["delta_long"] for v in seed_vals)
                }
                summary_rows.append(row)

    # Print to console
    print("\n" + "=" * 80)
    print("RQ2 Table 9: Temporal Macro-F1 and Robustness (5-Seed Means)")
    print("=" * 80)
    print(f"{'Dataset Setting':15} {'Model':22} {'Within F1':>10} {'Short F1':>10} {'Long F1':>10} {'Delta Short':>12} {'Delta Long':>12}")
    print("-" * 91)
    for r in summary_rows:
        print(f"{r['dataset_setting']:15} {r['model']:22} {r['within_f1']:10.4f} {r['short_f1']:10.4f} "
              f"{r['long_f1']:10.4f} {r['delta_short']:12.4f} {r['delta_long']:12.4f}")
    print("=" * 80 + "\n")

    # Save LaTeX table
    tex_path = out_dir / "RQ2_Table9.tex"
    tex_path.write_text(build_table9_latex(summary_rows), encoding="utf-8")
    print(f"Saved LaTeX table: {tex_path}")

    # Save CSV summary
    csv_path = out_dir / "RQ2_Table9_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved CSV summary: {csv_path}")

    return summary_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce RQ2 Table 9.")
    parser.add_argument("--archives-dir", type=Path, default=Path("archives"),
                        help="Path to directory containing zip archives")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/tables"),
                        help="Path to output directory for tables")
    args = parser.parse_args()
    generate_table9(args.archives_dir, args.out_dir)
