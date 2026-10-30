"""Script to reproduce Table 9 (RQ2: Temporal Robustness and Shift Degradation).

Computes five-seed means for Reference (Within), Short, and Long Macro-F1, along with
seed-paired deltas across Climate n500, Yelp n500, and Yelp n1000.

Adheres strictly to the paper's sign convention (Equation 6):
Delta_j = MacroF1(T_j) - MacroF1(T_0)  (Target minus Reference)
Positive indicates improvement; negative indicates decline.
"""

import argparse
import csv
import statistics
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.loader import ArchiveManager
from src.metrics import macro_f1
from src.tables import build_table9_latex

SEEDS = range(42, 47)
MODELS = ("distilbert", "tfidf_lr", "phi3.5:latest", "qwen2.5:3b-instruct")
SETTINGS = ("Climate n500", "Yelp n500", "Yelp n1000")
PERIODS = ("within", "short", "long")


def generate_table9(archives_dir: Path, out_dir: Path) -> list:
    archives_dir = Path(archives_dir).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with ArchiveManager(archives_dir) as mgr:
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

                    # Equation 6: Target minus Reference
                    delta_s = f1_by_period["short"] - f1_by_period["within"]
                    delta_l = f1_by_period["long"] - f1_by_period["within"]

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
    print(f"{'Dataset Setting':15} {'Model':22} {'Ref F1':>10} {'Short F1':>10} {'Long F1':>10} {'Delta Short':>12} {'Delta Long':>12}")
    print("-" * 91)
    for r in summary_rows:
        def fmt_delta(val: float) -> str:
            return f"+{val:.4f}" if val > 0 else f"{val:.4f}"
        print(f"{r['dataset_setting']:15} {r['model']:22} {r['within_f1']:10.4f} {r['short_f1']:10.4f} "
              f"{r['long_f1']:10.4f} {fmt_delta(r['delta_short']):>12} {fmt_delta(r['delta_long']):>12}")
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
    parser.add_argument("--archives-dir", type=Path, default=BASE_DIR / "archives",
                        help="Path to directory containing prediction archives")
    parser.add_argument("--out-dir", type=Path, default=BASE_DIR / "outputs" / "tables",
                        help="Output directory for generated tables")
    args = parser.parse_args()
    generate_table9(args.archives_dir, args.out_dir)
