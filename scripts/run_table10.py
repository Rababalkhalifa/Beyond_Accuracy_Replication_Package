"""Script to reproduce Table 10 (Period-Specific Calibration and Selective Risk).

Computes five-seed means for:
- Expected Calibration Error (ECE-10) across Reference, Short, and Long
- Selective risk at 80% coverage (R_80) across Reference, Short, and Long
- Selective risk at 100% coverage (R_100 = 1 - Accuracy) across Reference, Short, and Long

Adheres strictly to the paper's definition of equal fractional tie-handling.
"""

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.loader import ArchiveManager
from src.metrics import macro_f1, ece_10, risk_sweep
from src.tables import build_table10_latex

SEEDS = range(42, 47)
MODELS = ("distilbert", "tfidf_lr", "phi3.5:latest", "qwen2.5:3b-instruct")
SETTINGS = ("Climate n500", "Yelp n500", "Yelp n1000")
PERIODS = ("within", "short", "long")


def generate_table10(archives_dir: Path, out_dir: Path) -> dict:
    archives_dir = Path(archives_dir).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\nExtracting predictions and computing calibration & selective risk across 180 conditions...")

    with ArchiveManager(archives_dir) as mgr:
        records = []
        for setting in SETTINGS:
            n = 1000 if setting.endswith("1000") else 500
            labels = ("BELIEVER", "DENIER") if setting.startswith("Climate") else ("NEG", "POS")
            gold_col = "true_label" if setting.startswith("Climate") else "gold_label"
            pred_col = "predicted_label" if setting.startswith("Climate") else "pred_label"
            conf_col = "confidence"

            for period in PERIODS:
                for seed in SEEDS:
                    for model in MODELS:
                        if setting.startswith("Climate"):
                            rows = mgr.load_climate_predictions(model, seed, period)
                        else:
                            rows = mgr.load_yelp_predictions(model, n, seed, period)

                        wrong = [r[gold_col] != r[pred_col] for r in rows]
                        conf = [float(r[conf_col]) for r in rows]
                        acc = 1.0 - sum(wrong) / len(rows)
                        f1 = macro_f1(rows, labels, gold_col, pred_col)
                        ece = ece_10([not w for w in wrong], conf)

                        curve = risk_sweep(conf, wrong)

                        record = {
                            "setting": setting,
                            "model": model,
                            "seed": seed,
                            "period": period,
                            "accuracy": acc,
                            "macro_f1": f1,
                            "ece": ece,
                            "risk_at_80": curve[60],
                            "risk_at_100": curve[80]
                        }
                        records.append(record)

    grouped = defaultdict(list)
    for r in records:
        grouped[r["setting"], r["model"], r["period"]].append(r)

    means = {
        key: {
            field: statistics.mean(float(row[field]) for row in r_list)
            for field in ("accuracy", "macro_f1", "ece", "risk_at_80", "risk_at_100")
        }
        for key, r_list in grouped.items()
    }

    # Print to console
    print("\n" + "=" * 115)
    print("Table 10: Calibration (ECE) and Selective Risk (R_80, R_100) Across Periods (5-Seed Means)")
    print("=" * 115)
    print(f"{'Setting':15} {'Model':22} | {'ECE (Ref/Short/Long)':^24} | {'R_80 (Ref/Short/Long)':^24} | {'R_100 (Ref/Short/Long)':^24}")
    print("-" * 115)
    for setting in SETTINGS:
        for model in MODELS:
            ece_str = f"{means[setting, model, 'within']['ece']:.4f} / {means[setting, model, 'short']['ece']:.4f} / {means[setting, model, 'long']['ece']:.4f}"
            r80_str = f"{means[setting, model, 'within']['risk_at_80']:.4f} / {means[setting, model, 'short']['risk_at_80']:.4f} / {means[setting, model, 'long']['risk_at_80']:.4f}"
            r100_str = f"{means[setting, model, 'within']['risk_at_100']:.4f} / {means[setting, model, 'short']['risk_at_100']:.4f} / {means[setting, model, 'long']['risk_at_100']:.4f}"
            print(f"{setting:15} {model:22} | {ece_str:^24} | {r80_str:^24} | {r100_str:^24}")
    print("=" * 115 + "\n")

    # Save LaTeX table
    tex_path = out_dir / "RQ2_Table10_temporal_reliability.tex"
    tex_path.write_text(build_table10_latex(means, SETTINGS, MODELS), encoding="utf-8")
    print(f"Saved LaTeX table: {tex_path}")

    # Save CSV summary
    csv_path = out_dir / "RQ2_Table10_summary.csv"
    csv_rows = []
    for setting in SETTINGS:
        for model in MODELS:
            row = {
                "setting": setting,
                "model": model,
                "ece_reference": means[setting, model, "within"]["ece"],
                "ece_short": means[setting, model, "short"]["ece"],
                "ece_long": means[setting, model, "long"]["ece"],
                "r80_reference": means[setting, model, "within"]["risk_at_80"],
                "r80_short": means[setting, model, "short"]["risk_at_80"],
                "r80_long": means[setting, model, "long"]["risk_at_80"],
                "r100_reference": means[setting, model, "within"]["risk_at_100"],
                "r100_short": means[setting, model, "short"]["risk_at_100"],
                "r100_long": means[setting, model, "long"]["risk_at_100"],
            }
            csv_rows.append(row)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Saved CSV summary: {csv_path}")

    return means


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce Table 10 (Temporal Reliability & Selective Risk).")
    parser.add_argument("--archives-dir", type=Path, default=BASE_DIR / "archives",
                        help="Path to directory containing prediction archives")
    parser.add_argument("--out-dir", type=Path, default=BASE_DIR / "outputs" / "tables",
                        help="Output directory for generated tables")
    args = parser.parse_args()
    generate_table10(args.archives_dir, args.out_dir)
