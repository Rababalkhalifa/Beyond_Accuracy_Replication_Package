"""Script to generate publication figures and coverage rankings.

Reproduces:
- outputs/figures/RQ2_coverage_rankings.svg, .pdf, .png
- outputs/tables/RQ2_temporal_reliability_table.tex
- outputs/tables/RQ2_verified_seed_metrics.csv
- outputs/tables/RQ2_risk_coverage_per_seed.csv
"""

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.loader import ArchiveManager
from src.metrics import macro_f1, ece_10, risk_sweep
from src.plotting import plot_coverage_and_rankings
from src.tables import build_reliability_table_latex

SEEDS = range(42, 47)
MODELS = ("distilbert", "tfidf_lr", "phi3.5:latest", "qwen2.5:3b-instruct")
SETTINGS = ("Climate n500", "Yelp n500", "Yelp n1000")
PERIODS = ("within", "short", "long")


def run_figures_and_reliability(archives_dir: Path, figures_dir: Path, tables_dir: Path):
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    print("\nExtracting predictions and computing risk-coverage curves across 180 conditions...")

    with ArchiveManager(archives_dir) as mgr:
        records = []
        risk_rows = []

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
                            "n_per_class": n,
                            "accuracy": acc,
                            "macro_f1": f1,
                            "ece": ece,
                            "risk_at_80": curve[60],
                            "risk_at_100": curve[80]
                        }
                        records.append(record)

                        for pct, rk in enumerate(curve, start=20):
                            risk_rows.append({
                                "setting": setting,
                                "model": model,
                                "seed": seed,
                                "period": period,
                                "coverage": pct / 100.0,
                                "expected_risk": rk
                            })

    # Group means for reliability table
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

    # Save CSV outputs
    metrics_csv = tables_dir / "RQ2_verified_seed_metrics.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)
    print(f"Saved verified seed metrics: {metrics_csv}")

    risk_csv = tables_dir / "RQ2_risk_coverage_per_seed.csv"
    with risk_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(risk_rows[0].keys()))
        w.writeheader()
        w.writerows(risk_rows)
    print(f"Saved risk-coverage curves: {risk_csv}")

    # Save Reliability Table
    rel_tex = tables_dir / "RQ2_temporal_reliability_table.tex"
    rel_tex.write_text(build_reliability_table_latex(means, SETTINGS, MODELS), encoding="utf-8")
    print(f"Saved Reliability LaTeX Table: {rel_tex}")

    # Plot Figures
    saved_figs = plot_coverage_and_rankings(risk_rows, figures_dir)
    print("Generated Figures:")
    for sf in saved_figs:
        print(f"  {sf}")

    return saved_figs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate publication figures and reliability table.")
    parser.add_argument("--archives-dir", type=Path, default=Path("archives"),
                        help="Path to directory containing zip archives")
    parser.add_argument("--figures-dir", type=Path, default=Path("outputs/figures"),
                        help="Path to output directory for figures")
    parser.add_argument("--tables-dir", type=Path, default=Path("outputs/tables"),
                        help="Path to output directory for tables")
    args = parser.parse_args()
    run_figures_and_reliability(args.archives_dir, args.figures_dir, args.tables_dir)
