"""Script to generate all publication figures and risk-coverage curves.

Reproduces:
1. Figure 4: Risk-Coverage curves across periods and rankings
   - outputs/figures/RQ2_coverage_rankings.{svg,pdf,png}
   - outputs/tables/RQ2_temporal_reliability_table.tex
   - outputs/tables/RQ2_verified_seed_metrics.csv
   - outputs/tables/RQ2_risk_coverage_per_seed.csv
2. Figure 5: Temporal movement profiles and corresponding model-performance changes
   - outputs/figures/Figure5_movement_vs_response.{svg,pdf,png}
"""

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.loader import ArchiveManager
from src.metrics import macro_f1, ece_10, risk_sweep
from src.plotting import plot_coverage_and_rankings, plot_figure5_movement_vs_response
from src.tables import build_table10_latex
from src.forensics import compute_all_forensics

SEEDS = range(42, 47)
MODELS = ("distilbert", "tfidf_lr", "phi3.5:latest", "qwen2.5:3b-instruct")
SETTINGS = ("Climate n500", "Yelp n500", "Yelp n1000")
PERIODS = ("within", "short", "long")


def run_figures_and_reliability(archives_dir: Path, splits_dir: Path, figures_dir: Path, tables_dir: Path):
    archives_dir = Path(archives_dir).resolve()
    splits_dir = Path(splits_dir).resolve()
    figures_dir = Path(figures_dir).resolve()
    tables_dir = Path(tables_dir).resolve()

    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("STEP 1: Computing Risk-Coverage Curves Across 180 Conditions (Figure 4)")
    print("=" * 80)

    with ArchiveManager(archives_dir) as mgr:
        records = []
        risk_rows = []
        delta_rows = []

        for setting in SETTINGS:
            n = 1000 if setting.endswith("1000") else 500
            labels = ("BELIEVER", "DENIER") if setting.startswith("Climate") else ("NEG", "POS")
            gold_col = "true_label" if setting.startswith("Climate") else "gold_label"
            pred_col = "predicted_label" if setting.startswith("Climate") else "pred_label"
            conf_col = "confidence"

            for model in MODELS:
                for target in ("short", "long"):
                    deltas = []
                    for seed in SEEDS:
                        if setting.startswith("Climate"):
                            r_within = mgr.load_climate_predictions(model, seed, "within")
                            r_target = mgr.load_climate_predictions(model, seed, target)
                        else:
                            r_within = mgr.load_yelp_predictions(model, n, seed, "within")
                            r_target = mgr.load_yelp_predictions(model, n, seed, target)
                        f1_w = macro_f1(r_within, labels, gold_col, pred_col)
                        f1_t = macro_f1(r_target, labels, gold_col, pred_col)
                        deltas.append(f1_t - f1_w)

                    delta_rows.append({
                        "dataset_setting": setting,
                        "target": target,
                        "model": model,
                        "delta_macro_f1_mean": float(statistics.mean(deltas)),
                        "delta_macro_f1_sd": float(statistics.stdev(deltas))
                    })

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

    # Generate Figure 4 (Risk-Coverage & Ranking Curves)
    fig4_paths = plot_coverage_and_rankings(risk_rows, figures_dir, "RQ2_coverage_rankings")
    for p in fig4_paths:
        print(f"Generated Figure 4: {p}")

    # Generate Table 10 (Reliability)
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

    tex_table10 = tables_dir / "RQ2_temporal_reliability_table.tex"
    tex_table10.write_text(build_table10_latex(means, SETTINGS, MODELS), encoding="utf-8")
    print(f"Saved Table 10 LaTeX: {tex_table10}")

    # Save CSV caches
    csv_rec = tables_dir / "RQ2_verified_seed_metrics.csv"
    with csv_rec.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    csv_risk = tables_dir / "RQ2_risk_coverage_per_seed.csv"
    with csv_risk.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(risk_rows[0].keys()))
        writer.writeheader()
        writer.writerows(risk_rows)

    print("\n" + "=" * 80)
    print("STEP 2: Computing Forensics and Generating Movement vs Response (Figure 5)")
    print("=" * 80)

    forensics_rows = compute_all_forensics(splits_dir)
    fig5_paths = plot_figure5_movement_vs_response(forensics_rows, delta_rows, figures_dir, "Figure5_movement_vs_response")
    for p in fig5_paths:
        print(f"Generated Figure 5: {p}")

    print("\n>> All Figures & Curves Successfully Generated (Status: PASS)\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate figures and risk-coverage curves.")
    parser.add_argument("--archives-dir", type=Path, default=BASE_DIR / "archives",
                        help="Path to prediction archives directory")
    parser.add_argument("--splits-dir", type=Path, default=BASE_DIR / "data" / "splits",
                        help="Path to dataset splits directory")
    parser.add_argument("--figures-dir", type=Path, default=BASE_DIR / "outputs" / "figures",
                        help="Output directory for generated figures")
    parser.add_argument("--tables-dir", type=Path, default=BASE_DIR / "outputs" / "tables",
                        help="Output directory for generated tables")
    args = parser.parse_args()

    run_figures_and_reliability(args.archives_dir, args.splits_dir, args.figures_dir, args.tables_dir)
