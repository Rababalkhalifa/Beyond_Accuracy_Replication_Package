"""Script to generate all publication figures (Figures 3, 4, 5, and 6) and risk-coverage curves.

Reproduces:
1. Figure 3: 3x3 Confidence-based risk-coverage curves across historical evaluation periods
   - outputs/figures/Figure3_temporal_risk_coverage.{pdf,svg,png}
2. Figure 4: 3-panel Exploratory retained-error comparisons for RQ2 (11 cluster-bootstrap contrasts)
   - outputs/figures/Figure4_uncertainty_contrasts.{pdf,svg,png}
3. Figure 5: 3-panel Reference-period Macro-F1 across prompt configurations (P1, P3, P5)
   - outputs/figures/Figure5_prompt_comparison.{pdf,svg,png}
4. Figure 6: Dual-panel Temporal movement profiles and corresponding model-performance changes
   - outputs/figures/Figure6_movement_vs_response.{pdf,svg,png}
5. Supplemental within-period risk-coverage & ranking figure:
   - outputs/figures/RQ2_coverage_rankings.{pdf,svg,png}
"""

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.loader import ArchiveManager
from src.metrics import macro_f1, ece_10, risk_sweep
from src.plotting import (
    plot_figure3_temporal_risk_coverage,
    plot_figure4_uncertainty_contrasts,
    plot_figure5_prompt_comparison,
    plot_figure6_movement_vs_response,
    plot_coverage_and_rankings,
)
from src.tables import build_table10_latex
from src.forensics import compute_all_forensics
from scripts.run_table11 import generate_table11
from scripts.run_table12 import generate_table12

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
    print("STEP 1: Computing Risk-Coverage Curves Across 180 Conditions (Figure 3)")
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

    # Generate Figure 3 (3x3 Temporal Risk-Coverage Curves)
    fig3_paths = plot_figure3_temporal_risk_coverage(risk_rows, figures_dir, "Figure3_temporal_risk_coverage")
    for p in fig3_paths:
        print(f"Generated Figure 3: {p}")

    # Generate Supplemental Within-Period Coverage & Rankings Figure
    supp_paths = plot_coverage_and_rankings(risk_rows, figures_dir, "RQ2_coverage_rankings")
    for p in supp_paths:
        print(f"Generated Supplemental Coverage Rankings: {p}")

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
    print("STEP 2: Generating Figure 4 (Retained-Error Uncertainty Contrasts)")
    print("=" * 80)
    contrast_rows = generate_table11(archives_dir, tables_dir, recompute_bootstrap=False)
    fig4_paths = plot_figure4_uncertainty_contrasts(contrast_rows, figures_dir, "Figure4_uncertainty_contrasts")
    for p in fig4_paths:
        print(f"Generated Figure 4: {p}")

    print("\n" + "=" * 80)
    print("STEP 3: Generating Figure 5 (Reference-Period Macro-F1 Across Prompt Configurations)")
    print("=" * 80)
    _, _, prompt_summary_rows = generate_table12(archives_dir, tables_dir)
    fig5_paths = plot_figure5_prompt_comparison(prompt_summary_rows, figures_dir, "Figure5_prompt_comparison")
    for p in fig5_paths:
        print(f"Generated Figure 5: {p}")

    print("\n" + "=" * 80)
    print("STEP 4: Generating Figure 6 (Temporal Movement Profiles vs Model Response)")
    print("=" * 80)
    forensics_rows = compute_all_forensics(splits_dir)
    fig6_paths = plot_figure6_movement_vs_response(forensics_rows, delta_rows, figures_dir, "Figure6_movement_vs_response")
    for p in fig6_paths:
        print(f"Generated Figure 6: {p}")

    print("\n>> All Manuscript Figures (Figures 3, 4, 5, 6) Successfully Generated (Status: PASS)\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate all publication figures (Figures 3-6) and risk-coverage curves.")
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
