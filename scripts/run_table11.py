"""Script to reproduce Table 11 (Explicit P5 Abstention & Error Reduction) and Prompt Sensitivity.

Computes:
1. Table 11: Frequency and observed benefit of explicit P5 abstention on reference samples:
   - Total attempts N
   - Abstained count and proportion
   - Incorrect count among abstained predictions
   - Error percentage before abstention (all predictions)
   - Error percentage after abstention (retained predictions only)
   - Error reduction in percentage points (pp)
2. Equation 10 Prompt Sensitivity S_prompt:
   - Within-seed Macro-F1 range across prompt configurations {P1_minimal, P3_json_strict, P5_abstain}
"""

import argparse
import csv
import io
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.loader import ArchiveManager
from src.metrics import macro_f1
from src.tables import build_table11_latex

SEEDS = range(42, 47)
FROZEN_MODELS = [("phi3.5:latest", "Phi"), ("qwen2.5:3b-instruct", "Qwen")]
SETTINGS = [
    ("Climate n500", "climate", 500, ["BELIEVER", "DENIER"], "true_label", "predicted_label"),
    ("Yelp n500", "yelp", 500, ["POS", "NEG"], "gold_label", "pred_label"),
    ("Yelp n1000", "yelp", 1000, ["POS", "NEG"], "gold_label", "pred_label")
]


def generate_table11(archives_dir: Path, out_dir: Path) -> Tuple[List[Dict[str, any]], List[Dict[str, any]]]:
    archives_dir = Path(archives_dir).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\nAnalyzing explicit P5 abstention and prompt configurations across 5 seeds...")

    abstention_rows = []
    sensitivity_rows = []

    with ArchiveManager(archives_dir) as mgr:
        for setting_name, domain, n_per_label, labels, gold_k, pred_k in SETTINGS:
            for model_id, model_display in FROZEN_MODELS:
                # -------------------------------------------------------------
                # 1. P5 Abstention Statistics (Table 11)
                # -------------------------------------------------------------
                tot_abstained = 0
                tot_n = 0
                tot_inc_abst = 0
                err_before_list = []
                err_after_list = []

                for seed in SEEDS:
                    if domain == "climate":
                        rows = mgr.load_climate_prompt_predictions(model_id, "p5_abstain", seed, "within")
                    else:
                        rows = mgr.load_yelp_prompt_predictions(model_id, "p5_abstain", n_per_label, seed, "within")

                    err_before = np.mean([r[gold_k] != r[pred_k] for r in rows])
                    abstained = [r for r in rows if r["abstain"].strip().lower() == "true"]
                    inc_abst = sum(r[gold_k] != r[pred_k] for r in abstained)
                    retained = [r for r in rows if r["abstain"].strip().lower() != "true"]
                    err_after = np.mean([r[gold_k] != r[pred_k] for r in retained]) if retained else 0.0

                    err_before_list.append(err_before * 100.0)
                    err_after_list.append(err_after * 100.0)
                    tot_abstained += len(abstained)
                    tot_n += len(rows)
                    tot_inc_abst += inc_abst

                err_before_mean = float(np.mean(err_before_list))
                err_after_mean = float(np.mean(err_after_list))
                reduction_pp = err_before_mean - err_after_mean

                abst_slash_n = f"{tot_abstained:,}/{tot_n:,}"
                inc_among = f"{tot_inc_abst}/{tot_abstained}" if tot_abstained > 0 else "---"

                abstention_rows.append({
                    "setting": setting_name,
                    "model": model_display,
                    "abstained_slash_n": abst_slash_n,
                    "incorrect_among_abstained": inc_among,
                    "error_before_pct": err_before_mean,
                    "error_after_pct": err_after_mean,
                    "reduction_pp": reduction_pp
                })

                # -------------------------------------------------------------
                # 2. Equation 10: Prompt Sensitivity S_prompt across {P1, P3, P5}
                # -------------------------------------------------------------
                ranges = []
                for seed in SEEDS:
                    if domain == "climate":
                        r1 = mgr.load_climate_prompt_predictions(model_id, "p1_minimal", seed, "within")
                        r3 = mgr.load_climate_prompt_predictions(model_id, "p3_json_strict", seed, "within")
                        r5 = mgr.load_climate_prompt_predictions(model_id, "p5_abstain", seed, "within")
                        f1s = [
                            macro_f1(r1, labels, gold_k, pred_k),
                            macro_f1(r3, labels, gold_k, pred_k),
                            macro_f1(r5, labels, gold_k, pred_k)
                        ]
                    else:
                        # For Yelp, evaluate on matched valid outputs across prompt configurations
                        r1_raw = mgr.load_yelp_prompt_predictions(model_id, "p1_minimal", n_per_label, seed, "within")
                        r3_raw = mgr.load_yelp_prompt_predictions(model_id, "p3_json_strict", n_per_label, seed, "within")
                        r5_raw = mgr.load_yelp_prompt_predictions(model_id, "p5_abstain", n_per_label, seed, "within")

                        r1_map = {r["example_id"]: r for r in r1_raw if r.get("status", "ok") == "ok" and r.get(pred_k)}
                        r3_map = {r["example_id"]: r for r in r3_raw if r.get("status", "ok") == "ok" and r.get(pred_k)}
                        r5_map = {r["example_id"]: r for r in r5_raw if r.get("status", "ok") == "ok" and r.get(pred_k)}

                        # Also match with other model to get exact 6-combination intersection
                        other_id = "qwen2.5:3b-instruct" if model_id == "phi3.5:latest" else "phi3.5:latest"
                        o1 = {r["example_id"] for r in mgr.load_yelp_prompt_predictions(other_id, "p1_minimal", n_per_label, seed, "within") if r.get("status", "ok") == "ok" and r.get(pred_k)}
                        o3 = {r["example_id"] for r in mgr.load_yelp_prompt_predictions(other_id, "p3_json_strict", n_per_label, seed, "within") if r.get("status", "ok") == "ok" and r.get(pred_k)}
                        o5 = {r["example_id"] for r in mgr.load_yelp_prompt_predictions(other_id, "p5_abstain", n_per_label, seed, "within") if r.get("status", "ok") == "ok" and r.get(pred_k)}

                        common = sorted(set(r1_map) & set(r3_map) & set(r5_map) & o1 & o3 & o5)
                        f1s = [
                            macro_f1([r1_map[k] for k in common], labels, gold_k, pred_k),
                            macro_f1([r3_map[k] for k in common], labels, gold_k, pred_k),
                            macro_f1([r5_map[k] for k in common], labels, gold_k, pred_k)
                        ]

                    ranges.append(max(f1s) - min(f1s))

                sensitivity_rows.append({
                    "setting": setting_name,
                    "model": model_display,
                    "s_prompt_mean": float(np.mean(ranges)),
                    "s_prompt_sd": float(np.std(ranges, ddof=1))
                })

    # Print Table 11 to console
    print("\n" + "=" * 95)
    print("Table 11: Frequency and Observed Benefit of Explicit P5 Abstention (5-Seed Means)")
    print("=" * 95)
    print(f"{'Setting':15} {'Model':7} {'Abstained / N':>15} {'Incorrect Among':>18} {'Err Before (%)':>16} {'Err After (%)':>15} {'Reduction (pp)':>16}")
    print("-" * 95)
    for r in abstention_rows:
        print(f"{r['setting']:15} {r['model']:7} {r['abstained_slash_n']:>15} {r['incorrect_among_abstained']:>18} "
              f"{r['error_before_pct']:16.3f} {r['error_after_pct']:15.3f} {r['reduction_pp']:16.3f}")
    print("=" * 95 + "\n")

    # Print S_prompt to console
    print("=" * 70)
    print("Equation 10: Prompt Sensitivity S_prompt (Macro-F1 Range across P1, P3, P5)")
    print("=" * 70)
    print(f"{'Setting':15} {'Model':7} {'Mean S_prompt':>18} {'Sample SD':>15}")
    print("-" * 70)
    for r in sensitivity_rows:
        print(f"{r['setting']:15} {r['model']:7} {r['s_prompt_mean']:18.4f} {r['s_prompt_sd']:15.4f}")
    print("=" * 70 + "\n")

    # Save LaTeX table
    tex_path = out_dir / "RQ3_Table11_abstention.tex"
    tex_path.write_text(build_table11_latex(abstention_rows), encoding="utf-8")
    print(f"Saved LaTeX table: {tex_path}")

    # Save CSV summaries
    csv_abst_path = out_dir / "RQ3_Table11_summary.csv"
    with csv_abst_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(abstention_rows[0].keys()))
        writer.writeheader()
        writer.writerows(abstention_rows)
    print(f"Saved CSV summary: {csv_abst_path}")

    csv_sens_path = out_dir / "RQ3_prompt_sensitivity_summary.csv"
    with csv_sens_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(sensitivity_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sensitivity_rows)
    print(f"Saved CSV summary: {csv_sens_path}")

    return abstention_rows, sensitivity_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce Table 11 & Prompt Sensitivity.")
    parser.add_argument("--archives-dir", type=Path, default=BASE_DIR / "archives",
                        help="Path to directory containing prediction archives")
    parser.add_argument("--out-dir", type=Path, default=BASE_DIR / "outputs" / "tables",
                        help="Output directory for generated tables")
    args = parser.parse_args()
    generate_table11(args.archives_dir, args.out_dir)
