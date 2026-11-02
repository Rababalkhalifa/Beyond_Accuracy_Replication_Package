"""Script to reproduce Table 12 (Temporal-Forensics Evidence by Dataset and Horizon).

Computes five-seed means and sample standard deviations for:
- Lexical novelty (Equation 12)
- Distributional movement via base-2 Jensen-Shannon divergence (Equation 13)
- Top-term concentration change Delta C_100 (Equation 14)
- Document-length change Delta L (Equation 15)
- Vocabulary-overlap loss (Equation 16)
- Task-native class-marker asymmetry A_marker (Equation 17)

Directly derived from the authoritative split CSVs under lowercase word tokenization.
"""

import argparse
import csv
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.forensics import compute_all_forensics
from src.tables import build_table12_latex


def generate_table12(splits_dir: Path, out_dir: Path) -> list:
    splits_dir = Path(splits_dir).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\nComputing temporal forensics evidence across all horizons and seeds (42..46)...")
    summary_rows = compute_all_forensics(splits_dir)

    # Print to console
    print("\n" + "=" * 115)
    print("Table 12: Temporal-Forensics Evidence by Dataset and Target Horizon (5-Seed Means +- SD)")
    print("=" * 115)
    print(f"{'Setting':15} {'Target':6} {'Novelty':>18} {'JSD_2':>18} {'Delta C_100':>18} {'Delta L':>18} {'Overlap Loss':>18} {'Marker Asym':>18}")
    print("-" * 135)
    for r in summary_rows:
        nov_str = f"{r['lexical_novelty_mean']:.4f} +- {r['lexical_novelty_sd']:.4f}"
        jsd_str = f"{r['js_divergence_mean']:.4f} +- {r['js_divergence_sd']:.4f}"
        dC_str = f"{r['top100_mass_change_mean']:+.4f} +- {r['top100_mass_change_sd']:.4f}"
        dL_str = f"{r['mean_length_change_mean']:+.4f} +- {r['mean_length_change_sd']:.4f}"
        ovl_str = f"{r['overlap_loss_mean']:.4f} +- {r['overlap_loss_sd']:.4f}"
        asym_str = f"{r['class_marker_asymmetry_mean']:.4f} +- {r['class_marker_asymmetry_sd']:.4f}"
        print(f"{r['dataset_setting']:15} {r['target']:6} {nov_str:>18} {jsd_str:>18} {dC_str:>18} {dL_str:>18} {ovl_str:>18} {asym_str:>18}")
    print("=" * 115 + "\n")

    # Save LaTeX table
    tex_path = out_dir / "RQ5_Table12_temporal_forensics.tex"
    tex_path.write_text(build_table12_latex(summary_rows), encoding="utf-8")
    print(f"Saved LaTeX table: {tex_path}")

    # Save CSV summary
    csv_path = out_dir / "RQ5_Table12_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved CSV summary: {csv_path}")

    return summary_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce Table 12 (Temporal Forensics).")
    parser.add_argument("--splits-dir", type=Path, default=BASE_DIR / "data" / "splits",
                        help="Path to directory containing dataset splits")
    parser.add_argument("--out-dir", type=Path, default=BASE_DIR / "outputs" / "tables",
                        help="Output directory for generated tables")
    args = parser.parse_args()
    generate_table12(args.splits_dir, args.out_dir)
