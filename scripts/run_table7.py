"""Script to reproduce Table 7 (Descriptive Language Statistics of Sampled Temporal Splits).

Computes five-seed means and sample standard deviations for:
- Mean tokens per document
- Median tokens per document
- Total vocabulary size
- Type-token ratio (TTR)
- Top-100 token mass

Directly derived from the authoritative training (7,000) and evaluation (1,000 / 2,000) split CSVs.
"""

import argparse
import csv
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.descriptors import compute_all_descriptors
from src.tables import build_table7_latex


def generate_table7(splits_dir: Path, out_dir: Path) -> list:
    splits_dir = Path(splits_dir).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\nComputing language descriptors across all splits and seeds (42..46)...")
    summary_rows = compute_all_descriptors(splits_dir)

    # Print to console
    print("\n" + "=" * 90)
    print("Table 7: Descriptive Language Statistics of Sampled Temporal Splits")
    print("=" * 90)
    print(f"{'Setting':15} {'Split':7} {'Mean Len':>16} {'Med Len':>16} {'Vocab Size':>20} {'TTR':>15} {'Top-100 Mass':>16}")
    print("-" * 110)
    for r in summary_rows:
        mean_str = f"{r['mean_len_mean']:.2f} +- {r['mean_len_sd']:.2f}"
        med_str = f"{r['med_len_mean']:.2f} +- {r['med_len_sd']:.2f}"
        vocab_str = f"{r['vocab_size_mean']:.1f} +- {r['vocab_size_sd']:.1f}"
        ttr_str = f"{r['ttr_mean']:.3f} +- {r['ttr_sd']:.3f}"
        top_str = f"{r['top100_mass_mean']:.3f} +- {r['top100_mass_sd']:.3f}"
        print(f"{r['dataset_setting']:15} {r['split']:7} {mean_str:>16} {med_str:>16} {vocab_str:>20} {ttr_str:>15} {top_str:>16}")
    print("=" * 90 + "\n")

    # Save LaTeX table
    tex_path = out_dir / "Table7_descriptors.tex"
    tex_path.write_text(build_table7_latex(summary_rows), encoding="utf-8")
    print(f"Saved LaTeX table: {tex_path}")

    # Save CSV summary
    csv_path = out_dir / "Table7_descriptors_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved CSV summary: {csv_path}")

    return summary_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce Table 7 (Language Descriptors).")
    parser.add_argument("--splits-dir", type=Path, default=BASE_DIR / "data" / "splits",
                        help="Path to directory containing dataset splits")
    parser.add_argument("--out-dir", type=Path, default=BASE_DIR / "outputs" / "tables",
                        help="Output directory for generated tables")
    args = parser.parse_args()
    generate_table7(args.splits_dir, args.out_dir)
