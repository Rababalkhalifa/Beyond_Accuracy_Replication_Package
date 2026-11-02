"""Script to reproduce Table 13 (Temporal-Forensics Evidence by Dataset and Target Horizon).

Computes all 6 dataset-side movement metrics (Equations 12-17) across 5 seeds:
1. Lexical novelty (Eq. 12)
2. Jensen-Shannon divergence JSD_2 (Eq. 13)
3. Top-term concentration change Delta C_100 (Eq. 14)
4. Document-length change Delta L (Eq. 15)
5. Vocabulary-overlap loss (Eq. 16)
6. Task-native class-marker asymmetry A_marker (Eq. 17)
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.forensics import compute_all_forensics
from src.tables import build_table13_forensics_latex


def generate_table13(splits_dir: Path, out_dir: Path) -> List[Dict[str, any]]:
    splits_dir = Path(splits_dir).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\nComputing temporal forensics evidence across all horizons and seeds (42..46)...")
    rows = compute_all_forensics(splits_dir)

    print("\n" + "=" * 135)
    print("Table 13: Temporal-Forensics Evidence by Dataset and Target Horizon (5-Seed Means +- SD)")
    print("=" * 135)
    print(f"{'Setting':15} {'Target':7} {'Novelty':>18} {'JSD_2':>18} {'Delta C_100':>18} {'Delta L':>18} {'Overlap Loss':>18} {'Marker Asym':>18}")
    print("-" * 135)
    for r in rows:
        nov = f"{r['lexical_novelty_mean']:.4f} +- {r['lexical_novelty_sd']:.4f}"
        jsd = f"{r['js_divergence_mean']:.4f} +- {r['js_divergence_sd']:.4f}"
        dc = f"{r['top100_mass_change_mean']:+.4f} +- {r['top100_mass_change_sd']:.4f}"
        dl = f"{r['mean_length_change_mean']:+.4f} +- {r['mean_length_change_sd']:.4f}"
        ovl = f"{r['overlap_loss_mean']:.4f} +- {r['overlap_loss_sd']:.4f}"
        asym = f"{r['class_marker_asymmetry_mean']:.4f} +- {r['class_marker_asymmetry_sd']:.4f}"
        print(f"{r['dataset_setting']:15} {r['target']:7} {nov:>18} {jsd:>18} {dc:>18} {dl:>18} {ovl:>18} {asym:>18}")
    print("=" * 135 + "\n")

    tex_content = build_table13_forensics_latex(rows)
    for fname in ("RQ5_Table13_temporal_forensics.tex", "RQ5_Table12_temporal_forensics.tex"):
        tex_path = out_dir / fname
        tex_path.write_text(tex_content, encoding="utf-8")
    print(f"Saved LaTeX table: {out_dir / 'RQ5_Table13_temporal_forensics.tex'}")

    for fname in ("RQ5_Table13_summary.csv", "RQ5_Table12_summary.csv"):
        csv_path = out_dir / fname
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Saved CSV summary: {out_dir / 'RQ5_Table13_summary.csv'}")

    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce Table 13 (Temporal Forensics Evidence).")
    parser.add_argument("--splits-dir", type=Path, default=BASE_DIR / "data" / "splits",
                        help="Path to directory containing dataset splits")
    parser.add_argument("--out-dir", type=Path, default=BASE_DIR / "outputs" / "tables",
                        help="Output directory for generated tables")
    args = parser.parse_args()
    generate_table13(args.splits_dir, args.out_dir)
