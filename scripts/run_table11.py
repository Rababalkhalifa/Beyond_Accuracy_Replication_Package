"""Script to reproduce Table 11 (Focused Retained-Error Differences with Cluster Bootstrap).

Reproduces the 11 exploratory retained-error contrasts for RQ2 (Table 11 and Figure 4 of the manuscript)
using synchronized document-cluster bootstrap across seeds 42-46 (3,000 replicates, random seed 20261102).
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "code" / "analysis"))

from src.tables import build_table11_bootstrap_latex


def generate_table11(
    archives_dir: Path,
    out_dir: Path,
    recompute_bootstrap: bool = False,
    replicates: int = 3000,
    seed: int = 20261102
) -> List[Dict[str, any]]:
    archives_dir = Path(archives_dir).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "RQ2_Table11_bootstrap_contrasts.csv"

    if recompute_bootstrap or not csv_path.is_file():
        print(f"\nRunning {replicates:,} synchronized document-cluster bootstrap resamples across 11 RQ2 contrasts...")
        import rq2_paired_uncertainty as prior
        import rq2_uncertainty_improved_v2 as u2

        nb_path = BASE_DIR / "code" / "analysis" / "Temporal_coverage_rankings.ipynb"
        core = prior.get_core(nb_path)
        pools = u2.build_index(core, archives_dir)
        rows = u2.run(pools, replicates, seed)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved bootstrap CSV: {csv_path}")
    else:
        print("\nLoading verified 3,000-replicate synchronized document-cluster bootstrap contrasts (Table 11)...")
        with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))

    print("\n" + "=" * 115)
    print("Table 11: Focused Retained-Error Differences Across Evaluated Historical Cohorts (3,000 Bootstrap Resamples)")
    print("=" * 115)
    print(f"{'Setting':14} {'Period':10} {'Contrast':40} {'Diff (pp)':>10} {'95% CI (pp)':>20} {'Seed SD':>9} {'Clusters':>9}")
    print("-" * 115)
    for r in rows:
        diff = float(r["mean_difference_pp"])
        lo = float(r["ci_95_low_pp"])
        hi = float(r["ci_95_high_pp"])
        sd = float(r["seed_sd_pp"])
        clusters = int(r["unique_document_clusters_in_contrast"])
        ci_str = f"[{lo:+.3f}, {hi:+.3f}]"
        print(f"{r['setting']:14} {r['period']:10} {r['contrast']:40} {diff:+10.3f} {ci_str:>20} {sd:9.3f} {clusters:9,d}")
    print("=" * 115 + "\n")

    tex_path = out_dir / "RQ2_Table11_bootstrap_contrasts.tex"
    tex_path.write_text(build_table11_bootstrap_latex(rows), encoding="utf-8")
    print(f"Saved LaTeX table: {tex_path}")

    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce Table 11 (RQ2 Focused Retained-Error Bootstrap Contrasts).")
    parser.add_argument("--archives-dir", type=Path, default=BASE_DIR / "archives",
                        help="Path to directory containing prediction archives")
    parser.add_argument("--out-dir", type=Path, default=BASE_DIR / "outputs" / "tables",
                        help="Output directory for generated tables")
    parser.add_argument("--recompute-bootstrap", action="store_true",
                        help="Re-run the 3,000-replicate synchronized document-cluster bootstrap from scratch")
    args = parser.parse_args()
    generate_table11(args.archives_dir, args.out_dir, recompute_bootstrap=args.recompute_bootstrap)
