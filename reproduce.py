#!/usr/bin/env python3
"""Master Reproduction Script for 'Beyond Accuracy' Research Paper.

Usage:
    python reproduce.py --all          # Replicate all tables, figures, and run full audit
    python reproduce.py --table8       # Replicate RQ1 Table 8 (within-period canonical performance)
    python reproduce.py --table9       # Replicate RQ2 Table 9 (temporal robustness & deltas)
    python reproduce.py --figures      # Generate Figure panels (SVG, PDF, PNG) and Reliability Table
    python reproduce.py --audit        # Verify dataset splits SHA-256 and prediction integrity
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from scripts.audit_dataset_splits import run_audit
from scripts.run_table8 import generate_table8
from scripts.run_table9 import generate_table9
from scripts.plot_figures import run_figures_and_reliability


def main():
    parser = argparse.ArgumentParser(
        description="One-click replication suite for 'Beyond Accuracy' empirical results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python reproduce.py --all
  python reproduce.py --table8
  python reproduce.py --table9
  python reproduce.py --figures
  python reproduce.py --audit
"""
    )
    parser.add_argument("--all", action="store_true", help="Run full audit and reproduce all tables and figures")
    parser.add_argument("--table8", action="store_true", help="Reproduce Table 8 (RQ1 Canonical Performance)")
    parser.add_argument("--table9", action="store_true", help="Reproduce Table 9 (RQ2 Temporal Robustness)")
    parser.add_argument("--figures", action="store_true", help="Generate SVG/PDF/PNG figures and Reliability Table")
    parser.add_argument("--audit", action="store_true", help="Run split checksums and cross-model alignment audit")
    parser.add_argument("--archives-dir", type=Path, default=BASE_DIR / "archives", help="Directory with prediction ZIPs")
    parser.add_argument("--splits-dir", type=Path, default=BASE_DIR / "data" / "splits", help="Directory with dataset splits")
    parser.add_argument("--tables-dir", type=Path, default=BASE_DIR / "outputs" / "tables", help="Directory for table outputs")
    parser.add_argument("--figures-dir", type=Path, default=BASE_DIR / "outputs" / "figures", help="Directory for figure outputs")

    args = parser.parse_args()

    # If no flags provided, default to --all
    run_all = args.all or not (args.table8 or args.table9 or args.figures or args.audit)

    print("\n" + "#" * 80)
    print("  BEYOND ACCURACY: EMPIRICAL REPLICATION SUITE")
    print("#" * 80)

    if run_all or args.audit:
        print("\n>>> [1/4] Running Dataset & Prediction Provenance Audit...")
        run_audit(args.splits_dir, args.archives_dir)

    if run_all or args.table8:
        print("\n>>> [2/4] Generating RQ1 Table 8 (Within-Period Canonical Performance)...")
        generate_table8(args.archives_dir, args.tables_dir)

    if run_all or args.table9:
        print("\n>>> [3/4] Generating RQ2 Table 9 (Temporal Macro-F1 & Degradation)...")
        generate_table9(args.archives_dir, args.tables_dir)

    if run_all or args.figures:
        print("\n>>> [4/4] Generating Figures & Selective Reliability Table...")
        run_figures_and_reliability(args.archives_dir, args.figures_dir, args.tables_dir)

    print("\n" + "=" * 80)
    print("REPLICATION COMPLETE! Output files generated:")
    print("=" * 80)
    for p in sorted((BASE_DIR / "outputs").rglob("*.*")):
        if p.is_file():
            print(f"  - {p.relative_to(BASE_DIR)}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
