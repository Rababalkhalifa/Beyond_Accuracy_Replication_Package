#!/usr/bin/env python3
"""Unified master replication driver for:

'Aging Gracefully? Reliability Profiling and Temporal Forensics of Frozen and Adapted Text Classifiers'
Author: Rabab Alkhalifa
SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7560344

Provides an end-to-end, fully reproducible CLI to verify all mathematical formulas,
audit dataset splits, generate all publication tables (Tables 7-12), and render all
publication figures (Figures 4 and 5).

Usage:
    python reproduce.py --all
    python reproduce.py --audit
    python reproduce.py --table7
    python reproduce.py --table8
    python reproduce.py --table9
    python reproduce.py --table10
    python reproduce.py --table11
    python reproduce.py --table12
    python reproduce.py --figures
"""

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from scripts.audit_dataset_splits import run_audit
from scripts.run_table7 import generate_table7
from scripts.run_table8 import generate_table8
from scripts.run_table9 import generate_table9
from scripts.run_table10 import generate_table10
from scripts.run_table11 import generate_table11
from scripts.run_table12 import generate_table12
from scripts.plot_figures import run_figures_and_reliability


def main():
    parser = argparse.ArgumentParser(
        description="Master reproduction CLI for 'Aging Gracefully? Reliability Profiling and Temporal Forensics'.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python reproduce.py --all          # Run complete end-to-end replication
  python reproduce.py --table9       # Reproduce Table 9 (Temporal Macro-F1 & deltas)
  python reproduce.py --figures      # Generate Figure 4 and Figure 5
  python reproduce.py --audit        # Verify SHA-256 checksums and cohort integrity
"""
    )
    parser.add_argument("--all", action="store_true",
                        help="Execute complete replication suite: audit, tables 7-12, and publication figures")
    parser.add_argument("--audit", action="store_true",
                        help="Run cryptographic audit on 65 split files and prediction cohorts")
    parser.add_argument("--table7", action="store_true",
                        help="Reproduce Table 7 (Descriptive Language Statistics)")
    parser.add_argument("--table8", action="store_true",
                        help="Reproduce Table 8 (Canonical Within-Period Performance)")
    parser.add_argument("--table9", action="store_true",
                        help="Reproduce Table 9 (Temporal Robustness & Delta_j = Target - Ref)")
    parser.add_argument("--table10", action="store_true",
                        help="Reproduce Table 10 (Period-Specific ECE, R_80, and R_100)")
    parser.add_argument("--table11", action="store_true",
                        help="Reproduce Table 11 (P5 Abstention & Prompt Sensitivity S_prompt)")
    parser.add_argument("--table12", action="store_true",
                        help="Reproduce Table 12 (Temporal Forensics Movement Evidence)")
    parser.add_argument("--figures", action="store_true",
                        help="Generate Figure 4 (Risk-Coverage Rankings) and Figure 5 (Movement vs Response)")

    parser.add_argument("--splits-dir", type=Path, default=BASE_DIR / "data" / "splits",
                        help="Path to dataset splits directory")
    parser.add_argument("--archives-dir", type=Path, default=BASE_DIR / "archives",
                        help="Path to prediction archives directory")
    parser.add_argument("--tables-dir", type=Path, default=BASE_DIR / "outputs" / "tables",
                        help="Output directory for generated LaTeX and CSV tables")
    parser.add_argument("--figures-dir", type=Path, default=BASE_DIR / "outputs" / "figures",
                        help="Output directory for generated figures")

    args = parser.parse_args()

    # If no flags are provided, show help
    if not any([args.all, args.audit, args.table7, args.table8, args.table9,
                args.table10, args.table11, args.table12, args.figures]):
        parser.print_help()
        sys.exit(1)

    print("=" * 85)
    print("REPLICATION SUITE: Aging Gracefully? Reliability Profiling & Temporal Forensics")
    print("=" * 85)

    success = True

    if args.all or args.audit:
        ok = run_audit(args.splits_dir, args.archives_dir)
        success = success and ok

    if args.all or args.table7:
        generate_table7(args.splits_dir, args.tables_dir)

    if args.all or args.table8:
        generate_table8(args.archives_dir, args.tables_dir)

    if args.all or args.table9:
        generate_table9(args.archives_dir, args.tables_dir)

    if args.all or args.table10:
        generate_table10(args.archives_dir, args.tables_dir)

    if args.all or args.table11:
        generate_table11(args.archives_dir, args.tables_dir)

    if args.all or args.table12:
        generate_table12(args.splits_dir, args.tables_dir)

    if args.all or args.figures:
        run_figures_and_reliability(args.archives_dir, args.splits_dir, args.figures_dir, args.tables_dir)

    print("\n" + "=" * 85)
    print("REPLICATION EXECUTION COMPLETED")
    print(f"Artifacts written to:")
    print(f"  - Tables:  {args.tables_dir.resolve()}")
    print(f"  - Figures: {args.figures_dir.resolve()}")
    print("=" * 85 + "\n")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
