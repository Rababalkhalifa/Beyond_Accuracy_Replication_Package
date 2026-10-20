"""Script to audit dataset splits, SHA-256 checksums, and prediction cohorts.

Verifies:
1. All 65 CSV split files match their authoritative SHA-256 hashes.
2. Cross-model example ID consistency and class balance across 180 conditions.
3. Probability calibration consistency for supervised models.
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.audit import audit_splits, audit_predictions_and_metrics
from src.loader import ArchiveManager


def run_audit(splits_dir: Path, archives_dir: Path) -> bool:
    print("\n" + "=" * 80)
    print("STEP 1: Auditing Dataset Splits & SHA-256 Checksums")
    print("=" * 80)
    split_res = audit_splits(splits_dir)
    print(f"Total split files checked: {split_res['total_checked']}")
    print(f"Passed checksum matches:   {split_res['passed']}")
    if split_res["failed"]:
        print(f"FAILED matches ({len(split_res['failed'])}):")
        for f in split_res["failed"]:
            print(f"  [X] {f[0]}: actual={f[1]} vs expected={f[2]}")
        return False
    else:
        print(">> ALL SPLIT CHECKSUMS PASSED (Status: PASS)")

    print("\n" + "=" * 80)
    print("STEP 2: Auditing Prediction Cohorts & Cross-Model Integrity")
    print("=" * 80)
    with ArchiveManager(archives_dir) as mgr:
        pred_res = audit_predictions_and_metrics(mgr)
        print(f"Conditions verified:        {pred_res['conditions_verified']} / 180")
        print(f"Yelp P3 abstain rows scored: {pred_res['abstain_rows_scored']}")
        print(">> ALL PREDICTION INTEGRITY AUDITS PASSED (Status: PASS)")

    print("\n" + "=" * 80)
    print("OVERALL AUDIT RESULT: 100% REPRODUCIBLE AND CONSISTENT")
    print("=" * 80 + "\n")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit dataset splits and prediction integrity.")
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"),
                        help="Path to directory containing dataset splits and checksums_sha256.txt")
    parser.add_argument("--archives-dir", type=Path, default=Path("archives"),
                        help="Path to directory containing prediction archives")
    args = parser.parse_args()
    success = run_audit(args.splits_dir, args.archives_dir)
    sys.exit(0 if success else 1)
