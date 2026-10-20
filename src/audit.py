"""Auditing and validation utilities for splits, predictions, and provenance.

Validates that:
- All 65 dataset split CSVs match their authoritative SHA-256 checksums.
- Evaluated example IDs and text digests match exactly across model families.
- Class balance is preserved (N per class, 2N total per evaluation split).
- Supervised confidence scores reflect maximum predicted class probabilities.
"""

import hashlib
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Union

from src.loader import ArchiveManager


def audit_splits(splits_dir: Union[str, Path]) -> Dict[str, any]:
    """Verify SHA-256 checksums for all splits listed in checksums_sha256.txt."""
    sdir = Path(splits_dir)
    checksum_file = sdir / "checksums_sha256.txt"
    if not checksum_file.is_file():
        raise FileNotFoundError(f"Missing checksum file: {checksum_file}")

    total_checked = 0
    passed = 0
    failed = []

    with checksum_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            expected_hash, rel_path = parts
            rel_path = rel_path.strip().replace("\\", "/")
            if rel_path.startswith("*"):
                rel_path = rel_path[1:]

            target_file = sdir / rel_path
            if not target_file.is_file():
                failed.append((rel_path, "FILE_NOT_FOUND", expected_hash))
                continue

            h = hashlib.sha256()
            with target_file.open("rb") as tf:
                while chunk := tf.read(65536):
                    h.update(chunk)
            actual_hash = h.hexdigest()

            total_checked += 1
            if actual_hash.lower() == expected_hash.lower():
                passed += 1
            else:
                failed.append((rel_path, actual_hash, expected_hash))

    return {
        "total_checked": total_checked,
        "passed": passed,
        "failed": failed,
        "all_passed": (len(failed) == 0 and total_checked > 0)
    }


def audit_predictions_and_metrics(manager: ArchiveManager) -> Dict[str, any]:
    """Perform a comprehensive cross-model identity and probability alignment audit."""
    seeds = range(42, 47)
    periods = ("within", "short", "long")
    settings = ("Climate n500", "Yelp n500", "Yelp n1000")
    models = ("distilbert", "tfidf_lr", "phi3.5:latest", "qwen2.5:3b-instruct")

    total_conditions = 0
    anomalies_abstain = 0

    for setting in settings:
        n = 1000 if setting.endswith("1000") else 500
        labels = ("BELIEVER", "DENIER") if setting.startswith("Climate") else ("NEG", "POS")
        
        for period in periods:
            for seed in seeds:
                matched_ids = None
                for model in models:
                    if setting.startswith("Climate"):
                        rows = manager.load_climate_predictions(model, seed, period)
                        gold_col, pred_col, digest_col = "true_label", "predicted_label", "text_sha256"
                    else:
                        rows = manager.load_yelp_predictions(model, n, seed, period)
                        gold_col, pred_col, digest_col = "gold_label", "pred_label", "text_sha1"

                    # 1. Check size and class balance
                    if len(rows) != 2 * n:
                        raise ValueError(f"Size mismatch in {setting} {period} seed {seed} {model}: got {len(rows)}, expected {2*n}")
                    counts = Counter(r[gold_col] for r in rows)
                    if counts != {lab: n for lab in labels}:
                        raise ValueError(f"Balance mismatch in {setting} {period} seed {seed} {model}: {counts}")

                    # 2. Check matched IDs
                    ids = {(r["example_id"], r[digest_col], r[gold_col]) for r in rows}
                    if len(ids) != len(rows):
                        raise ValueError(f"Duplicate example IDs in {setting} {period} seed {seed} {model}")
                    if matched_ids is not None and matched_ids != ids:
                        raise ValueError(f"Example set divergence across models in {setting} {period} seed {seed} {model}")
                    matched_ids = ids

                    # 3. Check probability alignment for supervised models
                    if model in ("distilbert", "tfidf_lr"):
                        if setting.startswith("Climate"):
                            for r in rows:
                                conf = float(r["confidence"])
                                max_p = max(float(r["prob_believer"]), float(r["prob_denier"]))
                                if not math.isclose(conf, max_p, abs_tol=1e-5):
                                    raise ValueError(f"Confidence/prob mismatch in Climate {model}: {conf} vs {max_p}")
                        else:
                            for r in rows:
                                conf = float(r["confidence"])
                                p_pos = float(r["prob_pos"])
                                max_p = max(p_pos, 1.0 - p_pos)
                                if not math.isclose(conf, max_p, abs_tol=1e-5):
                                    raise ValueError(f"Confidence/prob mismatch in Yelp {model}: {conf} vs {max_p}")
                    elif not setting.startswith("Climate"):
                        anomalies_abstain += sum(r.get("abstain", "").lower() == "true" for r in rows)

                    total_conditions += 1

    return {
        "conditions_verified": total_conditions,
        "abstain_rows_scored": anomalies_abstain,
        "status": "PASS"
    }
