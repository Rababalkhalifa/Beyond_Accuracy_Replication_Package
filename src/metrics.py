"""Mathematical metrics computation for classification, calibration, and selective risk.

Guarantees exact parity with the paper's canonical definitions:
- Macro-F1: Balanced unweighted average of per-class F1 scores.
- ECE-10: 10 equal-width bins over [0, 1] with right-boundary closure for 1.0.
- Selective Risk: Exact expected error rate for fractional tie-handling.
"""

import math
import statistics
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np


def macro_f1(
    rows: Sequence[Dict[str, str]],
    labels: Sequence[str],
    gold_key: str,
    pred_key: str
) -> float:
    """Compute unweighted Macro-F1 across the given class labels."""
    f1_scores = []
    for label in labels:
        tp = sum(r[gold_key] == label and r[pred_key] == label for r in rows)
        fp = sum(r[gold_key] != label and r[pred_key] == label for r in rows)
        fn = sum(r[gold_key] == label and r[pred_key] != label for r in rows)
        denom = 2 * tp + fp + fn
        f1_scores.append((2.0 * tp / denom) if denom > 0 else 0.0)
    return float(statistics.mean(f1_scores))


def ece_10(
    correct: Sequence[Union[bool, int, float]],
    confidence: Sequence[float]
) -> float:
    """Compute Expected Calibration Error (ECE) using 10 equal-width bins.
    
    Bins span [0.0, 1.0] with width 0.1. An example with confidence 1.0 falls
    into the 10th bin [0.9, 1.0].
    """
    conf = np.asarray(confidence, dtype=float)
    corr = np.asarray(correct, dtype=float)

    if not np.isfinite(conf).all() or not ((conf >= 0.0) & (conf <= 1.0)).all():
        raise ValueError("Confidence values must be finite and within [0, 1].")

    bounds = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    n_total = len(conf)

    for bin_no in range(10):
        lo, hi = bounds[bin_no], bounds[bin_no + 1]
        if bin_no == 9:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        
        if mask.any():
            bin_size = int(mask.sum())
            bin_acc = float(corr[mask].mean())
            bin_conf = float(conf[mask].mean())
            ece += (bin_size / n_total) * abs(bin_acc - bin_conf)

    return float(ece)


def risk_sweep(
    scores: Sequence[float],
    errors: Sequence[Union[bool, int, float]]
) -> List[float]:
    """Compute expected error rate across coverage levels 20%..100% (step 1%).
    
    Implements fractional inclusion of tied confidence scores, representing the
    expected error under uniform random tie-breaking.
    """
    conf = np.asarray(scores, dtype=float)
    wrong = np.asarray(errors, dtype=float)

    groups, indices = np.unique(conf, return_inverse=True)
    counts = np.bincount(indices)[::-1]
    group_errors = np.bincount(indices, weights=wrong)[::-1]
    cumulative_count = counts.cumsum()
    cumulative_error = group_errors.cumsum()

    result = []
    n = len(conf)
    for pct in range(20, 101):
        kept = n * pct // 100
        boundary = int(np.searchsorted(cumulative_count, kept, side="left"))
        before = int(cumulative_count[boundary - 1]) if boundary > 0 else 0
        previous_error = float(cumulative_error[boundary - 1]) if boundary > 0 else 0.0
        fraction = (kept - before) / int(counts[boundary])
        retained_error = (previous_error + fraction * float(group_errors[boundary])) / kept
        result.append(float(retained_error))

    return result


def calc_condition_metrics(
    rows: Sequence[Dict[str, str]],
    gold_col: str,
    pred_col: str,
    labels: Sequence[str],
    conf_col: str,
    latency_col: Optional[str] = None
) -> Dict[str, float]:
    """Compute full suite of classification & calibration metrics for a single condition."""
    if not rows:
        raise ValueError("Prediction rows cannot be empty.")

    correct = [r[gold_col] == r[pred_col] for r in rows]
    conf = [float(r[conf_col]) for r in rows]

    acc = float(statistics.mean(correct))
    f1 = macro_f1(rows, labels, gold_col, pred_col)
    ece = ece_10(correct, conf)

    res = {
        "accuracy": acc,
        "macro_f1": f1,
        "ece": ece
    }

    if latency_col:
        times = [float(r[latency_col]) for r in rows if latency_col in r and r[latency_col] != ""]
        if times:
            res["latency_mean_sec"] = float(statistics.mean(times))

    return res
