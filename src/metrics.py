"""Mathematical metrics computation for classification, calibration, selective prediction, and temporal movement.

Guarantees exact parity with the paper's canonical mathematical definitions (Equations 1-17, Table 3):
- Precision, Recall, Class-F1, Macro-F1 (Equations 1-4)
- Accuracy (Equation 5)
- Temporal Macro-F1 Change Delta (Equation 6): Target minus Reference (positive = improvement, negative = decline)
- Expected Calibration Error with 10 bins (Equations 7-9): ECE-10
- Prompt Sensitivity (Equation 10): S_prompt = max(F1) - min(F1) across prompt configurations
- Selective Risk at Coverage c (Equation 11): R(c) with exact fractional tie-handling
- Explicit Abstention Metrics: Abstain Rate A, Retained Coverage C_ret, Post-abstention Error R_ret
"""

import math
import statistics
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np


def accuracy(
    correct: Sequence[Union[bool, int]]
) -> float:
    """Compute classification accuracy (Equation 5): (1/N) * sum I(y_hat == y)."""
    if not correct:
        raise ValueError("Sequence cannot be empty.")
    return float(np.mean(correct))


def macro_f1(
    rows: Sequence[Dict[str, str]],
    labels: Sequence[str],
    gold_key: str,
    pred_key: str
) -> float:
    """Compute unweighted Macro-F1 across class labels (Equations 1-4): (1/|Y|) * sum F1_k."""
    if not rows:
        raise ValueError("Prediction rows cannot be empty.")
    f1_scores = []
    for label in labels:
        tp = sum(r[gold_key] == label and r[pred_key] == label for r in rows)
        fp = sum(r[gold_key] != label and r[pred_key] == label for r in rows)
        fn = sum(r[gold_key] == label and r[pred_key] != label for r in rows)
        denom = 2 * tp + fp + fn
        f1_scores.append((2.0 * tp / denom) if denom > 0 else 0.0)
    return float(statistics.mean(f1_scores))


def temporal_delta(
    f1_target: float,
    f1_reference: float
) -> float:
    """Compute temporal robustness delta (Equation 6): Delta_j = MacroF1(T_j) - MacroF1(T_0).
    
    Positive indicates performance improvement on target; negative indicates degradation.
    """
    return float(f1_target - f1_reference)


def prompt_sensitivity(
    f1_scores_across_prompts: Sequence[float]
) -> float:
    """Compute prompt-configuration sensitivity (Equation 10): S_prompt = max(F1_p) - min(F1_p)."""
    if not f1_scores_across_prompts:
        raise ValueError("Prompt F1 scores list cannot be empty.")
    return float(max(f1_scores_across_prompts) - min(f1_scores_across_prompts))


def ece_10(
    correct: Sequence[Union[bool, int, float]],
    confidence: Sequence[float]
) -> float:
    """Compute Expected Calibration Error (ECE) using 10 equal-width bins (Equations 7-9).
    
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
    """Compute selective risk R(c) across coverage levels 20%..100% in 1% steps (Equation 11).
    
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


def abstention_metrics(
    true_labels: Sequence[str],
    pred_labels: Sequence[str],
    abstained: Sequence[bool]
) -> Dict[str, float]:
    """Compute explicit abstention metrics (Table 3):
    
    - total_n: total number of attempted examples
    - abstained_count: number of abstained predictions (a_i = 1)
    - abstain_rate: A = (1/N) * sum a_i
    - retained_coverage: C_ret = 1 - A
    - incorrect_among_abstained: number of abstained examples where pred != true
    - error_before_pct: percentage error before abstention
    - error_after_pct: percentage error on retained predictions
    - error_reduction_pp: percentage point reduction in error (before - after)
    """
    n = len(true_labels)
    if n == 0 or len(pred_labels) != n or len(abstained) != n:
        raise ValueError("Input sequences must be non-empty and of identical length.")

    wrong = [t != p for t, p in zip(true_labels, pred_labels)]
    abst_cnt = sum(abstained)
    retained_wrong = [w for w, a in zip(wrong, abstained) if not a]
    abst_wrong = sum(w for w, a in zip(wrong, abstained) if a)

    err_before = (sum(wrong) / n) * 100.0
    err_after = (sum(retained_wrong) / len(retained_wrong) * 100.0) if retained_wrong else 0.0

    return {
        "total_n": n,
        "abstained_count": abst_cnt,
        "abstain_rate": abst_cnt / n,
        "retained_coverage": 1.0 - (abst_cnt / n),
        "incorrect_among_abstained": abst_wrong,
        "error_before_pct": err_before,
        "error_after_pct": err_after,
        "error_reduction_pp": err_before - err_after
    }


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
