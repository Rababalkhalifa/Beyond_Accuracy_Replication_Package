"""Temporal forensics calculation for shifted evaluation horizons (Table 12, Equations 12-17).

Quantifies dataset-side movement between within-period reference (T_0) and shifted targets (T_1, T_2):
- Lexical novelty (Equation 12): |V_target \\ V_within| / |V_target|
- Distributional movement (Equation 13): JSD_2(P, Q) base-2 Jensen-Shannon divergence
- Top-term concentration change (Equation 14): Delta C_100 = C_100(target) - C_100(within)
- Document-length change (Equation 15): Delta L = mean_length(target) - mean_length(within)
- Vocabulary-overlap loss (Equation 16): 1 - (|V_within \\cap V_target| / |V_within \\cup V_target|)
- Class-marker asymmetry (Equation 17): A_marker = |churn_c1 - churn_c2| based on top-100 smoothed log-odds markers
"""

import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

TOKEN_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)


def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase word tokens."""
    return TOKEN_RE.findall(str(text).lower())


def token_counter(df: pd.DataFrame) -> Counter:
    """Compute token frequency counter for a dataframe text column."""
    counter = Counter()
    for text in df["text"].fillna(""):
        counter.update(tokenize(text))
    return counter


def document_lengths(df: pd.DataFrame) -> np.ndarray:
    """Compute array of document lengths in tokens."""
    return np.array([len(tokenize(t)) for t in df["text"].fillna("")], dtype=float)


def lexical_novelty(within_counter: Counter, target_counter: Counter) -> float:
    """Equation 12: Lexical novelty = |V_target \\ V_within| / |V_target|."""
    vw = set(within_counter)
    vt = set(target_counter)
    return len(vt - vw) / len(vt) if vt else 0.0


def overlap_loss(within_counter: Counter, target_counter: Counter) -> float:
    """Equation 16: Vocabulary-overlap loss = 1 - (|V_within \\cap V_target| / |V_within \\cup V_target|)."""
    vw = set(within_counter)
    vt = set(target_counter)
    union = vw | vt
    return (1.0 - (len(vw & vt) / len(union))) if union else 0.0


def js_divergence(within_counter: Counter, target_counter: Counter) -> float:
    """Equation 13: Jensen-Shannon divergence with base-2 logarithms.
    
    scipy.spatial.distance.jensenshannon returns the JS distance (square root of JS divergence).
    Squaring it yields the exact base-2 JS divergence JSD_2(P, Q) in [0, 1].
    """
    vocab = sorted(set(within_counter) | set(target_counter))
    if not vocab:
        return 0.0

    p = np.array([within_counter.get(t, 0) for t in vocab], dtype=float)
    q = np.array([target_counter.get(t, 0) for t in vocab], dtype=float)

    if p.sum() == 0 or q.sum() == 0:
        return 0.0

    p /= p.sum()
    q /= q.sum()

    return float(jensenshannon(p, q, base=2.0) ** 2)


def top100_mass(counter: Counter) -> float:
    """Compute fraction of total token mass accounted for by the 100 most frequent tokens."""
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return sum(count for _, count in counter.most_common(100)) / total


def class_log_odds_markers(
    df: pd.DataFrame,
    top_k: int = 100,
    alpha: float = 0.5
) -> Dict[str, set]:
    """Compute top-k class marker vocabulary using smoothed log-odds ratio with prior alpha."""
    labels = sorted(df["label"].dropna().astype(str).str.strip().unique())
    if len(labels) != 2:
        raise ValueError(f"Expected binary classification labels, found: {labels}")

    counters = {}
    totals = {}
    vocab = set()

    for label in labels:
        subset = df[df["label"].astype(str).str.strip() == label]
        c = token_counter(subset)
        counters[label] = c
        totals[label] = sum(c.values())
        vocab |= set(c)

    vocab_size = max(len(vocab), 1)
    markers = {}

    for label in labels:
        other = labels[1] if label == labels[0] else labels[0]
        scored = []
        for token in sorted(vocab):
            a = counters[label].get(token, 0)
            b = counters[other].get(token, 0)

            lo = (
                math.log((a + alpha) / (totals[label] - a + alpha * vocab_size))
                - math.log((b + alpha) / (totals[other] - b + alpha * vocab_size))
            )
            scored.append((token, lo))

        scored.sort(key=lambda x: (-x[1], x[0]))
        markers[label] = {token for token, _ in scored[:top_k]}

    return markers


def class_marker_metrics(
    within_df: pd.DataFrame,
    target_df: pd.DataFrame,
    top_k: int = 100
) -> Tuple[List[str], Dict[str, float], float]:
    """Equation 17: Class-marker asymmetry A_marker = |churn_c1 - churn_c2|."""
    within_markers = class_log_odds_markers(within_df, top_k=top_k)
    target_markers = class_log_odds_markers(target_df, top_k=top_k)

    labels = sorted(set(within_markers) & set(target_markers))
    if len(labels) != 2:
        raise RuntimeError(f"Expected two common class labels, found {labels}")

    churn = {}
    for label in labels:
        w_set = within_markers[label]
        t_set = target_markers[label]
        union = w_set | t_set
        churn[label] = (1.0 - len(w_set & t_set) / len(union)) if union else 0.0

    asymmetry = abs(churn[labels[0]] - churn[labels[1]])
    return labels, churn, asymmetry


def compute_condition_forensics(
    within_df: pd.DataFrame,
    target_df: pd.DataFrame
) -> Dict[str, float]:
    """Compute all 6 temporal forensics measures for a single seed and target split."""
    within_counter = token_counter(within_df)
    target_counter = token_counter(target_df)
    within_lengths = document_lengths(within_df)
    target_lengths = document_lengths(target_df)

    labels, churn, asymmetry = class_marker_metrics(within_df, target_df)

    nov = lexical_novelty(within_counter, target_counter)
    jsd = js_divergence(within_counter, target_counter)
    d_top100 = top100_mass(target_counter) - top100_mass(within_counter)
    d_length = float(target_lengths.mean() - within_lengths.mean())
    ovl = overlap_loss(within_counter, target_counter)

    return {
        "lexical_novelty": nov,
        "js_divergence": jsd,
        "top100_mass_change": d_top100,
        "mean_length_change": d_length,
        "overlap_loss": ovl,
        "class_marker_asymmetry": asymmetry,
        "labels": labels,
        "churn": churn
    }


def compute_all_forensics(
    splits_dir: Union[str, Path],
    seeds: Sequence[int] = (42, 43, 44, 45, 46)
) -> List[Dict[str, any]]:
    """Compute Table 12 temporal forensics across all settings and horizons."""
    splits_path = Path(splits_dir)

    settings = [
        ("Climate n500", "climate", 500),
        ("Yelp n500", "yelp", 500),
        ("Yelp n1000", "yelp", 1000)
    ]

    summary_rows = []

    for setting_name, folder, n in settings:
        for target_name in ("short", "long"):
            seed_metrics = []

            for seed in seeds:
                if folder == "climate":
                    within_fn = "within_2010_n500_per_label_total1000.csv"
                    target_fn = (
                        "short_2013_n500_per_label_total1000.csv"
                        if target_name == "short"
                        else "long_2018_2019_n500_per_label_total1000.csv"
                    )
                else:  # yelp
                    within_fn = f"within_2011_2012_n{n}_per_label_total{2*n}.csv"
                    target_fn = (
                        f"short_2015_n{n}_per_label_total{2*n}.csv"
                        if target_name == "short"
                        else f"long_2019_2020_n{n}_per_label_total{2*n}.csv"
                    )

                w_path = splits_path / folder / f"seed_{seed}" / within_fn
                t_path = splits_path / folder / f"seed_{seed}" / target_fn

                within_df = pd.read_csv(w_path)
                target_df = pd.read_csv(t_path)

                m = compute_condition_forensics(within_df, target_df)
                seed_metrics.append(m)

            # Five-seed aggregation
            nov_vals = [s["lexical_novelty"] for s in seed_metrics]
            jsd_vals = [s["js_divergence"] for s in seed_metrics]
            dC_vals = [s["top100_mass_change"] for s in seed_metrics]
            dL_vals = [s["mean_length_change"] for s in seed_metrics]
            ovl_vals = [s["overlap_loss"] for s in seed_metrics]
            asym_vals = [s["class_marker_asymmetry"] for s in seed_metrics]

            row = {
                "dataset_setting": setting_name,
                "target": target_name,
                "lexical_novelty_mean": float(np.mean(nov_vals)),
                "lexical_novelty_sd": float(np.std(nov_vals, ddof=1)),
                "js_divergence_mean": float(np.mean(jsd_vals)),
                "js_divergence_sd": float(np.std(jsd_vals, ddof=1)),
                "top100_mass_change_mean": float(np.mean(dC_vals)),
                "top100_mass_change_sd": float(np.std(dC_vals, ddof=1)),
                "mean_length_change_mean": float(np.mean(dL_vals)),
                "mean_length_change_sd": float(np.std(dL_vals, ddof=1)),
                "overlap_loss_mean": float(np.mean(ovl_vals)),
                "overlap_loss_sd": float(np.std(ovl_vals, ddof=1)),
                "class_marker_asymmetry_mean": float(np.mean(asym_vals)),
                "class_marker_asymmetry_sd": float(np.std(asym_vals, ddof=1)),
                "task_native_labels": "/".join(seed_metrics[0]["labels"])
            }
            summary_rows.append(row)

    return summary_rows
