from pathlib import Path
from collections import Counter
import math
import re

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon


ROOT = Path("/mnt/c/Research/Beyond_Accuracy_Reproduction")

CLIMATE_ROOT = Path(
    "/mnt/c/Research/Beyond_Accuracy_Replication_Package/data/splits/climate"
)

YELP_ROOT = (
    ROOT
    / "out"
    / "sentiment_adapted_on_frozen_20260722"
    / "recovered_shared_splits"
)

OUT = ROOT / "RQ5_reaudit"
OUT.mkdir(exist_ok=True)

SEEDS = [42, 43, 44, 45, 46]


# =========================================================
# 1. TOKENIZATION
# =========================================================
#
# IMPORTANT:
# Use the same lowercase regex tokenization already adopted
# in the corrected descriptor reconciliation:
#
#     re.findall(r"\b\w+\b", text.lower())
#
# This keeps RQ5 ΔL and ΔC100 consistent with Table 7.
# =========================================================

TOKEN_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)


def tokenize(text):
    return TOKEN_RE.findall(str(text).lower())


def token_counter(df):
    counter = Counter()

    for text in df["text"].fillna(""):
        counter.update(tokenize(text))

    return counter


def document_lengths(df):
    return np.array(
        [
            len(tokenize(text))
            for text in df["text"].fillna("")
        ],
        dtype=float,
    )


# =========================================================
# 2. BASIC FORENSIC METRICS
# =========================================================

def lexical_novelty(within_counter, target_counter):
    """
    Fraction of target vocabulary types not observed
    in the within-period vocabulary.
    """
    vw = set(within_counter)
    vt = set(target_counter)

    if not vt:
        return np.nan

    return len(vt - vw) / len(vt)


def overlap_loss(within_counter, target_counter):
    """
    1 - vocabulary Jaccard overlap.
    """
    vw = set(within_counter)
    vt = set(target_counter)

    union = vw | vt

    if not union:
        return np.nan

    return 1 - len(vw & vt) / len(union)


def js_divergence(within_counter, target_counter):
    """
    Jensen-Shannon divergence with base 2.
    scipy returns JS distance, so square it.
    """
    vocab = sorted(
        set(within_counter) | set(target_counter)
    )

    if not vocab:
        return np.nan

    p = np.array(
        [within_counter.get(t, 0) for t in vocab],
        dtype=float,
    )

    q = np.array(
        [target_counter.get(t, 0) for t in vocab],
        dtype=float,
    )

    if p.sum() == 0 or q.sum() == 0:
        return np.nan

    p /= p.sum()
    q /= q.sum()

    return float(
        jensenshannon(
            p,
            q,
            base=2.0,
        ) ** 2
    )


def top100_mass(counter):
    total = sum(counter.values())

    if total == 0:
        return np.nan

    top_mass = sum(
        count
        for _, count in counter.most_common(100)
    )

    return top_mass / total


# =========================================================
# 3. CLASS MARKERS
# =========================================================
#
# This is deliberately task-native.
#
# Climate:
#   BELIEVER vs DENIER
#
# Yelp:
#   NEG vs POS
#
# No conversion to POS/NEG is performed for Climate.
# =========================================================

def class_log_odds_markers(
    df,
    top_k=100,
    alpha=0.5,
):
    labels = sorted(
        df["label"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
    )

    if len(labels) != 2:
        raise ValueError(
            f"Expected exactly two labels, found {labels}"
        )

    counters = {}
    totals = {}
    vocab = set()

    for label in labels:
        subset = df[
            df["label"].astype(str).str.strip() == label
        ]

        counter = token_counter(subset)

        counters[label] = counter
        totals[label] = sum(counter.values())
        vocab |= set(counter)

    vocab_size = max(len(vocab), 1)

    markers = {}

    for label in labels:

        other = (
            labels[1]
            if label == labels[0]
            else labels[0]
        )

        scored = []

        for token in vocab:

            a = counters[label].get(token, 0)
            b = counters[other].get(token, 0)

            log_odds = math.log(
                (a + alpha)
                /
                (
                    totals[label]
                    - a
                    + alpha * vocab_size
                )
            ) - math.log(
                (b + alpha)
                /
                (
                    totals[other]
                    - b
                    + alpha * vocab_size
                )
            )

            scored.append(
                (token, log_odds)
            )

        scored.sort(
            key=lambda x: x[1],
            reverse=True,
        )

        markers[label] = {
            token
            for token, score
            in scored[:top_k]
        }

    return markers


def jaccard_distance(a, b):
    union = a | b

    if not union:
        return np.nan

    return 1 - len(a & b) / len(union)


def class_marker_metrics(
    within_df,
    target_df,
):
    within_markers = class_log_odds_markers(
        within_df,
        top_k=100,
    )

    target_markers = class_log_odds_markers(
        target_df,
        top_k=100,
    )

    labels = sorted(
        set(within_markers)
        & set(target_markers)
    )

    if len(labels) != 2:
        raise RuntimeError(
            f"Expected two common labels, found {labels}"
        )

    churn = {}

    for label in labels:
        churn[label] = jaccard_distance(
            within_markers[label],
            target_markers[label],
        )

    asymmetry = abs(
        churn[labels[0]]
        - churn[labels[1]]
    )

    return labels, churn, asymmetry


# =========================================================
# 4. PATH DEFINITIONS
# =========================================================

def climate_path(seed, split):

    files = {
        "within":
            "within_2010_n500_per_label_total1000.csv",

        "short":
            "short_2013_n500_per_label_total1000.csv",

        "long":
            "long_2018_2019_n500_per_label_total1000.csv",
    }

    return (
        CLIMATE_ROOT
        / f"seed_{seed}"
        / files[split]
    )


def yelp_path(seed, n, split):

    return (
        YELP_ROOT
        / f"seed{seed}"
        / f"n{n}_per_label"
        / f"{split}.csv"
    )


# =========================================================
# 5. COMPUTE ALL SIX RQ5 MEASURES
# =========================================================

rows = []


settings = [
    ("Climate n500", 500),
    ("Yelp n500", 500),
    ("Yelp n1000", 1000),
]


for setting, n in settings:

    print("\n" + "=" * 70)
    print(setting)
    print("=" * 70)

    for seed in SEEDS:

        if setting == "Climate n500":
            within_path = climate_path(
                seed,
                "within",
            )

        else:
            within_path = yelp_path(
                seed,
                n,
                "within",
            )

        within = pd.read_csv(within_path)

        within_counter = token_counter(within)
        within_lengths = document_lengths(within)

        for target_name in [
            "short",
            "long",
        ]:

            if setting == "Climate n500":
                target_path = climate_path(
                    seed,
                    target_name,
                )

            else:
                target_path = yelp_path(
                    seed,
                    n,
                    target_name,
                )

            target = pd.read_csv(target_path)

            target_counter = token_counter(target)
            target_lengths = document_lengths(target)

            labels, churn, asymmetry = (
                class_marker_metrics(
                    within,
                    target,
                )
            )

            d_length = (
                target_lengths.mean()
                - within_lengths.mean()
            )

            d_top100 = (
                top100_mass(target_counter)
                - top100_mass(within_counter)
            )

            row = {
                "dataset_setting": setting,
                "seed": seed,
                "target": target_name,

                "lexical_novelty":
                    lexical_novelty(
                        within_counter,
                        target_counter,
                    ),

                "js_divergence":
                    js_divergence(
                        within_counter,
                        target_counter,
                    ),

                "top100_mass_change":
                    d_top100,

                "mean_length_change":
                    d_length,

                "overlap_loss":
                    overlap_loss(
                        within_counter,
                        target_counter,
                    ),

                "class_label_1":
                    labels[0],

                "class_label_2":
                    labels[1],

                "marker_churn_class_1":
                    churn[labels[0]],

                "marker_churn_class_2":
                    churn[labels[1]],

                "class_marker_asymmetry":
                    asymmetry,

                "within_rows":
                    len(within),

                "target_rows":
                    len(target),

                "within_path":
                    str(within_path),

                "target_path":
                    str(target_path),
            }

            rows.append(row)

            print(
                f"seed={seed} "
                f"{target_name:5s} | "
                f"novelty={row['lexical_novelty']:.6f} "
                f"JSD={row['js_divergence']:.6f} "
                f"dC100={row['top100_mass_change']:+.6f} "
                f"dL={row['mean_length_change']:+.4f} "
                f"overlap={row['overlap_loss']:.6f} "
                f"asym={row['class_marker_asymmetry']:.6f} "
                f"labels={labels}"
            )


seed_level = pd.DataFrame(rows)

seed_level.to_csv(
    OUT / "gate2_rq5_seed_level.csv",
    index=False,
)


# =========================================================
# 6. FIVE-SEED AGGREGATION
# =========================================================

metric_cols = [
    "lexical_novelty",
    "js_divergence",
    "top100_mass_change",
    "mean_length_change",
    "overlap_loss",
    "class_marker_asymmetry",
]


summary_rows = []

for (
    setting,
    target
), group in seed_level.groupby(
    [
        "dataset_setting",
        "target",
    ],
    sort=False,
):

    row = {
        "dataset_setting": setting,
        "target": target,
        "n_seeds": group["seed"].nunique(),
    }

    for metric in metric_cols:

        values = group[metric].astype(float)

        row[f"{metric}_mean"] = (
            values.mean()
        )

        row[f"{metric}_sd"] = (
            values.std(ddof=1)
        )

    labels = sorted(
        set(
            group["class_label_1"]
        )
        | set(
            group["class_label_2"]
        )
    )

    row["task_native_labels"] = (
        "/".join(labels)
    )

    summary_rows.append(row)


summary = pd.DataFrame(summary_rows)


order = {
    "Climate n500": 0,
    "Yelp n500": 1,
    "Yelp n1000": 2,
}

target_order = {
    "short": 0,
    "long": 1,
}

summary["_dataset_order"] = (
    summary["dataset_setting"]
    .map(order)
)

summary["_target_order"] = (
    summary["target"]
    .map(target_order)
)

summary = (
    summary
    .sort_values(
        [
            "_dataset_order",
            "_target_order",
        ]
    )
    .drop(
        columns=[
            "_dataset_order",
            "_target_order",
        ]
    )
)


summary.to_csv(
    OUT / "gate2_rq5_summary.csv",
    index=False,
)


# =========================================================
# 7. DISPLAY FINAL TABLE
# =========================================================

print("\n")
print("=" * 100)
print("RQ5 GATE 2 — FIVE-SEED SUMMARY")
print("=" * 100)

display_cols = [
    "dataset_setting",
    "target",

    "lexical_novelty_mean",
    "lexical_novelty_sd",

    "js_divergence_mean",
    "js_divergence_sd",

    "top100_mass_change_mean",
    "top100_mass_change_sd",

    "mean_length_change_mean",
    "mean_length_change_sd",

    "overlap_loss_mean",
    "overlap_loss_sd",

    "class_marker_asymmetry_mean",
    "class_marker_asymmetry_sd",

    "task_native_labels",
]

print(
    summary[
        display_cols
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# =========================================================
# 8. SANITY CHECKS
# =========================================================

assert len(seed_level) == 30
assert len(summary) == 6

assert (
    seed_level[
        seed_level["dataset_setting"]
        == "Climate n500"
    ]["class_label_1"]
    .isin(
        [
            "BELIEVER",
            "DENIER",
        ]
    )
    .all()
)

assert (
    seed_level[
        seed_level["dataset_setting"]
        == "Climate n500"
    ]["class_label_2"]
    .isin(
        [
            "BELIEVER",
            "DENIER",
        ]
    )
    .all()
)

assert (
    seed_level[
        seed_level["dataset_setting"]
        .str.startswith("Yelp")
    ]["class_label_1"]
    .isin(
        [
            "NEG",
            "POS",
        ]
    )
    .all()
)

assert (
    seed_level[
        seed_level["dataset_setting"]
        .str.startswith("Yelp")
    ]["class_label_2"]
    .isin(
        [
            "NEG",
            "POS",
        ]
    )
    .all()
)


print("\nPASS: RQ5 Gate 2 regeneration completed.")

print(
    "Seed-level:",
    OUT / "gate2_rq5_seed_level.csv",
)

print(
    "Summary:",
    OUT / "gate2_rq5_summary.csv",
)
