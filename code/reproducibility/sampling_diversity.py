#!/usr/bin/env python3
"""Label-blind lexical-diversity ordering for temporal split sampling.

Feature extraction, temporal coverage, clustering, and ranking use only text and
source year. Labels are deliberately excluded. After the ordering is fixed,
labels may be consulted solely to enforce predeclared equal-class quotas.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import normalize


WORD_MAX_FEATURES = 12_000
CHAR_MAX_FEATURES = 12_000
SVD_COMPONENTS = 64
MAX_CLUSTERS = 512
YEAR_WEIGHTING_CHOICES = {"proportional": 1.0, "sqrt": 0.5, "equal": 0.0}


@dataclass(frozen=True)
class DiversityStructureReport:
    rows: int
    target_total: int
    years: tuple[str, ...]
    n_years: int
    n_clusters: int
    n_year_cluster_cells: int
    word_max_features: int
    char_max_features: int
    svd_components: int
    structure_sha256: str
    method: str = "text_year_tfidf_svd_minibatch_kmeans"
    labels_used_in_structure: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "target_total": self.target_total,
            "years": list(self.years),
            "n_years": self.n_years,
            "n_clusters": self.n_clusters,
            "n_year_cluster_cells": self.n_year_cluster_cells,
            "word_max_features": self.word_max_features,
            "char_max_features": self.char_max_features,
            "svd_components": self.svd_components,
            "structure_sha256": self.structure_sha256,
            "method": self.method,
            "labels_used_in_structure": self.labels_used_in_structure,
        }


@dataclass(frozen=True)
class DiversityOrderReport:
    structure: DiversityStructureReport
    label_blind_order_sha256: str
    year_weighting: str
    target_year_weights: dict[str, float]
    ordering_method: str
    labels_used_in_ordering: bool = False

    @property
    def method(self) -> str:
        return f"{self.structure.method}+{self.ordering_method}"

    @property
    def n_clusters(self) -> int:
        return self.structure.n_clusters

    @property
    def n_year_cluster_cells(self) -> int:
        return self.structure.n_year_cluster_cells

    def as_dict(self) -> dict[str, Any]:
        payload = self.structure.as_dict()
        payload.update(
            {
                "label_blind_order_sha256": self.label_blind_order_sha256,
                "year_weighting": self.year_weighting,
                "target_year_weights": self.target_year_weights,
                "ordering_method": self.ordering_method,
                "labels_used_in_ordering": self.labels_used_in_ordering,
                "full_method": self.method,
            }
        )
        return payload


def _stable_int(*parts: Any) -> int:
    payload = "||".join(str(part) for part in parts)
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16)


def _hash_lines(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _choose_cluster_count(n_rows: int, target_total: int, n_years: int) -> int:
    if n_rows <= 2:
        return 1
    diversity_goal = max(int(round(math.sqrt(n_rows))), max(8, target_total // 4))
    coverage_cap = max(8, target_total // max(2, 2 * max(1, n_years)))
    return int(max(2, min(n_rows, MAX_CLUSTERS, diversity_goal, coverage_cap)))


def _fit_text_embedding(texts: Sequence[str], seed: int) -> tuple[np.ndarray, int]:
    min_df = 2 if len(texts) >= 100 else 1
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=min_df,
                    max_df=0.995,
                    max_features=WORD_MAX_FEATURES,
                    sublinear_tf=True,
                    strip_accents="unicode",
                    dtype=np.float32,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=min_df,
                    max_df=0.995,
                    max_features=CHAR_MAX_FEATURES,
                    sublinear_tf=True,
                    dtype=np.float32,
                ),
            ),
        ]
    )
    matrix = features.fit_transform(list(texts))
    max_components = min(SVD_COMPONENTS, matrix.shape[0] - 1, matrix.shape[1] - 1)
    if max_components >= 2:
        svd = TruncatedSVD(
            n_components=int(max_components),
            n_iter=5,
            random_state=int(seed),
        )
        embedded = svd.fit_transform(matrix).astype(np.float32, copy=False)
        embedded = normalize(embedded, norm="l2", copy=False)
        return np.asarray(embedded, dtype=np.float32), int(max_components)

    dense = matrix.toarray() if issparse(matrix) else np.asarray(matrix)
    dense = normalize(dense.astype(np.float32, copy=False), norm="l2", copy=False)
    return np.asarray(dense, dtype=np.float32), int(dense.shape[1])


def prepare_label_blind_diversity_structure(
    frame: pd.DataFrame,
    *,
    salt: str,
    target_total: int,
    minimum_target_total: int | None = None,
    year_col: str = "source_year",
) -> tuple[pd.DataFrame, DiversityStructureReport]:
    """Fit one reusable text/year diversity structure for all sampling seeds."""
    required = {"text", "text_sha256", "example_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Diversity structure is missing columns: {missing}")
    if len(frame) < target_total:
        raise ValueError(
            f"Cannot select target_total={target_total} from only {len(frame)} rows."
        )

    work = frame.copy().reset_index(drop=True)
    if year_col not in work.columns:
        raise KeyError(f"Diversity structure requires year column {year_col!r}.")
    work[year_col] = work[year_col].astype(str)
    years = tuple(sorted(work[year_col].unique().tolist()))
    n_years = max(1, len(years))

    # Structure seed is fixed by the dataset/period identity, not a label or
    # sampling seed. All five split seeds therefore use the same lexical map.
    structure_seed = _stable_int(salt, "fixed_text_structure") % (2**31 - 1)
    embedded, used_components = _fit_text_embedding(
        work["text"].astype(str).tolist(), seed=structure_seed
    )
    cluster_target = int(minimum_target_total or target_total)
    n_clusters = _choose_cluster_count(len(work), cluster_target, n_years)

    if n_clusters == 1:
        cluster_ids = np.zeros(len(work), dtype=np.int32)
        distances = np.zeros(len(work), dtype=np.float32)
    else:
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=structure_seed,
            batch_size=min(4096, max(256, n_clusters * 8)),
            n_init=3,
            max_iter=120,
            reassignment_ratio=0.01,
        )
        cluster_ids = kmeans.fit_predict(embedded).astype(np.int32)
        centers = kmeans.cluster_centers_[cluster_ids]
        distances = np.sum((embedded - centers) ** 2, axis=1).astype(np.float32)

    work["_diversity_cluster"] = cluster_ids
    work["_diversity_distance"] = distances
    cells = work[[year_col, "_diversity_cluster"]].astype(str).agg("||".join, axis=1)
    structure_digest = _hash_lines(
        work["text_sha256"].astype(str)
        + "||"
        + work[year_col].astype(str)
        + "||"
        + work["_diversity_cluster"].astype(str)
    )
    report = DiversityStructureReport(
        rows=int(len(work)),
        target_total=int(target_total),
        years=years,
        n_years=int(n_years),
        n_clusters=int(n_clusters),
        n_year_cluster_cells=int(cells.nunique()),
        word_max_features=WORD_MAX_FEATURES,
        char_max_features=CHAR_MAX_FEATURES,
        svd_components=int(used_components),
        structure_sha256=structure_digest,
    )
    return work, report


def order_from_label_blind_structure(
    structured: pd.DataFrame,
    *,
    structure_report: DiversityStructureReport,
    seed: int,
    salt: str,
    year_col: str = "source_year",
    year_weighting: str = "sqrt",
) -> tuple[pd.DataFrame, DiversityOrderReport]:
    """Generate a seed-specific, year-controlled order from a label-blind structure.

    Every operation here uses only text-derived cluster information, source year,
    and the declared seed. Labels are not read. Each year gets its own diverse
    cluster order, and the year orders are merged with weighted fair queuing so
    every prefix follows the predeclared year distribution closely.
    """
    required = {
        "text_sha256",
        "example_id",
        year_col,
        "_diversity_cluster",
        "_diversity_distance",
    }
    missing = sorted(required - set(structured.columns))
    if missing:
        raise KeyError(f"Diversity ordering is missing columns: {missing}")

    if year_weighting not in YEAR_WEIGHTING_CHOICES:
        raise ValueError(
            f"Unknown year_weighting={year_weighting!r}; choose from "
            f"{sorted(YEAR_WEIGHTING_CHOICES)}."
        )

    work = structured.copy().reset_index(drop=True)
    work["_year_token"] = work[year_col].astype(str)
    year_counts = work["_year_token"].value_counts().sort_index().to_dict()
    exponent = YEAR_WEIGHTING_CHOICES[year_weighting]
    raw_weights = {
        str(year): float(count) ** float(exponent)
        for year, count in year_counts.items()
    }
    weight_total = sum(raw_weights.values())
    target_year_weights = {
        str(year): float(value / weight_total) for year, value in raw_weights.items()
    }

    year_orders: list[pd.DataFrame] = []
    for year_value, year_group in work.groupby("_year_token", sort=True):
        year_size = max(1, len(year_group))
        head_rows: list[pd.DataFrame] = []
        tail_rows: list[pd.DataFrame] = []

        for cluster_id, group in year_group.groupby("_diversity_cluster", sort=True):
            central = group.sort_values(
                ["_diversity_distance", "text_sha256", "example_id"],
                kind="mergesort",
            ).reset_index(drop=True)

            # The old implementation always selected the single nearest-centroid
            # row as the cluster head. In multi-year periods, those fixed heads
            # could occupy nearly the whole small sample and make all seeds almost
            # identical. Select the representative seed-specifically from a small
            # central candidate set instead. This remains completely label-blind.
            candidate_count = min(
                len(central),
                max(1, min(25, int(math.ceil(math.sqrt(len(central))))))
            )
            rng = np.random.default_rng(
                _stable_int(seed, salt, year_value, cluster_id, "cell") % (2**32 - 1)
            )
            head_position = int(rng.integers(0, candidate_count))
            head = central.iloc[[head_position]].copy()
            cell_share_within_year = float(len(group) / year_size)
            head["_cell_size"] = int(len(group))
            head["_cell_target_share"] = cell_share_within_year
            head["_cell_local_rank"] = 0
            head["_cell_priority"] = -1.0
            head["_tie_key"] = _stable_int(
                seed, salt, year_value, cluster_id, head.iloc[0]["text_sha256"], "head"
            )
            head_rows.append(head)

            remaining = central.drop(index=head_position).reset_index(drop=True)
            if remaining.empty:
                continue
            remaining = remaining.iloc[rng.permutation(len(remaining))].reset_index(drop=True)
            remaining["_cell_size"] = int(len(group))
            remaining["_cell_target_share"] = cell_share_within_year
            remaining["_cell_local_rank"] = np.arange(1, len(remaining) + 1)
            remaining["_cell_priority"] = (
                remaining["_cell_local_rank"].astype(float) - 0.5
            ) / max(cell_share_within_year, 1e-12)
            remaining["_tie_key"] = remaining["text_sha256"].map(
                lambda value: _stable_int(seed, salt, year_value, cluster_id, value)
            )
            tail_rows.append(remaining)

        # Cover clusters first within a year, then allocate additional rows in
        # proportion to cluster prevalence. This prevents a small year from
        # obtaining too many rows merely because it has many year-cluster cells.
        year_head = pd.concat(head_rows, ignore_index=True).sort_values(
            ["_cell_target_share", "_tie_key"],
            ascending=[False, True],
            kind="mergesort",
        )
        if tail_rows:
            year_tail = pd.concat(tail_rows, ignore_index=True).sort_values(
                ["_cell_priority", "_tie_key"], kind="mergesort"
            )
            year_order = pd.concat([year_head, year_tail], ignore_index=True)
        else:
            year_order = year_head.reset_index(drop=True)

        year_order["_year_local_rank"] = np.arange(len(year_order), dtype=np.int64)
        year_weight = target_year_weights[str(year_value)]
        year_order["_year_priority"] = (
            year_order["_year_local_rank"].astype(float) + 0.5
        ) / max(year_weight, 1e-12)
        year_order["_year_tie_key"] = year_order["text_sha256"].map(
            lambda value: _stable_int(seed, salt, year_value, value, "year_merge")
        )
        year_orders.append(year_order)

    ordered = pd.concat(year_orders, ignore_index=True).sort_values(
        ["_year_priority", "_year_tie_key"], kind="mergesort"
    ).reset_index(drop=True)
    ordered["_diversity_rank"] = np.arange(len(ordered), dtype=np.int64)
    report = DiversityOrderReport(
        structure=structure_report,
        label_blind_order_sha256=_hash_lines(ordered["text_sha256"].astype(str)),
        year_weighting=year_weighting,
        target_year_weights=target_year_weights,
        ordering_method=(
            "seed_specific_central_cluster_representatives_then_"
            f"{year_weighting}_year_weighted_fair_merge"
        ),
    )
    return ordered, report


def _largest_remainder_year_counts(
    weights: dict[str, float], total_rows: int
) -> dict[str, int]:
    """Convert year weights to exact integer totals without using labels."""
    if total_rows < 0:
        raise ValueError("total_rows must be non-negative")
    years = sorted(weights)
    raw = {year: float(weights[year]) * total_rows for year in years}
    counts = {year: int(math.floor(raw[year])) for year in years}
    remainder = total_rows - sum(counts.values())
    order = sorted(years, key=lambda y: (-(raw[y] - counts[y]), y))
    for year in order[:remainder]:
        counts[year] += 1
    return counts


def _binary_year_label_quotas(
    *,
    labels: Sequence[str],
    year_weights: dict[str, float],
    n_per_label: int,
    seed: int,
    context: str,
) -> dict[tuple[str, str], int]:
    """Predeclare exact year x class quotas while keeping ranking label-blind.

    The datasets in this paper are binary. Year totals are determined only from
    the requested sample size and source-year weights. Within each year, the two
    classes are split as evenly as integer arithmetic permits. Labels do not
    influence year totals, lexical features, clusters, or row ordering.
    """
    labels = [str(label) for label in labels]
    if len(labels) != 2:
        raise ValueError(
            "Exact year-controlled quota enforcement currently requires two labels; "
            f"received {labels}."
        )
    year_totals = _largest_remainder_year_counts(
        year_weights, total_rows=int(n_per_label) * len(labels)
    )
    quotas: dict[tuple[str, str], int] = {}
    odd_years: list[str] = []
    base_label_total = 0
    for year, total in year_totals.items():
        base = total // 2
        quotas[(year, labels[0])] = base
        quotas[(year, labels[1])] = base
        base_label_total += base
        if total % 2:
            odd_years.append(year)

    extras_for_first = int(n_per_label) - base_label_total
    odd_years = sorted(
        odd_years,
        key=lambda year: _stable_int(seed, context, year, "odd_year_assignment"),
    )
    first_extra_years = set(odd_years[:extras_for_first])
    for year in odd_years:
        label = labels[0] if year in first_extra_years else labels[1]
        quotas[(year, label)] += 1

    for label in labels:
        observed = sum(value for (year, lab), value in quotas.items() if lab == label)
        if observed != int(n_per_label):
            raise RuntimeError(
                f"{context}: internal quota error for {label}: {observed} != {n_per_label}"
            )
    return quotas


def balanced_year_label_prefixes_from_label_blind_order(
    ordered: pd.DataFrame,
    *,
    labels: Sequence[str],
    sizes_per_label: Sequence[int],
    context: str,
    seed: int,
    target_year_weights: dict[str, float],
    year_col: str = "source_year",
) -> dict[int, pd.DataFrame]:
    """Select nested balanced samples with fixed year totals after ranking.

    Labels are used only at this final quota-enforcement step. Each sample has
    exact class totals and year totals derived from the predeclared, label-blind
    year weights. Smaller requested samples are strict subsets of larger ones.
    """
    sizes = sorted({int(value) for value in sizes_per_label})
    maximum = max(sizes)
    quotas_by_size = {
        size: _binary_year_label_quotas(
            labels=labels,
            year_weights=target_year_weights,
            n_per_label=size,
            seed=seed,
            context=f"{context}/n{size}",
        )
        for size in sizes
    }

    # Verify that smaller joint quotas can be represented as nested prefixes of
    # the largest joint quotas. With paper-scale binary samples this should hold;
    # fail loudly rather than silently breaking nesting.
    for smaller, larger in zip(sizes, sizes[1:]):
        for key, small_value in quotas_by_size[smaller].items():
            if small_value > quotas_by_size[larger].get(key, 0):
                raise ValueError(
                    f"{context}: non-nested year-label quota for {key}: "
                    f"n{smaller}={small_value} > n{larger}="
                    f"{quotas_by_size[larger].get(key, 0)}"
                )

    maximum_quotas = quotas_by_size[maximum]
    counts = {key: 0 for key in maximum_quotas}
    accepted_indices: list[int] = []
    available = ordered.groupby(
        [ordered[year_col].astype(str), ordered["label"].astype(str)]
    ).size().to_dict()

    for index, row in ordered.iterrows():
        key = (str(row[year_col]), str(row["label"]))
        if key not in counts or counts[key] >= maximum_quotas[key]:
            continue
        accepted_indices.append(int(index))
        counts[key] += 1
        if all(counts[key] >= maximum_quotas[key] for key in maximum_quotas):
            break

    short = {
        f"{year}|{label}": maximum_quotas[(year, label)] - counts[(year, label)]
        for year, label in maximum_quotas
        if counts[(year, label)] < maximum_quotas[(year, label)]
    }
    if short:
        raise ValueError(
            f"{context}: label-blind order cannot satisfy fixed year x class quotas; "
            f"shortfall={short}; available={available}"
        )

    accepted = ordered.loc[accepted_indices].copy().sort_values(
        "_diversity_rank", kind="mergesort"
    )
    accepted["_accepted_year_label_rank"] = accepted.groupby(
        [accepted[year_col].astype(str), accepted["label"].astype(str)], sort=False
    ).cumcount()

    outputs: dict[int, pd.DataFrame] = {}
    for size in sizes:
        quotas = quotas_by_size[size]
        mask = pd.Series(False, index=accepted.index)
        for (year, label), quota in quotas.items():
            cell_mask = (
                accepted[year_col].astype(str).eq(year)
                & accepted["label"].astype(str).eq(label)
                & accepted["_accepted_year_label_rank"].lt(quota)
            )
            mask = mask | cell_mask
        subset = accepted.loc[mask].copy().sort_values(
            "_diversity_rank", kind="mergesort"
        ).reset_index(drop=True)
        expected = int(size) * len(labels)
        if len(subset) != expected:
            raise RuntimeError(
                f"{context}/n{size}: selected {len(subset)} rows; expected {expected}."
            )
        outputs[size] = subset
    return outputs

def build_label_blind_diversity_order(
    frame: pd.DataFrame,
    *,
    seed: int,
    salt: str,
    target_total: int,
    year_col: str = "source_year",
    year_weighting: str = "sqrt",
) -> tuple[pd.DataFrame, DiversityOrderReport]:
    """Convenience wrapper; preparation code should cache the structure."""
    structured, structure_report = prepare_label_blind_diversity_structure(
        frame, salt=salt, target_total=target_total, year_col=year_col
    )
    return order_from_label_blind_structure(
        structured,
        structure_report=structure_report,
        seed=seed,
        salt=salt,
        year_col=year_col,
        year_weighting=year_weighting,
    )


def balanced_prefixes_from_label_blind_order(
    ordered: pd.DataFrame,
    *,
    labels: Sequence[str],
    sizes_per_label: Sequence[int],
    context: str,
) -> dict[int, pd.DataFrame]:
    """Apply only predeclared class quotas after the text-only order is fixed."""
    sizes = sorted({int(value) for value in sizes_per_label})
    maximum = max(sizes)
    counts = {str(label): 0 for label in labels}
    accepted_indices: list[int] = []

    for index, label_value in ordered["label"].items():
        label = str(label_value)
        if label not in counts or counts[label] >= maximum:
            continue
        accepted_indices.append(int(index))
        counts[label] += 1
        if all(value >= maximum for value in counts.values()):
            break

    short = {label: maximum - counts[label] for label in counts if counts[label] < maximum}
    if short:
        raise ValueError(
            f"{context}: label-blind diversity order cannot satisfy the balanced quota; "
            f"shortfall={short}. Available label counts="
            f"{ordered['label'].value_counts().to_dict()}"
        )

    accepted = ordered.loc[accepted_indices].copy().sort_values(
        "_diversity_rank", kind="mergesort"
    )
    accepted["_accepted_label_rank"] = accepted.groupby("label", sort=False).cumcount()

    outputs: dict[int, pd.DataFrame] = {}
    for size in sizes:
        subset = accepted[accepted["_accepted_label_rank"] < size].copy()
        outputs[size] = subset.sort_values("_diversity_rank", kind="mergesort").reset_index(
            drop=True
        )
    return outputs


def label_blind_prefix(ordered: pd.DataFrame, total_rows: int) -> pd.DataFrame:
    if len(ordered) < total_rows:
        raise ValueError(f"Need {total_rows} rows, but order contains {len(ordered)}.")
    return ordered.iloc[:total_rows].copy().reset_index(drop=True)


def _distribution(series: pd.Series) -> dict[str, float]:
    counts = series.astype(str).value_counts(dropna=False)
    total = float(counts.sum())
    return {} if total == 0 else {str(k): float(v / total) for k, v in counts.items()}


def total_variation_distance(left: dict[str, float], right: dict[str, float]) -> float:
    keys = set(left) | set(right)
    return 0.5 * sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys)


def _simple_vocab(texts: Iterable[str]) -> set[str]:
    vocab: set[str] = set()
    for text in texts:
        vocab.update(token for token in re.findall(r"(?u)\b\w\w+\b", str(text).lower()))
    return vocab


def sampling_audit_row(
    source: pd.DataFrame,
    sample: pd.DataFrame,
    *,
    dataset: str,
    role: str,
    seed: int,
    variant: str,
    n_per_label: int | None,
    order_report: DiversityOrderReport,
    year_col: str = "source_year",
) -> dict[str, Any]:
    source_year = _distribution(source[year_col])
    sample_year = _distribution(sample[year_col])
    target_year = {str(k): float(v) for k, v in order_report.target_year_weights.items()}
    source_labels = _distribution(source["label"])
    sample_labels = _distribution(sample["label"])
    source_clusters = set(source["_diversity_cluster"].astype(int))
    sample_clusters = set(sample["_diversity_cluster"].astype(int))
    source_cells = set(zip(source[year_col].astype(str), source["_diversity_cluster"].astype(int)))
    sample_cells = set(zip(sample[year_col].astype(str), sample["_diversity_cluster"].astype(int)))
    source_vocab = _simple_vocab(source["text"])
    sample_vocab = _simple_vocab(sample["text"])

    return {
        "dataset": dataset,
        "role": role,
        "seed": int(seed),
        "variant": variant,
        "n_per_label": n_per_label,
        "rows": int(len(sample)),
        "source_rows": int(len(source)),
        "label_counts": json.dumps(sample["label"].value_counts().to_dict(), sort_keys=True),
        "year_counts": json.dumps(sample[year_col].astype(str).value_counts().to_dict(), sort_keys=True),
        "year_label_counts": json.dumps(
            {
                f"{year}|{label}": int(count)
                for (year, label), count in sample.groupby(
                    [sample[year_col].astype(str), sample["label"].astype(str)]
                ).size().to_dict().items()
            },
            sort_keys=True,
        ),
        "source_year_counts": json.dumps(source[year_col].astype(str).value_counts().to_dict(), sort_keys=True),
        "year_distribution_tvd": float(total_variation_distance(source_year, sample_year)),
        "target_year_weights": json.dumps(target_year, sort_keys=True),
        "year_target_distribution_tvd": float(
            total_variation_distance(target_year, sample_year)
        ),
        "year_weighting": order_report.year_weighting,
        "label_distribution_tvd": float(total_variation_distance(source_labels, sample_labels)),
        "cluster_coverage": float(len(sample_clusters) / max(1, len(source_clusters))),
        "year_cluster_cell_coverage": float(len(sample_cells) / max(1, len(source_cells))),
        "sample_clusters": int(len(sample_clusters)),
        "source_clusters": int(len(source_clusters)),
        "sample_year_cluster_cells": int(len(sample_cells)),
        "source_year_cluster_cells": int(len(source_cells)),
        "sample_vocab": int(len(sample_vocab)),
        "source_vocab": int(len(source_vocab)),
        "vocab_coverage": float(len(sample_vocab) / max(1, len(source_vocab))),
        "mean_tokens": float(sample["text"].str.split().map(len).mean()),
        "selection_order_label_blind": True,
        "labels_used_only_for_quota": variant == "balanced_quota_after_label_blind_order",
        "structure_sha256": order_report.structure.structure_sha256,
        "label_blind_order_sha256": order_report.label_blind_order_sha256,
        "diversity_method": order_report.method,
        "n_clusters": order_report.n_clusters,
        "n_year_cluster_cells": order_report.n_year_cluster_cells,
    }
