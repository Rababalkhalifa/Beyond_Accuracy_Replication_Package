#!/usr/bin/env python3
"""Prepare and audit immutable temporal splits for the paper rerun.

This is Phase 1 of the rerun pipeline. It creates one permanent split package
covering both datasets and all five seeds. The package is shared by every
frozen and adapted model in later phases.

Scientific design
-----------------
* Five paired seeds: 42, 43, 44, 45, 46.
* A balanced 10,000-row historical pool is sampled independently per seed.
* Exactly 7,000 rows are used for model fitting and 3,000 for internal
  validation/model selection.
* External temporal evaluation rows are never used for fitting or model
  selection.
* Climate uses the paper's n=500-per-label condition.
* Yelp uses n=500 and n=1,000 per label; n=500 is an exact nested subset of
  n=1,000 for each seed and temporal period.
* Exact normalized-text overlap is removed from the historical source before
  the 10,000-row pool is sampled.
* Every saved split is indexed and protected by a SHA-256 checksum.

The script intentionally does not run any model. Expensive runs should start
only after ``--stage audit`` passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sampling_diversity import (
    balanced_year_label_prefixes_from_label_blind_order,
    label_blind_prefix,
    order_from_label_blind_structure,
    prepare_label_blind_diversity_structure,
    sampling_audit_row,
)


DEFAULT_CONFIG = Path("datasets.json")
DEFAULT_SPLIT_ROOT = Path("shared_splits_paper_v6_joint_quota")
DEFAULT_DATASETS = ["climate", "yelp"]
DEFAULT_SEEDS = [42, 43, 44, 45, 46]
HISTORICAL_POOL_TOTAL = 10_000
TRAIN_ROWS = 7_000
VALIDATION_ROWS = 3_000
REQUIRED_ROLES = ["historical_train", "within", "short", "long"]
STANDARD_COLUMNS = [
    "example_id",
    "text",
    "label",
    "text_sha256",
    "source_dataset",
    "source_period",
    "source_year",
    "sample_seed",
]


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    task_type: str
    labels: tuple[str, str]
    text_col: str
    label_col: str
    id_col: str | None
    year_col: str
    allowed_years: Mapping[str, tuple[str, ...]]
    label_map: Mapping[str, str | None]
    paths: Mapping[str, Path]
    evaluation_sizes_per_label: tuple[int, ...]
    periods: Mapping[str, str]


@dataclass(frozen=True)
class PreparedArtifact:
    dataset: str
    seed: int
    role: str
    n_per_label: int | None
    path: Path
    rows: int
    label_counts: Mapping[str, int]
    sha256: str


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["prepare", "audit", "prepare-audit"],
        default="prepare-audit",
        help="Prepare splits, audit an existing package, or do both.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Dataset keys from datasets.json. Default: climate yelp.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Paired split/training seeds.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and rebuild an existing split package.",
    )
    parser.add_argument(
        "--evaluation-overlap-policy",
        choices=["earliest", "keep"],
        default="earliest",
        help=(
            "How to handle exact normalized-text duplicates across evaluation periods. "
            "'earliest' keeps the text in the earliest period (within, then short, then long) "
            "and samples replacements so all paper test sizes remain unchanged. "
            "'keep' preserves the source pools and reports overlap."
        ),
    )
    parser.add_argument(
        "--year-weighting",
        choices=["proportional", "sqrt", "equal"],
        default="sqrt",
        help=(
            "Label-blind year allocation used inside the lexical-diversity order. "
            "'sqrt' is the recommended compromise: it reduces domination by very large "
            "years without forcing equal-year sampling."
        ),
    )
    parser.add_argument(
        "--strict-eval-overlap",
        action="store_true",
        help=(
            "Treat any remaining exact overlap between within/short/long periods as fatal. "
            "With the default earliest policy, remaining overlap should be zero."
        ),
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(value: Any) -> str:
    return " ".join(str(value).strip().split())


def text_sha256(value: Any) -> str:
    return hashlib.sha256(normalize_text(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    temp.replace(path)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def remove_tree(path: Path) -> None:
    if not path.exists():
        return
    for child in sorted(path.rglob("*"), reverse=True):
        if child.is_file() or child.is_symlink():
            child.unlink()
        elif child.is_dir():
            child.rmdir()
    path.rmdir()


def ensure_binary_labels(labels: Sequence[str], dataset: str) -> tuple[str, str]:
    normalized = tuple(str(label).strip().upper() for label in labels)
    if len(normalized) != 2 or len(set(normalized)) != 2:
        raise ValueError(
            f"Dataset {dataset!r} must define exactly two distinct labels; got {labels!r}."
        )
    return normalized[0], normalized[1]


def normalize_label_map(raw: Mapping[Any, Any]) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for source, target in raw.items():
        key = str(source).strip()
        if target is None:
            result[key] = None
            continue
        target_token = str(target).strip().upper()
        result[key] = target_token if target_token else None
    return result


def resolve_config_path(config_path: Path, configured_path: str) -> Path:
    path = Path(configured_path).expanduser()
    if not path.is_absolute():
        path = (config_path.parent / path).resolve()
    return path


def load_specs(config_path: Path, requested: Sequence[str]) -> dict[str, DatasetSpec]:
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration not found: {config_path}. Copy datasets.template.json "
            "to datasets.json and set the source CSV paths."
        )
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    specs: dict[str, DatasetSpec] = {}
    for key in requested:
        if key not in raw:
            raise KeyError(f"Dataset {key!r} is not present in {config_path}.")
        cfg = raw[key]
        labels = ensure_binary_labels(cfg["labels"], key)
        task_type = str(cfg.get("task_type", "binary_text_classification")).strip()
        paths_cfg = cfg.get("paths", {})
        missing = [role for role in REQUIRED_ROLES if role not in paths_cfg]
        if missing:
            raise KeyError(f"Dataset {key!r} is missing paths for: {missing}")
        paths = {
            role: resolve_config_path(config_path, str(paths_cfg[role]))
            for role in REQUIRED_ROLES
        }
        eval_sizes = tuple(
            sorted({int(value) for value in cfg["evaluation_sizes_per_label"]})
        )
        if not eval_sizes or min(eval_sizes) <= 0:
            raise ValueError(f"Invalid evaluation sizes for {key!r}: {eval_sizes}")

        if not cfg.get("year_col"):
            raise KeyError(
                f"Dataset {key!r} must define year_col. The diversity sampler validates "
                "that every source row belongs to the declared temporal period."
            )
        years_cfg = cfg.get("years", {})
        missing_year_roles = [role for role in REQUIRED_ROLES if not years_cfg.get(role)]
        if missing_year_roles:
            raise KeyError(
                f"Dataset {key!r} must define non-empty years for every role; missing "
                f"{missing_year_roles}."
            )

        label_map = normalize_label_map(cfg.get("label_map", {}))
        # Allow already-canonical labels and common textual aliases without
        # forcing duplicate map entries. Numeric encodings are never guessed.
        for label in labels:
            label_map.setdefault(label, label)
        textual_aliases = {
            "NEG": ["NEGATIVE"],
            "POS": ["POSITIVE"],
            "BELIEVER": ["CLIMATE_BELIEVER"],
            "DENIER": ["CLIMATE_DENIER"],
        }
        for canonical, aliases in textual_aliases.items():
            if canonical in labels:
                for alias in aliases:
                    label_map.setdefault(alias, canonical)

        specs[key] = DatasetSpec(
            key=key,
            task_type=task_type,
            labels=labels,
            text_col=str(cfg.get("text_col", "text")),
            label_col=str(cfg.get("label_col", "label")),
            id_col=(str(cfg["id_col"]) if cfg.get("id_col") else None),
            year_col=str(cfg["year_col"]),
            allowed_years={
                role: tuple(str(value) for value in cfg.get("years", {}).get(role, []))
                for role in REQUIRED_ROLES
            },
            label_map=label_map,
            paths=paths,
            evaluation_sizes_per_label=eval_sizes,
            periods={
                role: str(cfg.get("periods", {}).get(role, role))
                for role in REQUIRED_ROLES
            },
        )
    return specs


def map_label(value: Any, spec: DatasetSpec) -> str | None:
    token = str(value).strip()
    candidates = [token, token.upper()]
    # Numeric labels often arrive as 0.0 after CSV parsing.
    try:
        number = float(token)
        if number.is_integer():
            candidates.extend([str(int(number)), f"{int(number)}.0"])
    except (TypeError, ValueError):
        pass

    for candidate in candidates:
        if candidate in spec.label_map:
            return spec.label_map[candidate]
    return None


def normalize_year(value: Any) -> str:
    if pd.isna(value):
        return ""
    token = str(value).strip()
    try:
        number = float(token)
        if number.is_integer():
            return str(int(number))
    except (TypeError, ValueError):
        pass
    return token


def load_source_frame(spec: DatasetSpec, role: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = spec.paths[role]
    if not path.exists():
        raise FileNotFoundError(f"Missing {spec.key}/{role} source file: {path}")

    frame = pd.read_csv(path)
    required = {spec.text_col, spec.label_col}
    required.add(spec.year_col)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"{path} is missing required columns: {missing}")

    original_rows = len(frame)
    raw_label_counts = {
        str(key): int(value)
        for key, value in frame[spec.label_col].astype(str).str.strip().value_counts(dropna=False).head(30).items()
    }
    work = pd.DataFrame()
    work["text"] = frame[spec.text_col].map(normalize_text)
    work["label"] = frame[spec.label_col].map(lambda value: map_label(value, spec))
    work["source_year"] = frame[spec.year_col].map(normalize_year)

    if spec.id_col and spec.id_col in frame.columns:
        raw_ids = frame[spec.id_col].astype(str).str.strip()
    else:
        raw_ids = pd.Series([""] * len(frame), index=frame.index, dtype="object")

    invalid_text = work["text"].eq("")
    unmapped_label = work["label"].isna()
    invalid_year = work["source_year"].eq("")
    if invalid_year.any():
        raise ValueError(
            f"{spec.key}/{role}: {int(invalid_year.sum())} rows have missing or invalid "
            f"year values in column {spec.year_col!r}. Period sampling requires a valid year."
        )
    keep = ~(invalid_text | unmapped_label)
    work = work.loc[keep].copy()
    raw_ids = raw_ids.loc[keep]

    configured_years = set(spec.allowed_years.get(role, ()))
    actual_years = set(work["source_year"].astype(str))
    if configured_years:
        outside = sorted(actual_years - configured_years)
        missing_years = sorted(configured_years - actual_years)
        if outside or missing_years:
            raise ValueError(
                f"{spec.key}/{role}: year-period validation failed. "
                f"configured={sorted(configured_years)}, observed={sorted(actual_years)}, "
                f"outside_period={outside}, configured_years_without_rows={missing_years}."
            )

    work["text_sha256"] = work["text"].map(text_sha256)
    generated_ids = spec.key + ":" + role + ":" + work["text_sha256"].str[:20]
    work["example_id"] = np.where(raw_ids.ne(""), raw_ids, generated_ids)
    work["source_dataset"] = spec.key
    work["source_period"] = spec.periods[role]
    work["sample_seed"] = pd.Series([pd.NA] * len(work), dtype="Int64")

    duplicate_text_rows = int(work.duplicated("text_sha256", keep=False).sum())
    duplicate_id_rows = int(work.duplicated("example_id", keep=False).sum())

    report = {
        "dataset": spec.key,
        "role": role,
        "path": str(path),
        "source_file_sha256": file_sha256(path),
        "original_rows": int(original_rows),
        "usable_rows": int(len(work)),
        "empty_text_rows_removed": int(invalid_text.sum()),
        "unmapped_label_rows_removed": int(unmapped_label.sum()),
        "year_col": spec.year_col,
        "configured_years": list(spec.allowed_years.get(role, ())),
        "observed_year_counts": {
            str(key): int(value)
            for key, value in work["source_year"].astype(str).value_counts().sort_index().items()
        },
        "raw_label_counts": raw_label_counts,
        "configured_label_map": dict(spec.label_map),
        "duplicate_text_rows_detected": duplicate_text_rows,
        "duplicate_id_rows_detected": duplicate_id_rows,
        "label_counts": {
            label: int((work["label"] == label).sum()) for label in spec.labels
        },
    }
    return work.reset_index(drop=True), report


def unique_by_text(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one deterministic row per normalized text hash."""
    return (
        frame.sort_values(["text_sha256", "example_id"], kind="mergesort")
        .drop_duplicates("text_sha256", keep="first")
        .reset_index(drop=True)
    )


def validate_balanced_counts(
    frame: pd.DataFrame,
    labels: Sequence[str],
    expected_total: int,
    context: str,
) -> None:
    if len(frame) != expected_total:
        raise RuntimeError(
            f"{context}: expected {expected_total} rows, found {len(frame)}."
        )
    expected_per_label = expected_total // len(labels)
    counts = frame["label"].value_counts().to_dict()
    for label in labels:
        if int(counts.get(label, 0)) != expected_per_label:
            raise RuntimeError(
                f"{context}: label {label} expected {expected_per_label}, "
                f"found {counts.get(label, 0)}."
            )


def save_split(
    frame: pd.DataFrame,
    path: Path,
    dataset: str,
    seed: int,
    role: str,
    n_per_label: int | None,
    labels: Sequence[str],
) -> PreparedArtifact:
    saved = frame.copy()
    saved["sample_seed"] = int(seed)
    missing = [column for column in STANDARD_COLUMNS if column not in saved.columns]
    if missing:
        raise RuntimeError(f"Cannot save {path}; missing columns: {missing}")
    saved = saved[STANDARD_COLUMNS]
    atomic_write_csv(path, saved)
    digest = file_sha256(path)
    counts = {label: int((saved["label"] == label).sum()) for label in labels}
    return PreparedArtifact(
        dataset=dataset,
        seed=seed,
        role=role,
        n_per_label=n_per_label,
        path=path,
        rows=len(saved),
        label_counts=counts,
        sha256=digest,
    )


def dataset_descriptive_rows(
    frame: pd.DataFrame,
    dataset: str,
    split: str,
    seed: int | None,
    n_per_label: int | None,
    labels: Sequence[str],
) -> list[dict[str, Any]]:
    token_lengths = frame["text"].str.split().map(len)
    rows: list[dict[str, Any]] = [
        {
            "dataset": dataset,
            "seed": seed,
            "split": split,
            "n_per_label": n_per_label,
            "label": "ALL",
            "rows": int(len(frame)),
            "unique_texts": int(frame["text_sha256"].nunique()),
            "year_counts": json.dumps(
                frame["source_year"].astype(str).value_counts().sort_index().to_dict(),
                sort_keys=True,
            ),
            "mean_tokens": float(token_lengths.mean()) if len(frame) else math.nan,
            "median_tokens": float(token_lengths.median()) if len(frame) else math.nan,
        }
    ]
    for label in labels:
        subset = frame[frame["label"] == label]
        lengths = subset["text"].str.split().map(len)
        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "split": split,
                "n_per_label": n_per_label,
                "label": label,
                "rows": int(len(subset)),
                "unique_texts": int(subset["text_sha256"].nunique()),
                "year_counts": json.dumps(
                    subset["source_year"].astype(str).value_counts().sort_index().to_dict(),
                    sort_keys=True,
                ),
                "mean_tokens": float(lengths.mean()) if len(subset) else math.nan,
                "median_tokens": float(lengths.median()) if len(subset) else math.nan,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Preparation
# ---------------------------------------------------------------------------


def prepare_package(
    config_path: Path,
    split_root: Path,
    datasets: Sequence[str],
    seeds: Sequence[int],
    overwrite: bool,
    evaluation_overlap_policy: str,
    year_weighting: str,
) -> None:
    if HISTORICAL_POOL_TOTAL != TRAIN_ROWS + VALIDATION_ROWS:
        raise RuntimeError("Training constants do not sum to the historical pool size.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique.")

    if split_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Split root already exists: {split_root}. Use --stage audit to inspect "
                "it, or --overwrite to rebuild it deliberately."
            )
        remove_tree(split_root)
    split_root.mkdir(parents=True, exist_ok=False)

    specs = load_specs(config_path, datasets)
    artifacts: list[PreparedArtifact] = []
    source_reports: list[dict[str, Any]] = []
    descriptives: list[dict[str, Any]] = []
    leakage_rows: list[dict[str, Any]] = []
    sampling_audit_rows: list[dict[str, Any]] = []
    dataset_manifests: dict[str, Any] = {}

    for dataset_key in datasets:
        spec = specs[dataset_key]
        print(f"\n[PREPARE] Loading {dataset_key} ({spec.task_type})")
        source_frames: dict[str, pd.DataFrame] = {}
        for role in REQUIRED_ROLES:
            frame, report = load_source_frame(spec, role)
            source_frames[role] = frame
            source_reports.append(report)
            descriptives.extend(
                dataset_descriptive_rows(
                    frame,
                    dataset=dataset_key,
                    split=f"source_{role}",
                    seed=None,
                    n_per_label=None,
                    labels=spec.labels,
                )
            )

        # Use unique normalized text for sampling so repeated source rows cannot
        # cross train/validation or be counted twice in an evaluation sample.
        unique_sources = {role: unique_by_text(frame) for role, frame in source_frames.items()}

        # First record the raw source overlap. These rows are evidence about the
        # input data, not necessarily a defect in the final saved split package.
        raw_eval_overlap: dict[str, int] = {}
        for left, right in [("within", "short"), ("within", "long"), ("short", "long")]:
            overlap = set(unique_sources[left]["text_sha256"]) & set(
                unique_sources[right]["text_sha256"]
            )
            raw_eval_overlap[f"{left}__{right}"] = int(len(overlap))
            leakage_rows.append(
                {
                    "dataset": dataset_key,
                    "seed": None,
                    "comparison": f"raw_source_{left}__source_{right}",
                    "overlap_rows": int(len(overlap)),
                    "severity": "resolved"
                    if overlap and evaluation_overlap_policy == "earliest"
                    else ("warning" if overlap else "ok"),
                }
            )

        # Make the evaluation pools mutually disjoint while preserving the paper's
        # chronological interpretation. Exact duplicates are assigned to their
        # earliest observed period: within -> short -> long. Sampling happens only
        # after this step, so replacements are drawn and test sizes remain fixed.
        eval_rows_removed: dict[str, Any] = {}
        if evaluation_overlap_policy == "earliest":
            seen_eval_hashes: set[str] = set()
            for role in ["within", "short", "long"]:
                frame = unique_sources[role]
                duplicate_mask = frame["text_sha256"].isin(seen_eval_hashes)
                removed = frame.loc[duplicate_mask].copy()
                kept = frame.loc[~duplicate_mask].reset_index(drop=True)
                unique_sources[role] = kept
                eval_rows_removed[role] = {
                    "total": int(len(removed)),
                    "by_label": {
                        label: int((removed["label"] == label).sum()) for label in spec.labels
                    },
                    "remaining_by_label": {
                        label: int((kept["label"] == label).sum()) for label in spec.labels
                    },
                }
                seen_eval_hashes.update(kept["text_sha256"].astype(str))
        else:
            eval_rows_removed = {
                role: {
                    "total": 0,
                    "by_label": {label: 0 for label in spec.labels},
                    "remaining_by_label": {
                        label: int((unique_sources[role]["label"] == label).sum())
                        for label in spec.labels
                    },
                }
                for role in ["within", "short", "long"]
            }

        # Verify the effective source pools after applying the overlap policy.
        for left, right in [("within", "short"), ("within", "long"), ("short", "long")]:
            overlap = set(unique_sources[left]["text_sha256"]) & set(
                unique_sources[right]["text_sha256"]
            )
            leakage_rows.append(
                {
                    "dataset": dataset_key,
                    "seed": None,
                    "comparison": f"source_{left}__source_{right}",
                    "overlap_rows": int(len(overlap)),
                    "severity": "warning" if overlap else "ok",
                }
            )

        maximum_eval_size = max(spec.evaluation_sizes_per_label)
        for role in ["within", "short", "long"]:
            for label in spec.labels:
                available = int((unique_sources[role]["label"] == label).sum())
                if available < maximum_eval_size:
                    raise ValueError(
                        f"{dataset_key}: after evaluation overlap policy "
                        f"{evaluation_overlap_policy!r}, period={role}, label={label} has "
                        f"only {available} unique rows; {maximum_eval_size} are required. "
                        f"Removed details={eval_rows_removed[role]}."
                    )

        eval_hashes = set().union(
            *[set(unique_sources[role]["text_sha256"]) for role in ["within", "short", "long"]]
        )
        historical_unique = unique_sources["historical_train"]
        overlap_mask = historical_unique["text_sha256"].isin(eval_hashes)
        historical_clean = historical_unique.loc[~overlap_mask].reset_index(drop=True)
        removed_train_eval_overlap = int(overlap_mask.sum())

        print(f"[DIVERSITY] Fitting fixed text/year structure for {dataset_key} historical pool")
        historical_structure, historical_structure_report = (
            prepare_label_blind_diversity_structure(
                historical_clean,
                salt=f"{dataset_key}:historical_pool",
                target_total=HISTORICAL_POOL_TOTAL,
                minimum_target_total=HISTORICAL_POOL_TOTAL,
            )
        )
        evaluation_structures: dict[str, tuple[pd.DataFrame, Any]] = {}
        maximum_eval_total = max(spec.evaluation_sizes_per_label) * len(spec.labels)
        for role in ["within", "short", "long"]:
            print(f"[DIVERSITY] Fitting fixed text/year structure for {dataset_key} {role}")
            evaluation_structures[role] = prepare_label_blind_diversity_structure(
                unique_sources[role],
                salt=f"{dataset_key}:{role}",
                target_total=maximum_eval_total,
                minimum_target_total=min(spec.evaluation_sizes_per_label) * len(spec.labels),
            )

        dataset_manifest: dict[str, Any] = {
            "task_type": spec.task_type,
            "labels": list(spec.labels),
            "periods": dict(spec.periods),
            "evaluation_sizes_per_label": list(spec.evaluation_sizes_per_label),
            "evaluation_overlap_policy": evaluation_overlap_policy,
            "raw_evaluation_source_overlap": raw_eval_overlap,
            "evaluation_source_rows_removed": eval_rows_removed,
            "historical_unique_rows_before_decontamination": int(len(historical_unique)),
            "historical_rows_removed_for_train_eval_overlap": removed_train_eval_overlap,
            "historical_unique_rows_after_decontamination": int(len(historical_clean)),
            "sampling_design": {
                "primary": (
                    "fixed year-by-class quotas applied after a text-and-year-only "
                    "seed-specific diversity order"
                ),
                "ordering_uses_labels": False,
                "labels_used_only_for_predeclared_year_by_class_quotas": True,
                "year_totals_derived_without_labels": True,
                "within_year_class_quota": "as even as integer arithmetic permits",
                "text_features": "word 1-2 gram TF-IDF + character 3-5 gram TF-IDF",
                "coverage": (
                    "seed-specific central representatives within year-cluster cells, "
                    f"then {year_weighting} weighted fair year merge"
                ),
                "year_weighting": year_weighting,
            },
            "seeds": {},
        }

        for label in spec.labels:
            before = int((historical_unique["label"] == label).sum())
            available = int((historical_clean["label"] == label).sum())
            removed_for_overlap = before - available
            required = HISTORICAL_POOL_TOTAL // 2
            if available < required:
                train_report = next(
                    report for report in source_reports
                    if report["dataset"] == dataset_key and report["role"] == "historical_train"
                )
                raise ValueError(
                    f"{dataset_key}: after decontamination, label {label} has only "
                    f"{available} unique rows; {required} are required. "
                    f"Before decontamination={before}; removed_for_train_eval_overlap="
                    f"{removed_for_overlap}; unmapped_source_rows="
                    f"{train_report['unmapped_label_rows_removed']}; raw_label_counts="
                    f"{train_report['raw_label_counts']}; configured_label_map="
                    f"{train_report['configured_label_map']}. "
                    "Fix the label_map or source paths, then rebuild with --overwrite."
                )

        for seed in seeds:
            seed = int(seed)
            print(f"[PREPARE] {dataset_key} seed={seed}")
            seed_dir = split_root / dataset_key / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=False)

            historical_order, historical_order_report = order_from_label_blind_structure(
                historical_structure,
                structure_report=historical_structure_report,
                seed=seed,
                salt=f"{dataset_key}:historical_pool",
                year_weighting=year_weighting,
            )
            historical_pool = balanced_year_label_prefixes_from_label_blind_order(
                historical_order,
                labels=spec.labels,
                sizes_per_label=[HISTORICAL_POOL_TOTAL // 2],
                context=f"{dataset_key}/seed_{seed}/historical_pool",
                seed=seed,
                target_year_weights=historical_order_report.target_year_weights,
            )[HISTORICAL_POOL_TOTAL // 2]
            natural_historical = label_blind_prefix(
                historical_order, HISTORICAL_POOL_TOTAL
            )
            sampling_audit_rows.append(
                sampling_audit_row(
                    historical_order,
                    historical_pool,
                    dataset=dataset_key,
                    role="historical_pool",
                    seed=seed,
                    variant="balanced_quota_after_label_blind_order",
                    n_per_label=HISTORICAL_POOL_TOTAL // 2,
                    order_report=historical_order_report,
                )
            )
            sampling_audit_rows.append(
                sampling_audit_row(
                    historical_order,
                    natural_historical,
                    dataset=dataset_key,
                    role="historical_pool",
                    seed=seed,
                    variant="fully_label_blind_prefix_audit",
                    n_per_label=None,
                    order_report=historical_order_report,
                )
            )
            historical_strata = (
                historical_pool["source_year"].astype(str)
                + "||"
                + historical_pool["label"].astype(str)
            )
            fit_df, validation_df = train_test_split(
                historical_pool,
                train_size=TRAIN_ROWS,
                test_size=VALIDATION_ROWS,
                random_state=seed,
                shuffle=True,
                stratify=historical_strata,
            )
            fit_df = fit_df.sample(frac=1.0, random_state=seed + 11).reset_index(drop=True)
            validation_df = validation_df.sample(
                frac=1.0, random_state=seed + 17
            ).reset_index(drop=True)
            validate_balanced_counts(
                fit_df, spec.labels, TRAIN_ROWS, f"{dataset_key}/seed_{seed}/train"
            )
            validate_balanced_counts(
                validation_df,
                spec.labels,
                VALIDATION_ROWS,
                f"{dataset_key}/seed_{seed}/validation",
            )

            train_path = seed_dir / "train_7000.csv"
            validation_path = seed_dir / "validation_3000.csv"
            artifacts.append(
                save_split(
                    fit_df,
                    train_path,
                    dataset_key,
                    seed,
                    "train",
                    None,
                    spec.labels,
                )
            )
            artifacts.append(
                save_split(
                    validation_df,
                    validation_path,
                    dataset_key,
                    seed,
                    "validation",
                    None,
                    spec.labels,
                )
            )
            descriptives.extend(
                dataset_descriptive_rows(
                    fit_df,
                    dataset_key,
                    "train",
                    seed,
                    None,
                    spec.labels,
                )
            )
            descriptives.extend(
                dataset_descriptive_rows(
                    validation_df,
                    dataset_key,
                    "validation",
                    seed,
                    None,
                    spec.labels,
                )
            )

            split_hashes = {
                "train": set(fit_df["text_sha256"]),
                "validation": set(validation_df["text_sha256"]),
            }
            leakage_rows.append(
                {
                    "dataset": dataset_key,
                    "seed": seed,
                    "comparison": "train__validation",
                    "overlap_rows": len(split_hashes["train"] & split_hashes["validation"]),
                    "severity": "fatal"
                    if split_hashes["train"] & split_hashes["validation"]
                    else "ok",
                }
            )

            seed_manifest: dict[str, Any] = {
                "train": str(train_path.relative_to(split_root)),
                "validation": str(validation_path.relative_to(split_root)),
                "historical_diversity_order": historical_order_report.as_dict(),
                "evaluation": {},
            }

            seed_eval_samples: dict[str, dict[int, pd.DataFrame]] = {}
            for role in ["within", "short", "long"]:
                maximum_total = maximum_eval_total
                role_structure, role_structure_report = evaluation_structures[role]
                diversity_order, diversity_order_report = order_from_label_blind_structure(
                    role_structure,
                    structure_report=role_structure_report,
                    seed=seed,
                    salt=f"{dataset_key}:{role}",
                    year_weighting=year_weighting,
                )
                nested = balanced_year_label_prefixes_from_label_blind_order(
                    diversity_order,
                    labels=spec.labels,
                    sizes_per_label=spec.evaluation_sizes_per_label,
                    context=f"{dataset_key}/seed_{seed}/{role}",
                    seed=seed,
                    target_year_weights=diversity_order_report.target_year_weights,
                )
                seed_eval_samples[role] = nested
                natural_prefix = label_blind_prefix(diversity_order, maximum_total)
                sampling_audit_rows.append(
                    sampling_audit_row(
                        diversity_order,
                        natural_prefix,
                        dataset=dataset_key,
                        role=role,
                        seed=seed,
                        variant="fully_label_blind_prefix_audit",
                        n_per_label=None,
                        order_report=diversity_order_report,
                    )
                )
                seed_manifest["evaluation"][role] = {
                    "diversity_order": diversity_order_report.as_dict(),
                    "balanced_samples": {},
                }
                for n_per_label, eval_df in nested.items():
                    expected_total = n_per_label * 2
                    validate_balanced_counts(
                        eval_df,
                        spec.labels,
                        expected_total,
                        f"{dataset_key}/seed_{seed}/{role}/n{n_per_label}",
                    )
                    period_tag = spec.periods[role].replace(" ", "_").replace("/", "-")
                    out_path = seed_dir / (
                        f"{role}_{period_tag}_n{n_per_label}_per_label_total{expected_total}.csv"
                    )
                    artifacts.append(
                        save_split(
                            eval_df,
                            out_path,
                            dataset_key,
                            seed,
                            role,
                            n_per_label,
                            spec.labels,
                        )
                    )
                    descriptives.extend(
                        dataset_descriptive_rows(
                            eval_df,
                            dataset_key,
                            role,
                            seed,
                            n_per_label,
                            spec.labels,
                        )
                    )
                    seed_manifest["evaluation"][role]["balanced_samples"][str(n_per_label)] = str(
                        out_path.relative_to(split_root)
                    )
                    sampling_audit_rows.append(
                        sampling_audit_row(
                            diversity_order,
                            eval_df,
                            dataset=dataset_key,
                            role=role,
                            seed=seed,
                            variant="balanced_quota_after_label_blind_order",
                            n_per_label=n_per_label,
                            order_report=diversity_order_report,
                        )
                    )

                    eval_hash_set = set(eval_df["text_sha256"])
                    for train_role, train_hash_set in split_hashes.items():
                        overlap_count = len(eval_hash_set & train_hash_set)
                        leakage_rows.append(
                            {
                                "dataset": dataset_key,
                                "seed": seed,
                                "comparison": f"{train_role}__{role}_n{n_per_label}",
                                "overlap_rows": overlap_count,
                                "severity": "fatal" if overlap_count else "ok",
                            }
                        )

                # Prove nested evaluation sizes for datasets with >1 condition.
                sizes = sorted(nested)
                for smaller, larger in zip(sizes, sizes[1:]):
                    small_hashes = set(nested[smaller]["text_sha256"])
                    large_hashes = set(nested[larger]["text_sha256"])
                    missing_from_large = len(small_hashes - large_hashes)
                    leakage_rows.append(
                        {
                            "dataset": dataset_key,
                            "seed": seed,
                            "comparison": f"nested_{role}_n{smaller}_inside_n{larger}",
                            "overlap_rows": missing_from_large,
                            "severity": "fatal" if missing_from_large else "ok",
                        }
                    )

            # Confirm that the actual diversity-selected evaluation samples are
            # disjoint across temporal periods for every paper-size condition.
            for n_per_label in spec.evaluation_sizes_per_label:
                role_hashes = {
                    role: set(seed_eval_samples[role][int(n_per_label)]["text_sha256"])
                    for role in ["within", "short", "long"]
                }
                for left, right in [("within", "short"), ("within", "long"), ("short", "long")]:
                    overlap_count = len(role_hashes[left] & role_hashes[right])
                    leakage_rows.append(
                        {
                            "dataset": dataset_key,
                            "seed": seed,
                            "comparison": f"{left}_n{n_per_label}__{right}_n{n_per_label}",
                            "overlap_rows": overlap_count,
                            "severity": "fatal" if overlap_count else "ok",
                        }
                    )

            dataset_manifest["seeds"][str(seed)] = seed_manifest
        dataset_manifests[dataset_key] = dataset_manifest

    artifact_rows = [
        {
            "dataset": item.dataset,
            "seed": item.seed,
            "role": item.role,
            "n_per_label": item.n_per_label,
            "path": str(item.path.relative_to(split_root)),
            "rows": item.rows,
            "label_counts_json": json.dumps(item.label_counts, sort_keys=True),
            "sha256": item.sha256,
        }
        for item in artifacts
    ]
    atomic_write_csv(split_root / "split_index.csv", pd.DataFrame(artifact_rows))
    atomic_write_csv(split_root / "dataset_descriptives.csv", pd.DataFrame(descriptives))
    atomic_write_csv(split_root / "leakage_audit.csv", pd.DataFrame(leakage_rows))
    atomic_write_csv(split_root / "sampling_audit.csv", pd.DataFrame(sampling_audit_rows))
    atomic_write_json(split_root / "source_files_manifest.json", source_reports)

    checksum_lines = [
        f"{row['sha256']}  {row['path']}" for row in sorted(artifact_rows, key=lambda r: r["path"])
    ]
    (split_root / "checksums_sha256.txt").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )

    manifest = {
        "schema_version": "4.0",
        "created_at_utc": utc_now(),
        "purpose": "paper-compatible temporal reliability rerun",
        "config_path": str(config_path.resolve()),
        "split_root": str(split_root.resolve()),
        "seeds": [int(seed) for seed in seeds],
        "historical_pool_total": HISTORICAL_POOL_TOTAL,
        "train_rows": TRAIN_ROWS,
        "validation_rows": VALIDATION_ROWS,
        "rules": {
            "external_evaluation_used_for_training": False,
            "external_evaluation_used_for_model_selection": False,
            "same_saved_evaluation_rows_for_all_models": True,
            "diversity_order_uses_text_and_year_only": True,
            "labels_used_only_after_ordering_for_fixed_class_quotas": True,
            "fully_label_blind_prefix_logged_for_bias_audit": True,
            "yelp_n500_nested_in_n1000": True,
            "exact_train_eval_text_overlap_removed_before_sampling": True,
            "evaluation_overlap_policy": evaluation_overlap_policy,
            "evaluation_periods_mutually_disjoint_when_policy_is_earliest": (
                evaluation_overlap_policy == "earliest"
            ),
            "checksums_required_before_model_execution": True,
        },
        "datasets": dataset_manifests,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    atomic_write_json(split_root / "split_manifest.json", manifest)
    print(f"\n[PREPARE] Wrote immutable split package: {split_root.resolve()}")


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


def load_saved_split(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [column for column in STANDARD_COLUMNS if column not in frame.columns]
    if missing:
        raise RuntimeError(f"{path} is missing required columns: {missing}")
    return frame


def audit_package(split_root: Path, strict_eval_overlap: bool) -> None:
    manifest_path = split_root / "split_manifest.json"
    index_path = split_root / "split_index.csv"
    leakage_path = split_root / "leakage_audit.csv"
    sampling_path = split_root / "sampling_audit.csv"
    for path in [manifest_path, index_path, leakage_path, sampling_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required split package file is missing: {path}")

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    index = pd.read_csv(index_path)
    leakage = pd.read_csv(leakage_path)
    sampling = pd.read_csv(sampling_path)

    problems: list[str] = []
    warnings: list[str] = []

    expected_seeds = [int(value) for value in manifest["seeds"]]
    if expected_seeds != DEFAULT_SEEDS:
        warnings.append(
            f"Manifest seeds are {expected_seeds}; paper defaults are {DEFAULT_SEEDS}."
        )

    # File existence, checksum, row counts, balance, and duplicate checks.
    loaded: dict[tuple[str, int, str, int | None], pd.DataFrame] = {}
    for _, row in index.iterrows():
        relative = Path(str(row["path"]))
        path = split_root / relative
        key = (
            str(row["dataset"]),
            int(row["seed"]),
            str(row["role"]),
            None if pd.isna(row["n_per_label"]) else int(row["n_per_label"]),
        )
        if not path.exists():
            problems.append(f"Missing split file: {path}")
            continue
        actual_sha = file_sha256(path)
        if actual_sha != str(row["sha256"]):
            problems.append(f"Checksum mismatch: {relative}")
            continue
        frame = load_saved_split(path)
        loaded[key] = frame
        if len(frame) != int(row["rows"]):
            problems.append(
                f"Row count mismatch for {relative}: index={row['rows']} actual={len(frame)}"
            )
        if frame["text_sha256"].duplicated().any():
            problems.append(f"Duplicate normalized texts inside {relative}")
        if frame["example_id"].duplicated().any():
            warnings.append(f"Duplicate example IDs inside {relative}")
        if set(frame["sample_seed"].astype(int).unique()) != {int(row["seed"])}:
            problems.append(f"Incorrect sample_seed values inside {relative}")

        dataset_cfg = manifest["datasets"][key[0]]
        labels = dataset_cfg["labels"]
        counts = frame["label"].value_counts().to_dict()
        if key[2] == "train":
            expected_total = TRAIN_ROWS
        elif key[2] == "validation":
            expected_total = VALIDATION_ROWS
        else:
            assert key[3] is not None
            expected_total = key[3] * 2
        if len(frame) != expected_total:
            problems.append(
                f"Unexpected rows in {relative}: expected {expected_total}, got {len(frame)}"
            )
        expected_per_label = expected_total // 2
        for label in labels:
            if int(counts.get(label, 0)) != expected_per_label:
                problems.append(
                    f"Class imbalance in {relative}: {label}={counts.get(label, 0)}, "
                    f"expected {expected_per_label}"
                )

    # Independent overlap checks from the saved files.
    for dataset, dataset_cfg in manifest["datasets"].items():
        for seed in expected_seeds:
            train = loaded.get((dataset, seed, "train", None))
            validation = loaded.get((dataset, seed, "validation", None))
            if train is None or validation is None:
                continue
            train_hashes = set(train["text_sha256"])
            validation_hashes = set(validation["text_sha256"])
            if train_hashes & validation_hashes:
                problems.append(f"Train/validation overlap: {dataset} seed={seed}")

            for role in ["within", "short", "long"]:
                size_frames: dict[int, pd.DataFrame] = {}
                for size in dataset_cfg["evaluation_sizes_per_label"]:
                    frame = loaded.get((dataset, seed, role, int(size)))
                    if frame is None:
                        problems.append(
                            f"Missing indexed condition: {dataset} seed={seed} {role} n={size}"
                        )
                        continue
                    size_frames[int(size)] = frame
                    eval_hashes = set(frame["text_sha256"])
                    if eval_hashes & train_hashes:
                        problems.append(
                            f"Train/evaluation overlap: {dataset} seed={seed} {role} n={size}"
                        )
                    if eval_hashes & validation_hashes:
                        problems.append(
                            f"Validation/evaluation overlap: {dataset} seed={seed} {role} n={size}"
                        )

                sizes = sorted(size_frames)
                for smaller, larger in zip(sizes, sizes[1:]):
                    small = set(size_frames[smaller]["text_sha256"])
                    large = set(size_frames[larger]["text_sha256"])
                    if not small.issubset(large):
                        problems.append(
                            f"Nesting failure: {dataset} seed={seed} {role} "
                            f"n={smaller} is not inside n={larger}"
                        )

            # Independently compare saved evaluation periods at each size.
            for size in dataset_cfg["evaluation_sizes_per_label"]:
                period_frames = {
                    role: loaded.get((dataset, seed, role, int(size)))
                    for role in ["within", "short", "long"]
                }
                if any(frame is None for frame in period_frames.values()):
                    continue
                period_hashes = {
                    role: set(frame["text_sha256"])
                    for role, frame in period_frames.items()
                    if frame is not None
                }
                for left, right in [("within", "short"), ("within", "long"), ("short", "long")]:
                    overlap_count = len(period_hashes[left] & period_hashes[right])
                    if overlap_count:
                        message = (
                            f"Saved evaluation overlap: {dataset} seed={seed} "
                            f"n={size} {left}__{right}={overlap_count}"
                        )
                        if strict_eval_overlap or dataset_cfg.get(
                            "evaluation_overlap_policy"
                        ) == "earliest":
                            problems.append(message)
                        else:
                            warnings.append(message)

    # Sampling-design audit: text/year ordering must be label-blind, and every
    # balanced primary sample must declare quota-only label use.
    if sampling.empty:
        problems.append("sampling_audit.csv is empty.")
    else:
        blind_values = sampling["selection_order_label_blind"].astype(str).str.lower()
        if not blind_values.isin(["true", "1"]).all():
            problems.append("One or more sampling orders were not declared label-blind.")
        balanced_rows = sampling[
            sampling["variant"] == "balanced_quota_after_label_blind_order"
        ]
        quota_values = balanced_rows["labels_used_only_for_quota"].astype(str).str.lower()
        if balanced_rows.empty or not quota_values.isin(["true", "1"]).all():
            problems.append(
                "Balanced samples do not consistently record quota-only label use."
            )
        low_cluster_coverage = balanced_rows[
            pd.to_numeric(balanced_rows["cluster_coverage"], errors="coerce") < 0.80
        ]
        if not low_cluster_coverage.empty:
            warnings.append(
                "One or more balanced samples cover less than 80% of text clusters; "
                "inspect sampling_audit.csv."
            )
        high_year_target_shift = balanced_rows[
            pd.to_numeric(
                balanced_rows["year_target_distribution_tvd"], errors="coerce"
            ) > 0.10
        ]
        if not high_year_target_shift.empty:
            warnings.append(
                "One or more balanced samples differ from the declared label-blind "
                "year-weighting target by TVD > 0.10; inspect sampling_audit.csv."
            )

    fatal_recorded = leakage[leakage["severity"] == "fatal"]
    if not fatal_recorded.empty and (fatal_recorded["overlap_rows"] > 0).any():
        problems.append("leakage_audit.csv records one or more fatal overlaps.")

    source_eval_overlap = leakage[
        leakage["comparison"].astype(str).str.startswith("source_")
        & (leakage["overlap_rows"] > 0)
    ]
    if not source_eval_overlap.empty:
        message = (
            "Exact text overlap exists between one or more source evaluation periods. "
            "See leakage_audit.csv."
        )
        if strict_eval_overlap:
            problems.append(message)
        else:
            warnings.append(message)

    audit_report = {
        "audited_at_utc": utc_now(),
        "split_root": str(split_root.resolve()),
        "status": "PASS" if not problems else "FAIL",
        "problems": problems,
        "warnings": warnings,
        "indexed_files": int(len(index)),
        "loaded_files": int(len(loaded)),
    }
    atomic_write_json(split_root / "audit_report.json", audit_report)

    print(f"\n[AUDIT] Indexed files: {len(index)}")
    for warning in warnings:
        print(f"[AUDIT][WARNING] {warning}")
    if problems:
        for problem in problems:
            print(f"[AUDIT][FAIL] {problem}")
        raise RuntimeError(
            f"Split audit failed with {len(problems)} problem(s). "
            f"See {split_root / 'audit_report.json'}"
        )
    print("[AUDIT] PASS: splits are balanced, immutable, aligned, and leakage-free.")
    print(f"[AUDIT] Report: {(split_root / 'audit_report.json').resolve()}")


def main() -> None:
    args = parse_args()
    if args.stage in {"prepare", "prepare-audit"}:
        prepare_package(
            config_path=args.config,
            split_root=args.split_root,
            datasets=args.datasets,
            seeds=args.seeds,
            overwrite=args.overwrite,
            evaluation_overlap_policy=args.evaluation_overlap_policy,
            year_weighting=args.year_weighting,
        )
    if args.stage in {"audit", "prepare-audit"}:
        audit_package(args.split_root, strict_eval_overlap=args.strict_eval_overlap)


if __name__ == "__main__":
    main()
