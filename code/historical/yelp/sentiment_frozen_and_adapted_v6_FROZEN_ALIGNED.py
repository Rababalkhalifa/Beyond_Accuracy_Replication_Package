#!/usr/bin/env python3
"""
Unified temporal sentiment experiment with frozen-aligned adapted baselines: frozen Ollama LLMs + TF-IDF LR + DistilBERT.

Scientific split rules
----------------------
1. A balanced 10,000-row historical sample is split stratifiably into 7,000
   model-fitting rows (70%) and 3,000 internal-validation rows (30%). The internal validation split is used for early
   stopping/model selection only.
2. The paper evaluation periods are never used for training or model selection:
      - within-period evaluation
      - short-horizon temporal evaluation
      - long-horizon temporal evaluation
3. Frozen and adapted approaches use the exact same saved evaluation samples.
4. For each seed, the 500-per-label evaluation sample is nested inside the
   1,000-per-label sample, so sample-size comparisons are paired.
5. The training pool is decontaminated against all evaluation periods by exact
   normalized-text SHA-1 before the 70/30 split. Evaluation rows are not altered.

Expected datasets.json entry
----------------------------
{
  "yelp": {
    "id_col": "id",
    "text_col": "text",
    "label_col": "label",
    "label_map": {"0": "NEG", "1": "NEU", "2": "POS"},  # NEU is excluded
    "date_col": "date",                     # optional
    "paths": {
      "train": "data/yelp_train_2007_2009.csv",
      "gold_dev": "data/yelp_within_2010.csv",
      "gold_test_short": "data/yelp_short_2013.csv",
      "gold_test_long": "data/yelp_long_2018_2019.csv"
    }
  }
}

Accepted path aliases are documented in PATH_ALIASES below.

Examples
--------
Prepare and inspect shared splits only:
    python sentiment_frozen_and_adapted_v6_FROZEN_ALIGNED.py --mode prepare

Run both model families using the newly prepared shared splits:
    python sentiment_frozen_and_adapted_v6_FROZEN_ALIGNED.py --mode both

Reuse an earlier shared split directory:
    python sentiment_frozen_and_adapted_v6_FROZEN_ALIGNED.py --mode both \
        --split-root out/sentiment_joint_YYYYMMDD_HHMMSS/shared_splits

Resume an interrupted combined run and retry only invalid frozen rows:
    python sentiment_frozen_and_adapted_v6_FROZEN_ALIGNED.py --mode both \
        --split-root out/sentiment_joint_YYYYMMDD_HHMMSS/shared_splits \
        --run-dir out/sentiment_joint_YYYYMMDD_HHMMSS --resume --retry-invalid

Run both adapted supervised baselines (TF-IDF LR + DistilBERT):
    pip install transformers accelerate
    python sentiment_frozen_and_adapted_v6_FROZEN_ALIGNED.py --mode adapted

Run only the missing TF-IDF LR baseline and merge it into an existing run:
    python sentiment_frozen_and_adapted_v6_FROZEN_ALIGNED.py --mode adapted \
        --adapted-models tfidf_lr \
        --split-root out/sentiment_joint_YYYYMMDD_HHMMSS/shared_splits \
        --run-dir out/sentiment_joint_YYYYMMDD_HHMMSS --resume

Rerun both adapted baselines on the exact rows used by an earlier frozen run:
    python sentiment_frozen_and_adapted_v6_FROZEN_ALIGNED.py --mode adapted \
        --reference-frozen-run-dir out/sentiment_joint_20260722_103442 \
        --run-dir out/sentiment_adapted_on_frozen_20260722 \
        --adapted-models tfidf_lr distilbert
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# =============================================================================
# USER SETTINGS
# =============================================================================

TASK_NAME = "sentiment"
DEFAULT_DATASET_KEY = "yelp"
DEFAULT_CONFIG_PATH = Path("datasets.json")

LABEL_ORDER = ["NEG", "POS"]
LABEL_SET = set(LABEL_ORDER)
DEFAULT_INVALID_LABEL = "NEG"

# The sentiment experiment is binary. Neutral rows may exist in the source
# dataset/configuration, but they are excluded before splitting and sampling.
BINARY_LABEL_ALIASES = {
    "NEG": "NEG",
    "NEGATIVE": "NEG",
    "POS": "POS",
    "POSITIVE": "POS",
}
EXCLUDED_LABEL_ALIASES = {"NEU", "NEUTRAL"}

# Paper evaluation sizes. These are per label, so n=500 means 1,000 total rows.
EVAL_SIZES_PER_LABEL = [500, 1000]
SAMPLE_SEEDS = [42, 43, 44, 45, 46]

# Conventional supervised split from a fixed, balanced historical sample.
# 10,000 total rows -> exactly 7,000 model-fitting rows and 3,000 internal
# validation rows. The paper's temporal evaluation sets remain separate.
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.30
HISTORICAL_POOL_TOTAL = 10_000

# Frozen LLM settings.
OLLAMA_HOST = "http://localhost:11434"
FROZEN_MODELS = ["qwen2.5:3b-instruct", "phi3.5:latest"]
NUM_WORKERS = min(6, os.cpu_count() or 1)
MAX_TEXT_CHARS = 800
NUM_PREDICT = 80
TIMEOUT_SEC = 600
RETRIES = 2
FROZEN_CHECKPOINT_EVERY = 25
DECODE_SEEDS = [1]
TEMPERATURE = 0.0
TOP_P = 1.0

PROMPT_IDS = ["p1_minimal", "p3_json_strict", "p5_abstain"]
CANONICAL_PROMPT = "p3_json_strict"

PROMPTS = {
    "p1_minimal": """You are a sentiment classifier.
Return ONLY valid JSON with keys: label, confidence, abstain.
Allowed labels: POS, NEG.
confidence MUST be a number between 0 and 1.
Text: {text}
""",
    "p3_json_strict": """Classify the sentiment of TEXT into exactly one label: POS or NEG.
Output MUST be one JSON object on one line. No markdown and no extra keys.
confidence must be a float in [0,1].
Use this exact shape:
{"label":"POS","confidence":0.9,"abstain":false}
TEXT:
{text}
""",
    "p5_abstain": """You are a cautious sentiment classifier.
Return ONLY one JSON object on one line. No markdown and no extra keys.
Allowed labels: POS, NEG.
Use this exact shape:
{"label":"NEG","confidence":0.0,"abstain":true}
If the sentiment is mixed, sarcastic, unclear, or genuinely uncertain:
- set abstain=true
- still provide your best-guess label
- confidence <= 0.5
If confident, set abstain=false and confidence closer to 1.
TEXT:
{text}
""",
}

JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "label": {"type": "string", "enum": LABEL_ORDER},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "abstain": {"type": "boolean"},
    },
    "required": ["label", "confidence", "abstain"],
}

SYSTEM_PROMPT = "Follow the user instructions exactly. Output only what is requested."

# Adapted TF-IDF Logistic Regression settings.
TFIDF_MAX_FEATURES = 50_000
TFIDF_MIN_DF = 2
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_LR_MAX_ITER = 2_000
TFIDF_LR_SOLVER = "liblinear"

# Adapted DistilBERT settings.
ADAPTED_MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 192
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
EARLY_STOPPING_PATIENCE = 1
# Memory-safe physical batches. Gradient accumulation preserves an effective
# training batch size of 8 while reducing peak GPU memory.
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 2
EVAL_ACCUMULATION_STEPS = 16

# datasets.json path aliases. Temporal files themselves remain unchanged.
PATH_ALIASES = {
    "historical_train": ["train", "historical_train", "weak_pool", "gold_train", "train_pool"],
    "within": ["gold_dev", "within", "gold_within", "within_test"],
    "short": ["gold_test_short", "short", "short_test"],
    "long": ["gold_test_long", "long", "long_test"],
}

STANDARD_COLUMNS = ["example_id", "text", "label", "text_sha1"]


# =============================================================================
# GENERIC HELPERS
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["prepare", "frozen", "adapted", "both"],
        default="both",
        help="prepare shared data only, or run one/both model families",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dataset", default=DEFAULT_DATASET_KEY)
    parser.add_argument(
        "--split-root",
        type=Path,
        default=None,
        help="Reuse an existing shared_splits directory. If omitted, create one.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Optional output directory. Default: out/sentiment_joint_<timestamp>",
    )
    parser.add_argument(
        "--eval-sizes",
        type=int,
        nargs="+",
        default=EVAL_SIZES_PER_LABEL,
        help="Balanced evaluation rows per label; paper defaults are 500 and 1000.",
    )
    parser.add_argument(
        "--historical-pool-total",
        type=int,
        default=HISTORICAL_POOL_TOTAL,
        help=(
            "Total balanced historical rows used before the 70/30 split. "
            "Default 10000 gives exactly 7000 training and 3000 validation rows."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=SAMPLE_SEEDS,
        help="Repeated sampling/training seeds.",
    )
    parser.add_argument(
        "--frozen-models",
        nargs="+",
        default=FROZEN_MODELS,
        help="Ollama model tags.",
    )
    parser.add_argument(
        "--adapted-models",
        nargs="+",
        choices=["tfidf_lr", "distilbert"],
        default=["tfidf_lr", "distilbert"],
        help=(
            "Adapted supervised baselines to run. Default runs both TF-IDF LR "
            "and DistilBERT. Use --adapted-models tfidf_lr to add only the "
            "missing classical baseline without retraining DistilBERT."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume an interrupted run in --run-dir. Completed prediction rows and "
            "metric conditions are reused instead of recomputed."
        ),
    )
    parser.add_argument(
        "--retry-invalid",
        action="store_true",
        help=(
            "With --resume, rerun only frozen predictions whose status is not 'ok'. "
            "Valid completed predictions are preserved."
        ),
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=FROZEN_CHECKPOINT_EVERY,
        help="Save frozen prediction progress after this many newly completed requests.",
    )
    parser.add_argument(
        "--reference-frozen-run-dir",
        type=Path,
        default=None,
        help=(
            "Earlier frozen-model run directory whose shared_splits and canonical "
            "prediction files define the evaluation rows for the adapted rerun. "
            "When provided, new evaluation samples are never generated."
        ),
    )
    parser.add_argument(
        "--reference-frozen-metrics",
        type=Path,
        default=None,
        help=(
            "Optional frozen metrics CSV to import into the new combined outputs. "
            "Default: <reference-frozen-run-dir>/metrics_frozen_partial.csv when present."
        ),
    )
    parser.add_argument(
        "--reference-frozen-prompt",
        choices=PROMPT_IDS,
        default=CANONICAL_PROMPT,
        help=(
            "Frozen prompt whose prediction files are used for row-identity alignment. "
            "Default: p3_json_strict."
        ),
    )
    parser.add_argument(
        "--skip-frozen-alignment-check",
        action="store_true",
        help=(
            "Skip prediction-file identity verification. This is not recommended; "
            "the shared frozen split manifest is still required."
        ),
    )
    parser.add_argument(
        "--overwrite-splits",
        action="store_true",
        help="Recreate shared split files if they already exist.",
    )
    return parser.parse_args()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]] | pd.DataFrame) -> None:
    """Write CSV atomically so an interrupted process does not corrupt progress."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def metric_key(row: dict[str, Any] | pd.Series) -> tuple[str, str, int, int, str, str]:
    """Stable identity for one model/seed/size/split/prompt measurement."""
    return (
        str(row["approach"]),
        str(row["model"]),
        int(row["seed"]),
        int(row["eval_n_per_label"]),
        str(row["split"]),
        str(row["prompt"]),
    )


def load_metric_map(path: Path) -> dict[tuple[str, str, int, int, str, str], dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    df = pd.read_csv(path)
    required = {"approach", "model", "seed", "eval_n_per_label", "split", "prompt"}
    if not required.issubset(df.columns):
        print(f"[WARN] Ignoring malformed partial metrics file: {path}")
        return {}
    out: dict[tuple[str, str, int, int, str, str], dict[str, Any]] = {}
    for _, row in df.iterrows():
        rec = row.replace({np.nan: None}).to_dict()
        out[metric_key(rec)] = rec
    return out


def save_metric_map(
    path: Path,
    rows: dict[tuple[str, str, int, int, str, str], dict[str, Any]],
) -> None:
    ordered = sorted(rows.values(), key=metric_key)
    write_csv(path, ordered)


def normalized_text(value: Any) -> str:
    return " ".join(str(value).strip().split())


def sha1_text(value: Any) -> str:
    return hashlib.sha1(normalized_text(value).encode("utf-8")).hexdigest()


def normalize_token(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def stable_offset(name: str) -> int:
    return int(hashlib.sha1(name.encode("utf-8")).hexdigest()[:8], 16) % 100_000


def mean_std(values: Iterable[float]) -> dict[str, Any]:
    vals = [float(v) for v in values if v is not None and not pd.isna(v)]
    if not vals:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals, ddof=0)),
        "n": len(vals),
    }


def resolve_path(paths: dict[str, str], role: str) -> Path:
    for key in PATH_ALIASES[role]:
        if key in paths:
            path = Path(paths[key])
            if not path.exists():
                raise FileNotFoundError(f"Configured {role} path does not exist: {path}")
            return path
    raise KeyError(
        f"Could not resolve {role!r}. Expected one of {PATH_ALIASES[role]} in paths; "
        f"found {sorted(paths)}"
    )


def normalize_label_map(raw_map: dict[Any, Any]) -> dict[Any, str | None]:
    """Normalize configured labels for the binary sentiment task.

    POS/POSITIVE and NEG/NEGATIVE are retained. NEU/NEUTRAL are deliberately
    mapped to ``None`` so those rows are removed before any train/validation or
    evaluation sampling. Other unknown targets still raise an error because
    they are more likely to be configuration mistakes than intended classes.
    """
    if not isinstance(raw_map, dict):
        raise TypeError("label_map must be a dictionary")

    out: dict[Any, str | None] = {}
    for raw_key, raw_value in raw_map.items():
        target_token = normalize_token(raw_value)
        if target_token in BINARY_LABEL_ALIASES:
            target: str | None = BINARY_LABEL_ALIASES[target_token]
        elif target_token in EXCLUDED_LABEL_ALIASES:
            target = None
        else:
            raise ValueError(
                f"Invalid mapped label {raw_value!r}. Use NEG/POS for retained "
                "binary classes, or NEU/NEUTRAL for rows that should be excluded."
            )

        for key in {raw_key, str(raw_key), str(raw_key).strip(), normalize_token(raw_key)}:
            out[key] = target
    return out


def load_source_frame(path: Path, cfg: dict[str, Any], role: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    text_col = cfg["text_col"]
    label_col = cfg["label_col"]
    id_col = cfg.get("id_col")
    date_col = cfg.get("date_col")

    missing = [c for c in [text_col, label_col] if c not in df.columns]
    if missing:
        raise KeyError(f"{path} is missing required columns {missing}; found {list(df.columns)}")

    label_map = normalize_label_map(cfg["label_map"])
    raw_label = df[label_col]
    mapped = raw_label.map(label_map)
    mapped = mapped.fillna(raw_label.astype(str).str.strip().map(label_map))
    mapped = mapped.fillna(raw_label.astype(str).str.strip().str.upper().map(label_map))

    out = pd.DataFrame()
    if id_col and id_col in df.columns:
        out["example_id"] = df[id_col].astype(str)
    else:
        out["example_id"] = [f"{role}_{i}" for i in range(len(df))]

    out["text"] = df[text_col].astype(str)
    out["label"] = mapped
    out["text_sha1"] = out["text"].map(sha1_text)
    out["source_role"] = role
    out["source_path"] = str(path)

    if date_col and date_col in df.columns:
        out["date_value"] = df[date_col].astype(str)
    else:
        detected = next((c for c in ["date", "created_at", "timestamp", "year"] if c in df.columns), None)
        if detected:
            out["date_value"] = df[detected].astype(str)

    before = len(out)
    out = out[out["label"].isin(LABEL_SET)].copy()
    out = out[out["text"].map(lambda x: bool(normalized_text(x)))].copy()
    dropped = before - len(out)
    if dropped:
        print(f"[WARN] {role}: dropped {dropped} rows with empty text or unmapped labels")

    return out.reset_index(drop=True)


def cap_historical_pool(
    df: pd.DataFrame, seed: int, historical_pool_total: int
) -> pd.DataFrame:
    """Create an exactly sized, class-balanced historical pool for one seed."""
    if historical_pool_total <= 0:
        raise ValueError("--historical-pool-total must be positive")
    if historical_pool_total % len(LABEL_ORDER) != 0:
        raise ValueError(
            f"--historical-pool-total must be divisible by {len(LABEL_ORDER)} "
            "to preserve exact class balance"
        )

    n_per_label = historical_pool_total // len(LABEL_ORDER)
    parts = []
    for label in LABEL_ORDER:
        group = df[df["label"] == label]
        if len(group) < n_per_label:
            raise ValueError(
                f"Historical label {label!r} has only {len(group)} rows, but "
                f"{n_per_label} are required for a total pool of {historical_pool_total}."
            )
        parts.append(
            group.sample(
                n=n_per_label,
                replace=False,
                random_state=seed + stable_offset(label),
            )
        )

    sampled = pd.concat(parts, ignore_index=True)
    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def balanced_order(df: pd.DataFrame, seed: int, split_name: str) -> dict[str, pd.DataFrame]:
    ordered: dict[str, pd.DataFrame] = {}
    for label in LABEL_ORDER:
        group = df[df["label"] == label]
        ordered[label] = group.sample(
            frac=1.0,
            random_state=seed + stable_offset(f"{split_name}:{label}"),
        ).reset_index(drop=True)
    return ordered


def nested_balanced_sample(
    ordered: dict[str, pd.DataFrame], n_per_label: int, split_name: str
) -> pd.DataFrame:
    shortages = {label: len(ordered[label]) for label in LABEL_ORDER if len(ordered[label]) < n_per_label}
    if shortages:
        raise ValueError(
            f"{split_name} cannot supply {n_per_label} rows per label. Available: {shortages}"
        )
    sample = pd.concat([ordered[label].iloc[:n_per_label] for label in LABEL_ORDER], ignore_index=True)
    return sample.sample(frac=1.0, random_state=n_per_label + stable_offset(split_name)).reset_index(drop=True)


def standard_saved_columns(df: pd.DataFrame) -> list[str]:
    extra = [c for c in ["date_value", "source_role", "source_path"] if c in df.columns]
    return STANDARD_COLUMNS + extra


def describe_frame(
    df: pd.DataFrame,
    stage: str,
    split: str,
    seed: int | None = None,
    eval_n_per_label: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    chars = df["text"].astype(str).str.len()
    words = df["text"].astype(str).str.split().str.len()

    date_min: str | None = None
    date_max: str | None = None
    year_min: int | None = None
    year_max: int | None = None
    if "date_value" in df.columns:
        parsed = pd.to_datetime(df["date_value"], errors="coerce", utc=True)
        if parsed.notna().any():
            date_min = parsed.min().isoformat()
            date_max = parsed.max().isoformat()
            year_min = int(parsed.dt.year.min())
            year_max = int(parsed.dt.year.max())
        else:
            numeric_year = pd.to_numeric(df["date_value"], errors="coerce")
            if numeric_year.notna().any():
                year_min = int(numeric_year.min())
                year_max = int(numeric_year.max())

    overall = {
        "stage": stage,
        "split": split,
        "seed": seed,
        "eval_n_per_label": eval_n_per_label,
        "n_rows": int(len(df)),
        "n_unique_texts": int(df["text_sha1"].nunique()),
        "duplicate_rows": int(len(df) - df["text_sha1"].nunique()),
        "char_mean": float(chars.mean()) if len(df) else None,
        "char_std": float(chars.std(ddof=0)) if len(df) else None,
        "char_median": float(chars.median()) if len(df) else None,
        "char_min": int(chars.min()) if len(df) else None,
        "char_max": int(chars.max()) if len(df) else None,
        "word_mean": float(words.mean()) if len(df) else None,
        "word_std": float(words.std(ddof=0)) if len(df) else None,
        "word_median": float(words.median()) if len(df) else None,
        "date_min": date_min,
        "date_max": date_max,
        "year_min": year_min,
        "year_max": year_max,
    }

    by_label: list[dict[str, Any]] = []
    for label in LABEL_ORDER:
        group = df[df["label"] == label]
        gchars = group["text"].astype(str).str.len()
        gwords = group["text"].astype(str).str.split().str.len()
        by_label.append(
            {
                "stage": stage,
                "split": split,
                "seed": seed,
                "eval_n_per_label": eval_n_per_label,
                "label": label,
                "n_rows": int(len(group)),
                "proportion": float(len(group) / len(df)) if len(df) else None,
                "char_mean": float(gchars.mean()) if len(group) else None,
                "char_std": float(gchars.std(ddof=0)) if len(group) else None,
                "word_mean": float(gwords.mean()) if len(group) else None,
                "word_std": float(gwords.std(ddof=0)) if len(group) else None,
            }
        )
    return overall, by_label


def print_descriptive_summary(overall_df: pd.DataFrame) -> None:
    cols = [
        "stage",
        "split",
        "seed",
        "eval_n_per_label",
        "n_rows",
        "n_unique_texts",
        "duplicate_rows",
        "char_mean",
        "word_mean",
        "year_min",
        "year_max",
    ]
    visible = overall_df[cols].copy()
    print("\n=== DATASET DESCRIPTIVE STATISTICS ===")
    print(visible.to_string(index=False))


# =============================================================================
# SHARED SPLIT PREPARATION
# =============================================================================


@dataclass(frozen=True)
class PreparedPaths:
    split_root: Path
    manifest_path: Path
    overall_stats_path: Path
    label_stats_path: Path


def prepare_shared_splits(
    config_path: Path,
    dataset_key: str,
    split_root: Path,
    seeds: Sequence[int],
    eval_sizes: Sequence[int],
    overwrite: bool,
    historical_pool_total: int = HISTORICAL_POOL_TOTAL,
) -> PreparedPaths:
    if not math.isclose(TRAIN_FRACTION + VALIDATION_FRACTION, 1.0, abs_tol=1e-12):
        raise ValueError("TRAIN_FRACTION + VALIDATION_FRACTION must equal 1.0")
    if not eval_sizes or min(eval_sizes) <= 0:
        raise ValueError("Evaluation sizes must be positive")

    manifest_path = split_root / "split_manifest.json"
    overall_stats_path = split_root / "dataset_descriptive_overall.csv"
    label_stats_path = split_root / "dataset_descriptive_by_label.csv"

    if manifest_path.exists() and not overwrite:
        existing = load_manifest(manifest_path)
        requested_seeds = {int(x) for x in seeds}
        available_seeds = {int(x) for x in existing.get("seeds", [])}
        requested_sizes = {int(x) for x in eval_sizes}
        available_sizes = {int(x) for x in existing.get("evaluation_sizes_per_label", [])}
        existing_pool_total = int(existing.get("historical_pool_total", -1))
        if existing.get("dataset") != dataset_key:
            raise ValueError(
                f"Existing split manifest is for dataset={existing.get('dataset')!r}, "
                f"not requested dataset={dataset_key!r}. Use another --split-root or --overwrite-splits."
            )
        if not requested_seeds.issubset(available_seeds):
            raise ValueError(
                f"Existing split manifest lacks requested seeds {sorted(requested_seeds - available_seeds)}. "
                "Use --overwrite-splits to rebuild it."
            )
        if not requested_sizes.issubset(available_sizes):
            raise ValueError(
                f"Existing split manifest lacks requested evaluation sizes "
                f"{sorted(requested_sizes - available_sizes)}. Use --overwrite-splits to rebuild it."
            )
        if existing_pool_total != int(historical_pool_total):
            raise ValueError(
                f"Existing split manifest uses historical_pool_total={existing_pool_total}, "
                f"but this run requests {historical_pool_total}. Use a new --split-root or "
                "pass --overwrite-splits to rebuild 7000/3000 training-validation files."
            )
        print(f"[DATA] Reusing existing shared split manifest: {manifest_path}")
        if overall_stats_path.exists():
            print_descriptive_summary(pd.read_csv(overall_stats_path))
        return PreparedPaths(split_root, manifest_path, overall_stats_path, label_stats_path)

    with config_path.open("r", encoding="utf-8") as handle:
        all_cfg = json.load(handle)
    if dataset_key not in all_cfg:
        raise KeyError(f"Dataset {dataset_key!r} is not present in {config_path}")
    cfg = all_cfg[dataset_key]

    paths_cfg = cfg["paths"]
    source_paths = {role: resolve_path(paths_cfg, role) for role in PATH_ALIASES}
    source_frames = {
        role: load_source_frame(path, cfg, role)
        for role, path in source_paths.items()
    }

    overall_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    for role, frame in source_frames.items():
        overall, by_label = describe_frame(frame, stage="source", split=role)
        overall_rows.append(overall)
        label_rows.extend(by_label)

    # Remove exact text overlap from training only; evaluation periods remain unchanged.
    all_eval_hashes = set().union(
        *[set(source_frames[role]["text_sha1"]) for role in ["within", "short", "long"]]
    )
    historical = source_frames["historical_train"]
    overlap_rows = int(historical["text_sha1"].isin(all_eval_hashes).sum())
    historical_clean = historical[~historical["text_sha1"].isin(all_eval_hashes)].copy()
    if overlap_rows:
        print(f"[DATA] Removed {overlap_rows} historical rows overlapping evaluation text")

    split_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "task": TASK_NAME,
        "dataset": dataset_key,
        "created_at": datetime.now().isoformat(),
        "config_path": str(config_path),
        "source_paths": {k: str(v) for k, v in source_paths.items()},
        "labels": LABEL_ORDER,
        "train_fraction": TRAIN_FRACTION,
        "validation_fraction": VALIDATION_FRACTION,
        "historical_pool_total": int(historical_pool_total),
        "expected_train_rows": int(round(historical_pool_total * TRAIN_FRACTION)),
        "expected_validation_rows": int(historical_pool_total - round(historical_pool_total * TRAIN_FRACTION)),
        "evaluation_sizes_per_label": sorted(set(int(x) for x in eval_sizes)),
        "seeds": [int(s) for s in seeds],
        "historical_overlap_rows_removed": overlap_rows,
        "rules": {
            "evaluation_periods_untouched_by_training": True,
            "same_saved_evaluation_rows_for_frozen_and_adapted": True,
            "nested_evaluation_samples": True,
            "within_evaluation_not_used_for_early_stopping": True,
        },
        "seeds_manifest": {},
    }

    max_eval_n = max(eval_sizes)
    for seed in seeds:
        seed = int(seed)
        seed_dir = split_root / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        historical_seed = cap_historical_pool(
            historical_clean, seed, historical_pool_total=historical_pool_total
        )
        fit_df, validation_df = train_test_split(
            historical_seed,
            train_size=TRAIN_FRACTION,
            test_size=VALIDATION_FRACTION,
            random_state=seed,
            shuffle=True,
            stratify=historical_seed["label"],
        )
        fit_df = fit_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        validation_df = validation_df.sample(frac=1.0, random_state=seed + 1).reset_index(drop=True)

        expected_train = int(round(historical_pool_total * TRAIN_FRACTION))
        expected_validation = historical_pool_total - expected_train
        if len(fit_df) != expected_train or len(validation_df) != expected_validation:
            raise RuntimeError(
                f"Unexpected split sizes for seed {seed}: train={len(fit_df)}, "
                f"validation={len(validation_df)}; expected "
                f"{expected_train}/{expected_validation}."
            )
        print(
            f"[DATA] seed={seed}: historical pool={historical_pool_total}, "
            f"train={len(fit_df)}, validation={len(validation_df)}"
        )

        train_path = seed_dir / "train_fit_70.csv"
        validation_path = seed_dir / "train_validation_30.csv"
        fit_df[standard_saved_columns(fit_df)].to_csv(train_path, index=False)
        validation_df[standard_saved_columns(validation_df)].to_csv(validation_path, index=False)

        for split_name, frame in [("train_fit_70", fit_df), ("train_validation_30", validation_df)]:
            overall, by_label = describe_frame(frame, stage="prepared", split=split_name, seed=seed)
            overall_rows.append(overall)
            label_rows.extend(by_label)

        eval_orders = {
            split: balanced_order(source_frames[split], seed, split)
            for split in ["within", "short", "long"]
        }
        seed_manifest: dict[str, Any] = {
            "train_fit": str(train_path),
            "train_validation": str(validation_path),
            "train_fit_rows": int(len(fit_df)),
            "train_validation_rows": int(len(validation_df)),
            "evaluation": {},
        }

        # Validate the largest size first. Smaller sizes are prefixes of the same ordering.
        for split in ["within", "short", "long"]:
            nested_balanced_sample(eval_orders[split], max_eval_n, split)

        for n_per_label in sorted(set(int(x) for x in eval_sizes)):
            n_dir = seed_dir / f"n{n_per_label}_per_label"
            n_dir.mkdir(parents=True, exist_ok=True)
            seed_manifest["evaluation"][str(n_per_label)] = {}

            for split in ["within", "short", "long"]:
                sample = nested_balanced_sample(eval_orders[split], n_per_label, split)
                out_path = n_dir / f"{split}.csv"
                sample[standard_saved_columns(sample)].to_csv(out_path, index=False)
                seed_manifest["evaluation"][str(n_per_label)][split] = str(out_path)

                overall, by_label = describe_frame(
                    sample,
                    stage="prepared",
                    split=split,
                    seed=seed,
                    eval_n_per_label=n_per_label,
                )
                overall_rows.append(overall)
                label_rows.extend(by_label)

        manifest["seeds_manifest"][str(seed)] = seed_manifest

    write_json(manifest_path, manifest)
    write_csv(overall_stats_path, overall_rows)
    write_csv(label_stats_path, label_rows)
    print_descriptive_summary(pd.DataFrame(overall_rows))
    print(f"[DATA] Wrote shared split manifest: {manifest_path}")
    return PreparedPaths(split_root, manifest_path, overall_stats_path, label_stats_path)


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("task") != TASK_NAME:
        raise ValueError(f"Expected task={TASK_NAME!r}, found {manifest.get('task')!r}")
    return manifest


def load_standard_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(STANDARD_COLUMNS) - set(df.columns)
    if missing:
        raise KeyError(f"{path} is missing standard columns: {sorted(missing)}")
    bad = sorted(set(df["label"].astype(str)) - LABEL_SET)
    if bad:
        raise ValueError(f"{path} contains invalid labels: {bad}")
    return df


def frozen_reference_split_root(reference_run_dir: Path) -> Path:
    """Return the only split root allowed for a frozen-aligned adapted rerun."""
    split_root = reference_run_dir / "shared_splits"
    manifest_path = split_root / "split_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            "Frozen-aligned rerun requires the original shared split manifest. "
            f"Expected: {manifest_path}. Do not regenerate the samples; copy the "
            "original shared_splits directory if it was moved."
        )
    return split_root


def _identity_frame_from_eval(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["example_id", "text_sha1", "label"]].copy()
    out["example_id"] = out["example_id"].astype(str)
    out["text_sha1"] = out["text_sha1"].astype(str)
    out["label"] = out["label"].astype(str)
    return out.drop_duplicates().sort_values(
        ["example_id", "text_sha1", "label"]
    ).reset_index(drop=True)


def _identity_frame_from_frozen_predictions(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    required = {"example_id", "text_sha1", "gold_label"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"{path} is missing frozen identity columns: {sorted(missing)}")
    out = df[["example_id", "text_sha1", "gold_label"]].copy()
    out.columns = ["example_id", "text_sha1", "label"]
    out["example_id"] = out["example_id"].astype(str)
    out["text_sha1"] = out["text_sha1"].astype(str)
    out["label"] = out["label"].astype(str)
    return out.drop_duplicates().sort_values(
        ["example_id", "text_sha1", "label"]
    ).reset_index(drop=True)


def audit_frozen_alignment(
    manifest: dict[str, Any],
    reference_run_dir: Path,
    run_dir: Path,
    prompt_id: str = CANONICAL_PROMPT,
) -> Path:
    """Verify exact row identity between frozen predictions and adapted evaluation files."""
    frozen_root = reference_run_dir / "predictions" / "frozen"
    if not frozen_root.exists():
        raise FileNotFoundError(
            f"Frozen prediction directory not found: {frozen_root}. "
            "The alignment audit needs the original row-level prediction CSVs."
        )

    audit_rows: list[dict[str, Any]] = []
    errors: list[str] = []
    checked_files = 0

    for seed_str, seed_plan in manifest["seeds_manifest"].items():
        seed = int(seed_str)
        for n_str, split_map in seed_plan["evaluation"].items():
            n_per_label = int(n_str)
            for split in ["within", "short", "long"]:
                eval_path = Path(split_map[split])
                eval_df = load_standard_csv(eval_path)
                expected = _identity_frame_from_eval(eval_df)
                pattern = f"n{n_per_label}_{split}_{prompt_id}.csv"
                candidates = sorted(frozen_root.glob(f"*/seed{seed}/{pattern}"))
                if not candidates:
                    errors.append(
                        f"No canonical frozen prediction file for seed={seed}, "
                        f"n={n_per_label}, split={split}, prompt={prompt_id}"
                    )
                    audit_rows.append({
                        "seed": seed,
                        "eval_n_per_label": n_per_label,
                        "split": split,
                        "prompt": prompt_id,
                        "frozen_prediction_path": None,
                        "evaluation_path": str(eval_path),
                        "expected_unique_rows": int(len(expected)),
                        "observed_unique_rows": 0,
                        "matched": False,
                        "reason": "missing_frozen_prediction_file",
                    })
                    continue

                for prediction_path in candidates:
                    checked_files += 1
                    observed = _identity_frame_from_frozen_predictions(
                        pd.read_csv(prediction_path), prediction_path
                    )
                    matched = expected.equals(observed)
                    reason = "ok"
                    if not matched:
                        expected_keys = set(map(tuple, expected.to_numpy()))
                        observed_keys = set(map(tuple, observed.to_numpy()))
                        missing_count = len(expected_keys - observed_keys)
                        extra_count = len(observed_keys - expected_keys)
                        reason = f"identity_mismatch:missing={missing_count},extra={extra_count}"
                        errors.append(
                            f"{prediction_path} does not match {eval_path}: "
                            f"missing={missing_count}, extra={extra_count}"
                        )
                    audit_rows.append({
                        "seed": seed,
                        "eval_n_per_label": n_per_label,
                        "split": split,
                        "prompt": prompt_id,
                        "frozen_prediction_path": str(prediction_path),
                        "evaluation_path": str(eval_path),
                        "expected_unique_rows": int(len(expected)),
                        "observed_unique_rows": int(len(observed)),
                        "matched": bool(matched),
                        "reason": reason,
                    })

    report_path = run_dir / "frozen_alignment_audit.csv"
    write_csv(report_path, audit_rows)
    write_json(
        run_dir / "frozen_alignment_audit.json",
        {
            "reference_frozen_run_dir": str(reference_run_dir),
            "prompt": prompt_id,
            "checked_prediction_files": checked_files,
            "conditions": len(audit_rows),
            "all_matched": not errors,
            "errors": errors,
            "csv_report": str(report_path),
        },
    )
    if errors:
        preview = "\n - ".join(errors[:10])
        more = f"\n... and {len(errors) - 10} more" if len(errors) > 10 else ""
        raise RuntimeError(
            "Frozen/adapted row alignment failed. Adapted training was not started.\n - "
            + preview + more + f"\nSee {report_path}"
        )
    print(
        f"[ALIGNMENT] Verified {checked_files} frozen prediction files against "
        f"the shared evaluation rows. Report: {report_path}"
    )
    return report_path


# =============================================================================
# METRICS
# =============================================================================


def ece_from_prediction_confidence(
    y_true: Sequence[str], y_pred: Sequence[str], confidence: Sequence[float], n_bins: int = 10
) -> float:
    correct = np.asarray([a == b for a, b in zip(y_true, y_pred)], dtype=float)
    conf = np.asarray(confidence, dtype=float)
    if len(correct) == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf >= lo) & (conf < hi) if i < n_bins - 1 else (conf >= lo) & (conf <= hi)
        if not mask.any():
            continue
        ece += float(mask.mean() * abs(correct[mask].mean() - conf[mask].mean()))
    return float(ece)


def core_metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=LABEL_ORDER)),
    }


def aggregate_metric_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    group_cols = [
        "approach",
        "model",
        "eval_n_per_label",
        "split",
        "prompt",
    ]
    metric_cols = [
        "accuracy",
        "macro_f1",
        "ece",
        "abstain_rate",
        "coverage",
        "risk_covered",
        "invalid_rate",
        "valid_accuracy",
        "valid_macro_f1",
        "valid_ece",
        "valid_n",
        "invalid_count",
        "json_error_rate",
        "schema_error_rate",
        "request_error_rate",
        "latency_mean_sec",
        "latency_p95_sec",
        "fit_time_sec",
        "n_eval",
    ]
    records: list[dict[str, Any]] = []
    for keys, group in df.groupby(group_cols, dropna=False):
        rec = dict(zip(group_cols, keys))
        rec["n_seeds"] = int(group["seed"].nunique())
        for col in metric_cols:
            if col not in group.columns:
                rec[f"{col}_mean"] = None
                rec[f"{col}_std"] = None
                continue
            vals = pd.to_numeric(group[col], errors="coerce").dropna()
            rec[f"{col}_mean"] = float(vals.mean()) if len(vals) else None
            rec[f"{col}_std"] = float(vals.std(ddof=0)) if len(vals) else None
        records.append(rec)
    return pd.DataFrame(records)


def temporal_delta_rows(metric_rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(metric_rows)
    if df.empty:
        return df
    canonical = df[df["prompt"] == CANONICAL_PROMPT].copy()
    records: list[dict[str, Any]] = []
    keys = ["approach", "model", "seed", "eval_n_per_label"]
    for group_key, group in canonical.groupby(keys):
        by_split = {r["split"]: r for _, r in group.iterrows()}
        if not {"within", "short", "long"}.issubset(by_split):
            continue
        base = dict(zip(keys, group_key))
        for target in ["short", "long"]:
            records.append(
                {
                    **base,
                    "target_split": target,
                    "delta_macro_f1_within_minus_target": float(
                        by_split["within"]["macro_f1"] - by_split[target]["macro_f1"]
                    ),
                    "delta_accuracy_within_minus_target": float(
                        by_split["within"]["accuracy"] - by_split[target]["accuracy"]
                    ),
                }
            )
    return pd.DataFrame(records)


def aggregate_temporal_deltas(delta_df: pd.DataFrame) -> pd.DataFrame:
    if delta_df.empty:
        return delta_df
    group_cols = ["approach", "model", "eval_n_per_label", "target_split"]
    rows: list[dict[str, Any]] = []
    for keys, group in delta_df.groupby(group_cols):
        rec = dict(zip(group_cols, keys))
        rec["n_seeds"] = int(group["seed"].nunique())
        for col in ["delta_macro_f1_within_minus_target", "delta_accuracy_within_minus_target"]:
            rec[f"{col}_mean"] = float(group[col].mean())
            rec[f"{col}_std"] = float(group[col].std(ddof=0))
        rows.append(rec)
    return pd.DataFrame(rows)


def print_measure_summary(aggregate_df: pd.DataFrame) -> None:
    if aggregate_df.empty:
        return
    canonical = aggregate_df[aggregate_df["prompt"] == CANONICAL_PROMPT].copy()
    cols = [
        "approach",
        "model",
        "eval_n_per_label",
        "split",
        "macro_f1_mean",
        "macro_f1_std",
        "accuracy_mean",
        "accuracy_std",
        "ece_mean",
        "latency_mean_sec_mean",
    ]
    print("\n=== CANONICAL PERFORMANCE MEASURES ===")
    print(canonical[cols].sort_values(cols[:4]).to_string(index=False))


# =============================================================================
# FROZEN OLLAMA MODELS
# =============================================================================


_thread_local = threading.local()


def get_http_session() -> requests.Session:
    if not hasattr(_thread_local, "session"):
        _thread_local.session = requests.Session()
    return _thread_local.session


def make_prompt(prompt_id: str, text: str) -> str:
    clean = str(text)
    if len(clean) > MAX_TEXT_CHARS:
        clean = clean[:MAX_TEXT_CHARS] + " ...[TRUNCATED]"
    return PROMPTS[prompt_id].replace("{text}", clean)


def warmup_ollama_model(model: str) -> None:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": 'Output ONLY this JSON: {"label":"POS","confidence":1.0,"abstain":false}',
            },
            {"role": "user", "content": "OK"},
        ],
        "options": {"temperature": 0.0, "top_p": 1.0, "seed": 1, "num_predict": 20},
        "format": JSON_SCHEMA,
        "stream": False,
    }
    try:
        requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=(10, 120)).raise_for_status()
    except Exception as exc:
        print(f"[WARN] Ollama warm-up failed for {model}: {exc}")


def ollama_chat(model: str, prompt: str, decode_seed: int) -> tuple[str, float, str | None]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "options": {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "seed": decode_seed,
            "num_predict": NUM_PREDICT,
        },
        "format": JSON_SCHEMA,
        "stream": False,
    }
    last_error: Exception | None = None
    for attempt in range(RETRIES + 1):
        try:
            t0 = time.perf_counter()
            response = get_http_session().post(
                f"{OLLAMA_HOST}/api/chat",
                json=payload,
                timeout=(10, TIMEOUT_SEC),
            )
            latency = time.perf_counter() - t0
            response.raise_for_status()
            content = response.json().get("message", {}).get("content", "")
            return str(content), float(latency), None
        except Exception as exc:
            last_error = exc
            if attempt < RETRIES:
                time.sleep(2.0 * (attempt + 1))
    return "", 0.0, repr(last_error)


def parse_frozen_json(raw: str) -> tuple[dict[str, Any] | None, str]:
    try:
        obj = json.loads(raw.strip())
    except Exception:
        return None, "json_error"
    if not isinstance(obj, dict) or set(obj) != {"label", "confidence", "abstain"}:
        return None, "schema_error"
    if obj["label"] not in LABEL_SET:
        return None, "schema_error"
    if not isinstance(obj["confidence"], (int, float)) or not 0.0 <= float(obj["confidence"]) <= 1.0:
        return None, "schema_error"
    if not isinstance(obj["abstain"], bool):
        return None, "schema_error"
    return obj, "ok"


def prediction_row_key(row: dict[str, Any] | pd.Series) -> tuple[str, str, int]:
    return (str(row["example_id"]), str(row["text_sha1"]), int(row["decode_seed"]))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def frozen_measures_from_predictions(pred_df: pd.DataFrame, prediction_path: Path) -> dict[str, Any]:
    """
    Report both all-example metrics (invalid outputs use the documented NEG/0.0
    fallback) and valid-only metrics. This prevents invalid generations from
    being hidden inside a single score.
    """
    pred_df = pred_df.copy()
    pred_df["abstain"] = pred_df["abstain"].map(_as_bool)
    pred_df["confidence"] = pd.to_numeric(pred_df["confidence"], errors="coerce").fillna(0.0)
    pred_df["latency_sec"] = pd.to_numeric(pred_df["latency_sec"], errors="coerce").fillna(0.0)
    pred_df["status"] = pred_df["status"].fillna("unknown").astype(str)

    y_true = pred_df["gold_label"].astype(str).tolist()
    y_pred = pred_df["pred_label"].astype(str).tolist()
    all_metrics = core_metrics(y_true, y_pred)

    valid_mask = pred_df["status"].eq("ok")
    valid_df = pred_df.loc[valid_mask]
    if len(valid_df):
        valid_metrics = core_metrics(
            valid_df["gold_label"].astype(str).tolist(),
            valid_df["pred_label"].astype(str).tolist(),
        )
        valid_ece = ece_from_prediction_confidence(
            valid_df["gold_label"].astype(str).tolist(),
            valid_df["pred_label"].astype(str).tolist(),
            valid_df["confidence"].tolist(),
        )
    else:
        valid_metrics = {"accuracy": None, "macro_f1": None}
        valid_ece = None

    correct = pred_df["gold_label"].astype(str) == pred_df["pred_label"].astype(str)
    covered = ~pred_df["abstain"]
    risk_covered = float(1.0 - correct[covered].mean()) if covered.any() else None

    status_counts = pred_df["status"].value_counts().to_dict()
    total = max(len(pred_df), 1)
    return {
        **all_metrics,
        "ece": ece_from_prediction_confidence(y_true, y_pred, pred_df["confidence"].tolist()),
        "valid_accuracy": valid_metrics["accuracy"],
        "valid_macro_f1": valid_metrics["macro_f1"],
        "valid_ece": valid_ece,
        "valid_n": int(valid_mask.sum()),
        "invalid_count": int((~valid_mask).sum()),
        "json_error_rate": float(status_counts.get("json_error", 0) / total),
        "schema_error_rate": float(status_counts.get("schema_error", 0) / total),
        "request_error_rate": float(status_counts.get("request_error", 0) / total),
        "abstain_rate": float(pred_df["abstain"].mean()),
        "coverage": float(covered.mean()),
        "risk_covered": risk_covered,
        "invalid_rate": float((~valid_mask).mean()),
        "latency_mean_sec": float(pred_df["latency_sec"].mean()),
        "latency_p95_sec": float(pred_df["latency_sec"].quantile(0.95)),
        "fit_time_sec": None,
        "n_eval": int(len(pred_df)),
        "invalid_fallback_label": DEFAULT_INVALID_LABEL,
        "prediction_path": str(prediction_path),
    }


def evaluate_frozen_split(
    model: str,
    df: pd.DataFrame,
    prompt_id: str,
    prediction_path: Path,
    *,
    resume: bool = False,
    retry_invalid: bool = False,
    checkpoint_every: int = FROZEN_CHECKPOINT_EVERY,
) -> dict[str, Any]:
    """Evaluate one condition with row-level checkpointing and safe resume."""
    if checkpoint_every < 1:
        raise ValueError("--checkpoint-every must be at least 1")

    df = df.reset_index(drop=True).copy()
    expected: list[tuple[str, str, int]] = []
    source_by_key: dict[tuple[str, str, int], pd.Series] = {}
    for decode_seed in DECODE_SEEDS:
        for _, source in df.iterrows():
            key = (str(source["example_id"]), str(source["text_sha1"]), int(decode_seed))
            if key in source_by_key:
                raise ValueError(
                    "Duplicate frozen prediction identity encountered: "
                    f"example_id={key[0]!r}, text_sha1={key[1]!r}, decode_seed={key[2]}"
                )
            expected.append(key)
            source_by_key[key] = source
    expected_set = set(expected)

    rows_by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    if resume and prediction_path.exists() and prediction_path.stat().st_size:
        try:
            prior = pd.read_csv(prediction_path)
            required = {
                "example_id", "text_sha1", "gold_label", "pred_label", "confidence",
                "abstain", "status", "request_error", "latency_sec", "decode_seed", "raw_output",
            }
            missing = required - set(prior.columns)
            if missing:
                print(f"[WARN] Cannot resume malformed prediction file {prediction_path}; missing {sorted(missing)}")
            else:
                for _, prior_row in prior.iterrows():
                    key = prediction_row_key(prior_row)
                    if key not in expected_set:
                        continue
                    rec = prior_row.replace({np.nan: None}).to_dict()
                    rec["decode_seed"] = int(rec["decode_seed"])
                    rec["abstain"] = _as_bool(rec["abstain"])
                    if retry_invalid and str(rec.get("status")) != "ok":
                        continue
                    rows_by_key[key] = rec
                print(
                    f"    [RESUME] reused {len(rows_by_key)}/{len(expected)} prediction rows"
                    + ("; invalid rows will be retried" if retry_invalid else "")
                )
        except Exception as exc:
            print(f"[WARN] Failed to read resume file {prediction_path}: {exc}; recomputing condition")
            rows_by_key = {}

    pending = [key for key in expected if key not in rows_by_key]

    def checkpoint() -> None:
        ordered_rows = [rows_by_key[key] for key in expected if key in rows_by_key]
        write_csv(prediction_path, ordered_rows)

    if pending:
        jobs: dict[Any, tuple[str, str, int]] = {}
        try:
            with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                for key in pending:
                    source = source_by_key[key]
                    future = executor.submit(
                        ollama_chat,
                        model,
                        make_prompt(prompt_id, source["text"]),
                        key[2],
                    )
                    jobs[future] = key

                newly_completed = 0
                for future in as_completed(jobs):
                    key = jobs[future]
                    source = source_by_key[key]
                    try:
                        raw, latency, request_error = future.result()
                    except Exception as exc:
                        raw, latency, request_error = "", 0.0, repr(exc)
                    parsed, status = (
                        parse_frozen_json(raw) if request_error is None else (None, "request_error")
                    )
                    pred = DEFAULT_INVALID_LABEL
                    confidence = 0.0
                    abstain = False
                    if parsed is not None:
                        pred = str(parsed["label"])
                        confidence = float(parsed["confidence"])
                        abstain = bool(parsed["abstain"])
                    rows_by_key[key] = {
                        "example_id": source["example_id"],
                        "text_sha1": source["text_sha1"],
                        "gold_label": source["label"],
                        "pred_label": pred,
                        "confidence": confidence,
                        "abstain": abstain,
                        "status": status,
                        "request_error": request_error,
                        "latency_sec": latency,
                        "decode_seed": key[2],
                        "raw_output": raw,
                    }
                    newly_completed += 1
                    if newly_completed % checkpoint_every == 0:
                        checkpoint()
                        print(
                            f"    [CHECKPOINT] {len(rows_by_key)}/{len(expected)} rows saved "
                            f"to {prediction_path.name}"
                        )
        except BaseException:
            checkpoint()
            print(f"    [INTERRUPTED] saved {len(rows_by_key)}/{len(expected)} rows")
            raise

    checkpoint()
    missing_keys = [key for key in expected if key not in rows_by_key]
    if missing_keys:
        raise RuntimeError(
            f"Frozen evaluation ended with {len(missing_keys)} missing predictions: {prediction_path}"
        )

    pred_df = pd.DataFrame([rows_by_key[key] for key in expected])
    measures = frozen_measures_from_predictions(pred_df, prediction_path)
    print(
        f"    [DONE] n={measures['n_eval']} invalid={measures['invalid_count']} "
        f"macro_f1={measures['macro_f1']:.4f}"
    )
    return measures


def run_frozen_models(
    manifest: dict[str, Any],
    run_dir: Path,
    models: Sequence[str],
    *,
    resume: bool = False,
    retry_invalid: bool = False,
    checkpoint_every: int = FROZEN_CHECKPOINT_EVERY,
) -> list[dict[str, Any]]:
    partial_path = run_dir / "metrics_frozen_partial.csv"
    metric_map = load_metric_map(partial_path) if resume else {}
    prediction_root = run_dir / "predictions" / "frozen"

    for model in models:
        print(f"\n[FROZEN MODEL] {model}")
        warmup_ollama_model(model)
        safe_model = model.replace(":", "_").replace("/", "_")

        for seed_str, seed_plan in manifest["seeds_manifest"].items():
            seed = int(seed_str)
            for n_str, eval_plan in seed_plan["evaluation"].items():
                n_per_label = int(n_str)
                for split in ["within", "short", "long"]:
                    df = load_standard_csv(eval_plan[split])
                    prompt_list = PROMPT_IDS if split == "within" else [CANONICAL_PROMPT]
                    for prompt_id in prompt_list:
                        key = ("frozen", model, seed, n_per_label, split, prompt_id)
                        print(
                            f"  [RUN] seed={seed} n={n_per_label} split={split} prompt={prompt_id}"
                        )
                        pred_path = (
                            prediction_root
                            / safe_model
                            / f"seed{seed}"
                            / f"n{n_per_label}_{split}_{prompt_id}.csv"
                        )
                        pred_path.parent.mkdir(parents=True, exist_ok=True)
                        measures = evaluate_frozen_split(
                            model,
                            df,
                            prompt_id,
                            pred_path,
                            resume=resume,
                            retry_invalid=retry_invalid,
                            checkpoint_every=checkpoint_every,
                        )
                        metric_map[key] = {
                            "approach": "frozen",
                            "model": model,
                            "seed": seed,
                            "eval_n_per_label": n_per_label,
                            "split": split,
                            "prompt": prompt_id,
                            **measures,
                        }
                        save_metric_map(partial_path, metric_map)
    return sorted(metric_map.values(), key=metric_key)



# =============================================================================
# ADAPTED TF-IDF LOGISTIC REGRESSION
# =============================================================================


def supervised_supervised_prediction_is_complete(path: Path, eval_df: pd.DataFrame) -> bool:
    """Verify that a saved supervised prediction file matches the evaluation rows."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        prior = pd.read_csv(path)
    except Exception:
        return False
    required = {
        "example_id", "text_sha1", "gold_label",
        "pred_label", "confidence", "prob_pos",
    }
    if not required.issubset(prior.columns) or len(prior) != len(eval_df):
        return False
    return (
        prior["text_sha1"].astype(str).tolist()
        == eval_df["text_sha1"].astype(str).tolist()
    )


def run_tfidf_lr_model(
    manifest: dict[str, Any],
    run_dir: Path,
    *,
    resume: bool = False,
) -> list[dict[str, Any]]:
    """Fit TF-IDF + Logistic Regression on train_fit and evaluate all paper splits."""
    partial_path = run_dir / "metrics_tfidf_partial.csv"
    metric_map = load_metric_map(partial_path) if resume else {}
    prediction_root = run_dir / "predictions" / "adapted_tfidf_lr"

    label_to_int = {label: i for i, label in enumerate(LABEL_ORDER)}

    for seed_str, seed_plan in manifest["seeds_manifest"].items():
        seed = int(seed_str)
        print(f"\n[TF-IDF LR] seed={seed}")

        seed_complete = bool(resume)
        if resume:
            for n_str, eval_plan in seed_plan["evaluation"].items():
                n_per_label = int(n_str)
                for split in ["within", "short", "long"]:
                    key = (
                        "adapted_supervised", "tfidf_lr", seed,
                        n_per_label, split, CANONICAL_PROMPT,
                    )
                    pred_path = (
                        prediction_root / f"seed{seed}" / f"n{n_per_label}_{split}.csv"
                    )
                    eval_df = load_standard_csv(eval_plan[split])
                    if (
                        key not in metric_map
                        or not supervised_supervised_prediction_is_complete(pred_path, eval_df)
                    ):
                        seed_complete = False
                        break
                if not seed_complete:
                    break
        if seed_complete:
            print("  [RESUME] all TF-IDF outputs for this seed are complete; skipping fit")
            continue

        fit_df = load_standard_csv(seed_plan["train_fit"])
        validation_df = load_standard_csv(seed_plan["train_validation"])

        pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        ngram_range=TFIDF_NGRAM_RANGE,
                        min_df=TFIDF_MIN_DF,
                        max_features=TFIDF_MAX_FEATURES,
                        dtype=np.float32,
                    ),
                ),
                (
                    "lr",
                    LogisticRegression(
                        max_iter=TFIDF_LR_MAX_ITER,
                        solver=TFIDF_LR_SOLVER,
                        random_state=seed,
                    ),
                ),
            ]
        )

        fit_start = time.perf_counter()
        pipeline.fit(fit_df["text"].astype(str), fit_df["label"].astype(str))
        fit_time = float(time.perf_counter() - fit_start)

        classes = [str(x) for x in pipeline.named_steps["lr"].classes_]
        missing_classes = [label for label in LABEL_ORDER if label not in classes]
        if missing_classes:
            raise RuntimeError(
                f"TF-IDF LR seed {seed} did not learn all labels. Missing: {missing_classes}"
            )

        def predict_probs(texts: Sequence[str]) -> np.ndarray:
            raw = np.asarray(pipeline.predict_proba(list(texts)), dtype=float)
            class_to_col = {label: idx for idx, label in enumerate(classes)}
            return np.column_stack([raw[:, class_to_col[label]] for label in LABEL_ORDER])

        # The internal 30% validation set is diagnostic only; it does not tune on
        # the paper's within/short/long evaluation periods.
        val_probs = predict_probs(validation_df["text"].astype(str).tolist())
        val_pred_int = val_probs.argmax(axis=1)
        val_pred_labels = [LABEL_ORDER[int(i)] for i in val_pred_int]
        val_true = validation_df["label"].astype(str).tolist()
        val_true_int = [label_to_int[x] for x in val_true]
        val_measures = core_metrics(val_true, val_pred_labels)
        val_prob_pos = val_probs[:, label_to_int["POS"]]
        write_json(
            run_dir / f"tfidf_training_diagnostics_seed{seed}.json",
            {
                "seed": seed,
                "fit_rows": int(len(fit_df)),
                "validation_rows": int(len(validation_df)),
                "train_fraction": TRAIN_FRACTION,
                "validation_fraction": VALIDATION_FRACTION,
                "fit_time_this_invocation_sec": fit_time,
                "validation_accuracy": val_measures["accuracy"],
                "validation_macro_f1": val_measures["macro_f1"],
                "validation_ece": ece_binary(val_true_int, val_prob_pos),
                "tfidf_max_features": TFIDF_MAX_FEATURES,
                "tfidf_min_df": TFIDF_MIN_DF,
                "tfidf_ngram_range": list(TFIDF_NGRAM_RANGE),
                "lr_max_iter": TFIDF_LR_MAX_ITER,
                "lr_solver": TFIDF_LR_SOLVER,
            },
        )

        for n_str, eval_plan in seed_plan["evaluation"].items():
            n_per_label = int(n_str)
            for split in ["within", "short", "long"]:
                key = (
                    "adapted_supervised", "tfidf_lr", seed,
                    n_per_label, split, CANONICAL_PROMPT,
                )
                eval_df = load_standard_csv(eval_plan[split])
                pred_path = (
                    prediction_root / f"seed{seed}" / f"n{n_per_label}_{split}.csv"
                )
                if (
                    resume
                    and key in metric_map
                    and supervised_supervised_prediction_is_complete(pred_path, eval_df)
                ):
                    print(f"  [RESUME] n={n_per_label} split={split} already complete")
                    continue

                t0 = time.perf_counter()
                probs = predict_probs(eval_df["text"].astype(str).tolist())
                elapsed = float(time.perf_counter() - t0)

                pred_int = probs.argmax(axis=1)
                pred_labels = [LABEL_ORDER[int(i)] for i in pred_int]
                confidence = probs.max(axis=1)
                prob_pos = probs[:, label_to_int["POS"]]
                y_true = eval_df["label"].astype(str).tolist()
                y_true_int = [label_to_int[x] for x in y_true]
                measures = core_metrics(y_true, pred_labels)
                ece = ece_binary(y_true_int, prob_pos)

                pred_df = pd.DataFrame(
                    {
                        "example_id": eval_df["example_id"],
                        "text_sha1": eval_df["text_sha1"],
                        "gold_label": y_true,
                        "pred_label": pred_labels,
                        "confidence": confidence,
                        "prob_pos": prob_pos,
                    }
                )
                pred_path.parent.mkdir(parents=True, exist_ok=True)
                write_csv(pred_path, pred_df)

                metric_map[key] = {
                    "approach": "adapted_supervised",
                    "model": "tfidf_lr",
                    "seed": seed,
                    "eval_n_per_label": n_per_label,
                    "split": split,
                    "prompt": CANONICAL_PROMPT,
                    **measures,
                    "ece": ece,
                    "valid_accuracy": measures["accuracy"],
                    "valid_macro_f1": measures["macro_f1"],
                    "valid_ece": ece,
                    "valid_n": int(len(eval_df)),
                    "invalid_count": 0,
                    "json_error_rate": 0.0,
                    "schema_error_rate": 0.0,
                    "request_error_rate": 0.0,
                    "abstain_rate": 0.0,
                    "coverage": 1.0,
                    "risk_covered": float(
                        1.0 - np.mean(np.asarray(y_true) == np.asarray(pred_labels))
                    ),
                    "invalid_rate": 0.0,
                    "latency_mean_sec": elapsed / max(len(eval_df), 1),
                    "latency_p95_sec": None,
                    "fit_time_sec": fit_time,
                    "n_eval": int(len(eval_df)),
                    "prediction_path": str(pred_path),
                }
                save_metric_map(partial_path, metric_map)
                print(
                    f"  [DONE] n={n_per_label} split={split} "
                    f"macro_f1={measures['macro_f1']:.4f}"
                )

        del pipeline

    return sorted(metric_map.values(), key=metric_key)


# =============================================================================
# ADAPTED DISTILBERT
# =============================================================================


def require_adapted_dependencies() -> dict[str, Any]:
    try:
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Adapted mode requires transformers and accelerate. Install with: "
            "pip install transformers accelerate"
        ) from exc
    return locals()


def ece_binary(y_true_int: Sequence[int], prob_pos: Sequence[float], n_bins: int = 10) -> float:
    p = np.asarray(prob_pos, dtype=float)
    y = np.asarray(y_true_int, dtype=int)
    pred = (p >= 0.5).astype(int)
    confidence = np.where(pred == 1, p, 1.0 - p)
    labels = [LABEL_ORDER[i] for i in y]
    predictions = [LABEL_ORDER[i] for i in pred]
    return ece_from_prediction_confidence(labels, predictions, confidence, n_bins=n_bins)


def run_distilbert_model(
    manifest: dict[str, Any],
    run_dir: Path,
    *,
    resume: bool = False,
) -> list[dict[str, Any]]:
    deps = require_adapted_dependencies()
    torch = deps["torch"]
    AutoModelForSequenceClassification = deps["AutoModelForSequenceClassification"]
    AutoTokenizer = deps["AutoTokenizer"]
    DataCollatorWithPadding = deps["DataCollatorWithPadding"]
    EarlyStoppingCallback = deps["EarlyStoppingCallback"]
    Trainer = deps["Trainer"]
    TrainingArguments = deps["TrainingArguments"]
    set_seed = deps["set_seed"]

    label_to_int = {label: i for i, label in enumerate(LABEL_ORDER)}
    int_to_label = {i: label for label, i in label_to_int.items()}

    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, texts: Sequence[str], labels: Sequence[str], tokenizer: Any):
            self.encodings = tokenizer(
                list(texts),
                truncation=True,
                max_length=MAX_LENGTH,
                padding=False,
            )
            self.labels = [label_to_int[x] for x in labels]

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, idx: int) -> dict[str, Any]:
            item = {k: self.encodings[k][idx] for k in self.encodings}
            item["labels"] = self.labels[idx]
            return item

    def compute_metrics_eval(eval_pred: Any) -> dict[str, float]:
        logits, labels = eval_pred
        pred = np.asarray(logits).argmax(axis=1)
        y_true = [int_to_label[int(x)] for x in labels]
        y_pred = [int_to_label[int(x)] for x in pred]
        return core_metrics(y_true, y_pred)

    partial_path = run_dir / "metrics_adapted_partial.csv"
    metric_map = load_metric_map(partial_path) if resume else {}
    prediction_root = run_dir / "predictions" / "adapted_distilbert"
    checkpoint_root = run_dir / "checkpoints" / "adapted_distilbert"

    for seed_str, seed_plan in manifest["seeds_manifest"].items():
        seed = int(seed_str)
        print(f"\n[ADAPTED MODEL] seed={seed}")

        # Skip training entirely when all six paper evaluation conditions for
        # this seed already have both metrics and complete prediction files.
        seed_complete = bool(resume)
        if resume:
            for n_str, eval_plan in seed_plan["evaluation"].items():
                n_per_label = int(n_str)
                for split in ["within", "short", "long"]:
                    key = (
                        "adapted_supervised", "distilbert", seed,
                        n_per_label, split, CANONICAL_PROMPT,
                    )
                    pred_path = prediction_root / f"seed{seed}" / f"n{n_per_label}_{split}.csv"
                    eval_df = load_standard_csv(eval_plan[split])
                    if key not in metric_map or not supervised_prediction_is_complete(pred_path, eval_df):
                        seed_complete = False
                        break
                if not seed_complete:
                    break
        if seed_complete:
            print("  [RESUME] all adapted outputs for this seed are complete; skipping training")
            continue

        set_seed(seed)
        fit_df = load_standard_csv(seed_plan["train_fit"])
        validation_df = load_standard_csv(seed_plan["train_validation"])

        tokenizer = AutoTokenizer.from_pretrained(ADAPTED_MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(
            ADAPTED_MODEL_NAME,
            num_labels=len(LABEL_ORDER),
            id2label=int_to_label,
            label2id=label_to_int,
        )
        train_ds = TextDataset(fit_df["text"], fit_df["label"], tokenizer)
        validation_ds = TextDataset(validation_df["text"], validation_df["label"], tokenizer)
        collator = DataCollatorWithPadding(tokenizer=tokenizer)

        output_dir = checkpoint_root / f"seed{seed}"
        last_checkpoint: Path | None = None
        if resume and output_dir.exists():
            candidates = []
            for candidate in output_dir.glob("checkpoint-*"):
                try:
                    step = int(candidate.name.rsplit("-", 1)[1])
                    candidates.append((step, candidate))
                except (IndexError, ValueError):
                    continue
            if candidates:
                last_checkpoint = max(candidates, key=lambda item: item[0])[1]
                print(f"  [RESUME] training from {last_checkpoint}")

        kwargs: dict[str, Any] = {
            "output_dir": str(output_dir),
            "overwrite_output_dir": not resume,
            "num_train_epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "warmup_ratio": WARMUP_RATIO,
            "per_device_train_batch_size": TRAIN_BATCH_SIZE,
            "per_device_eval_batch_size": EVAL_BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "eval_accumulation_steps": EVAL_ACCUMULATION_STEPS,
            "fp16": bool(torch.cuda.is_available()),
            "save_strategy": "epoch",
            "load_best_model_at_end": True,
            "metric_for_best_model": "macro_f1",
            "greater_is_better": True,
            "logging_strategy": "epoch",
            "save_total_limit": 1,
            "report_to": "none",
            "seed": seed,
            "data_seed": seed,
            "fp16": bool(torch.cuda.is_available()),
        }
        parameters = inspect.signature(TrainingArguments.__init__).parameters
        if "eval_strategy" in parameters:
            kwargs["eval_strategy"] = "epoch"
        else:
            kwargs["evaluation_strategy"] = "epoch"
        training_args = TrainingArguments(**kwargs)

        trainer_kwargs: dict[str, Any] = {
            "model": model,
            "args": training_args,
            "train_dataset": train_ds,
            "eval_dataset": validation_ds,
            "data_collator": collator,
            "compute_metrics": compute_metrics_eval,
            "callbacks": [EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
        }
        trainer_params = inspect.signature(Trainer.__init__).parameters
        if "processing_class" in trainer_params:
            trainer_kwargs["processing_class"] = tokenizer
        elif "tokenizer" in trainer_params:
            trainer_kwargs["tokenizer"] = tokenizer
        trainer = Trainer(**trainer_kwargs)

        fit_start = time.perf_counter()
        trainer.train(resume_from_checkpoint=str(last_checkpoint) if last_checkpoint else None)
        fit_time = float(time.perf_counter() - fit_start)

        # Internal validation is reported as training diagnostics only.
        val_output = trainer.predict(validation_ds)
        val_logits = np.asarray(val_output.predictions)
        val_probs = torch.softmax(torch.tensor(val_logits), dim=-1).cpu().numpy()
        val_pred = val_probs.argmax(axis=1)
        val_true = [label_to_int[x] for x in validation_df["label"]]
        val_metrics = core_metrics(
            validation_df["label"].tolist(),
            [int_to_label[int(x)] for x in val_pred],
        )
        write_json(
            run_dir / f"adapted_training_diagnostics_seed{seed}.json",
            {
                "seed": seed,
                "fit_rows": int(len(fit_df)),
                "validation_rows": int(len(validation_df)),
                "train_fraction": TRAIN_FRACTION,
                "validation_fraction": VALIDATION_FRACTION,
                "fit_time_this_invocation_sec": fit_time,
                "resumed_from_checkpoint": str(last_checkpoint) if last_checkpoint else None,
                "validation_accuracy": val_metrics["accuracy"],
                "validation_macro_f1": val_metrics["macro_f1"],
                "validation_ece": ece_binary(val_true, val_probs[:, label_to_int["POS"]]),
            },
        )

        for n_str, eval_plan in seed_plan["evaluation"].items():
            n_per_label = int(n_str)
            for split in ["within", "short", "long"]:
                key = (
                    "adapted_supervised", "distilbert", seed,
                    n_per_label, split, CANONICAL_PROMPT,
                )
                eval_df = load_standard_csv(eval_plan[split])
                pred_path = prediction_root / f"seed{seed}" / f"n{n_per_label}_{split}.csv"
                if resume and key in metric_map and supervised_prediction_is_complete(pred_path, eval_df):
                    print(f"  [RESUME] n={n_per_label} split={split} already complete")
                    continue

                eval_ds = TextDataset(eval_df["text"], eval_df["label"], tokenizer)
                t0 = time.perf_counter()
                output = trainer.predict(eval_ds)
                elapsed = float(time.perf_counter() - t0)
                logits = np.asarray(output.predictions)
                probs = torch.softmax(torch.tensor(logits), dim=-1).cpu().numpy()
                pred_int = probs.argmax(axis=1)
                pred_labels = [int_to_label[int(x)] for x in pred_int]
                confidence = probs.max(axis=1)
                prob_pos = probs[:, label_to_int["POS"]]
                y_true = eval_df["label"].tolist()
                y_true_int = [label_to_int[x] for x in y_true]
                measures = core_metrics(y_true, pred_labels)

                pred_df = pd.DataFrame(
                    {
                        "example_id": eval_df["example_id"],
                        "text_sha1": eval_df["text_sha1"],
                        "gold_label": y_true,
                        "pred_label": pred_labels,
                        "confidence": confidence,
                        "prob_pos": prob_pos,
                    }
                )
                pred_path.parent.mkdir(parents=True, exist_ok=True)
                write_csv(pred_path, pred_df)

                metric_map[key] = {
                    "approach": "adapted_supervised",
                    "model": "distilbert",
                    "seed": seed,
                    "eval_n_per_label": n_per_label,
                    "split": split,
                    "prompt": CANONICAL_PROMPT,
                    **measures,
                    "ece": ece_binary(y_true_int, prob_pos),
                    "valid_accuracy": measures["accuracy"],
                    "valid_macro_f1": measures["macro_f1"],
                    "valid_ece": ece_binary(y_true_int, prob_pos),
                    "valid_n": int(len(eval_df)),
                    "invalid_count": 0,
                    "json_error_rate": 0.0,
                    "schema_error_rate": 0.0,
                    "request_error_rate": 0.0,
                    "abstain_rate": 0.0,
                    "coverage": 1.0,
                    "risk_covered": float(1.0 - np.mean(np.asarray(y_true) == np.asarray(pred_labels))),
                    "invalid_rate": 0.0,
                    "latency_mean_sec": elapsed / max(len(eval_df), 1),
                    "latency_p95_sec": None,
                    "fit_time_sec": fit_time,
                    "n_eval": int(len(eval_df)),
                    "prediction_path": str(pred_path),
                }
                save_metric_map(partial_path, metric_map)

        del trainer, model, tokenizer, train_ds, validation_ds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return sorted(metric_map.values(), key=metric_key)


# =============================================================================
# RESULT ASSEMBLY
# =============================================================================


def prompt_sensitivity_rows(metric_rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(metric_rows)
    if df.empty:
        return df
    frozen_within = df[(df["approach"] == "frozen") & (df["split"] == "within")]
    rows: list[dict[str, Any]] = []
    for keys, group in frozen_within.groupby(["model", "seed", "eval_n_per_label"]):
        f1s = group.set_index("prompt")["macro_f1"].to_dict()
        if not set(PROMPT_IDS).issubset(f1s):
            continue
        values = [float(f1s[p]) for p in PROMPT_IDS]
        rows.append(
            {
                "model": keys[0],
                "seed": keys[1],
                "eval_n_per_label": keys[2],
                "prompt_f1_mean": float(statistics.mean(values)),
                "prompt_f1_variance": float(np.var(values)),
                "prompt_f1_range": float(max(values) - min(values)),
            }
        )
    return pd.DataFrame(rows)


def save_results(run_dir: Path, metric_rows: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    metrics_df = pd.DataFrame(metric_rows)
    metrics_path = run_dir / "metrics_per_seed.csv"
    metrics_df.to_csv(metrics_path, index=False)

    aggregate_df = aggregate_metric_rows(metric_rows)
    aggregate_df.to_csv(run_dir / "metrics_aggregated.csv", index=False)

    deltas = temporal_delta_rows(metric_rows)
    deltas.to_csv(run_dir / "temporal_deltas_per_seed.csv", index=False)
    aggregate_temporal_deltas(deltas).to_csv(run_dir / "temporal_deltas_aggregated.csv", index=False)

    sensitivity = prompt_sensitivity_rows(metric_rows)
    sensitivity.to_csv(run_dir / "prompt_sensitivity_per_seed.csv", index=False)

    result_json = {
        "task": TASK_NAME,
        "dataset": manifest["dataset"],
        "created_at": datetime.now().isoformat(),
        "split_manifest": manifest,
        "metrics_per_seed": metric_rows,
        "metrics_aggregated": aggregate_df.replace({np.nan: None}).to_dict(orient="records"),
        "temporal_deltas_per_seed": deltas.replace({np.nan: None}).to_dict(orient="records"),
        "prompt_sensitivity_per_seed": sensitivity.replace({np.nan: None}).to_dict(orient="records"),
    }
    write_json(run_dir / "results_sentiment_combined.json", result_json)
    print_measure_summary(aggregate_df)
    print(f"\n[RESULTS] Wrote {metrics_path}")
    print(f"[RESULTS] Wrote {run_dir / 'metrics_aggregated.csv'}")
    print(f"[RESULTS] Wrote {run_dir / 'temporal_deltas_aggregated.csv'}")
    print(f"[RESULTS] Wrote {run_dir / 'results_sentiment_combined.json'}")


def main() -> None:
    args = parse_args()
    if args.resume and args.run_dir is None:
        raise ValueError("--resume requires --run-dir pointing to the interrupted run directory")
    if args.retry_invalid and not args.resume:
        raise ValueError("--retry-invalid is only meaningful together with --resume")
    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every must be at least 1")
    if args.historical_pool_total <= 0:
        raise ValueError("--historical-pool-total must be positive")
    if args.historical_pool_total % len(LABEL_ORDER) != 0:
        raise ValueError(
            f"--historical-pool-total must be divisible by {len(LABEL_ORDER)} "
            "for an exactly balanced binary pool"
        )
    if args.reference_frozen_run_dir is not None and args.overwrite_splits:
        raise ValueError(
            "--overwrite-splits is forbidden with --reference-frozen-run-dir. "
            "Regenerating samples would invalidate the frozen/adapted comparison."
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.run_dir or Path("out") / f"sentiment_joint_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.reference_frozen_run_dir is not None:
        reference_run_dir = args.reference_frozen_run_dir
        reference_split_root = frozen_reference_split_root(reference_run_dir)
        if args.split_root is not None:
            requested = args.split_root.resolve()
            required = reference_split_root.resolve()
            if requested != required:
                raise ValueError(
                    "--split-root must point to the frozen run's original shared_splits "
                    f"directory: {reference_split_root}"
                )
        split_root = reference_split_root
        prepared = PreparedPaths(
            split_root=split_root,
            manifest_path=split_root / "split_manifest.json",
            overall_stats_path=split_root / "dataset_descriptive_overall.csv",
            label_stats_path=split_root / "dataset_descriptive_by_label.csv",
        )
        print(f"[DATA] Using frozen reference split manifest: {prepared.manifest_path}")
    else:
        reference_run_dir = None
        split_root = args.split_root or run_dir / "shared_splits"
        prepared = prepare_shared_splits(
            config_path=args.config,
            dataset_key=args.dataset,
            split_root=split_root,
            seeds=args.seeds,
            eval_sizes=args.eval_sizes,
            overwrite=args.overwrite_splits,
            historical_pool_total=args.historical_pool_total,
        )
    manifest = load_manifest(prepared.manifest_path)

    if reference_run_dir is not None and not args.skip_frozen_alignment_check:
        alignment_report_path = audit_frozen_alignment(
            manifest,
            reference_run_dir=reference_run_dir,
            run_dir=run_dir,
            prompt_id=args.reference_frozen_prompt,
        )
    else:
        alignment_report_path = None

    write_json(
        run_dir / "run_settings.json",
        {
            "task": TASK_NAME,
            "mode": args.mode,
            "dataset": args.dataset,
            "config": str(args.config),
            "split_root": str(split_root),
            "reference_frozen_run_dir": (
                str(reference_run_dir) if reference_run_dir is not None else None
            ),
            "reference_frozen_prompt": (
                args.reference_frozen_prompt if reference_run_dir is not None else None
            ),
            "frozen_alignment_checked": bool(
                reference_run_dir is not None and not args.skip_frozen_alignment_check
            ),
            "frozen_alignment_report": (
                str(alignment_report_path) if alignment_report_path is not None else None
            ),
            "eval_sizes_per_label": args.eval_sizes,
            "seeds": args.seeds,
            "train_fraction": TRAIN_FRACTION,
            "validation_fraction": VALIDATION_FRACTION,
            "historical_pool_total": int(args.historical_pool_total),
            "expected_train_rows": int(round(args.historical_pool_total * TRAIN_FRACTION)),
            "expected_validation_rows": int(args.historical_pool_total - round(args.historical_pool_total * TRAIN_FRACTION)),
            "train_batch_size": TRAIN_BATCH_SIZE,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "frozen_models": args.frozen_models,
            "adapted_models": args.adapted_models,
            "adapted_model": ADAPTED_MODEL_NAME,
            "tfidf_model": {
                "model_key": "tfidf_lr",
                "max_features": TFIDF_MAX_FEATURES,
                "min_df": TFIDF_MIN_DF,
                "ngram_range": list(TFIDF_NGRAM_RANGE),
                "lr_max_iter": TFIDF_LR_MAX_ITER,
                "lr_solver": TFIDF_LR_SOLVER,
            },
            "canonical_prompt": CANONICAL_PROMPT,
            "resume": bool(args.resume),
            "retry_invalid": bool(args.retry_invalid),
            "checkpoint_every": int(args.checkpoint_every),
        },
    )

    if args.mode == "prepare":
        print("\nDone. Shared sentiment splits and descriptive statistics are ready.")
        return

    # In resume mode, retain already completed model rows so running only the
    # missing TF-IDF baseline does not erase completed DistilBERT/frozen results.
    combined_metric_map: dict[
        tuple[str, str, int, int, str, str], dict[str, Any]
    ] = {}
    reference_metrics_path: Path | None = args.reference_frozen_metrics
    if reference_metrics_path is None and reference_run_dir is not None:
        candidate = reference_run_dir / "metrics_frozen_partial.csv"
        if candidate.exists():
            reference_metrics_path = candidate
    if reference_metrics_path is not None:
        if not reference_metrics_path.exists():
            raise FileNotFoundError(
                f"Reference frozen metrics file not found: {reference_metrics_path}"
            )
        frozen_metric_map = load_metric_map(reference_metrics_path)
        combined_metric_map.update(frozen_metric_map)
        print(
            f"[RESULTS] Imported {len(frozen_metric_map)} frozen metric rows from "
            f"{reference_metrics_path}"
        )

    if args.resume:
        for prior_path in [
            run_dir / "metrics_per_seed.csv",
            run_dir / "metrics_frozen_partial.csv",
            run_dir / "metrics_adapted_partial.csv",
            run_dir / "metrics_tfidf_partial.csv",
        ]:
            combined_metric_map.update(load_metric_map(prior_path))

    def add_rows(rows: Sequence[dict[str, Any]]) -> None:
        for row in rows:
            combined_metric_map[metric_key(row)] = row

    if args.mode in {"frozen", "both"}:
        add_rows(
            run_frozen_models(
                manifest,
                run_dir,
                args.frozen_models,
                resume=args.resume,
                retry_invalid=args.retry_invalid,
                checkpoint_every=args.checkpoint_every,
            )
        )

    if args.mode in {"adapted", "both"}:
        if "tfidf_lr" in args.adapted_models:
            add_rows(run_tfidf_lr_model(manifest, run_dir, resume=args.resume))
        if "distilbert" in args.adapted_models:
            add_rows(run_distilbert_model(manifest, run_dir, resume=args.resume))

    metric_rows = sorted(combined_metric_map.values(), key=metric_key)
    save_results(run_dir, metric_rows, manifest)


if __name__ == "__main__":
    main()
