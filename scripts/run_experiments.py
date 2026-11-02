#!/usr/bin/env python3
"""Run the paper's frozen and adapted models on the audited split package.

Models retained for paper compatibility
---------------------------------------
Frozen prompted models:
* qwen2.5:3b-instruct
* phi3.5:latest

Task-adapted supervised baselines:
* TF-IDF + Logistic Regression
* distilbert-base-uncased

The script never resamples data. It consumes only files listed in the audited
``split_manifest.json`` and verifies their SHA-256 checksums before execution.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline


DEFAULT_SPLIT_ROOT = Path("data/splits")
DEFAULT_MODELS = ["qwen2.5:3b-instruct", "phi3.5:latest"]
EXPECTED_MODEL_ARTIFACTS = {
    "qwen2.5:3b-instruct": {
        "digest": "357c53fb659c5076de1d65ccb0b397446227b71a42be9d1603d46168015c9e4b",
        "quantization": "Q4_K_M",
    },
    "phi3.5:latest": {
        "digest": "61819fb370a3c1a9be6694869331e5f85f867a079e9271d66cb223acb81d04ba",
        "quantization": "Q4_0",
    },
}
DEFAULT_ADAPTED = ["tfidf_lr", "distilbert"]
DEFAULT_PROMPTS = ["p1_minimal", "p3_json_strict", "p5_abstain"]
CANONICAL_PROMPT = "p3_json_strict"
OLLAMA_HOST = "http://localhost:11434"
CHECKPOINT_EVERY = 25
TIMEOUT_SEC = 600
RETRIES = 2
NUM_PREDICT = 96
MAX_TEXT_CHARS = 800
TFIDF_C_GRID = [0.25, 0.5, 1.0, 2.0, 4.0]
DISTILBERT_MODEL = "distilbert-base-uncased"
MAX_LENGTH = 192
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
EARLY_STOPPING_PATIENCE = 1
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 2
EVAL_ACCUMULATION_STEPS = 16

_thread_local = threading.local()


@dataclass(frozen=True)
class EvaluationCondition:
    dataset: str
    task_type: str
    labels: tuple[str, str]
    seed: int
    period: str
    n_per_label: int
    path: Path


# ---------------------------------------------------------------------------
# CLI and generic helpers
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["frozen", "adapted", "both"],
        default="both",
    )
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--frozen-models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument(
        "--adapted-models",
        nargs="+",
        choices=DEFAULT_ADAPTED,
        default=DEFAULT_ADAPTED,
    )
    parser.add_argument(
        "--prompt-ids",
        nargs="+",
        choices=DEFAULT_PROMPTS,
        default=DEFAULT_PROMPTS,
    )
    parser.add_argument("--ollama-host", default=OLLAMA_HOST)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-invalid", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    parser.add_argument(
        "--allow-failed-audit",
        action="store_true",
        help="Unsafe escape hatch. Normally model execution requires audit_report.json=PASS.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_manifest_and_index(
    split_root: Path, allow_failed_audit: bool
) -> tuple[dict[str, Any], pd.DataFrame]:
    manifest_path = split_root / "split_manifest.json"
    index_path = split_root / "split_index.csv"
    audit_path = split_root / "audit_report.json"
    for path in [manifest_path, index_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required split package file: {path}")
    if not audit_path.exists() and not allow_failed_audit:
        raise FileNotFoundError(
            f"Missing {audit_path}. Run prepare_and_audit_splits.py --stage audit first."
        )
    if audit_path.exists() and not allow_failed_audit:
        audit = load_json(audit_path)
        if audit.get("status") != "PASS":
            raise RuntimeError(
                f"Split audit status is {audit.get('status')!r}; refusing model execution."
            )

    manifest = load_json(manifest_path)
    index = pd.read_csv(index_path)
    for _, row in index.iterrows():
        path = split_root / str(row["path"])
        if not path.exists():
            raise FileNotFoundError(f"Indexed split file is missing: {path}")
        actual = file_sha256(path)
        if actual != str(row["sha256"]):
            raise RuntimeError(f"Split checksum mismatch: {path}")
    return manifest, index


def select_index_row(
    index: pd.DataFrame,
    dataset: str,
    seed: int,
    role: str,
    n_per_label: int | None,
) -> pd.Series:
    mask = (
        index["dataset"].astype(str).eq(dataset)
        & index["seed"].astype(int).eq(seed)
        & index["role"].astype(str).eq(role)
    )
    if n_per_label is None:
        mask &= index["n_per_label"].isna()
    else:
        mask &= index["n_per_label"].fillna(-1).astype(int).eq(n_per_label)
    rows = index.loc[mask]
    if len(rows) != 1:
        raise RuntimeError(
            f"Expected one indexed split for dataset={dataset}, seed={seed}, "
            f"role={role}, n_per_label={n_per_label}; found {len(rows)}."
        )
    return rows.iloc[0]


def load_indexed_frame(split_root: Path, row: pd.Series) -> pd.DataFrame:
    path = split_root / str(row["path"])
    frame = pd.read_csv(path)
    required = {"example_id", "text", "label", "text_sha256", "sample_seed"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"{path} is missing columns: {missing}")
    if file_sha256(path) != str(row["sha256"]):
        raise RuntimeError(f"Checksum changed during execution: {path}")
    return frame


def requested_scope(
    manifest: Mapping[str, Any], args: argparse.Namespace
) -> tuple[list[str], list[int]]:
    available_datasets = list(manifest["datasets"])
    datasets = args.datasets or available_datasets
    unknown = sorted(set(datasets) - set(available_datasets))
    if unknown:
        raise ValueError(f"Unknown datasets requested: {unknown}")
    available_seeds = [int(value) for value in manifest["seeds"]]
    seeds = args.seeds or available_seeds
    unavailable = sorted(set(seeds) - set(available_seeds))
    if unavailable:
        raise ValueError(f"Unavailable seeds requested: {unavailable}")
    return list(datasets), list(seeds)


def evaluation_conditions(
    manifest: Mapping[str, Any],
    index: pd.DataFrame,
    split_root: Path,
    datasets: Sequence[str],
    seeds: Sequence[int],
) -> list[EvaluationCondition]:
    conditions: list[EvaluationCondition] = []
    for dataset in datasets:
        cfg = manifest["datasets"][dataset]
        labels = tuple(cfg["labels"])
        for seed in seeds:
            for period in ["within", "short", "long"]:
                for n_per_label in cfg["evaluation_sizes_per_label"]:
                    row = select_index_row(
                        index, dataset, int(seed), period, int(n_per_label)
                    )
                    conditions.append(
                        EvaluationCondition(
                            dataset=dataset,
                            task_type=str(cfg["task_type"]),
                            labels=(str(labels[0]), str(labels[1])),
                            seed=int(seed),
                            period=period,
                            n_per_label=int(n_per_label),
                            path=split_root / str(row["path"]),
                        )
                    )
    return conditions


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def expected_calibration_error(
    correct: Sequence[bool], confidence: Sequence[float], n_bins: int = 10
) -> float:
    correct_arr = np.asarray(correct, dtype=float)
    confidence_arr = np.asarray(confidence, dtype=float)
    valid = np.isfinite(confidence_arr)
    if not valid.any():
        return math.nan
    correct_arr = correct_arr[valid]
    confidence_arr = np.clip(confidence_arr[valid], 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(confidence_arr)
    ece = 0.0
    for index in range(n_bins):
        lower, upper = edges[index], edges[index + 1]
        if index == n_bins - 1:
            mask = (confidence_arr >= lower) & (confidence_arr <= upper)
        else:
            mask = (confidence_arr >= lower) & (confidence_arr < upper)
        if not mask.any():
            continue
        ece += (mask.sum() / total) * abs(
            float(correct_arr[mask].mean()) - float(confidence_arr[mask].mean())
        )
    return float(ece)


def condition_metrics(
    predictions: pd.DataFrame,
    dataset: str,
    task_type: str,
    seed: int,
    period: str,
    n_per_label: int,
    family: str,
    model: str,
    prompt_id: str | None,
    confidence_source: str,
) -> dict[str, Any]:
    status = predictions.get("status", pd.Series(["ok"] * len(predictions)))
    valid_mask = status.astype(str).eq("ok") & predictions["predicted_label"].notna()
    valid = predictions.loc[valid_mask].copy()
    total = len(predictions)
    invalid = total - len(valid)

    if len(valid):
        accuracy = accuracy_score(valid["true_label"], valid["predicted_label"])
        macro_f1 = f1_score(
            valid["true_label"], valid["predicted_label"], average="macro"
        )
        correct = valid["true_label"].eq(valid["predicted_label"])
        confidence = pd.to_numeric(valid["confidence"], errors="coerce")
        ece = expected_calibration_error(correct, confidence)
        errors = int((~correct).sum())
    else:
        accuracy = macro_f1 = ece = math.nan
        errors = 0

    abstain = (
        predictions["abstain"].map(bool_value)
        if "abstain" in predictions.columns
        else pd.Series([False] * total)
    )
    covered = valid.loc[~abstain.loc[valid.index]] if len(valid) else valid
    covered_correct = (
        covered["true_label"].eq(covered["predicted_label"])
        if len(covered)
        else pd.Series(dtype=bool)
    )

    latency = pd.to_numeric(predictions.get("latency_sec"), errors="coerce")
    return {
        "dataset": dataset,
        "task_type": task_type,
        "seed": seed,
        "period": period,
        "n_per_label": n_per_label,
        "family": family,
        "model": model,
        "prompt_id": prompt_id,
        "rows": total,
        "valid_rows": int(len(valid)),
        "invalid_rows": int(invalid),
        "invalid_rate": float(invalid / total) if total else math.nan,
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "errors": errors,
        "error_rate_valid": float(errors / len(valid)) if len(valid) else math.nan,
        "confidence_source": confidence_source,
        "ece_type": (
            "self_reported_ece"
            if confidence_source == "llm_self_reported"
            else "probability_ece"
        ),
        "ece": float(ece),
        "abstain_rows": int(abstain.sum()),
        "abstain_rate": float(abstain.mean()) if total else math.nan,
        "coverage": float(len(covered) / total) if total else math.nan,
        "covered_accuracy": (
            float(covered_correct.mean()) if len(covered_correct) else math.nan
        ),
        "latency_mean_sec": float(latency.mean()) if latency.notna().any() else math.nan,
        "latency_median_sec": (
            float(latency.median()) if latency.notna().any() else math.nan
        ),
    }


def metrics_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    prompt_value = row.get("prompt_id")
    prompt_key = "" if prompt_value is None or pd.isna(prompt_value) else str(prompt_value)
    return (
        str(row["dataset"]),
        int(row["seed"]),
        str(row["period"]),
        int(row["n_per_label"]),
        str(row["family"]),
        str(row["model"]),
        prompt_key,
    )


def save_metric_rows(run_dir: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path = run_dir / "metrics_per_condition.csv"
    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    if path.exists():
        for row in pd.read_csv(path).to_dict(orient="records"):
            unique[metrics_key(row)] = row
    for row in rows:
        unique[metrics_key(row)] = dict(row)
    ordered = [unique[key] for key in sorted(unique)]
    atomic_write_csv(path, pd.DataFrame(ordered))


# ---------------------------------------------------------------------------
# Prompted frozen models
# ---------------------------------------------------------------------------


def task_definition(task_type: str, labels: Sequence[str]) -> str:
    first, second = labels
    if task_type == "climate_stance":
        return (
            f"{first}: accepts, supports, or expresses concern about climate change, "
            f"climate science, or climate action.\n"
            f"{second}: rejects, doubts, mocks, minimizes, or argues against climate "
            "change, climate science, or climate action."
        )
    if task_type == "sentiment":
        return (
            f"{first}: the overall opinion is negative.\n"
            f"{second}: the overall opinion is positive."
        )
    return f"Choose exactly one label: {first} or {second}."


def build_prompt(
    prompt_id: str, task_type: str, labels: Sequence[str], text: str
) -> str:
    first, second = labels
    definition = task_definition(task_type, labels)
    clipped = text[:MAX_TEXT_CHARS]
    task_name = "climate stance" if task_type == "climate_stance" else "sentiment"

    if prompt_id == "p1_minimal":
        return (
            f"Classify the {task_name} of TEXT as {first} or {second}.\n"
            "Return exactly one JSON object with keys label, confidence, abstain.\n"
            f"Definitions:\n{definition}\n"
            "confidence must be between 0 and 1. abstain must be false.\n"
            f"TEXT:\n{clipped}"
        )
    if prompt_id == "p3_json_strict":
        return (
            f"Determine the {task_name} label for TEXT. Allowed labels are exactly "
            f"{first} and {second}.\n"
            "Output one JSON object only. Do not add markdown or explanation.\n"
            f"Definitions:\n{definition}\n"
            "Use this exact structure: "
            f'{{"label":"{first}","confidence":0.90,"abstain":false}}\n'
            "confidence must be a number in [0,1]. abstain must be false.\n"
            f"TEXT:\n{clipped}"
        )
    if prompt_id == "p5_abstain":
        return (
            f"Classify the {task_name} of TEXT as {first} or {second}.\n"
            "Return one JSON object only with keys label, confidence, abstain.\n"
            f"Definitions:\n{definition}\n"
            "Set abstain=true only when the text is genuinely mixed, unclear, or "
            "insufficient for a defensible decision. Even when abstaining, provide the "
            "best-guess label and confidence no higher than 0.50.\n"
            f"TEXT:\n{clipped}"
        )
    raise KeyError(f"Unknown prompt id: {prompt_id}")


def json_schema(labels: Sequence[str], prompt_id: str) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "label": {"type": "string", "enum": list(labels)},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "abstain": (
                {"type": "boolean"}
                if prompt_id == "p5_abstain"
                else {"type": "boolean", "const": False}
            ),
        },
        "required": ["label", "confidence", "abstain"],
    }


def get_http_session() -> requests.Session:
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        _thread_local.session = session
    return session


def ollama_model_inventory(host: str) -> dict[str, Any]:
    response = requests.get(f"{host.rstrip('/')}/api/tags", timeout=30)
    response.raise_for_status()
    payload = response.json()
    return {
        str(item.get("name") or item.get("model")): item
        for item in payload.get("models", [])
    }


def warmup_ollama(host: str, model: str) -> None:
    response = requests.post(
        f"{host.rstrip('/')}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": "Reply with OK."}],
            "stream": False,
            "keep_alive": "30m",
            "options": {"temperature": 0.0, "num_predict": 4, "seed": 1},
        },
        timeout=TIMEOUT_SEC,
    )
    response.raise_for_status()


def call_ollama(
    host: str,
    model: str,
    prompt: str,
    labels: Sequence[str],
    prompt_id: str,
) -> tuple[str, float, str | None]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "Follow the classification instructions exactly.",
            },
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "format": json_schema(labels, prompt_id),
        "keep_alive": "30m",
        "options": {
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 1,
            "num_predict": NUM_PREDICT,
        },
    }
    last_error: str | None = None
    for attempt in range(RETRIES + 1):
        start = time.perf_counter()
        try:
            response = get_http_session().post(
                f"{host.rstrip('/')}/api/chat", json=payload, timeout=TIMEOUT_SEC
            )
            response.raise_for_status()
            elapsed = time.perf_counter() - start
            data = response.json()
            return str(data.get("message", {}).get("content", "")), elapsed, None
        except Exception as exc:  # noqa: BLE001 - preserve operational error text
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < RETRIES:
                time.sleep(2**attempt)
    return "", math.nan, last_error


def parse_frozen_response(
    raw: str, labels: Sequence[str], prompt_id: str
) -> tuple[str | None, float, bool, str]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            return None, math.nan, False, "invalid_json"
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None, math.nan, False, "invalid_json"

    label = str(payload.get("label", "")).strip().upper()
    confidence = safe_float(payload.get("confidence"))
    abstain = bool_value(payload.get("abstain", False))
    if label not in set(labels):
        return None, confidence, abstain, "invalid_label"
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        return label, confidence, abstain, "invalid_confidence"
    if prompt_id != "p5_abstain" and abstain:
        return label, confidence, abstain, "unexpected_abstain"
    if prompt_id == "p5_abstain" and abstain and confidence > 0.5:
        return label, confidence, abstain, "abstain_confidence_violation"
    return label, confidence, abstain, "ok"


def frozen_prediction_path(
    run_dir: Path,
    model: str,
    prompt_id: str,
    condition: EvaluationCondition,
) -> Path:
    return (
        run_dir
        / "predictions"
        / "frozen"
        / slugify(model)
        / prompt_id
        / condition.dataset
        / f"seed_{condition.seed}"
        / f"{condition.period}_n{condition.n_per_label}_per_label.csv"
    )


def frozen_existing_map(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    if "text_sha256" not in frame.columns:
        return {}
    return {
        str(row["text_sha256"]): row.to_dict()
        for _, row in frame.drop_duplicates("text_sha256", keep="last").iterrows()
    }


def run_frozen_condition(
    host: str,
    model: str,
    prompt_id: str,
    condition: EvaluationCondition,
    output_path: Path,
    num_workers: int,
    resume: bool,
    retry_invalid: bool,
    checkpoint_every: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    eval_df = pd.read_csv(condition.path)
    existing = frozen_existing_map(output_path) if resume else {}
    rows: dict[str, dict[str, Any]] = dict(existing)

    pending: list[dict[str, Any]] = []
    for record in eval_df.to_dict(orient="records"):
        key = str(record["text_sha256"])
        prior = existing.get(key)
        if prior is None:
            pending.append(record)
        elif retry_invalid and str(prior.get("status")) != "ok":
            pending.append(record)

    def classify(record: Mapping[str, Any]) -> dict[str, Any]:
        prompt = build_prompt(
            prompt_id,
            condition.task_type,
            condition.labels,
            str(record["text"]),
        )
        raw, latency, request_error = call_ollama(
            host, model, prompt, condition.labels, prompt_id
        )
        if request_error:
            predicted, confidence, abstain, status = None, math.nan, False, "request_error"
        else:
            predicted, confidence, abstain, status = parse_frozen_response(
                raw, condition.labels, prompt_id
            )
        return {
            "dataset": condition.dataset,
            "task_type": condition.task_type,
            "seed": condition.seed,
            "period": condition.period,
            "n_per_label": condition.n_per_label,
            "family": "frozen",
            "model": model,
            "prompt_id": prompt_id,
            "example_id": record["example_id"],
            "text_sha256": record["text_sha256"],
            "text": record["text"],
            "true_label": record["label"],
            "predicted_label": predicted,
            "confidence": confidence,
            "confidence_source": "llm_self_reported",
            "abstain": abstain,
            "is_correct": bool(predicted == record["label"]) if predicted else False,
            "status": status,
            "request_error": request_error,
            "latency_sec": latency,
            "raw_response": raw,
        }

    completed_since_save = 0
    if pending:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(classify, record): record for record in pending}
            for future in as_completed(futures):
                result = future.result()
                rows[str(result["text_sha256"])] = result
                completed_since_save += 1
                if completed_since_save >= checkpoint_every:
                    ordered = pd.DataFrame(
                        [rows[str(hash_value)] for hash_value in eval_df["text_sha256"] if str(hash_value) in rows]
                    )
                    atomic_write_csv(output_path, ordered)
                    completed_since_save = 0

    missing = [
        str(value) for value in eval_df["text_sha256"] if str(value) not in rows
    ]
    if missing:
        raise RuntimeError(
            f"Frozen condition remains incomplete: {model}/{prompt_id}/"
            f"{condition.dataset}/seed={condition.seed}/{condition.period}/"
            f"n={condition.n_per_label}; missing {len(missing)} rows."
        )
    ordered = pd.DataFrame([rows[str(value)] for value in eval_df["text_sha256"]])
    atomic_write_csv(output_path, ordered)
    metric = condition_metrics(
        ordered,
        condition.dataset,
        condition.task_type,
        condition.seed,
        condition.period,
        condition.n_per_label,
        "frozen",
        model,
        prompt_id,
        "llm_self_reported",
    )
    return ordered, metric


def run_frozen(
    manifest: Mapping[str, Any],
    conditions: Sequence[EvaluationCondition],
    run_dir: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    inventory = ollama_model_inventory(args.ollama_host)
    missing_models = [model for model in args.frozen_models if model not in inventory]
    if missing_models:
        available = sorted(inventory)
        raise RuntimeError(
            f"Ollama models not found: {missing_models}. Available models: {available}"
        )
    atomic_write_json(
        run_dir / "ollama_model_inventory.json",
        {model: inventory[model] for model in args.frozen_models},
    )

    metric_rows: list[dict[str, Any]] = []
    for model in args.frozen_models:
        print(f"\n[FROZEN] Warming {model}")
        warmup_ollama(args.ollama_host, model)
        for prompt_id in args.prompt_ids:
            for condition in conditions:
                if getattr(args, "paper_scope", False) and prompt_id != "p3_json_strict" and condition.period != "within":
                    continue
                print(
                    f"[FROZEN] {model} {prompt_id} {condition.dataset} "
                    f"seed={condition.seed} {condition.period} n={condition.n_per_label}"
                )
                output = frozen_prediction_path(run_dir, model, prompt_id, condition)
                _, metric = run_frozen_condition(
                    args.ollama_host,
                    model,
                    prompt_id,
                    condition,
                    output,
                    args.num_workers,
                    args.resume,
                    args.retry_invalid,
                    args.checkpoint_every,
                )
                metric_rows.append(metric)
                save_metric_rows(run_dir, metric_rows)
    return metric_rows


# ---------------------------------------------------------------------------
# TF-IDF Logistic Regression
# ---------------------------------------------------------------------------


def adapted_prediction_path(
    run_dir: Path, model: str, condition: EvaluationCondition
) -> Path:
    return (
        run_dir
        / "predictions"
        / "adapted"
        / model
        / condition.dataset
        / f"seed_{condition.seed}"
        / f"{condition.period}_n{condition.n_per_label}_per_label.csv"
    )


def prediction_complete(path: Path, eval_df: pd.DataFrame) -> bool:
    if not path.exists():
        return False
    try:
        pred = pd.read_csv(path)
    except Exception:
        return False
    if len(pred) != len(eval_df) or "text_sha256" not in pred.columns:
        return False
    return list(pred["text_sha256"].astype(str)) == list(
        eval_df["text_sha256"].astype(str)
    )


def fit_tfidf_model(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    labels: Sequence[str],
    seed: int,
) -> tuple[Pipeline, dict[str, Any]]:
    best_model: Pipeline | None = None
    best_score = -math.inf
    trials: list[dict[str, Any]] = []
    for c_value in TFIDF_C_GRID:
        pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=50_000,
                        min_df=2,
                        ngram_range=(1, 2),
                        sublinear_tf=True,
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        C=c_value,
                        max_iter=2_000,
                        solver="liblinear",
                        random_state=seed,
                    ),
                ),
            ]
        )
        start = time.perf_counter()
        pipeline.fit(train_df["text"], train_df["label"])
        validation_pred = pipeline.predict(validation_df["text"])
        score = f1_score(validation_df["label"], validation_pred, average="macro")
        trials.append(
            {
                "C": c_value,
                "validation_macro_f1": float(score),
                "fit_seconds": float(time.perf_counter() - start),
            }
        )
        if score > best_score:
            best_score = float(score)
            best_model = pipeline
    assert best_model is not None
    return best_model, {
        "selection_metric": "validation_macro_f1",
        "selected_C": float(best_model.named_steps["classifier"].C),
        "best_validation_macro_f1": best_score,
        "trials": trials,
        "refit_on_validation": False,
        "training_rows": int(len(train_df)),
        "validation_rows": int(len(validation_df)),
        "labels": list(labels),
    }


def predict_supervised(
    model: Any,
    eval_df: pd.DataFrame,
    labels: Sequence[str],
    dataset: str,
    task_type: str,
    seed: int,
    period: str,
    n_per_label: int,
    model_name: str,
) -> pd.DataFrame:
    start = time.perf_counter()
    predicted = model.predict(eval_df["text"])
    probabilities = model.predict_proba(eval_df["text"])
    elapsed = time.perf_counter() - start
    classes = [str(value) for value in model.classes_]
    class_to_index = {label: index for index, label in enumerate(classes)}
    confidence = probabilities.max(axis=1)

    result = pd.DataFrame(
        {
            "dataset": dataset,
            "task_type": task_type,
            "seed": seed,
            "period": period,
            "n_per_label": n_per_label,
            "family": "adapted",
            "model": model_name,
            "prompt_id": None,
            "example_id": eval_df["example_id"],
            "text_sha256": eval_df["text_sha256"],
            "text": eval_df["text"],
            "true_label": eval_df["label"],
            "predicted_label": predicted,
            "confidence": confidence,
            "confidence_source": "predicted_probability",
            "abstain": False,
            "status": "ok",
            "latency_sec": elapsed / max(len(eval_df), 1),
        }
    )
    for label in labels:
        result[f"prob_{slugify(label).lower()}"] = probabilities[:, class_to_index[label]]
    result["is_correct"] = result["true_label"].eq(result["predicted_label"])
    return result


def run_tfidf(
    manifest: Mapping[str, Any],
    index: pd.DataFrame,
    split_root: Path,
    conditions: Sequence[EvaluationCondition],
    run_dir: Path,
    datasets: Sequence[str],
    seeds: Sequence[int],
    resume: bool,
) -> list[dict[str, Any]]:
    metric_rows: list[dict[str, Any]] = []
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("joblib is required for TF-IDF model persistence") from exc

    for dataset in datasets:
        cfg = manifest["datasets"][dataset]
        labels = tuple(cfg["labels"])
        task_type = str(cfg["task_type"])
        for seed in seeds:
            seed_conditions = [
                condition
                for condition in conditions
                if condition.dataset == dataset and condition.seed == seed
            ]
            train_row = select_index_row(index, dataset, seed, "train", None)
            validation_row = select_index_row(index, dataset, seed, "validation", None)
            train_df = load_indexed_frame(split_root, train_row)
            validation_df = load_indexed_frame(split_root, validation_row)

            model_dir = run_dir / "models" / "tfidf_lr" / dataset / f"seed_{seed}"
            model_path = model_dir / "model.joblib"
            selection_path = model_dir / "selection.json"
            if resume and model_path.exists() and selection_path.exists():
                model = joblib.load(model_path)
            else:
                print(f"\n[ADAPTED][TFIDF] fitting {dataset} seed={seed}")
                model, selection = fit_tfidf_model(train_df, validation_df, labels, seed)
                model_dir.mkdir(parents=True, exist_ok=True)
                joblib.dump(model, model_path)
                atomic_write_json(selection_path, selection)

            for condition in seed_conditions:
                eval_df = pd.read_csv(condition.path)
                output = adapted_prediction_path(run_dir, "tfidf_lr", condition)
                if resume and prediction_complete(output, eval_df):
                    predictions = pd.read_csv(output)
                else:
                    predictions = predict_supervised(
                        model,
                        eval_df,
                        labels,
                        dataset,
                        task_type,
                        seed,
                        condition.period,
                        condition.n_per_label,
                        "tfidf_lr",
                    )
                    atomic_write_csv(output, predictions)
                metric_rows.append(
                    condition_metrics(
                        predictions,
                        dataset,
                        task_type,
                        seed,
                        condition.period,
                        condition.n_per_label,
                        "adapted",
                        "tfidf_lr",
                        None,
                        "predicted_probability",
                    )
                )
                save_metric_rows(run_dir, metric_rows)
    return metric_rows


# ---------------------------------------------------------------------------
# DistilBERT
# ---------------------------------------------------------------------------


def require_transformers() -> dict[str, Any]:
    try:
        import torch
        import transformers
        from datasets import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError(
            "DistilBERT requires torch, transformers, datasets, and accelerate. "
            "Install the full requirements before rerunning."
        ) from exc
    return locals()


def build_training_arguments(TrainingArguments: Any, output_dir: Path) -> Any:
    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "per_device_train_batch_size": TRAIN_BATCH_SIZE,
        "per_device_eval_batch_size": EVAL_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "eval_accumulation_steps": EVAL_ACCUMULATION_STEPS,
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": 100,
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "save_total_limit": 2,
        "report_to": [],
        "seed": 0,
        "data_seed": 0,
        "dataloader_num_workers": 0,
        "fp16": False,
    }
    signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"
    return TrainingArguments(**kwargs)


def train_distilbert(
    deps: Mapping[str, Any],
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    labels: Sequence[str],
    output_dir: Path,
    seed: int,
) -> tuple[Any, Any, dict[str, Any]]:
    torch = deps["torch"]
    Dataset = deps["Dataset"]
    AutoTokenizer = deps["AutoTokenizer"]
    AutoModelForSequenceClassification = deps["AutoModelForSequenceClassification"]
    DataCollatorWithPadding = deps["DataCollatorWithPadding"]
    EarlyStoppingCallback = deps["EarlyStoppingCallback"]
    Trainer = deps["Trainer"]
    TrainingArguments = deps["TrainingArguments"]

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    label_to_id = {label: index for index, label in enumerate(labels)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_MODEL, use_fast=True)

    def make_dataset(frame: pd.DataFrame) -> Any:
        dataset = Dataset.from_dict(
            {
                "text": frame["text"].astype(str).tolist(),
                "labels": [label_to_id[value] for value in frame["label"]],
            }
        )
        return dataset.map(
            lambda batch: tokenizer(
                batch["text"], truncation=True, max_length=MAX_LENGTH
            ),
            batched=True,
            remove_columns=["text"],
        )

    train_ds = make_dataset(train_df)
    validation_ds = make_dataset(validation_df)
    model = AutoModelForSequenceClassification.from_pretrained(
        DISTILBERT_MODEL,
        num_labels=len(labels),
        id2label=id_to_label,
        label2id=label_to_id,
    )

    def compute_metrics(eval_prediction: Any) -> dict[str, float]:
        logits, true_ids = eval_prediction
        pred_ids = np.asarray(logits).argmax(axis=1)
        return {
            "accuracy": float(accuracy_score(true_ids, pred_ids)),
            "macro_f1": float(f1_score(true_ids, pred_ids, average="macro")),
        }

    arguments = build_training_arguments(TrainingArguments, output_dir)
    arguments.seed = seed
    arguments.data_seed = seed
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=train_ds,
        eval_dataset=validation_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
    )
    result = trainer.train()
    trainer.save_model(str(output_dir / "best_model"))
    tokenizer.save_pretrained(str(output_dir / "best_model"))
    evaluation = trainer.evaluate(validation_ds)
    metadata = {
        "base_model": DISTILBERT_MODEL,
        "training_rows": int(len(train_df)),
        "validation_rows": int(len(validation_df)),
        "refit_on_validation": False,
        "seed": seed,
        "train_runtime": safe_float(result.metrics.get("train_runtime")),
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "validation_metrics": {
            key: safe_float(value) for key, value in evaluation.items()
        },
    }
    return trainer, tokenizer, metadata


def load_distilbert_trainer(deps: Mapping[str, Any], model_dir: Path) -> tuple[Any, Any]:
    AutoTokenizer = deps["AutoTokenizer"]
    AutoModelForSequenceClassification = deps["AutoModelForSequenceClassification"]
    DataCollatorWithPadding = deps["DataCollatorWithPadding"]
    Trainer = deps["Trainer"]
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    return trainer, tokenizer


def predict_distilbert(
    deps: Mapping[str, Any],
    trainer: Any,
    tokenizer: Any,
    eval_df: pd.DataFrame,
    labels: Sequence[str],
    condition: EvaluationCondition,
) -> pd.DataFrame:
    Dataset = deps["Dataset"]
    torch = deps["torch"]
    dataset = Dataset.from_dict({"text": eval_df["text"].astype(str).tolist()})
    tokenized = dataset.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH),
        batched=True,
        remove_columns=["text"],
    )
    start = time.perf_counter()
    prediction = trainer.predict(tokenized)
    elapsed = time.perf_counter() - start
    logits = np.asarray(prediction.predictions)
    probabilities = torch.softmax(torch.tensor(logits), dim=1).cpu().numpy()
    pred_ids = probabilities.argmax(axis=1)
    predicted = [labels[index] for index in pred_ids]
    confidence = probabilities.max(axis=1)
    result = pd.DataFrame(
        {
            "dataset": condition.dataset,
            "task_type": condition.task_type,
            "seed": condition.seed,
            "period": condition.period,
            "n_per_label": condition.n_per_label,
            "family": "adapted",
            "model": "distilbert",
            "prompt_id": None,
            "example_id": eval_df["example_id"],
            "text_sha256": eval_df["text_sha256"],
            "text": eval_df["text"],
            "true_label": eval_df["label"],
            "predicted_label": predicted,
            "confidence": confidence,
            "confidence_source": "predicted_probability",
            "abstain": False,
            "status": "ok",
            "latency_sec": elapsed / max(len(eval_df), 1),
        }
    )
    for index, label in enumerate(labels):
        result[f"prob_{slugify(label).lower()}"] = probabilities[:, index]
    result["is_correct"] = result["true_label"].eq(result["predicted_label"])
    return result


def run_distilbert(
    manifest: Mapping[str, Any],
    index: pd.DataFrame,
    split_root: Path,
    conditions: Sequence[EvaluationCondition],
    run_dir: Path,
    datasets: Sequence[str],
    seeds: Sequence[int],
    resume: bool,
) -> list[dict[str, Any]]:
    deps = require_transformers()
    metric_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        cfg = manifest["datasets"][dataset]
        labels = tuple(cfg["labels"])
        for seed in seeds:
            seed_conditions = [
                condition
                for condition in conditions
                if condition.dataset == dataset and condition.seed == seed
            ]
            train_df = load_indexed_frame(
                split_root, select_index_row(index, dataset, seed, "train", None)
            )
            validation_df = load_indexed_frame(
                split_root, select_index_row(index, dataset, seed, "validation", None)
            )
            model_root = run_dir / "models" / "distilbert" / dataset / f"seed_{seed}"
            best_model = model_root / "best_model"
            metadata_path = model_root / "training_metadata.json"
            if resume and best_model.exists() and metadata_path.exists():
                trainer, tokenizer = load_distilbert_trainer(deps, best_model)
            else:
                print(f"\n[ADAPTED][DISTILBERT] fitting {dataset} seed={seed}")
                model_root.mkdir(parents=True, exist_ok=True)
                trainer, tokenizer, metadata = train_distilbert(
                    deps, train_df, validation_df, labels, model_root, seed
                )
                atomic_write_json(metadata_path, metadata)

            for condition in seed_conditions:
                eval_df = pd.read_csv(condition.path)
                output = adapted_prediction_path(run_dir, "distilbert", condition)
                if resume and prediction_complete(output, eval_df):
                    predictions = pd.read_csv(output)
                else:
                    predictions = predict_distilbert(
                        deps, trainer, tokenizer, eval_df, labels, condition
                    )
                    atomic_write_csv(output, predictions)
                metric_rows.append(
                    condition_metrics(
                        predictions,
                        condition.dataset,
                        condition.task_type,
                        condition.seed,
                        condition.period,
                        condition.n_per_label,
                        "adapted",
                        "distilbert",
                        None,
                        "predicted_probability",
                    )
                )
                save_metric_rows(run_dir, metric_rows)
    return metric_rows


# ---------------------------------------------------------------------------
# Run metadata and main
# ---------------------------------------------------------------------------


def command_output(command: Sequence[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except Exception:
        return None
    text = (result.stdout or result.stderr).strip()
    return text or None


def environment_metadata(script_path: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "created_at_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "script": str(script_path.resolve()),
        "script_sha256": file_sha256(script_path),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "ollama_version": command_output(["ollama", "--version"]),
        "nvidia_smi": command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
    }
    try:
        import sklearn

        metadata["scikit_learn"] = sklearn.__version__
    except Exception:
        pass
    try:
        import torch

        metadata["torch"] = torch.__version__
        metadata["cuda_available"] = bool(torch.cuda.is_available())
        metadata["cuda_version"] = torch.version.cuda
        metadata["gpu_name"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
    except Exception:
        metadata["torch"] = None
    try:
        import transformers

        metadata["transformers"] = transformers.__version__
    except Exception:
        metadata["transformers"] = None
    return metadata


def merge_existing_metrics(run_dir: Path, new_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    existing_path = run_dir / "metrics_per_condition.csv"
    merged: dict[tuple[Any, ...], dict[str, Any]] = {}
    if existing_path.exists():
        for row in pd.read_csv(existing_path).to_dict(orient="records"):
            merged[metrics_key(row)] = row
    for row in new_rows:
        merged[metrics_key(row)] = dict(row)
    return [merged[key] for key in sorted(merged)]


def main() -> None:
    args = parse_args()
    if args.retry_invalid and not args.resume:
        raise ValueError("--retry-invalid requires --resume")
    if args.num_workers < 1 or args.checkpoint_every < 1:
        raise ValueError("--num-workers and --checkpoint-every must be positive")

    manifest, index = load_manifest_and_index(
        args.split_root, args.allow_failed_audit
    )
    datasets, seeds = requested_scope(manifest, args)
    conditions = evaluation_conditions(
        manifest, index, args.split_root, datasets, seeds
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.run_dir or Path("runs") / f"paper_rerun_{timestamp}"
    if args.resume and not run_dir.exists():
        raise FileNotFoundError(f"Cannot resume missing run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    atomic_write_json(run_dir / "environment.json", environment_metadata(Path(__file__)))
    atomic_write_json(
        run_dir / "run_settings.json",
        {
            "created_at_utc": utc_now(),
            "mode": args.mode,
            "split_root": str(args.split_root.resolve()),
            "split_manifest_sha256": file_sha256(
                args.split_root / "split_manifest.json"
            ),
            "datasets": datasets,
            "seeds": seeds,
            "conditions": len(conditions),
            "frozen_models": args.frozen_models,
            "adapted_models": args.adapted_models,
            "prompt_ids": args.prompt_ids,
            "canonical_prompt": CANONICAL_PROMPT,
            "resume": bool(args.resume),
            "retry_invalid": bool(args.retry_invalid),
            "ollama_host": args.ollama_host,
            "num_workers": args.num_workers,
            "tfidf_C_grid": TFIDF_C_GRID,
            "distilbert": {
                "base_model": DISTILBERT_MODEL,
                "max_length": MAX_LENGTH,
                "epochs": NUM_EPOCHS,
                "learning_rate": LEARNING_RATE,
                "train_batch_size": TRAIN_BATCH_SIZE,
                "eval_batch_size": EVAL_BATCH_SIZE,
                "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            },
        },
    )

    all_new_rows: list[dict[str, Any]] = []
    if args.mode in {"frozen", "both"}:
        rows = run_frozen(manifest, conditions, run_dir, args)
        all_new_rows.extend(rows)
        save_metric_rows(run_dir, merge_existing_metrics(run_dir, all_new_rows))

    if args.mode in {"adapted", "both"}:
        if "tfidf_lr" in args.adapted_models:
            rows = run_tfidf(
                manifest,
                index,
                args.split_root,
                conditions,
                run_dir,
                datasets,
                seeds,
                args.resume,
            )
            all_new_rows.extend(rows)
            save_metric_rows(run_dir, merge_existing_metrics(run_dir, all_new_rows))
        if "distilbert" in args.adapted_models:
            rows = run_distilbert(
                manifest,
                index,
                args.split_root,
                conditions,
                run_dir,
                datasets,
                seeds,
                args.resume,
            )
            all_new_rows.extend(rows)
            save_metric_rows(run_dir, merge_existing_metrics(run_dir, all_new_rows))

    final_rows = merge_existing_metrics(run_dir, all_new_rows)
    save_metric_rows(run_dir, final_rows)
    print(f"\n[DONE] Run directory: {run_dir.resolve()}")
    print(f"[DONE] Metrics: {(run_dir / 'metrics_per_condition.csv').resolve()}")


if __name__ == "__main__":
    main()
