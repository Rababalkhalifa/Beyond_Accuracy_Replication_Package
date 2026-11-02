#!/usr/bin/env python3
"""Create reviewer-facing analysis tables from a completed rerun.

The analysis deliberately keeps supervised predicted probabilities separate
from prompted-LLM self-reported confidence. Cross-family calibration values are
reported but never pooled into one calibration ranking.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix


CANONICAL_PROMPT = "p3_json_strict"
MODEL_ORDER = ["tfidf_lr", "distilbert", "qwen2.5:3b-instruct", "phi3.5:latest"]
PERIOD_ORDER = ["within", "short", "long"]
BOOTSTRAP_SAMPLES = 10_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--failure-threshold", type=float, default=0.70)
    parser.add_argument("--top-error-examples", type=int, default=50)
    parser.add_argument("--top-shift-terms", type=int, default=25)
    return parser.parse_args()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    tmp.replace(path)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def prompt_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def canonical_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    prompts = metrics["prompt_id"].map(prompt_value)
    return metrics[(metrics["family"] == "adapted") | prompts.eq(CANONICAL_PROMPT)].copy()


def bootstrap_mean_ci(
    values: Sequence[float], seed: int = 20260803
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return math.nan, math.nan
    if len(array) == 1:
        return float(array[0]), float(array[0])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(BOOTSTRAP_SAMPLES, len(array)))
    means = array[indices].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def aggregate_seed_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "dataset",
        "task_type",
        "period",
        "n_per_label",
        "family",
        "model",
        "prompt_id",
        "confidence_source",
        "ece_type",
    ]
    metric_cols = [
        "accuracy",
        "macro_f1",
        "ece",
        "invalid_rate",
        "abstain_rate",
        "coverage",
        "covered_accuracy",
        "latency_mean_sec",
        "errors",
        "error_rate_valid",
    ]
    rows: list[dict[str, Any]] = []
    work = metrics.copy()
    work["prompt_id"] = work["prompt_id"].map(prompt_value)
    for keys, group in work.groupby(group_cols, dropna=False, sort=True):
        row = dict(zip(group_cols, keys))
        row["seed_count"] = int(group["seed"].nunique())
        for metric in metric_cols:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy()
            row[f"{metric}_mean"] = float(np.mean(values)) if len(values) else math.nan
            row[f"{metric}_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else math.nan
            )
            lower, upper = bootstrap_mean_ci(values)
            row[f"{metric}_ci95_low"] = lower
            row[f"{metric}_ci95_high"] = upper
        rows.append(row)
    return pd.DataFrame(rows)


def exact_sign_flip_pvalue(differences: Sequence[float]) -> float:
    diff = np.asarray(differences, dtype=float)
    diff = diff[np.isfinite(diff)]
    if len(diff) == 0:
        return math.nan
    observed = abs(float(diff.mean()))
    means = []
    for signs in itertools.product([-1.0, 1.0], repeat=len(diff)):
        means.append(abs(float((diff * np.asarray(signs)).mean())))
    return float(np.mean(np.asarray(means) >= observed - 1e-15))


def holm_adjust(pvalues: Sequence[float]) -> list[float]:
    values = np.asarray(pvalues, dtype=float)
    adjusted = np.full(len(values), np.nan)
    valid_indices = np.where(np.isfinite(values))[0]
    if not len(valid_indices):
        return adjusted.tolist()
    order = valid_indices[np.argsort(values[valid_indices])]
    running = 0.0
    m = len(order)
    for rank, index in enumerate(order):
        candidate = min(1.0, (m - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def paired_model_tests(metrics: pd.DataFrame) -> pd.DataFrame:
    work = canonical_metrics(metrics)
    rows: list[dict[str, Any]] = []
    setting_cols = ["dataset", "period", "n_per_label"]
    for setting, group in work.groupby(setting_cols, sort=True):
        pivot = group.pivot_table(index="seed", columns="model", values="macro_f1", aggfunc="first")
        models = [model for model in MODEL_ORDER if model in pivot.columns]
        local_rows: list[dict[str, Any]] = []
        for model_a, model_b in itertools.combinations(models, 2):
            paired = pivot[[model_a, model_b]].dropna()
            differences = paired[model_a].to_numpy() - paired[model_b].to_numpy()
            if len(differences) == 0:
                continue
            lower, upper = bootstrap_mean_ci(differences, seed=20260803 + len(rows))
            sd = float(np.std(differences, ddof=1)) if len(differences) > 1 else math.nan
            effect_dz = float(np.mean(differences) / sd) if sd and math.isfinite(sd) else math.nan
            local_rows.append(
                {
                    "dataset": setting[0],
                    "period": setting[1],
                    "n_per_label": int(setting[2]),
                    "model_a": model_a,
                    "model_b": model_b,
                    "paired_seeds": int(len(differences)),
                    "mean_macro_f1_difference_a_minus_b": float(np.mean(differences)),
                    "median_difference": float(np.median(differences)),
                    "difference_ci95_low": lower,
                    "difference_ci95_high": upper,
                    "paired_effect_dz": effect_dz,
                    "exact_sign_flip_p": exact_sign_flip_pvalue(differences),
                }
            )
        adjusted = holm_adjust([row["exact_sign_flip_p"] for row in local_rows])
        for row, p_adjusted in zip(local_rows, adjusted):
            row["holm_adjusted_p"] = p_adjusted
            rows.append(row)
    return pd.DataFrame(rows)


def temporal_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    work = canonical_metrics(metrics)
    key_cols = ["dataset", "task_type", "n_per_label", "family", "model", "seed"]
    rows: list[dict[str, Any]] = []
    for keys, group in work.groupby(key_cols, sort=True):
        period_map = {row["period"]: row for _, row in group.iterrows()}
        if "within" not in period_map:
            continue
        baseline = period_map["within"]
        for target in ["short", "long"]:
            if target not in period_map:
                continue
            target_row = period_map[target]
            rows.append(
                {
                    **dict(zip(key_cols, keys)),
                    "target_period": target,
                    "within_macro_f1": baseline["macro_f1"],
                    "target_macro_f1": target_row["macro_f1"],
                    "macro_f1_change_target_minus_within": target_row["macro_f1"] - baseline["macro_f1"],
                    "within_errors": baseline["errors"],
                    "target_errors": target_row["errors"],
                    "error_count_change": target_row["errors"] - baseline["errors"],
                }
            )
    return pd.DataFrame(rows)


def decision_test(metrics: pd.DataFrame, threshold: float) -> pd.DataFrame:
    work = canonical_metrics(metrics)
    rows: list[dict[str, Any]] = []
    for (dataset, n_per_label, seed), group in work.groupby(
        ["dataset", "n_per_label", "seed"], sort=True
    ):
        within = group[group["period"] == "within"].dropna(subset=["macro_f1"])
        if within.empty:
            continue
        headline = within.sort_values(
            ["macro_f1", "invalid_rate", "model"], ascending=[False, True, True]
        ).iloc[0]
        selected_model = headline["model"]

        eligible = within[
            (within["invalid_rate"].fillna(1.0) <= 0.01)
            & (within["coverage"].fillna(0.0) >= 0.90)
        ]
        if eligible.empty:
            constrained = headline
        else:
            constrained = eligible.sort_values(
                ["macro_f1", "model"], ascending=[False, True]
            ).iloc[0]
        constrained_model = constrained["model"]

        for target in ["short", "long"]:
            future = group[group["period"] == target].dropna(subset=["macro_f1"])
            if future.empty:
                continue
            oracle = future.sort_values(["macro_f1", "model"], ascending=[False, True]).iloc[0]
            selected_future = future[future["model"] == selected_model]
            constrained_future = future[future["model"] == constrained_model]
            if selected_future.empty or constrained_future.empty:
                continue
            selected_future = selected_future.iloc[0]
            constrained_future = constrained_future.iloc[0]
            rows.append(
                {
                    "dataset": dataset,
                    "n_per_label": int(n_per_label),
                    "seed": int(seed),
                    "target_period": target,
                    "headline_selected_model": selected_model,
                    "headline_within_macro_f1": float(headline["macro_f1"]),
                    "headline_future_macro_f1": float(selected_future["macro_f1"]),
                    "headline_future_failure_below_threshold": bool(
                        selected_future["macro_f1"] < threshold
                    ),
                    "reliability_constrained_model": constrained_model,
                    "reliability_constrained_future_macro_f1": float(
                        constrained_future["macro_f1"]
                    ),
                    "oracle_future_model": oracle["model"],
                    "oracle_future_macro_f1": float(oracle["macro_f1"]),
                    "headline_regret": float(
                        oracle["macro_f1"] - selected_future["macro_f1"]
                    ),
                    "reliability_constrained_regret": float(
                        oracle["macro_f1"] - constrained_future["macro_f1"]
                    ),
                    "headline_selected_future_winner": bool(
                        selected_model == oracle["model"]
                    ),
                    "reliability_constrained_future_winner": bool(
                        constrained_model == oracle["model"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def prompt_sensitivity(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frozen = metrics[metrics["family"] == "frozen"].copy()
    frozen["prompt_id"] = frozen["prompt_id"].map(prompt_value)
    keys = ["dataset", "seed", "period", "n_per_label", "model"]
    wording_rows: list[dict[str, Any]] = []
    abstention_rows: list[dict[str, Any]] = []
    for key, group in frozen.groupby(keys, sort=True):
        by_prompt = {row["prompt_id"]: row for _, row in group.iterrows()}
        if "p1_minimal" in by_prompt and "p3_json_strict" in by_prompt:
            a = by_prompt["p1_minimal"]
            b = by_prompt["p3_json_strict"]
            wording_rows.append(
                {
                    **dict(zip(keys, key)),
                    "macro_f1_absolute_difference": abs(a["macro_f1"] - b["macro_f1"]),
                    "accuracy_absolute_difference": abs(a["accuracy"] - b["accuracy"]),
                    "invalid_rate_difference": a["invalid_rate"] - b["invalid_rate"],
                }
            )
        if "p5_abstain" in by_prompt:
            p5 = by_prompt["p5_abstain"]
            abstention_rows.append(
                {
                    **dict(zip(keys, key)),
                    "abstain_rate": p5["abstain_rate"],
                    "coverage": p5["coverage"],
                    "covered_accuracy": p5["covered_accuracy"],
                    "macro_f1_on_valid_rows": p5["macro_f1"],
                }
            )
    return pd.DataFrame(wording_rows), pd.DataFrame(abstention_rows)


def prediction_files(run_dir: Path) -> list[Path]:
    return sorted((run_dir / "predictions").rglob("*.csv"))


def load_all_predictions(run_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in prediction_files(run_dir):
        frame = pd.read_csv(path)
        frame["prediction_file"] = str(path.relative_to(run_dir))
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True, sort=False)
    result["prompt_id"] = result.get("prompt_id", "").map(prompt_value)
    return result


def error_analysis(predictions: pd.DataFrame, top_n: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if predictions.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    canonical = predictions[
        (predictions["family"] == "adapted")
        | predictions["prompt_id"].eq(CANONICAL_PROMPT)
    ].copy()
    canonical["status"] = canonical["status"].astype(str)
    canonical["is_error"] = (
        canonical["status"].eq("ok")
        & canonical["predicted_label"].notna()
        & canonical["true_label"].ne(canonical["predicted_label"])
    )
    group_cols = [
        "dataset",
        "task_type",
        "seed",
        "period",
        "n_per_label",
        "family",
        "model",
        "prompt_id",
    ]
    counts = (
        canonical.groupby(group_cols, dropna=False)
        .agg(
            rows=("text_sha256", "size"),
            valid_rows=("status", lambda values: int((values == "ok").sum())),
            error_rows=("is_error", "sum"),
            invalid_rows=("status", lambda values: int((values != "ok").sum())),
            abstain_rows=("abstain", lambda values: int(pd.Series(values).astype(bool).sum())),
        )
        .reset_index()
    )
    counts["error_percentage_valid"] = 100 * counts["error_rows"] / counts["valid_rows"].replace(0, np.nan)

    confusion_rows: list[dict[str, Any]] = []
    for keys, group in canonical[canonical["status"] == "ok"].groupby(group_cols, dropna=False):
        labels = sorted(set(group["true_label"].dropna()) | set(group["predicted_label"].dropna()))
        matrix = confusion_matrix(group["true_label"], group["predicted_label"], labels=labels)
        base = dict(zip(group_cols, keys))
        for true_index, true_label in enumerate(labels):
            for pred_index, pred_label in enumerate(labels):
                confusion_rows.append(
                    {
                        **base,
                        "true_label": true_label,
                        "predicted_label": pred_label,
                        "count": int(matrix[true_index, pred_index]),
                    }
                )

    errors = canonical[canonical["is_error"]].copy()
    errors["confidence"] = pd.to_numeric(errors["confidence"], errors="coerce")
    high_confidence = (
        errors.sort_values("confidence", ascending=False)
        .groupby(group_cols, dropna=False, group_keys=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return counts, pd.DataFrame(confusion_rows), high_confidence


def risk_coverage(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    canonical = predictions[
        ((predictions["family"] == "adapted") | predictions["prompt_id"].eq(CANONICAL_PROMPT))
        & predictions["status"].astype(str).eq("ok")
        & predictions["confidence"].notna()
    ].copy()
    group_cols = [
        "dataset",
        "seed",
        "period",
        "n_per_label",
        "family",
        "model",
        "confidence_source",
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in canonical.groupby(group_cols, sort=True):
        ordered = group.sort_values("confidence", ascending=False).reset_index(drop=True)
        correct = ordered["true_label"].eq(ordered["predicted_label"]).to_numpy()
        for target_coverage in [1.0, 0.9, 0.8, 0.7, 0.5]:
            keep = max(1, int(math.ceil(len(ordered) * target_coverage)))
            retained = correct[:keep]
            rows.append(
                {
                    **dict(zip(group_cols, keys)),
                    "target_coverage": target_coverage,
                    "actual_coverage": keep / len(ordered),
                    "retained_rows": keep,
                    "covered_risk": float(1.0 - retained.mean()),
                    "covered_accuracy": float(retained.mean()),
                    "minimum_retained_confidence": float(ordered.iloc[keep - 1]["confidence"]),
                }
            )
    return pd.DataFrame(rows)


def simple_tokens(text: str) -> list[str]:
    return re.findall(r"(?u)\b\w\w+\b", str(text).lower())


def js_divergence(train_texts: Sequence[str], target_texts: Sequence[str]) -> float:
    train_counts: dict[str, int] = {}
    target_counts: dict[str, int] = {}
    for text in train_texts:
        for token in simple_tokens(text):
            train_counts[token] = train_counts.get(token, 0) + 1
    for text in target_texts:
        for token in simple_tokens(text):
            target_counts[token] = target_counts.get(token, 0) + 1
    vocabulary = sorted(set(train_counts) | set(target_counts))
    if not vocabulary:
        return math.nan
    p = np.asarray([train_counts.get(token, 0) for token in vocabulary], dtype=float)
    q = np.asarray([target_counts.get(token, 0) for token in vocabulary], dtype=float)
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def top_shift_terms(
    train_texts: Sequence[str],
    target_texts: Sequence[str],
    top_n: int,
) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    vectorizer = CountVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=3,
        binary=True,
        max_features=50_000,
    )
    combined = list(train_texts) + list(target_texts)
    try:
        matrix = vectorizer.fit_transform(combined)
    except ValueError:
        return [], []
    train_matrix = matrix[: len(train_texts)]
    target_matrix = matrix[len(train_texts) :]
    train_df = np.asarray(train_matrix.mean(axis=0)).ravel()
    target_df = np.asarray(target_matrix.mean(axis=0)).ravel()
    difference = target_df - train_df
    terms = np.asarray(vectorizer.get_feature_names_out())
    emerging_indices = np.argsort(difference)[::-1][:top_n]
    declining_indices = np.argsort(difference)[:top_n]
    emerging = [(str(terms[i]), float(difference[i])) for i in emerging_indices if difference[i] > 0]
    declining = [(str(terms[i]), float(difference[i])) for i in declining_indices if difference[i] < 0]
    return emerging, declining


def first_example_containing(texts: Sequence[str], term: str) -> str | None:
    pattern = re.compile(rf"\b{re.escape(term)}\b", flags=re.IGNORECASE)
    for text in texts:
        if pattern.search(str(text)):
            return str(text)[:500]
    return None


def temporal_forensics(split_root: Path, top_n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = json.loads((split_root / "split_manifest.json").read_text(encoding="utf-8"))
    index = pd.read_csv(split_root / "split_index.csv")
    metric_rows: list[dict[str, Any]] = []
    term_rows: list[dict[str, Any]] = []
    for dataset, cfg in manifest["datasets"].items():
        for seed in manifest["seeds"]:
            train_row = index[
                (index["dataset"] == dataset)
                & (index["seed"].astype(int) == int(seed))
                & (index["role"] == "train")
                & index["n_per_label"].isna()
            ].iloc[0]
            train = pd.read_csv(split_root / train_row["path"])
            train_vocab = set(token for text in train["text"] for token in simple_tokens(text))
            train_lengths = train["text"].map(lambda value: len(simple_tokens(value)))
            for period in ["within", "short", "long"]:
                for size in cfg["evaluation_sizes_per_label"]:
                    row = index[
                        (index["dataset"] == dataset)
                        & (index["seed"].astype(int) == int(seed))
                        & (index["role"] == period)
                        & (index["n_per_label"].fillna(-1).astype(int) == int(size))
                    ].iloc[0]
                    target = pd.read_csv(split_root / row["path"])
                    target_tokens = [token for text in target["text"] for token in simple_tokens(text)]
                    target_vocab = set(target_tokens)
                    novelty = len(target_vocab - train_vocab) / max(len(target_vocab), 1)
                    target_lengths = target["text"].map(lambda value: len(simple_tokens(value)))
                    metric_rows.append(
                        {
                            "dataset": dataset,
                            "seed": int(seed),
                            "period": period,
                            "n_per_label": int(size),
                            "lexical_novelty_rate": novelty,
                            "vocabulary_overlap_jaccard": len(train_vocab & target_vocab)
                            / max(len(train_vocab | target_vocab), 1),
                            "token_distribution_js_divergence": js_divergence(
                                train["text"], target["text"]
                            ),
                            "mean_document_length_train": float(train_lengths.mean()),
                            "mean_document_length_target": float(target_lengths.mean()),
                            "mean_length_change": float(target_lengths.mean() - train_lengths.mean()),
                        }
                    )
                    emerging, declining = top_shift_terms(train["text"], target["text"], top_n)
                    for direction, values in [("emerging", emerging), ("declining", declining)]:
                        for rank, (term, change) in enumerate(values, 1):
                            term_rows.append(
                                {
                                    "dataset": dataset,
                                    "seed": int(seed),
                                    "period": period,
                                    "n_per_label": int(size),
                                    "direction": direction,
                                    "rank": rank,
                                    "term": term,
                                    "document_frequency_change": change,
                                    "illustrative_target_example": (
                                        first_example_containing(target["text"], term)
                                        if direction == "emerging"
                                        else None
                                    ),
                                }
                            )
    return pd.DataFrame(metric_rows), pd.DataFrame(term_rows)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.run_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.run_dir / "metrics_per_condition.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
    metrics = pd.read_csv(metrics_path)

    aggregate = aggregate_seed_metrics(metrics)
    paired = paired_model_tests(metrics)
    deltas = temporal_deltas(metrics)
    decisions = decision_test(metrics, args.failure_threshold)
    wording, abstention = prompt_sensitivity(metrics)
    predictions = load_all_predictions(args.run_dir)
    errors, confusion, high_confidence_errors = error_analysis(
        predictions, args.top_error_examples
    )
    risk = risk_coverage(predictions)
    drift, shift_terms = temporal_forensics(args.split_root, args.top_shift_terms)

    outputs = {
        "metrics_aggregated_with_ci.csv": aggregate,
        "paired_model_tests_macro_f1.csv": paired,
        "temporal_deltas_per_seed.csv": deltas,
        "model_selection_decision_test.csv": decisions,
        "prompt_wording_sensitivity.csv": wording,
        "abstention_policy_results.csv": abstention,
        "error_counts_by_model_period.csv": errors,
        "confusion_matrices.csv": confusion,
        "high_confidence_errors.csv": high_confidence_errors,
        "risk_coverage_fixed_levels.csv": risk,
        "temporal_forensics_metrics.csv": drift,
        "temporal_shift_terms_and_examples.csv": shift_terms,
    }
    for name, frame in outputs.items():
        atomic_write_csv(output_dir / name, frame)

    atomic_write_json(
        output_dir / "analysis_manifest.json",
        {
            "canonical_frozen_prompt": CANONICAL_PROMPT,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "paired_test": "exact paired sign-flip permutation test across seeds",
            "multiple_testing": "Holm adjustment within each dataset-period-sample-size setting",
            "decision_failure_threshold": args.failure_threshold,
            "confidence_interpretation": {
                "adapted": "predicted class probability",
                "frozen": "LLM self-reported confidence",
                "cross_family_calibration_ranking_permitted": False,
                "risk_coverage_comparison": "operational comparison with confidence source retained",
            },
            "prompt_analysis": {
                "wording_sensitivity": "p1_minimal versus p3_json_strict; same JSON schema and no abstention",
                "abstention_policy": "p5_abstain analyzed separately",
            },
            "generated_files": list(outputs),
        },
    )
    print(f"[ANALYSIS] Wrote reviewer-facing outputs to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
