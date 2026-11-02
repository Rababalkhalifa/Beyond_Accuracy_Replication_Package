#!/usr/bin/env python3
"""Check data paths, Python packages, GPU visibility, and Ollama models before rerun."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import requests


REQUIRED_PACKAGES = ["numpy", "pandas", "sklearn", "scipy", "requests", "joblib"]
DISTILBERT_PACKAGES = ["torch", "transformers", "datasets", "accelerate"]
OLLAMA_MODELS = ["qwen2.5:3b-instruct", "phi3.5:latest"]
ROLES = ["historical_train", "within", "short", "long"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("datasets.json"))
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    return parser.parse_args()


def resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def check_import(name: str) -> tuple[bool, str]:
    try:
        module = importlib.import_module(name)
        return True, str(getattr(module, "__version__", "installed"))
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def main() -> None:
    args = parse_args()
    failures: list[str] = []
    warnings: list[str] = []

    print("[PREFLIGHT] Python packages")
    for package in REQUIRED_PACKAGES + DISTILBERT_PACKAGES:
        ok, detail = check_import(package)
        print(f"  {'OK' if ok else 'MISSING'} {package}: {detail}")
        if not ok:
            failures.append(f"Missing package: {package}")

    torch_ok, _ = check_import("torch")
    if torch_ok:
        import torch

        print(f"[PREFLIGHT] CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[PREFLIGHT] GPU: {torch.cuda.get_device_name(0)}")
        else:
            warnings.append("CUDA is unavailable; DistilBERT will be very slow on CPU.")

    if not args.config.exists():
        failures.append(f"Missing configuration: {args.config}")
    else:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        print("[PREFLIGHT] Dataset files")
        for dataset, cfg in config.items():
            for role in ROLES:
                configured = cfg.get("paths", {}).get(role)
                if not configured:
                    failures.append(f"{dataset}: missing configured path for {role}")
                    continue
                path = resolve(args.config.parent, str(configured))
                if not path.exists():
                    print(f"  MISSING {dataset}/{role}: {path}")
                    failures.append(f"Missing source CSV: {path}")
                    continue
                try:
                    columns = list(pd.read_csv(path, nrows=2).columns)
                except Exception as exc:  # noqa: BLE001
                    failures.append(f"Cannot read {path}: {exc}")
                    continue
                year_col = cfg.get("year_col")
                if not year_col:
                    failures.append(f"{dataset}: missing year_col")
                    year_col = "year"
                required = [
                    cfg.get("text_col", "text"),
                    cfg.get("label_col", "label"),
                    year_col,
                ]
                missing_columns = [column for column in required if column not in columns]
                if missing_columns:
                    failures.append(f"{path} missing columns {missing_columns}")
                    continue

                configured_years = {
                    str(int(float(value))) if str(value).replace('.', '', 1).isdigit() else str(value)
                    for value in cfg.get("years", {}).get(role, [])
                }
                if not configured_years:
                    failures.append(f"{dataset}: missing configured years for {role}")
                    continue
                observed_years: set[str] = set()
                missing_year_values = 0
                try:
                    for chunk in pd.read_csv(path, usecols=[year_col], chunksize=100_000):
                        values = chunk[year_col]
                        missing_year_values += int(values.isna().sum())
                        for value in values.dropna().astype(str).str.strip().unique():
                            try:
                                number = float(value)
                                observed_years.add(str(int(number)) if number.is_integer() else value)
                            except ValueError:
                                observed_years.add(value)
                except Exception as exc:  # noqa: BLE001
                    failures.append(f"Cannot validate years in {path}: {exc}")
                    continue
                if missing_year_values:
                    failures.append(
                        f"{dataset}/{role}: {missing_year_values} missing year values"
                    )
                outside = sorted(observed_years - configured_years)
                absent = sorted(configured_years - observed_years)
                if outside or absent:
                    failures.append(
                        f"{dataset}/{role}: configured years={sorted(configured_years)}, "
                        f"observed={sorted(observed_years)}, outside={outside}, absent={absent}"
                    )
                else:
                    print(
                        f"  OK {dataset}/{role}: {path} years={sorted(observed_years)}"
                    )

    print("[PREFLIGHT] Ollama")
    try:
        response = requests.get(f"{args.ollama_host.rstrip('/')}/api/tags", timeout=15)
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        available = {
            str(item.get("name") or item.get("model"))
            for item in payload.get("models", [])
        }
        for model in OLLAMA_MODELS:
            if model in available:
                print(f"  OK {model}")
            else:
                print(f"  MISSING {model}")
                failures.append(f"Missing Ollama model: {model}")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"Cannot connect to Ollama: {exc}")

    for warning in warnings:
        print(f"[PREFLIGHT][WARNING] {warning}")
    if failures:
        print("\n[PREFLIGHT] FAIL")
        for failure in failures:
            print(f"  - {failure}")
        sys.exit(1)
    print("\n[PREFLIGHT] PASS")


if __name__ == "__main__":
    main()
