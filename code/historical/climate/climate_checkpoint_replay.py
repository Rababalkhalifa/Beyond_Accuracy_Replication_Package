"""Read-only replay of saved Climate checkpoints against archived predictions.

This does not train or select models. It verifies whether one saved checkpoint
per seed reproduces the archived within/short/long predictions for that seed.
It cannot establish the original execution history or checkpoint timestamps.
"""

import argparse
import csv
import io
import json
import sys
import zipfile
from pathlib import Path


PERIODS = ("within", "short", "long")
LABELS = ("BELIEVER", "DENIER")
MAX_LENGTH = 192  # run_paper_models_paper_scope.py


def archived_rows(archive, model, seed, period):
    suffix = (
        f"predictions/adapted/{model}/climate/seed_{seed}/"
        f"{period}_n500_per_label.csv"
    )
    matches = [name for name in archive.namelist()
               if name.replace("\\", "/").endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"Expected one archive entry ending in {suffix}; found {len(matches)}")
    with archive.open(matches[0]) as raw:
        rows = list(csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")))
    if len(rows) != 1000:
        raise ValueError(f"{suffix}: expected 1000 rows, found {len(rows)}")
    if len({row["example_id"] for row in rows}) != len(rows):
        raise ValueError(f"{suffix}: duplicate example IDs")
    for row in rows:
        if (row["dataset"] != "climate" or row["model"] != model
                or int(row["seed"]) != seed or row["period"] != period
                or int(row["n_per_label"]) != 500):
            raise ValueError(f"{suffix}: inconsistent row metadata")
    return rows


def load_predictor(run_dir, model, seed, device):
    root = run_dir / "models" / model / "climate" / f"seed_{seed}"
    if model == "tfidf_lr":
        import joblib
        if not (root / "selection.json").is_file():
            raise FileNotFoundError(root / "selection.json")
        pipeline = joblib.load(root / "model.joblib")
        classes = [str(label) for label in pipeline.classes_]
        if set(classes) != set(LABELS):
            raise ValueError(f"Unexpected TF-IDF labels: {classes}")

        def predict(texts):
            predictions = [str(label) for label in pipeline.predict(texts)]
            matrix = pipeline.predict_proba(texts)
            probabilities = [
                {label: float(values[classes.index(label)]) for label in LABELS}
                for values in matrix
            ]
            return predictions, probabilities

        return predict

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if not (root / "training_metadata.json").is_file():
        raise FileNotFoundError(root / "training_metadata.json")
    model_dir = root / "best_model"
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    network = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir), local_files_only=True
    )
    labels = [network.config.id2label[i] for i in range(2)]
    if set(labels) != set(LABELS):
        raise ValueError(f"Unexpected DistilBERT labels: {labels}")
    target = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if target == "auto":
        target = "cpu"
    network.to(target).eval()

    def predict(texts):
        predictions, probabilities = [], []
        with torch.inference_mode():
            for start in range(0, len(texts), 16):
                tokens = tokenizer(
                    texts[start:start + 16], truncation=True,
                    max_length=MAX_LENGTH, padding=True, return_tensors="pt"
                )
                logits = network(**{key: value.to(target) for key, value in tokens.items()}).logits
                matrix = torch.softmax(logits, dim=1).cpu().numpy()
                for values in matrix:
                    by_label = {label: float(values[i]) for i, label in enumerate(labels)}
                    predictions.append(max(by_label, key=by_label.get))
                    probabilities.append(by_label)
        return predictions, probabilities

    return predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="completed_prediction_audit.zip")
    parser.add_argument("run_dir", type=Path, help="Directory containing models/")
    parser.add_argument("--model", choices=("tfidf_lr", "distilbert"), default="tfidf_lr")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--tolerance", type=float, default=0.0001,
                        help="Absolute probability tolerance (default 0.0001)")
    parser.add_argument("--csv", type=Path, help="Optional audit summary CSV")
    args = parser.parse_args()
    if not args.archive.is_file() or not args.run_dir.is_dir():
        parser.error("Archive or run directory does not exist")

    summaries = []
    with zipfile.ZipFile(args.archive) as archive:
        for seed in args.seeds:
            if seed not in range(42, 47):
                parser.error("Use archived seeds 42 through 46")
            predictor = load_predictor(args.run_dir, args.model, seed, args.device)
            for period in PERIODS:
                rows = archived_rows(archive, args.model, seed, period)
                predicted, probabilities = predictor([row["text"] for row in rows])
                bad_labels = sum(str(predicted[i]) != row["predicted_label"]
                                 for i, row in enumerate(rows))
                max_diff = max(
                    abs(probabilities[i][label] - float(row[f"prob_{label.lower()}"]))
                    for i, row in enumerate(rows) for label in LABELS
                )
                passed = bad_labels == 0 and max_diff <= args.tolerance
                item = dict(model=args.model, seed=seed, period=period,
                            rows=len(rows), label_mismatches=bad_labels,
                            max_probability_difference=max_diff,
                            within_tolerance=passed)
                summaries.append(item)
                print(f"{args.model:10} seed {seed} {period:6}: "
                      f"label mismatches {bad_labels}/{len(rows)}; "
                      f"max probability difference {max_diff:.8g}; "
                      f"{'PASS' if passed else 'REVIEW'}", flush=True)

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
        print(f"Saved audit: {args.csv}")
    print("Replay compares saved model outputs with archived rows; it does not "
          "prove which checkpoint was used during the original run.")
    return 0 if all(item["within_tolerance"] for item in summaries) else 1


if __name__ == "__main__":
    sys.exit(main())
