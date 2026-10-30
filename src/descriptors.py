"""Language descriptors calculation for sampled dataset splits (Table 7).

Computes descriptive language statistics under lowercase word tokenization (\\b\\w+\\b):
- Mean tokens per document
- Median tokens per document
- Total vocabulary size
- Type-token ratio (TTR = vocab / total_tokens)
- Top-100 token mass (fraction of total token mass accounted for by the 100 most frequent tokens)
"""

import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union
import numpy as np
import pandas as pd

TOKEN_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)


def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase word tokens."""
    return TOKEN_RE.findall(str(text).lower())


def compute_split_statistics(df: pd.DataFrame) -> Dict[str, float]:
    """Compute linguistic descriptor statistics for a single dataset split dataframe."""
    lengths = []
    counts = Counter()

    for text in df["text"].fillna(""):
        tokens = tokenize(text)
        lengths.append(len(tokens))
        counts.update(tokens)

    total_tokens = sum(lengths)
    vocab_size = len(counts)
    ttr = (vocab_size / total_tokens) if total_tokens > 0 else 0.0
    top100_mass = (sum(count for _, count in counts.most_common(100)) / total_tokens) if total_tokens > 0 else 0.0

    return {
        "n_docs": float(len(lengths)),
        "mean_len": float(np.mean(lengths)),
        "med_len": float(np.median(lengths)),
        "vocab_size": float(vocab_size),
        "ttr": float(ttr),
        "top100_mass": float(top100_mass),
        "total_tokens": float(total_tokens)
    }


def compute_all_descriptors(
    splits_dir: Union[str, Path],
    seeds: Sequence[int] = (42, 43, 44, 45, 46)
) -> List[Dict[str, any]]:
    """Compute Table 7 descriptive language statistics across all dataset settings and splits."""
    splits_path = Path(splits_dir)

    settings = [
        ("Climate n500", "climate", 500),
        ("Yelp n500", "yelp", 500),
        ("Yelp n1000", "yelp", 1000)
    ]

    splits_config = [
        ("train", "train_7000.csv"),
        ("within", "within_2010_n500_per_label_total1000.csv"),
        ("short", "short_2013_n500_per_label_total1000.csv"),
        ("long", "long_2018_2019_n500_per_label_total1000.csv")
    ]

    results = []

    for setting_name, folder, n in settings:
        for split_name in ("train", "within", "short", "long"):
            per_seed_stats = []
            for seed in seeds:
                if folder == "climate":
                    if split_name == "train":
                        fn = "train_7000.csv"
                    elif split_name == "within":
                        fn = "within_2010_n500_per_label_total1000.csv"
                    elif split_name == "short":
                        fn = "short_2013_n500_per_label_total1000.csv"
                    else:
                        fn = "long_2018_2019_n500_per_label_total1000.csv"
                else:  # yelp
                    if split_name == "train":
                        fn = "train_7000.csv"
                    elif split_name == "within":
                        fn = f"within_2011_2012_n{n}_per_label_total{2*n}.csv"
                    elif split_name == "short":
                        fn = f"short_2015_n{n}_per_label_total{2*n}.csv"
                    else:
                        fn = f"long_2019_2020_n{n}_per_label_total{2*n}.csv"

                file_path = splits_path / folder / f"seed_{seed}" / fn
                if not file_path.exists():
                    raise FileNotFoundError(f"Missing expected split file: {file_path}")

                df = pd.read_csv(file_path)
                per_seed_stats.append(compute_split_statistics(df))

            # Aggregate across 5 seeds: mean and sample standard deviation (ddof=1)
            mean_len_vals = [s["mean_len"] for s in per_seed_stats]
            med_len_vals = [s["med_len"] for s in per_seed_stats]
            vocab_vals = [s["vocab_size"] for s in per_seed_stats]
            ttr_vals = [s["ttr"] for s in per_seed_stats]
            top100_vals = [s["top100_mass"] for s in per_seed_stats]

            row = {
                "dataset_setting": setting_name,
                "split": split_name,
                "mean_len_mean": float(np.mean(mean_len_vals)),
                "mean_len_sd": float(np.std(mean_len_vals, ddof=1)),
                "med_len_mean": float(np.mean(med_len_vals)),
                "med_len_sd": float(np.std(med_len_vals, ddof=1)),
                "vocab_size_mean": float(np.mean(vocab_vals)),
                "vocab_size_sd": float(np.std(vocab_vals, ddof=1)),
                "ttr_mean": float(np.mean(ttr_vals)),
                "ttr_sd": float(np.std(ttr_vals, ddof=1)),
                "top100_mass_mean": float(np.mean(top100_vals)),
                "top100_mass_sd": float(np.std(top100_vals, ddof=1)),
            }
            results.append(row)

    return results
