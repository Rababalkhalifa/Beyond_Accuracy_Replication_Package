"""Archive loading and prediction extraction utilities.

Encapsulates reading zipped prediction cohorts and cached metrics for both
Climate stance classification and Yelp sentiment classification.
"""

import csv
import io
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

DEFAULT_ARCHIVES = {
    "climate": "completed_prediction_audit.zip",
    "yelp_frozen": "original_yelp_frozen_predictions.zip",
    "yelp_aligned": "yelp_prediction_alignment.zip",
    "yelp_metrics": "code_and_remaining_results.zip",
    "yelp_frozen_metrics": "yelp_results_audit.zip"
}

MODEL_SLUGS = {
    "distilbert": "distilbert",
    "tfidf_lr": "tfidf_lr",
    "phi3.5:latest": "phi3.5_latest",
    "qwen2.5:3b-instruct": "qwen2.5_3b-instruct"
}


class ArchiveManager:
    """Manages zipfile handles to verified prediction and metric archives."""

    def __init__(self, archives_dir: Union[str, Path], file_map: Optional[Dict[str, str]] = None):
        self.archives_dir = Path(archives_dir)
        self.file_map = file_map or DEFAULT_ARCHIVES
        self._handles: Dict[str, zipfile.ZipFile] = {}

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def open(self):
        """Open all required zip archive handles."""
        for key, fname in self.file_map.items():
            path = self.archives_dir / fname
            if not path.is_file():
                # Try finding recursively if not in root of archives_dir
                matches = list(self.archives_dir.rglob(fname))
                if not matches:
                    raise FileNotFoundError(f"Missing required archive: {fname} under {self.archives_dir}")
                path = matches[0]
            self._handles[key] = zipfile.ZipFile(path, "r")

    def close(self):
        """Close all open archive handles."""
        for z in self._handles.values():
            try:
                z.close()
            except Exception:
                pass
        self._handles.clear()

    def get_csv(self, archive_key: str, suffix: str) -> List[Dict[str, str]]:
        """Read a single CSV file by suffix from a designated zip archive."""
        if archive_key not in self._handles:
            raise KeyError(f"Archive '{archive_key}' is not opened.")
        z = self._handles[archive_key]
        matches = [name for name in z.namelist() if name.replace("\\", "/").endswith(suffix)]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly 1 file ending with '{suffix}' in '{archive_key}', found {len(matches)}.")
        with z.open(matches[0]) as f:
            return list(csv.DictReader(io.TextIOWrapper(f, encoding="utf-8-sig", newline="")))

    def load_climate_predictions(self, model: str, seed: int, period: str) -> List[Dict[str, str]]:
        """Load Climate stance predictions for a given model, seed, and temporal period."""
        slug = MODEL_SLUGS[model]
        if model in ("distilbert", "tfidf_lr"):
            path = f"predictions/adapted/{slug}/climate/seed_{seed}/{period}_n500_per_label.csv"
        else:
            path = f"predictions/frozen/{slug}/p3_json_strict/climate/seed_{seed}/{period}_n500_per_label.csv"
        return self.get_csv("climate", path)

    def load_yelp_predictions(self, model: str, n_per_label: int, seed: int, period: str) -> List[Dict[str, str]]:
        """Load Yelp sentiment predictions for a given model, sample size, seed, and period."""
        if model in ("distilbert", "tfidf_lr"):
            folder = "later_distilbert" if model == "distilbert" else "aligned_tfidf"
            path = f"{folder}/seed{seed}/n{n_per_label}_{period}.csv"
            return self.get_csv("yelp_aligned", path)
        else:
            slug = MODEL_SLUGS[model]
            path = f"predictions/frozen/{slug}/seed{seed}/n{n_per_label}_{period}_p3_json_strict.csv"
            return self.get_csv("yelp_frozen", path)

    def load_climate_metrics_index(self) -> Dict[Tuple[str, str, int, str], Dict[str, str]]:
        """Load cached Climate metrics index: (setting, model, seed, period) -> record."""
        index = {}
        rows = self.get_csv("climate", "metrics_per_condition.csv")
        for r in rows:
            if r.get("dataset") != "climate":
                continue
            m = r["model"]
            if m not in MODEL_SLUGS:
                continue
            period = r["period"]
            seed = int(r["seed"])
            exp_prompt = "" if m in ("distilbert", "tfidf_lr") else "p3_json_strict"
            if r.get("prompt_id", "") != exp_prompt or int(r.get("n_per_label", 0)) != 500:
                continue
            key = ("Climate n500", m, seed, period)
            index[key] = r
        return index

    def load_yelp_metrics_indices(self) -> Dict[Tuple[str, str, int, str], Dict[str, str]]:
        """Load cached Yelp metrics indices for both supervised and frozen models."""
        index = {}
        # Supervised from later run
        sup_rows = self.get_csv("yelp_metrics", "later_run/metrics_per_seed.csv")
        for r in sup_rows:
            m = r["model"]
            if m not in ("distilbert", "tfidf_lr"):
                continue
            split = r["split"]
            n = int(r["eval_n_per_label"])
            seed = int(r["seed"])
            if r.get("prompt", "") != "p3_json_strict" and r.get("prompt", "") != "":
                pass
            key = (f"Yelp n{n}", m, seed, split)
            index[key] = r

        # Frozen from audit
        frozen_rows = self.get_csv("yelp_frozen_metrics", "metrics_frozen_partial.csv")
        for r in frozen_rows:
            m = r["model"]
            if m not in ("phi3.5:latest", "qwen2.5:3b-instruct"):
                continue
            if r.get("prompt") != "p3_json_strict":
                continue
            split = r["split"]
            n = int(r["eval_n_per_label"])
            seed = int(r["seed"])
            key = (f"Yelp n{n}", m, seed, split)
            index[key] = r

        return index
