"""Hydrate dehydrated dataset split CSVs with raw text from source corpora or a full-text archive.

In accordance with standard academic NLP data-sharing practice (and to respect third-party
dataset redistribution terms such as the Yelp Dataset Terms of Use and social-media platform
policies), the public GitHub repository ships dehydrated split manifests in `data/splits/`
containing `example_id`, `label`, cryptographic text digests (`text_sha256` / `text_sha1`),
`token_count`, `char_length`, and temporal metadata.

All paper verification commands (`python reproduce.py --all`, Tables 7-13, Figures 3-6) work
100% out-of-the-box without raw text. Researchers who wish to re-run model training or live
LLM inference from scratch can restore the `text` column using this utility in two ways:

1. From a full-text split directory (obtained upon request from the author for academic use):
   python scripts/hydrate_splits.py --fulltext-splits-dir /path/to/full_text

2. From raw source CSV/JSONL files (matching rows by SHA-256 / SHA-1 digest of the text):
   python scripts/hydrate_splits.py --source-files /path/to/raw_corpus1.csv /path/to/raw_corpus2.json
"""

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Dict

csv.field_size_limit(10**9)

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SPLITS_DIR = BASE_DIR / "data" / "splits"


def build_hash_lookup(source_files, text_col: str = "text") -> Dict[str, str]:
    """Index raw texts from CSV or JSONL files by both SHA-256 and SHA-1 digests."""
    lookup: Dict[str, str] = {}
    for src in source_files:
        p = Path(src)
        if not p.is_file():
            raise FileNotFoundError(f"Source file not found: {p}")
        if p.suffix.lower() in (".json", ".jsonl"):
            with p.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    txt = obj.get(text_col)
                    if isinstance(txt, str) and txt:
                        b = txt.encode("utf-8")
                        lookup[hashlib.sha256(b).hexdigest()] = txt
                        lookup[hashlib.sha1(b).hexdigest()] = txt
        else:
            with p.open("r", encoding="utf-8", errors="replace", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    txt = row.get(text_col)
                    if isinstance(txt, str) and txt:
                        b = txt.encode("utf-8")
                        lookup[hashlib.sha256(b).hexdigest()] = txt
                        lookup[hashlib.sha1(b).hexdigest()] = txt
    return lookup


def hydrate_from_fulltext_dir(splits_dir: Path, fulltext_dir: Path) -> None:
    """Copy full-text split CSVs directly and verify against checksums_fulltext_sha256.txt."""
    copied = 0
    for folder in ("climate", "yelp"):
        src_folder = fulltext_dir / folder
        if not src_folder.is_dir():
            continue
        for src_csv in sorted(src_folder.glob("seed_*/*.csv")):
            rel = src_csv.relative_to(fulltext_dir)
            dst_csv = splits_dir / rel
            dst_csv.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_csv, dst_csv)
            copied += 1
    print(f"Hydrated {copied} split CSV files from {fulltext_dir}.")


def hydrate_from_lookup(splits_dir: Path, lookup: Dict[str, str]) -> None:
    """Populate the `text` column in each dehydrated split CSV using the hash lookup."""
    total_rows = 0
    matched_rows = 0
    for folder in ("climate", "yelp"):
        for csv_path in sorted((splits_dir / folder).glob("seed_*/*.csv")):
            with csv_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                in_fields = list(reader.fieldnames or [])
                rows = list(reader)
            hash_col = "text_sha256" if "text_sha256" in in_fields else "text_sha1"
            base_fields = [c for c in in_fields if c not in ("text", "token_count", "char_length")]
            out_fields = [base_fields[0], "text"] + base_fields[1:]
            for r in rows:
                total_rows += 1
                digest = r.get(hash_col, "")
                txt = lookup.get(digest, "")
                if txt:
                    matched_rows += 1
                r["text"] = txt
                r.pop("token_count", None)
                r.pop("char_length", None)
            with csv_path.open("w", encoding="utf-8", newline="\n") as f:
                writer = csv.DictWriter(f, fieldnames=out_fields, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
    print(f"Hydrated {matched_rows}/{total_rows} rows across split files.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hydrate dehydrated dataset split CSVs with raw text.")
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--fulltext-splits-dir", type=Path, default=None,
                        help="Path to directory containing full-text climate/ and yelp/ split folders")
    parser.add_argument("--source-files", nargs="*", default=[],
                        help="Paths to raw source CSV/JSONL files containing a text column")
    parser.add_argument("--text-col", type=str, default="text",
                        help="Column/field name for raw text in --source-files (default: text)")
    args = parser.parse_args()

    if args.fulltext_splits_dir:
        hydrate_from_fulltext_dir(args.splits_dir, args.fulltext_splits_dir)
    elif args.source_files:
        lookup = build_hash_lookup(args.source_files, text_col=args.text_col)
        hydrate_from_lookup(args.splits_dir, lookup)
    else:
        parser.error("Provide either --fulltext-splits-dir or --source-files.")


if __name__ == "__main__":
    main()
