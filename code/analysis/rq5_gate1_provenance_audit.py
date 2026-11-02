from pathlib import Path
import pandas as pd
import json

ROOT = Path("/mnt/c/Research/Beyond_Accuracy_Reproduction")

SEEDS = [42, 43, 44, 45, 46]

# ---------------------------------------------------------
# Authoritative split roots already established in our audit
# ---------------------------------------------------------

CLIMATE_CANDIDATES = [
    Path("/mnt/c/Research/Beyond_Accuracy_Replication_Package/data/splits/climate"),
    ROOT / "temporal_reliability_rerun_v8/shared_splits_paper_v6_joint_quota/climate",
]

YELP_CANDIDATES = [
    ROOT / "out/sentiment_adapted_on_frozen_20260722/recovered_shared_splits",
    ROOT / "provenance_audit_20261014_001156/aligned_yelp_splits",
]

def first_existing(candidates):
    for p in candidates:
        if p.exists():
            return p
    return None

CLIMATE_ROOT = first_existing(CLIMATE_CANDIDATES)
YELP_ROOT = first_existing(YELP_CANDIDATES)

print("=" * 70)
print("RQ5 GATE 1 — PROVENANCE AUDIT")
print("=" * 70)

print("\nAuthoritative roots")
print("Climate:", CLIMATE_ROOT)
print("Yelp   :", YELP_ROOT)

if CLIMATE_ROOT is None:
    raise FileNotFoundError("No Climate authoritative split root found.")

if YELP_ROOT is None:
    raise FileNotFoundError("No Yelp authoritative split root found.")


# ---------------------------------------------------------
# Helper: inspect one CSV without making assumptions about
# exact column names.
# ---------------------------------------------------------

TEXT_CANDIDATES = [
    "text",
    "review",
    "review_text",
    "content",
    "tweet",
]

LABEL_CANDIDATES = [
    "label",
    "gold_label",
    "target",
    "stance",
    "sentiment",
]


def detect_column(columns, candidates):
    lookup = {str(c).lower(): c for c in columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


def inspect_csv(path, setting, seed, split):
    df = pd.read_csv(path)

    text_col = detect_column(df.columns, TEXT_CANDIDATES)
    label_col = detect_column(df.columns, LABEL_CANDIDATES)

    labels = None

    if label_col is not None:
        labels = sorted(
            df[label_col]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
            .tolist()
        )

    return {
        "setting": setting,
        "seed": seed,
        "split": split,
        "path": str(path),
        "exists": True,
        "rows": len(df),
        "text_column": text_col,
        "label_column": label_col,
        "labels": json.dumps(labels, ensure_ascii=False),
        "duplicate_rows": int(df.duplicated().sum()),
    }


audit_rows = []


# ---------------------------------------------------------
# Climate
# Expected filenames from the authoritative replication
# package.
# ---------------------------------------------------------

climate_files = {
    "within": "within_2010_n500_per_label_total1000.csv",
    "short": "short_2013_n500_per_label_total1000.csv",
    "long": "long_2018_2019_n500_per_label_total1000.csv",
}

for seed in SEEDS:

    seed_dir = CLIMATE_ROOT / f"seed_{seed}"

    # Some mirrors may use seed42 instead of seed_42.
    if not seed_dir.exists():
        alt = CLIMATE_ROOT / f"seed{seed}"
        if alt.exists():
            seed_dir = alt

    for split, filename in climate_files.items():

        path = seed_dir / filename

        if not path.exists():
            audit_rows.append({
                "setting": "Climate n500",
                "seed": seed,
                "split": split,
                "path": str(path),
                "exists": False,
            })
            continue

        audit_rows.append(
            inspect_csv(
                path,
                setting="Climate n500",
                seed=seed,
                split=split,
            )
        )


# ---------------------------------------------------------
# Yelp
# Historical frozen-aligned evaluation splits.
# ---------------------------------------------------------

for n_per_label in [500, 1000]:

    setting = f"Yelp n{n_per_label}"

    for seed in SEEDS:

        seed_dir = (
            YELP_ROOT
            / f"seed{seed}"
            / f"n{n_per_label}_per_label"
        )

        for split in ["within", "short", "long"]:

            path = seed_dir / f"{split}.csv"

            if not path.exists():
                audit_rows.append({
                    "setting": setting,
                    "seed": seed,
                    "split": split,
                    "path": str(path),
                    "exists": False,
                })
                continue

            audit_rows.append(
                inspect_csv(
                    path,
                    setting=setting,
                    seed=seed,
                    split=split,
                )
            )


audit = pd.DataFrame(audit_rows)

out_dir = ROOT / "RQ5_reaudit"
out_dir.mkdir(exist_ok=True)

audit.to_csv(
    out_dir / "gate1_split_provenance.csv",
    index=False,
)


# ---------------------------------------------------------
# Human-readable checks
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("SPLIT AUDIT")
print("=" * 70)

show_cols = [
    "setting",
    "seed",
    "split",
    "exists",
    "rows",
    "label_column",
    "labels",
]

print(
    audit.reindex(columns=show_cols)
    .to_string(index=False)
)


print("\n" + "=" * 70)
print("MISSING FILES")
print("=" * 70)

missing = audit[audit["exists"] != True]

if len(missing):
    print(
        missing[
            ["setting", "seed", "split", "path"]
        ].to_string(index=False)
    )
else:
    print("None")


print("\n" + "=" * 70)
print("LABEL SETS BY DATASET")
print("=" * 70)

existing = audit[audit["exists"] == True].copy()

for setting in [
    "Climate n500",
    "Yelp n500",
    "Yelp n1000",
]:

    subset = existing[existing["setting"] == setting]

    label_sets = sorted(
        subset["labels"]
        .dropna()
        .unique()
        .tolist()
    )

    print(f"{setting}: {label_sets}")


# ---------------------------------------------------------
# Assertions — intentionally conservative.
# Stop immediately if the task-native label spaces are not
# what the manuscript now says they are.
# ---------------------------------------------------------

climate_labels = set()

for value in existing.loc[
    existing["setting"] == "Climate n500",
    "labels",
].dropna():

    climate_labels.update(json.loads(value))


yelp_labels = set()

for value in existing.loc[
    existing["setting"].str.startswith("Yelp"),
    "labels",
].dropna():

    yelp_labels.update(json.loads(value))


print("\nDetected Climate labels:", sorted(climate_labels))
print("Detected Yelp labels   :", sorted(yelp_labels))


assert climate_labels == {"BELIEVER", "DENIER"}, (
    "Climate label audit failed: "
    f"{sorted(climate_labels)}"
)

assert yelp_labels == {"NEG", "POS"}, (
    "Yelp label audit failed: "
    f"{sorted(yelp_labels)}"
)


# Expected evaluation sizes.
expected_rows = {
    "Climate n500": 1000,
    "Yelp n500": 1000,
    "Yelp n1000": 2000,
}

bad_sizes = existing[
    existing.apply(
        lambda r:
            r["rows"] != expected_rows[r["setting"]],
        axis=1,
    )
]

print("\n" + "=" * 70)
print("ROW-COUNT CHECK")
print("=" * 70)

if len(bad_sizes):
    print(bad_sizes[
        ["setting", "seed", "split", "rows", "path"]
    ].to_string(index=False))

    raise RuntimeError(
        "Unexpected evaluation split sizes detected."
    )

print("All evaluation split sizes are correct.")

print("\nPASS: RQ5 Gate 1 provenance/label audit.")
print(
    "Saved:",
    out_dir / "gate1_split_provenance.csv",
)
