import os
import csv
import zipfile
import hashlib
from pathlib import Path

# CHANGE THESE FOLDERS if needed
SEARCH_ROOTS = [
    r"/mnt/c/Research",
    r"/mnt/e/2026_Desktop",
]

KEYWORDS = [
    "qwen",
    "seed42",
    "n500",
]

def normalize_path(p):
    return str(p).replace("\\", "/").lower()

def looks_relevant(path):
    s = normalize_path(path)

    return (
        "qwen" in s
        and ("seed42" in s or "seed_42" in s or "seed-42" in s)
        and (
            "n500" in s
            or "500_per_label" in s
            or "500_per_class" in s
        )
        and (
            "p1" in s
            or "minimal" in s
        )
        and (
            "within" in s
            or "reference" in s
        )
    )

def find_col(columns, candidates):
    lower = {c.lower(): c for c in columns}

    for x in candidates:
        if x in lower:
            return lower[x]

    return None

def text_hash(text):
    text = str(text).strip()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def get_identity(row):
    columns = row.keys()

    text_col = find_col(
        columns,
        [
            "text",
            "review_text",
            "input_text",
            "content",
            "sentence",
        ],
    )

    if text_col and row.get(text_col):
        return "TEXT:" + text_hash(row[text_col])

    id_col = find_col(
        columns,
        [
            "text_hash",
            "input_hash",
            "example_id",
            "sample_id",
            "record_id",
            "row_id",
            "id",
        ],
    )

    if id_col and row.get(id_col):
        return "ID:" + str(row[id_col]).strip()

    return None

def row_failed(row):
    low = {
        str(k).lower(): str(v).strip().lower()
        for k, v in row.items()
    }

    # Explicit failure fields
    for key in [
        "success",
        "ok",
        "request_ok",
    ]:
        if key in low and low[key] in ["false", "0", "no"]:
            return True

    for key in [
        "status",
        "request_status",
    ]:
        if key in low:
            value = low[key]

            if (
                "fail" in value
                or "error" in value
                or "timeout" in value
                or "exception" in value
            ):
                return True

    for key in [
        "error",
        "request_error",
        "error_type",
    ]:
        if key in low:
            value = low[key]

            if value not in [
                "",
                "none",
                "null",
                "nan",
                "false",
                "0",
            ]:
                return True

    return False

def row_successful(row):
    if row_failed(row):
        return False

    low = {
        str(k).lower(): str(v).strip().lower()
        for k, v in row.items()
    }

    for key in [
        "success",
        "ok",
        "request_ok",
    ]:
        if key in low and low[key] in ["true", "1", "yes"]:
            return True

    for key in [
        "status",
        "request_status",
    ]:
        if key in low and low[key] in [
            "success",
            "successful",
            "ok",
            "completed",
        ]:
            return True

    # Fallback: usable prediction exists
    pred_col = find_col(
        row.keys(),
        [
            "pred_label",
            "predicted_label",
            "prediction",
            "pred",
            "y_pred",
            "predicted",
        ],
    )

    if pred_col:
        value = str(row.get(pred_col, "")).strip().lower()

        if value not in [
            "",
            "none",
            "null",
            "nan",
            "failed",
            "error",
        ]:
            return True

    return False

def read_csv_file(path):
    rows = []

    try:
        with open(
            path,
            "r",
            encoding="utf-8-sig",
            errors="replace",
            newline="",
        ) as f:

            reader = csv.DictReader(f)

            for row in reader:
                rows.append(row)

    except Exception:
        return None

    return rows

def read_csv_from_zip(zip_path, member):
    try:
        with zipfile.ZipFile(zip_path) as z:
            raw = z.read(member)

        text = raw.decode(
            "utf-8-sig",
            errors="replace",
        )

        reader = csv.DictReader(
            text.splitlines()
        )

        return list(reader)

    except Exception:
        return None

all_files = []

print("\nSearching...\n")

for root in SEARCH_ROOTS:

    root = Path(root)

    if not root.exists():
        print("Folder not found:", root)
        continue

    for path in root.rglob("*"):

        if not path.is_file():
            continue

        suffix = path.suffix.lower()

        # Normal CSV
        if suffix == ".csv":

            if looks_relevant(path):

                rows = read_csv_file(path)

                if rows is not None:

                    all_files.append(
                        {
                            "name": str(path),
                            "rows": rows,
                        }
                    )

        # ZIP archive
        elif suffix == ".zip":

            try:
                with zipfile.ZipFile(path) as z:

                    for member in z.namelist():

                        if not member.lower().endswith(".csv"):
                            continue

                        combined = str(path) + "/" + member

                        if looks_relevant(combined):

                            rows = read_csv_from_zip(
                                path,
                                member,
                            )

                            if rows is not None:

                                all_files.append(
                                    {
                                        "name": combined,
                                        "rows": rows,
                                    }
                                )

            except Exception:
                pass


print("Relevant prediction files found:")
print("=" * 80)

for i, file in enumerate(all_files):

    failed = sum(
        row_failed(r)
        for r in file["rows"]
    )

    successful = sum(
        row_successful(r)
        for r in file["rows"]
    )

    print(
        f"[{i}] rows={len(file['rows'])} "
        f"successful={successful} "
        f"failed={failed}"
    )

    print(file["name"])
    print()


# Find original failure-bearing file
failure_sources = []

for i, file in enumerate(all_files):

    failures = [
        r
        for r in file["rows"]
        if row_failed(r)
    ]

    if failures:

        failure_sources.append(
            (i, file, failures)
        )


print("\nRECOVERY CHECK")
print("=" * 80)

if not failure_sources:

    print(
        "No file containing explicit failures was found."
    )

else:

    for source_index, source, failures in failure_sources:

        failed_ids = {
            get_identity(r)
            for r in failures
            if get_identity(r)
        }

        print(
            f"\nFailure file [{source_index}]"
        )

        print(source["name"])

        print(
            "Failed rows:",
            len(failures),
        )

        print(
            "Identifiable failed reviews:",
            len(failed_ids),
        )

        for target_index, target in enumerate(all_files):

            if target_index == source_index:
                continue

            successful_ids = {
                get_identity(r)
                for r in target["rows"]
                if (
                    row_successful(r)
                    and get_identity(r)
                )
            }

            recovered = (
                failed_ids
                & successful_ids
            )

            if recovered:

                print()

                print(
                    f"Possible recovery file [{target_index}]"
                )

                print(target["name"])

                print(
                    f"Recovered: "
                    f"{len(recovered)} / "
                    f"{len(failed_ids)}"
                )

                if recovered == failed_ids:

                    print(
                        "*** ALL FAILED REVIEWS "
                        "WERE FOUND WITH "
                        "SUCCESSFUL PREDICTIONS ***"
                    )

                else:

                    print(
                        "Only some failed reviews "
                        "were recovered."
                    )


print("\nDone.")