#!/usr/bin/env python3
"""Rebuild an isolated, verified Windows run directory for Climate Qwen P1.

Requires Python 3.10+ standard library only. The evaluation runner itself
requires its original machine's ML dependencies and local Ollama server.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import zipfile

EXPECTED_MANIFEST_SHA256 = "3ea19ee6af7594a59dfc28e2099c211f1d210d65e8231cefa5b67b4d3dbe5dc3"
EXPECTED_RUNNER_SHA256 = "6eb5aa9789a04df666a002ae1ded0b695c49f4317cb8bbf92976970bf17403ba"
SPLIT_ROOT = "shared_splits_paper_v6_joint_quota"
MODEL_FOLDER = "qwen2.5_3b-instruct"
PERIOD_FILES = {
    "within": "within_2010_n500_per_label_total1000.csv",
    "short": "short_2013_n500_per_label_total1000.csv",
    "long": "long_2018_2019_n500_per_label_total1000.csv",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_csv_bytes(data):
    return list(csv.DictReader(io.StringIO(data.decode("utf-8-sig", errors="strict"), newline="")))


def err(message):
    raise RuntimeError(message)


def archive_entry(zf, filename):
    try:
        return zf.read(filename)
    except KeyError:
        err(f"Missing archive entry: {filename}")


def clean_extract(zf, parent):
    """Extract only paths whose components stay under parent."""
    base = parent.resolve()
    for info in zf.infolist():
        # The attached source uses forward-slash paths, no symlinks.
        name = info.filename.replace("\\", "/")
        parts = name.split("/")
        if name.startswith("/") or ".." in parts or ":" in name:
            err(f"Unsafe ZIP path: {info.filename}")
        dest = (parent / name).resolve()
        if not dest.is_relative_to(base):
            err(f"Unsafe ZIP path: {info.filename}")
        if info.is_dir():
            dest.mkdir(parents=True, exist_ok=True)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as source, dest.open("wb") as target:
                shutil.copyfileobj(source, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, required=True, help="Original shared_splits ZIP")
    parser.add_argument("--p1", type=Path, required=True, help="Saved p1_minimal.zip")
    parser.add_argument("--runner", type=Path, required=True, help="Original run_paper_models.py")
    parser.add_argument("--metrics", type=Path, required=True, help="Original metrics_per_condition(3).csv")
    parser.add_argument("--out", type=Path, default=Path("climate_recovery"), help="NEW recovery folder")
    args = parser.parse_args()

    for p in [args.splits, args.p1, args.runner, args.metrics]:
        if not p.is_file():
            err(f"Input not found: {p}")
    out = args.out.resolve()
    if out.exists():
        err(f"To protect previous results, the output folder must NOT already exist: {out}")
    if digest(args.runner.read_bytes()) != EXPECTED_RUNNER_SHA256:
        err("Runner script SHA-256 differs from original recorded script. Refusing to mix experimental versions.")

    with zipfile.ZipFile(args.splits) as sz, zipfile.ZipFile(args.p1) as pz:
        manifest_bytes = archive_entry(sz, f"{SPLIT_ROOT}/split_manifest.json")
        if digest(manifest_bytes) != EXPECTED_MANIFEST_SHA256:
            err("Split manifest SHA-256 does not match the original frozen run settings.")
        audit = json.loads(archive_entry(sz, f"{SPLIT_ROOT}/audit_report.json"))
        if audit.get("status") != "PASS":
            err("The original split audit is not PASS.")
        index = read_csv_bytes(archive_entry(sz, f"{SPLIT_ROOT}/split_index.csv"))
        for item in index:
            member = f"{SPLIT_ROOT}/{item['path'].replace(chr(92), '/')}"
            if digest(archive_entry(sz, member)) != item["sha256"]:
                err(f"Checksum mismatch in split: {member}")

        report = []
        checked_predictions = []
        for seed in range(42, 47):
            for period, ev_file in PERIOD_FILES.items():
                eval_member = f"{SPLIT_ROOT}/climate/seed_{seed}/{ev_file}"
                ev_rows = read_csv_bytes(archive_entry(sz, eval_member))
                if len(ev_rows) != 1000:
                    err(f"Expected 1,000 evaluation examples: {eval_member}")
                expected = {r["text_sha256"]: r for r in ev_rows}
                if len(expected) != 1000:
                    err(f"Duplicate evaluation hash in {eval_member}")
                pred_member = f"p1_minimal/climate/seed_{seed}/{period}_n500_per_label.csv"
                if pred_member in pz.namelist():
                    data = pz.read(pred_member)
                    rows = read_csv_bytes(data)
                    used = set()
                    valid = timeouts = 0
                    for r in rows:
                        key = r.get("text_sha256")
                        if key in used or key not in expected:
                            err(f"Duplicate or unexpected prediction: {pred_member}, {key}")
                        used.add(key)
                        orig = expected[key]
                        if (r.get("true_label") != orig["label"] or
                                r.get("text") != orig["text"] or
                                r.get("example_id") != orig["example_id"] or
                                r.get("model") != "qwen2.5:3b-instruct" or
                                r.get("prompt_id") != "p1_minimal" or
                                str(r.get("seed")) != str(seed) or
                                r.get("period") != period):
                            err(f"Saved prediction does not match original split: {pred_member}, {key}")
                        if r["status"] == "ok":
                            valid += 1
                        elif r["status"] == "request_error" and "ReadTimeout" in r.get("request_error", ""):
                            timeouts += 1
                        else:
                            err(f"Unexpected saved prediction status: {pred_member}, {r['status']}")
                else:
                    data = None
                    valid = timeouts = 0
                    used = set()
                checked_predictions.append((seed, period, data))
                report.append({
                    "seed": seed, "period": period, "expected": 1000,
                    "saved": len(used), "valid": valid, "timeouts": timeouts,
                    "not_started": 1000 - len(used),
                    "calls_remaining": 1000 - valid,
                })

        metric_rows = read_csv_bytes(args.metrics.read_bytes())
        if len(metric_rows) != 100:
            err(f"Expected 100 original metric rows, found {len(metric_rows)}. Inspect source metrics.")

        # No local folder is modified until all checks have passed.
        out.mkdir(parents=True)
        clean_extract(sz, out)
        runner_dest = out / "run_paper_models.py"
        shutil.copy2(args.runner, runner_dest)
        run_dir = out / "recovered_run"
        run_dir.mkdir()
        shutil.copy2(args.metrics, run_dir / "metrics_per_condition.csv")
        shutil.copy2(args.metrics, run_dir / "metrics_original_backup.csv")
        for seed, period, data in checked_predictions:
            if data is None:
                continue
            p = (run_dir / "predictions" / "frozen" / MODEL_FOLDER / "p1_minimal" /
                 "climate" / f"seed_{seed}" / f"{period}_n500_per_label.csv")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        with (out / "prediction_recovery_audit.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(report[0]))
            writer.writeheader()
            writer.writerows(report)

        command = (f'python "{runner_dest}" --mode frozen '
                   f'--split-root "{out / SPLIT_ROOT}" '
                   f'--run-dir "{run_dir}" '
                   '--datasets climate --seeds 45 46 '
                   '--frozen-models "qwen2.5:3b-instruct" '
                   '--prompt-ids p1_minimal --resume --retry-invalid '
                   '--num-workers 1 --checkpoint-every 10')
        (out / "RESUME_QWEN_P1.bat").write_text(
            "@echo off\r\n"
            "echo Make sure Ollama is running and qwen2.5:3b-instruct is installed.\r\n"
            + command + "\r\n"
            "pause\r\n", encoding="utf-8")
        (out / "RESUME_COMMAND.txt").write_text(command + "\n", encoding="utf-8")

        totals = {key: sum(row[key] for row in report)
                  for key in ["expected", "saved", "valid", "timeouts", "not_started", "calls_remaining"]}
        report_obj = {
            "split_audit": audit.get("status"), "split_files_sha256_passed": len(index),
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "runner_sha256": EXPECTED_RUNNER_SHA256,
            "metrics_rows_preserved": len(metric_rows),
            "qwen_p1_climate": totals,
            "batch_to_resume": "qwen2.5:3b-instruct p1_minimal climate seeds 45 and 46",
            "no_existing_input_modified": True,
        }
        (out / "recovery_summary.json").write_text(json.dumps(report_obj, indent=2), encoding="utf-8")
        print(json.dumps(report_obj, indent=2))
        print("\nRecovery prepared. On Windows, after verifying local Ollama is responsive, run:")
        print(command)
        print("\nOriginal files were not changed. See prediction_recovery_audit.csv.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, RuntimeError, zipfile.BadZipFile) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
