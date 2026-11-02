#!/usr/bin/env python3
r"""Reproduce Figure 3: 3 x 3 temporal risk–coverage curves, without footnotes.

Input files are derived, seed-level results from the completed prediction audit.
This script only checks and visualizes saved results; it does not fit or run models.

Example (PowerShell):
  python .\plot_rq2_coverage.py --metrics .\RQ2_verified_seed_metrics.csv `
      --curves .\RQ2_risk_coverage_per_seed.csv `
      --svg .\RQ2_coverage_rankings.svg --png .\RQ2_coverage_rankings.png
"""

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter


SETTINGS = ("Climate n500", "Yelp n500", "Yelp n1000")
TITLES = ("Climate stance · n500", "Yelp sentiment · n500", "Yelp sentiment · n1000")
MODELS = ("distilbert", "tfidf_lr", "phi3.5:latest", "qwen2.5:3b-instruct")
PERIODS = ("within", "short", "long")
SEEDS = {42, 43, 44, 45, 46}
COVERAGES = tuple(range(20, 101))
RANK_COVERAGES = (60, 80, 100)
LABEL = {"distilbert": "DistilBERT", "tfidf_lr": "TF–IDF+LR",
         "phi3.5:latest": "Phi", "qwen2.5:3b-instruct": "Qwen"}
COLOR = {"distilbert": "#176FAD", "tfidf_lr": "#626A75",
         "phi3.5:latest": "#08896B", "qwen2.5:3b-instruct": "#CB5900"}
MARKER = dict(zip(MODELS, ("o", "s", "^", "D")))
LINE = dict(zip(MODELS, ("-", "--", "-.", ":")))


def read_csv(path, fields):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not set(fields).issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: expected columns {', '.join(fields)}")
        return list(reader)


def check_unit(value, name):
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError(f"{name} must be finite and between 0 and 1: {value}")
    return number


def load_verified(metrics_path, curves_path):
    metrics = read_csv(metrics_path,
                       ("setting", "model", "seed", "period", "accuracy",
                        "risk_at_80", "risk_at_100"))
    curves = read_csv(curves_path,
                      ("setting", "model", "seed", "period", "coverage", "expected_risk"))
    if len(metrics) != 180 or len(curves) != 14580:
        raise ValueError("Expected 180 metric rows and 14,580 curve rows "
                         "(3 settings × 4 models × 3 periods × 5 seeds × 81 coverages).")

    metric_by_key, curve_by_key = {}, {}
    for row in metrics:
        key = (row["setting"], row["model"], row["period"], int(row["seed"]))
        if key in metric_by_key:
            raise ValueError(f"Duplicate metric row: {key}")
        metric_by_key[key] = row

    for row in curves:
        pct = round(float(row["coverage"]) * 100)
        if not math.isclose(float(row["coverage"]) * 100, pct, abs_tol=1e-7):
            raise ValueError(f"Unexpected coverage: {row['coverage']}")
        key = (row["setting"], row["model"], row["period"], int(row["seed"]), pct)
        if key in curve_by_key:
            raise ValueError(f"Duplicate curve row: {key}")
        curve_by_key[key] = check_unit(row["expected_risk"], "expected_risk")

    expected_metric = {(setting, model, period, seed)
                       for setting in SETTINGS for model in MODELS
                       for period in PERIODS for seed in SEEDS}
    expected_curve = {(*key, pct) for key in expected_metric for pct in COVERAGES}
    if metric_by_key.keys() != expected_metric or curve_by_key.keys() != expected_curve:
        raise ValueError("Rows do not cover exactly the expected settings, models, "
                         "periods, seeds 42–46 and coverage 20–100% by 1%.")

    for key, row in metric_by_key.items():
        for pct, field in ((80, "risk_at_80"), (100, "risk_at_100")):
            if not math.isclose(curve_by_key[(*key, pct)], check_unit(row[field], field),
                                abs_tol=1e-10, rel_tol=0):
                raise ValueError(f"Saved {field} differs from the curve for {key}")
        if not math.isclose(curve_by_key[(*key, 100)],
                            1 - check_unit(row["accuracy"], "accuracy"),
                            abs_tol=1e-10, rel_tol=0):
            raise ValueError(f"Full-coverage risk differs from 1 − accuracy for {key}")

    print("Verified 180 metric rows and 14,580 curve rows; five seeds in all conditions.")
    print("Seed-level risks at 80% and 100% match the metrics file; "
          "full-coverage risk matches 1 − accuracy.")
    return curve_by_key


def plot(curve, pdf_path, svg_path=None, png_path=None):
    """Figure 3: rows=settings, columns=historical periods. No bottom note."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.labelsize": 9, "axes.titlesize": 10,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "legend.fontsize": 9, "svg.fonttype": "none", "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#CBD2D8", "savefig.facecolor": "white",
    })
    fig, axs = plt.subplots(3, 3, figsize=(11.4, 8.2), sharex=False, sharey='row')
    fig.subplots_adjust(left=.105, right=.985, top=.88, bottom=.095,
                        wspace=.32, hspace=.43)
    period_titles = {"within": "Reference", "short": "Short", "long": "Long"}
    row_titles = ["Climate stance\nn = 500 per class",
                  "Yelp sentiment\nn = 500 per class",
                  "Yelp sentiment\nn = 1,000 per class"]

    for i, setting in enumerate(SETTINGS):
        for j, period in enumerate(PERIODS):
            ax = axs[i, j]
            ax.set_title(period_titles[period], loc="left", fontweight="bold", pad=8)
            ax.set_xlim(20, 100)
            ax.set_ylim(0, 45 if i == 0 else 12.5)
            ax.set_xticks((20, 40, 60, 80, 100))
            ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
            ax.grid(axis="y", color="#E7ECF1", linewidth=.7)
            ax.axvline(80, color="#BFC6CC", linewidth=.9, linestyle=":", zorder=1)
            if i == 2:
                ax.set_xlabel("Coverage retained")
            if j == 0:
                ax.set_ylabel("Retained error")
                ax.annotate(row_titles[i], xy=(-.36, .5), xycoords="axes fraction",
                            ha="center", va="center", rotation=90,
                            fontweight="bold", fontsize=9)
            for model in MODELS:
                samples = [[curve[setting, model, period, seed, pct] * 100
                            for seed in sorted(SEEDS)] for pct in COVERAGES]
                means = [statistics.mean(values) for values in samples]
                sds = [statistics.stdev(values) for values in samples]
                ax.fill_between(COVERAGES,
                                [max(0, m - sd) for m, sd in zip(means, sds)],
                                [m + sd for m, sd in zip(means, sds)],
                                color=COLOR[model], alpha=.12, linewidth=0)
                ax.plot(COVERAGES, means, color=COLOR[model], lw=1.85,
                        linestyle="-", zorder=3)
                for pct in (60, 80, 95, 100):
                    ax.plot(pct, means[pct-20], marker=MARKER[model],
                            color=COLOR[model], markersize=3.8,
                            markeredgecolor="white", markeredgewidth=.4, zorder=4)

    handles = [Line2D([0], [0], color=COLOR[m], marker=MARKER[m],
                      linestyle="-", lw=1.9, markersize=5, label=LABEL[m])
               for m in MODELS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.54, .985),
               ncol=4, frameon=False, columnspacing=1.8, handlelength=2.5)
    # Intentionally no fig.text notes underneath the nine panels.
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, format="pdf")
    print(f"Saved: {pdf_path.resolve()}")
    if svg_path is not None:
        svg_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(svg_path, format="svg")
        print(f"Saved: {svg_path.resolve()}")
    if png_path is not None:
        png_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_path, format="png", dpi=300)
        print(f"Saved: {png_path.resolve()}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--curves", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, default=Path("RQ2_temporal_risk_coverage.pdf"))
    parser.add_argument("--svg", type=Path)
    parser.add_argument("--png", type=Path)
    args = parser.parse_args()
    plot(load_verified(args.metrics, args.curves), args.pdf, args.svg, args.png)


if __name__ == "__main__":
    main()
