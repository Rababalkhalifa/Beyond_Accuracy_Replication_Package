#!/usr/bin/env python3
r"""Plot exploratory within-period RQ2 risk–coverage and model rankings.

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


def plot(curve, svg_path, png_path=None):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8.2,
        "axes.titlesize": 9, "xtick.labelsize": 7.4, "ytick.labelsize": 7.4,
        "legend.fontsize": 7.3, "svg.fonttype": "none", "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#aab2bc", "savefig.facecolor": "white",
    })
    fig = plt.figure(figsize=(7.2, 5.7))
    grid = fig.add_gridspec(2, 3, left=.10, right=.985, top=.875, bottom=.12,
                            height_ratios=[1.55, 1.0], hspace=.47, wspace=.30)

    for j, setting in enumerate(SETTINGS):
        ax = fig.add_subplot(grid[0, j])
        ax.set_title(TITLES[j], loc="left", fontweight="bold", pad=7)
        ax.set_xlim(20, 100)
        ax.set_ylim(0, 42 if j == 0 else 11)  # Distinct scales: label in caption.
        ax.set_xticks((20, 40, 60, 80, 100))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        ax.grid(axis="y", color="#E7ECF1", linewidth=.7)
        ax.axvline(80, color="#b7bec7", linewidth=1, linestyle=":")
        ax.set_xlabel("Coverage retained (%)")
        if j == 0:
            ax.set_ylabel("Error among retained predictions")

        risk_at = {}
        for model in MODELS:
            samples = [[curve[setting, model, "within", seed, pct] * 100
                        for seed in sorted(SEEDS)] for pct in COVERAGES]
            means = [statistics.mean(values) for values in samples]
            sds = [statistics.stdev(values) for values in samples]
            risk_at[model] = {pct: means[pct - 20] for pct in RANK_COVERAGES}
            ax.fill_between(COVERAGES,
                            [m - sd for m, sd in zip(means, sds)],
                            [m + sd for m, sd in zip(means, sds)],
                            color=COLOR[model], alpha=.09, linewidth=0)
            ax.plot(COVERAGES, means, color=COLOR[model], linewidth=1.65,
                    linestyle=LINE[model])
            for pct in (80, 100):
                ax.plot(pct, risk_at[model][pct], marker=MARKER[model],
                        color=COLOR[model], markersize=3.5, markeredgecolor="white",
                        markeredgewidth=.4, zorder=4)

        rank_ax = fig.add_subplot(grid[1, j])
        rank_ax.set_title("Rank by mean retained error", loc="left",
                          color="#33435B", pad=4, fontsize=7.8)
        rank_ax.set_xlim(52, 108)
        rank_ax.set_ylim(4.4, .58)
        rank_ax.set_xticks(RANK_COVERAGES, ("60%", "80%", "100%"))
        rank_ax.set_yticks((1, 2, 3, 4))
        rank_ax.set_yticklabels(("1st", "2nd", "3rd", "4th") if j == 0 else ())
        rank_ax.spines["left"].set_color("#d2d7de")
        rank_ax.spines["bottom"].set_color("#d2d7de")
        rank_ax.grid(axis="y", color="#edf0f4", linewidth=.6)
        if j == 0:
            rank_ax.set_ylabel("Rank (lower risk first)")
        ranks = {pct: sorted(MODELS, key=lambda m: (risk_at[m][pct], MODELS.index(m)))
                 for pct in RANK_COVERAGES}
        for model in MODELS:
            ys = [ranks[pct].index(model) + 1 for pct in RANK_COVERAGES]
            rank_ax.plot(RANK_COVERAGES, ys, color=COLOR[model], linewidth=1.4,
                         linestyle=LINE[model], zorder=2)
            for pct, y in zip(RANK_COVERAGES, ys):
                rank_ax.plot(pct, y, marker=MARKER[model], color=COLOR[model],
                             markersize=5.2, zorder=3)
                rank_ax.annotate(f"{risk_at[model][pct]:.2f}", (pct, y),
                                 xytext=(0, -10), textcoords="offset points",
                                 ha="center", va="center", fontsize=6,
                                 color=COLOR[model], fontweight="bold",
                                 bbox={"facecolor": "white", "edgecolor": "none",
                                       "pad": .15, "alpha": .8}, zorder=4)

    handles = [Line2D([0], [0], color=COLOR[m], linestyle=LINE[m],
                      marker=MARKER[m], linewidth=1.7, markersize=4.3,
                      label=LABEL[m]) for m in MODELS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.54, .972),
               ncol=4, frameon=False, columnspacing=1.8, handlelength=2.5)
    fig.text(.10, .055, "Marker labels show mean retained error (%). "
             "Bands show ±1 sample SD across five seeds.",
             color="#435267", fontsize=7.2)

    svg_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(svg_path, format="svg")
    print(f"Saved resolution-independent SVG: {svg_path.resolve()}")
    if png_path:
        png_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_path, dpi=300)
        print(f"Saved 300-DPI preview: {png_path.resolve()}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True,
                        help="Path to RQ2_verified_seed_metrics.csv")
    parser.add_argument("--curves", type=Path, required=True,
                        help="Path to RQ2_risk_coverage_per_seed.csv")
    parser.add_argument("--svg", type=Path, required=True,
                        help="SVG output path")
    parser.add_argument("--png", type=Path, help="Optional 300-DPI PNG preview path")
    args = parser.parse_args()
    plot(load_verified(args.metrics, args.curves), args.svg, args.png)


if __name__ == "__main__":
    main()
