"""Publication figure generation for risk-coverage curves and model rankings.

Reproduces Figure: 'Coverage changes the model ranking' in SVG, PDF, and PNG formats.
"""

from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Dict, List, Sequence, Union

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

SEEDS = range(42, 47)
SETTINGS = ("Climate n500", "Yelp n500", "Yelp n1000")
MODELS = ("distilbert", "tfidf_lr", "phi3.5:latest", "qwen2.5:3b-instruct")
DISPLAY = ("DistilBERT", "TF-IDF+LR", "phi3.5:latest", "qwen2.5:3b-instruct")
COLORS = ("#0072B2", "#72777F", "#009E73", "#D55E00")
MARKERS = ("o", "s", "^", "D")


def plot_coverage_and_rankings(
    risk_rows: Sequence[Dict[str, any]],
    out_dir: Union[str, Path],
    filename_stem: str = "RQ2_coverage_rankings"
) -> List[Path]:
    """Generate high-resolution multi-panel risk vs. coverage and ranking figure."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    grouped = defaultdict(dict)
    for r in risk_rows:
        pct = int(round(float(r["coverage"]) * 100))
        grouped[r["setting"], r["model"], int(r["seed"]), r["period"]][pct] = float(r["expected_risk"])

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False
    })

    fig = plt.figure(figsize=(12.4, 8.5), facecolor="white")
    grid = fig.add_gridspec(2, 3, height_ratios=[1.55, 1], left=0.07, right=0.98,
                            top=0.79, bottom=0.14, wspace=0.24, hspace=0.43)

    fig.text(0.07, 0.955, "Coverage changes the model ranking", fontsize=21, fontweight="bold", color="#172B4D")
    fig.text(0.07, 0.913, "Within-period predictions • matched examples per setting • five sampling seeds • frozen models use P3",
             fontsize=11, color="#526071")

    legend = [Line2D([0], [0], color=c, marker=m, lw=2, label=l)
              for c, m, l in zip(COLORS, MARKERS, DISPLAY)]
    fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(0.063, 0.89), ncol=4,
               frameon=False, fontsize=10.7, columnspacing=2.3)

    xs = np.arange(20, 101) / 100.0

    for j, setting in enumerate(SETTINGS):
        upper = fig.add_subplot(grid[0, j])
        lower = fig.add_subplot(grid[1, j])
        means = {}

        upper.axvspan(0.79, 0.81, color="#E8EDF4", zorder=0)
        upper.axvline(0.8, color="#8393A7", lw=0.9, ls=":", zorder=1)

        for i, model in enumerate(MODELS):
            mat = np.asarray([[grouped[setting, model, seed, "within"][p] for p in range(20, 101)]
                              for seed in SEEDS])
            mu = mat.mean(axis=0)
            sd = mat.std(axis=0, ddof=1)
            means[model] = mu

            upper.fill_between(xs, np.maximum(0, mu - sd), mu + sd, color=COLORS[i], alpha=0.10, linewidth=0)
            upper.plot(xs, mu, color=COLORS[i], lw=2.1)
            upper.scatter([0.8, 1.0], [mu[60], mu[80]], color=COLORS[i], marker=MARKERS[i],
                          s=34, zorder=4, edgecolors="white", linewidth=0.6)

        upper.set_title(("Climate stance • n500", "Yelp sentiment • n500", "Yelp sentiment • n1000")[j],
                        loc="left", fontsize=12, fontweight="bold", pad=14)
        upper.set_xlim(0.2, 1.015)
        upper.set_ylim(0, 0.43 if j == 0 else 0.115)
        upper.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
        upper.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        upper.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        upper.set_xlabel("Coverage retained")
        if j == 0:
            upper.set_ylabel("Error among retained predictions")
        upper.text(0.8, 1.015, "80%", transform=upper.get_xaxis_transform(), ha="center", color="#66788A", fontsize=9)
        upper.grid(axis="y", color="#E5EAF0", lw=0.7)
        upper.set_axisbelow(True)

        lower.axvspan(0.91, 1.09, color="#F1F4F8", zorder=0)
        ticks = (60, 80, 100)
        rank = {model: [] for model in MODELS}
        for p in ticks:
            v = {model: means[model][p - 20] for model in MODELS}
            for model in MODELS:
                rank[model].append(1 + sum(x < v[model] - 1e-12 for x in v.values()))

        for i, model in enumerate(MODELS):
            lower.plot(range(3), rank[model], color=COLORS[i], lw=2.2, alpha=0.85)
            for k, p in enumerate(ticks):
                shown = str(Decimal(str(round(float(means[model][p - 20] * 100), 10))).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP))
                lower.text(k, rank[model][k], shown, color="white", va="center", ha="center",
                           fontsize=8.8, fontweight="bold", zorder=4,
                           bbox=dict(boxstyle="round,pad=0.4,rounding_size=0.65",
                                     facecolor=COLORS[i], edgecolor="white", linewidth=1.1))

        lower.set_xlim(-0.3, 2.3)
        lower.set_ylim(4.48, 0.5)
        lower.set_xticks(range(3), ["60%", "80%", "100%"])
        lower.set_yticks([1, 2, 3, 4])
        lower.set_yticklabels(["1st", "2nd", "3rd", "4th"])
        lower.set_xlabel("Coverage retained")
        if j == 0:
            lower.set_ylabel("Rank by mean retained error")
        lower.set_title("Ranking at three operating points", fontsize=10.5, loc="left", pad=10, color="#526071")
        lower.grid(axis="y", color="#E5EAF0", lw=0.7)
        lower.set_axisbelow(True)

        for ax in (upper, lower):
            ax.spines["left"].set_color("#C4CDD7")
            ax.spines["bottom"].set_color("#C4CDD7")
            ax.tick_params(labelsize=9)

    fig.text(0.07, 0.078, "Top: mean risk with ±1 sample SD. Bottom: numbers inside markers are retained error percentages; ranks describe means only.",
             fontsize=9, color="#526071")
    fig.text(0.07, 0.052, "Coverage sweep: 20–100%. Tied confidence scores use expected risk under uniform random tie-breaking; P5 abstention is excluded.",
             fontsize=9, color="#526071")
    fig.text(0.07, 0.026, "n is examples per class. Each model may retain different cases. Small rank differences do not establish a statistically reliable ordering.",
             fontsize=9, color="#526071")

    saved_paths = []
    for ext in ("svg", "pdf", "png"):
        p = out / f"{filename_stem}.{ext}"
        fig.savefig(p, dpi=220, facecolor="white")
        saved_paths.append(p)
    plt.close(fig)

    return saved_paths
