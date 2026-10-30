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


def plot_figure5_movement_vs_response(
    forensics_rows: Sequence[Dict[str, any]],
    delta_rows: Sequence[Dict[str, any]],
    out_dir: Union[str, Path],
    filename_stem: str = "Figure5_movement_vs_response"
) -> List[Path]:
    """Generate publication Figure 5: Movement vs. Response dual-panel heatmap."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    condition_order = [
        ("Climate n500", "short"),
        ("Climate n500", "long"),
        ("Yelp n500", "short"),
        ("Yelp n500", "long"),
        ("Yelp n1000", "short"),
        ("Yelp n1000", "long"),
    ]

    model_order = [
        "distilbert",
        "tfidf_lr",
        "phi3.5:latest",
        "qwen2.5:3b-instruct",
    ]

    model_names = {
        "distilbert": "DistilBERT",
        "tfidf_lr": "TF-IDF LR",
        "phi3.5:latest": "Phi",
        "qwen2.5:3b-instruct": "Qwen",
    }

    metric_cols = {
        "Novelty": "lexical_novelty_mean",
        "JSD": "js_divergence_mean",
        "ΔC100": "top100_mass_change_mean",
        "Δ length": "mean_length_change_mean",
        "Overlap loss": "overlap_loss_mean",
        "Marker\nasymmetry": "class_marker_asymmetry_mean",
    }
    metric_names = list(metric_cols.keys())

    # Map forensics rows
    rq5_map = {(r["dataset_setting"], r["target"]): r for r in forensics_rows}

    # Format annotations and relative intensities
    def format_forensic_value(metric, value):
        if metric == "Δ length":
            return f"↑ {abs(value):.1f}" if value > 0 else (f"↓ {abs(value):.1f}" if value < 0 else "0.0")
        if metric == "ΔC100":
            return f"↑ {abs(value):.3f}" if value > 0 else (f"↓ {abs(value):.3f}" if value < 0 else "0.000")
        return f"{value:.3f}"

    heat_a = []
    annot_a = []

    for setting, target in condition_order:
        row = rq5_map[setting, target]
        heat_row = []
        text_row = []
        for metric, col in metric_cols.items():
            val = float(row[col])
            all_vals = [abs(float(rq5_map[s, t][col])) if metric in ("ΔC100", "Δ length")
                        else float(rq5_map[s, t][col]) for s, t in condition_order]
            max_val = max(all_vals)
            mag = abs(val) if metric in ("ΔC100", "Δ length") else val
            rel = (mag / max_val) if max_val > 0 else 0.0
            heat_row.append(rel)
            text_row.append(format_forensic_value(metric, val))
        heat_a.append(heat_row)
        annot_a.append(text_row)

    heat_a = np.array(heat_a)
    row_labels = [f"{dataset}\n{target.capitalize()}" for dataset, target in condition_order]

    # Panel B Response Matrix
    delta_map = {(r["dataset_setting"], r["target"], r["model"]): r for r in delta_rows}
    response_matrix = np.zeros((len(condition_order), len(model_order)))

    for i, (setting, target) in enumerate(condition_order):
        for j, model in enumerate(model_order):
            if (setting, target, model) in delta_map:
                row = delta_map[setting, target, model]
                # Check for either delta_macro_f1_mean or delta_short/delta_long
                if "delta_macro_f1_mean" in row:
                    val = float(row["delta_macro_f1_mean"])
                elif target == "short" and "delta_short" in row:
                    val = float(row["delta_short"])
                elif target == "long" and "delta_long" in row:
                    val = float(row["delta_long"])
                else:
                    val = 0.0
                response_matrix[i, j] = val

    response_limit = float(np.max(np.abs(response_matrix))) if np.max(np.abs(response_matrix)) > 0 else 0.1

    # Plot dual-panel figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 9), gridspec_kw={"width_ratios": [1.25, 0.95], "wspace": 0.28})

    im1 = ax1.imshow(heat_a, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax1.set_title("A. Dataset-side temporal movement", fontweight="bold", fontsize=15, pad=16)
    ax1.set_xticks(np.arange(len(metric_names)))
    ax1.set_xticklabels(metric_names, fontweight="bold")
    ax1.set_yticks(np.arange(len(row_labels)))
    ax1.set_yticklabels(row_labels, fontweight="bold")

    ax1.set_xticks(np.arange(-0.5, len(metric_names), 1), minor=True)
    ax1.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax1.grid(which="minor", color="white", linewidth=2)
    ax1.tick_params(which="minor", bottom=False, left=False)

    for i in range(heat_a.shape[0]):
        for j in range(heat_a.shape[1]):
            strength = heat_a[i, j]
            color = "white" if strength >= 0.62 else "black"
            ax1.text(j, i, annot_a[i][j], ha="center", va="center", fontsize=12, fontweight="bold", color=color)

    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.045, pad=0.035)
    cbar1.set_label("Relative magnitude within each metric", fontsize=11, fontweight="bold")

    im2 = ax2.imshow(response_matrix, aspect="auto", cmap="RdYlGn", vmin=-response_limit, vmax=response_limit)
    ax2.set_title("B. Model response", fontweight="bold", fontsize=15, pad=16)
    ax2.set_xticks(np.arange(len(model_order)))
    ax2.set_xticklabels([model_names[m] for m in model_order], fontweight="bold")
    ax2.set_yticks(np.arange(len(row_labels)))
    ax2.set_yticklabels(row_labels, fontweight="bold")

    ax2.set_xticks(np.arange(-0.5, len(model_order), 1), minor=True)
    ax2.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax2.grid(which="minor", color="white", linewidth=2)
    ax2.tick_params(which="minor", bottom=False, left=False)

    for i in range(response_matrix.shape[0]):
        for j in range(response_matrix.shape[1]):
            val = response_matrix[i, j]
            rel = (abs(val) / response_limit) if response_limit > 0 else 0
            text_color = "white" if rel >= 0.60 else "black"
            cell_text = f"↑ {val:.3f}" if val > 0 else (f"↓ {abs(val):.3f}" if val < 0 else "0.000")
            ax2.text(j, i, cell_text, ha="center", va="center", fontsize=12, fontweight="bold", color=text_color)

    ax2.text(0.0, 1.025, "↓ lower Macro-F1", transform=ax2.transAxes, ha="left", va="bottom", fontsize=10.5, fontweight="bold")
    ax2.text(1.0, 1.025, "↑ higher Macro-F1", transform=ax2.transAxes, ha="right", va="bottom", fontsize=10.5, fontweight="bold")

    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.050, pad=0.04)
    cbar2.set_label("ΔMacro-F1 (target − within)", fontsize=11, fontweight="bold")

    for ax in [ax1, ax2]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Temporal movement profiles and corresponding model-performance changes", fontsize=17, fontweight="bold", y=0.98)

    saved_paths = []
    for ext in ("svg", "pdf", "png"):
        p = out / f"{filename_stem}.{ext}"
        fig.savefig(p, dpi=220, facecolor="white", bbox_inches="tight")
        saved_paths.append(p)
    plt.close(fig)

    return saved_paths
