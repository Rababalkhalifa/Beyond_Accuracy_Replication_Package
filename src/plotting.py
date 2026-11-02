"""Publication figure generation for the manuscript:
- Figure 3: 3x3 Confidence-based risk-coverage curves across historical evaluation periods
- Figure 4: 3-panel Exploratory retained-error comparisons for RQ2 (11 cluster-bootstrap contrasts)
- Figure 5: 3-panel Reference-period Macro-F1 across prompt configurations (P1, P3, P5)
- Figure 6: Dual-panel Temporal movement profiles and corresponding model-performance changes
- Supplemental: Within-period risk-coverage and model ranking plot (RQ2_coverage_rankings)
"""

import statistics
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
PERIODS = ("within", "short", "long")
MODELS = ("distilbert", "tfidf_lr", "phi3.5:latest", "qwen2.5:3b-instruct")
DISPLAY = ("DistilBERT", "TF-IDF+LR", "phi3.5:latest", "qwen2.5:3b-instruct")
COLORS = ("#0072B2", "#72777F", "#009E73", "#D55E00")
MARKERS = ("o", "s", "^", "D")


def plot_figure3_temporal_risk_coverage(
    risk_rows: Sequence[Dict[str, any]],
    out_dir: Union[str, Path],
    filename_stem: str = "Figure3_temporal_risk_coverage"
) -> List[Path]:
    """Generate manuscript Figure 3: 3x3 temporal risk-coverage curves across periods and settings."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    curve = {}
    for r in risk_rows:
        pct = int(round(float(r["coverage"]) * 100))
        curve[r["setting"], r["model"], r["period"], int(r["seed"]), pct] = float(r["expected_risk"])

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.labelsize": 9, "axes.titlesize": 10,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "legend.fontsize": 9, "svg.fonttype": "none", "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#CBD2D8", "savefig.facecolor": "white",
    })
    fig, axs = plt.subplots(3, 3, figsize=(11.4, 8.2), sharex=False, sharey="row")
    fig.subplots_adjust(left=.105, right=.985, top=.88, bottom=.095, wspace=.32, hspace=.43)

    period_titles = {"within": "Reference", "short": "Short", "long": "Long"}
    row_titles = [
        "Climate stance\nn = 500 per class",
        "Yelp sentiment\nn = 500 per class",
        "Yelp sentiment\nn = 1,000 per class"
    ]
    label_map = {
        "distilbert": "DistilBERT",
        "tfidf_lr": "TF–IDF+LR",
        "phi3.5:latest": "Phi",
        "qwen2.5:3b-instruct": "Qwen"
    }
    color_map = {
        "distilbert": "#176FAD",
        "tfidf_lr": "#626A75",
        "phi3.5:latest": "#08896B",
        "qwen2.5:3b-instruct": "#CB5900"
    }
    marker_map = dict(zip(MODELS, ("o", "s", "^", "D")))
    coverages = tuple(range(20, 101))

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
                            ha="center", va="center", rotation=90, fontweight="bold", fontsize=9)
            for model in MODELS:
                samples = [[curve[setting, model, period, seed, pct] * 100
                            for seed in sorted(SEEDS)] for pct in coverages]
                means = [statistics.mean(values) for values in samples]
                sds = [statistics.stdev(values) for values in samples]
                ax.fill_between(coverages,
                                [max(0, m - sd) for m, sd in zip(means, sds)],
                                [m + sd for m, sd in zip(means, sds)],
                                color=color_map[model], alpha=.12, linewidth=0)
                ax.plot(coverages, means, color=color_map[model], lw=1.85, linestyle="-", zorder=3)
                for pct in (60, 80, 95, 100):
                    ax.plot(pct, means[pct - 20], marker=marker_map[model],
                            color=color_map[model], markersize=3.8,
                            markeredgecolor="white", markeredgewidth=.4, zorder=4)

    handles = [Line2D([0], [0], color=color_map[m], marker=marker_map[m],
                      linestyle="-", lw=1.9, markersize=5, label=label_map[m])
               for m in MODELS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.54, .985),
               ncol=4, frameon=False, columnspacing=1.8, handlelength=2.5)

    saved_paths = []
    for stem in (filename_stem, "RQ2_temporal_risk_coverage"):
        for ext in ("pdf", "svg", "png"):
            p = out / f"{stem}.{ext}"
            fig.savefig(p, dpi=300 if ext == "png" else None, facecolor="white")
            if stem == filename_stem:
                saved_paths.append(p)
    plt.close(fig)
    return saved_paths


def plot_figure4_uncertainty_contrasts(
    contrast_rows: Sequence[Dict[str, any]],
    out_dir: Union[str, Path],
    filename_stem: str = "Figure4_uncertainty_contrasts"
) -> List[Path]:
    """Generate manuscript Figure 4: 3-panel exploratory retained-error comparisons for RQ2."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    period_labels = {"within": "Reference", "short": "Short", "long": "Long"}
    sections = [
        ("A  Yelp: Phi minus Qwen at 80%", "#1875A5",
         [("Yelp n500", "within", "phi_minus_qwen_at_80", "n500 · Reference"),
          ("Yelp n500", "short", "phi_minus_qwen_at_80", "n500 · Short"),
          ("Yelp n500", "long", "phi_minus_qwen_at_80", "n500 · Long"),
          ("Yelp n1000", "within", "phi_minus_qwen_at_80", "n1000 · Reference"),
          ("Yelp n1000", "short", "phi_minus_qwen_at_80", "n1000 · Short"),
          ("Yelp n1000", "long", "phi_minus_qwen_at_80", "n1000 · Long")]),
        ("B  Climate: Qwen 80% minus 100%", "#C16A23",
         [("Climate n500", p, "qwen_80_minus_100", period_labels[p])
          for p in ("within", "short", "long")]),
        ("C  Climate: DistilBERT temporal change", "#17835D",
         [("Climate n500", "short", "distilbert_period_minus_reference_at_80", "Short − Reference"),
          ("Climate n500", "long", "distilbert_period_minus_reference_at_80", "Long − Reference")])
    ]

    lookup = {
        (r["setting"], r["period"], r["contrast"]): (
            float(r["mean_difference_pp"]),
            float(r["ci_95_low_pp"]),
            float(r["ci_95_high_pp"])
        )
        for r in contrast_rows
    }

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 5.2), gridspec_kw={"width_ratios": [1.55, 1, 0.9]})
    for ax, (title, color, items) in zip(axes, sections):
        values = np.array([lookup[(setting, period, contrast)] for setting, period, contrast, _ in items])
        x, lo, hi = values.T
        y = np.arange(len(items))[::-1]
        ax.errorbar(x, y, xerr=[x - lo, hi - x], fmt="o", markersize=5.8, color=color,
                    ecolor=color, elinewidth=1.5, capsize=3, zorder=3)
        ax.axvline(0, color="#5C646E", linewidth=1, linestyle="--")
        ax.set_yticks(y, [label for *_, label in items], fontsize=9)
        ax.set_title(title, fontsize=10.5, loc="left", weight="bold", pad=12)
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=9)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.set_ylim(-.65, len(items) - .35)
        ax.set_xlabel("Retained-error difference (percentage points)", fontsize=9)
        ax.set_xlim(min(lo.min() - .2, -.2), max(hi.max() + .2, .2))

    fig.subplots_adjust(left=.13, right=.99, top=.81, bottom=.19, wspace=.43)

    saved_paths = []
    for stem in (filename_stem, "RQ2_uncertainty_contrasts"):
        for ext in ("pdf", "svg", "png"):
            p = out / f"{stem}.{ext}"
            fig.savefig(p, bbox_inches="tight", dpi=240 if ext == "png" else None, facecolor="white")
            if stem == filename_stem:
                saved_paths.append(p)
    plt.close(fig)
    return saved_paths


def plot_figure5_prompt_comparison(
    prompt_summary_rows: Sequence[Dict[str, any]],
    out_dir: Union[str, Path],
    filename_stem: str = "Figure5_prompt_comparison"
) -> List[Path]:
    """Generate manuscript Figure 5: Reference-period Macro-F1 across prompt configurations (P1, P3, P5)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    prompts = ["p1_minimal", "p3_json_strict", "p5_abstain"]
    frozen_models = [("phi3.5:latest", "Phi", "o"), ("qwen2.5:3b-instruct", "Qwen", "s")]
    lookup = {(r["setting"], r["model"], r["prompt"]): r for r in prompt_summary_rows}

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "svg.fonttype": "none",
        "pdf.fonttype": 42
    })

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    for ax, setting in zip(axes, SETTINGS):
        for model_id, model_label, marker in frozen_models:
            means = [float(lookup[(setting, model_id, p)]["macro_f1_mean"]) for p in prompts]
            sds = [float(lookup[(setting, model_id, p)]["macro_f1_sd"]) for p in prompts]
            ax.errorbar(
                range(3),
                means,
                yerr=sds,
                marker=marker,
                linewidth=1.7,
                markersize=6,
                capsize=4,
                label=model_label
            )

        ax.set_title(setting)
        ax.set_ylabel("Macro-F1")
        ax.set_xticks(range(3), ["P1", "P3", "P5"])
        if setting == "Climate n500":
            ax.set_ylim(0.54, 0.74)
        else:
            ax.set_ylim(0.925, 0.98)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.22)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.88))

    saved_paths = []
    for stem in (filename_stem, "Figure4_prompt_comparison_corrected"):
        for ext in ("pdf", "svg", "png"):
            p = out / f"{stem}.{ext}"
            fig.savefig(p, dpi=300 if ext == "png" else None, bbox_inches="tight", facecolor="white")
            if stem == filename_stem:
                saved_paths.append(p)
    plt.close(fig)
    return saved_paths


def plot_coverage_and_rankings(
    risk_rows: Sequence[Dict[str, any]],
    out_dir: Union[str, Path],
    filename_stem: str = "RQ2_coverage_rankings"
) -> List[Path]:
    """Generate high-resolution multi-panel within-period risk vs. coverage and ranking figure."""
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


def plot_figure6_movement_vs_response(
    forensics_rows: Sequence[Dict[str, any]],
    delta_rows: Sequence[Dict[str, any]],
    out_dir: Union[str, Path],
    filename_stem: str = "Figure6_movement_vs_response"
) -> List[Path]:
    """Generate manuscript Figure 6: Movement vs. Response dual-panel heatmap."""
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

    rq5_map = {(r["dataset_setting"], r["target"]): r for r in forensics_rows}

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

    delta_map = {(r["dataset_setting"], r["target"], r["model"]): r for r in delta_rows}
    response_matrix = np.zeros((len(condition_order), len(model_order)))

    for i, (setting, target) in enumerate(condition_order):
        for j, model in enumerate(model_order):
            if (setting, target, model) in delta_map:
                row = delta_map[setting, target, model]
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
    for stem in (filename_stem, "Figure5_movement_vs_response"):
        for ext in ("svg", "pdf", "png"):
            p = out / f"{stem}.{ext}"
            fig.savefig(p, dpi=220, facecolor="white", bbox_inches="tight")
            if stem == filename_stem:
                saved_paths.append(p)
    plt.close(fig)

    return saved_paths


# Backwards-compatible alias
plot_figure5_movement_vs_response = plot_figure6_movement_vs_response
