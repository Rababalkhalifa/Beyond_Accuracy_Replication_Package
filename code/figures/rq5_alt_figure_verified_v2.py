from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# PATHS
# =========================================================

ROOT = Path("/mnt/c/Research/Beyond_Accuracy_Reproduction")

RQ5_FILE = ROOT / "RQ5_reaudit" / "gate2_rq5_summary.csv"
CLIMATE_F1 = ROOT / "climate_temporal_f1.csv"
YELP_F1 = ROOT / "yelp_temporal_f1.csv"

OUT = ROOT / "RQ5_reaudit" / "alt_figure_verified_v2"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 43, 44, 45, 46]


# =========================================================
# ORDER / LABELS
# =========================================================

CONDITION_ORDER = [
    ("Climate n500", "short"),
    ("Climate n500", "long"),
    ("Yelp n500", "short"),
    ("Yelp n500", "long"),
    ("Yelp n1000", "short"),
    ("Yelp n1000", "long"),
]

MODEL_ORDER = [
    "distilbert",
    "tfidf_lr",
    "phi3.5:latest",
    "qwen2.5:3b-instruct",
]

MODEL_NAMES = {
    "distilbert": "DistilBERT",
    "tfidf_lr": "TF-IDF LR",
    "phi3.5:latest": "Phi",
    "qwen2.5:3b-instruct": "Qwen",
}

MODEL_MARKERS = {
    "distilbert": "o",
    "tfidf_lr": "s",
    "phi3.5:latest": "^",
    "qwen2.5:3b-instruct": "D",
}

MODEL_COLORS = {
    "distilbert": "#0072B2",
    "tfidf_lr": "#E69F00",
    "phi3.5:latest": "#009E73",
    "qwen2.5:3b-instruct": "#D55E00",
}


# =========================================================
# GLOBAL FONT SETTINGS
# =========================================================

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10.5,
})


# =========================================================
# 1. LOAD VERIFIED RQ5 DATA
# =========================================================

rq5 = pd.read_csv(RQ5_FILE)

rq5["key"] = list(zip(
    rq5["dataset_setting"],
    rq5["target"],
))

order_map = {
    key: i
    for i, key in enumerate(CONDITION_ORDER)
}

rq5["order"] = rq5["key"].map(order_map)

assert rq5["order"].notna().all(), (
    "Unexpected RQ5 dataset/target condition."
)

rq5 = (
    rq5
    .sort_values("order")
    .reset_index(drop=True)
)

assert len(rq5) == 6


# =========================================================
# 2. LOAD RQ2 SOURCE DATA
# =========================================================

climate = pd.read_csv(CLIMATE_F1)
yelp = pd.read_csv(YELP_F1)

print("Climate source:", CLIMATE_F1)
print("Yelp source   :", YELP_F1)


# =========================================================
# 3. RECOMPUTE RQ2 DELTAS FROM SEED-LEVEL F1
# =========================================================
#
# Correct definition:
#
#     ΔMacro-F1 = target - within
#
# Positive = improvement
# Negative = degradation
#
# We deliberately ignore stored mean_delta_* columns.
# =========================================================

delta_rows = []


def add_model_deltas(setting, row):

    model = row["model"]

    for target in ["short", "long"]:

        seed_deltas = []

        for seed in SEEDS:

            within_f1 = float(
                row[f"seed_{seed}_within_f1"]
            )

            target_f1 = float(
                row[f"seed_{seed}_{target}_f1"]
            )

            seed_deltas.append(
                target_f1 - within_f1
            )

        delta_rows.append({
            "dataset_setting": setting,
            "target": target,
            "model": model,

            "delta_macro_f1_mean":
                float(np.mean(seed_deltas)),

            "delta_macro_f1_sd":
                float(np.std(seed_deltas, ddof=1)),

            **{
                f"seed_{seed}_delta":
                    seed_deltas[i]
                for i, seed in enumerate(SEEDS)
            }
        })


for _, row in climate.iterrows():
    add_model_deltas(
        "Climate n500",
        row,
    )

for _, row in yelp.iterrows():
    add_model_deltas(
        row["setting"],
        row,
    )


delta = pd.DataFrame(delta_rows)

delta["key"] = list(zip(
    delta["dataset_setting"],
    delta["target"],
))

delta["order"] = delta["key"].map(order_map)

delta = (
    delta
    .sort_values(["order", "model"])
    .reset_index(drop=True)
)

assert len(delta) == 24


# =========================================================
# 4. VERIFY AGAINST FINAL RQ2 TABLE
# =========================================================

expected = {
    ("Climate n500", "short", "distilbert"): -0.0258,
    ("Climate n500", "short", "tfidf_lr"): -0.0881,
    ("Climate n500", "short", "phi3.5:latest"): -0.0067,
    ("Climate n500", "short", "qwen2.5:3b-instruct"): -0.0285,

    ("Climate n500", "long", "distilbert"): -0.0170,
    ("Climate n500", "long", "tfidf_lr"): -0.0282,
    ("Climate n500", "long", "phi3.5:latest"): +0.0275,
    ("Climate n500", "long", "qwen2.5:3b-instruct"): +0.1019,

    ("Yelp n500", "short", "distilbert"): +0.0184,
    ("Yelp n500", "short", "tfidf_lr"): +0.0084,
    ("Yelp n500", "short", "phi3.5:latest"): +0.0198,
    ("Yelp n500", "short", "qwen2.5:3b-instruct"): +0.0134,

    ("Yelp n500", "long", "distilbert"): +0.0282,
    ("Yelp n500", "long", "tfidf_lr"): +0.0272,
    ("Yelp n500", "long", "phi3.5:latest"): +0.0254,
    ("Yelp n500", "long", "qwen2.5:3b-instruct"): +0.0154,

    ("Yelp n1000", "short", "distilbert"): +0.0190,
    ("Yelp n1000", "short", "tfidf_lr"): +0.0121,
    ("Yelp n1000", "short", "phi3.5:latest"): +0.0194,
    ("Yelp n1000", "short", "qwen2.5:3b-instruct"): +0.0139,

    ("Yelp n1000", "long", "distilbert"): +0.0275,
    ("Yelp n1000", "long", "tfidf_lr"): +0.0230,
    ("Yelp n1000", "long", "phi3.5:latest"): +0.0277,
    ("Yelp n1000", "long", "qwen2.5:3b-instruct"): +0.0172,
}


for _, r in delta.iterrows():

    key = (
        r["dataset_setting"],
        r["target"],
        r["model"],
    )

    got = r["delta_macro_f1_mean"]
    exp = expected[key]

    assert np.isclose(
        got,
        exp,
        atol=0.00015,
    ), (
        f"RQ2 mismatch for {key}: "
        f"got={got:.6f}, expected={exp:.4f}"
    )


print(
    "\nPASS: all 24 RQ2 deltas reproduced "
    "from seed-level F1."
)


# =========================================================
# 5. PREPARE PANEL A
# =========================================================

metric_cols = {
    "Novelty":
        "lexical_novelty_mean",

    "JSD":
        "js_divergence_mean",

    "ΔC100":
        "top100_mass_change_mean",

    "Δ length":
        "mean_length_change_mean",

    "Overlap loss":
        "overlap_loss_mean",

    "Marker\nasymmetry":
        "class_marker_asymmetry_mean",
}

metric_names = list(metric_cols.keys())

heat = []
annotations = []


def display_value(metric, value):

    if metric == "Δ length":

        if value > 0:
            return f"↑ {abs(value):.1f}"

        if value < 0:
            return f"↓ {abs(value):.1f}"

        return "0.0"

    if metric == "ΔC100":

        if value > 0:
            return f"↑ {abs(value):.3f}"

        if value < 0:
            return f"↓ {abs(value):.3f}"

        return "0.000"

    return f"{value:.3f}"


for _, row in rq5.iterrows():

    heat_row = []
    text_row = []

    for metric, col in metric_cols.items():

        value = float(row[col])

        all_values = rq5[col].astype(float)

        if metric in {
            "ΔC100",
            "Δ length",
        }:

            magnitude = abs(value)
            maximum = all_values.abs().max()

        else:

            magnitude = value
            maximum = all_values.max()

        norm = (
            magnitude / maximum
            if maximum > 0
            else 0.0
        )

        heat_row.append(norm)

        text_row.append(
            display_value(
                metric,
                value,
            )
        )

    heat.append(heat_row)
    annotations.append(text_row)


heat = np.array(heat)


row_labels = [
    f"{dataset}\n{target.capitalize()}"
    for dataset, target in CONDITION_ORDER
]


# =========================================================
# 6. CREATE FIGURE
# =========================================================

fig, (ax1, ax2) = plt.subplots(
    1,
    2,
    figsize=(18, 9),
    gridspec_kw={
        "width_ratios": [1.35, 1.15],
        "wspace": 0.30,
    },
)


# =========================================================
# PANEL A — DATASET MOVEMENT
# =========================================================

im = ax1.imshow(
    heat,
    aspect="auto",
    cmap="YlOrRd",
    vmin=0,
    vmax=1,
)


ax1.set_title(
    "A. Dataset-side temporal movement",
    fontweight="bold",
    pad=16,
)


ax1.set_xticks(
    np.arange(len(metric_names))
)

ax1.set_xticklabels(
    metric_names,
    rotation=0,
    ha="center",
    fontweight="bold",
)


ax1.set_yticks(
    np.arange(len(row_labels))
)

ax1.set_yticklabels(
    row_labels,
    fontweight="bold",
)


# ---------------------------------------------------------
# Cell borders
# ---------------------------------------------------------

ax1.set_xticks(
    np.arange(-0.5, len(metric_names), 1),
    minor=True,
)

ax1.set_yticks(
    np.arange(-0.5, len(row_labels), 1),
    minor=True,
)

ax1.grid(
    which="minor",
    color="white",
    linewidth=2,
)

ax1.tick_params(
    which="minor",
    bottom=False,
    left=False,
)


# ---------------------------------------------------------
# Large annotations
# ---------------------------------------------------------

for i in range(heat.shape[0]):

    for j in range(heat.shape[1]):

        strength = heat[i, j]

        text_color = (
            "white"
            if strength >= 0.62
            else "black"
        )

        ax1.text(
            j,
            i,
            annotations[i][j],
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color=text_color,
        )


# ---------------------------------------------------------
# Colorbar
# ---------------------------------------------------------

cbar = fig.colorbar(
    im,
    ax=ax1,
    fraction=0.045,
    pad=0.035,
)

cbar.set_label(
    "Relative magnitude within each metric",
    fontsize=11,
    fontweight="bold",
)

cbar.ax.tick_params(
    labelsize=10,
)


# =========================================================
# PANEL B — MODEL RESPONSE
# =========================================================

y_base = np.arange(
    len(CONDITION_ORDER)
)

# Small vertical offsets so four models do not sit directly
# on top of each other.
offsets = {
    "distilbert": -0.18,
    "tfidf_lr": -0.06,
    "phi3.5:latest": +0.06,
    "qwen2.5:3b-instruct": +0.18,
}


# ---------------------------------------------------------
# Background guide rows
# ---------------------------------------------------------

for y in y_base:

    ax2.axhline(
        y,
        color="#DDDDDD",
        linewidth=0.8,
        zorder=0,
    )


# ---------------------------------------------------------
# Zero line
# ---------------------------------------------------------

ax2.axvline(
    0,
    color="black",
    linewidth=1.6,
    linestyle="--",
    zorder=1,
)


# ---------------------------------------------------------
# Plot each model
# ---------------------------------------------------------

for model in MODEL_ORDER:

    subset = (
        delta[
            delta["model"] == model
        ]
        .sort_values("order")
    )

    x = subset[
        "delta_macro_f1_mean"
    ].to_numpy()

    xerr = subset[
        "delta_macro_f1_sd"
    ].to_numpy()

    y = (
        y_base
        + offsets[model]
    )

    ax2.errorbar(
        x,
        y,
        xerr=xerr,
        fmt=MODEL_MARKERS[model],
        markersize=9,
        markeredgewidth=0.8,
        capsize=4,
        elinewidth=1.4,
        color=MODEL_COLORS[model],
        label=MODEL_NAMES[model],
        zorder=3,
    )


    # -----------------------------------------------------
    # Point values
    # -----------------------------------------------------

    for xi, yi in zip(x, y):

        if xi >= 0:
            dx = 0.004
            ha = "left"

        else:
            dx = -0.004
            ha = "right"

        ax2.text(
            xi + dx,
            yi,
            f"{xi:+.3f}",
            va="center",
            ha=ha,
            fontsize=9.5,
            fontweight="bold",
            color=MODEL_COLORS[model],
        )


# ---------------------------------------------------------
# Axis labels
# ---------------------------------------------------------

ax2.set_yticks(
    y_base
)

ax2.set_yticklabels(
    row_labels,
    fontweight="bold",
)

ax2.invert_yaxis()


ax2.set_xlabel(
    "ΔMacro-F1  (target − within)",
    fontweight="bold",
)

ax2.set_title(
    "B. Model response",
    fontweight="bold",
    pad=16,
)


# ---------------------------------------------------------
# Improvement/degradation labels
# ---------------------------------------------------------

ax2.text(
    0.02,
    1.025,
    "← lower Macro-F1",
    transform=ax2.transAxes,
    ha="left",
    va="bottom",
    fontsize=10,
    fontweight="bold",
)

ax2.text(
    0.98,
    1.025,
    "higher Macro-F1 →",
    transform=ax2.transAxes,
    ha="right",
    va="bottom",
    fontsize=10,
    fontweight="bold",
)


# ---------------------------------------------------------
# Symmetric x-axis
# ---------------------------------------------------------

largest = max(
    abs(
        delta[
            "delta_macro_f1_mean"
        ].min()
    ),
    abs(
        delta[
            "delta_macro_f1_mean"
        ].max()
    ),
)

limit = max(
    0.12,
    largest + 0.02,
)

ax2.set_xlim(
    -limit,
    limit,
)


ax2.grid(
    axis="x",
    alpha=0.18,
)


# ---------------------------------------------------------
# Legend above plot
# ---------------------------------------------------------

ax2.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.11),
    ncol=4,
    frameon=False,
)


# =========================================================
# CLEAN SPINES
# =========================================================

for ax in [ax1, ax2]:

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# =========================================================
# OVERALL TITLE
# =========================================================

fig.suptitle(
    "Temporal movement profiles and corresponding model-performance changes",
    fontsize=17,
    fontweight="bold",
    y=0.98,
)


fig.subplots_adjust(
    top=0.88,
    bottom=0.16,
    left=0.10,
    right=0.97,
)


# =========================================================
# SAVE
# =========================================================

for ext in [
    "png",
    "pdf",
    "svg",
]:

    path = (
        OUT
        / f"Figure5_movement_vs_response_v2.{ext}"
    )

    fig.savefig(
        path,
        dpi=350 if ext == "png" else None,
        bbox_inches="tight",
    )


plt.close(fig)


# =========================================================
# 7. SAVE EXACT AUDIT DATA
# =========================================================

delta.to_csv(
    OUT
    / "Figure5_RQ2_recomputed_deltas.csv",
    index=False,
)

rq5.to_csv(
    OUT
    / "Figure5_RQ5_verified_inputs.csv",
    index=False,
)


# =========================================================
# 8. CAPTION
# =========================================================

caption = """
Figure 5: Temporal movement profiles and corresponding
model-performance changes. Panel A summarizes six
dataset-side temporal-forensics measures. Cell shading
represents each condition's magnitude relative to the
largest observed value of that metric across the six
dataset-target conditions; annotations report the original
metric values, and arrows retain the direction of signed
top-term-concentration and document-length changes.
Panel B shows mean within-seed change in Macro-F1
(target minus within-period reference), with error bars
representing sample standard deviation across seeds 42–46.
Positive values indicate higher Macro-F1 on the sampled
target and negative values indicate lower Macro-F1.
The juxtaposition is descriptive and does not imply that
any individual forensic measure causes the observed
performance change.
""".strip()


(
    OUT
    / "Figure5_caption_v2.txt"
).write_text(
    caption,
    encoding="utf-8",
)


# =========================================================
# 9. FINAL REPORT
# =========================================================

print("\n" + "=" * 90)
print("FINAL FIGURE 5 V2")
print("=" * 90)

print(
    "\nPASS: all 24 RQ2 deltas were reproduced "
    "from seed-level F1."
)

print("\nSaved to:")
print(OUT)

print("\nFiles:")

for p in sorted(
    OUT.glob("Figure5_*")
):
    print(" ", p)
