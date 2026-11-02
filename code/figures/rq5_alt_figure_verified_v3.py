from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# PATHS
# =========================================================

ROOT = Path("/mnt/c/Research/Beyond_Accuracy_Reproduction")

RQ5_FILE = (
    ROOT
    / "RQ5_reaudit"
    / "gate2_rq5_summary.csv"
)

CLIMATE_F1 = (
    ROOT
    / "climate_temporal_f1.csv"
)

YELP_F1 = (
    ROOT
    / "yelp_temporal_f1.csv"
)

OUT = (
    ROOT
    / "RQ5_reaudit"
    / "alt_figure_verified_v3"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

SEEDS = [42, 43, 44, 45, 46]


# =========================================================
# CONDITION / MODEL ORDER
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
# GLOBAL FIGURE STYLE
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
# 1. LOAD VERIFIED RQ5 SUMMARY
# =========================================================

rq5 = pd.read_csv(
    RQ5_FILE
)

rq5["key"] = list(
    zip(
        rq5["dataset_setting"],
        rq5["target"],
    )
)

condition_order_map = {
    key: i
    for i, key in enumerate(
        CONDITION_ORDER
    )
}

rq5["order"] = (
    rq5["key"]
    .map(condition_order_map)
)

assert rq5["order"].notna().all(), (
    "Unexpected dataset-target condition "
    "in RQ5 summary."
)

rq5 = (
    rq5
    .sort_values("order")
    .reset_index(drop=True)
)

assert len(rq5) == 6


# =========================================================
# 2. LOAD VERIFIED RQ2 SOURCE F1 FILES
# =========================================================

climate = pd.read_csv(
    CLIMATE_F1
)

yelp = pd.read_csv(
    YELP_F1
)

print(
    "Climate source:",
    CLIMATE_F1,
)

print(
    "Yelp source   :",
    YELP_F1,
)


# =========================================================
# 3. RECOMPUTE RQ2 DELTAS
# =========================================================
#
# Correct definition:
#
# ΔMacro-F1 = target - within
#
# Positive = higher target-period Macro-F1
# Negative = lower target-period Macro-F1
#
# Stored historical mean_delta_* columns are NOT used.
# =========================================================

delta_rows = []


def add_model_deltas(
    setting,
    row,
):

    model = row["model"]

    for target in [
        "short",
        "long",
    ]:

        seed_deltas = []

        for seed in SEEDS:

            within_f1 = float(
                row[
                    f"seed_{seed}_within_f1"
                ]
            )

            target_f1 = float(
                row[
                    f"seed_{seed}_{target}_f1"
                ]
            )

            seed_deltas.append(
                target_f1
                - within_f1
            )

        record = {
            "dataset_setting":
                setting,

            "target":
                target,

            "model":
                model,

            "delta_macro_f1_mean":
                float(
                    np.mean(
                        seed_deltas
                    )
                ),

            "delta_macro_f1_sd":
                float(
                    np.std(
                        seed_deltas,
                        ddof=1,
                    )
                ),
        }

        for i, seed in enumerate(
            SEEDS
        ):

            record[
                f"seed_{seed}_delta"
            ] = seed_deltas[i]

        delta_rows.append(
            record
        )


# Climate
for _, row in climate.iterrows():

    add_model_deltas(
        "Climate n500",
        row,
    )


# Yelp
for _, row in yelp.iterrows():

    add_model_deltas(
        row["setting"],
        row,
    )


delta = pd.DataFrame(
    delta_rows
)

delta["key"] = list(
    zip(
        delta["dataset_setting"],
        delta["target"],
    )
)

delta["order"] = (
    delta["key"]
    .map(condition_order_map)
)

delta = (
    delta
    .sort_values(
        [
            "order",
            "model",
        ]
    )
    .reset_index(drop=True)
)

assert len(delta) == 24


# =========================================================
# 4. VERIFY AGAINST FINAL RQ2 TABLE
# =========================================================

expected = {

    # Climate short
    (
        "Climate n500",
        "short",
        "distilbert",
    ): -0.0258,

    (
        "Climate n500",
        "short",
        "tfidf_lr",
    ): -0.0881,

    (
        "Climate n500",
        "short",
        "phi3.5:latest",
    ): -0.0067,

    (
        "Climate n500",
        "short",
        "qwen2.5:3b-instruct",
    ): -0.0285,

    # Climate long
    (
        "Climate n500",
        "long",
        "distilbert",
    ): -0.0170,

    (
        "Climate n500",
        "long",
        "tfidf_lr",
    ): -0.0282,

    (
        "Climate n500",
        "long",
        "phi3.5:latest",
    ): +0.0275,

    (
        "Climate n500",
        "long",
        "qwen2.5:3b-instruct",
    ): +0.1019,

    # Yelp n500 short
    (
        "Yelp n500",
        "short",
        "distilbert",
    ): +0.0184,

    (
        "Yelp n500",
        "short",
        "tfidf_lr",
    ): +0.0084,

    (
        "Yelp n500",
        "short",
        "phi3.5:latest",
    ): +0.0198,

    (
        "Yelp n500",
        "short",
        "qwen2.5:3b-instruct",
    ): +0.0134,

    # Yelp n500 long
    (
        "Yelp n500",
        "long",
        "distilbert",
    ): +0.0282,

    (
        "Yelp n500",
        "long",
        "tfidf_lr",
    ): +0.0272,

    (
        "Yelp n500",
        "long",
        "phi3.5:latest",
    ): +0.0254,

    (
        "Yelp n500",
        "long",
        "qwen2.5:3b-instruct",
    ): +0.0154,

    # Yelp n1000 short
    (
        "Yelp n1000",
        "short",
        "distilbert",
    ): +0.0190,

    (
        "Yelp n1000",
        "short",
        "tfidf_lr",
    ): +0.0121,

    (
        "Yelp n1000",
        "short",
        "phi3.5:latest",
    ): +0.0194,

    (
        "Yelp n1000",
        "short",
        "qwen2.5:3b-instruct",
    ): +0.0139,

    # Yelp n1000 long
    (
        "Yelp n1000",
        "long",
        "distilbert",
    ): +0.0275,

    (
        "Yelp n1000",
        "long",
        "tfidf_lr",
    ): +0.0230,

    (
        "Yelp n1000",
        "long",
        "phi3.5:latest",
    ): +0.0277,

    (
        "Yelp n1000",
        "long",
        "qwen2.5:3b-instruct",
    ): +0.0172,
}


for _, row in delta.iterrows():

    key = (
        row["dataset_setting"],
        row["target"],
        row["model"],
    )

    observed = (
        row[
            "delta_macro_f1_mean"
        ]
    )

    expected_value = (
        expected[key]
    )

    assert np.isclose(
        observed,
        expected_value,
        atol=0.00015,
    ), (
        f"RQ2 mismatch {key}: "
        f"got {observed:.6f}, "
        f"expected {expected_value:.4f}"
    )


print(
    "\nPASS: all 24 RQ2 deltas "
    "reproduced from seed-level F1."
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


metric_names = list(
    metric_cols.keys()
)


def display_value(
    metric,
    value,
):

    # Document length
    if metric == "Δ length":

        if value > 0:
            return (
                f"↑ {abs(value):.1f}"
            )

        if value < 0:
            return (
                f"↓ {abs(value):.1f}"
            )

        return "0.0"

    # Top-100 concentration
    if metric == "ΔC100":

        if value > 0:
            return (
                f"↑ {abs(value):.3f}"
            )

        if value < 0:
            return (
                f"↓ {abs(value):.3f}"
            )

        return "0.000"

    # Other measures
    return f"{value:.3f}"


heat = []
annotations = []


for _, row in rq5.iterrows():

    heat_row = []
    annotation_row = []

    for metric, col in (
        metric_cols.items()
    ):

        value = float(
            row[col]
        )

        all_values = (
            rq5[col]
            .astype(float)
        )

        if metric in {
            "ΔC100",
            "Δ length",
        }:

            magnitude = abs(
                value
            )

            maximum = (
                all_values
                .abs()
                .max()
            )

        else:

            magnitude = value

            maximum = (
                all_values
                .max()
            )

        relative = (
            magnitude / maximum
            if maximum > 0
            else 0.0
        )

        heat_row.append(
            relative
        )

        annotation_row.append(
            display_value(
                metric,
                value,
            )
        )

    heat.append(
        heat_row
    )

    annotations.append(
        annotation_row
    )


heat = np.array(
    heat
)


row_labels = [
    f"{dataset}\n{target.capitalize()}"
    for dataset, target
    in CONDITION_ORDER
]


# =========================================================
# 6. CREATE FIGURE
# =========================================================

fig, (
    ax1,
    ax2,
) = plt.subplots(
    1,
    2,
    figsize=(18, 9.5),
    gridspec_kw={
        "width_ratios":
            [1.35, 1.20],

        "wspace":
            0.28,
    },
)


# =========================================================
# PANEL A — TEMPORAL MOVEMENT
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
    fontsize=15,
    pad=16,
)


ax1.set_xticks(
    np.arange(
        len(
            metric_names
        )
    )
)

ax1.set_xticklabels(
    metric_names,
    rotation=0,
    ha="center",
    fontweight="bold",
)


ax1.set_yticks(
    np.arange(
        len(
            row_labels
        )
    )
)

ax1.set_yticklabels(
    row_labels,
    fontweight="bold",
)


# =========================================================
# CELL BORDERS
# =========================================================

ax1.set_xticks(
    np.arange(
        -0.5,
        len(metric_names),
        1,
    ),
    minor=True,
)

ax1.set_yticks(
    np.arange(
        -0.5,
        len(row_labels),
        1,
    ),
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


# =========================================================
# CELL ANNOTATIONS
# =========================================================

for i in range(
    heat.shape[0]
):

    for j in range(
        heat.shape[1]
    ):

        relative = (
            heat[i, j]
        )

        text_color = (
            "white"
            if relative >= 0.62
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


# =========================================================
# COLOR BAR
# =========================================================

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
    len(
        CONDITION_ORDER
    )
)


# More spacing between the models
vertical_offsets = {
    "distilbert": -0.27,
    "tfidf_lr": -0.09,
    "phi3.5:latest": +0.09,
    "qwen2.5:3b-instruct": +0.27,
}


# =========================================================
# GUIDE LINES
# =========================================================

for y in y_base:

    ax2.axhline(
        y,
        color="#D5D5D5",
        linewidth=0.9,
        zorder=0,
    )


# =========================================================
# ZERO LINE
# =========================================================

ax2.axvline(
    0,
    color="black",
    linewidth=1.5,
    linestyle="--",
    zorder=1,
)


# =========================================================
# MODELS
# =========================================================

for model in MODEL_ORDER:

    subset = (
        delta[
            delta["model"]
            == model
        ]
        .sort_values(
            "order"
        )
    )

    x = (
        subset[
            "delta_macro_f1_mean"
        ]
        .to_numpy()
    )

    xerr = (
        subset[
            "delta_macro_f1_sd"
        ]
        .to_numpy()
    )

    y = (
        y_base
        + vertical_offsets[
            model
        ]
    )


    # -----------------------------------------------------
    # Error bars + markers
    # -----------------------------------------------------

    ax2.errorbar(
        x,
        y,
        xerr=xerr,

        fmt=(
            MODEL_MARKERS[
                model
            ]
        ),

        markersize=9.5,

        markeredgewidth=0.8,

        capsize=4,

        elinewidth=1.4,

        color=(
            MODEL_COLORS[
                model
            ]
        ),

        label=(
            MODEL_NAMES[
                model
            ]
        ),

        zorder=3,
    )


    # -----------------------------------------------------
    # Value labels ABOVE points
    # -----------------------------------------------------
    #
    # Because the y-axis is inverted, subtracting moves
    # text visually upward.
    # -----------------------------------------------------

    label_y_shift = -0.105

    for xi, yi in zip(
        x,
        y,
    ):

        ax2.text(
            xi,
            yi + label_y_shift,

            f"{xi:+.3f}",

            ha="center",
            va="bottom",

            fontsize=10,

            fontweight="bold",

            color=(
                MODEL_COLORS[
                    model
                ]
            ),

            zorder=5,
        )


# =========================================================
# PANEL B AXES
# =========================================================

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
    fontsize=12,
    fontweight="bold",
)


ax2.set_title(
    "B. Model response",
    fontsize=15,
    fontweight="bold",
    pad=16,
)


# =========================================================
# DIRECTION LABELS
# =========================================================

ax2.text(
    0.02,
    1.025,

    "← lower Macro-F1",

    transform=ax2.transAxes,

    ha="left",
    va="bottom",

    fontsize=10.5,
    fontweight="bold",
)


ax2.text(
    0.98,
    1.025,

    "higher Macro-F1 →",

    transform=ax2.transAxes,

    ha="right",
    va="bottom",

    fontsize=10.5,
    fontweight="bold",
)


# =========================================================
# SYMMETRIC X AXIS
# =========================================================

largest_delta = max(
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

x_limit = max(
    0.12,
    largest_delta + 0.025,
)

ax2.set_xlim(
    -x_limit,
    x_limit,
)


ax2.grid(
    axis="x",
    alpha=0.17,
)


# =========================================================
# LEGEND
# =========================================================

ax2.legend(
    loc="upper center",
    bbox_to_anchor=(
        0.5,
        -0.11,
    ),
    ncol=4,
    frameon=False,
    columnspacing=1.6,
    handletextpad=0.5,
)


# =========================================================
# CLEAN SPINES
# =========================================================

for ax in [
    ax1,
    ax2,
]:

    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)


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
# 7. SAVE FIGURE
# =========================================================

for ext in [
    "png",
    "pdf",
    "svg",
]:

    output_path = (
        OUT
        / (
            "Figure5_"
            "movement_vs_response_"
            f"v3.{ext}"
        )
    )

    fig.savefig(
        output_path,

        dpi=(
            350
            if ext == "png"
            else None
        ),

        bbox_inches="tight",
    )


plt.close(fig)


# =========================================================
# 8. SAVE AUDIT INPUTS
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
# 9. CAPTION
# =========================================================

caption = """
Figure 5: Temporal movement profiles and corresponding
model-performance changes. Panel A summarizes six
dataset-side temporal-forensics measures. Cell shading
represents each condition's magnitude relative to the
largest observed value of that metric across the six
dataset-target conditions; annotations report the original
metric values, while arrows retain the direction of signed
top-term-concentration and document-length changes.
Panel B shows mean within-seed change in Macro-F1
(target minus within-period reference), with error bars
showing sample standard deviation across seeds 42–46.
Positive values indicate higher Macro-F1 on the sampled
target and negative values indicate lower Macro-F1.
The juxtaposition is descriptive and does not imply that
any individual forensic measure causes the observed
performance change.
""".strip()


(
    OUT
    / "Figure5_caption_v3.txt"
).write_text(
    caption,
    encoding="utf-8",
)


# =========================================================
# 10. REPORT DELTAS
# =========================================================

print(
    "\n"
    + "=" * 100
)

print(
    "FIGURE 5 V3 — RQ2 DELTAS"
)

print(
    "=" * 100
)


print(
    delta[
        [
            "dataset_setting",
            "target",
            "model",
            "delta_macro_f1_mean",
            "delta_macro_f1_sd",
        ]
    ]
    .to_string(
        index=False,
        float_format=(
            lambda x:
                f"{x:.6f}"
        ),
    )
)


# =========================================================
# FINAL
# =========================================================

print(
    "\n"
    + "=" * 100
)

print(
    "FIGURE 5 V3 COMPLETE"
)

print(
    "=" * 100
)

print(
    "\nPASS: Figure 5 uses verified "
    "RQ5 data and independently "
    "recomputed RQ2 deltas."
)

print(
    "\nSaved to:"
)

print(
    OUT
)

print(
    "\nFiles:"
)

for path in sorted(
    OUT.glob(
        "Figure5_*"
    )
):

    print(
        " ",
        path,
    )
