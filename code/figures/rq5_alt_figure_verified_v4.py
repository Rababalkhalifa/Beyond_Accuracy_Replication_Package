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
    / "alt_figure_verified_v4"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

SEEDS = [42, 43, 44, 45, 46]


# =========================================================
# ORDER
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


# =========================================================
# STYLE
# =========================================================

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})


# =========================================================
# 1. LOAD VERIFIED RQ5
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
    for i, key
    in enumerate(CONDITION_ORDER)
}

rq5["order"] = (
    rq5["key"]
    .map(condition_order_map)
)

assert rq5["order"].notna().all(), (
    "Unexpected RQ5 condition."
)

rq5 = (
    rq5
    .sort_values("order")
    .reset_index(drop=True)
)

assert len(rq5) == 6


# =========================================================
# 2. LOAD RQ2 SOURCE FILES
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
# 3. RECOMPUTE RQ2 DELTAS FROM SEED F1
# =========================================================
#
# Correct definition:
#
# ΔMacro-F1 = target - within
#
# positive = improved target-period score
# negative = lower target-period score
#
# Historical stored delta columns are ignored.
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
    .map(
        condition_order_map
    )
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
# 4. VERIFY RQ2 AGAINST TABLE 9
# =========================================================

expected = {

    ("Climate n500", "short", "distilbert"):
        -0.0258,

    ("Climate n500", "short", "tfidf_lr"):
        -0.0881,

    ("Climate n500", "short", "phi3.5:latest"):
        -0.0067,

    ("Climate n500", "short", "qwen2.5:3b-instruct"):
        -0.0285,

    ("Climate n500", "long", "distilbert"):
        -0.0170,

    ("Climate n500", "long", "tfidf_lr"):
        -0.0282,

    ("Climate n500", "long", "phi3.5:latest"):
        +0.0275,

    ("Climate n500", "long", "qwen2.5:3b-instruct"):
        +0.1019,

    ("Yelp n500", "short", "distilbert"):
        +0.0184,

    ("Yelp n500", "short", "tfidf_lr"):
        +0.0084,

    ("Yelp n500", "short", "phi3.5:latest"):
        +0.0198,

    ("Yelp n500", "short", "qwen2.5:3b-instruct"):
        +0.0134,

    ("Yelp n500", "long", "distilbert"):
        +0.0282,

    ("Yelp n500", "long", "tfidf_lr"):
        +0.0272,

    ("Yelp n500", "long", "phi3.5:latest"):
        +0.0254,

    ("Yelp n500", "long", "qwen2.5:3b-instruct"):
        +0.0154,

    ("Yelp n1000", "short", "distilbert"):
        +0.0190,

    ("Yelp n1000", "short", "tfidf_lr"):
        +0.0121,

    ("Yelp n1000", "short", "phi3.5:latest"):
        +0.0194,

    ("Yelp n1000", "short", "qwen2.5:3b-instruct"):
        +0.0139,

    ("Yelp n1000", "long", "distilbert"):
        +0.0275,

    ("Yelp n1000", "long", "tfidf_lr"):
        +0.0230,

    ("Yelp n1000", "long", "phi3.5:latest"):
        +0.0277,

    ("Yelp n1000", "long", "qwen2.5:3b-instruct"):
        +0.0172,
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
        f"Mismatch {key}: "
        f"{observed:.6f} "
        f"vs {expected_value:.4f}"
    )


print(
    "\nPASS: all 24 RQ2 deltas reproduced "
    "from seed-level F1."
)


# =========================================================
# 5. PANEL A — MOVEMENT HEATMAP
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


def format_forensic_value(
    metric,
    value,
):

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


heat_a = []
annot_a = []


for _, row in rq5.iterrows():

    heat_row = []
    text_row = []

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
            else 0
        )

        heat_row.append(
            relative
        )

        text_row.append(
            format_forensic_value(
                metric,
                value,
            )
        )

    heat_a.append(
        heat_row
    )

    annot_a.append(
        text_row
    )


heat_a = np.array(
    heat_a
)


row_labels = [
    f"{dataset}\n{target.capitalize()}"
    for dataset, target
    in CONDITION_ORDER
]


# =========================================================
# 6. PANEL B — RESPONSE MATRIX
# =========================================================

response_matrix = np.zeros(
    (
        len(CONDITION_ORDER),
        len(MODEL_ORDER),
    )
)


response_sd_matrix = np.zeros(
    (
        len(CONDITION_ORDER),
        len(MODEL_ORDER),
    )
)


for i, (
    setting,
    target,
) in enumerate(
    CONDITION_ORDER
):

    for j, model in enumerate(
        MODEL_ORDER
    ):

        subset = delta[
            (
                delta["dataset_setting"]
                == setting
            )
            &
            (
                delta["target"]
                == target
            )
            &
            (
                delta["model"]
                == model
            )
        ]

        assert len(subset) == 1

        response_matrix[
            i,
            j,
        ] = float(
            subset.iloc[0][
                "delta_macro_f1_mean"
            ]
        )

        response_sd_matrix[
            i,
            j,
        ] = float(
            subset.iloc[0][
                "delta_macro_f1_sd"
            ]
        )


# =========================================================
# 7. CREATE FIGURE
# =========================================================

fig, (
    ax1,
    ax2,
) = plt.subplots(
    1,
    2,
    figsize=(17, 9),
    gridspec_kw={
        "width_ratios":
            [1.25, 0.95],

        "wspace":
            0.28,
    },
)


# =========================================================
# PANEL A
# =========================================================

im1 = ax1.imshow(
    heat_a,
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
        len(metric_names)
    )
)

ax1.set_xticklabels(
    metric_names,
    fontweight="bold",
)


ax1.set_yticks(
    np.arange(
        len(row_labels)
    )
)

ax1.set_yticklabels(
    row_labels,
    fontweight="bold",
)


# Cell borders
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


# Numbers
for i in range(
    heat_a.shape[0]
):

    for j in range(
        heat_a.shape[1]
    ):

        strength = (
            heat_a[
                i,
                j,
            ]
        )

        color = (
            "white"
            if strength >= 0.62
            else "black"
        )

        ax1.text(
            j,
            i,
            annot_a[i][j],
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color=color,
        )


# Colorbar A
cbar1 = fig.colorbar(
    im1,
    ax=ax1,
    fraction=0.045,
    pad=0.035,
)

cbar1.set_label(
    "Relative magnitude within each metric",
    fontsize=11,
    fontweight="bold",
)


# =========================================================
# PANEL B
# =========================================================

response_limit = float(
    np.max(
        np.abs(
            response_matrix
        )
    )
)


im2 = ax2.imshow(
    response_matrix,
    aspect="auto",
    cmap="RdYlGn",
    vmin=-response_limit,
    vmax=response_limit,
)


ax2.set_title(
    "B. Model response",
    fontweight="bold",
    fontsize=15,
    pad=16,
)


ax2.set_xticks(
    np.arange(
        len(MODEL_ORDER)
    )
)

ax2.set_xticklabels(
    [
        MODEL_NAMES[m]
        for m in MODEL_ORDER
    ],
    fontweight="bold",
)


ax2.set_yticks(
    np.arange(
        len(row_labels)
    )
)

ax2.set_yticklabels(
    row_labels,
    fontweight="bold",
)


# Cell borders
ax2.set_xticks(
    np.arange(
        -0.5,
        len(MODEL_ORDER),
        1,
    ),
    minor=True,
)

ax2.set_yticks(
    np.arange(
        -0.5,
        len(row_labels),
        1,
    ),
    minor=True,
)

ax2.grid(
    which="minor",
    color="white",
    linewidth=2,
)

ax2.tick_params(
    which="minor",
    bottom=False,
    left=False,
)


# =========================================================
# PANEL B CELL NUMBERS
# =========================================================

for i in range(
    response_matrix.shape[0]
):

    for j in range(
        response_matrix.shape[1]
    ):

        value = (
            response_matrix[
                i,
                j,
            ]
        )

        relative = (
            abs(value)
            / response_limit
            if response_limit > 0
            else 0
        )

        text_color = (
            "white"
            if relative >= 0.60
            else "black"
        )

        if value > 0:

            cell_text = (
                f"↑ {value:.3f}"
            )

        elif value < 0:

            cell_text = (
                f"↓ {abs(value):.3f}"
            )

        else:

            cell_text = (
                "0.000"
            )

        ax2.text(
            j,
            i,
            cell_text,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color=text_color,
        )


# =========================================================
# PANEL B DIRECTION LABELS
# =========================================================

ax2.text(
    0.0,
    1.025,
    "↓ lower Macro-F1",
    transform=ax2.transAxes,
    ha="left",
    va="bottom",
    fontsize=10.5,
    fontweight="bold",
)


ax2.text(
    1.0,
    1.025,
    "↑ higher Macro-F1",
    transform=ax2.transAxes,
    ha="right",
    va="bottom",
    fontsize=10.5,
    fontweight="bold",
)


# =========================================================
# COLORBAR B
# =========================================================

cbar2 = fig.colorbar(
    im2,
    ax=ax2,
    fraction=0.050,
    pad=0.04,
)


cbar2.set_label(
    "ΔMacro-F1 (target − within)",
    fontsize=11,
    fontweight="bold",
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
    bottom=0.12,
    left=0.10,
    right=0.97,
)


# =========================================================
# 8. SAVE FIGURE
# =========================================================

for ext in [
    "png",
    "pdf",
    "svg",
]:

    path = (
        OUT
        / (
            "Figure5_"
            "movement_vs_response_"
            f"v4.{ext}"
        )
    )

    fig.savefig(
        path,

        dpi=(
            350
            if ext == "png"
            else None
        ),

        bbox_inches="tight",
    )


plt.close(
    fig
)


# =========================================================
# 9. SAVE AUDIT DATA
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
# 10. SAVE RESPONSE MATRIX
# =========================================================

response_export = pd.DataFrame(
    response_matrix,
    columns=[
        MODEL_NAMES[m]
        for m in MODEL_ORDER
    ],
)

response_export.insert(
    0,
    "condition",
    [
        f"{d} {t}"
        for d, t
        in CONDITION_ORDER
    ],
)

response_export.to_csv(
    OUT
    / "Figure5_response_matrix.csv",
    index=False,
)


# =========================================================
# 11. CAPTION
# =========================================================

caption = """
Figure 5: Temporal movement profiles and corresponding
model-performance changes. Panel A summarizes six
dataset-side temporal-forensics measures. Cell shading
represents each condition's magnitude relative to the
largest observed value of the same metric across the six
dataset-target conditions; annotations report the original
metric values, and arrows retain the direction of signed
top-term-concentration and document-length changes.
Panel B shows the mean within-seed change in Macro-F1
(target minus within-period reference) for each model.
Green cells indicate higher target-period Macro-F1 and
red cells indicate lower target-period Macro-F1; cell
annotations report the corresponding mean temporal
change. Temporal changes are calculated within each of
seeds 42--46 before aggregation. The juxtaposition is
descriptive and does not imply that any individual
forensic measure causes the observed performance change.
""".strip()


(
    OUT
    / "Figure5_caption_v4.txt"
).write_text(
    caption,
    encoding="utf-8",
)


# =========================================================
# 12. REPORT RESPONSE MATRIX
# =========================================================

print(
    "\n"
    + "=" * 100
)

print(
    "FIGURE 5 V4 — MODEL RESPONSE MATRIX"
)

print(
    "=" * 100
)


print(
    response_export.to_string(
        index=False,
        float_format=(
            lambda x:
                f"{x:+.4f}"
        ),
    )
)


# =========================================================
# FINAL REPORT
# =========================================================

print(
    "\n"
    + "=" * 100
)

print(
    "FIGURE 5 V4 COMPLETE"
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
