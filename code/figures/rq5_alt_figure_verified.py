from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


ROOT = Path("/mnt/c/Research/Beyond_Accuracy_Reproduction")

RQ5_FILE = ROOT / "RQ5_reaudit" / "gate2_rq5_summary.csv"
CLIMATE_F1 = ROOT / "climate_temporal_f1.csv"
YELP_F1 = ROOT / "yelp_temporal_f1.csv"

OUT = ROOT / "RQ5_reaudit" / "alt_figure_verified"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 43, 44, 45, 46]

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
# 1. LOAD VERIFIED RQ5
# =========================================================

rq5 = pd.read_csv(RQ5_FILE)

rq5["key"] = list(zip(
    rq5["dataset_setting"],
    rq5["target"],
))

order_map = {
    key: i for i, key in enumerate(CONDITION_ORDER)
}

rq5["order"] = rq5["key"].map(order_map)
rq5 = rq5.sort_values("order").reset_index(drop=True)

assert len(rq5) == 6


# =========================================================
# 2. LOAD RQ2 SOURCE F1 FILES
# =========================================================

climate = pd.read_csv(CLIMATE_F1)
yelp = pd.read_csv(YELP_F1)

print("Climate source:", CLIMATE_F1)
print("Yelp source   :", YELP_F1)


# =========================================================
# 3. RECOMPUTE DELTAS FROM SEED-LEVEL F1
# =========================================================
#
# DO NOT use mean_delta_short / mean_delta_long.
#
# Those columns contain the historical old-sign definition.
#
# Correct definition:
#
# delta = target F1 - within F1
#
# Positive = improved target-period Macro-F1
# Negative = degraded target-period Macro-F1
# =========================================================

delta_rows = []


def add_model_deltas(setting, row):

    model = row["model"]

    for target in ["short", "long"]:

        seed_deltas = []

        for seed in SEEDS:

            within = float(
                row[f"seed_{seed}_within_f1"]
            )

            target_f1 = float(
                row[f"seed_{seed}_{target}_f1"]
            )

            delta = target_f1 - within
            seed_deltas.append(delta)

        delta_rows.append({
            "dataset_setting": setting,
            "target": target,
            "model": model,
            "delta_macro_f1_mean":
                np.mean(seed_deltas),
            "delta_macro_f1_sd":
                np.std(seed_deltas, ddof=1),
            **{
                f"seed_{seed}_delta":
                    seed_deltas[i]
                for i, seed in enumerate(SEEDS)
            },
        })


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
# 4. VERIFY AGAINST TABLE-9 VALUES
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
        f"RQ2 mismatch {key}: "
        f"got {got:.6f}, "
        f"expected {exp:.4f}"
    )


print("\nPASS: all 24 RQ2 deltas reproduced from seed-level F1.")


# =========================================================
# 5. PANEL A — MOVEMENT PROFILE
# =========================================================

metric_cols = {
    "Novelty":
        "lexical_novelty_mean",

    "JSD":
        "js_divergence_mean",

    "ΔC100":
        "top100_mass_change_mean",

    "ΔL":
        "mean_length_change_mean",

    "Overlap":
        "overlap_loss_mean",

    "Marker asym.":
        "class_marker_asymmetry_mean",
}

metric_names = list(metric_cols.keys())

heat = []
annotations = []

for _, row in rq5.iterrows():

    heat_row = []
    annotation_row = []

    for name, col in metric_cols.items():

        value = float(row[col])

        all_values = rq5[col].astype(float)

        # Signed measures:
        # normalize magnitude only for visual intensity.
        if name in {"ΔC100", "ΔL"}:
            magnitude = abs(value)
            max_value = all_values.abs().max()
        else:
            magnitude = value
            max_value = all_values.max()

        norm = (
            magnitude / max_value
            if max_value > 0
            else 0
        )

        heat_row.append(norm)

        if name == "ΔL":
            annotation_row.append(
                f"{value:+.1f}"
            )

        elif name == "ΔC100":
            annotation_row.append(
                f"{value:+.3f}"
            )

        else:
            annotation_row.append(
                f"{value:.3f}"
            )

    heat.append(heat_row)
    annotations.append(annotation_row)

heat = np.array(heat)


row_labels = [
    f"{setting}\n{target.capitalize()}"
    for setting, target in CONDITION_ORDER
]


# =========================================================
# 6. CREATE FIGURE
# =========================================================

fig, (ax1, ax2) = plt.subplots(
    1,
    2,
    figsize=(14, 7),
    gridspec_kw={
        "width_ratios": [1.2, 1.0]
    },
)


# -------------------------
# Panel A
# -------------------------

im = ax1.imshow(
    heat,
    aspect="auto",
)

ax1.set_title(
    "A. Dataset-side temporal movement"
)

ax1.set_xticks(
    np.arange(len(metric_names))
)

ax1.set_xticklabels(
    metric_names,
    rotation=35,
    ha="right",
)

ax1.set_yticks(
    np.arange(len(row_labels))
)

ax1.set_yticklabels(
    row_labels
)

for i in range(heat.shape[0]):
    for j in range(heat.shape[1]):

        ax1.text(
            j,
            i,
            annotations[i][j],
            ha="center",
            va="center",
            fontsize=8,
        )


cbar = fig.colorbar(
    im,
    ax=ax1,
    fraction=0.046,
    pad=0.04,
)

cbar.set_label(
    "Relative within-metric magnitude"
)


# =========================================================
# 7. PANEL B — MODEL RESPONSE
# =========================================================

y = np.arange(len(CONDITION_ORDER))

markers = {
    "distilbert": "o",
    "tfidf_lr": "s",
    "phi3.5:latest": "^",
    "qwen2.5:3b-instruct": "D",
}

for model in MODEL_ORDER:

    subset = (
        delta[
            delta["model"] == model
        ]
        .sort_values("order")
    )

    ax2.errorbar(
        subset["delta_macro_f1_mean"],
        y,
        xerr=subset["delta_macro_f1_sd"],
        fmt=markers[model],
        capsize=3,
        label=MODEL_NAMES[model],
    )


ax2.axvline(
    0,
    linewidth=1,
)

ax2.set_yticks(y)
ax2.set_yticklabels(row_labels)
ax2.invert_yaxis()

ax2.set_xlabel(
    "ΔMacro-F1 (target − within)"
)

ax2.set_title(
    "B. Corresponding model response"
)

ax2.grid(
    axis="x",
    alpha=0.25,
)

ax2.legend(
    frameon=False,
    loc="best",
)


# =========================================================
# 8. FINALIZE
# =========================================================

fig.suptitle(
    "Temporal movement profiles and model-performance changes",
    fontsize=14,
)

fig.tight_layout(
    rect=[0, 0, 1, 0.96]
)


for ext in ["png", "pdf", "svg"]:

    path = (
        OUT
        / f"Figure5_movement_vs_response.{ext}"
    )

    fig.savefig(
        path,
        dpi=300 if ext == "png" else None,
        bbox_inches="tight",
    )

plt.close(fig)


# =========================================================
# 9. SAVE AUDIT DATA
# =========================================================

delta.to_csv(
    OUT / "Figure5_RQ2_recomputed_deltas.csv",
    index=False,
)

rq5.to_csv(
    OUT / "Figure5_RQ5_verified_inputs.csv",
    index=False,
)


print("\nRQ2 deltas used in Panel B:")

print(
    delta[
        [
            "dataset_setting",
            "target",
            "model",
            "delta_macro_f1_mean",
            "delta_macro_f1_sd",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


print("\nSaved figure to:")
print(OUT)

print(
    "\nPASS: Figure 5 generated entirely "
    "from verified upstream RQ2 and RQ5 data."
)
