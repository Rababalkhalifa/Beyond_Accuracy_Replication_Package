from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


ROOT = Path("/mnt/c/Research/Beyond_Accuracy_Reproduction")

INPUT = ROOT / "RQ5_reaudit" / "gate2_rq5_summary.csv"

OUT = ROOT / "RQ5_reaudit" / "gate3"
OUT.mkdir(parents=True, exist_ok=True)


# =========================================================
# 1. LOAD VERIFIED GATE-2 SUMMARY
# =========================================================

df = pd.read_csv(INPUT)

expected_settings = {
    "Climate n500",
    "Yelp n500",
    "Yelp n1000",
}

expected_targets = {
    "short",
    "long",
}

assert set(df["dataset_setting"]) == expected_settings
assert set(df["target"]) == expected_targets
assert len(df) == 6

print("Loaded verified Gate-2 summary:")
print(INPUT)


# =========================================================
# 2. TABLE 12 COLUMNS
# =========================================================

table12 = df[
    [
        "dataset_setting",
        "target",

        "lexical_novelty_mean",
        "lexical_novelty_sd",

        "js_divergence_mean",
        "js_divergence_sd",

        "top100_mass_change_mean",
        "top100_mass_change_sd",

        "mean_length_change_mean",
        "mean_length_change_sd",

        "overlap_loss_mean",
        "overlap_loss_sd",

        "class_marker_asymmetry_mean",
        "class_marker_asymmetry_sd",

        "task_native_labels",
    ]
].copy()


# =========================================================
# 3. FORMAT MEAN ± SD
# =========================================================

def pm(mean, sd, digits=4):
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


formatted_rows = []

for _, r in table12.iterrows():

    formatted_rows.append(
        {
            "Dataset": r["dataset_setting"],
            "Target": r["target"],

            "Novelty":
                pm(
                    r["lexical_novelty_mean"],
                    r["lexical_novelty_sd"],
                ),

            "Distributional movement":
                pm(
                    r["js_divergence_mean"],
                    r["js_divergence_sd"],
                ),

            "Top-term concentration":
                pm(
                    r["top100_mass_change_mean"],
                    r["top100_mass_change_sd"],
                ),

            "Document-length change":
                pm(
                    r["mean_length_change_mean"],
                    r["mean_length_change_sd"],
                ),

            "Overlap loss":
                pm(
                    r["overlap_loss_mean"],
                    r["overlap_loss_sd"],
                ),

            "Class-marker asymmetry":
                pm(
                    r["class_marker_asymmetry_mean"],
                    r["class_marker_asymmetry_sd"],
                ),

            "Task-native labels":
                r["task_native_labels"],
        }
    )


table12_formatted = pd.DataFrame(formatted_rows)

table12_formatted.to_csv(
    OUT / "Table12_corrected.csv",
    index=False,
)

print("\n" + "=" * 110)
print("CORRECTED TABLE 12")
print("=" * 110)

print(
    table12_formatted.to_string(index=False)
)


# =========================================================
# 4. WRITE LATEX TABLE BODY
# =========================================================

latex_lines = []

for _, r in table12.iterrows():

    dataset = r["dataset_setting"]
    target = r["target"]

    line = (
        f"{dataset} & {target} & "
        f"{r['lexical_novelty_mean']:.4f} $\\pm$ "
        f"{r['lexical_novelty_sd']:.4f} & "
        f"{r['js_divergence_mean']:.4f} $\\pm$ "
        f"{r['js_divergence_sd']:.4f} & "
        f"{r['top100_mass_change_mean']:.4f} $\\pm$ "
        f"{r['top100_mass_change_sd']:.4f} & "
        f"{r['mean_length_change_mean']:.4f} $\\pm$ "
        f"{r['mean_length_change_sd']:.4f} & "
        f"{r['overlap_loss_mean']:.4f} $\\pm$ "
        f"{r['overlap_loss_sd']:.4f} & "
        f"{r['class_marker_asymmetry_mean']:.4f} $\\pm$ "
        f"{r['class_marker_asymmetry_sd']:.4f} \\\\"
    )

    latex_lines.append(line)


latex_body = "\n".join(latex_lines)

(OUT / "Table12_corrected_latex_rows.txt").write_text(
    latex_body,
    encoding="utf-8",
)


# =========================================================
# 5. CORRECTED TABLE 12 CAPTION
# =========================================================

caption = r"""
Temporal-forensics evidence by dataset and target horizon.
Values quantify dataset movement from the within-period
reference to each shifted target period and are reported as
mean $\pm$ sample standard deviation across five sampled
seeds (42--46). Lexical novelty is the proportion of target
vocabulary types absent from the within-period vocabulary;
distributional movement is measured using Jensen--Shannon
divergence; top-term concentration is the paired within-seed
target-minus-reference change in top-100 token mass
($\Delta C_{100}$); document-length change is the paired
within-seed target-minus-reference change in mean token
length ($\Delta L$); overlap loss is one minus vocabulary
Jaccard overlap; and class-marker asymmetry is the absolute
difference between the two task-native class-marker churn
values.
""".strip()

(OUT / "Table12_corrected_caption.txt").write_text(
    caption,
    encoding="utf-8",
)


# =========================================================
# 6. FIGURE 5
# =========================================================
#
# Goal:
# Normalize each metric independently across the six
# dataset-target conditions.
#
# This preserves the original Figure-5 interpretation:
# 1.00 = largest observed magnitude for that metric.
#
# For signed metrics:
#   ΔC100 and ΔL
# use absolute magnitude because Figure 5 summarizes
# movement intensity, not movement direction.
#
# The quantitative table retains the sign.
# =========================================================

plot_df = df.copy()

plot_metrics = {
    "Lexical novelty":
        "lexical_novelty_mean",

    "Distributional movement":
        "js_divergence_mean",

    "Top-term concentration":
        "top100_mass_change_mean",

    "Document-length change":
        "mean_length_change_mean",

    "Overlap loss":
        "overlap_loss_mean",

    "Class-marker asymmetry":
        "class_marker_asymmetry_mean",
}


# =========================================================
# 7. COMPUTE DRIVER INTENSITY
# =========================================================

intensity_rows = []

for _, row in plot_df.iterrows():

    record = {
        "dataset_setting":
            row["dataset_setting"],

        "target":
            row["target"],
    }

    for display_name, col in plot_metrics.items():

        raw = row[col]

        # Signed change metrics are converted to magnitude.
        if col in {
            "top100_mass_change_mean",
            "mean_length_change_mean",
        }:
            raw = abs(raw)

        values = plot_df[col].copy()

        if col in {
            "top100_mass_change_mean",
            "mean_length_change_mean",
        }:
            values = values.abs()

        max_value = values.max()

        intensity = (
            raw / max_value
            if max_value > 0
            else 0
        )

        record[display_name] = intensity

    intensity_rows.append(record)


intensity_df = pd.DataFrame(intensity_rows)

intensity_df.to_csv(
    OUT / "Figure5_normalized_driver_intensity.csv",
    index=False,
)


# =========================================================
# 8. EXTRACT TOP THREE DRIVERS PER CONDITION
# =========================================================

top_rows = []

metric_names = list(plot_metrics.keys())

for _, row in intensity_df.iterrows():

    values = {
        metric:
            row[metric]
        for metric in metric_names
    }

    ranking = sorted(
        values.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    for rank, (metric, value) in enumerate(
        ranking[:3],
        start=1,
    ):

        top_rows.append(
            {
                "dataset_setting":
                    row["dataset_setting"],

                "target":
                    row["target"],

                "rank":
                    rank,

                "driver":
                    metric,

                "normalized_intensity":
                    value,
            }
        )


top3 = pd.DataFrame(top_rows)

top3.to_csv(
    OUT / "Figure5_top3_drivers.csv",
    index=False,
)


print("\n" + "=" * 110)
print("FIGURE 5 — TOP THREE NORMALIZED DRIVERS")
print("=" * 110)

print(
    top3.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# =========================================================
# 9. DRAW FIGURE 5
# =========================================================

condition_order = [
    ("Climate n500", "short"),
    ("Climate n500", "long"),
    ("Yelp n500", "short"),
    ("Yelp n500", "long"),
    ("Yelp n1000", "short"),
    ("Yelp n1000", "long"),
]


fig, axes = plt.subplots(
    2,
    3,
    figsize=(15, 8),
)

axes = axes.flatten()


for ax, (setting, target) in zip(
    axes,
    condition_order,
):

    subset = top3[
        (top3["dataset_setting"] == setting)
        &
        (top3["target"] == target)
    ].sort_values(
        "normalized_intensity",
        ascending=True,
    )

    ax.barh(
        subset["driver"],
        subset["normalized_intensity"],
    )

    ax.set_xlim(0, 1.05)

    ax.set_xlabel(
        "Relative driver intensity"
    )

    ax.set_title(
        f"{setting} — {target.capitalize()}"
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(
        axis="x",
        alpha=0.2,
    )


fig.suptitle(
    "Temporal-forensics movement signatures",
    fontsize=14,
)

fig.tight_layout(
    rect=[0, 0, 1, 0.95]
)


# =========================================================
# 10. SAVE FIGURE
# =========================================================

fig.savefig(
    OUT / "Figure5_corrected.png",
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    OUT / "Figure5_corrected.pdf",
    bbox_inches="tight",
)

fig.savefig(
    OUT / "Figure5_corrected.svg",
    bbox_inches="tight",
)

plt.close(fig)


# =========================================================
# 11. FIGURE 5 CAPTION
# =========================================================

figure5_caption = r"""
Figure 5: Top temporal-forensics drivers across the evaluated
dataset-target conditions. For each forensic metric, movement
magnitude is normalized independently across the six
dataset-target conditions so that 1.00 represents the largest
observed value for that metric. Absolute magnitudes are used
for signed top-term-concentration and document-length changes,
while their original directions are retained in Table 12.
For readability, only the three highest normalized drivers are
shown for each condition. The figure is a descriptive summary
of movement signatures and should not be interpreted as a
causal attribution of model-performance changes.
""".strip()

(OUT / "Figure5_corrected_caption.txt").write_text(
    figure5_caption,
    encoding="utf-8",
)


# =========================================================
# 12. MANUSCRIPT-NUMBER SANITY CHECKS
# =========================================================

climate_short = df[
    (df["dataset_setting"] == "Climate n500")
    &
    (df["target"] == "short")
].iloc[0]

climate_long = df[
    (df["dataset_setting"] == "Climate n500")
    &
    (df["target"] == "long")
].iloc[0]

yelp500_long = df[
    (df["dataset_setting"] == "Yelp n500")
    &
    (df["target"] == "long")
].iloc[0]

yelp1000_long = df[
    (df["dataset_setting"] == "Yelp n1000")
    &
    (df["target"] == "long")
].iloc[0]


assert np.isclose(
    climate_short[
        "top100_mass_change_mean"
    ],
    0.039757,
    atol=1e-6,
)

assert np.isclose(
    climate_short[
        "mean_length_change_mean"
    ],
    -1.0878,
    atol=1e-6,
)

assert np.isclose(
    climate_long[
        "mean_length_change_mean"
    ],
    8.0938,
    atol=1e-6,
)

assert np.isclose(
    yelp500_long[
        "mean_length_change_mean"
    ],
    -35.1940,
    atol=1e-6,
)

assert np.isclose(
    yelp1000_long[
        "mean_length_change_mean"
    ],
    -35.0015,
    atol=1e-6,
)


# =========================================================
# 13. FINAL OUTPUT
# =========================================================

print("\n" + "=" * 110)
print("RQ5 GATE 3 COMPLETE")
print("=" * 110)

for path in [
    OUT / "Table12_corrected.csv",
    OUT / "Table12_corrected_latex_rows.txt",
    OUT / "Table12_corrected_caption.txt",
    OUT / "Figure5_normalized_driver_intensity.csv",
    OUT / "Figure5_top3_drivers.csv",
    OUT / "Figure5_corrected.png",
    OUT / "Figure5_corrected.pdf",
    OUT / "Figure5_corrected.svg",
    OUT / "Figure5_corrected_caption.txt",
]:
    print(path)

print("\nPASS: corrected Table 12 and Figure 5 regenerated.")
