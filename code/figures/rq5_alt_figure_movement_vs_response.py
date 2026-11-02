from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("/mnt/c/Research/Beyond_Accuracy_Reproduction")

RQ5_SUMMARY = ROOT / "RQ5_reaudit" / "gate2_rq5_summary.csv"
DELTA_CSV = ROOT / "RQ5_reaudit" / "rq2_delta_macro_f1_for_alt_figure.csv"

OUTDIR = ROOT / "RQ5_reaudit" / "alt_figure"
OUTDIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 1. LOAD RQ5 SUMMARY
# =========================================================

rq5 = pd.read_csv(RQ5_SUMMARY)

order = [
    ("Climate n500", "short"),
    ("Climate n500", "long"),
    ("Yelp n500", "short"),
    ("Yelp n500", "long"),
    ("Yelp n1000", "short"),
    ("Yelp n1000", "long"),
]

rq5["row_key"] = list(zip(rq5["dataset_setting"], rq5["target"]))
rq5["row_order"] = rq5["row_key"].map({k: i for i, k in enumerate(order)})
rq5 = rq5.sort_values("row_order").reset_index(drop=True)

assert len(rq5) == 6, f"Expected 6 RQ5 rows, got {len(rq5)}"


# =========================================================
# 2. PREPARE PANEL A DATA
# =========================================================
#
# We keep signs for table interpretation, but normalize
# magnitudes for heatmap shading.
# =========================================================

metric_map = {
    "Novelty": "lexical_novelty_mean",
    "JSD": "js_divergence_mean",
    "ΔC100": "top100_mass_change_mean",
    "ΔL": "mean_length_change_mean",
    "Overlap": "overlap_loss_mean",
    "Marker asym.": "class_marker_asymmetry_mean",
}

metric_labels = list(metric_map.keys())

raw_values = pd.DataFrame(index=rq5.index)

for display_name, col in metric_map.items():
    vals = rq5[col].astype(float).copy()

    # Normalize magnitudes for signed metrics
    if display_name in {"ΔC100", "ΔL"}:
        vals_for_norm = vals.abs()
    else:
        vals_for_norm = vals.copy()

    max_val = vals_for_norm.max()
    raw_values[display_name] = vals
    raw_values[f"{display_name}_norm"] = vals_for_norm / max_val if max_val > 0 else 0.0

heat = raw_values[[f"{m}_norm" for m in metric_labels]].to_numpy()


# Text annotations inside cells
def fmt_metric(metric, value):
    if metric in {"ΔC100", "ΔL"}:
        return f"{value:+.2f}" if metric == "ΔL" else f"{value:+.3f}"
    return f"{value:.3f}"


annot = []
for i in range(len(rq5)):
    row = []
    for m in metric_labels:
        row.append(fmt_metric(m, raw_values.loc[i, m]))
    annot.append(row)


# Row labels
row_labels = [
    f"{ds}\n{target.capitalize()}"
    for ds, target in zip(rq5["dataset_setting"], rq5["target"])
]


# =========================================================
# 3. LOAD OR CREATE PANEL B TEMPLATE
# =========================================================
#
# delta_macro_f1 should be TARGET - WITHIN
# positive = improvement on shifted target
# negative = degradation on shifted target
# =========================================================

expected_models = [
    "distilbert",
    "tfidf_lr",
    "phi3.5:latest",
    "qwen2.5:3b-instruct",
]

if not DELTA_CSV.exists():
    rows = []
    for ds, target in order:
        for model in expected_models:
            rows.append(
                {
                    "dataset_setting": ds,
                    "target": target,
                    "model": model,
                    "delta_macro_f1": np.nan,
                }
            )
    template = pd.DataFrame(rows)
    template.to_csv(DELTA_CSV, index=False)

    print("\nCreated template:")
    print(DELTA_CSV)
    print("\nFill the delta_macro_f1 column and rerun.")
    print("Definition: delta_macro_f1 = target_macro_f1 - within_macro_f1")
    raise SystemExit(0)

delta = pd.read_csv(DELTA_CSV)

required_cols = {
    "dataset_setting",
    "target",
    "model",
    "delta_macro_f1",
}
missing = required_cols - set(delta.columns)
if missing:
    raise ValueError(f"Missing columns in {DELTA_CSV}: {sorted(missing)}")

delta["row_key"] = list(zip(delta["dataset_setting"], delta["target"]))
delta["row_order"] = delta["row_key"].map({k: i for i, k in enumerate(order)})

delta = delta.sort_values(["row_order", "model"]).reset_index(drop=True)

# Basic checks
expected_n = len(order) * len(expected_models)
assert len(delta) == expected_n, f"Expected {expected_n} delta rows, got {len(delta)}"

if delta["delta_macro_f1"].isna().any():
    missing_rows = delta[delta["delta_macro_f1"].isna()]
    print("\nStill missing delta values in:")
    print(missing_rows.to_string(index=False))
    raise SystemExit("Fill all delta_macro_f1 values and rerun.")


# =========================================================
# 4. PLOT
# =========================================================

fig = plt.figure(figsize=(14, 7.5))
gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0])

# -------------------------
# Panel A: movement heatmap
# -------------------------
ax1 = fig.add_subplot(gs[0, 0])

im = ax1.imshow(heat, aspect="auto")

ax1.set_xticks(range(len(metric_labels)))
ax1.set_xticklabels(metric_labels, rotation=30, ha="right")

ax1.set_yticks(range(len(row_labels)))
ax1.set_yticklabels(row_labels)

ax1.set_title("A. Temporal movement profile")

for i in range(heat.shape[0]):
    for j in range(heat.shape[1]):
        ax1.text(
            j, i,
            annot[i][j],
            ha="center", va="center",
            fontsize=8,
        )

cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
cbar.set_label("Relative movement intensity (0–1)")


# -------------------------
# Panel B: model response
# -------------------------
ax2 = fig.add_subplot(gs[0, 1])

model_order = expected_models
model_names = {
    "distilbert": "DistilBERT",
    "tfidf_lr": "TF-IDF LR",
    "phi3.5:latest": "Phi",
    "qwen2.5:3b-instruct": "Qwen",
}
markers = {
    "distilbert": "o",
    "tfidf_lr": "s",
    "phi3.5:latest": "^",
    "qwen2.5:3b-instruct": "D",
}

y_positions = np.arange(len(order))

for model in model_order:
    sub = delta[delta["model"] == model].sort_values("row_order")
    ax2.plot(
        sub["delta_macro_f1"].to_numpy(),
        y_positions,
        marker=markers[model],
        linestyle="",
        label=model_names[model],
    )

ax2.axvline(0.0, linewidth=1)

ax2.set_yticks(y_positions)
ax2.set_yticklabels(row_labels)
ax2.invert_yaxis()

ax2.set_title("B. Corresponding model response")
ax2.set_xlabel("ΔMacro-F1 (target − within)")
ax2.grid(axis="x", alpha=0.25)
ax2.legend(frameon=False, loc="best")


# -------------------------
# Overall title
# -------------------------
fig.suptitle(
    "Temporal movement profiles and corresponding model-performance changes",
    fontsize=14,
)
fig.tight_layout(rect=(0, 0, 1, 0.96))

# Save
png = OUTDIR / "Figure5_alt_movement_vs_response.png"
pdf = OUTDIR / "Figure5_alt_movement_vs_response.pdf"
svg = OUTDIR / "Figure5_alt_movement_vs_response.svg"

fig.savefig(png, dpi=300, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
fig.savefig(svg, bbox_inches="tight")
plt.close(fig)

print("\nSaved:")
print(" ", png)
print(" ", pdf)
print(" ", svg)

# Also save the panel-A values for auditing
panel_a_export = rq5[
    [
        "dataset_setting",
        "target",
        "lexical_novelty_mean",
        "js_divergence_mean",
        "top100_mass_change_mean",
        "mean_length_change_mean",
        "overlap_loss_mean",
        "class_marker_asymmetry_mean",
    ]
].copy()

panel_a_export.to_csv(
    OUTDIR / "Figure5_alt_panelA_values.csv",
    index=False,
)

delta.to_csv(
    OUTDIR / "Figure5_alt_panelB_values.csv",
    index=False,
)

caption = """
Figure 5: Temporal movement profiles and corresponding model-performance changes.
Panel A summarizes six forensic measures for each dataset-target condition. Each
metric is normalized independently to the 0–1 range for visualization, while the
cell annotations report the underlying values; signed ΔC100 and ΔL values retain
their original directions. Panel B shows the corresponding temporal change in
Macro-F1 (target minus within) for DistilBERT, TF-IDF LR, Phi, and Qwen.
Positive values indicate improved target-period performance, whereas negative
values indicate degradation. The figure is descriptive and should not be interpreted
as a causal attribution of performance change to any single forensic measure.
""".strip()

(OUTDIR / "Figure5_alt_caption.txt").write_text(caption, encoding="utf-8")
print(" ", OUTDIR / "Figure5_alt_caption.txt")
