from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("/mnt/c/Research/Beyond_Accuracy_Reproduction")

# Corrected Yelp results after recovering the 168 Qwen P1 failures
YELP_METRICS = (
    ROOT
    / "out"
    / "sentiment_joint_20260722_103442_YELP_RETRY"
    / "metrics_per_seed.csv"
)

# Existing verified RQ3 results are still valid for Climate
candidates = [
    ROOT / "RQ3_Verifier" / "common_matched_scores.csv",
    ROOT / "outputs" / "common_matched_scores.csv",
    ROOT / "RQ3_Notebook_Output" / "common_matched_scores.csv",
]

CLIMATE_SOURCE = next((p for p in candidates if p.exists()), None)

if CLIMATE_SOURCE is None:
    hits = list(ROOT.rglob("common_matched_scores.csv"))
    if not hits:
        raise FileNotFoundError("Could not find common_matched_scores.csv")
    CLIMATE_SOURCE = hits[0]

OUT = ROOT / "RQ3_Figure4_corrected"
OUT.mkdir(exist_ok=True)

print("Climate source:", CLIMATE_SOURCE)
print("Yelp source:", YELP_METRICS)


# =========================================================
# 1. CLIMATE — unchanged verified results
# =========================================================

climate = pd.read_csv(CLIMATE_SOURCE)

climate = climate[
    (climate["setting"] == "Climate n500")
    & climate["model"].isin([
        "phi3.5:latest",
        "qwen2.5:3b-instruct"
    ])
    & climate["prompt"].isin([
        "p1_minimal",
        "p3_json_strict",
        "p5_abstain"
    ])
][
    ["setting", "model", "seed", "prompt", "macro_f1"]
].copy()


# =========================================================
# 2. YELP — corrected complete reference samples
# =========================================================

yelp = pd.read_csv(YELP_METRICS)

yelp = yelp[
    (yelp["approach"] == "frozen")
    & yelp["model"].isin([
        "phi3.5:latest",
        "qwen2.5:3b-instruct"
    ])
    & (yelp["split"] == "within")
    & yelp["prompt"].isin([
        "p1_minimal",
        "p3_json_strict",
        "p5_abstain"
    ])
    & yelp["eval_n_per_label"].isin([500, 1000])
].copy()

yelp["setting"] = yelp["eval_n_per_label"].map({
    500: "Yelp n500",
    1000: "Yelp n1000"
})

yelp = yelp[
    ["setting", "model", "seed", "prompt", "macro_f1"]
]


# =========================================================
# 3. COMBINE AND AUDIT
# =========================================================

scores = pd.concat([climate, yelp], ignore_index=True)

for setting in ["Climate n500", "Yelp n500", "Yelp n1000"]:
    tmp = scores[scores["setting"] == setting]

    print(
        setting,
        "rows =", len(tmp),
        "seeds =", sorted(tmp["seed"].unique())
    )

    assert len(tmp) == 30
    assert not tmp.duplicated(
        ["model", "seed", "prompt"]
    ).any()


scores.to_csv(
    OUT / "Figure4_per_seed_scores.csv",
    index=False
)


# =========================================================
# 4. FIGURE MEANS + SAMPLE SD
# =========================================================

summary = (
    scores
    .groupby(["setting", "model", "prompt"], as_index=False)
    .agg(
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_sd=("macro_f1", lambda x: x.std(ddof=1)),
        n_seeds=("seed", "nunique")
    )
)

summary.to_csv(
    OUT / "Figure4_summary.csv",
    index=False
)

print("\n=== FIGURE 4 VALUES ===")
print(summary.to_string(index=False))


# =========================================================
# 5. PROMPT-SENSITIVITY RANGE
# range within seed first, then average across 5 seeds
# =========================================================

ranges_seed = (
    scores
    .groupby(["setting", "model", "seed"], as_index=False)
    .agg(
        prompt_range=(
            "macro_f1",
            lambda x: x.max() - x.min()
        )
    )
)

ranges = (
    ranges_seed
    .groupby(["setting", "model"], as_index=False)
    .agg(
        mean_seed_range=("prompt_range", "mean"),
        sample_sd=("prompt_range", lambda x: x.std(ddof=1))
    )
)

ranges_seed.to_csv(
    OUT / "Figure4_prompt_ranges_per_seed.csv",
    index=False
)

ranges.to_csv(
    OUT / "Figure4_prompt_ranges_summary.csv",
    index=False
)

print("\n=== CORRECTED PROMPT SENSITIVITY ===")
print(ranges.to_string(index=False))


# =========================================================
# 6. REGENERATE FIGURE 4
# =========================================================

SETTINGS = [
    "Climate n500",
    "Yelp n500",
    "Yelp n1000"
]

MODELS = [
    "phi3.5:latest",
    "qwen2.5:3b-instruct"
]

MODEL_NAMES = {
    "phi3.5:latest": "Phi",
    "qwen2.5:3b-instruct": "Qwen"
}

PROMPTS = [
    "p1_minimal",
    "p3_json_strict",
    "p5_abstain"
]

MARKERS = {
    "phi3.5:latest": "o",
    "qwen2.5:3b-instruct": "s"
}

plt.rcParams.update({
    "font.size": 11,
    "svg.fonttype": "none"
})

fig, axes = plt.subplots(
    1, 3,
    figsize=(12.5, 3.8)
)

for ax, setting in zip(axes, SETTINGS):

    for model in MODELS:

        sub = (
            summary[
                (summary["setting"] == setting)
                & (summary["model"] == model)
            ]
            .set_index("prompt")
            .loc[PROMPTS]
        )

        ax.errorbar(
            range(3),
            sub["macro_f1_mean"],
            yerr=sub["macro_f1_sd"],
            marker=MARKERS[model],
            linewidth=1.7,
            markersize=6,
            capsize=4,
            label=MODEL_NAMES[model]
        )

    ax.set_title(setting)
    ax.set_ylabel("Macro-F1")

    ax.set_xticks(
        range(3),
        ["P1", "P3", "P5"]
    )

    if setting == "Climate n500":
        ax.set_ylim(0.54, 0.74)
    else:
        ax.set_ylim(0.925, 0.98)

    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.22)


handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=2,
    frameon=False
)

fig.tight_layout(
    rect=(0, 0, 1, 0.88)
)

fig.savefig(
    OUT / "Figure4_prompt_comparison_corrected.png",
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    OUT / "Figure4_prompt_comparison_corrected.svg",
    bbox_inches="tight"
)

fig.savefig(
    OUT / "Figure4_prompt_comparison_corrected.pdf",
    bbox_inches="tight"
)

plt.close(fig)


# =========================================================
# 7. FINAL CAPTION
# =========================================================

caption = """Full reference samples; means ±1 sample SD, not confidence intervals. Vertical scales differ.

Figure 4: Reference-period Macro-F1 across prompt configurations. Points show five-seed means and error bars show sample standard deviations. Comparisons use the same reference samples across all six model–prompt combinations within each seed. P5 predictions are scored using the required best-guess labels, including abstained cases. Vertical scales differ across panels.
"""

(OUT / "Figure4_caption_corrected.txt").write_text(
    caption,
    encoding="utf-8"
)

print("\nDONE.")
print("Outputs:", OUT)