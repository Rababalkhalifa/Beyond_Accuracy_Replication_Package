"""Table generation and LaTeX formatting utilities for paper tables.

Guarantees exact matching with the paper's LaTeX templates and table numbering:
- Table 8 (RQ1): Canonical within-period performance across datasets and models.
- Table 9 (RQ2): Temporal Macro-F1 degradation and seed-paired deltas.
- Reliability Table: Period-specific ECE and 80% coverage selective risk.
"""

from typing import Dict, List, Sequence

LATEX_MODEL_NAMES = {
    "distilbert": "DistilBERT",
    "tfidf_lr": "TF--IDF+LR",
    "phi3.5:latest": r"\texttt{phi}",
    "qwen2.5:3b-instruct": r"\texttt{qwen}"
}

LATEX_MODEL_LONG_NAMES = {
    "distilbert": "DistilBERT",
    "tfidf_lr": "TF--IDF+LR",
    "phi3.5:latest": r"\texttt{phi3.5:latest}",
    "qwen2.5:3b-instruct": r"\texttt{qwen2.5:3b-instruct}"
}


def build_table8_latex(rows: Sequence[Dict[str, any]]) -> str:
    """Format Table 8 (RQ1 Canonical Performance) as a LaTeX table."""
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Canonical within-period performance across datasets and model families. "
        r"Values are means across five sampling seeds. Climate is stance classification and "
        r"Yelp is sentiment classification. ECE uses predicted probabilities for supervised models "
        r"and self-reported confidence for prompted LLMs. Latency is the recorded mean per evaluation example; "
        r"timing procedures differ between model families.}",
        r"\label{tab:canonical-performance}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"\textbf{Dataset setting} & \textbf{Model} & \textbf{Accuracy} & \textbf{Macro-F1} & \textbf{ECE} & \textbf{Mean latency (s)} \\",
        r"\midrule"
    ]

    for i, row in enumerate(rows):
        if i > 0 and rows[i - 1]["dataset_setting"] != row["dataset_setting"]:
            lines.append(r"\midrule")
        
        setting = (r"\textbf{" + row["dataset_setting"] + "}"
                   if i == 0 or rows[i - 1]["dataset_setting"] != row["dataset_setting"] else "")
        model_name = LATEX_MODEL_NAMES.get(row["model"], row["model"])
        
        lines.append(
            f"{setting} & {model_name} & {row['accuracy']:.4f} & "
            f"{row['macro_f1']:.4f} & {row['ece']:.4f} & "
            f"{row['latency_mean_sec']:.6f} " + r"\\"
        )

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def build_table9_latex(rows: Sequence[Dict[str, any]]) -> str:
    """Format Table 9 (RQ2 Temporal Robustness) as a LaTeX table."""
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Temporal Macro-F1 across the within-period reference and short- and long-distance "
        r"target periods. Values are means across five sampling seeds. For each seed, $\Delta$ is within-period "
        r"Macro-F1 minus target-period Macro-F1; negative values indicate higher performance on the target sample. "
        r"Climate is stance classification and Yelp is sentiment classification.}",
        r"\label{tab:temporal-robustness}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"\textbf{Dataset setting} & \textbf{Model} & \textbf{Within F1} & \textbf{Short F1} & \textbf{Long F1} & "
        r"$\Delta$ \textbf{short} & $\Delta$ \textbf{long} \\",
        r"\midrule"
    ]

    for i, row in enumerate(rows):
        if i > 0 and rows[i - 1]["dataset_setting"] != row["dataset_setting"]:
            lines.append(r"\midrule")
        
        setting = (r"\textbf{" + row["dataset_setting"] + "}"
                   if i == 0 or rows[i - 1]["dataset_setting"] != row["dataset_setting"] else "")
        model_name = LATEX_MODEL_NAMES.get(row["model"], row["model"])
        
        lines.append(
            f"{setting} & {model_name} & {row['within_f1']:.4f} & {row['short_f1']:.4f} & {row['long_f1']:.4f} & "
            f"{row['delta_short']:.4f} & {row['delta_long']:.4f} " + r"\\"
        )

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def build_reliability_table_latex(means: Dict[tuple, Dict[str, float]], settings: Sequence[str], models: Sequence[str]) -> str:
    """Format Period-specific ECE and 80% Selective Risk as a LaTeX table."""
    lines = [
        r"\begin{table*}[t]",
        r"\centering\scriptsize",
        r"\caption{Period-specific ECE and expected retained error at 80\% confidence-based coverage "
        r"(five-seed means). ECE uses 10 equal-width bins. The coverage analysis derives from canonical P3 predictions "
        r"using fractional tie-breaking; explicit abstention is excluded. Lower numbers denote better performance.}",
        r"\label{tab:temporal-reliability}",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Dataset setting & Model & \multicolumn{3}{c}{Expected Calibration Error (ECE)} & "
        r"\multicolumn{3}{c}{Expected risk at 80\% coverage} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
        r" & & Within & Short & Long & Within & Short & Long \\",
        r"\midrule"
    ]

    for si, setting in enumerate(settings):
        if si > 0:
            lines.append(r"\midrule")
        for i, model in enumerate(models):
            eces = [means[setting, model, p]["ece"] for p in ("within", "short", "long")]
            r80s = [means[setting, model, p]["risk_at_80"] for p in ("within", "short", "long")]
            prefix = r"\textbf{" + setting + r"}" if i == 0 else ""
            m_name = LATEX_MODEL_LONG_NAMES.get(model, model)
            lines.append(
                f"{prefix} & {m_name} & {eces[0]:.4f} & {eces[1]:.4f} & {eces[2]:.4f} & "
                f"{r80s[0]:.4f} & {r80s[1]:.4f} & {r80s[2]:.4f} " + r"\\"
            )

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines) + "\n"
