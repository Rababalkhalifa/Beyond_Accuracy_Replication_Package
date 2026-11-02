"""Table generation and LaTeX formatting utilities for paper tables.

Guarantees 1-to-1 exact matching with the manuscript LaTeX templates and table numbering:
- Table 7: Descriptive language statistics across sampled splits
- Table 8: Canonical within-period performance across datasets and model families
- Table 9: Temporal Macro-F1 across Reference, Short, and Long, with Delta = Target - Reference
- Table 10: Calibration and confidence-based selective risk across evaluation periods (ECE, R_80, R_100)
- Table 11: Focused retained-error differences across evaluated historical cohorts (11 contrasts, 3,000 bootstrap replicates)
- Table 12: Frequency and observed benefit of explicit Abstention-aware (P5) abstention on reference samples
- Table 13: Temporal-forensics evidence by dataset and target horizon
"""

from decimal import Decimal, ROUND_HALF_UP
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


def fmt_half_up(val: float, decimals: int = 4) -> str:
    """Format a floating-point number with standard half-up rounding."""
    q = "0." + ("0" * decimals) if decimals > 0 else "0"
    return str(Decimal(str(round(float(val), 10))).quantize(Decimal(q), rounding=ROUND_HALF_UP))


def build_table7_latex(rows: Sequence[Dict[str, any]]) -> str:
    """Format Table 7 (Descriptive Language Statistics) as LaTeX."""
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{\textbf{Descriptive language statistics of the sampled temporal splits.} "
        r"Values are computed from the authoritative training (7,000 examples) and evaluation "
        r"(1,000 or 2,000 examples) split CSVs under lowercase word tokenization "
        r"(\texttt{\textbackslash b\textbackslash w+\textbackslash b}) and reported as "
        r"mean $\pm$ sample standard deviation across seeds 42--46.}",
        r"\label{tab:sample_language_descriptors}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"\textbf{Dataset setting} & \textbf{Split} & \textbf{Mean tokens/doc} & \textbf{Median tokens/doc} & "
        r"\textbf{Vocabulary size} & \textbf{Type-token ratio} & \textbf{Top-100 token mass} \\",
        r"\midrule"
    ]

    for i, r in enumerate(rows):
        if i > 0 and rows[i - 1]["dataset_setting"] != r["dataset_setting"]:
            lines.append(r"\midrule")
        setting = (r"\textbf{" + r["dataset_setting"] + "}"
                   if i == 0 or rows[i - 1]["dataset_setting"] != r["dataset_setting"] else "")
        lines.append(
            f"{setting} & {r['split']} & "
            f"{r['mean_len_mean']:.2f} $\\pm$ {r['mean_len_sd']:.2f} & "
            f"{r['med_len_mean']:.2f} $\\pm$ {r['med_len_sd']:.2f} & "
            f"{r['vocab_size_mean']:.1f} $\\pm$ {r['vocab_size_sd']:.1f} & "
            f"{r['ttr_mean']:.3f} $\\pm$ {r['ttr_sd']:.3f} & "
            f"{r['top100_mass_mean']:.3f} $\\pm$ {r['top100_mass_sd']:.3f} " + r"\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}"
    ]
    return "\n".join(lines) + "\n"


def build_table8_latex(rows: Sequence[Dict[str, any]]) -> str:
    """Format Table 8 (RQ1 Canonical Within-Period Performance) as LaTeX."""
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
    """Format Table 9 (RQ2 Temporal Robustness) as LaTeX.

    Adheres strictly to the paper's sign convention: Delta_j = MacroF1(T_j) - MacroF1(T_0).
    Negative indicates decline; positive indicates improvement.
    """
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Macro-F1 across the reference ($T_0$), short-distance ($T_1$), and long-distance ($T_2$) "
        r"evaluation periods. Values are means over five sampling seeds. Temporal changes are computed within "
        r"each seed as $\Delta_j=F_1(T_j)-F_1(T_0)$ and then averaged; positive changes indicate higher "
        r"target-period performance. Boldface marks the highest mean Macro-F1 in each setting and period, "
        r"not a statistically established difference.}",
        r"\label{tab:temporal-robustness}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"\textbf{Dataset setting} & \textbf{Model} & \textbf{Reference F1} & \textbf{Short F1} & \textbf{Long F1} & "
        r"$\Delta$ \textbf{short} & $\Delta$ \textbf{long} \\",
        r"\midrule"
    ]

    for i, row in enumerate(rows):
        if i > 0 and rows[i - 1]["dataset_setting"] != row["dataset_setting"]:
            lines.append(r"\midrule")

        setting = (r"\textbf{" + row["dataset_setting"] + "}"
                   if i == 0 or rows[i - 1]["dataset_setting"] != row["dataset_setting"] else "")
        model_name = LATEX_MODEL_NAMES.get(row["model"], row["model"])

        def fmt_delta(val: float) -> str:
            return f"+{val:.4f}" if val > 0 else f"{val:.4f}"

        lines.append(
            f"{setting} & {model_name} & {row['within_f1']:.4f} & {row['short_f1']:.4f} & {row['long_f1']:.4f} & "
            f"{fmt_delta(row['delta_short'])} & {fmt_delta(row['delta_long'])} " + r"\\"
        )

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def build_table10_latex(
    means: Dict[tuple, Dict[str, float]],
    settings: Sequence[str],
    models: Sequence[str]
) -> str:
    """Format Table 10 (Calibration and Selective Risk across Periods) as LaTeX."""
    lines = [
        r"\begin{table*}[t]",
        r"\centering\footnotesize",
        r"\caption{Calibration and confidence-based selective risk across evaluation periods (five-seed means). "
        r"$R_{80}$ and $R_{100}$ are retained error at 80\% and 100\% coverage, respectively; $R_{100}=1-\mathrm{Accuracy}$. "
        r"Boldface marks the lowest mean for each metric and period within a dataset setting and does not establish "
        r"a statistically reliable difference. Supervised confidence is the maximum predicted class probability; "
        r"frozen models use \texttt{Strict JSON} self-reported confidence. Ties at the coverage cutoff receive equal fractional inclusion.}",
        r"\label{tab:temporal-reliability}",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\begin{tabular}{llrrrrrrrrr}",
        r"\toprule",
        r"& & \multicolumn{3}{c}{ECE} & \multicolumn{3}{c}{$R_{80}$} & \multicolumn{3}{c}{$R_{100}$} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
        r"Dataset setting & Model & Reference & Short & Long & Reference & Short & Long & Reference & Short & Long \\",
        r"\midrule"
    ]

    for si, setting in enumerate(settings):
        if si > 0:
            lines.append(r"\midrule")

        min_ece = [min(means[setting, m, p]["ece"] for m in models) for p in ("within", "short", "long")]
        min_r80 = [min(means[setting, m, p]["risk_at_80"] for m in models) for p in ("within", "short", "long")]
        min_r100 = [min(means[setting, m, p]["risk_at_100"] for m in models) for p in ("within", "short", "long")]

        for i, model in enumerate(models):
            prefix = r"\textbf{" + setting + r"}" if i == 0 else ""
            m_name = LATEX_MODEL_LONG_NAMES.get(model, model)

            def fmt_val(val: float, best_val: float) -> str:
                s = fmt_half_up(val, 4)
                return r"\textbf{" + s + "}" if abs(val - best_val) < 1e-5 else s

            eces = [fmt_val(means[setting, model, p]["ece"], min_ece[pi]) for pi, p in enumerate(("within", "short", "long"))]
            r80s = [fmt_val(means[setting, model, p]["risk_at_80"], min_r80[pi]) for pi, p in enumerate(("within", "short", "long"))]
            r100s = [fmt_val(means[setting, model, p]["risk_at_100"], min_r100[pi]) for pi, p in enumerate(("within", "short", "long"))]

            lines.append(
                f"{prefix} & {m_name} & {eces[0]} & {eces[1]} & {eces[2]} & "
                f"{r80s[0]} & {r80s[1]} & {r80s[2]} & {r100s[0]} & {r100s[1]} & {r100s[2]} " + r"\\"
            )

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def build_table11_bootstrap_latex(rows: Sequence[Dict[str, any]]) -> str:
    """Format Table 11 (Focused Retained-Error Differences with Cluster Bootstrap) as LaTeX."""
    period_labels = {
        ("within", "phi_minus_qwen_at_80"): ("Reference", "Phi - Qwen at 80%"),
        ("short", "phi_minus_qwen_at_80"): ("Short", "Phi - Qwen at 80%"),
        ("long", "phi_minus_qwen_at_80"): ("Long", "Phi - Qwen at 80%"),
        ("within", "qwen_80_minus_100"): ("Reference", "Qwen 80% - 100%"),
        ("short", "qwen_80_minus_100"): ("Short", "Qwen 80% - 100%"),
        ("long", "qwen_80_minus_100"): ("Long", "Qwen 80% - 100%"),
        ("short", "distilbert_period_minus_reference_at_80"): ("Short - Ref.", "DistilBERT at 80%"),
        ("long", "distilbert_period_minus_reference_at_80"): ("Long - Ref.", "DistilBERT at 80%"),
    }

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\footnotesize",
        r"\caption{Focused retained-error differences across evaluated historical cohorts. "
        r"Differences are five-seed means in percentage points (pp). Conditional 95\% intervals use "
        r"3,000 synchronized document-cluster bootstrap resamples conditional on the archived models, "
        r"fixed seeds, and evaluated splits; intervals are not adjusted for multiple comparisons.}",
        r"\label{tab:rq2-bootstrap-contrasts}",
        r"\begin{tabular}{lllrrrrr}",
        r"\toprule",
        r"\textbf{Setting} & \textbf{Period} & \textbf{Contrast} & \textbf{Diff. (pp)} & "
        r"\textbf{95\% CI (pp)} & \textbf{Seed SD} & \textbf{Seed signs} & \textbf{Clusters} \\",
        r"\midrule"
    ]

    for i, r in enumerate(rows):
        prev_contrast = rows[i - 1]["contrast"] if i > 0 else None
        if i > 0 and (rows[i - 1]["setting"] != r["setting"] or prev_contrast != r["contrast"]):
            lines.append(r"\midrule")

        p_str, c_str = period_labels.get((r["period"], r["contrast"]), (r["period"], r["contrast"]))
        diff = float(r["mean_difference_pp"])
        lo = float(r["ci_95_low_pp"])
        hi = float(r["ci_95_high_pp"])
        sd = float(r["seed_sd_pp"])
        neg = int(r["seed_diffs_negative"])
        pos = int(r["seed_diffs_positive"])
        clusters = int(r["unique_document_clusters_in_contrast"])

        sign_str = f"{neg} neg." if neg == 5 else (f"{pos} pos." if pos == 5 else f"{neg} neg. / {pos} pos.")

        lines.append(
            f"{r['setting']} & {p_str} & {c_str} & {diff:+.3f} & "
            f"[{lo:+.3f}, {hi:+.3f}] & {sd:.3f} & {sign_str} & {clusters:,} " + r"\\"
        )

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def build_table12_abstention_latex(rows: Sequence[Dict[str, any]]) -> str:
    """Format Table 12 (Explicit Abstention-aware Abstention Frequency and Benefit) as LaTeX."""
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\footnotesize",
        r"\caption{Frequency and observed benefit of explicit \texttt{Abstention-aware} abstention on the reference samples. "
        r"Error rates are percentages averaged across five sampling seeds. ``Before'' scores all \texttt{Abstention-aware} best-guess labels; "
        r"``after'' scores only retained predictions. Counts are row occurrences across sampling runs and may include "
        r"repeated texts. A dash indicates that no abstentions occurred. No \texttt{Abstention-aware} request or response-format failures occurred.}",
        r"\label{tab:rq3-abstention}",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"\textbf{Setting} & \textbf{Model} & \textbf{Abstained/$N$} & \textbf{Incorrect among abstained} & "
        r"\textbf{Error before (\%)} & \textbf{Error after (\%)} & \textbf{Reduction (pp)} \\",
        r"\midrule"
    ]

    for i, r in enumerate(rows):
        if i > 0 and rows[i - 1]["setting"] != r["setting"]:
            lines.append(r"\midrule")
        setting_str = r"\textbf{" + r["setting"] + "}" if i == 0 or rows[i - 1]["setting"] != r["setting"] else ""
        lines.append(
            f"{setting_str} & {r['model']} & {r['abstained_slash_n']} & {r['incorrect_among_abstained']} & "
            f"{r['error_before_pct']:.3f} & {r['error_after_pct']:.3f} & {r['reduction_pp']:.3f} " + r"\\"
        )

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def build_table13_forensics_latex(rows: Sequence[Dict[str, any]]) -> str:
    """Format Table 13 (Temporal Forensics Evidence) as LaTeX."""
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Temporal-forensics evidence by dataset and target horizon. Values quantify movement "
        r"from the within-period reference to each shifted target period and are reported as "
        r"mean $\pm$ sample standard deviation across five sampled seeds (42--46). Lexical novelty is the "
        r"proportion of target vocabulary types absent from the within-period vocabulary; distributional "
        r"movement is measured using Jensen--Shannon divergence; top-term concentration is reported as "
        r"paired within-seed target-minus-reference change in top-100 token mass ($\Delta C_{100}$); "
        r"document-length change is reported as paired within-seed signed target-minus-reference change in "
        r"mean token length ($\Delta L$); overlap loss is defined as one minus vocabulary Jaccard overlap; "
        r"and class-marker asymmetry is the absolute difference between the two task-native class-marker churn values.}",
        r"\label{tab:rq5-temporal-forensics}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"\textbf{Dataset} & \textbf{Target} & \textbf{Novelty} & \textbf{Distributional movement} & "
        r"\textbf{Top-term concentration} & \textbf{Document-length change} & \textbf{Overlap loss} & "
        r"\textbf{Class-marker asymmetry} \\",
        r"\midrule"
    ]

    for i, r in enumerate(rows):
        if i > 0 and rows[i - 1]["dataset_setting"] != r["dataset_setting"]:
            lines.append(r"\midrule")
        d_str = r"\textbf{" + r["dataset_setting"] + "}" if i == 0 or rows[i - 1]["dataset_setting"] != r["dataset_setting"] else ""
        lines.append(
            f"{d_str} & {r['target']} & "
            f"{r['lexical_novelty_mean']:.4f} $\\pm$ {r['lexical_novelty_sd']:.4f} & "
            f"{r['js_divergence_mean']:.4f} $\\pm$ {r['js_divergence_sd']:.4f} & "
            f"{r['top100_mass_change_mean']:.4f} $\\pm$ {r['top100_mass_change_sd']:.4f} & "
            f"{r['mean_length_change_mean']:.4f} $\\pm$ {r['mean_length_change_sd']:.4f} & "
            f"{r['overlap_loss_mean']:.4f} $\\pm$ {r['overlap_loss_sd']:.4f} & "
            f"{r['class_marker_asymmetry_mean']:.4f} $\\pm$ {r['class_marker_asymmetry_sd']:.4f} " + r"\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}"
    ]
    return "\n".join(lines) + "\n"


# Backwards-compatible aliases
build_table11_latex = build_table12_abstention_latex
build_table12_latex = build_table13_forensics_latex
