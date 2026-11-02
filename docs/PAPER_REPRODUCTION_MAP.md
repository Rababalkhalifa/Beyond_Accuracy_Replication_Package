# Paper Reproduction Mapping Matrix

**Scope:** Itemized provenance and reproduction map connecting every table (Tables 7–13), figure (Figures 3–6), and empirical claim in *"Aging Gracefully? Reliability Profiling and Temporal Forensics of Frozen and Adapted Text Classifiers"* to its underlying prediction cohorts, analysis code, and output artifacts.

---

## 1. Comprehensive Paper Reproduction Matrix

| Paper Item | Core Empirical Finding | Source Predictions / Data | Analysis Code | Figure / Table Code | Output Artifact | Reproducible? | Provenance & Methodological Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: | :--- |
| **Table 7: Split Descriptives** | Mean/median tokens/doc, vocab size, TTR, and top-100 token mass across splits (seeds 42–46). | `data/splits/climate/`<br>`data/splits/yelp/` (65 CSVs + `forensics_and_descriptors_cache.json`) | `src/descriptors.py`<br>`scripts/run_table7.py` | `src/tables.py`<br>`code/analysis/rq5_gate2_regenerate_forensics.py` | `outputs/tables/Table7_descriptors.tex`<br>`outputs/tables/Table7_descriptors_summary.csv` | **100% Exact** | Lowercase regex word tokenization `\b\w+\b`. Exact match to Table 7 across all 12 reported rows. |
| **Table 8: Canonical Benchmark (RQ1)** | Within-period reference ($T_0$) performance. DistilBERT leads on Climate (F1 0.8876); Qwen leads on Yelp (F1 0.9602 / 0.9601). | `archives/completed_prediction_audit.zip`<br>`archives/original_yelp_frozen_predictions.zip`<br>`archives/yelp_prediction_alignment.zip` | `src/metrics.py`<br>`scripts/run_table8.py` | `src/tables.py`<br>`code/reproducibility/run_paper_models.py` | `outputs/tables/RQ1_Table8.tex`<br>`outputs/tables/RQ1_Table8_summary.csv` | **100% Exact** | Evaluated on `Strict-JSON` (`p3_json_strict`) for LLMs. Accuracy, Macro-F1, ECE-10, and mean inference time match Table 8. |
| **Table 9: Temporal Macro-F1 (RQ2)** | Temporal robustness deltas: $\Delta_j = \text{MacroF1}(T_j) - \text{MacroF1}(T_0)$. Negative deltas indicate decline. | Same prediction archives across $T_0, T_1, T_2$. | `src/metrics.py`<br>`scripts/run_table9.py` | `src/tables.py`<br>`code/historical/climate/climate_temporal_f1.py`<br>`code/historical/yelp/yelp_temporal_f1.py` | `outputs/tables/RQ2_Table9.tex`<br>`outputs/tables/RQ2_Table9_summary.csv` | **100% Exact** | Standardized sign convention: Target minus Reference. Climate DistilBERT drops `-0.0258` ($T_1$) and `-0.0170` ($T_2$); Qwen improves `+0.1019` on Climate $T_2$. |
| **Table 10: Calibration & Risk (RQ2)** | ECE-10, $R_{80}$ (retained error at 80% coverage), and $R_{100} = 1 - \text{Accuracy}$ across all three temporal horizons. | Same prediction archives (confidence fields). | `src/metrics.py`<br>`scripts/run_table10.py` | `src/tables.py`<br>`code/analysis/Temporal_coverage_rankings.py` | `outputs/tables/RQ2_Table10_temporal_reliability.tex`<br>`outputs/tables/RQ2_Table10_summary.csv` | **100% Exact** | Strict 10-bin ECE and equal fractional tie-handling for $R_{80}, R_{100}$ with standard half-up display rounding matching Table 10. |
| **Table 11: Bootstrap Contrasts (RQ2)** | 11 focused retained-error differences (pp) and conditional 95% document-cluster bootstrap intervals (3,000 resamples). | Matched prediction pools clustered by text digest + gold label across seeds 42–46. | `code/analysis/rq2_uncertainty_improved_v2.py`<br>`scripts/run_table11.py` | `src/tables.py` | `outputs/tables/RQ2_Table11_bootstrap_contrasts.tex`<br>`outputs/tables/RQ2_Table11_bootstrap_contrasts.csv` | **100% Exact** | Synchronized document-cluster bootstrap (`replicates=3000`, `seed=20261102`). All 11 intervals exclude zero and match Table 11 exactly. |
| **Table 12: Explicit Abstention & $S_{\text{prompt}}$ (RQ3)** | Explicit `Abstention-aware` (`p5_abstain`) abstention frequency and post-abstention error reduction, plus $S_{\text{prompt}}$ across `P1, P3, P5`. | P1, P3, and P5 prediction cohorts at reference horizon $T_0$. | `src/metrics.py`<br>`scripts/run_table12.py` | `src/tables.py`<br>`code/figures/rebuild_figure4_corrected.py` | `outputs/tables/RQ3_Table12_abstention.tex`<br>`outputs/tables/RQ3_Table12_summary.csv`<br>`outputs/tables/RQ3_prompt_sensitivity_summary.csv` | **100% Exact** | Qwen abstains 4/5,000 on Climate & Yelp n500 (4/4 wrong) and 3/10,000 on Yelp n1000 (3/3 wrong); Phi abstains 0. $S_{\text{prompt}}$: Climate `0.0631` vs `0.0206`; Yelp n500 `0.0147` vs `0.0107`; Yelp n1000 `0.0107` vs `0.0091`. |
| **Table 13: Temporal Forensics (RQ5)** | 6 dataset movement signals: Lexical Novelty, base-2 JSD, $\Delta C_{100}$, $\Delta L$, Overlap Loss, and Marker Asymmetry. | `data/splits/` across $T_0, T_1, T_2$ (`forensics_and_descriptors_cache.json`). | `src/forensics.py`<br>`scripts/run_table13.py` | `src/tables.py`<br>`code/analysis/rq5_gate2_regenerate_forensics.py` | `outputs/tables/RQ5_Table13_temporal_forensics.tex`<br>`outputs/tables/RQ5_Table13_summary.csv` | **100% Exact** | Synchronized with Gate 2 re-audit (`gate2_rq5_seed_level.csv`). Matches all 36 mean $\pm$ SD cells in Table 13. |
| **Figure 3: Risk–Coverage Curves** | 3×3 grid of confidence-based risk–coverage curves (20–100% coverage) across settings and historical periods. | All 180 evaluation conditions across seeds 42–46 (14,580 curve points). | `src/metrics.py`<br>`src/plotting.py` | `scripts/plot_figures.py`<br>`code/figures/plot_figure3_temporal_no_note.py` | `outputs/figures/Figure3_temporal_risk_coverage.pdf`<br>`outputs/figures/Figure3_temporal_risk_coverage.svg`<br>`outputs/figures/Figure3_temporal_risk_coverage.png` | **100% Exact** | Rows = Climate n500, Yelp n500, Yelp n1000; Columns = Reference, Short, Long. Matches Figure 3 (p. 22). |
| **Figure 4: Uncertainty Contrasts** | 3-panel forest plot of the 11 exploratory retained-error contrasts and 95% cluster-bootstrap intervals. | `outputs/tables/RQ2_Table11_bootstrap_contrasts.csv` | `src/plotting.py`<br>`scripts/plot_figures.py` | `code/figures/plot_rq2_uncertainty_contrasts.py` | `outputs/figures/Figure4_uncertainty_contrasts.pdf`<br>`outputs/figures/Figure4_uncertainty_contrasts.svg`<br>`outputs/figures/Figure4_uncertainty_contrasts.png` | **100% Exact** | Panels A (Yelp Phi − Qwen at 80%), B (Climate Qwen 80% − 100%), C (Climate DistilBERT temporal change). Matches Figure 4 (p. 23). |
| **Figure 5: Prompt Configurations** | 3-panel plot of reference-period Macro-F1 (mean $\pm$ SD) across `P1`, `P3`, `P5`. | `outputs/tables/RQ3_Figure5_prompt_summary.csv` | `src/plotting.py`<br>`scripts/plot_figures.py` | `code/figures/rebuild_figure4_corrected.py` | `outputs/figures/Figure5_prompt_comparison.pdf`<br>`outputs/figures/Figure5_prompt_comparison.svg`<br>`outputs/figures/Figure5_prompt_comparison.png` | **100% Exact** | Compares `Phi` and `Qwen` across `P1`, `P3`, `P5` on matched reference samples. Matches Figure 5 (p. 24). |
| **Figure 6: Movement vs. Response** | Dual-panel heatmap connecting dataset-side temporal movement (Panel A) to $\Delta\text{Macro-F1}$ response (Panel B). | Forensics summaries (Table 13) + Model deltas (Table 9). | `src/plotting.py`<br>`src/forensics.py` | `scripts/plot_figures.py`<br>`code/figures/rq5_alt_figure_verified_v4.py` | `outputs/figures/Figure6_movement_vs_response.pdf`<br>`outputs/figures/Figure6_movement_vs_response.svg`<br>`outputs/figures/Figure6_movement_vs_response.png` | **100% Exact** | Matches Figure 6 (p. 27) across all 6 dataset–horizon rows. |

---

## 2. Command-Level Execution Mapping

All paper outputs can be deterministically regenerated via the top-level driver:

```bash
# 1. Master execution reproducing all paper tables (Tables 7-13) and figures (Figures 3-6):
python reproduce.py --all

# 2. Individual targeted executions:
python reproduce.py --audit     # Verifies data integrity & SHA-256 hashes
python reproduce.py --table7    # Table 7: Language Descriptors
python reproduce.py --table8    # Table 8: Canonical Reference Performance (RQ1)
python reproduce.py --table9    # Table 9: Temporal Macro-F1 Deltas (RQ2)
python reproduce.py --table10   # Table 10: ECE-10 & Selective Risk R_80, R_100 (RQ2)
python reproduce.py --table11   # Table 11: 11 RQ2 Retained-Error Bootstrap Contrasts (RQ2)
python reproduce.py --table12   # Table 12: Explicit Abstention & Prompt Sensitivity S_prompt (RQ3)
python reproduce.py --table13   # Table 13: Temporal Forensics Movement Evidence (RQ5)
python reproduce.py --figures   # Figures 3, 4, 5, and 6
```
