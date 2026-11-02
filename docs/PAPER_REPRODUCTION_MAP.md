# Paper Reproduction Mapping Matrix

**Scope:** Itemized provenance and reproduction map connecting every table, figure, and empirical claim in the manuscript to its underlying prediction cohorts, analysis code, and output artifacts.

---

## 1. Comprehensive Paper Reproduction Matrix

| Paper Item | Core Empirical Finding | Source Predictions | Analysis Code | Figure / Table Code | Output Artifact | Reproducible? | Provenance & Methodological Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: | :--- |
| **Table 7: Split Descriptives** | Word count distributions, vocab size, TTR, and top-100 term mass across splits. | `data/splits/climate/`<br>`data/splits/yelp/` (all 65 CSVs) | `src/descriptors.py`<br>`scripts/run_table7.py` | `src/tables.py`<br>`code/analysis/rq5_gate2_regenerate_forensics.py` | `outputs/tables/Table7_descriptors.tex`<br>`outputs/tables/Table7_descriptors_summary.csv` | **100% Exact** | Lowercase regex word tokenization `\b\w+\b`. Exact match to Table 7 across all 12 reported rows. |
| **Table 8: Canonical Benchmark (RQ1)** | Within-period reference ($T_0$) performance. DistilBERT leads on Climate (F1 0.8876); Qwen leads on Yelp (F1 0.9602). | `archives/completed_prediction_audit.zip`<br>`archives/original_yelp_frozen_predictions.zip`<br>`archives/yelp_prediction_alignment.zip` | `src/metrics.py`<br>`scripts/run_table8.py` | `src/tables.py`<br>`code/reproducibility/run_paper_models.py` | `outputs/tables/RQ1_Table8.tex`<br>`outputs/tables/RQ1_Table8_summary.csv` | **100% Exact** | Evaluated on P3 for LLMs. Accuracy, Macro-F1, ECE-10, and inference latency matching reported values. |
| **Table 9: Temporal Macro-F1 (RQ2)** | Temporal robustness deltas: $\Delta_j = \text{MacroF1}(T_j) - \text{MacroF1}(T_0)$. Negative deltas indicate degradation. | Same prediction archives across $T_0, T_1, T_2$. | `src/metrics.py`<br>`scripts/run_table9.py` | `src/tables.py`<br>`code/historical/climate/climate_temporal_f1.py`<br>`code/historical/yelp/yelp_temporal_f1.py` | `outputs/tables/RQ2_Table9.tex`<br>`outputs/tables/RQ2_Table9_summary.csv` | **100% Exact** | Standardized sign convention: Target minus Reference. Climate DistilBERT drops -0.0911 at $T_2$; Qwen drops -0.0655. |
| **Table 10: Calibration & Risk (RQ2)** | ECE-10, $R_{80}$ (risk at 80% coverage), and $R_{100}$ (full risk) across all three temporal horizons. | Same prediction archives (confidence fields). | `src/metrics.py`<br>`scripts/run_table10.py` | `src/tables.py`<br>`code/historical/climate/climate_temporal_selective_risk.py`<br>`code/historical/yelp/yelp_temporal_selective_risk.py` | `outputs/tables/RQ2_Table10_temporal_reliability.tex`<br>`outputs/tables/RQ2_Table10_summary.csv` | **100% Exact** | Strict 10-bin ECE and empirical selective risk. Ranking inversions between ECE and selective risk documented. |
| **Table 11: Abstention & Sensitivity (RQ3)** | Explicit abstention behavior under P5 ($A_{\text{total}}, A_{\text{rate}}, A_{\text{pct}}, R_{\text{cov}}$) and prompt sensitivity $S_{\text{prompt}}$. | P1, P3, and P5 prediction cohorts at reference horizon $T_0$. | `src/metrics.py`<br>`scripts/run_table11.py` | `src/tables.py`<br>`code/analysis/verify_rq3.py` | `outputs/tables/RQ3_Table11_abstention.tex`<br>`outputs/tables/RQ3_Table11_summary.csv`<br>`outputs/tables/RQ3_prompt_sensitivity_summary.csv` | **100% Exact** | Qwen abstains 1,029/5,000 on Climate; Phi abstains 0/5,000. $S_{\text{prompt}}$ matches Table 11 range values across seeds. |
| **Table 12: Temporal Forensics (RQ5)** | 6 dataset movement signals: Lexical Novelty, base-2 JSD, $\Delta C_{100}$, $\Delta L$, Overlap Loss, and Marker Asymmetry. | `data/splits/` CSV text files across $T_0, T_1, T_2$. | `src/forensics.py`<br>`scripts/run_table12.py` | `src/tables.py`<br>`code/analysis/rq5_gate2_regenerate_forensics.py` | `outputs/tables/RQ5_Table12_temporal_forensics.tex`<br>`outputs/tables/RQ5_Table12_summary.csv` | **100% Exact** | Synchronized with Gate 2 re-audit. Climate long JSD: 0.3323; Novelty: 0.7177; Overlap loss: 0.8036. |
| **Figure 4: Risk-Coverage Curves** | Selective risk curves $R(c)$ as coverage $c$ varies from 0.05 to 1.0; highlights ranking reversals under temporal shift. | All 180 evaluation conditions across seeds 42–46. | `src/metrics.py`<br>`src/plotting.py` | `scripts/plot_figures.py`<br>`code/figures/plot_rq2_coverage.py`<br>`code/figures/rebuild_figure4_corrected.py` | `outputs/figures/RQ2_coverage_rankings.pdf`<br>`outputs/figures/RQ2_coverage_rankings.svg`<br>`outputs/figures/RQ2_coverage_rankings.png` | **100% Exact** | High-resolution multi-panel curves plotting mean selective risk across 5 seeds for all 4 models and horizons. |
| **Figure 5: Movement vs. Response** | Radar / profile visualization connecting dataset forensic movement to model degradation deltas. | Forensics summaries (Table 12) + Model deltas (Table 9). | `src/plotting.py`<br>`src/forensics.py` | `scripts/plot_figures.py`<br>`code/figures/rq5_alt_figure_verified_v4.py` | `outputs/figures/Figure5_movement_vs_response.pdf`<br>`outputs/figures/Figure5_movement_vs_response.svg`<br>`outputs/figures/Figure5_movement_vs_response.png` | **100% Exact** | Renders standardized z-score movement profiles alongside observed Macro-F1 delta responses. |
| **RQ2 Uncertainty Contrasts & Bootstrap** | 11 pairwise model retained-error contrasts and 95% bootstrap confidence intervals. | Synced prediction pairs matching document IDs. | `code/analysis/rq2_uncertainty_improved_v2.py` | `code/figures/plot_rq2_uncertainty_contrasts.py` | `C:\Research\RQ2_Uncertainty_Improved_V2\RQ2_uncertainty_contrasts.pdf`<br>`RQ2_focused_cluster_bootstrap.csv` | **100% Exact** | Synchronized document-cluster bootstrap (10,000 iterations) accounting for seed clustering and paired contrasts. |

---

## 2. Command-Level Execution Mapping

All paper outputs can be deterministically regenerated via the top-level driver:

```bash
# 1. Master execution reproducing all paper tables and figures:
python reproduce.py --all

# 2. Individual targeted executions:
python reproduce.py --audit     # Verifies data integrity & SHA-256 hashes
python reproduce.py --table7    # Table 7: Language Descriptors
python reproduce.py --table8    # Table 8: Canonical Reference Performance (RQ1)
python reproduce.py --table9    # Table 9: Temporal Macro-F1 Deltas (RQ2)
python reproduce.py --table10   # Table 10: ECE-10 & Selective Risk (RQ2)
python reproduce.py --table11   # Table 11: Explicit Abstention & Prompt Sensitivity (RQ3)
python reproduce.py --table12   # Table 12: Temporal Forensics Movement Evidence (RQ5)
python reproduce.py --figures   # Figure 4 (Risk-Coverage) & Figure 5 (Movement Profiles)
```
