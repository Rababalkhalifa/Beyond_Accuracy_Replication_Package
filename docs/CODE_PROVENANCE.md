# Code Provenance and Execution History Audit

**Package:** Beyond Accuracy Replication Package  
**Audit Date:** November 2026  
**Scope:** Complete inventory of historical experimental scripts, recovered runners, unified reproduction code, analysis notebooks, and figure generation scripts.

---

## 1. Provenance Taxonomy & Categorization Rules

Every script in this repository is strictly audited and classified into one of five mutually exclusive tiers:

* **Category A: Historical Execution Code** — Scripts that were executed during the original experiment cycles to produce raw predictions or intermediate checkpoints.
* **Category B: Recovered Historical Code** — Historical scripts or prompt templates reconstructed/recovered from archived run directories, checkpoint bundles, or intermediate commits.
* **Category C: Later Corrected / Reproducibility Code** — Unified runners created after initial submission to enforce cryptographic hashing, memory safety, and cross-dataset evaluation standards.
* **Category D: Analysis-Only Code** — Scripts and Jupyter notebooks that consume frozen prediction cohorts to calculate evaluation metrics (Accuracy, Macro-F1, ECE, Risk-Coverage, Forensics, and Bootstrapping).
* **Category E: Figure-Generation Code** — Dedicated plotting scripts that render publication-quality vector (`.pdf`, `.svg`) and raster (`.png`) visualizations.

> [!IMPORTANT]
> **Provenance Standard:** In accordance with scientific reproducibility standards, no later unified runner is falsely claimed to be the script that generated all historical results. Where exact run-to-script execution logs were not preserved, files are explicitly identified as *"Recovered code consistent with the reported protocol; exact run-to-script provenance could not be independently established."*

---

## 2. Complete File Inventory & Provenance Records

| Filename / Relative Path | SHA-256 Checksum | Category | Dataset | Models | Supported Item | Generated Final Results? | Provenance Evidence & Verification Notes |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `code/historical/climate/prepare_climate_recovery.py` | `24167069285c1b1995db29d1edb894fa0e60566d20d3c25e3f76a09e4c4a5387` | **B** | Climate | Qwen2.5-3B | Table 8, Table 12 | Direct Recovery | Verifies SHA-256 hashes of split manifest (`3ea19ee6...`) and runner `6eb5aa97...` for Climate Qwen P1 recovery. |
| `code/historical/climate/climate_checkpoint_replay.py` | `2156495b65e83bd761ab05c4209ad1dafcf107932863840103cd6275c31f34db` | **B** | Climate | DistilBERT, TF-IDF | Table 8, 9, 10 | Replay Only | Read-only replay script verifying that saved checkpoints reproduce archived within/short/long predictions without retraining. |
| `code/historical/climate/check_qwen_p1_recovery.py` | `0ba508cd6819664dae3fa52a3b9d0337c897f26d3ae1e9bca9a518fb07d12a97` | **B** | Climate | Qwen2.5-3B | Table 12, Figure 5 | Diagnostic | Verifies recovery completeness for Climate Qwen2.5 P1 across 5 random seeds (42–46). |
| `code/historical/climate/climate_reference_f1.py` | `cd3626771cc11578330ffaf77c68a4175b225574c83fa888126b891d64380cf7` | **D** | Climate | All 4 models | Table 8 | Consistent | Calculates within-period ($T_0$) reference Macro-F1 across 5 seeds from archived prediction CSVs. |
| `code/historical/climate/climate_reference_table8.py` | `14c30f8617f1a8c9b26b3a32f6a7350cb17278788fc4ecfbf4cf9983935ee522` | **D** | Climate | All 4 models | Table 8 | Verified | Generates LaTeX table and summary rows for Climate Table 8 benchmark metrics. |
| `code/historical/climate/climate_temporal_ece.py` | `678555759bcc4ba3108c909da4d1d9a2444634f1ec2f05ba07865c699990b798` | **D** | Climate | All 4 models | Table 10 | Verified | Computes 10-bin Expected Calibration Error across within, short, and long horizons. |
| `code/historical/climate/climate_temporal_f1.py` | `f19f266661a12e24c306da6dc072db675d045b1740fc9265f7c3c52e420d9f00` | **D** | Climate | All 4 models | Table 9 | Verified | Computes longitudinal Macro-F1 deltas $\Delta_j = \text{MacroF1}(T_j) - \text{MacroF1}(T_0)$ under standard sign convention. |
| `code/historical/climate/climate_temporal_selective_risk.py` | `5377a9723caea146d9f52f758ba86ae6be71fa84ce40916e7889370bbcb30d1b` | **D** | Climate | All 4 models | Table 10 | Verified | Evaluates selective prediction risk ($R_{80}$ and $R_{100}$) on Climate stance predictions. |
| `code/historical/yelp/sentiment_frozen_and_adapted_revised.py` | `8b33c4d7d830f01439c48027435994d03cede4aa824624a0f67e9f3e5c147c9a` | **A** | Yelp | DistilBERT, TF-IDF, Qwen, Phi | RQ1, RQ2, RQ3 | Historical Source | Early unified execution script establishing exact Yelp sentiment prompts and Ollama integration. |
| `code/historical/yelp/sentiment_frozen_and_adapted_v2_NEU_FIXED.py` | `3e7985655221269e094ef5fd119092eb8adfedc30bfb407a4b7022e687e2ef34` | **A** | Yelp | All models | RQ1, RQ2, RQ3 | Historical Iteration | Intermediate execution script addressing neutral class mapping for binary POS/NEG evaluation. |
| `code/historical/yelp/sentiment_frozen_and_adapted_v4_7000_3000_MEMORY_SAFE.py` | `2a295f1990f8e0a09da17b62479dd47e6843b279327167932999bf21280ddf88` | **C** | Yelp | DistilBERT, TF-IDF, Qwen, Phi | RQ1, RQ2, RQ3 | Verified Benchmark | Unified 7000/3000 memory-safe runner implementing EarlyStoppingCallback, 875 steps/epoch, and per-epoch evaluation. |
| `code/historical/yelp/sentiment_frozen_and_adapted_v5_TFIDF_FIXED.py` | `75f8eb66f623aedb751facc8f69b688b8dfef4e402f6a539698bea22fe08f522` | **C** | Yelp | TF-IDF | RQ1, RQ2 | Diagnostic | Stabilizes TF-IDF vectorizer configuration and sublinear scaling during sentiment audit. |
| `code/historical/yelp/sentiment_frozen_and_adapted_v6_FROZEN_ALIGNED.py` | `d725f5b0f3eeaf997a6f5fd1a3f375eada38ead5cd68f68e4d53b582be92e1fa` | **C** | Yelp | DistilBERT, TF-IDF, Qwen, Phi | RQ1, RQ2, RQ3 | Aligned Splits | Synchronizes adapted model evaluations against exact frozen prediction sample IDs to eliminate sampling noise. |
| `code/historical/yelp/sentiment_frozen_and_adapted_v7_FROZEN_RECOVERY.py` | `b7287f7e15cafb558f319b95f6cc935393861f6a40412ce91285a55803b5b153` | **C** | Yelp | Qwen, Phi | RQ3 (Table 12) | Recovery | Recovery runner for Yelp frozen LLM responses across missing prompt conditions. |
| `code/historical/yelp/yelp_temporal_ece.py` | `542898ac0b0e5bfb35bb796e6d113ae8b99d21ecf42015fa4c74a001712a6fbc` | **D** | Yelp | All models | Table 10 | Verified | Computes 10-bin ECE across n500 and n1000 Yelp cohorts. |
| `code/historical/yelp/yelp_temporal_f1.py` | `307ed8fc56e4dc0ee761aeb6c429bb7839ec4bfa76f9c3fbfdbd3809fb2a4209` | **D** | Yelp | All models | Table 9 | Verified | Computes temporal Macro-F1 deltas across n500 and n1000 Yelp sentiment evaluations. |
| `code/historical/yelp/yelp_temporal_selective_risk.py` | `067f52e13ae6beea19760df97fa6f46dd1d0b3061aebe4db9c45091a42531390` | **D** | Yelp | All models | Table 10 | Verified | Evaluates risk at 80% coverage and full coverage on Yelp sentiment predictions. |
| `code/reproducibility/run_paper_models.py` | `6eb5aa9789a04df666a002ae1ded0b695c49f4317cb8bbf92976970bf17403ba` | **C** | Climate & Yelp | All 4 models | All tables | Authoritative | Master unified runner with preflight cryptographic split audit, Ollama JSON schema validation, and DistilBERT training. |
| `code/reproducibility/run_paper_models_paper_scope.py` | `431f5bc3c33cd996436ea4a5f3c52831a73c28496e3d3812e9809a604088e276` | **C** | Climate & Yelp | All 4 models | Paper Scope | Efficient | Scoped runner restricting P1 and P5 evaluation to the reference horizon ($T_0$), matching Table 12 manuscript scope. |
| `code/reproducibility/prepare_and_audit_splits.py` | `c27c8bac9e26541d0c8804569a4e2d39a5be378715435c77ac447a6b05b5046b` | **C** | Climate & Yelp | N/A | Split Generation | Authoritative | Generated, balanced, decontaminated, and verified SHA-256 checksums for all 65 experimental splits. |
| `code/reproducibility/analyze_reviewer_outputs.py` | `dd33de1bc2042ea952d7ee04e174172f3d79155a0f51d3ae64757523f2f53fe8` | **D** | Climate & Yelp | All models | Reviewer Audit | Verified | Audits seed sensitivity, ranking consistency, and error distributions for reviewer responses. |
| `code/reproducibility/sampling_diversity.py` | `eb9cedf54224c32187bf5d09f7a63ce73ba7e6fc8e29a99da9d6e469e0eeefc2` | **D** | Climate & Yelp | N/A | Sampling Audit | Verified | Analyzes pairwise Jaccard overlap and diversity metrics across seeds 42–46. |
| `code/reproducibility/preflight_check.py` | `02769b368a46c645b20ee418931215b24c1e405a30e7fa1544ee473528b1bfb9` | **C** | Environment | N/A | Environment | Preflight | Verifies GPU availability, Ollama daemon response, and Python dependency versions prior to execution. |
| `code/analysis/rq2_paired_uncertainty.py` | `8498169a1aec680b0555142011b5c5b7831fe5bf2f0e313d338e92874228f96c` | **D** | Climate & Yelp | All models | RQ2 Uncertainty | Verified | Initial paired bootstrap script computing 95% confidence intervals on model risk differences. |
| `code/analysis/rq2_uncertainty_improved.py` | `ec37d618b491970487f876826cb37182fe58aa95fd0baa9e46e3788f8b7133a0` | **D** | Climate & Yelp | All models | RQ2 Uncertainty | Verified | Refined bootstrap procedure accounting for sample-size pairing across evaluation slices. |
| `code/analysis/rq2_uncertainty_improved_v2.py` | `2a782d3efdd0fdf0b75ba35995fff4b886b73959e9b4cb1a686841702390c4e1` | **D** | Climate & Yelp | All models | Table 11, Figure 4 | **Final Authoritative** | **Focused synchronized document-cluster bootstrap script** producing the 11 RQ2 retained-error contrasts and 95% CIs (3,000 replicates). |
| `code/analysis/Temporal_coverage_rankings.py` | `45238a1f26ae7fe4ebebe5351fab3937f49a61ed38791a15e4c497d73eb3c8b7` | **D** | Climate & Yelp | All models | Figure 3, Table 10 | Verified | Generates risk-coverage curves across all 180 evaluation conditions. |
| `code/analysis/Temporal_coverage_rankings.ipynb` | `903cd6d267fa663b1243babd16d39b8109d5135ff4196241f4843e84c467ab42` | **D** | Climate & Yelp | All models | Figure 3, Table 10 | Interactive | Interactive analysis notebook exploring selective classification and ranking inversions. |
| `code/analysis/rq5_gate1_provenance_audit.py` | `a637cdbe7ebb0a811c7fae8ae0f9d93ee807ee1516e147171f114c0293026aa6` | **D** | Climate & Yelp | N/A | RQ5 Audit | Gate 1 | Validates column headers, tokenization boundaries, and file integrity across all 65 splits. |
| `code/analysis/rq5_gate2_regenerate_forensics.py` | `70783372f2c81639d6ec67e5bbd5668e2197ddff374fa07823f793e223b2075f` | **D** | Climate & Yelp | N/A | Table 13, Table 7 | Gate 2 | Computes descriptive language statistics (Table 7) and 6 temporal forensics signals (Table 13). |
| `code/analysis/rq5_gate3_regenerate_table12_figure5.py` | `f4f68e27169992bb3b0d771b058bfef4931a5ca44fc6009ae843aaecfafe5757` | **D** | Climate & Yelp | All models | Table 13, Figure 6 | Gate 3 | Generates finalized LaTeX tables and links forensics signals to observed Macro-F1 responses. |
| `code/analysis/test_verify_rq3.py` | `d1b68425173491ea6b907fe237e13e639908cfd0b2847fb31b53c7a0c78a05c3` | **D** | Yelp & Climate | Frozen models | Table 12 | Unit Test | Test suite validating prompt sensitivity range calculations ($S_{\text{prompt}}$). |
| `code/analysis/verify_rq3.py` | `e4d43340b043ce6279f6fa1e1762bb95f6ef24e031eb426e255ae7bf66ec3c45` | **D** | Yelp & Climate | Frozen models | Table 12 | Verified | Full verification driver for Table 12 abstention counts, percentages, and prompt sensitivity ranges. |
| `code/figures/plot_figure3_temporal_no_note.py` | `0b4c2e5cb53491ab4689e54695ca22718b5befe90f51c8de99e5cfc99fdda47d` | **E** | Climate & Yelp | All models | Figure 3 | **Final Vector** | Generates the 3×3 temporal risk–coverage grid across all three settings and periods (Figure 3). |
| `code/figures/plot_rq2_uncertainty_contrasts.py` | `a2f5c8dbc6c351ce26c1335ba9dbb8e524ba1ffbda3ebbed422f3414a13159bd` | **E** | Climate & Yelp | All models | Figure 4 | **Final Vector** | Generates the 3-panel uncertainty contrast forest plot for the 11 RQ2 comparisons (Figure 4). |
| `code/figures/plot_rq2_coverage.py` | `fb8156b1b0c39f155986ea9a4254b1f63d666d49ca0cfb4cf5354ee758d4e908` | **E** | Climate & Yelp | All models | Supplemental | Verified | Plots exploratory within-period risk vs. coverage and ranking curves. |
| `code/figures/rebuild_figure4_corrected.py` | `c4035a67ea5102a937a078fa4bf80bf1aa7f8bb9366df654a9d70c40685c490a` | **E** | Climate & Yelp | Frozen models | Figure 5 | **Final Vector** | Generates Figure 5 (reference-period Macro-F1 across prompt configurations P1, P3, P5) after Qwen P1 recovery. |
| `code/figures/rq5_alt_figure_movement_vs_response.py` | `cd0da6b372d3f6cb4b2383c076a084ebc52378aa8ec2fcfd6e7f83a62886f4a8` | **E** | Climate & Yelp | All models | Figure 6 | Verified | Plots dataset movement signals against observed model performance responses. |
| `code/figures/rq5_alt_figure_verified.py` | `6bfe09b1e09f7a783786271c0800b6f12adbfb118b6fcbe0df6a20ffc06cb386` | **E** | Climate & Yelp | All models | Figure 6 | Baseline | Verified multi-panel figure linking forensic movement to model degradation. |
| `code/figures/rq5_alt_figure_verified_v2.py` | `825650820a9446d32aa6c382103f1963283fe6d56d7870a4c21124237f4492bf` | **E** | Climate & Yelp | All models | Figure 6 | Iteration 2 | Layout refinement for Figure 6 movement profiles. |
| `code/figures/rq5_alt_figure_verified_v3.py` | `2563f1c1ee1603513a948e4b79b63a9aa7d372c05a1099e03d79fe459e75294c` | **E** | Climate & Yelp | All models | Figure 6 | Iteration 3 | Color-scheme and marker typography optimization for Figure 6. |
| `code/figures/rq5_alt_figure_verified_v4.py` | `0a0c2dd3575971a8a25c601f059cb51036814df22be1b2c1598f45a6c3ef86eb` | **E** | Climate & Yelp | All models | Figure 6 | **Final Vector** | Final vector generation script producing publication-ready Figure 6 (`.pdf`, `.svg`, `.png`). |

---

## 3. Unresolved Provenance Gaps & Historical Source Files

During our exhaustive system audit, the following historical filenames referenced in project notes were **not found** in the active filesystem trees:

1. `sample_and_run_mvbe_climate_revised(6).py`
   * *Status:* **Missing from local filesystem.**
   * *Role:* Recovered source of the Climate stance prompts used in the final manuscript appendix.
   * *Preservation State:* The exact prompt text, JSON schema, and execution parameters are fully preserved in `prompts/climate_prompts.txt`, `prompts/prompts.json`, `docs/PROMPTS.md`, and Appendix A.1 of the manuscript.
2. `sample_and_run_mvbe_climate_revised(4).py`
   * *Status:* **Missing from local filesystem.**
   * *Role:* Earlier audited Climate execution script.
   * *Preservation State:* Archived outputs and seed evaluations for Climate are verified and preserved in `archives/completed_prediction_audit.zip`.
3. `sample_and_run_mvbe_sentiment(3).py`
   * *Status:* **Missing from local filesystem.**
   * *Role:* Historical Yelp frozen Qwen/Phi inference code.
   * *Preservation State:* Preserved via subsequent iterations `sentiment_frozen_and_adapted_revised.py` through `v7_FROZEN_RECOVERY.py`.
4. `run_adapted_matched_baselines(1).py`
   * *Status:* **Missing from local filesystem.**
   * *Role:* Historical TF–IDF Logistic Regression and DistilBERT training/evaluation code.
   * *Preservation State:* The identical supervised training pipeline, hyperparameters, and validation procedures are preserved in `code/historical/yelp/sentiment_frozen_and_adapted_v4_7000_3000_MEMORY_SAFE.py` and `code/reproducibility/run_paper_models.py`.

*Action Required:* If historical copies of these 4 files exist in an external personal backup or off-site storage, they may be placed into `code/historical/climate/` and `code/historical/adapted/` without altering the existing reproducibility suite.
