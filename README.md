# Aging Gracefully? Reliability Profiling and Temporal Forensics of Frozen and Adapted Text Classifiers

[![SSRN Preprint](https://img.shields.io/badge/SSRN-Preprint%207560344-blue.svg)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7560344)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Replication Status](https://img.shields.io/badge/Replication-100%25%20Verified-success.svg)](reproduce.py)

Official publication and reproduction package for the research paper:  
**"Beyond Accuracy in Dynamic Sentiment Analysis: Reliability Profiling and Temporal Forensics for Frozen and Adapted Models"**  
Author: **Rabab Alkhalifa** (*Department of Computer Science, College of Computer Science and Information Technology, Imam Abdulrahman Bin Faisal University, Dammam, Saudi Arabia*)  
Preprint: [SSRN Abstract 7560344](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7560344) | DOI: [10.2139/ssrn.7560344](https://doi.org/10.2139/ssrn.7560344)

---

## Table of Contents

1. [Paper Scope & Research Questions](#1-paper-scope--research-questions)
2. [Repository Structure](#2-repository-structure)
3. [Environment Setup](#3-environment-setup)
4. [Required Python Packages](#4-required-python-packages)
5. [Ollama Installation & Service Setup](#5-ollama-installation--service-setup)
6. [Required Model Tags & Immutable Manifests](#6-required-model-tags--immutable-manifests)
7. [Data Preparation & Sampling Protocol](#7-data-preparation--sampling-protocol)
8. [Historical Code Provenance & Script Inventory](#8-historical-code-provenance--script-inventory)
9. [Execution & Reproduction Commands](#9-execution--reproduction-commands)
10. [Table & Figure Reproduction Map](#10-table--figure-reproduction-map)
11. [Reproducibility Limitations](#11-reproducibility-limitations)
12. [Citation](#12-citation)

---

## 1. Paper Scope & Research Questions

Evaluating natural language processing classifiers strictly on in-domain reference accuracy conceals critical operational vulnerabilities. When models encounter temporally displaced language streams, performance degrades non-uniformly across model families, confidence estimates lose calibration, and prompt formulations alter predictive behavior.

This study conducts a rigorous comparative audit across **four distinct model architectures**, **two domain tasks**, **five random sampling seeds (42–46)**, and **three temporal horizons** (within-period reference $T_0$, short-distance target $T_1$, and long-distance target $T_2$):

* **Frozen Prompted Local LLMs:**
  * `qwen2.5:3b-instruct` (Alibaba Cloud, 3.1B parameters, `Q4_K_M` quantization)
  * `phi3.5:latest` (Microsoft Research, 3.8B parameters, `Q4_0` quantization)
* **Task-Adapted Supervised Baselines:**
  * DistilBERT (`distilbert-base-uncased`, Hugging Face, 66M parameters)
  * TF–IDF Logistic Regression (Unigram + Bigram, Sublinear TF, $L_2$ Regularization)
* **Datasets & Tasks:**
  * **Climate:** Social media stance classification (`BELIEVER` vs. `DENIER`)
  * **Yelp:** Consumer review sentiment classification (`POS` vs. `NEG`)
* **Prompt Formulations:**
  * `p1_minimal` (minimal task instruction)
  * `p3_json_strict` (canonical zero-shot JSON format with verbalized confidence)
  * `p5_abstain` (explicit confidence thresholding with permitted abstention)

### Core Research Questions Addressed

* **RQ1 (In-Domain Performance):** How do small task-adapted models compare with compact frozen instruction-tuned LLMs when evaluated on temporally matched in-domain test data?
* **RQ2 (Temporal Robustness & Reliability):** How severely does temporal shift degrade classification accuracy, Macro-F1 ($\Delta_j = \text{MacroF1}(T_j) - \text{MacroF1}(T_0)$), Expected Calibration Error (ECE-10), and confidence-based selective risk ($R_{80}, R_{100}$)?
* **RQ3 (Prompt Sensitivity & Explicit Abstention):** Does permitting explicit abstention (`p5_abstain`) reduce post-abstention error, and how sensitive are frozen LLMs to prompt variation ($S_{\text{prompt}}$)?
* **RQ4 (Supervised vs. Verbalized Uncertainty):** How do supervised softmax probabilities compare to LLM verbalized self-reported confidence in selective prediction workflows?
* **RQ5 (Temporal Forensics):** Can observed empirical performance decay be explained through corpus-level distributional shifts (lexical novelty, base-2 JSD, top-100 term concentration, document length change, vocabulary overlap loss, and class-marker asymmetry)?

---

## 2. Repository Structure

The repository is organized to strictly isolate historical experimental execution scripts, later reproducibility wrappers, analysis/forensics code, verified data splits, and cryptographically anchored prediction cohorts:

```text
Beyond_Accuracy_Replication_Package/
├── README.md                      # Academic reproduction guide & equation documentation
├── reproduce.py                   # Master replication CLI driver (100% self-contained)
├── requirements.txt               # Pinned Python package dependencies
├── environment.yml                # Conda environment definition
│
├── code/                          # Complete script repository (preserved & segregated)
│   ├── historical/                # Exact, byte-for-byte preserved historical code
│   │   ├── climate/               # Historical Climate stance execution & recovery scripts
│   │   ├── yelp/                  # Historical Yelp sentiment execution & recovery scripts
│   │   └── adapted/               # Historical adapted baseline documentation
│   ├── reproducibility/           # Authoritative unified runners & audit scripts
│   │   ├── run_paper_models.py    # Unified Ollama & HF runner for all paper models
│   │   ├── prepare_and_audit_splits.py # Verifies split generation & cryptographic parity
│   │   └── preflight_check.py     # System environment & model availability verification
│   ├── analysis/                  # Statistical verification, uncertainty & forensics
│   │   ├── rq2_uncertainty_improved_v2.py # Synchronized document bootstrap for RQ2 contrasts
│   │   ├── Temporal_coverage_rankings.py  # Risk-coverage ranking inversion engine
│   │   ├── Temporal_coverage_rankings.ipynb # Interactive ranking analysis notebook
│   │   └── rq5_gate*.py           # Three-gate temporal forensics audit pipeline
│   └── figures/                   # Scripts generating manuscript Figures 4 and 5
│       ├── rebuild_figure4_corrected.py   # Canonical Figure 4 generator
│       └── rq5_alt_figure_verified.py     # Canonical Figure 5 generator
│
├── config/                        # Verified execution hyperparameters & machine settings
│   ├── EXECUTION_SETTINGS.md      # Full prose documentation of model & system settings
│   └── run_settings.json          # Machine-readable JSON runtime parameters
│
├── prompts/                       # Exact verbatim prompt templates (No paraphrasing)
│   ├── climate_prompts.txt        # Verbatim P1, P3, P5 prompts for Climate (BELIEVER/DENIER)
│   ├── yelp_prompts.txt           # Verbatim P1, P3, P5 prompts for Yelp (POS/NEG)
│   └── prompts.json               # Structured JSON prompts with JSON schemas
│
├── docs/                          # Audit reports, schemas, and methodology documentation
│   ├── CODE_PROVENANCE.md         # Exhaustive audit of all 41 code files with SHA256 hashes
│   ├── DATA.md                    # Data sources, chronological windows, and label mapping
│   ├── SAMPLING.md                # 5-seed sampling protocol, class quotas, and nested pairing
│   ├── OUTPUT_SCHEMA.md           # 21-column schema for saved prediction outputs
│   ├── PAPER_REPRODUCTION_MAP.md  # Detailed mapping for Tables 7–12, Figs 4–5, and contrasts
│   ├── ENVIRONMENT.md             # Hardware specifications, OS, and Ollama configuration
│   └── PROVENANCE_DISCREPANCIES.md# Technical discrepancies, sign conventions, and resolutions
│
├── data/                          # Audited experimental data splits
│   ├── README.md                  # Split specifications & sampling overview
│   └── splits/                    # 65 authoritative CSV splits + checksums_sha256.txt
│       ├── climate/               # Climate stance splits (seeds 42..46: train, within, short, long)
│       └── yelp/                  # Yelp sentiment splits (seeds 42..46: train, within, short, long)
│
├── archives/                      # Cryptographically verified prediction cohorts (ZIP)
│   ├── completed_prediction_audit.zip           # Climate predictions (adapted + frozen P1/P3/P5)
│   ├── original_yelp_frozen_predictions.zip     # Yelp frozen predictions (P1/P3/P5)
│   ├── yelp_prediction_alignment.zip            # Yelp aligned supervised predictions
│   ├── code_and_remaining_results.zip           # Latency metrics & supervised summaries
│   └── yelp_results_audit.zip                   # Audit manifests & cross-model mapping logs
│
├── src/                           # Pure modular scientific library
│   ├── metrics.py                 # Equations 1–11: Accuracy, Macro-F1, Deltas, ECE-10, S_prompt
│   ├── descriptors.py             # Table 7: Language statistics (tokens/doc, vocab, TTR, mass)
│   ├── forensics.py               # Equations 12–17: Novelty, JSD_2, Delta C100, Delta L, Churn
│   ├── loader.py                  # Seamless archive reader & prediction cohort extractor
│   ├── audit.py                   # SHA-256 verification and cross-model ID matching
│   ├── tables.py                  # Formats LaTeX & CSV for Tables 7–12
│   └── plotting.py                # High-res vector/bitmap plotting for Figures 4 and 5
│
├── scripts/                       # Modular standalone execution scripts
│   ├── run_experiments.py         # End-to-end model execution (DistilBERT, TF-IDF, Ollama)
│   └── run_table7.py ... run_table12.py # Modular table generation scripts
│
└── outputs/                       # Regenerated publication tables and figures
    ├── tables/                    # LaTeX (.tex) and summary (.csv) files for all tables
    └── figures/                   # Vector (.pdf, .svg) and bitmap (.png) figures
```

---

## 3. Environment Setup

The primary verification suite (`reproduce.py`) evaluates all 180 experimental conditions, computes confidence intervals, reproduces Tables 7–12, and renders Figures 4–5 directly from the verified prediction cohorts using pure Python and standard scientific packages. It requires no specialized GPU hardware.

### Option A: Python Virtual Environment (venv)
```bash
# Clone the repository
git clone https://github.com/Rababalkhalifa/Beyond_Accuracy_Replication_Package.git
cd Beyond_Accuracy_Replication_Package

# Create and activate a clean Python 3.10 virtual environment
python -m venv venv
# On Linux/macOS:
source venv/bin/activate
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# Upgrade pip and install pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Option B: Conda Environment
```bash
conda env create -f environment.yml
conda activate beyond-accuracy
```

---

## 4. Required Python Packages

The package dependencies are pinned in [`requirements.txt`](file:///C:/Research/Beyond_Accuracy_Replication_Package/requirements.txt):

```text
# Core Scientific & Reproduction Stack
numpy>=1.24.0,<2.0.0
pandas>=2.0.0,<3.0.0
scipy>=1.10.0,<2.0.0
scikit-learn>=1.3.0,<1.6.0
matplotlib>=3.7.0,<4.0.0
seaborn>=0.12.0,<0.14.0

# Optional Model Retraining & Live Prompting Stack
torch>=2.0.0
transformers>=4.35.0
requests>=2.28.0
tqdm>=4.65.0
```

* For fast reproduction of paper findings, tables, figures, and bootstrap statistics: only the **Core Stack** is required.
* For retraining DistilBERT or sending live HTTP requests to Ollama: the **Optional Stack** and a running Ollama daemon are required.

---

## 5. Ollama Installation & Service Setup

To re-run the frozen LLMs from scratch, install Ollama and configure the local service:

1. **Install Ollama:** Download and install Ollama from [ollama.com](https://ollama.com).
2. **Start the Ollama Daemon:**
   ```bash
   ollama serve
   ```
   By default, the service listens at `http://localhost:11434`.
3. **Verify Daemon Connectivity:**
   ```bash
   curl http://localhost:11434/api/tags
   ```

---

## 6. Required Model Tags & Immutable Manifests

The study evaluated exact model tags pulled via Ollama. Weight manifests and quantization parameters were extracted directly from the installed Ollama library:

| Model Tag | Parameters | Quantization | Ollama Manifest Digest (SHA-256) | Native Context |
| :--- | :---: | :---: | :--- | :---: |
| `qwen2.5:3b-instruct` | 3.09B | **`Q4_K_M`** | `357c53fb659c5076de1d65ccb0b397446227b71a42be9d1603d46168015c9e4b` | 32,768 tokens |
| `phi3.5:latest` | 3.82B | **`Q4_0`** | `61819fb370a3c1a9be6694869331e5f85f867a079e9271d66cb223acb81d04ba` | 131,072 tokens |

To pull these exact models:
```bash
ollama pull qwen2.5:3b-instruct
ollama pull phi3.5:latest
```

*Note on Quantization:* Quantization levels (`Q4_K_M` for Qwen2.5 and `Q4_0` for Phi-3.5) are determined by the compiled GGUF model container distributed under the corresponding Ollama tag. See [`config/EXECUTION_SETTINGS.md`](file:///C:/Research/Beyond_Accuracy_Replication_Package/config/EXECUTION_SETTINGS.md) for full metadata.

---

## 7. Data Preparation & Sampling Protocol

Detailed data acquisition, chronological horizons, and sampling specifications are documented in [`docs/DATA.md`](file:///C:/Research/Beyond_Accuracy_Replication_Package/docs/DATA.md) and [`docs/SAMPLING.md`](file:///C:/Research/Beyond_Accuracy_Replication_Package/docs/SAMPLING.md).

### Chronological Horizons & Splits
* **Climate Stance:** Social media posts categorized into `BELIEVER` vs. `DENIER`.
  * Training: Historical partition (2008–2016), 10,000 documents (5,000 per class).
  * Within-Period Reference ($T_0$): Historical partition held-out, 1,000 documents (500 per class).
  * Short-Distance Target ($T_1$): Near-future partition (2017), 1,000 documents (500 per class).
  * Long-Distance Target ($T_2$): Distant partition (2018), 1,000 documents (500 per class).
* **Yelp Sentiment:** Restaurant and business reviews categorized into `POS` (5 stars) vs. `NEG` (1 star).
  * Training: In-period partition (2005–2012), 7,000 documents (3,500 per class).
  * Within-Period Reference ($T_0$): Held-out in-period partition (2005–2012), 2,000 documents (1,000 per class).
  * Short-Distance Target ($T_1$): Subsequent partition (2013–2015), 2,000 documents (1,000 per class).
  * Long-Distance Target ($T_2$): Distant partition (2016–2019), 2,000 documents (1,000 per class).
  * Evaluated under both `n500` (nested 1,000-document subset) and `n1000` (full 2,000-document set).

### Sampling Seeds, Dehydrated Manifests & Data Availability
All sampling was performed across five random seeds: `42, 43, 44, 45, 46`. Following standard academic NLP practice for third-party text corpora (and in compliance with the [Yelp Dataset Terms of Use](https://www.yelp.com/dataset) and social-media platform redistribution policies), the 65 CSV files in [`data/splits/`](data/splits/) and the prediction cohorts in [`archives/`](archives/) are distributed in **dehydrated form**: each row records `example_id`, gold `label`, cryptographic text digest (`text_sha256` or `text_sha1`), `token_count`, `char_length`, and temporal metadata, while omitting raw third-party review and post text.

* **Out-of-the-Box Reproduction (`reproduce.py --all`):** All audits, Tables 7–12, and Figures 4–5 run **100% out-of-the-box** without raw text using the dehydrated split manifests, `data/splits/checksums_sha256.txt`, the prediction archives, and the precomputed non-reconstructible token statistics in `data/splits/forensics_and_descriptors_cache.json`:
  ```bash
  python scripts/audit_dataset_splits.py
  python reproduce.py --all
  ```
* **Rehydrating Raw Text (for Model Retraining or Live Prompting):** Researchers who wish to re-fit TF–IDF / DistilBERT or re-run live Ollama prompting from scratch can restore the `text` column using [`scripts/hydrate_splits.py`](scripts/hydrate_splits.py):
  1. **From official source datasets:** Download the source corpora from their original providers (e.g., [Yelp Open Dataset](https://www.yelp.com/dataset)) and match rows by SHA-256 / SHA-1 text digest:
     ```bash
     python scripts/hydrate_splits.py --source-files /path/to/source_corpus.csv
     ```
  2. **Available upon academic request:** The full-text split archive for non-commercial academic verification can be requested directly from the corresponding author and hydrated via:
     ```bash
     python scripts/hydrate_splits.py --fulltext-splits-dir /path/to/full_text
     ```
     Both hydration methods are automatically verified against `data/splits/checksums_fulltext_sha256.txt`.

---

## 8. Historical Code Provenance & Script Inventory

To ensure scientific transparency, this repository preserves the exact code that was used or recovered during the research project **without modifying, cleaning up, refactoring, or silently renaming** historical scripts.

Every code artifact is cataloged in [`docs/CODE_PROVENANCE.md`](file:///C:/Research/Beyond_Accuracy_Replication_Package/docs/CODE_PROVENANCE.md) with its exact filename, SHA-256 hash, dataset, model, role, provenance status (A: Historical execution, B: Recovered historical, C: Later reproducibility runner, D: Analysis-only, E: Figure generation), and verification evidence.

### Summary of Script Categories

1. **Preserved Historical Scripts (`code/historical/`):**
   * Climate execution & recovery: `prepare_climate_recovery.py`, `climate_checkpoint_replay.py`, `check_qwen_p1_recovery.py`, `climate_reference_f1.py`, `climate_reference_table8.py`, `climate_temporal_ece.py`, `climate_temporal_f1.py`, `climate_temporal_selective_risk.py`.
   * Yelp execution & recovery: `sentiment_frozen_and_adapted_revised.py`, `v2_NEU_FIXED.py`, `v4_7000_3000_MEMORY_SAFE.py`, `v5_TFIDF_FIXED.py`, `v6_FROZEN_ALIGNED.py`, `v7_FROZEN_RECOVERY.py`, `yelp_temporal_ece.py`, `yelp_temporal_f1.py`, `yelp_temporal_selective_risk.py`.
2. **Unified Reproducibility Runners (`code/reproducibility/`):**
   * `run_paper_models.py`: Unified end-to-end execution pipeline supporting DistilBERT, TF–IDF, and Ollama LLMs with identical prompt logic.
   * `prepare_and_audit_splits.py`: Split generation and verification engine.
3. **Statistical Analysis & Uncertainty Contrasts (`code/analysis/`):**
   * `rq2_uncertainty_improved_v2.py`: Focused synchronized document-cluster bootstrap engine generating empirical 95% confidence intervals for all 11 RQ2 contrasts reported in Section 5.2.
   * `Temporal_coverage_rankings.py` / `.ipynb`: Selective risk-coverage ranking inversion analysis.
   * `rq5_gate1_provenance_audit.py`, `rq5_gate2_regenerate_forensics.py`, `rq5_gate3_regenerate_table12_figure5.py`: Three-gate temporal forensics audit pipeline.
4. **Manuscript Figure Generators (`code/figures/`):**
   * `rebuild_figure4_corrected.py`: Generates Figure 4 (Selective Risk vs. Coverage Curves).
   * `rq5_alt_figure_verified.py`: Generates Figure 5 (Temporal Movement Profiles vs. Model Degradation).

*Unresolved Historical Scripts Notice:* Four earlier intermediate filenames (`sample_and_run_mvbe_climate_revised(6).py`, `sample_and_run_mvbe_climate_revised(4).py`, `sample_and_run_mvbe_sentiment(3).py`, and `run_adapted_matched_baselines(1).py`) are documented in [`code/historical/climate/MISSING_HISTORICAL_FILES.md`](file:///C:/Research/Beyond_Accuracy_Replication_Package/code/historical/climate/MISSING_HISTORICAL_FILES.md). Their exact prompt templates, hyperparameters, and operational logic are preserved in `run_paper_models.py`, `prompts/`, and `config/EXECUTION_SETTINGS.md`.

---

## 9. Execution & Reproduction Commands

### Master One-Click Reproduction Driver
To reproduce all paper tables, figures, and audit checks in ~15–25 seconds:
```bash
python reproduce.py --all
```

### Reproducing Historical & Unified Pipelines Directly

#### 1. Running the Unified Paper Runner (Inference / Retraining from Scratch)
```bash
# Run both adapted baselines and frozen LLMs across all 5 seeds
python code/reproducibility/run_paper_models.py --mode both

# Run only task-adapted models (TF--IDF and DistilBERT)
python code/reproducibility/run_paper_models.py --mode adapted

# Run only frozen Ollama LLMs (qwen2.5:3b-instruct and phi3.5:latest)
python code/reproducibility/run_paper_models.py --mode frozen --ollama-host http://localhost:11434

# Run specific dataset or seeds
python code/reproducibility/run_paper_models.py --datasets climate --seeds 42 43 --prompt-ids p3_json_strict
```

#### 2. Running the Historical Yelp Memory-Safe Pipeline
```bash
python code/historical/yelp/sentiment_frozen_and_adapted_v4_7000_3000_MEMORY_SAFE.py
```

#### 3. Running the Synchronized Document-Cluster Bootstrap (RQ2 Uncertainty Contrasts)
```bash
python code/analysis/rq2_uncertainty_improved_v2.py
```

---

## 10. Table & Figure Reproduction Map

The following table maps every empirical table, figure, and quantitative claim in the manuscript to its reproduction command, script, and output artifact:

| Manuscript Item | Research Question | Underlying Mathematical Equation | CLI Reproduction Command | Output File |
| :--- | :---: | :---: | :--- | :--- |
| **Audit** | — | — | `python reproduce.py --audit` | Console report (`PASS: 65/65 splits, 180 conditions`) |
| **Table 7** | Descriptors | Eq. (1)–(3), Word Regex | `python reproduce.py --table7` | `outputs/tables/Table7_descriptors.tex`<br>`outputs/tables/Table7_descriptors_summary.csv` |
| **Table 8** | RQ1 | Eq. (4) [Macro-F1], Eq. (5) [Acc], Eq. (9) [ECE] | `python reproduce.py --table8` | `outputs/tables/RQ1_Table8.tex`<br>`outputs/tables/RQ1_Table8.csv` |
| **Table 9** | RQ2 | Eq. (6) [$\Delta_j = \text{Target} - \text{Ref}$] | `python reproduce.py --table9` | `outputs/tables/RQ2_Table9.tex`<br>`outputs/tables/RQ2_Table9.csv` |
| **Table 10** | RQ2 | Eq. (9) [ECE], Eq. (11) [$R(c)$ at $c \in \{0.8, 1.0\}$] | `python reproduce.py --table10` | `outputs/tables/RQ2_Table10_temporal_reliability.tex`<br>`outputs/tables/RQ2_verified_seed_metrics.csv` |
| **Table 11** | RQ3 | Eq. (10) [$S_{\text{prompt}}$], Explicit Abstention $A$ | `python reproduce.py --table11` | `outputs/tables/RQ3_Table11_abstention.tex`<br>`outputs/tables/RQ3_prompt_sensitivity_summary.csv` |
| **Table 12** | RQ5 | Eq. (12)–(17) [Novelty, JSD$_2$, $\Delta C_{100}$, $\Delta L$, Churn] | `python reproduce.py --table12` | `outputs/tables/RQ5_Table12_temporal_forensics.tex`<br>`outputs/tables/RQ5_Table12_summary.csv` |
| **Figure 4** | RQ2 / RQ4 | Eq. (11) [Selective Risk $R(c)$ vs. Coverage $c$] | `python reproduce.py --figures` | `outputs/figures/RQ2_coverage_rankings.pdf`<br>`outputs/figures/RQ2_coverage_rankings.svg` |
| **Figure 5** | RQ5 | Forensic shifts vs. empirical accuracy drop | `python reproduce.py --figures` | `outputs/figures/Figure5_movement_vs_response.pdf`<br>`outputs/figures/Figure5_movement_vs_response.svg` |
| **RQ2 Contrasts**| RQ2 | Synchronized document-cluster bootstrap (1,000 resamples)| `python code/analysis/rq2_uncertainty_improved_v2.py` | Console report & contrast summary table |

For full tabular breakdowns of the 11 RQ2 bootstrap contrasts and exact numerical comparisons, see [`docs/PAPER_REPRODUCTION_MAP.md`](file:///C:/Research/Beyond_Accuracy_Replication_Package/docs/PAPER_REPRODUCTION_MAP.md).

---

## 11. Reproducibility Limitations

When replicating or extending this study, researchers should consider several technical constraints inherent to LLM inference, hardware divergence, and data access:

1. **Non-Deterministic Inference Across GPU Architectures:**  
   Even when setting `temperature=0.0`, `top_p=1.0`, and fixing decode seed `1`, small numerical differences can arise across different GPU hardware architectures, CUDA compute capabilities, and Ollama/llama.cpp backend compilation flags (e.g., AVX2 vs. AVX-512 vs. cuBLAS matrix multiplication order). This floating-point non-associativity can occasionally flip marginal predictions near decision boundaries.
2. **Exact Model Digest & Quantization Reproducibility:**  
   Ollama periodically updates model tags when new upstream GGUF revisions are released. Replicating the exact empirical results requires pulling or verifying the exact immutable SHA-256 model digests specified in [Section 6](#6-required-model-tags--immutable-manifests) and [`config/EXECUTION_SETTINGS.md`](file:///C:/Research/Beyond_Accuracy_Replication_Package/config/EXECUTION_SETTINGS.md). Running with unverified model digests or different quantization types (e.g., `Q8_0` or `FP16`) will alter accuracy and latency profiles.
3. **Input Text Truncation (`MAX_TEXT_CHARS = 800`):**  
   Prompted LLM inference scripts enforced a hard input truncation of the first 800 characters (`text[:800]`) to ensure consistent memory bounds and execution throughput. Removing or altering this character limit on longer reviews will alter token distributions and classification predictions.
4. **Hardware Constraints (VRAM / RAM):**  
   Evaluating frozen LLMs locally requires a minimum of 8 GB system RAM (CPU execution) or 4 GB dedicated VRAM (GPU offloading). Fine-tuning DistilBERT requires at least 4 GB VRAM or will fall back to CPU execution with significantly longer wall-clock runtimes.
5. **Third-Party Data Redistribution & Split Hydration:**  
   To respect third-party dataset and platform terms (including the [Yelp Dataset Terms of Use](https://www.yelp.com/dataset)), raw review and post text is not redistributed in the public repository; splits are published as dehydrated manifests (`example_id`, `label`, `text_sha256`/`text_sha1`, `token_count`, `char_length`, and temporal metadata) alongside precomputed non-reconstructible token statistics (`data/splits/forensics_and_descriptors_cache.json`). All paper tables, figures, and audits (`python reproduce.py --all`) run out-of-the-box without raw text. Retraining models or running live LLM inference requires hydrating the splits via `python scripts/hydrate_splits.py` from the official source datasets or requesting the full-text splits from the corresponding author for academic verification.
6. **Dependency Version Drift:**  
   Differences across major Hugging Face `transformers` versions (e.g., v4.30 vs. v4.45) or `scikit-learn` versions can alter default optimization parameters (such as AdamW epsilon or default logistic regression solvers). Using the pinned `requirements.txt` environment ensures exact algorithmic equivalence.

---

## 12. Citation

If you use this replication package, dataset splits, or forensic methodology in your research, please cite:

```bibtex
@article{alkhalifa2026beyond,
  title={Beyond Accuracy in Dynamic Sentiment Analysis: Reliability Profiling and Temporal Forensics for Frozen and Adapted Models},
  author={Alkhalifa, Rabab},
  journal={SSRN Electronic Journal},
  year={2026},
  doi={10.2139/ssrn.7560344},
  url={https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7560344}
}
```
