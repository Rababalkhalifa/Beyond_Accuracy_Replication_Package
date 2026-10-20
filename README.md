# Beyond Accuracy: Temporal Reliability & Robustness Replication Package

Official, verified reproduction suite for the empirical results reported in the research paper:  
**"Beyond Accuracy: Temporal Reliability and Robustness of NLP and LLM Classifiers"**

This self-contained package replicates all canonical tables, degradation metrics, calibration scores, and selective risk curves from verified prediction cohorts without requiring model re-training.

---

## 1. Quickstart

### Environment Setup
You can use standard Python virtual environments (`venv`) or Conda:

**Option A: Standard pip**
```bash
pip install -r requirements.txt
```

**Option B: Conda**
```bash
conda env create -f environment.yml
conda activate beyond-accuracy
```

### Full Replication (One Command)
Run the master replication script to audit data splits, generate all LaTeX tables, CSV summaries, and high-resolution figures:

```bash
python reproduce.py --all
```
*Expected runtime: ~8–12 seconds on standard commodity hardware (CPU-only, no GPU required).*

---

## 2. Command-to-Paper Mapping

Every table and figure in the manuscript maps directly to a modular CLI command:

| Manuscript Item | Description | Replication Command | Output Artifact |
| :--- | :--- | :--- | :--- |
| **Data Audit** | Validates SHA-256 for all 65 splits & 180 prediction conditions | `python reproduce.py --audit` | Terminal verification report (`PASS`) |
| **RQ1: Table 8** | Canonical within-period performance (Acc, Macro-F1, ECE, Latency) | `python reproduce.py --table8` | `outputs/tables/RQ1_Table8.tex`<br>`outputs/tables/RQ1_Table8_summary.csv` |
| **RQ2: Table 9** | Temporal Macro-F1 across within, short, long periods & paired $\Delta$ | `python reproduce.py --table9` | `outputs/tables/RQ2_Table9.tex`<br>`outputs/tables/RQ2_Table9_summary.csv` |
| **Figure Panel** | Coverage sweep (20–100%) and model ranking inversion | `python reproduce.py --figures` | `outputs/figures/RQ2_coverage_rankings.svg`<br>`outputs/figures/RQ2_coverage_rankings.pdf`<br>`outputs/figures/RQ2_coverage_rankings.png` |
| **Reliability Table**| Period-specific ECE and 80% coverage selective risk | `python reproduce.py --figures` | `outputs/tables/RQ2_temporal_reliability_table.tex` |

---

## 3. Repository Architecture

```text
Beyond_Accuracy_Replication_Package/
│
├── README.md                      # Academic reproduction guide (this document)
├── reproduce.py                   # Master replication entry point
├── requirements.txt               # Pinned Python package dependencies
├── environment.yml                # Conda environment definition
│
├── configs/                       # Configuration definitions
│   └── settings.yaml              # Seeds (42–46), model definitions, cohort parameters
│
├── data/                          # Audited experimental data splits
│   ├── README.md                  # Split methodology & joint quota specification
│   └── splits/                    # 65 verified split CSVs + checksums_sha256.txt
│
├── archives/                      # Cryptographically verified prediction cohorts
│   ├── completed_prediction_audit.zip
│   ├── original_yelp_frozen_predictions.zip
│   ├── yelp_prediction_alignment.zip
│   ├── code_and_remaining_results.zip
│   └── yelp_results_audit.zip
│
├── src/                           # Pure modular scientific library
│   ├── __init__.py
│   ├── metrics.py                 # Macro-F1, 10-bin ECE, fractional tie risk-sweep
│   ├── loader.py                  # Seamless archive reader & prediction loader
│   ├── audit.py                   # Checksum verification and cross-model ID matching
│   ├── tables.py                  # Exact LaTeX & CSV table formatting
│   └── plotting.py                # High-resolution figure rendering (SVG, PDF, PNG)
│
├── scripts/                       # Focused standalone execution scripts
│   ├── audit_dataset_splits.py    # Audits split integrity & prediction alignment
│   ├── run_table8.py              # RQ1 Within-period table generator
│   ├── run_table9.py              # RQ2 Temporal robustness table generator
│   └── plot_figures.py            # Generates figures & reliability tables
│
├── notebooks/                     # Interactive Jupyter analysis
│   └── Beyond_Accuracy_Audit.ipynb # Self-contained audit & interactive figure inspection
│
└── outputs/                       # Generated replication artifacts matching paper
    ├── tables/                    # RQ1_Table8.tex, RQ2_Table9.tex, summaries
    └── figures/                   # SVG, PDF, and PNG figure panels
```

---

## 4. Scientific & Mathematical Fidelity

This replication package preserves exact mathematical definitions:

1. **Macro-F1:**
   $$\text{Macro-F1} = \frac{1}{|C|} \sum_{c \in C} F1_c$$
   Calculated per-seed before taking arithmetic means across the five random seeds ($42, 43, 44, 45, 46$).

2. **Expected Calibration Error (ECE-10):**
   10 equal-width bins spanning $[0.0, 1.0]$ with bin width $0.1$. Bins are left-closed, right-open $[b_{i}, b_{i+1})$, with the final bin closed on both ends $[0.9, 1.0]$. Probabilities for supervised models derive from model softmax; LLMs use self-reported confidence.

3. **Temporal Degradation Paired Deltas:**
   $$\Delta_{\text{short}} = \text{F1}_{\text{within}} - \text{F1}_{\text{short}}, \quad \Delta_{\text{long}} = \text{F1}_{\text{within}} - \text{F1}_{\text{long}}$$
   Computed on a per-seed basis for each model family, strictly pairing identical sample seeds before taking the five-seed mean.

4. **Confidence-Based Selective Risk:**
   Calculated for coverage sweeps from 20% to 100% in 1% increments. When a selection threshold cuts through identical confidence scores, exact fractional tie-handling is applied, representing the expected error under uniform random tie-breaking.

---

## 5. Summary of Empirical Results

### Table 8: Canonical Within-Period Performance (5-Seed Means)

| Dataset Setting | Model | Accuracy | Macro-F1 | ECE | Mean Latency (s) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Climate n500** | DistilBERT | 0.8878 | 0.8876 | 0.0994 | 0.001422 |
| | TF-IDF+LR | 0.8018 | 0.8001 | 0.0264 | 0.000036 |
| | phi3.5:latest | 0.7126 | 0.7124 | 0.2233 | 1.886351 |
| | qwen2.5:3b-instruct | 0.6520 | 0.6263 | 0.2841 | 0.627379 |
| **Yelp n500** | DistilBERT | 0.9324 | 0.9324 | 0.0559 | 0.002246 |
| | TF-IDF+LR | 0.9046 | 0.9045 | 0.1581 | 0.000124 |
| | phi3.5:latest | 0.9452 | 0.9452 | 0.0360 | 10.688317 |
| | qwen2.5:3b-instruct | 0.9602 | 0.9602 | 0.0654 | 3.710427 |
| **Yelp n1000** | DistilBERT | 0.9348 | 0.9348 | 0.0536 | 0.002204 |
| | TF-IDF+LR | 0.9063 | 0.9063 | 0.1597 | 0.000118 |
| | phi3.5:latest | 0.9448 | 0.9448 | 0.0340 | 10.861398 |
| | qwen2.5:3b-instruct | 0.9601 | 0.9601 | 0.0646 | 3.663848 |

### Table 9: Temporal Robustness (5-Seed Means)

| Dataset Setting | Model | Within F1 | Short F1 | Long F1 | $\Delta$ Short | $\Delta$ Long |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Climate n500** | DistilBERT | 0.8876 | 0.8618 | 0.8706 | +0.0258 | +0.0170 |
| | TF-IDF+LR | 0.8001 | 0.7120 | 0.7719 | +0.0881 | +0.0282 |
| | phi3.5:latest | 0.7124 | 0.7057 | 0.7398 | +0.0067 | -0.0275 |
| | qwen2.5:3b-instruct | 0.6263 | 0.5978 | 0.7282 | +0.0285 | -0.1019 |
| **Yelp n500** | DistilBERT | 0.9324 | 0.9508 | 0.9606 | -0.0184 | -0.0282 |
| | TF-IDF+LR | 0.9045 | 0.9130 | 0.9318 | -0.0084 | -0.0272 |
| | phi3.5:latest | 0.9452 | 0.9650 | 0.9706 | -0.0198 | -0.0254 |
| | qwen2.5:3b-instruct | 0.9602 | 0.9736 | 0.9756 | -0.0134 | -0.0154 |
| **Yelp n1000** | DistilBERT | 0.9348 | 0.9538 | 0.9623 | -0.0190 | -0.0275 |
| | TF-IDF+LR | 0.9063 | 0.9184 | 0.9293 | -0.0121 | -0.0230 |
| | phi3.5:latest | 0.9448 | 0.9642 | 0.9725 | -0.0194 | -0.0277 |
| | qwen2.5:3b-instruct | 0.9601 | 0.9740 | 0.9773 | -0.0139 | -0.0172 |

---

## 6. Citation & Contact

If you use or adapt this code or benchmark, please cite the corresponding research paper:

```bibtex
@article{alkhalifa7560344beyond,
  title={Beyond Accuracy in Dynamic Sentiment Analysis: Reliability Profiling and Temporal Forensics for Frozen and Adapted Models},
  author={Alkhalifa, Rabab},
  journal={Available at SSRN 7560344}
}
```
