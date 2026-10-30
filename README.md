# Aging Gracefully? Reliability Profiling and Temporal Forensics of Frozen and Adapted Text Classifiers

[![SSRN Preprint](https://img.shields.io/badge/SSRN-Preprint%207560344-blue.svg)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7560344)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Replication Status](https://img.shields.io/badge/Replication-100%25%20Verified-success.svg)](reproduce.py)

Official open-source replication package for the research paper:  
**"Aging Gracefully? Reliability Profiling and Temporal Forensics of Frozen and Adapted Text Classifiers"**  
Author: **Rabab Alkhalifa** (*Imam Abdulrahman Bin Faisal University*)  
Preprint available on SSRN: [https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7560344](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7560344)

---

## 1. Overview & Key Contributions

Evaluating text classifiers strictly on in-domain reference accuracy conceals critical operational vulnerabilities. When models encounter temporally displaced language streams, performance degrades non-uniformly across model families, confidence estimates lose calibration, and prompt formulations alter predictive behavior.

This replication package reproduces the multidimensional reliability framework and empirical findings of the study across two distinct domains (Climate stance classification and Yelp sentiment classification), four model architectures, five random sampling seeds (42–46), and three temporal horizons (within-period reference $T_0$, short-distance target $T_1$, and long-distance target $T_2$):

1. **Multidimensional Reliability Profiling:** Evaluates task-adapted supervised models (DistilBERT, TF–IDF Logistic Regression) alongside frozen instruction-tuned local LLMs (`phi3.5:latest`, `qwen2.5:3b-instruct`) across canonical predictive accuracy, temporal robustness, verbalized calibration, prompt stability, selective prediction, and structured-output consistency.
2. **Standardized Temporal Delta Sign Convention:** Formulates temporal robustness as $\Delta_j = \text{MacroF1}(T_j) - \text{MacroF1}(T_0)$ (Target minus Reference), where negative values indicate predictive decline under temporal shift and positive values indicate improvement.
3. **Temporal Forensics Layer:** Characterizes dataset-side temporal movement through six signals (lexical novelty, base-2 Jensen–Shannon divergence, top-100 term concentration change, document-length change, vocabulary-overlap loss, and task-native class-marker asymmetry).
4. **100% Self-Contained Parity:** All 65 sampled dataset splits, 180 evaluation conditions, and 40,000+ prediction outputs are cryptographically anchored with SHA-256 hashes, enabling instant reproduction without model re-training.

---

## 2. Mathematical Framework & Equation Mapping

The replication package strictly implements all equations (1)–(17) and Table 3 from the manuscript:

### Summary of Evaluation Measures (Manuscript Table 3)

| Component | Measure | Mathematical Formulation | Eq. | Interpretation |
| :--- | :--- | :--- | :---: | :--- |
| **Predictive Performance** | Accuracy | $\text{Accuracy} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\hat{y}_i = y_i)$ | (5) | Fraction of correctly classified instances |
| | Macro-F1 | $\text{MacroF1} = \frac{1}{\|\mathcal{Y}\|} \sum_{k \in \mathcal{Y}} F_{1,k}$ | (4) | Balanced unweighted average across class-level F1 |
| **Temporal Robustness** | Temporal Macro-F1 Change ($\Delta_j$) | $\Delta_j = \text{MacroF1}(T_j) - \text{MacroF1}(T_0), \quad j \in \{1, 2\}$ | (6) | Target minus reference: negative = decline, positive = gain |
| **Calibration** | Expected Calibration Error | $\text{ECE} = \sum_{b=1}^{10} \frac{\|B_b\|}{N} \|\text{acc}(B_b) - \text{conf}(B_b)\|$ | (9) | 10 equal-width bins on $[0, 1]$ with right closure on 1.0 |
| **Efficiency** | Mean Latency per Example | $\bar{t} = \frac{1}{S} \sum_{s=1}^S \frac{T_s}{N_s}$ | — | Mean recorded wall-clock inference time in seconds |
| **Prompt Stability** | Prompt Sensitivity ($S_{\text{prompt}}$) | $S_{\text{prompt}} = \max_{p \in \mathcal{P}} \text{MacroF1}_p - \min_{p \in \mathcal{P}} \text{MacroF1}_p$ | (10) | Within-seed Macro-F1 range across prompts $\{\text{P1}, \text{P3}, \text{P5}\}$ |
| **Selective Prediction** | Selective Risk at Coverage $c$ | $R(c) = \frac{1}{\|S_c\|} \sum_{i \in S_c} \mathbb{I}(\hat{y}_i \neq y_i)$ | (11) | Error rate on retained subset with fractional tie-breaking |
| **Explicit Abstention** | Abstain Rate ($A$) | $A = \frac{1}{N} \sum_{i=1}^N a_i$ | — | Fraction of examples with explicit deferral ($a_i = 1$) |
| | Retained Coverage ($C_{\text{ret}}$) | $C_{\text{ret}} = 1 - A$ | — | Fraction of examples retained for scoring |
| | Post-Abstention Error ($R_{\text{ret}}$) | $R_{\text{ret}} = \frac{\sum_{i=1}^N (1 - a_i)\mathbb{I}(\hat{y}_i \neq y_i)}{\sum_{i=1}^N (1 - a_i)}$ | — | Observed error rate strictly among retained predictions |
| **Output Validity** | Invalid-Output Rate | $R_{\text{invalid}} = \frac{1}{N} \sum_{i=1}^N v_i$ | — | Fraction of outputs failing JSON schema formatting ($v_i = 1$) |
| **Temporal Forensics** | Lexical Novelty | $\text{Novelty} = \frac{\|V_{\text{target}} \setminus V_{\text{within}}\|}{\|V_{\text{target}}\|}$ | (12) | Proportion of target vocabulary types absent from reference |
| | Distributional Movement | $\text{JSD}_2(P, Q) = \frac{1}{2} D_{\text{KL},2}(P \parallel M) + \frac{1}{2} D_{\text{KL},2}(Q \parallel M)$ | (13) | Base-2 Jensen–Shannon divergence ($M = \frac{1}{2}(P+Q)$) |
| | Top-Term Concentration Change | $\Delta C_{100} = C_{100}(D_{\text{target}}) - C_{100}(D_{\text{within}})$ | (14) | Target minus reference change in top-100 token mass |
| | Document-Length Change | $\Delta L = \bar{L}_{\text{target}} - \bar{L}_{\text{within}}$ | (15) | Target minus reference change in mean document token length |
| | Vocabulary-Overlap Loss | $1 - \frac{\|V_{\text{within}} \cap V_{\text{target}}\|}{\|V_{\text{within}} \cup V_{\text{target}}\|}$ | (16) | One minus Jaccard vocabulary overlap |
| | Class-Marker Asymmetry | $A_{\text{marker}} = \|\text{churn}_{c1} - \text{churn}_{c2}\|$ | (17) | Absolute difference between task-native pivot marker churns |

*Notes on notation:* $T_0$ is the within-period reference; $T_1$ and $T_2$ are the short- and long-distance shifted targets. Task-native label sets are $\mathcal{Y} = \{\text{POS}, \text{NEG}\}$ for Yelp sentiment and $\mathcal{Y} = \{\text{BELIEVER}, \text{DENIER}\}$ for Climate stance.

---

## 3. Quickstart & Installation

The replication suite requires Python 3.10+ and standard numerical libraries. No GPU or specialized hardware is required.

### Setup via pip
```bash
# Clone the repository
git clone https://github.com/rabab-alkhalifa/Beyond_Accuracy_Replication_Package.git
cd Beyond_Accuracy_Replication_Package

# Install dependencies
pip install -r requirements.txt
```

### Setup via Conda
```bash
conda env create -f environment.yml
conda activate beyond-accuracy
```

---

## 4. One-Click Reproduction CLI

The unified master driver `reproduce.py` reproduces the entire paper or individual analyses:

```bash
# Execute complete replication suite (Audit, Tables 7-12, Figures 4-5)
python reproduce.py --all
```
*Expected runtime: ~15–25 seconds on commodity CPU hardware.*

### Individual Component Commands

| Paper Item | Description | CLI Command | Output Artifacts |
| :--- | :--- | :--- | :--- |
| **Audit** | Verify 65 split hashes & 180 prediction conditions | `python reproduce.py --audit` | Console summary report (`PASS`) |
| **Table 7** | Descriptive language statistics across splits | `python reproduce.py --table7` | `outputs/tables/Table7_descriptors.{tex,csv}` |
| **Table 8** | Canonical within-period performance (RQ1) | `python reproduce.py --table8` | `outputs/tables/RQ1_Table8.{tex,csv}` |
| **Table 9** | Temporal Macro-F1 deltas ($\Delta = \text{Target} - \text{Ref}$) | `python reproduce.py --table9` | `outputs/tables/RQ2_Table9.{tex,csv}` |
| **Table 10** | Calibration (ECE) & selective risk ($R_{80}, R_{100}$) | `python reproduce.py --table10` | `outputs/tables/RQ2_Table10_temporal_reliability.tex` |
| **Table 11** | P5 explicit abstention & Prompt Sensitivity ($S_{\text{prompt}}$) | `python reproduce.py --table11` | `outputs/tables/RQ3_Table11_abstention.tex`<br>`outputs/tables/RQ3_prompt_sensitivity_summary.csv` |
| **Table 12** | Temporal forensics dataset movement evidence (RQ5) | `python reproduce.py --table12` | `outputs/tables/RQ5_Table12_temporal_forensics.{tex,csv}` |
| **Figure 4** | Risk-coverage curves & ranking inversions | `python reproduce.py --figures` | `outputs/figures/RQ2_coverage_rankings.{svg,pdf,png}` |
| **Figure 5** | Movement profiles vs. model response | `python reproduce.py --figures` | `outputs/figures/Figure5_movement_vs_response.{svg,pdf,png}` |

---

## 5. Verified Experimental Results

### Table 7: Descriptive Language Statistics of Sampled Splits
*Values reported as mean $\pm$ sample standard deviation across seeds 42–46 under lowercase word tokenization (`\b\w+\b`).*

| Setting | Split | Mean Tokens/Doc | Median Tokens/Doc | Vocabulary Size | Type-Token Ratio | Top-100 Token Mass |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Climate n500** | train | $19.00 \pm 0.04$ | $20.00 \pm 0.00$ | $18,153.6 \pm 62.5$ | $0.136 \pm 0.001$ | $0.509 \pm 0.001$ |
| | within ($T_0$) | $18.47 \pm 0.11$ | $19.40 \pm 0.55$ | $4,756.2 \pm 48.0$ | $0.257 \pm 0.003$ | $0.504 \pm 0.002$ |
| | short ($T_1$) | $17.39 \pm 0.11$ | $18.00 \pm 0.00$ | $4,235.2 \pm 67.3$ | $0.244 \pm 0.004$ | $0.543 \pm 0.004$ |
| | long ($T_2$) | $26.57 \pm 0.36$ | $22.80 \pm 0.45$ | $6,606.4 \pm 43.7$ | $0.249 \pm 0.003$ | $0.497 \pm 0.002$ |
| **Yelp n500** | train | $147.13 \pm 1.97$ | $114.80 \pm 1.92$ | $27,755.6 \pm 193.8$ | $0.027 \pm 0.000$ | $0.534 \pm 0.001$ |
| | within ($T_0$) | $142.11 \pm 3.62$ | $108.00 \pm 3.16$ | $9,997.6 \pm 209.9$ | $0.070 \pm 0.001$ | $0.537 \pm 0.001$ |
| | short ($T_1$) | $117.69 \pm 1.72$ | $83.60 \pm 1.67$ | $8,656.4 \pm 98.0$ | $0.074 \pm 0.001$ | $0.540 \pm 0.001$ |
| | long ($T_2$) | $106.92 \pm 1.30$ | $73.80 \pm 1.52$ | $8,091.4 \pm 64.7$ | $0.076 \pm 0.001$ | $0.542 \pm 0.002$ |
| **Yelp n1000** | train | $147.13 \pm 1.97$ | $114.80 \pm 1.92$ | $27,755.6 \pm 193.8$ | $0.027 \pm 0.000$ | $0.534 \pm 0.001$ |
| | within ($T_0$) | $141.30 \pm 1.45$ | $107.40 \pm 2.19$ | $14,113.0 \pm 102.1$ | $0.050 \pm 0.000$ | $0.537 \pm 0.001$ |
| | short ($T_1$) | $117.55 \pm 1.78$ | $82.80 \pm 1.30$ | $12,202.4 \pm 127.5$ | $0.052 \pm 0.001$ | $0.541 \pm 0.002$ |
| | long ($T_2$) | $106.30 \pm 1.79$ | $74.80 \pm 1.92$ | $11,423.0 \pm 73.8$ | $0.054 \pm 0.001$ | $0.542 \pm 0.001$ |

---

### Table 8: Canonical Within-Period Performance (RQ1)
*Five-seed means on the reference sample ($T_0$). Supervised models use predicted probabilities; frozen LLMs use P3 verbalized confidence.*

| Dataset Setting | Model Family / Model | Accuracy | Macro-F1 | ECE-10 | Mean Latency (s) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Climate n500** | DistilBERT | **0.8878** | **0.8876** | 0.0994 | 0.001422 |
| | TF--IDF Logistic Regression | 0.8018 | 0.8001 | **0.0264** | 0.000036 |
| | `phi3.5:latest` (P3) | 0.7126 | 0.7124 | 0.2233 | 1.886351 |
| | `qwen2.5:3b-instruct` (P3) | 0.6520 | 0.6263 | 0.2841 | 0.627379 |
| **Yelp n500** | DistilBERT | 0.9324 | 0.9324 | 0.0559 | 0.002246 |
| | TF--IDF Logistic Regression | 0.9046 | 0.9045 | 0.1581 | 0.000124 |
| | `phi3.5:latest` (P3) | 0.9452 | 0.9452 | **0.0360** | 10.688317 |
| | `qwen2.5:3b-instruct` (P3) | **0.9602** | **0.9602** | 0.0654 | 3.710427 |
| **Yelp n1000** | DistilBERT | 0.9348 | 0.9348 | 0.0536 | 0.002204 |
| | TF--IDF Logistic Regression | 0.9063 | 0.9063 | 0.1597 | 0.000118 |
| | `phi3.5:latest` (P3) | 0.9448 | 0.9448 | **0.0340** | 10.861398 |
| | `qwen2.5:3b-instruct` (P3) | **0.9601** | **0.9601** | 0.0646 | 3.663848 |

---

### Table 9: Temporal Macro-F1 Robustness and Paired Deltas (RQ2)
*Five-seed means. $\Delta_j = \text{MacroF1}(T_j) - \text{MacroF1}(T_0)$ (Target minus Reference; negative = decline, positive = improvement).*

| Dataset Setting | Model | Reference F1 ($T_0$) | Short F1 ($T_1$) | Long F1 ($T_2$) | $\Delta$ Short | $\Delta$ Long |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Climate n500** | DistilBERT | **0.8876** | **0.8618** | **0.8706** | $-0.0258$ | $-0.0170$ |
| | TF--IDF Logistic Regression | 0.8001 | 0.7120 | 0.7719 | $-0.0881$ | $-0.0282$ |
| | `phi3.5:latest` | 0.7124 | 0.7057 | 0.7398 | $-0.0067$ | $+0.0275$ |
| | `qwen2.5:3b-instruct` | 0.6263 | 0.5978 | 0.7282 | $-0.0285$ | $+0.1019$ |
| **Yelp n500** | DistilBERT | 0.9324 | 0.9508 | 0.9606 | $+0.0184$ | $+0.0282$ |
| | TF--IDF Logistic Regression | 0.9045 | 0.9130 | 0.9318 | $+0.0084$ | $+0.0272$ |
| | `phi3.5:latest` | 0.9452 | 0.9650 | 0.9706 | $+0.0198$ | $+0.0254$ |
| | `qwen2.5:3b-instruct` | **0.9602** | **0.9736** | **0.9756** | $+0.0134$ | $+0.0154$ |
| **Yelp n1000** | DistilBERT | 0.9348 | 0.9538 | 0.9623 | $+0.0190$ | $+0.0275$ |
| | TF--IDF Logistic Regression | 0.9063 | 0.9184 | 0.9293 | $+0.0121$ | $+0.0230$ |
| | `phi3.5:latest` | 0.9448 | 0.9642 | 0.9725 | $+0.0194$ | $+0.0277$ |
| | `qwen2.5:3b-instruct` | **0.9601** | **0.9740** | **0.9773** | $+0.0139$ | $+0.0172$ |

---

### Table 10: Calibration and Confidence-Based Selective Risk Across Periods
*Five-seed means. $R_{80}$ and $R_{100}$ are retained error rates at 80% and 100% coverage, respectively ($R_{100} = 1 - \text{Accuracy}$). Boldface marks the lowest error within setting and period.*

| Dataset Setting | Model | ECE (Ref / Short / Long) | $R_{80}$ (Ref / Short / Long) | $R_{100}$ (Ref / Short / Long) |
| :--- | :--- | :---: | :---: | :---: |
| **Climate n500** | DistilBERT | 0.0994 / 0.1227 / 0.1142 | **0.0488** / **0.0800** / **0.0685** | **0.1122** / **0.1376** / **0.1292** |
| | TF--IDF Logistic Regression | **0.0264** / **0.0563** / **0.0267** | 0.1377 / 0.2240 / 0.1698 | 0.1982 / 0.2746 / 0.2260 |
| | `phi3.5:latest` | 0.2233 / 0.2309 / 0.1949 | 0.2672 / 0.2721 / 0.2288 | 0.2874 / 0.2942 / 0.2588 |
| | `qwen2.5:3b-instruct` | 0.2841 / 0.3085 / 0.1989 | 0.3771 / 0.3954 / 0.2883 | 0.3480 / 0.3720 / 0.2646 |
| **Yelp n500** | DistilBERT | 0.0559 / **0.0396** / **0.0316** | 0.0198 / 0.0132 / 0.0105 | 0.0676 / 0.0492 / 0.0394 |
| | TF--IDF Logistic Regression | 0.1581 / 0.1549 / 0.1662 | 0.0452 / 0.0302 / 0.0195 | 0.0954 / 0.0870 / 0.0682 |
| | `phi3.5:latest` | **0.0360** / 0.0404 / 0.0397 | **0.0182** / **0.0095** / **0.0077** | 0.0548 / 0.0350 / 0.0294 |
| | `qwen2.5:3b-instruct` | 0.0654 / 0.0755 / 0.0741 | 0.0252 / 0.0139 / 0.0153 | **0.0398** / **0.0264** / **0.0244** |
| **Yelp n1000** | DistilBERT | 0.0536 / **0.0372** / **0.0300** | **0.0177** / 0.0112 / 0.0083 | 0.0652 / 0.0462 / 0.0377 |
| | TF--IDF Logistic Regression | 0.1597 / 0.1586 / 0.1647 | 0.0403 / 0.0275 / 0.0204 | 0.0937 / 0.0816 / 0.0707 |
| | `phi3.5:latest` | **0.0340** / 0.0404 / 0.0385 | 0.0181 / **0.0100** / **0.0072** | 0.0552 / 0.0358 / 0.0275 |
| | `qwen2.5:3b-instruct` | 0.0646 / 0.0753 / 0.0765 | 0.0249 / 0.0134 / 0.0135 | **0.0399** / **0.0260** / **0.0227** |

---

### Table 11: Explicit P5 Abstention & Error Reduction (RQ3)
*Frequency and observed error reduction from explicit P5 abstention on reference samples (means across five seeds).*

| Setting | Model | Abstained / $N$ | Incorrect Among Abstained | Error Before (%) | Error After (%) | Reduction (pp) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Climate n500** | Phi | 0 / 5,000 | --- | 29.760% | 29.760% | 0.000 pp |
| | Qwen | 4 / 5,000 | 4 / 4 | 38.920% | 38.871% | 0.049 pp |
| **Yelp n500** | Phi | 0 / 5,000 | --- | 6.520% | 6.520% | 0.000 pp |
| | Qwen | 4 / 5,000 | 4 / 4 | 5.360% | 5.284% | 0.076 pp |
| **Yelp n1000** | Phi | 0 / 10,000 | --- | 6.420% | 6.420% | 0.000 pp |
| | Qwen | 3 / 10,000 | 3 / 3 | 5.010% | 4.981% | 0.029 pp |

#### Prompt Sensitivity $S_{\text{prompt}}$ (Equation 10)
*Within-seed Macro-F1 range across prompt templates $\{\text{P1\_minimal}, \text{P3\_json\_strict}, \text{P5\_abstain}\}$:*
- **Climate n500:** Phi: $0.0206 \pm 0.0048$ | Qwen: $0.0631 \pm 0.0117$
- **Yelp n500:** Phi: $0.0105 \pm 0.0039$ | Qwen: $0.0146 \pm 0.0066$
- **Yelp n1000:** Phi: $0.0091 \pm 0.0039$ | Qwen: $0.0107 \pm 0.0026$

---

### Table 12: Temporal Forensics Movement Evidence (RQ5)
*Values quantify dataset-side movement from the within-period reference to each target period (mean $\pm$ sample SD across seeds 42–46).*

| Dataset Setting | Target | Novelty (Eq. 12) | JSD$_2$ (Eq. 13) | $\Delta C_{100}$ (Eq. 14) | $\Delta L$ (Eq. 15) | Overlap Loss (Eq. 16) | Marker Asym (Eq. 17) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Climate n500** | short | $0.6324 \pm 0.0081$ | $0.2894 \pm 0.0047$ | $+0.0398 \pm 0.0057$ | $-1.0878 \pm 0.1465$ | $0.7907 \pm 0.0034$ | $0.0308 \pm 0.0238$ |
| | long | $0.7177 \pm 0.0073$ | $0.3323 \pm 0.0033$ | $-0.0063 \pm 0.0010$ | $+8.0938 \pm 0.3244$ | $0.8036 \pm 0.0051$ | $0.0125 \pm 0.0102$ |
| **Yelp n500** | short | $0.3803 \pm 0.0068$ | $0.0590 \pm 0.0010$ | $+0.0031 \pm 0.0023$ | $-24.4268 \pm 3.9735$ | $0.5964 \pm 0.0023$ | $0.0370 \pm 0.0141$ |
| | long | $0.3709 \pm 0.0071$ | $0.0676 \pm 0.0005$ | $+0.0056 \pm 0.0031$ | $-35.1940 \pm 3.0609$ | $0.6084 \pm 0.0025$ | $0.0357 \pm 0.0124$ |
| **Yelp n1000** | short | $0.3588 \pm 0.0057$ | $0.0421 \pm 0.0005$ | $+0.0034 \pm 0.0018$ | $-23.7559 \pm 2.6326$ | $0.5769 \pm 0.0020$ | $0.0756 \pm 0.0281$ |
| | long | $0.3494 \pm 0.0090$ | $0.0504 \pm 0.0008$ | $+0.0044 \pm 0.0017$ | $-35.0015 \pm 1.8090$ | $0.5895 \pm 0.0049$ | $0.0703 \pm 0.0236$ |

---

## 6. Repository Architecture

```text
Beyond_Accuracy_Replication_Package/
│
├── README.md                      # Academic reproduction guide & equation documentation
├── reproduce.py                   # Master replication CLI driver
├── requirements.txt               # Pinned Python dependencies
├── environment.yml                # Conda environment definition
│
├── data/                          # Audited experimental data splits
│   ├── README.md                  # Sampling methodology & split specifications
│   └── splits/                    # 65 authoritative CSV splits + checksums_sha256.txt
│       ├── climate/               # Climate stance splits (seeds 42..46: train, within, short, long)
│       └── yelp/                  # Yelp sentiment splits (seeds 42..46: train, within, short, long for n500/n1000)
│
├── archives/                      # Cryptographically verified prediction cohorts
│   ├── completed_prediction_audit.zip           # Climate predictions (adapted + frozen P1/P3/P5)
│   ├── original_yelp_frozen_predictions.zip     # Yelp frozen predictions (P1/P3/P5 across n500/n1000)
│   ├── yelp_prediction_alignment.zip            # Yelp aligned supervised predictions (DistilBERT + TF-IDF)
│   ├── code_and_remaining_results.zip           # Latency metrics & supervised cached summaries
│   └── yelp_results_audit.zip                   # Audit manifests & cross-model mapping logs
│
├── src/                           # Pure modular scientific library
│   ├── __init__.py
│   ├── metrics.py                 # Equations 1-11: Accuracy, Macro-F1, Deltas, ECE-10, S_prompt, Selective Risk
│   ├── descriptors.py             # Table 7: Language statistics (tokens/doc, vocab, TTR, top-100 mass)
│   ├── forensics.py               # Equations 12-17: Novelty, base-2 JSD, Delta C_100, Delta L, Overlap loss, Marker asym
│   ├── loader.py                  # Seamless archive reader & prediction cohort extractor
│   ├── audit.py                   # SHA-256 verification and cross-model ID matching
│   ├── tables.py                  # Formats LaTeX & CSV for Tables 7, 8, 9, 10, 11, 12
│   └── plotting.py                # High-res vector/bitmap plotting for Figures 4 and 5
│
├── scripts/                       # Modular standalone execution scripts
│   ├── audit_dataset_splits.py    # Audits 65 split files and 180 evaluation conditions
│   ├── run_table7.py              # Reproduces Table 7 (Language Descriptors)
│   ├── run_table8.py              # Reproduces Table 8 (Canonical Performance)
│   ├── run_table9.py              # Reproduces Table 9 (Temporal Macro-F1 Robustness)
│   ├── run_table10.py             # Reproduces Table 10 (Reliability & Selective Risk)
│   ├── run_table11.py             # Reproduces Table 11 (P5 Abstention & Prompt Sensitivity)
│   ├── run_table12.py             # Reproduces Table 12 (Temporal Forensics Movement)
│   └── plot_figures.py            # Generates Figure 4 and Figure 5
│
└── outputs/                       # Regenerated publication tables and figures
    ├── tables/                    # LaTeX (.tex) and summary (.csv) files for all tables
    └── figures/                   # Vector (.pdf, .svg) and bitmap (.png) figures
```

---

## 7. Citation

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

---

## 8. License

The code in this repository is licensed under the [MIT License](LICENSE).  
The datasets, annotations, and documentation are licensed under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
