# Prediction Output Schema & Confidence Scoring Protocols

**Scope:** Schema specification of archived and regenerated prediction CSV files across all 180 evaluation conditions, with documentation of confidence estimation mechanisms.

---

## 1. Prediction Record Schema (21 Columns)

Every prediction CSV in `archives/` and output directory contains the following standardized fields:

| Column Name | Type | Description |
| :--- | :--- | :--- |
| `dataset` | `string` | Target dataset identifier (`climate` or `yelp`). |
| `task_type` | `string` | Task category (`climate_stance` or `sentiment`). |
| `seed` | `integer` | Sampling seed controlling data selection and initialization (`42..46`). |
| `period` | `string` | Temporal horizon: `within` ($T_0$), `short` ($T_1$), or `long` ($T_2$). |
| `n_per_label` | `integer` | Sample size per class: `500` or `1000`. |
| `family` | `string` | Model paradigm: `adapted` (supervised) or `frozen` (prompted LLM). |
| `model` | `string` | Evaluated system: `distilbert`, `tfidf_lr`, `qwen2.5:3b-instruct`, `phi3.5:latest`. |
| `prompt_id` | `string` | Prompt template: `p1_minimal`, `p3_json_strict` (canonical), or `p5_abstain` (or `None` for adapted). |
| `example_id` | `string` | Unique document identifier ensuring 1-to-1 matching across all models. |
| `text_sha256` | `string` | SHA-256 cryptographic hash of raw input document string. |
| `text` | `string` | Input text passed to the tokenizer or prompt template. |
| `true_label` | `string` | Ground truth gold label (`BELIEVER`/`DENIER` or `POS`/`NEG`). |
| `predicted_label`| `string` | Assigned prediction class. |
| `confidence` | `float` | Numerical confidence value bounded in $[0.0, 1.0]$. |
| `confidence_source`| `string` | Mechanism generating confidence: `posterior_probability` or `verbalized_self_report`. |
| `abstain` | `boolean` | Flag indicating explicit abstention (`true` or `false`). |
| `is_correct` | `boolean` | Binary indicator: `predicted_label == true_label`. |
| `status` | `string` | Parsing/execution status: `ok`, `invalid_json`, `invalid_label`, `invalid_confidence`, etc. |
| `request_error` | `string` | Exception message or timeout diagnostic (if request failed; else `null`). |
| `latency_sec` | `float` | Wall-clock execution time in seconds for the inference call. |
| `raw_response` | `string` | Unparsed textual response payload emitted by the model. |

---

## 2. Confidence Estimation: Two Disparate Paradigms

A critical methodological foundation of this paper is distinguishing between two fundamentally different confidence estimation mechanisms:

### A. Supervised Class Posterior Probability (`posterior_probability`)
* **Applicable Models:** DistilBERT, TF–IDF Logistic Regression.
* **Mechanism:**
  - For DistilBERT: Softmax transformation of model output logits:
    $$\hat{c} = \max_{k \in \{0, 1\}} \frac{\exp(z_k)}{\sum_{j} \exp(z_j)}$$
  - For TF–IDF LR: Logistic sigmoid decision function:
    $$\hat{c} = \max ( \sigma(w^T x + b), 1 - \sigma(w^T x + b) )$$
* **Characteristics:** Derived from parametric loss minimization during supervised training on $T_0$. Probabilities sum to 1 across the binary support.

### B. Prompted Self-Reported Verbalized Confidence (`verbalized_self_report`)
* **Applicable Models:** `qwen2.5:3b-instruct`, `phi3.5:latest`.
* **Mechanism:** The model emits a self-reported scalar in the JSON payload (`"confidence": 0.90`) in response to direct natural language instructions.
* **Characteristics:** These values are generated through causal language modeling generation rather than output token probabilities. They reflect the model's self-assessment/verbal calibration under the prompt framing.

### C. Evaluation Considerations
* **Expected Calibration Error (ECE-10):** Evaluates whether confidence $\hat{c} \in [0, 1]$ corresponds to empirical accuracy across 10 equal-width bins $[0.0, 0.1), \dots, [0.9, 1.0]$.
* **Selective Risk Curves ($R_c$):** Documents are sorted in descending order of $\hat{c}$, and risk is evaluated at coverage thresholds (e.g. 80% coverage $R_{80}$ and 100% coverage $R_{100}$).
