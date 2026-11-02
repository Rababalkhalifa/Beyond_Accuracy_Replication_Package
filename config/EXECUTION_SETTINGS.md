# Empirical Execution Settings & Hyperparameter Provenance

**Scope:** Verifiable operational parameters, inference controls, hardware environments, and model training configurations extracted directly from:
1. Executed run manifests and checkpoint metadata (`Yelp_Original_Checkpoint_Evidence`, `run_settings.json`, `trainer_state.json`, `training_args.json`);
2. Local Ollama daemon manifests (`http://localhost:11434/api/tags`);
3. Historical and reproducibility source code defaults where run metadata are unrecorded.

---

## 1. Frozen Prompted Local LLM Configurations

### A. Evaluated Model Artifacts & Quantization
Inference was conducted locally using Ollama (`v0.40.13`) under direct API integration. Model weights and quantization levels are extracted directly from the installed Ollama manifests:

```text
qwen2.5:3b-instruct
digest: 357c53fb659c5076de1d65ccb0b397446227b71a42be9d1603d46168015c9e4b
quantization: Q4_K_M

phi3.5:latest
digest: 61819fb370a3c1a9be6694869331e5f85f867a079e9271d66cb223acb81d04ba
quantization: Q4_0
```

| Metric / Attribute | `qwen2.5:3b-instruct` | `phi3.5:latest` |
| :--- | :--- | :--- |
| **Exact Ollama Tag** | `qwen2.5:3b-instruct` | `phi3.5:latest` |
| **Immutable Manifest Digest** | `357c53fb659c5076de1d65ccb0b397446227b71a42be9d1603d46168015c9e4b` | `61819fb370a3c1a9be6694869331e5f85f867a079e9271d66cb223acb81d04ba` |
| **Model Format / Container** | GGUF | GGUF |
| **Model Family** | `qwen2` | `phi3` |
| **Parameter Size** | 3.09B (reported as 3.1B) | 3.82B (reported as 3.8B) |
| **Quantization Level** | **`Q4_K_M`** | **`Q4_0`** |
| **Native Context Length** | 32,768 tokens | 131,072 tokens |
| **Embedding Length** | 2,048 | 3,072 |
| **Ollama Backend Runner** | `ggml` | `ggml` |
| **Installed Artifact Timestamp**| `2026-08-05T23:07:49.329Z` | `2026-08-05T23:07:52.691Z` |

> [!NOTE]
> *Quantization Provenance:* Quantization levels (`Q4_K_M` for Qwen2.5 and `Q4_0` for Phi-3.5) were directly determined by the installed Ollama library artifacts and verified through API inspection.

---

### B. Decoding Parameters & Generation Controls
* **Temperature ($T$):** `0.0` (strictly deterministic greedy argmax decoding).
* **Top-p Nucleus:** `1.0` (unrestricted probability mass).
* **Random Decode Seed:** `1` (fixed integer passed via Ollama options block).
* **Output Token Limit (`num_predict`):**
  * Authoritative unified runner (`run_paper_models.py`): `96` tokens.
  * Earlier historical runner (`sample_and_run_mvbe`): `80` tokens.
* **Input Text Truncation (`MAX_TEXT_CHARS`):** First `800` characters (`text[:800]`).
* **Streaming:** `False` (blocking JSON response).
* **Keep-Alive:** `"30m"` (prevents memory eviction between evaluation seeds).
* **Structured Generation / Format:** Enforced via Ollama API JSON Schema (`format` parameter matching `{label, confidence, abstain}`).
* **Warm-Up Procedure:** Prior to each model evaluation block, a single dummy probe is transmitted (`"Reply with OK."`, `num_predict=4`, `temperature=0.0`) to load weights into VRAM.

---

### C. Concurrency, Latency, and Error Handling
* **Client Concurrency:** Multi-threaded client-side concurrency managed by `ThreadPoolExecutor`.
  * *Actual Run Setting:* `num_workers = 4` (bounded by `min(4, os.cpu_count())`).
  * *Code Default Notice:* Earlier exploratory scripts contained code defaults permitting up to 6 workers, but all final audited runs used `4` workers.
* **Request Batching:** `False`. Requests were queued as individual document HTTP POST requests to `http://localhost:11434/api/chat`.
* **Inference Timing Definition:** Wall-clock duration calculated via `time.perf_counter() - start` around the POST request.
* **Retry Behavior:** Maximum 2 retries per document with exponential sleep backoff (`time.sleep(2**attempt)`). Sleep intervals are excluded from recorded latency; operational error strings are captured in `request_error`.

---

## 2. Task-Adapted Supervised Baseline Configurations

### A. DistilBERT Fine-Tuning
Settings extracted from `trainer_state.json`, `training_args.json`, and `run_settings.json` in `Yelp_Original_Checkpoint_Evidence`:

| Hyperparameter | Documented Value | Verification Source |
| :--- | :--- | :--- |
| **Base Architecture** | `distilbert-base-uncased` | Hugging Face Hub (66M parameters) |
| **Max Sequence Length** | `192` tokens | Tokenizer truncation / padding setting |
| **Training Epochs** | `3` epochs | `num_train_epochs: 3` in `training_args.json` |
| **Optimization Algorithm** | AdamW | `transformers.Trainer` default |
| **Learning Rate** | `2e-5` | Verified in run logs (`1.486e-5` at step 875 under linear decay) |
| **Weight Decay** | `0.01` | AdamW regularizer |
| **Warmup Ratio** | `0.10` (10% total steps) | Linear warmup with linear decay |
| **Per-Device Train Batch Size** | `4` | `train_batch_size: 4` in `trainer_state.json` |
| **Gradient Accumulation Steps** | `2` | `effective_batch_size = 8` |
| **Per-Device Eval Batch Size** | `8` | Evaluated every epoch on validation split |
| **Total Steps per Epoch** | `875` steps | $7{,}000 \text{ rows} / 4 / 2 = 875 \text{ steps}$ |
| **Total Training Steps** | `2,625` steps | $875 \times 3 \text{ epochs}$ |
| **Checkpoint Selection Rule** | Validation **Macro-F1** | `metric_for_best_model: "macro_f1"`, `greater_is_better: True` |
| **Model Selection Mechanism** | `load_best_model_at_end=True` | Evaluated at each epoch (`save_strategy="epoch"`) |
| **Early Stopping** | Patience = 1 | `EarlyStoppingCallback(early_stopping_patience=1)` |
| **Saved Model Limit** | `save_total_limit = 1` | Prevents disk exhaustion |
| **Probability Extraction** | Softmax class probabilities | $\max_k P(y=k \mid x)$ via PyTorch logits |

---

### B. TF–IDF + Logistic Regression
Settings extracted from `run_settings.json` and `run_paper_models.py`:

| Parameter | Documented Value | Notes |
| :--- | :--- | :--- |
| **N-gram Range** | `(1, 2)` | Unigrams and bigrams |
| **Tokenization / Casing** | `lowercase=True` | Standard alphanumeric token regex |
| **Sublinear TF Scaling** | `sublinear_tf=True` | Replaces $\text{tf}$ with $1 + \log(\text{tf})$ |
| **Vocabulary Pruning** | `min_df=2`, `max_features=50000` | Prunes rare hapax legomena |
| **Classifier Solver** | `liblinear` (historical Yelp) / `lbfgs` (unified) | L-BFGS used for multinomial convergence in unified runner |
| **Regularization Grid ($C$)** | `[0.25, 0.5, 1.0, 2.0, 4.0]` | Grid search evaluated on held-out validation Macro-F1 |
| **Penalty** | `l2` | Ridge regularization |
| **Max Iterations** | `1000` (up to `2000` in Yelp run settings) | Ensures gradient convergence |
| **Class Weighting** | `None` | Balanced binary splits require no reweighting |
| **Probability Extraction** | Calibrated posterior via `predict_proba` | Sigmoid / Softmax decision score |

---

## 3. Host System Hardware Specifications

* **Operating System:** Microsoft Windows 11 Home 64-bit (Build 10.0.26300)
* **Processor (CPU):** 11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz (8 physical cores, 16 logical threads)
* **System RAM:** 16.0 GB physical memory
* **Discrete GPU:** NVIDIA GeForce RTX 3050 Ti Laptop GPU (4.0 GB VRAM, Driver 32.0.15.6614)
* **Integrated GPU:** Intel(R) UHD Graphics
* **Python Runtime:** Python 3.10.11 (64-bit)
