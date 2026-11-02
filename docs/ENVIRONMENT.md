# Hardware and Software Environment Specifications

**Scope:** Documented hardware configuration, host operating system, local inference daemon, and Python software dependencies supporting replication and historical execution.

---

## 1. Hardware Environment

The primary evaluation runs and local LLM inferences were executed on the following workstation host:

* **Processor (CPU):** 11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz
  - Physical Cores: 8
  - Logical Processors (Threads): 16
  - Architecture: x86_64 / 64-bit
* **System Memory (RAM):** 16.0 GB physical memory
* **Dedicated GPU:** NVIDIA GeForce RTX 3050 Ti Laptop GPU
  - Dedicated VRAM: 4.0 GB (4,293,918,720 bytes)
  - Driver Version: 32.0.15.6614
* **Integrated GPU:** Intel(R) UHD Graphics (128 MB reserved)
* **Host Operating System:** Microsoft Windows 11 Home 64-bit (Build 10.0.26300)

---

## 2. Local LLM Runtime Daemon (Ollama)

* **Ollama Version:** `0.40.13`
* **Local Server Endpoint:** `http://localhost:11434`
* **Inference Backend:** `ggml` / `llama.cpp` runner embedded in Ollama
* **Installed Model Artifacts:**
  * `qwen2.5:3b-instruct`:
    - SHA-256 Digest: `357c53fb659c5076de1d65ccb0b397446227b71a42be9d1603d46168015c9e4b`
    - Container Format: GGUF
    - Quantization Level: `Q4_K_M`
    - Parameter Count: 3.09B
    - Context Window: 32,768 tokens
  * `phi3.5:latest`:
    - SHA-256 Digest: `61819fb370a3c1a9be6694869331e5f85f867a079e9271d66cb223acb81d04ba`
    - Container Format: GGUF
    - Quantization Level: `Q4_0`
    - Parameter Count: 3.82B
    - Context Window: 131,072 tokens

---

## 3. Python Software Stack

### A. Core Reproduction Stack (Standard Standardized Packages)
Used for the deterministic reproduction suite (`reproduce.py`, Tables 7–12, Figures 4–5):
* Python: `>= 3.10`
* `numpy`: `>= 1.20.0`
* `scipy`: `>= 1.7.0`
* `pandas`: `>= 1.3.0`
* `matplotlib`: `>= 3.5.0`
* `pyyaml`: `>= 6.0`
* `jupyter`: `>= 1.0.0`
* `ipython`: `>= 7.0.0`

### B. Live Model Execution Stack (Adapted Training & Inference)
Required for running end-to-end model training and live Ollama queries (`scripts/run_experiments.py`):
* `requests`: `>= 2.28.0`
* `scikit-learn`: `>= 1.0.0`
* `joblib`: `>= 1.2.0`
* `torch`: `>= 2.0.0`
* `transformers`: `>= 4.30.0`
* `accelerate`: `>= 0.20.0`

> [!NOTE]
> *Version Provenance Notice:* Package versions above represent tested operational bounds. In accordance with provenance integrity rules, earlier unrecorded pip wheel versions from historical runs prior to August 2026 are not fabricated; the exact reproducible requirements are documented in `requirements.txt` and `environment.yml`.
