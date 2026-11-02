# Missing Historical Adapted Baseline Code Notice

During the repository provenance audit, the following historical baseline runner was **not found** on the local filesystem:

1. `run_adapted_matched_baselines(1).py`
   * **Role:** Historical TF–IDF Logistic Regression and DistilBERT training/evaluation code.
   * **Preservation Status:** The identical training procedures, hyperparameters, cross-validation protocols, and evaluation metrics are preserved in:
     - `code/historical/yelp/sentiment_frozen_and_adapted_v4_7000_3000_MEMORY_SAFE.py` (L1464–1540)
     - `code/reproducibility/run_paper_models.py` (L510–680)
     - `config/run_settings.json`
     - `checkpoints/adapted_distilbert/` (in `Yelp_Original_Checkpoint_Evidence`)

If the author possesses an exact copy of `run_adapted_matched_baselines(1).py`, copy it into this folder under its exact historical filename.
