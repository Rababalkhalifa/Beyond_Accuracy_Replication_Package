# Missing Historical Climate Files Notice

During the repository provenance audit, the following historical execution files referenced in external audit notes were **not found** on the local filesystem:

1. `sample_and_run_mvbe_climate_revised(6).py`
   * **Role:** Recovered source of the Climate stance prompts used in the final manuscript appendix.
   * **Preservation Status:** The exact prompts, instructions, class definitions (`BELIEVER`/`DENIER`), and response schemas from this file are fully preserved in:
     - `prompts/climate_prompts.txt`
     - `prompts/prompts.json`
     - `docs/PROMPTS.md`
     - `code/reproducibility/run_paper_models.py`
     - Appendix B of the manuscript.

2. `sample_and_run_mvbe_climate_revised(4).py`
   * **Role:** Earlier audited Climate execution script.
   * **Preservation Status:** Retained in audit manifests; predictions and seed outputs generated under this protocol are preserved in `archives/completed_prediction_audit.zip`.

If the author possesses exact copies of these historical files from an external archive, copy them into this folder under their exact historical names. Do not overwrite existing recovery scripts.
