# Missing Historical Yelp Files Notice

During the repository provenance audit, the following historical execution file referenced in project notes was **not found** on the local filesystem:

1. `sample_and_run_mvbe_sentiment(3).py`
   * **Role:** Historical Yelp frozen Qwen/Phi inference code.
   * **Preservation Status:** The Yelp frozen LLM inference logic, prompt templates (`p1_minimal`, `p3_json_strict`, `p5_abstain`), and JSON schema formats are preserved in:
     - `code/historical/yelp/sentiment_frozen_and_adapted_revised.py`
     - `code/historical/yelp/sentiment_frozen_and_adapted_v4_7000_3000_MEMORY_SAFE.py`
     - `prompts/yelp_prompts.txt`
     - `archives/original_yelp_frozen_predictions.zip`

If the author possesses an exact copy of `sample_and_run_mvbe_sentiment(3).py`, copy it into this folder under its exact historical filename.
