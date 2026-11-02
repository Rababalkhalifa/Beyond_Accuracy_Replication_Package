# Dataset Provenance, Chronological Horizons, and Acquisition

**Scope:** Description of source corpora, temporal horizons, class construction, decontamination procedures, and data availability / hydration policies.

---

## 1. Corpus Descriptions & Longitudinal Horizons

The study investigates temporal degradation and reliability across two fundamentally distinct language streams:

### A. Climate Stance Corpus
* **Domain:** Longitudinal social media discourse on climate science, policy, and natural phenomena.
* **Class Definition:** Binary stance classification:
  * `BELIEVER`: Accepts, supports, or expresses concern about climate change, climate science, or climate action.
  * `DENIER`: Rejects, doubts, mocks, minimizes, or argues against climate change, climate science, or climate action.
* **Temporal Slices:**
  * **Historical Training Pool:** 2007–2009 ($N_{\text{fit}} = 7{,}000$, $N_{\text{val}} = 3{,}000$).
  * **Within-Period Reference ($T_0$):** 2010 ($N = 1{,}000$; 500 per class).
  * **Short-Distance Target ($T_1$):** 2013 ($N = 1{,}000$; 500 per class; +3-year chronological displacement).
  * **Long-Distance Target ($T_2$):** 2018–2019 ($N = 1{,}000$; 500 per class; +8-to-9-year chronological displacement).

### B. Yelp Sentiment Corpus
* **Domain:** Customer review stream from commercial business profiles in the [Yelp Open Dataset](https://www.yelp.com/dataset).
* **Class Definition:** Binary sentiment classification:
  * `POS`: Highly favorable reviews (original ratings 4 and 5 stars).
  * `NEG`: Highly critical reviews (original ratings 1 and 2 stars).
  * *Exclusion:* 3-star ratings are neutral/ambiguous and were pruned from the binary task scope.
* **Temporal Slices:**
  * **Historical Training Pool:** 2006–2010 ($N_{\text{fit}} = 7{,}000$, $N_{\text{val}} = 3{,}000$).
  * **Within-Period Reference ($T_0$):** 2011–2012 ($N = 1{,}000$ and $N = 2{,}000$; 500 and 1,000 per class).
  * **Short-Distance Target ($T_1$):** 2015 ($N = 1{,}000$ and $N = 2{,}000$; +3-to-4-year displacement).
  * **Long-Distance Target ($T_2$):** 2019–2020 ($N = 1{,}000$ and $N = 2{,}000$; +7-to-9-year displacement).

---

## 2. Leakage Prevention & Exact Text Decontamination

To ensure empirical validity, historical training pools were rigorously decontaminated against all evaluation periods:
1. **Cryptographic Text Hashing:** Every document text was hashed (SHA-256 for Climate, SHA-1 for Yelp) and verified for uniqueness.
2. **Prior Decontamination:** Any historical record matching an evaluation period document digest was purged from the training corpus *before* conducting the 70/30 train/validation split.
3. **Purity Audit:** All 65 sampled CSV splits were audited with zero exact-text overlap between training/validation and reported test cohorts (verified via `scripts/audit_dataset_splits.py` and `data/splits/leakage_audit.csv`).

---

## 3. Data Redistribution Policy & Split Hydration

Following standard academic NLP practice for third-party text corpora (and in compliance with the [Yelp Dataset Terms of Use](https://www.yelp.com/dataset/agreement) and social-media platform redistribution terms):
* **Dehydrated Split Manifests (`data/splits/climate/` and `data/splits/yelp/`):** The 65 split files in the public repository record `example_id`, gold `label`, cryptographic text digests (`text_sha256` / `text_sha1`), `token_count`, `char_length`, and temporal metadata, with raw review and post text omitted.
* **100% Out-of-the-Box Reproduction:** All manuscript tables (Tables 7–12), figures (Figures 4–5), and audits (`python reproduce.py --all`) run directly without raw text using the saved prediction cohorts in `archives/` and the precomputed non-reconstructible token statistics in `data/splits/forensics_and_descriptors_cache.json`.
* **Hydrating Raw Text (Source Reconstruction or Upon Request):**
  - Researchers can reconstruct the `text` column from their own copy of the official source datasets using `python scripts/hydrate_splits.py --source-files ...`, which matches rows by `text_sha256` / `text_sha1` and validates against `data/splits/checksums_fulltext_sha256.txt`.
  - Alternatively, the full-text research splits can be provided upon reasonable request to the corresponding author for non-commercial academic verification (`python scripts/hydrate_splits.py --fulltext-splits-dir ...`).
