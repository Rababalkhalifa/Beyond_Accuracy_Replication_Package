# Dataset Splits & Provenance

This directory contains the audited dataset split manifests for the **Beyond Accuracy** study under the joint quota temporal evaluation protocol (`shared_splits_paper_v6_joint_quota`).

---

## 1. Experimental Settings

| Setting | Domain / Task | Classes | Time Spans | Sample Size ($N$ per class) | Total per Evaluation Split ($2N$) |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **Climate n500** | Stance Classification | `BELIEVER`, `DENIER` | Train: 2007–2009<br>Within: 2010<br>Short: 2013<br>Long: 2018–2019 | 500 | 1,000 |
| **Yelp n500** | Sentiment Analysis | `NEG`, `POS` | Train: 2006–2010<br>Within: 2011–2012<br>Short: 2015<br>Long: 2019–2020 | 500 | 1,000 |
| **Yelp n1000** | Sentiment Analysis | `NEG`, `POS` | Same as Yelp n500 | 1,000 | 2,000 |

---

## 2. Dehydrated Split Format & Data Availability

Following standard academic NLP practice for third-party text corpora (and in compliance with the [Yelp Dataset Terms of Use](https://www.yelp.com/dataset) and social-media platform terms), the 65 CSV files in `splits/climate/` and `splits/yelp/` are distributed as **dehydrated split manifests**:
- Each row records `example_id`, gold `label`, cryptographic text digest (`text_sha256` for Climate, `text_sha1` for Yelp), `token_count`, `char_length`, and temporal metadata (`source_year` / `date_value`, `source_period` / `source_role`, `sample_seed`), omitting raw third-party text.
- `splits/forensics_and_descriptors_cache.json` stores the precomputed non-reconstructible split-level token statistics required for Table 7 (language descriptors) and Table 12 / Figure 5 (temporal forensics), so **`python reproduce.py --all` runs 100% out-of-the-box without raw text**.
- `splits/checksums_sha256.txt` stores the authoritative SHA-256 checksums for the 65 dehydrated split files, while `splits/checksums_fulltext_sha256.txt` stores the SHA-256 checksums for the hydrated full-text split files (`audit_splits` accepts either state automatically).

### Hydrating Raw Text for Model Retraining or Live Prompting

Researchers who wish to re-run model training or live LLM inference from scratch can populate the `text` column using `scripts/hydrate_splits.py`:
1. **From official source datasets** (matching rows by SHA-256 / SHA-1 text digest):
   ```bash
   python scripts/hydrate_splits.py --source-files /path/to/raw_source.csv
   ```
2. **Upon academic request from the corresponding author** (for non-commercial scientific verification):
   ```bash
   python scripts/hydrate_splits.py --fulltext-splits-dir /path/to/full_text
   ```

---

## 3. Auditing Checksums

To verify that all 65 splits and 180 prediction cohorts match their authoritative hashes:

```bash
python reproduce.py --audit
```
Or directly:
```bash
python scripts/audit_dataset_splits.py
```
