# Dataset Splits & Provenance

This directory contains the audited dataset splits for the **Beyond Accuracy** study under the joint quota temporal evaluation protocol (`shared_splits_paper_v6_joint_quota`).

---

## 1. Experimental Settings

| Setting | Domain / Task | Classes | Time Spans | Sample Size ($N$ per class) | Total per Evaluation Split ($2N$) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Climate n500** | Stance Classification | `BELIEVER`, `DENIER` | Train: 2007–2009<br>Within: 2010<br>Short: 2013<br>Long: 2018–2019 | 500 | 1,000 |
| **Yelp n500** | Sentiment Analysis | `NEG`, `POS` | Train: 2006–2010<br>Within: 2011–2012<br>Short: 2015<br>Long: 2019–2020 | 500 | 1,000 |
| **Yelp n1000** | Sentiment Analysis | `NEG`, `POS` | Same as Yelp n500 | 1,000 | 2,000 |

---

## 2. Directory Layout

- `splits/climate/`: Per-seed evaluation split CSVs across seeds 42, 43, 44, 45, and 46.
- `splits/yelp/`: Per-seed evaluation split CSVs for both $n=500$ and $n=1000$ cohorts.
- `splits/checksums_sha256.txt`: Authoritative cryptographic SHA-256 hashes for all 65 split files.
- `splits/audit_report.json`: Verification manifest confirming zero data leakage and exact seed balance.
- `splits/sampling_audit.csv`: Descriptive sampling diagnostics across temporal windows.

---

## 3. Auditing Checksums

To verify that the splits have not been modified or corrupted:

```bash
python reproduce.py --audit
```
Or directly:
```bash
python scripts/audit_dataset_splits.py
```
All 65 split files are dynamically verified against `checksums_sha256.txt`.
