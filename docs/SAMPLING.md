# Sampling Protocol, Seed Randomization, and Cohort Nesting

**Scope:** Formal specification of the random sampling framework, balanced class quota allocations, five-seed randomization, and nested cohort structures.

---

## 1. Five-Seed Experimental Design

To guard against sampling variance and ensure statistical robustness, all dataset partitions, model initializations, and evaluation samples were replicated across **five independent random seeds**:
$$\mathcal{S} = \{42, 43, 44, 45, 46\}$$

Every seed controls:
1. The pseudorandom selection of historical documents for the training pool ($N = 10{,}000$);
2. The stratified 70/30 split into model fitting ($N_{\text{fit}} = 7{,}000$) and internal validation ($N_{\text{val}} = 3{,}000$);
3. The balanced sampling of evaluation instances across each temporal period.

---

## 2. Sample Sizes and Nested Evaluation Design

### A. Class-Balanced Sampling Quotas
All evaluation samples are strictly class-balanced to preserve a 50/50 prior distribution:
* **$n = 500$ per class (1,000 instances total):** Evaluated across all datasets (Climate stance, Yelp sentiment).
* **$n = 1{,}000$ per class (2,000 instances total):** Evaluated on Yelp sentiment to measure the empirical effect of evaluation sample scale.
* *Note on Scope:* Climate $n = 1{,}000$ was excluded from the experimental scope due to historical corpus size limits in earlier periods.

### B. Paired Nested Subset Structure
To eliminate sampling noise when analyzing the effect of evaluation sample size, the $n=500$ sample is **strictly nested** within the $n=1{,}000$ sample for each seed and period:
$$\mathcal{D}_{\text{eval}}^{(n=500, s, T_j)} \subset \mathcal{D}_{\text{eval}}^{(n=1000, s, T_j)}$$
This guarantees that performance differences between $n=500$ and $n=1{,}000$ reflect sample scale rather than disjoint document variance.

---

## 3. Matched-Example Evaluation Across Models

Under every evaluation condition (defined by dataset, seed $s$, period $T_j$, and size $n$), all four model architectures are evaluated on the **exact same instance cohort**:
$$\forall m \in \{\text{DistilBERT}, \text{TF--IDF}, \text{Qwen2.5}, \text{Phi-3.5}\}, \quad \text{EvalInstances}(m) \equiv \mathcal{D}_{\text{eval}}^{(n, s, T_j)}$$

Predictions are indexed by unique `example_id` and cryptographically anchored by document SHA-256 hashes. This allows strictly paired statistical contrasts (e.g. McNemar's tests, paired bootstrap confidence intervals, and selective risk differences).

---

## 4. Summary of Generated Split Cohorts

The replication suite contains 65 audited CSV split files:
* **Climate Stance:** 5 seeds $\times$ 5 splits (`train`, `validation`, `within`, `short`, `long`) = 25 files.
* **Yelp Sentiment:** 5 seeds $\times$ 8 splits (`train`, `validation`, and `within`, `short`, `long` for both $n=500$ and $n=1{,}000$) = 40 files.
* **Total:** 65 CSV split files, indexed in `data/splits/split_index.csv` and verified against `data/splits/checksums_sha256.txt`.
