# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Byte Builders
**Team Members:** Pantham Prasanth Venkat Sai, Sai Manikanta Kukkala, Manikanta Chowdavaram, Adapa Sankar Sandeep
**Submission Date:** [TODO — fill on final submission]

---

## 1. Executive Summary

We solve the challenge with a **blocking + supervised classifier** pipeline. Text is normalized (legal-suffix stripping, address-abbreviation standardization), candidates are generated with TF-IDF character n-gram nearest-neighbour search inside a dynamically matched same-country pool, and each candidate pair is scored by a logistic-regression model over 16 string-similarity features (RapidFuzz ratios, Jaccard, TF-IDF cosine, country and numeric-address signals). The decision threshold is tuned directly on macro-averaged **F_0.5** using a held-out split of training S1 entities. The design respects the open country label set (no hardcoded US/India), uses only MIT/Apache-2.0 licensed models far below the 8B parameter cap, and performs no external lookups.

---

## 2. Methodology

### 2.1 Problem Analysis

- **No shared identifiers.** Records link only through noisy `business_name` / `business_address` strings plus a `country` label, so the task reduces to pairwise record linkage at scale.
- **Name noise:** abbreviations (`Corp` vs `Corporation`, `Pvt` vs `Private`, `Ltd` vs `Limited`), legal-suffix inconsistency, DBA/trade names, punctuation (`&` vs `and`), word-order transpositions, typos, transliteration variants.
- **Address noise:** abbreviations (`Rd`/`Road`, `St`/`Street`), missing components (no PIN, no state), landmark-based references (`Near SBI ATM`), municipal numbering formats, component reordering.
- **Open country set.** Training covers US and India; the test set adds France. The pipeline therefore treats `country` as an arbitrary string label and never hardcodes a value list.
- **Metric is precision-heavy.** Macro F_0.5 (β=0.5) penalizes false merges more than missed matches, and singletons score 1.0 only when predicted exactly empty — so the operating point must be conservative and empty predictions must be first-class outputs.

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier (supervised pairwise matching)
**Core Innovation:** A lightweight, fully local pipeline where blocking recall is guarded by a same-country TF-IDF character n-gram index, and a calibrated logistic-regression model over complementary similarity features (fuzzy ratios, token/char Jaccard, TF-IDF cosine, address-number recall) is thresholded by direct F_0.5 optimization on a held-out entity split — trading recall for the precision the metric rewards.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** TF-IDF over character n-grams (`char_wb`, 3–5 grams) of normalized `name + " " + address`; cosine nearest neighbours (`sklearn.NearestNeighbors`), **top-K = 80** per S1 entity, queried against the S2+S3 pool filtered to the S1 record's country label (matched dynamically — no hardcoded country list). Records with an empty country label, or a same-country pool smaller than `MIN_POOL_SIZE`, fall back to the global S2+S3 pool so no entity is starved of candidates.
- **Candidate pairs generated:** [TODO — fill with real-data run count]
- **How you ensured true matches were not lost:** (a) character n-grams are robust to typos, transliterations and word-order changes that defeat token-exact blocking; (b) K=80 is generous relative to the small true match multiplicities (each S1 matches zero, one, or a few records); (c) the country filter is label-dynamic with a global-pool fallback, so the open-set test country (France) and missing labels are handled; (d) blocking recall is measured directly against the training ground truth before any model tuning — on the synthetic smoke test it retrieved 453/453 ground-truth pairs (recall 1.0). If real-data blocking recall drops, the first levers are raising K and adding a name-only fallback index.

---

## 4. Matching Model

**Features used (16 per candidate pair):**
- Name features: RapidFuzz `ratio`, `token_set_ratio`, `token_sort_ratio`, `partial_ratio` on normalized names; token Jaccard; character-bigram Jaccard.
- Address features: RapidFuzz `ratio`, `token_set_ratio`, `token_sort_ratio`, `partial_ratio` on normalized addresses; token Jaccard; fraction of the S1 record's address numbers present in the candidate address (numeric recall).
- Other: TF-IDF cosine of the combined normalized `name + address` text; country exact-match flag (1/0); normalized name-length difference; normalized address-length difference.

**Model type:** `sklearn.linear_model.LogisticRegression` with `class_weight="balanced"` (scikit-learn is BSD-licensed, far below the 8B-parameter cap). Training pairs: positives = ground-truth pairs recovered by blocking; negatives = all other blocked candidates. Positives missed by blocking cannot be learned — hence the blocking-recall guard in §3.

**Threshold selection method:** Direct **F_0.5 optimization on a held-out validation split** (25% of training S1 entities, split at the entity level so no S1 leaks across folds). The per-pair probability threshold is swept over [0.05, 1.0]; macro F_0.5 is computed per S1 entity (singletons included) and averaged; the chosen threshold is the **midpoint of the maximum-F_0.5 plateau** to avoid overfitting a single argmax. The model is then retrained on all training pairs and the threshold applied to the test candidates. Per entity, all candidates above threshold are kept (probability-sorted); empty lists are allowed and encouraged for singletons.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** [TODO — real-data validation score; synthetic smoke test: 1.0000, threshold 0.56, vs fuzzy-rule baseline 0.9539 on the same split. The synthetic generator produces fairly separable entities, so this reflects an easy benchmark, not expected real-data performance.]
- **Common false positives (wrong merges):** [TODO — real-data analysis] Expected modes: name twins (same normalized name, different address) with coincidentally similar addresses; heavy abbreviation collisions after suffix stripping.
- **Common false negatives (missed matches):** [TODO — real-data analysis] Expected modes: pairs lost at the blocking stage (true match outside top-80 or cross-country label mismatch); extreme transliteration/abbreviation variants scoring below the conservative threshold.

---

## 6. Conclusion

A blocking-plus-classifier pipeline with TF-IDF character n-gram blocking, 16 complementary similarity features, and F_0.5-tuned logistic regression gives a fast (~seconds), fully local, license-compliant solution that treats country as an open label set and rewards the precision the metric demands. On synthetic data it achieves a clean validation F_0.5 above a fuzzy-rule baseline; the remaining work is tuning on the real training data and confirming blocking recall there.

---

## Appendix

### A. Code Artefacts

```
code/business_entity_resolution/
└── src/
    ├── solution.py         # entry point: normalize -> block -> features ->
    │                       #   train/tune -> predict -> output/*.tsv
    ├── make_synthetic.py  # seedable synthetic data generator (dev only)
    ├── README.md          # exact run instructions
    └── requirements.txt   # pinned: pandas, numpy, scikit-learn, rapidfuzz
```

Reproduce end-to-end against the challenge layout:

```bash
pip install -r requirements.txt
python3 src/solution.py --data-dir dataset --output-dir output
# writes output/matching_results.tsv and output/candidate_pairs.tsv
```

`solution.py` prints the validation macro F_0.5, the chosen threshold, output format-check results, and runtime. `candidate_pairs.tsv` is exactly the candidate set the model scores (the last stage before the model), so every matched ID is a subset of it by construction.

### B. Additional Results

[TODO — real-data validation curves, threshold sweep, blocking-recall numbers, and representative error examples once the training set is available.]
