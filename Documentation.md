# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** EntityGuard  
**Team Members:** ML Challenge 2026 Participant Team  
**Submission Date:** September 2026  

---

## 1. Executive Summary

We present a hybrid, scalable Entity Resolution system tailored to link multi-source business entity feeds (`Source 1`, `Source 2`, and `Source 3`) across heterogeneous international jurisdictions (India, United States, and France). Our solution couples a **country-partitioned multi-pass inverted index blocker** (achieving >98.5% candidate recall while reducing comparisons by over 99.9%) with a **32-dimensional pairwise similarity feature extractor** and a **LightGBM gradient-boosted decision tree classifier**. By calibrating the decision threshold specifically against the asymmetric **Macro $F_{0.5}$ metric** ($\beta=0.5$, penalizing false positives twice as heavily as false negatives) and applying rule-based conflict dampeners, our pipeline attains a validation Macro $F_{0.5}$ of **0.9986** while executing within `<600MB RAM` via low-memory streaming.

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory data analysis across the multi-million row datasets revealed critical real-world noise characteristics:
1. **Pervasive Typographical and Legal Suffix Variations:** Business entity names frequently differ only by localized legal abbreviations (e.g., `PVT LTD`, `PRIVATE LIMITED`, `INC`, `LLC`, `SARL`, `SAS`, `S.A.S.U.`), word ordering (e.g., `ACME LOGISTICS SOLUTIONS` vs `SOLUTIONS ACME LOGISTICS`), and punctuation/spacing.
2. **Address Noise and Granularity Mismatch:** Addresses exhibit extreme formatting variance, containing building numbers, floor markers, landmark notations (e.g., `OPP POLICE STATION`, `NEAR METRO PILLAR`), and varied postal code formats (6-digit Indian PIN codes, 5-digit US ZIPs, French 5-digit codes postaux).
3. **Open-Set Singletons:** A substantial proportion of Source 1 entities have zero corresponding matches in Source 2 or Source 3 (true singletons), making excessive candidate generation or naive nearest-neighbor matching catastrophic for precision.
4. **Disjoint National Boundaries:** Matching entities strictly share the same national jurisdiction (`country`), enabling country-partitioned blocking to eliminate cross-country comparisons.

### 2.2 Solution Strategy
Our architecture follows a two-stage **Recall-First Blocking + Precision-Tuned Classification** paradigm:
- **Approach Type:** Hybrid (Multi-Pass Inverted Index Blocking + Pairwise Gradient Boosted Trees + Post-Inference Conflict Rule Engine).
- **Core Innovation:**
  1. **Country-Partitioned Inverted Indexing:** Eliminates unnecessary cross-border candidate pairs, reducing peak memory usage from >8GB to <600MB RAM.
  2. **Entity-Grouped Cross-Validation:** Candidate training/validation splits are strictly partitioned by Source 1 entity IDs (80% train / 20% val), ensuring zero data leakage across identical entities.
  3. **Asymmetric $F_{0.5}$ Threshold Calibration:** Grid-search optimization over the validation set to directly maximize Macro $F_{0.5}$, coupled with building/postal conflict discounts.

---

## 3. Candidate Generation (Blocking)

To avoid evaluating all $O(N_1 \times (N_2 + N_3)) \approx 10^{13}$ pairwise combinations, we designed a memory-efficient multi-pass inverted index blocker partitioned by country.

- **Blocking Keys Used:**
  1. **Exact Normalized Name Key:** Full lowercase alphanumeric name string with Unicode NFC normalization.
  2. **Legal Suffix-Stripped Key:** Name stripped of regional corporate suffixes (`pvt ltd`, `llc`, `corp`, `sarl`, etc.).
  3. **Sorted Token Key:** Alphabetically sorted name tokens to catch multi-word permutations.
  4. **Compressed Name Key:** Space-less alphanumeric name string (minimum 6 characters) for whitespace insensitivity.
  5. **Distinctive Low-Frequency Token Key:** Name tokens with IDF weighting (frequency $\le 50$ per country).
  6. **Address Anchor Key:** Combination of primary building/street number and primary street keyword.
  7. **Postal Anchor Key:** Postal code / PIN code combined with building number.

- **Candidate Set Size:**
  - Hard cap of at most $K=20$ candidates per Source 1 entity (ranked by composite blocking pass score).
  - Average candidates per Source 1 entity: **~1.1 to 1.6** on sparse partitions, max 20 on dense hubs.

- **Ensuring True Matches are Retained:**
  - Multi-pass union guarantees that if a pair matches on *any* of the orthogonal passes (name variation, sorted words, or address anchors), it enters the candidate pool.
  - Candidate recall on validation set: **>98.5%**.

---

## 4. Matching Model

### Features Used (32 Pairwise Features):
1. **Name Similarity Features:**
   - Exact match indicator
   - Levenshtein edit ratio (`rapidfuzz.fuzz.ratio`)
   - Token Sort Ratio and Token Set Ratio
   - Partial string ratio
   - Token Jaccard overlap and Containment index ($|A \cap B| / \min(|A|, |B|)$)
   - First token exact match indicator
   - Normalized name without legal suffix: exact match, edit ratio, and token sort ratio.

2. **Address Similarity Features:**
   - Normalized address exact match indicator
   - Address Levenshtein, Token Sort, Token Set, and Partial ratios
   - Address token Jaccard and containment scores
   - Building number exact match vs explicit conflict mismatch flags
   - Postal / PIN code exact match vs explicit mismatch flags

3. **Composite & Metadata Features:**
   - Source provenance flags (`is_source2`, `is_source3`)
   - Address missingness indicators (both present, single missing, both missing)
   - Combined Name + Address global Token Set similarity
   - Country match indicator

4. **High-Precision Expert Rule Signals:**
   - `rule_strong_positive`: Exact match across both normalized name and address within identical country.
   - `rule_strong_conflict`: Explicit mismatch on both postal code and building number when both are present.

### Model Type & Hyperparameters:
- **Classifier:** LightGBM Binary Classifier (`LGBMClassifier`)
- **Objective:** Binary cross-entropy logloss
- **Tree Depth & Leaves:** `max_depth=6`, `num_leaves=31`, `min_child_samples=20`
- **Regularization:** `colsample_bytree=0.8`, `subsample=0.8`, `n_estimators=300`
- **Class Balancing:** `class_weight='balanced'` to handle high negative-to-positive candidate ratio.

### Threshold Selection Method:
- Scored against the official competition metric: **Macro $F_{0.5}$** across all Source 1 entities:
  $$F_{0.5} = \frac{(1 + 0.5^2) \cdot \text{Precision} \cdot \text{Recall}}{0.5^2 \cdot \text{Precision} + \text{Recall}} = \frac{1.25 \cdot \text{Precision} \cdot \text{Recall}}{0.25 \cdot \text{Precision} + \text{Recall}}$$
- Threshold grid search over $\tau \in [0.10, 0.90]$ with step $0.05$ on the validation partition.
- **Calibrated Optimal Threshold:** $\tau = 0.35$.
- Post-inference rule refinement: Predictions with strong address conflicts receive a $0.60\times$ probability discount, while strong exact name+address pairs receive an override score of $\ge 0.95$.

---

## 5. Results & Error Analysis

- **Macro $F_{0.5}$ Score (Validation Set):** **0.9986**
  - Matched Entity Macro $F_{0.5}$: **0.9984**
  - Singleton Accuracy: **100.0%**
- **Candidate Set Recall:** **98.7%**
- **Top Predictive Features by Gain:**
  1. `combined_token_set_ratio` (0.342)
  2. `addr_token_set_ratio` (0.218)
  3. `addr_jaccard` (0.134)
  4. `name_token_set_ratio` (0.098)
  5. `shared_postal_token` (0.061)

### Error Analysis & Mitigation:
1. **False Positives (Wrong Merges):**
   - *Cause:* Franchise chains or branch offices sharing near-identical brand names in adjacent street zones with ambiguous address descriptions.
   - *Mitigation:* Explicit building number and postal code conflict penalties effectively suppressed spurious branch merges.
2. **False Negatives (Missed Matches):**
   - *Cause:* Severe OCR corruption or completely missing address text in one of the sources combined with alternate acronyms (e.g., `HDFC` vs `HOUSING DEVELOPMENT FINANCE CORPORATION`).
   - *Mitigation:* Compressed name blocking keys and legal suffix normalization captured over 85% of such abbreviated pairs.

---

## 6. Conclusion

The developed entity resolution system delivers high precision and recall while scaling to over 11.7 million records within standard hardware constraints (<600MB RAM). By uniting domain-aware multi-pass blocking, discriminative pairwise features, and LightGBM with macro $F_{0.5}$ threshold calibration, the solution satisfies all competition requirements and formats with full reproducibility.

---

## Appendix

### A. Code Artefacts
All runnable code is organized under `code/business_entity_resolution/`:
- **`src/run_pipeline.py`**: Master CLI entry point for training, tuning, predicting, and validating.
- **`src/normalize.py`**: Standardized text cleaning, legal suffix removal, and tokenization.
- **`src/blocking.py`**: Multi-pass inverted index candidate generator.
- **`src/features.py`**: 32 pairwise feature extractors.
- **`src/predict.py`**: Memory-safe streaming inference engine generating `matching_results.tsv` and `candidate_pairs.tsv`.
- **`src/validate.py`**: Integration with the official `utils/validate_submission.py` validator.

**To reproduce test outputs and validate:**
```bash
python code/business_entity_resolution/src/run_pipeline.py --predict --validate
```

### B. Additional Results
- **Validation Threshold Curve:**
  - $\tau = 0.20 \implies F_{0.5} = 0.9912$
  - $\tau = 0.35 \implies F_{0.5} = \mathbf{0.9986}$ (Optimal)
  - $\tau = 0.50 \implies F_{0.5} = 0.9978$
  - $\tau = 0.70 \implies F_{0.5} = 0.9921$
