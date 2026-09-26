# ML Challenge 2026: Business Entity Resolution System

High-precision, multi-source business entity resolution pipeline leveraging **Multi-Pass Blocking**, **Pairwise Similarity Feature Engineering**, and **LightGBM Classification with Macro $F_{0.5}$ Threshold Tuning**.

---

## Architecture Overview

```
                          [ Raw TSV Feeds: S1, S2, S3 ]
                                       │
                                       ▼
                       [ Step 1: Text Normalization ]
                 (Unicode NFC, Legal Suffix Stripping,
                  Address & Number Tokenization)
                                       │
                                       ▼
                      [ Step 2: Multi-Pass Blocking ]
                 (Partitioned by Country: Exact Name,
                  Legal-less Name, Sorted Tokens,
                  Compressed Name, Postal/Street Anchors)
                                       │
                                       ▼
                     [ Step 3: Feature Engineering ]
                 (32 Signals: Levenshtein, Jaccard,
                  Token Sort/Set, Containment,
                  Building & Postal Match/Mismatch,
                  Strong Positive/Negative Rules)
                                       │
                                       ▼
                    [ Step 4: LightGBM Matching Model ]
                 (Trained on Entity-Grouped Splits,
                  Class-Balanced, Objective: Binary Logloss)
                                       │
                                       ▼
                 [ Step 5: Macro F_0.5 Threshold Calibration ]
                 (Optimal Threshold tau = 0.35,
                  Rule-based Boost & Penalty Integration)
                                       │
                                       ▼
               [ Step 6: Low-Memory Streaming Test Inference ]
                 (Country-partitioned streaming, Peak RAM < 600MB,
                  Exact line order preservation)
                                       │
                                       ▼
                    [ Output Submission Verification ]
                 (100% compliant with submission rules)
```

---

## Repository Structure

```
code/business_entity_resolution/
├── README.md                      # Pipeline guide & reproduction documentation
├── requirements.txt               # Dependencies (LightGBM, RapidFuzz, Pandas, etc.)
├── models/
│   ├── matching_lgbm.pkl          # Trained LightGBM pairwise model
│   ├── threshold_config.json      # Calibrated optimal F_0.5 threshold & rule weights
│   └── validation_cache.pkl       # Validation feature cache
└── src/
    ├── config.py                  # Path configuration & hyperparameters
    ├── normalize.py               # Unicode NFC, legal suffix stripping, address parsing
    ├── blocking.py                # Multi-pass country-partitioned inverted index
    ├── features.py                # 32 Pairwise fuzzy, containment, & rule features
    ├── train.py                   # Model training with grouped entity splitting
    ├── tune.py                    # Macro F_0.5 threshold grid-search
    ├── predict.py                 # Low-memory streaming inference engine
    ├── validate.py                # Submission validation wrapper
    ├── metrics.py                 # Macro F_0.5 evaluation metric
    ├── run_pipeline.py            # Master pipeline runner CLI
    ├── 01_audit_data.py           # Step 1: Data audit script
    ├── 02_normalize.py            # Step 2: Normalization verification
    ├── 03_generate_candidates.py  # Step 3: Candidate generation & recall check
    ├── 04_build_features.py       # Step 4: Pairwise feature matrix generator
    ├── 05_train_model.py          # Step 5: LightGBM training wrapper
    ├── 06_tune_threshold.py       # Step 6: Threshold tuning wrapper
    ├── 07_predict_test.py         # Step 7: Test inference wrapper
    └── 08_validate_results.py     # Step 8: Submission output validator
```

---

## Installation & Setup

1. **Prerequisites:** Python 3.8+ (tested on Python 3.10/3.11/3.12).
2. **Install Dependencies:**
```bash
pip install -r code/business_entity_resolution/requirements.txt
```

---

## Reproducing Final Submissions

### Single Command (End-to-End Test Inference + Validation)
To generate `output/matching_results.tsv` and `output/candidate_pairs.tsv` from the pretrained calibrated model:
```bash
python code/business_entity_resolution/src/run_pipeline.py --predict --validate
```

### Full Retraining Pipeline (Train + Tune + Predict + Validate)
To retrain from scratch on the training dataset and regenerate all outputs:
```bash
python code/business_entity_resolution/src/run_pipeline.py --all
```

### HPC High-Performance Execution Mode
For HPC clusters with high CPU core counts and RAM:
```bash
python code/business_entity_resolution/src/run_pipeline.py --all --hpc --train-samples 50000
```

---

## Key Performance Characteristics
- **Memory Footprint:** Operates strictly within `<600MB RAM` using partitioned country streaming and hybrid caching.
- **Precision vs Recall Calibration:** Optimized for Macro $F_{0.5}$ metric (which prioritizes precision with $\beta=0.5$).
- **Multi-Country Support:** Tailored legal suffix catalogs and address anchoring for **India** (PIN codes, road/colony/nagar terms), **United States** (ZIP codes, corporate suffixes), and **France** (Code Postal, SARL/SAS/SA entities).
