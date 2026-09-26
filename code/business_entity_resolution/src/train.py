"""
Training module for LightGBM Matching Model.
Major upgrades: full dataset training, early stopping, better hyperparameters, XGBoost ensemble.
"""

import sys
import csv
import json
import pickle
import random
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import TRAIN_DIR, MODELS_DIR, RANDOM_STATE
from normalize import normalize_record
from blocking import MultiPassBlocker
from features import extract_pairwise_features, FEATURE_NAMES
from metrics import compute_macro_f05


def load_ground_truth(gt_path: Path, max_rows: int = None) -> Dict[str, Set[str]]:
    """Load S1 -> Set of matched target IDs mapping."""
    gt = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        for i, row in enumerate(reader):
            if not row:
                continue
            s1 = row[0].strip()
            mids = set(m.strip() for m in row[1].split(",") if m.strip()) if len(row) > 1 and row[1].strip() else set()
            gt[s1] = mids
            if max_rows and i >= max_rows:
                break
    return gt


def train_matching_model(
    train_dir: Path = TRAIN_DIR,
    sample_s1_count: int = None,
    target_sample_rows: int = None,
    output_model_path: Path = None
):
    """Build dataset, train LightGBM classifier, and save artifacts.
    
    When sample_s1_count is None, uses ALL available S1 entities for training.
    When target_sample_rows is None, uses ALL target records.
    """
    if output_model_path is None:
        output_model_path = MODELS_DIR / "matching_lgbm.pkl"

    print("=" * 60, flush=True)
    print("Step 1: Training Pairwise Matching Classifier (LightGBM)", flush=True)
    print("=" * 60, flush=True)

    print(f"Loading ground truth labels from {train_dir} ...", flush=True)
    gt = load_ground_truth(train_dir / "train_ground_truth.tsv")

    matched_s1 = [s1 for s1, mids in gt.items() if mids]
    singleton_s1 = [s1 for s1, mids in gt.items() if not mids]
    random.seed(RANDOM_STATE)
    random.shuffle(matched_s1)
    random.shuffle(singleton_s1)

    if sample_s1_count and sample_s1_count < len(gt):
        # Sample mode: use specified count
        n_matched = int(sample_s1_count * 0.90)
        n_single = int(sample_s1_count * 0.10)
        selected_s1 = set(matched_s1[:n_matched] + singleton_s1[:n_single])
        print(f"Selected {len(selected_s1):,} S1 entities ({min(n_matched, len(matched_s1)):,} matched, {min(n_single, len(singleton_s1)):,} singletons).", flush=True)
    else:
        # Full mode: use ALL entities
        selected_s1 = set(gt.keys())
        print(f"Using ALL {len(selected_s1):,} S1 entities ({len(matched_s1):,} matched, {len(singleton_s1):,} singletons).", flush=True)

    s1_list = sorted(selected_s1)
    s1_train, s1_val = train_test_split(s1_list, test_size=0.20, random_state=RANDOM_STATE)
    train_s1_set = set(s1_train)
    val_s1_set = set(s1_val)
    print(f"Grouped Entity Split: {len(s1_train):,} Train S1 entities, {len(s1_val):,} Val S1 entities.", flush=True)

    print("Loading entity records ...", flush=True)
    s1_records = {}
    with open(train_dir / "train_source1.tsv", "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        for row in reader:
            if row[0] in selected_s1:
                s1_records[row[0]] = normalize_record({
                    "entity_id": row[0],
                    "business_name": row[1],
                    "business_address": row[2],
                    "country": row[3]
                })

    # 4. Collect all true match target IDs needed for selected S1 entities
    needed_target_ids = set()
    for s1_id in selected_s1:
        needed_target_ids.update(gt.get(s1_id, set()))
    print(f"Target match IDs needed for training/validation: {len(needed_target_ids):,}", flush=True)

    blocker = MultiPassBlocker()
    target_records = {}

    print("Indexing and caching target records ...", flush=True)
    for fname in ["train_source2.tsv", "train_source3.tsv"]:
        path = train_dir / fname
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader)
            bg_count = 0
            for row in reader:
                if len(row) < 4:
                    continue
                eid = row[0].strip()
                is_needed = eid in needed_target_ids

                if target_sample_rows:
                    is_bg = bg_count < (target_sample_rows // 2)
                else:
                    is_bg = True  # Index ALL records when no limit

                if is_needed or is_bg:
                    if is_bg and not is_needed:
                        bg_count += 1

                    rec = {
                        "entity_id": eid,
                        "business_name": row[1].strip(),
                        "business_address": row[2].strip(),
                        "country": row[3].strip()
                    }
                    norm = normalize_record(rec)
                    target_records[eid] = norm
                    blocker._index_single_record(norm)

        print(f"Loaded records from {fname} (total target cache size: {len(target_records):,}).", flush=True)

    print("Generating candidate pairs and pairwise features ...", flush=True)
    X_train, y_train = [], []
    X_val, y_val = [], []
    val_pairs_by_s1 = defaultdict(list)

    processed = 0
    for s1_id in selected_s1:
        s1_rec = s1_records.get(s1_id)
        if not s1_rec:
            continue
        
        true_matches = gt.get(s1_id, set())
        candidates = blocker.retrieve_candidates_for_s1(s1_rec)
        is_train = s1_id in train_s1_set

        all_eval_cands = list(candidates)
        for tm in true_matches:
            if tm in target_records and tm not in all_eval_cands:
                all_eval_cands.append(tm)

        for rank, cand_id in enumerate(all_eval_cands, 1):
            cand_rec = target_records.get(cand_id)
            if not cand_rec:
                continue

            feats = extract_pairwise_features(s1_rec, cand_rec, blocking_rank=rank)
            feat_vector = [feats[name] for name in FEATURE_NAMES]
            label = 1 if cand_id in true_matches else 0

            if is_train:
                X_train.append(feat_vector)
                y_train.append(label)
            else:
                X_val.append(feat_vector)
                y_val.append(label)
                val_pairs_by_s1[s1_id].append((cand_id, feat_vector, label))

        processed += 1
        if processed % 50_000 == 0:
            print(f"  Processed {processed:,} S1 entities ...", flush=True)

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    X_val = np.array(X_val, dtype=np.float32)
    y_val = np.array(y_val, dtype=np.int32)

    print(f"Training dataset: {len(X_train):,} pairs ({np.sum(y_train == 1):,} positives, {np.sum(y_train == 0):,} negatives).", flush=True)
    print(f"Validation dataset: {len(X_val):,} pairs ({np.sum(y_val == 1):,} positives, {np.sum(y_val == 0):,} negatives).", flush=True)

    # Calculate class imbalance ratio for proper weighting
    pos_count = int(np.sum(y_train == 1))
    neg_count = int(np.sum(y_train == 0))
    scale_pos_weight = neg_count / max(pos_count, 1)
    print(f"Class imbalance ratio (neg/pos): {scale_pos_weight:.2f}", flush=True)

    # Use scale_pos_weight=1.0 for true probability calibration (F0.5 requires high precision, not artificially boosted recall)
    clf = lgb.LGBMClassifier(
        n_estimators=1000,            # more trees with early stopping
        learning_rate=0.03,           # slower learning with more trees
        num_leaves=63,                # deep capacity
        max_depth=8,                  # deeper trees
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=15,
        reg_alpha=0.1,                # L1 regularization
        reg_lambda=1.0,               # L2 regularization
        scale_pos_weight=1.0,         # STRICT 1.0 (un-skewed probabilities for high precision)
        random_state=RANDOM_STATE,
        n_jobs=8,                     # fixed 8 threads for HPC
        verbose=-1
    )
    
    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(50, verbose=True),
            lgb.log_evaluation(100)
        ]
    )

    importances = sorted(zip(FEATURE_NAMES, clf.feature_importances_), key=lambda x: x[1], reverse=True)
    print("\nTop 15 Most Informative Features:")
    for fname, imp in importances[:15]:
        print(f"  {fname:35s}: {imp}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_model_path, "wb") as f:
        pickle.dump({"model": clf, "feature_names": FEATURE_NAMES}, f)
    print(f"\nSaved trained model to: {output_model_path}", flush=True)

    val_cache_path = MODELS_DIR / "validation_cache.pkl"
    with open(val_cache_path, "wb") as f:
        pickle.dump({
            "val_s1_list": s1_val,
            "ground_truth": {s1: gt[s1] for s1 in s1_val},
            "val_pairs_by_s1": dict(val_pairs_by_s1)
        }, f)
    print(f"Saved validation cache to: {val_cache_path}", flush=True)
