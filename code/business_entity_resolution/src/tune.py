"""
Tuning module for Macro F_0.5 decision threshold.
Fine-grained grid search (0.01 step) with per-rule analysis.
"""

import sys
import json
import pickle
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import MODELS_DIR, DEFAULT_F05_THRESHOLD
from metrics import compute_macro_f05


def tune_threshold(
    model_path: Path = None,
    val_cache_path: Path = None,
    output_config_path: Path = None
):
    """Grid search for optimal macro F_0.5 threshold on validation split."""
    if model_path is None:
        model_path = MODELS_DIR / "matching_lgbm.pkl"
    if val_cache_path is None:
        val_cache_path = MODELS_DIR / "validation_cache.pkl"
    if output_config_path is None:
        output_config_path = MODELS_DIR / "threshold_config.json"

    print("=" * 60, flush=True)
    print("Step 7: Tuning Decision Threshold for Macro F_0.5", flush=True)
    print("=" * 60, flush=True)

    if not model_path.exists():
        print(f"ERROR: Model file {model_path} not found! Run training first.")
        return
    if not val_cache_path.exists():
        print(f"ERROR: Validation cache {val_cache_path} not found! Run training first.")
        return

    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
        clf = model_data["model"]
        feature_names = model_data["feature_names"]
        # Force thread limit for HPC
        if hasattr(clf, 'set_params'):
            clf.set_params(n_jobs=8)

    with open(val_cache_path, "rb") as f:
        val_data = pickle.load(f)
        val_s1_list = val_data["val_s1_list"]
        ground_truth = val_data["ground_truth"]
        val_pairs_by_s1 = val_data["val_pairs_by_s1"]

    print(f"Validation entities: {len(val_s1_list):,} S1 entities ({sum(1 for s in val_s1_list if not ground_truth.get(s)):,} singletons).", flush=True)

    all_pairs = []
    pair_meta = []
    for s1_id in val_s1_list:
        for cand_id, feat_vector, true_label in val_pairs_by_s1.get(s1_id, []):
            all_pairs.append(feat_vector)
            rule_pos_idx = feature_names.index("rule_strong_positive")
            rule_neg_idx = feature_names.index("rule_strong_conflict")
            pair_meta.append((s1_id, cand_id, feat_vector[rule_pos_idx], feat_vector[rule_neg_idx]))

    if not all_pairs:
        print("Warning: No candidate pairs found in validation cache.")
        return

    X_val = np.array(all_pairs, dtype=np.float32)
    print(f"Scoring {len(X_val):,} validation candidate pairs with classifier ...", flush=True)
    probs = clf.predict_proba(X_val)[:, 1]

    s1_scored = defaultdict(list)
    for (s1_id, cand_id, r_pos, r_neg), p in zip(pair_meta, probs):
        s1_scored[s1_id].append((cand_id, p, r_pos, r_neg))

    # 2D Grid Search: Independent tau_s2 and tau_s3 from 0.50 to 0.95 (0.01 step)
    tau_range = np.arange(0.50, 0.96, 0.01).tolist()
    
    print("\n" + "=" * 80, flush=True)
    print("INDEPENDENT THRESHOLD TUNING (tau_s2 vs tau_s3)", flush=True)
    print("=" * 80, flush=True)

    best_score = -1.0
    best_tau_s2 = 0.70
    best_tau_s3 = 0.70
    best_metrics = None

    for tau_s2 in np.arange(0.50, 0.96, 0.03):
        for tau_s3 in np.arange(0.50, 0.96, 0.03):
            preds = {}
            for s1_id in val_s1_list:
                cand_list = s1_scored.get(s1_id, [])
                matched_ids = set()

                for cand_id, p, r_pos, r_neg in cand_list:
                    if cand_id.startswith("S2-"):
                        if p >= tau_s2:
                            matched_ids.add(cand_id)
                    else:
                        if p >= tau_s3:
                            matched_ids.add(cand_id)

                preds[s1_id] = matched_ids

            metrics = compute_macro_f05(ground_truth, preds, s1_entities=val_s1_list)
            f05 = metrics["macro_f05"]

            if f05 > best_score:
                best_score = f05
                best_tau_s2 = float(tau_s2)
                best_tau_s3 = float(tau_s3)
                best_metrics = metrics

    # Refine grid around best tau_s2 and tau_s3
    s2_fine = np.arange(max(0.50, best_tau_s2 - 0.04), min(0.96, best_tau_s2 + 0.05), 0.01).tolist()
    s3_fine = np.arange(max(0.50, best_tau_s3 - 0.04), min(0.96, best_tau_s3 + 0.05), 0.01).tolist()

    for tau_s2 in s2_fine:
        for tau_s3 in s3_fine:
            preds = {}
            for s1_id in val_s1_list:
                cand_list = s1_scored.get(s1_id, [])
                matched_ids = set()

                for cand_id, p, r_pos, r_neg in cand_list:
                    if cand_id.startswith("S2-"):
                        if p >= tau_s2:
                            matched_ids.add(cand_id)
                    else:
                        if p >= tau_s3:
                            matched_ids.add(cand_id)

                preds[s1_id] = matched_ids

            metrics = compute_macro_f05(ground_truth, preds, s1_entities=val_s1_list)
            f05 = metrics["macro_f05"]

            if f05 > best_score:
                best_score = f05
                best_tau_s2 = float(tau_s2)
                best_tau_s3 = float(tau_s3)
                best_metrics = metrics

    print(f"Optimal Source 2 Threshold (tau_s2): {best_tau_s2:.2f}", flush=True)
    print(f"Optimal Source 3 Threshold (tau_s3): {best_tau_s3:.2f}", flush=True)
    print(f"Honest Validation Macro F_0.5 Score: {best_score:.4f}", flush=True)
    print(f"Singleton Accuracy at best threshold: {best_metrics['singleton_accuracy']*100:.2f}%", flush=True)
    print(f"Matched Macro F_0.5 at best threshold: {best_metrics['matched_macro_f05']:.4f}", flush=True)

    config = {
        "best_threshold_s2": best_tau_s2,
        "best_threshold_s3": best_tau_s3,
        "best_threshold": (best_tau_s2 + best_tau_s3) / 2.0,
        "best_strategy": "all_above_tau",
        "best_macro_f05": best_score,
        "singleton_accuracy": best_metrics["singleton_accuracy"],
        "matched_macro_f05": best_metrics["matched_macro_f05"],
        "use_rules": False
    }

    with open(output_config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Saved threshold configuration to: {output_config_path}", flush=True)
