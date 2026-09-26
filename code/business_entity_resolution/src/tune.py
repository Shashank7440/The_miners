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

    # Grid search: thresholds from 0.30 to 0.95 with per-source selection strategies
    thresholds = np.arange(0.30, 0.96, 0.02).tolist()
    
    print("\n" + "-" * 95, flush=True)
    print(f"{'Mode':<18} | {'Tau':<6} | {'Macro F0.5':<12} | {'Singleton Acc':<14} | {'Matched F0.5':<12} | {'Total Preds':<10}", flush=True)
    print("-" * 95, flush=True)

    best_score = -1.0
    best_threshold = DEFAULT_F05_THRESHOLD
    best_metrics = None
    best_strategy = "all_above_tau"
    best_use_rules = False

    # Strategies: 'top_dynamic', 'top1_per_source', 'top2_per_source', 'all_above_tau'
    strategies = ["top_dynamic", "top1_per_source", "top2_per_source", "all_above_tau"]

    for strat in strategies:
        for use_rules in [False, True]:
            rule_tag = "rules=OFF" if not use_rules else "rules=ON"
            mode_name = f"{strat} ({rule_tag})"
            local_best_score = -1.0
            local_best_tau = DEFAULT_F05_THRESHOLD

            for tau in thresholds:
                preds = {}
                total_predicted = 0

                for s1_id in val_s1_list:
                    cand_list = s1_scored.get(s1_id, [])
                    matched_ids = set()

                    # Separate by target source (S2 vs S3)
                    s2_scored = []
                    s3_scored = []

                    for cand_id, p, r_pos, r_neg in cand_list:
                        eff_p = p
                        if use_rules:
                            if r_neg > 0:
                                eff_p *= 0.60
                            if r_pos > 0:
                                eff_p = max(eff_p, 0.95)

                        if eff_p >= tau:
                            if cand_id.startswith("S2-"):
                                s2_scored.append((cand_id, eff_p))
                            else:
                                s3_scored.append((cand_id, eff_p))

                    # Sort descending by probability
                    s2_scored.sort(key=lambda x: x[1], reverse=True)
                    s3_scored.sort(key=lambda x: x[1], reverse=True)

                    if strat == "top_dynamic":
                        if s2_scored:
                            top_p = s2_scored[0][1]
                            for cid, p in s2_scored:
                                if p >= top_p - 0.12:
                                    matched_ids.add(cid)
                        if s3_scored:
                            top_p = s3_scored[0][1]
                            for cid, p in s3_scored:
                                if p >= top_p - 0.12:
                                    matched_ids.add(cid)
                    elif strat == "top1_per_source":
                        if s2_scored:
                            matched_ids.add(s2_scored[0][0])
                        if s3_scored:
                            matched_ids.add(s3_scored[0][0])
                    elif strat == "top2_per_source":
                        for item in s2_scored[:2]:
                            matched_ids.add(item[0])
                        for item in s3_scored[:2]:
                            matched_ids.add(item[0])
                    else:  # 'all_above_tau'
                        for item in s2_scored:
                            matched_ids.add(item[0])
                        for item in s3_scored:
                            matched_ids.add(item[0])

                    preds[s1_id] = matched_ids
                    total_predicted += len(matched_ids)

                metrics = compute_macro_f05(ground_truth, preds, s1_entities=val_s1_list)
                f05 = metrics["macro_f05"]
                s_acc = metrics["singleton_accuracy"]
                m_f05 = metrics["matched_macro_f05"]

                if f05 > local_best_score:
                    local_best_score = f05
                    local_best_tau = tau

                if f05 > best_score:
                    best_score = f05
                    best_threshold = tau
                    best_metrics = metrics
                    best_use_rules = use_rules
                    best_strategy = strat

            print(f"{mode_name:<18} | Best tau={local_best_tau:.2f} | F0.5={local_best_score:.4f}", flush=True)

    print("-" * 95, flush=True)
    print(f"\nOptimal Decision Threshold for Macro F_0.5: {best_threshold:.2f}", flush=True)
    print(f"Optimal Candidate Selection Strategy: {best_strategy}", flush=True)
    print(f"Best Validation Macro F_0.5 Score: {best_score:.4f}", flush=True)
    print(f"Rules active: {best_use_rules}", flush=True)
    print(f"Singleton Accuracy at best threshold: {best_metrics['singleton_accuracy']*100:.2f}%", flush=True)
    print(f"Matched Macro F_0.5 at best threshold: {best_metrics['matched_macro_f05']:.4f}", flush=True)

    config = {
        "best_threshold": best_threshold,
        "best_strategy": best_strategy,
        "best_macro_f05": best_score,
        "singleton_accuracy": best_metrics["singleton_accuracy"],
        "matched_macro_f05": best_metrics["matched_macro_f05"],
        "use_rules": best_use_rules,
        "rule_conflict_discount": 0.60,
        "rule_positive_boost": 0.95
    }

    with open(output_config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Saved threshold configuration to: {output_config_path}", flush=True)
