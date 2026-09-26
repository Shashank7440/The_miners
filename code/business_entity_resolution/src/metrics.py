"""
Evaluation metrics for ML Challenge 2026: Business Entity Resolution.
Implements the macro-averaged F_0.5 evaluation metric per official challenge specs.
"""

from typing import Dict, Set, Iterable


def compute_f05_single(true_set: Set[str], pred_set: Set[str]) -> float:
    """Compute F_0.5 score for a single Source 1 entity.
    
    Precision-heavy (beta = 0.5):
        F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
        
    Special cases:
        - True is empty & Pred is empty: 1.0 (correct singleton)
        - True is empty & Pred is non-empty: 0.0 (false merge on singleton)
        - True is non-empty & Pred is empty: 0.0 (missed all matches)
        - No intersection: 0.0
    """
    if not true_set:
        return 1.0 if not pred_set else 0.0

    if not pred_set:
        return 0.0

    tp = len(true_set & pred_set)
    if tp == 0:
        return 0.0

    precision = tp / len(pred_set)
    recall = tp / len(true_set)

    denom = 0.25 * precision + recall
    if denom == 0.0:
        return 0.0

    return (1.25 * precision * recall) / denom


def compute_macro_f05(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
    s1_entities: Iterable[str] = None
) -> Dict[str, float]:
    """Compute macro-averaged F_0.5 across all Source 1 entities."""
    if s1_entities is None:
        s1_entities = ground_truth.keys()

    total_f05 = 0.0
    total_count = 0
    singleton_count = 0
    singleton_correct = 0
    matched_count = 0
    matched_f05_sum = 0.0

    for s1 in s1_entities:
        t_set = ground_truth.get(s1, set())
        p_set = predictions.get(s1, set())
        score = compute_f05_single(t_set, p_set)
        total_f05 += score
        total_count += 1

        if not t_set:
            singleton_count += 1
            if not p_set:
                singleton_correct += 1
        else:
            matched_count += 1
            matched_f05_sum += score

    macro_f05 = total_f05 / max(total_count, 1)
    singleton_acc = singleton_correct / max(singleton_count, 1) if singleton_count else 0.0
    matched_macro_f05 = matched_f05_sum / max(matched_count, 1) if matched_count else 0.0

    return {
        "macro_f05": macro_f05,
        "total_entities": total_count,
        "singleton_count": singleton_count,
        "singleton_accuracy": singleton_acc,
        "matched_count": matched_count,
        "matched_macro_f05": matched_macro_f05
    }
