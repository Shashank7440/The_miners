"""
Step 4: Create Pairwise Matching Features
CLI testing wrapper for pairwise feature engineering.
"""

import sys
from pathlib import Path

# Adjust path for internal imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from normalize import normalize_record
from features import extract_pairwise_features, FEATURE_NAMES


if __name__ == "__main__":
    rec1 = normalize_record({"entity_id": "S1-100", "business_name": "ABC Technologies Inc", "business_address": "100 Main St, New York, NY 10001", "country": "US"})
    rec2 = normalize_record({"entity_id": "S2-200", "business_name": "ABC Technologies", "business_address": "100 Main Street, NY", "country": "US"})
    rec3 = normalize_record({"entity_id": "S3-300", "business_name": "XYZ Corp", "business_address": "500 Oak St, Dallas, TX 75001", "country": "US"})

    print("--- Testing Feature Extraction (Step 4) ---")
    pos_feats = extract_pairwise_features(rec1, rec2, blocking_rank=1)
    neg_feats = extract_pairwise_features(rec1, rec3, blocking_rank=10)

    print("Positive Pair Features (S1-100, S2-200):")
    for k in ["name_token_set_ratio", "name_no_suffix_exact", "addr_token_set_ratio", "shared_building_number", "rule_strong_positive"]:
        print(f"  {k}: {pos_feats[k]}")

    print("\nNegative Pair Features (S1-100, S3-300):")
    for k in ["name_token_set_ratio", "name_no_suffix_exact", "addr_token_set_ratio", "shared_building_number", "rule_strong_conflict"]:
        print(f"  {k}: {neg_feats[k]}")
