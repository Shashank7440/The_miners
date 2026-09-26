"""
Master Pipeline Runner for ML Challenge 2026: Business Entity Resolution.
Orchestrates:
1. Data normalization & indexing
2. Multi-pass candidate generation
3. Pairwise feature building
4. LightGBM classifier training
5. Macro F_0.5 threshold tuning
6. Final test predictions (matching_results.tsv & candidate_pairs.tsv)
7. Official submission validation

Usage:
  # Run full pipeline locally (small sample):
  python code/business_entity_resolution/src/run_pipeline.py --all

  # Run on HPC with FULL training data:
  python code/business_entity_resolution/src/run_pipeline.py --all --hpc

  # Run only inference on test set:
  python code/business_entity_resolution/src/run_pipeline.py --predict --validate
"""

import sys
import argparse
from pathlib import Path

# Adjust path for internal imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import TRAIN_DIR, TEST_DIR, OUTPUT_DIR, MATCHING_OUTPUT, CANDIDATE_OUTPUT, MODELS_DIR

# Import pipeline steps
from train import train_matching_model
from tune import tune_threshold
from predict import run_test_prediction
from validate import run_official_validator


def main():
    parser = argparse.ArgumentParser(description="End-to-End Business Entity Resolution Pipeline")
    parser.add_argument("--all", action="store_true", help="Run full pipeline end-to-end")
    parser.add_argument("--train", action="store_true", help="Train matching model")
    parser.add_argument("--tune", action="store_true", help="Tune decision threshold for F_0.5")
    parser.add_argument("--predict", action="store_true", help="Generate test predictions")
    parser.add_argument("--validate", action="store_true", help="Validate generated outputs")
    parser.add_argument("--train-samples", type=int, default=None, help="Number of S1 samples for training (None = ALL)")
    parser.add_argument("--target-samples", type=int, default=None, help="Number of target records to index for training (None = ALL)")
    parser.add_argument("--test-dir", type=Path, default=TEST_DIR, help="Path to test directory")
    parser.add_argument("--train-dir", type=Path, default=TRAIN_DIR, help="Path to train directory")
    parser.add_argument("--hpc", action="store_true", help="Flag for high-performance HPC execution")
    args = parser.parse_args()

    if not any([args.all, args.train, args.tune, args.predict, args.validate]):
        parser.print_help()
        print("\nPlease specify at least one action, e.g. --all or --predict --validate")
        sys.exit(1)

    print("=" * 70, flush=True)
    print("ML CHALLENGE 2026: BUSINESS ENTITY RESOLUTION PIPELINE")
    print(f"Mode: {'HPC Cluster (FULL DATA)' if args.hpc else 'Standard Workstation'}", flush=True)
    print("=" * 70, flush=True)

    # 1. Train Model
    if args.all or args.train:
        print("\n>>> STEP 1: Training Pairwise Matching Classifier <<<", flush=True)
        if args.hpc:
            # HPC mode: train on ALL data (None = unlimited)
            train_s1 = args.train_samples       # None = use ALL
            train_target = args.target_samples   # None = use ALL
        else:
            # Local mode: use small sample for quick iteration
            train_s1 = args.train_samples if args.train_samples else 5000
            train_target = args.target_samples if args.target_samples else 40000
        
        train_matching_model(
            train_dir=args.train_dir,
            sample_s1_count=train_s1,
            target_sample_rows=train_target
        )

    # 2. Tune Threshold
    if args.all or args.tune:
        print("\n>>> STEP 2: Calibrating F_0.5 Threshold <<<", flush=True)
        tune_threshold()

    # 3. Predict Test Set
    if args.all or args.predict:
        print("\n>>> STEP 3: Generating Final Test Predictions <<<", flush=True)
        run_test_prediction(
            test_dir=args.test_dir,
            matching_output=MATCHING_OUTPUT,
            candidate_output=CANDIDATE_OUTPUT
        )

    # 4. Validate Submission
    if args.all or args.validate:
        print("\n>>> STEP 4: Validating Submission Outputs <<<", flush=True)
        ret = run_official_validator(
            matching_file=MATCHING_OUTPUT,
            candidate_file=CANDIDATE_OUTPUT,
            test_dir=args.test_dir
        )
        if ret == 0:
            print("\nSUCCESS: All files verified and 100% compliant with submission rules!", flush=True)
        else:
            print("\nWARNING: Validator reported issues. Please check output log above.", flush=True)


if __name__ == "__main__":
    main()
