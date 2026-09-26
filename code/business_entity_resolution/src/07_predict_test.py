"""
Step 9: Generate Final Test Predictions
CLI wrapper for end-to-end test set inference.
"""

import sys
import argparse
from pathlib import Path

# Adjust path for internal imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import TEST_DIR, MATCHING_OUTPUT, CANDIDATE_OUTPUT
from predict import run_test_prediction


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Test Predictions")
    parser.add_argument("--test-dir", type=Path, default=TEST_DIR, help="Path to test directory")
    parser.add_argument("--model", type=Path, default=None, help="Trained model path")
    parser.add_argument("--config", type=Path, default=None, help="Threshold config path")
    parser.add_argument("--matching-out", type=Path, default=MATCHING_OUTPUT, help="Output matching results TSV")
    parser.add_argument("--candidate-out", type=Path, default=CANDIDATE_OUTPUT, help="Output candidate pairs TSV")
    parser.add_argument("--max-s1", type=int, default=None, help="Max S1 rows to process (for debugging)")
    parser.add_argument("--max-target", type=int, default=None, help="Max S2/S3 rows to index")
    args = parser.parse_args()

    run_test_prediction(
        test_dir=args.test_dir,
        model_path=args.model,
        threshold_config_path=args.config,
        matching_output=args.matching_out,
        candidate_output=args.candidate_out,
        max_s1=args.max_s1,
        max_target=args.max_target
    )
