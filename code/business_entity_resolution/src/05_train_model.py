"""
Step 5: Train a Binary Matching Model
CLI wrapper for training the LightGBM matching model.
"""

import sys
import argparse
from pathlib import Path

# Adjust path for internal imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import train_matching_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LightGBM Matching Model")
    parser.add_argument("--s1-samples", type=int, default=5000, help="Number of S1 samples for training")
    parser.add_argument("--target-samples", type=int, default=40000, help="Number of S2/S3 records to index")
    args = parser.parse_args()

    train_matching_model(
        sample_s1_count=args.s1_samples,
        target_sample_rows=args.target_samples
    )
