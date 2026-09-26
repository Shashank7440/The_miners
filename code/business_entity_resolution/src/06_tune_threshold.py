"""
Step 6 & 7: Tune Decision Threshold for Macro F_0.5
CLI wrapper for grid-search threshold tuning.
"""

import sys
import argparse
from pathlib import Path

# Adjust path for internal imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tune import tune_threshold


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune F_0.5 Threshold on Validation Split")
    parser.add_argument("--model", type=Path, default=None, help="Path to trained LightGBM model")
    parser.add_argument("--cache", type=Path, default=None, help="Path to validation cache")
    args = parser.parse_args()

    tune_threshold(
        model_path=args.model,
        val_cache_path=args.cache
    )
