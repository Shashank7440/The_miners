"""
Step 10: Validate Submission Files and Perform Error Analysis
CLI wrapper for submission validation and error analysis.
"""

import sys
import argparse
from pathlib import Path

# Adjust path for internal imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import TEST_DIR, MATCHING_OUTPUT, CANDIDATE_OUTPUT
from validate import run_official_validator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Submission Outputs")
    parser.add_argument("--matching", type=Path, default=MATCHING_OUTPUT, help="Path to matching_results.tsv")
    parser.add_argument("--candidate", type=Path, default=CANDIDATE_OUTPUT, help="Path to candidate_pairs.tsv")
    parser.add_argument("--test-dir", type=Path, default=TEST_DIR, help="Path to test dir with test_source1.tsv")
    parser.add_argument("--check-ids", action="store_true", help="Check ID existence in target sources")
    args = parser.parse_args()

    ret = run_official_validator(args.matching, args.candidate, args.test_dir, check_ids=args.check_ids)
    sys.exit(ret)
