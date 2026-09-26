"""
Validation and error analysis module.
"""

import sys
import subprocess
import argparse
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import BASE_DIR, TEST_DIR, MATCHING_OUTPUT, CANDIDATE_OUTPUT
from metrics import compute_f05_single, compute_macro_f05


def find_validator_script() -> Path:
    candidates = [
        BASE_DIR / "student_resource" / "utils" / "validate_submission.py",
        BASE_DIR / "utils" / "validate_submission.py",
        Path("utils/validate_submission.py").resolve(),
        Path("student_resource/utils/validate_submission.py").resolve()
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def run_official_validator(matching_file: Path, candidate_file: Path, test_dir: Path, check_ids: bool = False) -> int:
    validator = find_validator_script()
    if not validator:
        print("ERROR: Official validator script utils/validate_submission.py not found!", flush=True)
        return 1

    print("\n" + "=" * 60, flush=True)
    print(f"Running Official Submission Validator: {validator.name}", flush=True)
    print("=" * 60, flush=True)

    cmd = [
        sys.executable,
        str(validator),
        "--matching", str(matching_file),
        "--test-dir", str(test_dir)
    ]
    if candidate_file and candidate_file.exists():
        cmd.extend(["--candidate", str(candidate_file)])
    if check_ids:
        cmd.append("--check-ids")

    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    print(res.stdout, flush=True)
    if res.stderr:
        print("STDERR:\n", res.stderr, flush=True)

    return res.returncode
