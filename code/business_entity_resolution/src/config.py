"""
Configuration and shared path definitions for Business Entity Resolution.
Supports both local runs and HPC cluster execution via environment variables or CLI flags.
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

# Detect dataset location (student_resource/dataset or dataset)
if (BASE_DIR / "student_resource" / "dataset").exists():
    DATASET_DIR = BASE_DIR / "student_resource" / "dataset"
elif (BASE_DIR / "dataset").exists():
    DATASET_DIR = BASE_DIR / "dataset"
else:
    DATASET_DIR = Path("dataset").resolve()

TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"

# Submission and output directories
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", BASE_DIR / "output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MATCHING_OUTPUT = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_OUTPUT = OUTPUT_DIR / "candidate_pairs.tsv"

# Models and artifacts directory
MODELS_DIR = BASE_DIR / "code" / "business_entity_resolution" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Hardware and worker settings
DEFAULT_WORKERS = max(1, os.cpu_count() or 4)
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 100_000))

# Blocking parameters
MAX_CANDIDATES_PER_S1 = 100         # set to 100 per source for maximum candidate retrieval recall
MIN_POSTAL_LEN = 3

# Model hyperparameters
RANDOM_STATE = 42
DEFAULT_F05_THRESHOLD = 0.40        # was 0.65 → lowered; fine-grained tuning will override
