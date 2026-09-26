"""
Step 1: Load and Audit the Data
Performs data audit across train and test datasets:
- Row counts and column headers
- Entity ID prefix validation (S1-, S2-, S3-)
- Duplicate entity_id detection
- Missing values in name, address, country
- Country distribution (training vs test open set)
- Ground truth match statistics and singleton percentage
"""

import os
import sys
import argparse
import pandas as pd
from collections import Counter
from pathlib import Path

# Adjust path for internal imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import TRAIN_DIR, TEST_DIR


def audit_source_file(file_path: Path, expected_prefix: str, sample_rows: int = 1000, full_scan: bool = False):
    """Audit an entity source TSV file."""
    print(f"\n==================================================", flush=True)
    print(f"Auditing file: {file_path.name}", flush=True)
    print(f"Path: {file_path}", flush=True)
    print(f"==================================================", flush=True)

    if not file_path.exists():
        print(f"ERROR: File {file_path} does not exist!", flush=True)
        return

    # 1. Quick sample audit with pandas
    df_sample = pd.read_csv(file_path, sep="\t", dtype=str, nrows=sample_rows)
    print(f"Columns found: {list(df_sample.columns)}", flush=True)
    print(f"Sample preview (first 2 rows):", flush=True)
    for idx, row in df_sample.head(2).iterrows():
        print(f"  [{row['entity_id']}] Name: '{row.get('business_name', '')}' | Addr: '{row.get('business_address', '')}' | Country: '{row.get('country', '')}'", flush=True)

    # 2. Scanning file metrics
    total_rows = 0
    missing_names = 0
    missing_addrs = 0
    wrong_prefix_count = 0
    country_counts = Counter()
    seen_ids = set()
    duplicate_ids = 0

    chunksize = 50_000
    max_rows = None if full_scan else 50_000

    print(f"Scanning records (mode: {'FULL SCAN' if full_scan else f'Audit Sample ({max_rows:,} rows)'})...", flush=True)
    
    for chunk in pd.read_csv(file_path, sep="\t", dtype=str, chunksize=chunksize, keep_default_na=False):
        for _, row in chunk.iterrows():
            eid = row["entity_id"].strip()
            name = row.get("business_name", "").strip()
            addr = row.get("business_address", "").strip()
            country = row.get("country", "").strip()

            total_rows += 1
            if not eid.startswith(expected_prefix):
                wrong_prefix_count += 1
            if full_scan:
                if eid in seen_ids:
                    duplicate_ids += 1
                else:
                    seen_ids.add(eid)

            if not name:
                missing_names += 1
            if not addr:
                missing_addrs += 1
            country_counts[country] += 1

            if max_rows and total_rows >= max_rows:
                break
        if max_rows and total_rows >= max_rows:
            break

    print(f"Total Rows Scanned: {total_rows:,}", flush=True)
    print(f"Prefix Validity ({expected_prefix}): {'PASS (all valid)' if wrong_prefix_count == 0 else f'FAIL ({wrong_prefix_count} invalid)'}", flush=True)
    if full_scan:
        print(f"Duplicate IDs: {duplicate_ids} ({'PASS (unique)' if duplicate_ids == 0 else 'FAIL'})", flush=True)
    print(f"Missing Business Names: {missing_names:,} ({missing_names/total_rows*100:.2f}%)", flush=True)
    print(f"Missing Business Addresses: {missing_addrs:,} ({missing_addrs/total_rows*100:.2f}%)", flush=True)
    print(f"Country Breakdown: {dict(country_counts)}", flush=True)


def audit_ground_truth(gt_path: Path, full_scan: bool = False):
    """Audit ground truth matching file and singleton stats."""
    print(f"\n==================================================", flush=True)
    print(f"Auditing Ground Truth: {gt_path.name}", flush=True)
    print(f"==================================================", flush=True)
    
    if not gt_path.exists():
        print(f"ERROR: Ground truth {gt_path} not found!", flush=True)
        return

    total = 0
    singletons = 0
    matched_count = 0
    match_lengths = []
    seen_s1 = set()
    dup_s1 = 0
    s2_matches = 0
    s3_matches = 0
    max_rows = None if full_scan else 50_000

    for chunk in pd.read_csv(gt_path, sep="\t", dtype=str, chunksize=50_000, keep_default_na=False):
        for _, row in chunk.iterrows():
            s1 = row["source1_entity_id"].strip()
            total += 1
            if s1 in seen_s1:
                dup_s1 += 1
            seen_s1.add(s1)

            mids_str = row["matched_entity_ids"].strip()
            if not mids_str:
                singletons += 1
            else:
                matched_count += 1
                mids = [m.strip() for m in mids_str.split(",") if m.strip()]
                match_lengths.append(len(mids))
                for m in mids:
                    if m.startswith("S2-"):
                        s2_matches += 1
                    elif m.startswith("S3-"):
                        s3_matches += 1
            if max_rows and total >= max_rows:
                break
        if max_rows and total >= max_rows:
            break

    print(f"Total Ground Truth Records: {total:,}", flush=True)
    print(f"Duplicate S1 rows: {dup_s1} (should be 0)", flush=True)
    print(f"Singletons (Zero matches): {singletons:,} ({singletons/total*100:.2f}%)", flush=True)
    print(f"Entities with Matches: {matched_count:,} ({matched_count/total*100:.2f}%)", flush=True)
    if match_lengths:
        print(f"Matches per Matched Entity: Mean = {sum(match_lengths)/len(match_lengths):.2f}, Min = {min(match_lengths)}, Max = {max(match_lengths)}", flush=True)
    print(f"Total S2 Matches: {s2_matches:,}, Total S3 Matches: {s3_matches:,}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Audit ML Challenge 2026 Datasets")
    parser.add_argument("--train-dir", type=Path, default=TRAIN_DIR, help="Path to train directory")
    parser.add_argument("--test-dir", type=Path, default=TEST_DIR, help="Path to test directory")
    parser.add_argument("--full", action="store_true", help="Perform full dataset scan (slower)")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 60)
    print("ML Challenge 2026: Business Entity Resolution - Data Audit")
    print("=" * 60)

    # Audit Training Files
    print(f"\n--- Checking Training Files in: {args.train_dir} ---")
    audit_source_file(args.train_dir / "train_source1.tsv", "S1-", full_scan=args.full)
    audit_source_file(args.train_dir / "train_source2.tsv", "S2-", full_scan=args.full)
    audit_source_file(args.train_dir / "train_source3.tsv", "S3-", full_scan=args.full)
    audit_ground_truth(args.train_dir / "train_ground_truth.tsv")

    # Audit Test Files
    print(f"\n--- Checking Test Files in: {args.test_dir} ---")
    audit_source_file(args.test_dir / "test_source1.tsv", "S1-", full_scan=args.full)
    audit_source_file(args.test_dir / "test_source2.tsv", "S2-", full_scan=args.full)
    audit_source_file(args.test_dir / "test_source3.tsv", "S3-", full_scan=args.full)
    print("\nData Audit Completed Successfully!")


if __name__ == "__main__":
    main()
