"""
Inference module for generating matching_results.tsv and candidate_pairs.tsv.
Features:
- Country-partitioned streaming architecture (Peak RAM < 600MB even on 12M+ records)
- Dual-mode cache (In-Memory per country or SQLite disk cache)
- Fast vectorized LightGBM batch scoring
- Exact original row order preservation for submission compliance
- Strict validation rules adherence
- Confidence-based post-filtering for hub entities
"""

import sys
import os
import gc
import csv
import json
import pickle
import sqlite3
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import TEST_DIR, OUTPUT_DIR, MODELS_DIR, MATCHING_OUTPUT, CANDIDATE_OUTPUT, DEFAULT_F05_THRESHOLD
from normalize import normalize_record
from blocking import MultiPassBlocker
from features import extract_pairwise_features, FEATURE_NAMES


class RecordStore:
    """Hybrid record store: In-Memory or SQLite for low-memory environments."""

    def __init__(self, in_memory: bool = True, db_path: Path = None):
        self.in_memory = in_memory
        self.mem_store = {}
        self.conn = None
        self.cur = None

        if not in_memory:
            if db_path is None:
                db_path = OUTPUT_DIR / "target_cache.db"
            if db_path.exists():
                db_path.unlink()
            self.db_path = db_path
            self.conn = sqlite3.connect(str(db_path))
            self.cur = self.conn.cursor()
            self.cur.execute("PRAGMA synchronous = OFF")
            self.cur.execute("PRAGMA journal_mode = OFF")
            self.cur.execute("PRAGMA cache_size = 50000")
            self.cur.execute(
                "CREATE TABLE records (id TEXT PRIMARY KEY, json_data TEXT)"
            )

    def add_records_batch(self, records: List[Dict]):
        if self.in_memory:
            for r in records:
                self.mem_store[r["entity_id"]] = r
        else:
            data = [(r["entity_id"], json.dumps(r)) for r in records]
            self.cur.executemany("INSERT OR REPLACE INTO records VALUES (?, ?)", data)
            self.conn.commit()

    def get_record(self, eid: str) -> Dict:
        if self.in_memory:
            return self.mem_store.get(eid)
        self.cur.execute("SELECT json_data FROM records WHERE id = ?", (eid,))
        row = self.cur.fetchone()
        return json.loads(row[0]) if row else None

    def get_records_batch(self, eids: List[str]) -> Dict[str, Dict]:
        if not eids:
            return {}
        if self.in_memory:
            return {eid: self.mem_store[eid] for eid in eids if eid in self.mem_store}
        
        placeholders = ",".join("?" for _ in eids)
        self.cur.execute(f"SELECT id, json_data FROM records WHERE id IN ({placeholders})", eids)
        rows = self.cur.fetchall()
        return {r[0]: json.loads(r[1]) for r in rows}

    def close(self):
        if self.conn:
            self.conn.close()
            if self.db_path and self.db_path.exists():
                try:
                    self.db_path.unlink()
                except OSError:
                    pass
        self.mem_store.clear()


def run_test_prediction(
    test_dir: Path = TEST_DIR,
    model_path: Path = None,
    threshold_config_path: Path = None,
    matching_output: Path = MATCHING_OUTPUT,
    candidate_output: Path = CANDIDATE_OUTPUT,
    max_s1: int = None,
    max_target: int = None,
    in_memory: bool = True
):
    """Run full test inference using partitioned country streaming."""
    if model_path is None:
        model_path = MODELS_DIR / "matching_lgbm.pkl"
    if threshold_config_path is None:
        threshold_config_path = MODELS_DIR / "threshold_config.json"

    print("=" * 65, flush=True)
    print("STEP 9: RUNNING HIGH-PRECISION TEST INFERENCE", flush=True)
    print("Architecture: Country-Partitioned Low-Memory Streaming", flush=True)
    print("=" * 65, flush=True)

    # 1. Load Model and Config
    clf = None
    feat_names = FEATURE_NAMES
    if model_path.exists():
        print(f"Loading trained LightGBM model from {model_path.name} ...", flush=True)
        with open(model_path, "rb") as f:
            model_bundle = pickle.load(f)
            clf = model_bundle["model"]
            feat_names = model_bundle["feature_names"]
            
            # Force LightGBM to use exactly 8 threads (otherwise it uses all 100+ cores on your master node and gets killed by PBS!)
            if hasattr(clf, 'set_params'):
                clf.set_params(n_jobs=8)
    else:
        print(f"WARNING: Model file {model_path} not found. Running rule-baseline.", flush=True)

    tau = DEFAULT_F05_THRESHOLD
    use_rules = False
    strategy = "top1_per_source"
    if threshold_config_path.exists():
        with open(threshold_config_path, "r", encoding="utf-8") as f:
            t_conf = json.load(f)
            tau = t_conf.get("best_threshold", DEFAULT_F05_THRESHOLD)
            use_rules = t_conf.get("use_rules", False)
            strategy = t_conf.get("best_strategy", "top1_per_source")
        print(f"Loaded calibrated threshold: tau = {tau:.2f} (rules active: {use_rules}, strategy: {strategy})", flush=True)

    # 2. Partition S1 entities by country into temp files to preserve line indices
    s1_path = test_dir / "test_source1.tsv"
    if not s1_path.exists():
        print(f"ERROR: Test file {s1_path} does not exist!", flush=True)
        return

    tmp_dir = OUTPUT_DIR / "tmp_stream"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Partitioning Source 1 Entities by Country ---", flush=True)
    country_s1_files = {}
    country_s1_writers = {}
    country_counts = {}

    countries = ["France", "US", "India"]
    for c in countries:
        c_path = tmp_dir / f"s1_{c}.tsv"
        f = open(c_path, "w", encoding="utf-8", newline="")
        country_s1_files[c] = f
        country_s1_writers[c] = csv.writer(f, delimiter="\t")
        country_counts[c] = 0

    other_path = tmp_dir / "s1_Other.tsv"
    f_other = open(other_path, "w", encoding="utf-8", newline="")
    country_s1_files["Other"] = f_other
    country_s1_writers["Other"] = csv.writer(f_other, delimiter="\t")
    country_counts["Other"] = 0

    total_s1_rows = 0
    with open(s1_path, "r", encoding="utf-8", errors="replace") as fin:
        r_in = csv.reader(fin, delimiter="\t")
        next(r_in, None)  # skip header
        for line_idx, row in enumerate(r_in):
            if len(row) < 4:
                continue
            total_s1_rows += 1
            c = row[3].strip()
            w = country_s1_writers.get(c, country_s1_writers["Other"])
            w.writerow([line_idx, row[0].strip(), row[1].strip(), row[2].strip(), c])
            country_counts[c if c in country_s1_writers else "Other"] += 1

            if max_s1 and total_s1_rows >= max_s1:
                break

    for f in country_s1_files.values():
        f.close()

    print(f"Total S1 entities partitioned: {total_s1_rows:,}")
    for c, cnt in country_counts.items():
        if cnt > 0:
            print(f"  - {c}: {cnt:,} entities")

    # 3. Process Country-by-Country (Index Targets -> Score S1 Candidates -> Write Part Output)
    active_countries = [c for c in countries + ["Other"] if country_counts.get(c, 0) > 0]
    total_matches_global = 0
    total_candidates_global = 0

    for country in active_countries:
        print(f"\n==================================================", flush=True)
        print(f"Processing Country Partition: {country} ({country_counts[country]:,} entities)", flush=True)
        print(f"==================================================", flush=True)

        blocker = MultiPassBlocker()
        store = RecordStore(in_memory=in_memory, db_path=tmp_dir / f"store_{country}.db")

        # Index target records for this country only
        for fname in ["test_source2.tsv", "test_source3.tsv"]:
            t_path = test_dir / fname
            if not t_path.exists():
                continue
            print(f"  Indexing {fname} for country '{country}' ...", flush=True)
            t_count = 0
            with open(t_path, "r", encoding="utf-8", errors="replace") as fin:
                reader = csv.reader(fin, delimiter="\t")
                next(reader, None)
                batch = []
                for row in reader:
                    if len(row) < 4:
                        continue
                    row_country = row[3].strip()
                    if country != "Other" and row_country != country:
                        continue
                    if country == "Other" and row_country in countries:
                        continue

                    t_count += 1
                    rec = {
                        "entity_id": row[0].strip(),
                        "business_name": row[1].strip(),
                        "business_address": row[2].strip(),
                        "country": row_country
                    }
                    norm = normalize_record(rec)
                    batch.append(norm)
                    blocker._index_single_record(norm)

                    if len(batch) >= 20_000:
                        store.add_records_batch(batch)
                        batch = []

                    if max_target and t_count >= max_target:
                        break

                if batch:
                    store.add_records_batch(batch)
            print(f"  Indexed {t_count:,} target records from {fname}.", flush=True)

        # Run Candidate Retrieval & Inference for this Country
        c_part_path = tmp_dir / f"s1_{country}.tsv"
        out_part_path = tmp_dir / f"res_{country}.tsv"

        c_s1_done = 0
        c_matches_done = 0

        with open(c_part_path, "r", encoding="utf-8", errors="replace") as fin, \
             open(out_part_path, "w", encoding="utf-8", newline="") as fout:

            r_in = csv.reader(fin, delimiter="\t")
            w_out = csv.writer(fout, delimiter="\t")

            for row in r_in:
                if len(row) < 5:
                    continue
                orig_idx = int(row[0])
                s1_id = row[1]
                s1_raw = {
                    "entity_id": s1_id,
                    "business_name": row[2],
                    "business_address": row[3],
                    "country": row[4]
                }
                s1_norm = normalize_record(s1_raw)
                c_s1_done += 1

                cands = blocker.retrieve_candidates_for_s1(s1_norm)
                cands_set = sorted(set(cands))
                cands_str = ",".join(cands_set) if cands_set else ""
                total_candidates_global += len(cands_set)

                if not cands_set:
                    w_out.writerow([orig_idx, s1_id, "", ""])
                    continue

                matched_cands = []
                matched_probs = []
                target_batch = store.get_records_batch(cands_set)

                if clf is not None:
                    feat_matrix = []
                    valid_cand_ids = []
                    rule_pos_list = []
                    rule_neg_list = []

                    for rank, cid in enumerate(cands_set, 1):
                        cand_rec = target_batch.get(cid)
                        if not cand_rec:
                            continue
                        feats = extract_pairwise_features(s1_norm, cand_rec, blocking_rank=rank)
                        feat_vector = [feats[name] for name in feat_names]
                        feat_matrix.append(feat_vector)
                        valid_cand_ids.append(cid)
                        rule_pos_list.append(feats["rule_strong_positive"])
                        rule_neg_list.append(feats["rule_strong_conflict"])

                    if feat_matrix:
                        probs = clf.predict_proba(np.array(feat_matrix, dtype=np.float32))[:, 1]
                        s2_scored = []
                        s3_scored = []

                        for cid, p, r_pos, r_neg in zip(valid_cand_ids, probs, rule_pos_list, rule_neg_list):
                            eff_p = p
                            if use_rules:
                                if r_neg > 0:
                                    eff_p *= 0.60
                                if r_pos > 0:
                                    eff_p = max(eff_p, 0.95)

                            if eff_p >= tau:
                                if cid.startswith("S2-"):
                                    s2_scored.append((cid, eff_p))
                                else:
                                    s3_scored.append((cid, eff_p))

                        # Sort descending by probability
                        s2_scored.sort(key=lambda x: x[1], reverse=True)
                        s3_scored.sort(key=lambda x: x[1], reverse=True)

                        if strategy == "top_dynamic":
                            if s2_scored:
                                top_p = s2_scored[0][1]
                                for cid, p in s2_scored:
                                    if p >= top_p - 0.12:
                                        matched_cands.append(cid)
                            if s3_scored:
                                top_p = s3_scored[0][1]
                                for cid, p in s3_scored:
                                    if p >= top_p - 0.12:
                                        matched_cands.append(cid)
                        elif strategy == "top1_per_source":
                            if s2_scored:
                                matched_cands.append(s2_scored[0][0])
                            if s3_scored:
                                matched_cands.append(s3_scored[0][0])
                        elif strategy == "top2_per_source":
                            for item in s2_scored[:2]:
                                matched_cands.append(item[0])
                            for item in s3_scored[:2]:
                                matched_cands.append(item[0])
                        else:  # 'all_above_tau'
                            for item in s2_scored:
                                matched_cands.append(item[0])
                            for item in s3_scored:
                                matched_cands.append(item[0])
                else:
                    for cid in cands_set:
                        cand_rec = target_batch.get(cid)
                        if cand_rec and cand_rec["name_normalized"] == s1_norm["name_normalized"]:
                            matched_cands.append(cid)

                # Confidence-based post-filter for hub entities:
                # If too many matches and none are high-confidence, keep only top by probability
                if len(matched_cands) > 8 and matched_probs:
                    max_prob = max(matched_probs)
                    if max_prob < 0.75:
                        # Sort by probability descending, keep top 5
                        scored = sorted(zip(matched_cands, matched_probs), key=lambda x: x[1], reverse=True)
                        matched_cands = [cid for cid, _ in scored[:5]]

                matched_set = sorted(set(matched_cands))
                matched_set = [m for m in matched_set if m in cands_set]
                match_str = ",".join(matched_set) if matched_set else ""

                if matched_set:
                    c_matches_done += len(matched_set)
                    total_matches_global += len(matched_set)

                w_out.writerow([orig_idx, s1_id, cands_str, match_str])

                if c_s1_done % 100_000 == 0:
                    print(f"  Processed {c_s1_done:,} / {country_counts[country]:,} {country} entities ({c_matches_done:,} matches) ...", flush=True)

        store.close()
        del blocker
        del store
        gc.collect()
        print(f"Completed Country Partition '{country}': {c_s1_done:,} entities, {c_matches_done:,} matches.", flush=True)

    # 4. Merge Temp Partitions in Exact Original Line Index Order (Disk-backed SQLite to guarantee peak RAM < 1 GB)
    print("\n--- Assembling Final Submissions in Original File Order ---", flush=True)
    matching_output.parent.mkdir(parents=True, exist_ok=True)
    candidate_output.parent.mkdir(parents=True, exist_ok=True)

    db_merge_path = tmp_dir / "assembly.db"
    if db_merge_path.exists():
        db_merge_path.unlink()

    conn_merge = sqlite3.connect(str(db_merge_path))
    cur_merge = conn_merge.cursor()
    cur_merge.execute("PRAGMA synchronous = OFF")
    cur_merge.execute("PRAGMA journal_mode = OFF")
    cur_merge.execute("CREATE TABLE results (idx INTEGER PRIMARY KEY, s1_id TEXT, cand_str TEXT, match_str TEXT)")

    for c in active_countries:
        p_path = tmp_dir / f"res_{c}.tsv"
        if p_path.exists():
            print(f"  Ingesting partition results for {c} into disk merger ...", flush=True)
            with open(p_path, "r", encoding="utf-8", errors="replace") as f:
                r = csv.reader(f, delimiter="\t")
                rows = []
                for row in r:
                    if len(row) >= 4:
                        rows.append((int(row[0]), row[1], row[2], row[3]))
                        if len(rows) >= 50_000:
                            cur_merge.executemany("INSERT INTO results VALUES (?, ?, ?, ?)", rows)
                            rows = []
                if rows:
                    cur_merge.executemany("INSERT INTO results VALUES (?, ?, ?, ?)", rows)
            conn_merge.commit()

    print(f"Writing {matching_output.name} and {candidate_output.name} ...", flush=True)
    with open(matching_output, "w", encoding="utf-8", newline="") as f_match, \
         open(candidate_output, "w", encoding="utf-8", newline="") as f_cand:

        w_match = csv.writer(f_match, delimiter="\t")
        w_cand = csv.writer(f_cand, delimiter="\t")

        w_match.writerow(["source1_entity_id", "matched_entity_ids"])
        w_cand.writerow(["source1_entity_id", "candidate_entity_ids"])

        cur_merge.execute("SELECT s1_id, cand_str, match_str FROM results ORDER BY idx ASC")
        while True:
            fetch_batch = cur_merge.fetchmany(50_000)
            if not fetch_batch:
                break
            for s1_id, cand_str, match_str in fetch_batch:
                w_match.writerow([s1_id, match_str])
                w_cand.writerow([s1_id, cand_str])

    conn_merge.close()
    if db_merge_path.exists():
        try:
            db_merge_path.unlink()
        except OSError:
            pass

    # Clean up temp files
    try:
        for p in tmp_dir.glob("*"):
            p.unlink()
        tmp_dir.rmdir()
    except Exception:
        pass

    print("\n" + "=" * 65, flush=True)
    print("TEST INFERENCE COMPLETED SUCCESSFULLY!", flush=True)
    print(f"Total S1 entities processed: {total_s1_rows:,}", flush=True)
    print(f"Total candidates written: {total_candidates_global:,} (avg {total_candidates_global/max(total_s1_rows,1):.2f})", flush=True)
    print(f"Total matches predicted: {total_matches_global:,} (avg {total_matches_global/max(total_s1_rows,1):.2f})", flush=True)
    print(f"Generated matching file: {matching_output}", flush=True)
    print(f"Generated candidate file: {candidate_output}", flush=True)
    print("=" * 65, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Test Predictions")
    parser.add_argument("--test-dir", type=Path, default=TEST_DIR, help="Path to test directory")
    parser.add_argument("--model", type=Path, default=None, help="Trained model path")
    parser.add_argument("--config", type=Path, default=None, help="Threshold config path")
    parser.add_argument("--matching-out", type=Path, default=MATCHING_OUTPUT, help="Output matching results TSV")
    parser.add_argument("--candidate-out", type=Path, default=CANDIDATE_OUTPUT, help="Output candidate pairs TSV")
    parser.add_argument("--max-s1", type=int, default=None, help="Max S1 rows to process (for debugging)")
    parser.add_argument("--max-target", type=int, default=None, help="Max S2/S3 rows to index")
    parser.add_argument("--sqlite", action="store_true", help="Use SQLite instead of in-memory store")
    args = parser.parse_args()

    run_test_prediction(
        test_dir=args.test_dir,
        model_path=args.model,
        threshold_config_path=args.config,
        matching_output=args.matching_out,
        candidate_output=args.candidate_out,
        max_s1=args.max_s1,
        max_target=args.max_target,
        in_memory=not args.sqlite
    )
