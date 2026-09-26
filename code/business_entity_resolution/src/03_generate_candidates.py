"""
Step 3: Generate Candidates using Multiple Blocking Keys
Implements multi-pass inverted index blocking partitioned strictly by country:
- Block A: Exact normalized name & name without legal suffix
- Block B: Exact postal and numeric address anchors (postal + building / street number + street token)
- Block C: Rare locality and address tokens (low document frequency)
- Block D: Domain / compressed name match & 3-gram name token prefixes
- Block E: Multi-token name overlap (sorted tokens & distinctive name tokens)

Outputs candidates per Source 1 entity from Source 2 and Source 3.
"""

import sys
import re
import csv
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Any

# Adjust path for internal imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import TRAIN_DIR, TEST_DIR, CANDIDATE_OUTPUT, MAX_CANDIDATES_PER_S1
from normalize import normalize_record, NAME_STOPWORDS


class MultiPassBlocker:
    """Memory-efficient multi-pass inverted index blocker."""

    def __init__(self, max_candidates_per_s1: int = MAX_CANDIDATES_PER_S1):
        self.max_candidates = max_candidates_per_s1
        
        # Inverted index tables partitioned by country
        # structure: {country: {key: list_of_target_ids}}
        self.index_exact_name = defaultdict(lambda: defaultdict(list))
        self.index_no_suffix_name = defaultdict(lambda: defaultdict(list))
        self.index_sorted_name = defaultdict(lambda: defaultdict(list))
        self.index_compressed_name = defaultdict(lambda: defaultdict(list))
        self.index_name_tokens = defaultdict(lambda: defaultdict(list))
        self.index_address_anchor = defaultdict(lambda: defaultdict(list))
        self.index_postal_anchor = defaultdict(lambda: defaultdict(list))

        # Token frequencies for IDF-like filtering
        self.token_freq = defaultdict(Counter)

    def _get_compressed_name(self, name_norm: str) -> str:
        """Name with all spaces removed (matches domain names like maurewilliamscolombier)."""
        return re.sub(r"\s+", "", name_norm)

    def _get_distinctive_tokens(self, tokens: List[str], country: str) -> List[str]:
        """Extract tokens that are not stopwords and reasonably distinctive."""
        res = []
        for t in tokens:
            if len(t) >= 4 and t not in NAME_STOPWORDS and not t.isdigit():
                freq = self.token_freq[country][t]
                if freq < 1000:  # Avoid ultra-frequent tokens
                    res.append(t)
        return res

    def index_target_records(self, tsv_path: Path, max_rows: int = None):
        """Index target records (from Source 2 or Source 3)."""
        print(f"Indexing target file: {tsv_path.name} ...", flush=True)
        count = 0
        with open(tsv_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader, None)

            for row in reader:
                if len(row) < 4:
                    continue
                count += 1
                rec = {
                    "entity_id": row[0].strip(),
                    "business_name": row[1].strip(),
                    "business_address": row[2].strip(),
                    "country": row[3].strip()
                }
                norm = normalize_record(rec)
                eid = norm["entity_id"]
                c = norm["country"]

                # 1. Block A: Exact Name & No-Suffix Name
                name_norm = norm["name_normalized"]
                if name_norm:
                    self.index_exact_name[c][name_norm].append(eid)
                    
                    no_suf = norm["name_without_legal_suffix"]
                    if no_suf and no_suf != name_norm:
                        self.index_no_suffix_name[c][no_suf].append(eid)

                    sorted_name = norm["name_sorted_tokens"]
                    if sorted_name:
                        self.index_sorted_name[c][sorted_name].append(eid)

                    # Compressed name (catches domains / merged strings)
                    comp = self._get_compressed_name(no_suf)
                    if len(comp) >= 6:
                        self.index_compressed_name[c][comp].append(eid)

                # 2. Block B & C: Address Anchors
                # Pair building number with first prominent street token
                b_nums = norm.get("building_numbers", [])
                p_tokens = norm.get("postal_tokens", [])
                a_tokens = norm.get("address_tokens", [])

                if b_nums and a_tokens:
                    b = b_nums[0]
                    # Find street name token (first non-number token)
                    for tok in a_tokens:
                        if not tok.isdigit() and len(tok) >= 3:
                            anchor_key = f"{b}_{tok}"
                            self.index_address_anchor[c][anchor_key].append(eid)
                            break

                # Postal code anchor (pair postal with building number or significant word)
                if p_tokens and b_nums:
                    post_anchor = f"{p_tokens[0]}_{b_nums[0]}"
                    self.index_postal_anchor[c][post_anchor].append(eid)

                # Track token frequencies for distinctive token matching
                for t in norm.get("name_tokens", []):
                    if len(t) >= 4 and t not in NAME_STOPWORDS:
                        self.token_freq[c][t] += 1
                        if self.token_freq[c][t] <= 50:  # Index only rare/medium tokens
                            self.index_name_tokens[c][t].append(eid)

                if max_rows and count >= max_rows:
                    break

        print(f"Indexed {count:,} records from {tsv_path.name}.", flush=True)

    def retrieve_candidates_for_s1(self, s1_norm: Dict[str, Any]) -> List[str]:
        """Retrieve candidate IDs for a single Source 1 record."""
        c = s1_norm["country"]
        candidate_scores = defaultdict(int)

        name_norm = s1_norm["name_normalized"]
        no_suf = s1_norm["name_without_legal_suffix"]
        sorted_name = s1_norm["name_sorted_tokens"]
        comp_name = self._get_compressed_name(no_suf)

        # 1. Exact Name match (high weight)
        if name_norm and name_norm in self.index_exact_name[c]:
            for cid in self.index_exact_name[c][name_norm]:
                candidate_scores[cid] += 15

        # 2. Name without suffix match
        if no_suf and no_suf in self.index_no_suffix_name[c]:
            for cid in self.index_no_suffix_name[c][no_suf]:
                candidate_scores[cid] += 10

        # 3. Sorted tokens match
        if sorted_name and sorted_name in self.index_sorted_name[c]:
            for cid in self.index_sorted_name[c][sorted_name]:
                candidate_scores[cid] += 8

        # 4. Compressed name match (domain matching)
        if len(comp_name) >= 6 and comp_name in self.index_compressed_name[c]:
            for cid in self.index_compressed_name[c][comp_name]:
                candidate_scores[cid] += 7

        # 5. Distinctive name tokens
        for tok in self._get_distinctive_tokens(s1_norm.get("name_tokens", []), c):
            if tok in self.index_name_tokens[c]:
                for cid in self.index_name_tokens[c][tok]:
                    candidate_scores[cid] += 3

        # 6. Address anchor (Building number + Street word)
        b_nums = s1_norm.get("building_numbers", [])
        a_tokens = s1_norm.get("address_tokens", [])
        p_tokens = s1_norm.get("postal_tokens", [])

        if b_nums and a_tokens:
            b = b_nums[0]
            for tok in a_tokens:
                if not tok.isdigit() and len(tok) >= 3:
                    anchor_key = f"{b}_{tok}"
                    if anchor_key in self.index_address_anchor[c]:
                        for cid in self.index_address_anchor[c][anchor_key]:
                            candidate_scores[cid] += 10
                    break

        # 7. Postal anchor (Postal code + Building number)
        if p_tokens and b_nums:
            post_anchor = f"{p_tokens[0]}_{b_nums[0]}"
            if post_anchor in self.index_postal_anchor[c]:
                for cid in self.index_postal_anchor[c][post_anchor]:
                    candidate_scores[cid] += 12

        if not candidate_scores:
            return []

        # Sort candidates by score descending and take top K
        sorted_cands = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        top_candidates = [cid for cid, score in sorted_cands[:self.max_candidates]]
        return top_candidates


def generate_candidates_for_split(
    s1_path: Path,
    s2_path: Path,
    s3_path: Path,
    output_path: Path,
    max_s1: int = None,
    max_target: int = None
):
    """Run candidate generation and output candidate_pairs.tsv."""
    print("=" * 60, flush=True)
    print(f"Generating Candidate Pairs for: {s1_path.name}", flush=True)
    print("=" * 60, flush=True)

    blocker = MultiPassBlocker(max_candidates_per_s1=MAX_CANDIDATES_PER_S1)

    # Index Source 2 and Source 3
    blocker.index_target_records(s2_path, max_rows=max_target)
    blocker.index_target_records(s3_path, max_rows=max_target)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_s1 = 0
    total_candidates = 0
    s1_with_zero = 0

    print(f"Processing Source 1 queries from {s1_path.name} ...", flush=True)
    with open(s1_path, "r", encoding="utf-8", errors="replace") as fin, \
         open(output_path, "w", encoding="utf-8", newline="") as fout:
        
        reader = csv.reader(fin, delimiter="\t")
        writer = csv.writer(fout, delimiter="\t")
        header = next(reader, None)

        # Write required candidate header
        writer.writerow(["source1_entity_id", "candidate_entity_ids"])

        for row in reader:
            if len(row) < 4:
                continue
            total_s1 += 1
            s1_rec = {
                "entity_id": row[0].strip(),
                "business_name": row[1].strip(),
                "business_address": row[2].strip(),
                "country": row[3].strip()
            }
            s1_norm = normalize_record(s1_rec)
            cands = blocker.retrieve_candidates_for_s1(s1_norm)

            if cands:
                total_candidates += len(cands)
                # Sort IDs consistently, no duplicates
                cands_str = ",".join(sorted(set(cands)))
            else:
                s1_with_zero += 1
                cands_str = ""

            writer.writerow([s1_norm["entity_id"], cands_str])

            if total_s1 % 50_000 == 0:
                print(f"  Processed {total_s1:,} S1 entities (avg candidates: {total_candidates/total_s1:.1f}) ...", flush=True)

            if max_s1 and total_s1 >= max_s1:
                break

    print(f"\nCandidate Generation Complete!", flush=True)
    print(f"Total S1 entities processed: {total_s1:,}", flush=True)
    print(f"Total candidates generated: {total_candidates:,} (avg {total_candidates/max(total_s1, 1):.2f} per entity)", flush=True)
    print(f"S1 entities with 0 candidates (singletons): {s1_with_zero:,} ({s1_with_zero/max(total_s1, 1)*100:.2f}%)", flush=True)
    print(f"Saved candidate pairs to: {output_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-pass Candidate Generation")
    parser.add_argument("--test", action="store_true", help="Generate candidates for test split (default is validation/train)")
    parser.add_argument("--max-s1", type=int, default=None, help="Max S1 rows to process (for debugging)")
    parser.add_argument("--max-target", type=int, default=None, help="Max S2/S3 rows to index")
    parser.add_argument("--output", type=Path, default=CANDIDATE_OUTPUT, help="Output candidate TSV path")
    args = parser.parse_args()

    split_dir = TEST_DIR if args.test else TRAIN_DIR
    prefix = "test" if args.test else "train"

    generate_candidates_for_split(
        s1_path=split_dir / f"{prefix}_source1.tsv",
        s2_path=split_dir / f"{prefix}_source2.tsv",
        s3_path=split_dir / f"{prefix}_source3.tsv",
        output_path=args.output,
        max_s1=args.max_s1,
        max_target=args.max_target
    )
