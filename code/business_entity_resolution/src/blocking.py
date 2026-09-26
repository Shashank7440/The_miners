"""
Core Blocking Engine for Business Entity Resolution.
Multi-pass inverted index partitioned by country.
Added: phonetic blocking, accent-stripped blocking, first-token+city blocking.
"""

import sys
import re
import csv
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import MAX_CANDIDATES_PER_S1
from normalize import normalize_record, NAME_STOPWORDS, remove_accents, get_char_trigrams


def get_initialism(tokens: List[str], stopwords: Set[str]) -> str:
    useful = [t for t in tokens if t and t not in stopwords and not t.isdigit()]
    return "".join(t[0] for t in useful)


class MultiPassBlocker:
    """Memory-efficient multi-pass inverted index blocker."""

    def __init__(self, max_candidates_per_s1: int = MAX_CANDIDATES_PER_S1):
        self.max_candidates = max_candidates_per_s1
        
        self.index_exact_name = defaultdict(lambda: defaultdict(list))
        self.index_no_suffix_name = defaultdict(lambda: defaultdict(list))
        self.index_sorted_name = defaultdict(lambda: defaultdict(list))
        self.index_compressed_name = defaultdict(lambda: defaultdict(list))
        self.index_name_tokens = defaultdict(lambda: defaultdict(list))
        self.index_address_anchor = defaultdict(lambda: defaultdict(list))
        self.index_postal_anchor = defaultdict(lambda: defaultdict(list))

        # NEW blocking indices for 95%+ recall & initialisms
        self.index_no_accent_name = defaultdict(lambda: defaultdict(list))
        self.index_first_token_city = defaultdict(lambda: defaultdict(list))
        self.index_postal_name = defaultdict(lambda: defaultdict(list))
        self.index_initialism = defaultdict(lambda: defaultdict(list))

        self.token_freq = defaultdict(Counter)

    def _get_compressed_name(self, name_norm: str) -> str:
        return re.sub(r"\s+", "", name_norm)

    def _get_distinctive_tokens(self, tokens: List[str], country: str) -> List[str]:
        res = []
        for t in tokens:
            if len(t) >= 3 and t not in NAME_STOPWORDS and not t.isdigit():
                freq = self.token_freq[country][t]
                if freq < 1000:
                    res.append(t)
        return res

    def _get_city_token(self, address_tokens: List[str]) -> str:
        """Extract likely city token (last non-numeric token with len>=4)."""
        non_num = [t for t in address_tokens if not t.isdigit() and len(t) >= 4]
        return non_num[-1] if non_num else ""

    def _index_single_record(self, norm: Dict[str, Any]):
        """Index a single normalized record into all blocking indices."""
        eid = norm["entity_id"]
        c = norm["country"]

        name_norm = norm["name_normalized"]
        if name_norm:
            self.index_exact_name[c][name_norm].append(eid)
            
            no_suf = norm["name_without_legal_suffix"]
            if no_suf and no_suf != name_norm:
                self.index_no_suffix_name[c][no_suf].append(eid)

            sorted_name = norm["name_sorted_tokens"]
            if sorted_name:
                self.index_sorted_name[c][sorted_name].append(eid)

            comp = self._get_compressed_name(no_suf)
            if len(comp) >= 5:
                self.index_compressed_name[c][comp].append(eid)

            # Accent-stripped name index (helps French entities)
            no_accent = norm.get("name_no_accent", "")
            if no_accent and no_accent != no_suf:
                self.index_no_accent_name[c][no_accent].append(eid)

        b_nums = norm.get("building_numbers", [])
        p_tokens = norm.get("postal_tokens", [])
        a_tokens = norm.get("address_tokens", [])
        name_tokens = norm.get("name_tokens", [])

        if b_nums and a_tokens:
            b = b_nums[0]
            for tok in a_tokens:
                if not tok.isdigit() and len(tok) >= 3:
                    anchor_key = f"{b}_{tok}"
                    self.index_address_anchor[c][anchor_key].append(eid)

        if p_tokens and b_nums:
            post_anchor = f"{p_tokens[0]}_{b_nums[0]}"
            self.index_postal_anchor[c][post_anchor].append(eid)

        # Co-index postal code + name token (catches entities with street variations)
        if p_tokens and name_tokens:
            for t in name_tokens:
                if t not in NAME_STOPWORDS and len(t) >= 3:
                    self.index_postal_name[c][f"{p_tokens[0]}_{t}"].append(eid)

        # First-name-token + city blocking
        city = self._get_city_token(a_tokens)
        if name_tokens and city:
            ft = name_tokens[0]
            if len(ft) >= 3:
                self.index_first_token_city[c][f"{ft}_{city}"].append(eid)

        # Initialism indexing (e.g. HDFC, SBI)
        init_key = get_initialism(name_tokens, NAME_STOPWORDS)
        if len(init_key) >= 2:
            self.index_initialism[c][init_key].append(eid)

        for t in name_tokens:
            if len(t) >= 3 and t not in NAME_STOPWORDS and not t.isdigit():
                self.token_freq[c][t] += 1
                if self.token_freq[c][t] <= 500:
                    self.index_name_tokens[c][t].append(eid)

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
                self._index_single_record(norm)

                if max_rows and count >= max_rows:
                    break

        print(f"Indexed {count:,} records from {tsv_path.name}.", flush=True)

    def retrieve_candidates_for_s1(self, s1_norm: Dict[str, Any]) -> List[str]:
        """Retrieve candidate IDs for a single Source 1 record."""
        c = s1_norm["country"]
        candidate_scores = defaultdict(int)
        strong_candidates = set()

        name_norm = s1_norm["name_normalized"]
        no_suf = s1_norm["name_without_legal_suffix"]
        sorted_name = s1_norm["name_sorted_tokens"]
        comp_name = self._get_compressed_name(no_suf)
        b_nums = s1_norm.get("building_numbers", [])
        p_tokens = s1_norm.get("postal_tokens", [])
        a_tokens = s1_norm.get("address_tokens", [])
        name_tokens = s1_norm.get("name_tokens", [])

        # Pass 1: Exact normalized name
        if name_norm and name_norm in self.index_exact_name[c]:
            for cid in self.index_exact_name[c][name_norm]:
                candidate_scores[cid] += 20
                if p_tokens or b_nums:
                    strong_candidates.add(cid)

        # Pass 2: Legal suffix-stripped name
        if no_suf and no_suf in self.index_no_suffix_name[c]:
            for cid in self.index_no_suffix_name[c][no_suf]:
                candidate_scores[cid] += 15

        # Pass 3: Sorted token name
        if sorted_name and sorted_name in self.index_sorted_name[c]:
            for cid in self.index_sorted_name[c][sorted_name]:
                candidate_scores[cid] += 12

        # Pass 4: Compressed name (whitespace-insensitive)
        if len(comp_name) >= 5 and comp_name in self.index_compressed_name[c]:
            for cid in self.index_compressed_name[c][comp_name]:
                candidate_scores[cid] += 10

        # Pass 5: Accent-stripped name (French entities)
        no_accent = s1_norm.get("name_no_accent", "")
        if no_accent and no_accent in self.index_no_accent_name[c]:
            for cid in self.index_no_accent_name[c][no_accent]:
                candidate_scores[cid] += 10

        # Pass 6: Initialisms (e.g. HDFC, SBI)
        init_key = get_initialism(name_tokens, NAME_STOPWORDS)
        if len(init_key) >= 2 and init_key in self.index_initialism[c]:
            for cid in self.index_initialism[c][init_key]:
                candidate_scores[cid] += 8

        # Pass 7: Distinctive low-frequency name tokens
        for tok in name_tokens:
            if len(tok) >= 3 and tok not in NAME_STOPWORDS and tok in self.index_name_tokens[c]:
                freq = self.token_freq[c][tok]
                weight = 10 if freq < 50 else (5 if freq < 200 else 2)
                for cid in self.index_name_tokens[c][tok]:
                    candidate_scores[cid] += weight

        # Pass 8: Address anchor (building_number + street keyword)
        if b_nums and a_tokens:
            b = b_nums[0]
            for tok in a_tokens:
                if not tok.isdigit() and len(tok) >= 3:
                    anchor_key = f"{b}_{tok}"
                    if anchor_key in self.index_address_anchor[c]:
                        for cid in self.index_address_anchor[c][anchor_key]:
                            candidate_scores[cid] += 12

        # Pass 9: Postal anchor (postal_code + building_number)
        if p_tokens and b_nums:
            post_anchor = f"{p_tokens[0]}_{b_nums[0]}"
            if post_anchor in self.index_postal_anchor[c]:
                for cid in self.index_postal_anchor[c][post_anchor]:
                    candidate_scores[cid] += 15

        # Pass 10: Postal code + name token
        if p_tokens and name_tokens:
            for t in name_tokens:
                if t not in NAME_STOPWORDS and len(t) >= 3:
                    key = f"{p_tokens[0]}_{t}"
                    if key in self.index_postal_name[c]:
                        for cid in self.index_postal_name[c][key]:
                            candidate_scores[cid] += 10

        # Pass 11: First-name-token + city
        city = self._get_city_token(a_tokens)
        if name_tokens and city:
            ft = name_tokens[0]
            if len(ft) >= 3:
                key = f"{ft}_{city}"
                if key in self.index_first_token_city[c]:
                    for cid in self.index_first_token_city[c][key]:
                        candidate_scores[cid] += 6

        if not candidate_scores:
            return []

        # Strong candidate protection: Keep strong candidates, then fill remaining slots with top fuzzy candidates
        remaining_slots = max(0, self.max_candidates - len(strong_candidates))
        fuzzy_sorted = sorted(
            [(cid, sc) for cid, sc in candidate_scores.items() if cid not in strong_candidates],
            key=lambda x: x[1],
            reverse=True
        )[:remaining_slots]

        result_cands = list(strong_candidates)
        result_cands.extend(cid for cid, _ in fuzzy_sorted)
        return result_cands
