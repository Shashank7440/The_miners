"""
Core Feature Extraction Module for Business Entity Resolution.
Extracts name, address, numbers, and composite similarity signals.
Expanded: Jaro-Winkler, trigram dice, initialism, length ratios, shared distinctive tokens.
"""

from typing import Dict, List, Any
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from normalize import NAME_STOPWORDS, remove_accents, get_initialism, get_char_trigrams, get_char_bigrams


def compute_jaccard(tokens1: List[str], tokens2: List[str]) -> float:
    """Compute token Jaccard similarity."""
    if not tokens1 or not tokens2:
        return 0.0
    s1, s2 = set(tokens1), set(tokens2)
    union_len = len(s1 | s2)
    return len(s1 & s2) / union_len if union_len > 0 else 0.0


def compute_containment(tokens1: List[str], tokens2: List[str]) -> float:
    """Compute token containment similarity: |A ∩ B| / min(|A|, |B|)."""
    if not tokens1 or not tokens2:
        return 0.0
    s1, s2 = set(tokens1), set(tokens2)
    min_len = min(len(s1), len(s2))
    return len(s1 & s2) / min_len if min_len > 0 else 0.0


def trigram_dice(s1: str, s2: str, n: int = 3) -> float:
    """Dice coefficient on character trigrams (spaces removed)."""
    if not s1 or not s2:
        return 0.0
    g1 = get_char_trigrams(s1, n)
    g2 = get_char_trigrams(s2, n)
    if not g1 or not g2:
        return 0.0
    return 2.0 * len(g1 & g2) / (len(g1) + len(g2))


def bigram_dice(s1: str, s2: str) -> float:
    """Dice coefficient on character bigrams (spaces removed)."""
    if not s1 or not s2:
        return 0.0
    g1 = get_char_bigrams(s1)
    g2 = get_char_bigrams(s2)
    if not g1 or not g2:
        return 0.0
    return 2.0 * len(g1 & g2) / (len(g1) + len(g2))


def count_shared_distinctive(t1: List[str], t2: List[str]) -> tuple:
    """Count shared distinctive tokens (len>=4, not stopwords)."""
    s1 = {t for t in t1 if t not in NAME_STOPWORDS and len(t) >= 4}
    s2 = {t for t in t2 if t not in NAME_STOPWORDS and len(t) >= 4}
    shared = len(s1 & s2)
    total = len(s1 | s2)
    return shared, total


def extract_pairwise_features(s1: Dict[str, Any], cand: Dict[str, Any], blocking_rank: int = 1) -> Dict[str, float]:
    """Extract full numeric feature vector for a candidate pair."""
    feats = {}

    # 1. Blocking Metadata
    feats["blocking_rank"] = float(blocking_rank)
    feats["is_source2"] = 1.0 if cand["entity_id"].startswith("S2-") else 0.0
    feats["is_source3"] = 1.0 if cand["entity_id"].startswith("S3-") else 0.0
    feats["country_match"] = 1.0 if s1["country"] == cand["country"] else 0.0

    # 2. Missingness flags
    feats["s1_has_empty_addr"] = 1.0 if s1["is_addr_empty"] else 0.0
    feats["cand_has_empty_addr"] = 1.0 if cand["is_addr_empty"] else 0.0
    feats["both_have_address"] = 1.0 if (not s1["is_addr_empty"] and not cand["is_addr_empty"]) else 0.0

    # 3. Name features (Exact and Fuzzy)
    n1 = s1["name_normalized"]
    n2 = cand["name_normalized"]

    feats["name_exact_match"] = 1.0 if n1 and n1 == n2 else 0.0
    feats["name_levenshtein_ratio"] = fuzz.ratio(n1, n2) / 100.0 if n1 and n2 else 0.0
    feats["name_token_sort_ratio"] = fuzz.token_sort_ratio(n1, n2) / 100.0 if n1 and n2 else 0.0
    feats["name_token_set_ratio"] = fuzz.token_set_ratio(n1, n2) / 100.0 if n1 and n2 else 0.0
    feats["name_partial_ratio"] = fuzz.partial_ratio(n1, n2) / 100.0 if n1 and n2 else 0.0
    
    t1 = s1.get("name_tokens", [])
    t2 = cand.get("name_tokens", [])
    feats["name_jaccard"] = compute_jaccard(t1, t2)
    feats["name_containment"] = compute_containment(t1, t2)
    feats["same_first_token"] = 1.0 if (t1 and t2 and t1[0] == t2[0]) else 0.0

    # Name without legal suffix
    ns1 = s1["name_without_legal_suffix"]
    ns2 = cand["name_without_legal_suffix"]
    feats["name_no_suffix_exact"] = 1.0 if ns1 and ns1 == ns2 else 0.0
    feats["name_no_suffix_ratio"] = fuzz.ratio(ns1, ns2) / 100.0 if ns1 and ns2 else 0.0
    feats["name_no_suffix_sort"] = fuzz.token_sort_ratio(ns1, ns2) / 100.0 if ns1 and ns2 else 0.0

    # NEW: Jaro-Winkler similarity (good for short strings/typos)
    feats["name_jaro_winkler"] = JaroWinkler.similarity(n1, n2) if n1 and n2 else 0.0
    
    # NEW: Character trigram & bigram Dice coefficient
    feats["name_trigram_dice"] = trigram_dice(ns1, ns2)
    feats["name_bigram_dice"] = bigram_dice(ns1.replace(" ", ""), ns2.replace(" ", ""))

    # NEW: Name length ratio (catches abbreviation vs full name)
    feats["name_length_ratio"] = min(len(n1), len(n2)) / max(len(n1), len(n2), 1) if n1 and n2 else 0.0
    feats["name_token_count_ratio"] = min(len(t1), len(t2)) / max(len(t1), len(t2), 1) if t1 and t2 else 0.0

    # NEW: Initialism match (HDFC ↔ housing development finance corporation)
    if t1 and t2:
        init1 = get_initialism(t1)
        init2 = get_initialism(t2)
        n1_comp = n1.replace(" ", "")
        n2_comp = n2.replace(" ", "")
        feats["name_initialism_match"] = 1.0 if (
            init1 == n2_comp or init2 == n1_comp or (len(init1) >= 2 and init1 == init2)
        ) else 0.0
    else:
        feats["name_initialism_match"] = 0.0

    # NEW: Shared distinctive tokens count & ratio
    shared, total = count_shared_distinctive(t1, t2)
    feats["name_shared_distinctive_count"] = float(shared)
    feats["name_shared_distinctive_ratio"] = shared / max(total, 1)

    # NEW: Accent-stripped name match (French entities: réseau → reseau)
    na1 = s1.get("name_no_accent", ns1)
    na2 = cand.get("name_no_accent", ns2)
    feats["name_no_accent_exact"] = 1.0 if na1 and na1 == na2 else 0.0

    # 4. Address features
    a1 = s1["address_normalized"]
    a2 = cand["address_normalized"]

    feats["addr_exact_match"] = 1.0 if a1 and a1 == a2 else 0.0
    feats["addr_levenshtein_ratio"] = fuzz.ratio(a1, a2) / 100.0 if a1 and a2 else 0.0
    feats["addr_token_sort_ratio"] = fuzz.token_sort_ratio(a1, a2) / 100.0 if a1 and a2 else 0.0
    feats["addr_token_set_ratio"] = fuzz.token_set_ratio(a1, a2) / 100.0 if a1 and a2 else 0.0
    feats["addr_partial_ratio"] = fuzz.partial_ratio(a1, a2) / 100.0 if a1 and a2 else 0.0

    at1 = s1.get("address_tokens", [])
    at2 = cand.get("address_tokens", [])
    feats["addr_jaccard"] = compute_jaccard(at1, at2)
    feats["addr_containment"] = compute_containment(at1, at2)

    # NEW: Address Jaro-Winkler
    feats["addr_jaro_winkler"] = JaroWinkler.similarity(a1, a2) if a1 and a2 else 0.0

    # NEW: Address trigram dice
    feats["addr_trigram_dice"] = trigram_dice(a1, a2)

    # Building & Postal Numbers overlap
    b1 = set(s1.get("building_numbers", []))
    b2 = set(cand.get("building_numbers", []))
    feats["shared_building_number"] = 1.0 if (b1 and b2 and (b1 & b2)) else 0.0
    feats["building_number_mismatch"] = 1.0 if (b1 and b2 and not (b1 & b2)) else 0.0

    p1 = set(s1.get("postal_tokens", []))
    p2 = set(cand.get("postal_tokens", []))
    feats["shared_postal_token"] = 1.0 if (p1 and p2 and (p1 & p2)) else 0.0
    feats["postal_token_mismatch"] = 1.0 if (p1 and p2 and not (p1 & p2)) else 0.0

    # 5. Combined Global Text Similarity
    c1 = f"{n1} {a1}".strip()
    c2 = f"{n2} {a2}".strip()
    feats["combined_token_set_ratio"] = fuzz.token_set_ratio(c1, c2) / 100.0 if c1 and c2 else 0.0

    # 6. High-Precision Rule Features
    feats["rule_strong_positive"] = 1.0 if (feats["name_exact_match"] == 1.0 and feats["addr_exact_match"] == 1.0 and feats["country_match"] == 1.0) else 0.0
    feats["rule_strong_conflict"] = 1.0 if (feats["building_number_mismatch"] == 1.0 or feats["postal_token_mismatch"] == 1.0) else 0.0

    return feats


FEATURE_NAMES = [
    "blocking_rank",
    "is_source2",
    "is_source3",
    "country_match",
    "s1_has_empty_addr",
    "cand_has_empty_addr",
    "both_have_address",
    "name_exact_match",
    "name_levenshtein_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_partial_ratio",
    "name_jaccard",
    "name_containment",
    "same_first_token",
    "name_no_suffix_exact",
    "name_no_suffix_ratio",
    "name_no_suffix_sort",
    # NEW features
    "name_jaro_winkler",
    "name_trigram_dice",
    "name_bigram_dice",
    "name_length_ratio",
    "name_token_count_ratio",
    "name_initialism_match",
    "name_shared_distinctive_count",
    "name_shared_distinctive_ratio",
    "name_no_accent_exact",
    # Address features
    "addr_exact_match",
    "addr_levenshtein_ratio",
    "addr_token_sort_ratio",
    "addr_token_set_ratio",
    "addr_partial_ratio",
    "addr_jaccard",
    "addr_containment",
    "addr_jaro_winkler",
    "addr_trigram_dice",
    # Number features
    "shared_building_number",
    "building_number_mismatch",
    "shared_postal_token",
    "postal_token_mismatch",
    # Combined
    "combined_token_set_ratio",
    "rule_strong_positive",
    "rule_strong_conflict"
]
