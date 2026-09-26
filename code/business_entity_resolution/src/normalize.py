"""
Core Normalization Library for Business Entity Resolution.
Supports US, India, and France entities with Unicode NFC preservation.
"""

import re
import sys
import unicodedata
from typing import Dict, List, Set, Tuple, Optional

# Ensure UTF-8 console output on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ---------------------------------------------------------------------------
# Legal suffixes and abbreviations dictionary
# ---------------------------------------------------------------------------
LEGAL_EXPANSIONS = {
    r"\bpvt\b": "private",
    r"\bltd\b": "limited",
    r"\bpte\b": "private",
    r"\bcorp\b": "corporation",
    r"\binc\b": "incorporated",
    r"\bco\b": "company",
    r"\bllp\b": "limited liability partnership",
    r"\bllc\b": "limited liability company",
    r"\bplc\b": "public limited company",
    r"\bsarl\b": "sarl",
    r"\bsas\b": "sas",
    r"\bsasu\b": "sasu",
    r"\bsci\b": "sci",
    r"\beurl\b": "eurl",
    r"\bgmbh\b": "gmbh",
    r"\bag\b": "ag",
}

# Regex pattern for legal suffix removal (end of string or standalone)
LEGAL_SUFFIX_RE = re.compile(
    r"\b(private limited|pvt ltd|private ltd|pvt limited|limited|corporation|incorporated|"
    r"company|corp|inc|llp|llc|co|sarl|sas|sasu|sci|eurl|sa|gmbh|ag|holdings|enterprises)\b",
    re.IGNORECASE
)

# Address abbreviations dictionary (expanded for India & France)
ADDRESS_EXPANSIONS = {
    r"\brd\b": "road",
    r"\bst\b": "street",
    r"\bave\b": "avenue",
    r"\bav\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bbvd\b": "boulevard",
    r"\bhwy\b": "highway",
    r"\bdr\b": "drive",
    r"\bln\b": "lane",
    r"\bct\b": "court",
    r"\bpl\b": "place",
    r"\bsq\b": "square",
    r"\bpkwy\b": "parkway",
    r"\bste\b": "suite",
    r"\bapt\b": "apartment",
    r"\bfl\b": "floor",
    r"\bbldg\b": "building",
    r"\boff\b": "office",
    r"\bdist\b": "district",
    r"\btq\b": "taluk",
    r"\br\b": "rue",
    r"\bbd\b": "boulevard",
    # French-specific
    r"\bres\b": "residence",
    r"\bimm\b": "immeuble",
    r"\bbt\b": "batiment",
    r"\bapp\b": "appartement",
    r"\bcrs\b": "cours",
    r"\bpas\b": "passage",
    r"\bche\b": "chemin",
    # Indian-specific
    r"\bnr\b": "near",
    r"\bopp\b": "opposite",
    r"\bvill\b": "village",
}

NAME_STOPWORDS = {
    "and", "the", "of", "in", "for", "at", "by", "from", "with", "to", "on",
    "services", "solutions", "international", "group", "enterprises", "trading",
    "industries", "consultants", "associates", "technologies", "technology", "global",
    "management", "consulting", "logistics", "retail", "works", "systems",
    "agency", "marketing", "center", "care", "india", "france", "us"
}

PINCODE_INDIA_RE = re.compile(r"\b[1-9][0-9]{5}\b")
ZIPCODE_US_RE = re.compile(r"\b[0-9]{5}(?:-[0-9]{4})?\b")
POSTAL_FRANCE_RE = re.compile(r"\b[0-9]{5}\b")
NUMBER_RE = re.compile(r"\b\d+\b")
URL_CLEAN_RE = re.compile(r"(https?://)?(www\.)?([a-z0-9\-]+)\.(com|in|org|net|fr|co|io|edu)", re.IGNORECASE)
PUNCT_RE = re.compile(r"""[\.,\-_/\\:;!?'"()\[\]{}*+=@#$%^&|~`<>«»""''—–•·]+""")


def normalize_text(text: Optional[str]) -> str:
    """Base normalization: NFC unicode normalization, lowercase, '&' -> 'and', clean punctuation."""
    if not text or str(text).lower() in ("nan", "null", "none"):
        return ""
    
    text = str(text)

    # Strip URL formatting if it looks like a website domain
    m = URL_CLEAN_RE.search(text)
    if m:
        text = URL_CLEAN_RE.sub(r"\3", text)

    # Normalize unicode to canonical composition (preserves Indic matras and accented Latin)
    text = unicodedata.normalize("NFC", text)
    # Convert '&' to 'and'
    text = text.replace("&", " and ")
    # Lowercase
    text = text.lower()
    # Replace punctuation with space, preserving letters, digits, and combining marks across all languages
    text = PUNCT_RE.sub(" ", text)
    # Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_accents(text: str) -> str:
    """Convert accented characters to ASCII equivalents (é→e, ü→u, etc.)."""
    if not text:
        return ""
    nfkd = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in nfkd if not unicodedata.combining(c))


def get_initialism(tokens: list) -> str:
    """Get initialism from first letter of each token: ['housing','development'] -> 'hd'."""
    return "".join(t[0] for t in tokens if t)


def get_char_trigrams(text: str, n: int = 3) -> set:
    """Get character n-gram set from a string (spaces removed)."""
    text = text.replace(" ", "")
    if len(text) < n:
        return set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}


def normalize_business_name(name: Optional[str]) -> Dict[str, any]:
    """Generate multiple normalized views of a business name."""
    norm = normalize_text(name)
    if not norm:
        return {
            "name_normalized": "",
            "name_without_legal_suffix": "",
            "name_sorted_tokens": "",
            "name_tokens": [],
            "name_no_accent": "",
        }

    # Expand common abbreviations
    expanded = norm
    for pattern, repl in LEGAL_EXPANSIONS.items():
        expanded = re.sub(pattern, repl, expanded)
    expanded = re.sub(r"\s+", " ", expanded).strip()

    # Remove legal suffixes for comparison
    no_suffix = LEGAL_SUFFIX_RE.sub(" ", expanded)
    no_suffix = re.sub(r"\s+", " ", no_suffix).strip()
    if not no_suffix:
        no_suffix = expanded

    # Sorted tokens to handle word order permutations
    tokens = [t for t in expanded.split() if t]
    sorted_tokens = " ".join(sorted(tokens))

    # Accent-stripped version (for French entity matching)
    no_accent = remove_accents(no_suffix)

    return {
        "name_normalized": expanded,
        "name_without_legal_suffix": no_suffix,
        "name_sorted_tokens": sorted_tokens,
        "name_tokens": tokens,
        "name_no_accent": no_accent,
    }


def normalize_business_address(address: Optional[str]) -> Dict[str, any]:
    """Generate normalized address, tokens, street numbers, and postal tokens."""
    norm = normalize_text(address)
    if not norm:
        return {
            "address_normalized": "",
            "address_tokens": [],
            "address_numbers": [],
            "building_numbers": [],
            "postal_tokens": []
        }

    # Expand address abbreviations
    expanded = norm
    for pattern, repl in ADDRESS_EXPANSIONS.items():
        expanded = re.sub(pattern, repl, expanded)
    expanded = re.sub(r"\s+", " ", expanded).strip()

    tokens = [t for t in expanded.split() if t]
    numbers = NUMBER_RE.findall(expanded)
    
    # Extract postal tokens (5-digit or 6-digit codes)
    postal_tokens = [n for n in numbers if len(n) in (5, 6)]
    
    # Building / street numbers (usually first 1 or 2 small numbers)
    building_numbers = [n for n in numbers if len(n) <= 4]

    return {
        "address_normalized": expanded,
        "address_tokens": tokens,
        "address_numbers": numbers,
        "building_numbers": building_numbers,
        "postal_tokens": postal_tokens
    }


def normalize_record(record: Dict[str, str]) -> Dict[str, any]:
    """Normalize full entity record containing entity_id, business_name, business_address, country."""
    raw_name = record.get("business_name", "")
    raw_addr = record.get("business_address", "")
    country = record.get("country", "").strip().upper()
    eid = record.get("entity_id", "").strip()

    name_dict = normalize_business_name(raw_name)
    addr_dict = normalize_business_address(raw_addr)

    return {
        "entity_id": eid,
        "country": country,
        "raw_name": raw_name,
        "raw_address": raw_addr,
        "is_name_empty": len(name_dict["name_normalized"]) == 0,
        "is_addr_empty": len(addr_dict["address_normalized"]) == 0,
        **name_dict,
        **addr_dict
    }
