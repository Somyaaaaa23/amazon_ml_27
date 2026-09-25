"""
Stage 1: Normalization module.
Handles:
- P1-3: Rule-based Indic script transliteration to Latin (Devanagari, Tamil, Telugu, etc.)
- P2-1 & P2-2: Exact legal-form stripping (including French forms eurl, sasu, sci, ei, snc)
- P2-3, P2-4, P2-5: House/plot number isolation, address component tokenization, landmark handling, and addr_missing flag.
"""

import re
import unicodedata
from typing import Tuple, Dict, Any, List
import pandas as pd


# -----------------------------------------------------------------------------
# 1. Indic Script Transliteration Table (Brahmic Offset Map)
# -----------------------------------------------------------------------------
BRAHMIC_OFFSET_MAP = {
    0x02: "m",   # Anusvara
    0x03: "h",   # Visarga
    0x05: "a",   0x06: "aa",  0x07: "i",   0x08: "ee",  0x09: "u",   0x0A: "oo",  0x0B: "ri",
    0x0E: "e",   0x0F: "ai",  0x10: "ai",  0x12: "o",   0x13: "au",  0x14: "au",
    0x15: "k",   0x16: "kh",  0x17: "g",   0x18: "gh",  0x19: "ng",
    0x1A: "ch",  0x1B: "chh", 0x1C: "j",   0x1D: "jh",  0x1E: "ny",
    0x1F: "t",   0x20: "th",  0x21: "d",   0x22: "dh",  0x23: "n",
    0x24: "t",   0x25: "th",  0x26: "d",   0x27: "dh",  0x28: "n",   0x29: "nn",
    0x2A: "p",   0x2B: "ph",  0x2C: "b",   0x2D: "bh",  0x2E: "m",
    0x2F: "y",   0x30: "r",   0x31: "rr",  0x32: "l",   0x33: "ll",  0x34: "lll", 0x35: "v",
    0x36: "sh",  0x37: "sh",  0x38: "s",   0x39: "h",
    0x3E: "aa",  0x3F: "i",   0x40: "ee",  0x41: "u",   0x42: "oo",  0x43: "ri",
    0x46: "e",   0x47: "e",   0x48: "ai",  0x4A: "o",   0x4B: "o",   0x4C: "au",
    0x4D: "",    # Virama (vowel suppressor / halant / pulli)
}

def transliterate_indic(text: str) -> str:
    """
    Transliterates Brahmic scripts (Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada, Malayalam)
    to Latin characters using universal unicode block offsets.
    """
    if not text or not isinstance(text, str):
        return ""
    res = []
    for ch in text:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x0D7F:
            base = cp & 0xFF80
            offset = cp - base
            res.append(BRAHMIC_OFFSET_MAP.get(offset, " "))
        else:
            res.append(ch)
    return "".join(res)


def strip_accents(text: str) -> str:
    if not text:
        return ""
    text_nfkd = unicodedata.normalize("NFKD", str(text))
    return "".join([c for c in text_nfkd if not unicodedata.combining(c)])


# -----------------------------------------------------------------------------
# 2. Refined Legal Suffixes (P2-1, P2-2)
# Strictly legal forms; descriptive words like services/solutions/group/enterprises are preserved in core name!
# -----------------------------------------------------------------------------
TRUE_LEGAL_SUFFIXES = [
    # US / UK / India forms
    r"\bprivate limited\b", r"\bpvt ltd\b", r"\bpvt\b", r"\bltd\b", r"\blimited\b",
    r"\bcorporation\b", r"\bcorp\b", r"\bincorporated\b", r"\binc\b",
    r"\bllc\b", r"\bllp\b", r"\bl\.l\.c\b", r"\bl\.l\.p\b",
    r"\bplc\b", r"\bgmbh\b", r"\bco\b", r"\bcompany\b",
    # French forms
    r"\bsarl\b", r"\bs\.a\.r\.l\b", r"\bsas\b", r"\bs\.a\.s\b", r"\bsasu\b", r"\bs\.a\.s\.u\b",
    r"\bsa\b", r"\bs\.a\b", r"\beurl\b", r"\be\.u\.r\.l\b", r"\bsci\b", r"\bs\.c\.i\b",
    r"\bsnc\b", r"\bs\.n\.c\b", r"\bselarl\b", r"\bs\.e\.l\.a\.r\.l\b",
    r"\bsociete anonyme\b", r"\bsociete par actions simplifiee\b", r"\bentreprise individuelle\b",
    r"\bei\b",
]
LEGAL_SUFFIX_REGEX = re.compile(r"\b(" + "|".join([p.strip(r"\b") for p in TRUE_LEGAL_SUFFIXES]) + r")\b", flags=re.IGNORECASE)

# Phonetic expansions for transliterated & common abbreviations
NAME_ABBREVIATIONS = {
    r"\bpraivet\b": "private",
    r"\bpvt\b": "private",
    r"\bltd\b": "limited",
    r"\bcorp\b": "corporation",
    r"\binc\b": "incorporated",
    r"\beleipi\b": "llp",
    r"\belepi\b": "llp",
    r"\beleli\b": "llc",
    r"\bintl\b": "international",
    r"\bmfg\b": "manufacturing",
    r"\btech\b": "technologies",
}

ADDRESS_ABBREVIATIONS = {
    r"\brd\b": "road",
    r"\bst\b": "street",
    r"\bave\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bln\b": "lane",
    r"\bdr\b": "drive",
    r"\bct\b": "court",
    r"\bpkwy\b": "parkway",
    r"\bhwy\b": "highway",
    r"\bste\b": "suite",
    r"\bflr\b": "floor",
    r"\bindl\b": "industrial",
    r"\bph\b": "phase",
    r"\bpl\b": "place",
    r"\br\b": "rue",
    r"\bbd\b": "boulevard",
    r"\bav\b": "avenue",
    r"\bopp\b": "opposite",
    r"\bnr\b": "near",
    r"\badj\b": "adjacent",
}

# Regex for landmark extraction
LANDMARK_PATTERN = re.compile(
    r"\b(near|opp|opposite|behind|beside|adjacent to|adj|next to|in front of)\s+([^,]+)",
    flags=re.IGNORECASE
)

# House / Plot / Leading Number pattern (P2-3)
HOUSE_NUMBER_PATTERN = re.compile(r"^\s*([0-9]+[a-zA-Z]?(?:[\/-][0-9]+[a-zA-Z]?)?)\b|\bplot\s*(?:no\.?|#)?\s*([0-9]+[a-zA-Z]?)\b", flags=re.IGNORECASE)


def normalize_name(raw_name: str) -> Tuple[str, str]:
    """
    Returns (name_norm, name_core)
    - name_norm: fully cleaned name with transliteration & abbreviation expansion
    - name_core: true legal-form-stripped variant preserving descriptive terms
    """
    if pd.isna(raw_name) or not raw_name:
        return "", ""

    # Transliterate Indic scripts first
    text = transliterate_indic(str(raw_name))
    text = strip_accents(text).lower()

    # Normalize ampersands & special characters
    text = re.sub(r"\s*&\s*", " and ", text)
    text = re.sub(r"[\/\+\@]", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)

    # Expand abbreviations
    for pattern, replacement in NAME_ABBREVIATIONS.items():
        text = re.sub(pattern, replacement, text)

    text = re.sub(r"\s+", " ", text).strip()

    # Remove legal suffixes
    core = LEGAL_SUFFIX_REGEX.sub(" ", text)
    core = re.sub(r"\s+", " ", core).strip()

    return text, core


def normalize_address(raw_addr: str) -> Tuple[str, str, str, bool]:
    """
    Returns (addr_norm, landmark, house_number, addr_missing)
    """
    if pd.isna(raw_addr) or not str(raw_addr).strip() or str(raw_addr).strip().lower() in ["nan", "null", "none"]:
        return "", "", "", True

    # Transliterate Indic scripts
    text = transliterate_indic(str(raw_addr))
    text = strip_accents(text).lower()

    # Extract landmark
    landmark = ""
    lm_match = LANDMARK_PATTERN.search(text)
    if lm_match:
        landmark = lm_match.group(0).strip()

    # Extract house / plot number
    house_num = ""
    hn_match = HOUSE_NUMBER_PATTERN.search(text)
    if hn_match:
        house_num = hn_match.group(1) or hn_match.group(2) or ""

    # Standardize abbreviations
    text = re.sub(r"\s*&\s*", " and ", text)
    text = re.sub(r"[^\w\s,]", " ", text)  # Preserve commas for component parsing

    for pattern, replacement in ADDRESS_ABBREVIATIONS.items():
        text = re.sub(pattern, replacement, text)

    text = re.sub(r"\s+", " ", text).strip()

    return text, landmark, house_num, False


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized preprocessing for a source DataFrame.
    Adds:
    - name_norm, name_core
    - addr_norm, landmark, house_number, addr_missing
    - combined_text
    """
    df = df.copy()

    for col in ["entity_id", "business_name", "business_address", "country"]:
        if col not in df.columns:
            df[col] = ""

    df["country_clean"] = df["country"].fillna("").astype(str).str.strip().str.lower()

    name_tuples = [normalize_name(x) for x in df["business_name"]]
    df["name_norm"] = [t[0] for t in name_tuples]
    df["name_core"] = [t[1] for t in name_tuples]

    addr_tuples = [normalize_address(x) for x in df["business_address"]]
    df["addr_norm"] = [t[0] for t in addr_tuples]
    df["landmark"] = [t[1] for t in addr_tuples]
    df["house_number"] = [t[2] for t in addr_tuples]
    df["addr_missing"] = [1.0 if t[3] else 0.0 for t in addr_tuples]

    df["combined_text"] = df["name_norm"] + " " + df["addr_norm"]

    return df
