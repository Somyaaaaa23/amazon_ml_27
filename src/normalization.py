"""
Stage 1: Normalization module.
Builds cleaned, standardized views for business names and addresses.
Works generically across multi-lingual inputs (US, India, France).
"""

import re
import unicodedata
from typing import Tuple, Dict, Any
import pandas as pd


# Regex for Unicode NFKD accent stripping
def strip_accents(text: str) -> str:
    if not text:
        return ""
    text_nfkd = unicodedata.normalize("NFKD", str(text))
    return "".join([c for c in text_nfkd if not unicodedata.combining(c)])


# Common legal suffixes to isolate core business names
LEGAL_SUFFIXES = [
    r"\bprivate limited\b", r"\bpvt ltd\b", r"\bpvt\b", r"\bltd\b", r"\blimited\b",
    r"\bcorporation\b", r"\bcorp\b", r"\bincorporated\b", r"\binc\b",
    r"\bcompany\b", r"\bco\b", r"\bllc\b", r"\bllp\b", r"\bl\.l\.c\b",
    r"\bsarl\b", r"\bs\.a\.r\.l\b", r"\bsas\b", r"\bs\.a\.s\b", r"\bsa\b", r"\bs\.a\b",
    r"\bgmbh\b", r"\bplc\b", r"\bholding\b", r"\bholdings\b", r"\bgroup\b",
    r"\benterprises\b", r"\bent\b", r"\bservices\b", r"\bsolutions\b"
]
LEGAL_SUFFIX_REGEX = re.compile("|".join(LEGAL_SUFFIXES), flags=re.IGNORECASE)

# Common name abbreviation expansions
NAME_ABBREVIATIONS = {
    r"\bpvt\b": "private",
    r"\bltd\b": "limited",
    r"\bcorp\b": "corporation",
    r"\binc\b": "incorporated",
    r"\bco\b": "company",
    r"\bent\b": "enterprises",
    r"\bsoln\b": "solutions",
    r"\bintl\b": "international",
    r"\bassoc\b": "associates",
    r"\bmfg\b": "manufacturing",
    r"\btech\b": "technologies",
}

# Address abbreviation mappings
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
    r"\bfl\b": "floor",
    r"\bindl\b": "industrial",
    r"\bind\b": "industrial",
    r"\bph\b": "phase",
    r"\bpl\b": "place",
    r"\br\b": "rue",
    r"\bopp\b": "opposite",
    r"\bnr\b": "near",
    r"\badj\b": "adjacent",
}

# Landmark extraction regex
LANDMARK_PATTERN = re.compile(
    r"\b(near|opp|opposite|behind|beside|adjacent to|adj|next to|in front of)\s+([^,]+)",
    flags=re.IGNORECASE
)

# Postal code extractor (5-digit US, 6-digit India, 5-digit France)
POSTAL_CODE_PATTERN = re.compile(r"\b(\d{5,6})\b")


def normalize_name(raw_name: str) -> Tuple[str, str]:
    """
    Returns (name_norm, name_core)
    - name_norm: fully cleaned and normalized name with expanded abbreviations
    - name_core: legal-suffix-stripped core name
    """
    if pd.isna(raw_name) or not raw_name:
        return "", ""

    text = str(raw_name).strip()
    text = strip_accents(text).lower()

    # Replace ampersands
    text = re.sub(r"\s*&\s*", " and ", text)
    text = re.sub(r"[\/\+\@]", " ", text)

    # Remove non-alphanumeric except whitespace
    text = re.sub(r"[^\w\s]", " ", text)

    # Expand abbreviations
    for pattern, replacement in NAME_ABBREVIATIONS.items():
        text = re.sub(pattern, replacement, text)

    text = re.sub(r"\s+", " ", text).strip()

    # Create core name by removing legal suffixes
    core = LEGAL_SUFFIX_REGEX.sub(" ", text)
    core = re.sub(r"\s+", " ", core).strip()

    return text, core


def normalize_address(raw_addr: str) -> Tuple[str, str, str]:
    """
    Returns (addr_norm, landmark, postal_code)
    - addr_norm: standardized address without noise
    - landmark: extracted landmark phrase if present (empty string if none)
    - postal_code: 5/6 digit postal code if found
    """
    if pd.isna(raw_addr) or not raw_addr:
        return "", "", ""

    text = str(raw_addr).strip()
    text = strip_accents(text).lower()

    # Extract landmark
    landmark = ""
    lm_match = LANDMARK_PATTERN.search(text)
    if lm_match:
        landmark = lm_match.group(0).strip()
        # Remove landmark from core address view
        text = LANDMARK_PATTERN.sub(" ", text)

    # Extract postal code
    postal_code = ""
    pc_match = POSTAL_CODE_PATTERN.search(text)
    if pc_match:
        postal_code = pc_match.group(1).strip()

    # Replace ampersands & punctuation
    text = re.sub(r"\s*&\s*", " and ", text)
    text = re.sub(r"[^\w\s]", " ", text)

    # Standardize address tokens
    for pattern, replacement in ADDRESS_ABBREVIATIONS.items():
        text = re.sub(pattern, replacement, text)

    text = re.sub(r"\s+", " ", text).strip()

    return text, landmark, postal_code


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies normalization pipeline to a source DataFrame.
    Adds columns:
    - name_norm, name_core
    - addr_norm, landmark, postal_code
    - combined_text (name_norm + ' ' + addr_norm)
    """
    df = df.copy()

    # Ensure required columns exist
    for col in ["entity_id", "business_name", "business_address", "country"]:
        if col not in df.columns:
            df[col] = ""

    # Clean country
    df["country_clean"] = df["country"].fillna("").astype(str).str.strip().str.lower()

    name_tuples = [normalize_name(x) for x in df["business_name"]]
    df["name_norm"] = [t[0] for t in name_tuples]
    df["name_core"] = [t[1] for t in name_tuples]

    addr_tuples = [normalize_address(x) for x in df["business_address"]]
    df["addr_norm"] = [t[0] for t in addr_tuples]
    df["landmark"] = [t[1] for t in addr_tuples]
    df["postal_code"] = [t[2] for t in addr_tuples]

    df["combined_text"] = df["name_norm"] + " " + df["addr_norm"]

    return df
