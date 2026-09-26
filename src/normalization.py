"""
Stage 1: normalization of names and addresses (all countries, no country-specific branches).
- Indic scripts (Devanagari, Bengali, Gujarati, Tamil, Telugu, Kannada, Malayalam, ...) are
  transliterated to Latin with our own rule table.
- Accents stripped, apostrophes joined, website parts / junk ids / leading zeros removed.
- name_core: name without true legal forms (incl. French SARL/SAS/EURL/SASU/SCI/EI);
  name_compact: name_core without spaces.
- Addresses: abbreviations expanded, house/plot number extracted from any comma component,
  commas kept for component features. (A landmark phrase is also extracted but is not used.)
"""

import re
import unicodedata
from typing import Tuple, Dict, Any, List
import pandas as pd


# -----------------------------------------------------------------------------
# 1. Indic Script Transliteration Table (Brahmic Offset Map)
# -----------------------------------------------------------------------------
# The nine Brahmic Unicode blocks (Devanagari 0x0900 ... Malayalam 0x0D00, 0x80 each) share one
# layout: the same offset is the same sound. Values target the English spellings used in Source 1
# (long vowels folded to short ones, फ -> f), since Indic-script records are mostly transliterated
# English business names.
_INDEP_VOWELS = {0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u", 0x0B: "ri", 0x0C: "li",
                 0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o", 0x13: "o", 0x14: "au"}
_CONSONANTS = {0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "n",
               0x1A: "ch", 0x1B: "chh", 0x1C: "j", 0x1D: "jh", 0x1E: "n",
               0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n",
               0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n",
               0x2A: "p", 0x2B: "f", 0x2C: "b", 0x2D: "bh", 0x2E: "m",
               0x2F: "y", 0x30: "r", 0x31: "r", 0x32: "l", 0x33: "l", 0x34: "l", 0x35: "v",
               0x36: "sh", 0x37: "sh", 0x38: "s", 0x39: "h",
               0x58: "q", 0x59: "kh", 0x5A: "g", 0x5B: "z", 0x5C: "d", 0x5D: "rh", 0x5E: "f", 0x5F: "y"}
_VOWEL_SIGNS = {0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u", 0x43: "ri", 0x44: "ri",
                0x45: "e", 0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o", 0x4C: "au",
                0x62: "li", 0x63: "li"}
_OTHER = {0x01: "n", 0x02: "n", 0x03: "h",           # candrabindu, anusvara, visarga
          0x7A: "n", 0x7B: "n", 0x7C: "r", 0x7D: "l", 0x7E: "l", 0x7F: "k",  # Malayalam chillu letters
          0x4E: "r"}
_VIRAMA, _NUKTA = 0x4D, 0x3C
_SILENT = {0x55, 0x56, 0x57, 0x70, 0x71}  # length marks etc.
_ZERO_WIDTH = {"‌", "‍"}

# Transliterated spellings of common business words -> the English word used in Source 1
_TRANSLIT_WORDS = {
    "praivet": "private", "praivat": "private", "piraivet": "private", "prayvet": "private", "praivetu": "private",
    "limitet": "limited", "limitedu": "limited", "limitad": "limited", "limittad": "limited", "limittadu": "limited",
    "elelpi": "llp", "elelsi": "llc",
}
_PRA_LI = re.compile(r"\bpra\.?\s*li\b\.?")


def _is_brahmic(ch: str) -> bool:
    return 0x0900 <= ord(ch) <= 0x0D7F


def transliterate_indic(text: str) -> str:
    """Rule-based transliteration of Brahmic scripts to Latin (our own table, no external data).
    Adds the inherent vowel 'a' after a consonant unless a vowel sign / virama follows or the
    consonant ends the word (word-final schwa deletion)."""
    if not text or not isinstance(text, str):
        return ""
    if not any(_is_brahmic(c) for c in text):
        return text
    chars = [c for c in text if c not in _ZERO_WIDTH]
    out = []
    n = len(chars)
    for i, ch in enumerate(chars):
        if not _is_brahmic(ch):
            out.append(ch)
            continue
        off = ord(ch) & 0x7F
        if off in _CONSONANTS:
            # Malayalam റ്റ (rra + virama + rra) is "tt"
            if ord(ch) == 0x0D31 and i + 2 < n and ord(chars[i + 1]) == 0x0D4D and ord(chars[i + 2]) == 0x0D31:
                out.append("t")
                continue
            if ord(ch) == 0x0D31 and i >= 2 and ord(chars[i - 1]) == 0x0D4D and ord(chars[i - 2]) == 0x0D31:
                out.append("t")
            else:
                out.append(_CONSONANTS[off])
            j = i + 1
            while j < n and _is_brahmic(chars[j]) and (ord(chars[j]) & 0x7F) == _NUKTA:
                j += 1
            nxt = (ord(chars[j]) & 0x7F) if j < n and _is_brahmic(chars[j]) else None
            if nxt is None or nxt in _SILENT:
                continue  # word-final consonant: no inherent vowel
            if nxt in _VOWEL_SIGNS or nxt == _VIRAMA:
                continue
            out.append("a")
        elif off in _INDEP_VOWELS:
            out.append(_INDEP_VOWELS[off])
        elif off in _VOWEL_SIGNS:
            out.append(_VOWEL_SIGNS[off])
        elif off in _OTHER:
            out.append(_OTHER[off])
        elif off in (_VIRAMA, _NUKTA) or off in _SILENT:
            continue
        elif 0x66 <= off <= 0x6F:
            out.append(str(off - 0x66))
        else:
            out.append(" ")
    res = "".join(out)
    res = _PRA_LI.sub("private limited", res)
    return " ".join(_TRANSLIT_WORDS.get(w, w) for w in res.split(" "))


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
    r"\btrl\b": "trail",
    r"\bcir\b": "circle",
    r"\bterr?\b": "terrace",
    r"\bpky\b": "parkway",
    r"\bsq\b": "square",
    r"\bhts\b": "heights",
    r"\bcres\b": "crescent",
    r"\bxing\b": "crossing",
    r"\bexpy\b": "expressway",
    r"\bfwy\b": "freeway",
    r"\bapt\b": "apartment",
    r"\bbldg\b": "building",
    r"\bche\b": "chemin",
    r"\bchem\b": "chemin",
    r"\bimp\b": "impasse",
    r"\brte\b": "route",
    r"\bfg\b": "faubourg",
}

# Zero-padded numbers ("0031" vs "31", "Building No. 0253") are noise, not a different number
_LEADING_ZEROS = re.compile(r"(?<!\d)0+(?=\d)")
# Website-style names ("highlandintelligence.com", "www.gilddova.com") and junk ids in names
_URL_PARTS = re.compile(r"https?://|\bwww\.|\.(?:co\.in|com|net|org|in|fr|biz|info|io|us)\b")
_NAME_JUNK = re.compile(r"\bid\s*[:#]?\s*\d+|\b\d{6,}\b|\bd\s*/\s*b\s*/\s*a\b|\bdba\b")

# Regex for landmark extraction
LANDMARK_PATTERN = re.compile(
    r"\b(near|opp|opposite|behind|beside|adjacent to|adj|next to|in front of)\s+([^,]+)",
    flags=re.IGNORECASE
)

# House / plot / street number. Addresses are often reordered, so the number is looked for in
# every comma component, not only at the start. Priority: explicit markers ("plot no 12",
# "door no 183", "#51"), then the first component that starts with a number.
_NUM = r"(\d+[a-z]?(?:\s*[/-]\s*\d+[a-z]?)*)"
_HN_EXPLICIT = re.compile(r"\b(?:plot|door|house|flat|shop|h|d|kh|khasra|survey|sy|s)\s*(?:no|n|number|#)?[\s\.:#-]*" + _NUM, re.I)
_HN_NO = re.compile(r"(?:\bno|#|\bn[o\u00b0\u00ba])[\s\.:#-]*" + _NUM, re.I)
_HN_COMPONENT = re.compile(r"^\s*([a-z]{0,2}[-/]?\d+[a-z]?(?:\s*[/-]\s*\d+[a-z]?)*)(?:\s+\d+/\d+)?(?:\s+(?:bis|ter))?(?:\s+[a-z]|\s*$)", re.I)


def extract_house_number(text: str) -> str:
    for pat in (_HN_EXPLICIT, _HN_NO):
        m = pat.search(text)
        if m:
            return re.sub(r"\s+", "", m.group(1))
    for comp in text.split(","):
        m = _HN_COMPONENT.match(comp)
        if m:
            return re.sub(r"\s+", "", m.group(1))
    return ""


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

    # Apostrophes join rather than split ("Orelee's" -> "orelees")
    text = re.sub(r"['\u2019`]", "", text)
    text = _URL_PARTS.sub(" ", text)
    text = _NAME_JUNK.sub(" ", text)
    text = _LEADING_ZEROS.sub("", text)

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

    text = _LEADING_ZEROS.sub("", text)

    # Extract house / plot number
    house_num = extract_house_number(text)

    # Standardize abbreviations
    text = re.sub(r"['\u2019`]", "", text)
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
    df["name_compact"] = df["name_core"].str.replace(" ", "", regex=False)

    addr_tuples = [normalize_address(x) for x in df["business_address"]]
    df["addr_norm"] = [t[0] for t in addr_tuples]
    df["landmark"] = [t[1] for t in addr_tuples]
    df["house_number"] = [t[2] for t in addr_tuples]
    df["addr_missing"] = [1.0 if t[3] else 0.0 for t in addr_tuples]

    df["combined_text"] = df["name_norm"] + " " + df["addr_norm"]

    return df
