"""Field normalization for business names and addresses.

Design notes (grounded in reports/data_analysis_report.md):

* Source 1 names are 100% ASCII; ~11-15% of Source 2/3 names are not.
  Of the non-ASCII names, the single largest group (~48%) is plain Latin
  text corrupted with accent marks (e.g. "Énterprises", "Bóral") -- not a
  script change at all. Unicode accent-folding (NFKD + strip combining
  marks) recovers these exactly and is applied universally.
* The second largest group (~35-44%) is Devanagari script (Hindi/Marathi).
  A hand-built phonetic transliteration table is applied so that common
  English loanwords transliterated into Devanagari (e.g. "प्राइवेट लिमिटेड"
  for "Private Limited") recover partial character/token overlap with the
  Latin spelling on the Source-1 side.
* The remaining Indic scripts (Tamil, Telugu, Kannada, Gujarati, Bengali,
  Malayalam, Gurmukhi, Oriya) are each individually a small minority and
  building reliable phonetic tables for all of them from memory risks more
  harm (wrong mappings) than benefit. Verified empirically that addresses
  stay in Latin script even when the name doesn't (~82-85% of the time),
  so blocking/matching for these cases leans on address + country + a
  coarse script-family feature rather than transliteration.
"""
from __future__ import annotations

import re
import unicodedata

import pandas as pd

# ---------------------------------------------------------------------------
# Devanagari -> Latin phonetic transliteration (approximate, rule-based)
# ---------------------------------------------------------------------------

_DEV_VOWELS_INDEP = {
    0x0904: "a", 0x0905: "a", 0x0906: "aa", 0x0907: "i", 0x0908: "ii",
    0x0909: "u", 0x090A: "uu", 0x090B: "ri", 0x090C: "lri", 0x090D: "ae",
    0x090E: "e", 0x090F: "e", 0x0910: "ai", 0x0911: "o", 0x0912: "o",
    0x0913: "o", 0x0914: "au", 0x0960: "rri", 0x0961: "lli",
    0x0962: "l", 0x0963: "l",
}

_DEV_MATRAS = {
    0x093E: "aa", 0x093F: "i", 0x0940: "ii", 0x0941: "u", 0x0942: "uu",
    0x0943: "ri", 0x0944: "rri", 0x0945: "ae", 0x0946: "e", 0x0947: "e",
    0x0948: "ai", 0x0949: "o", 0x094A: "o", 0x094B: "o", 0x094C: "au",
}

_DEV_CONSONANTS = {
    0x0915: "k", 0x0916: "kh", 0x0917: "g", 0x0918: "gh", 0x0919: "ng",
    0x091A: "ch", 0x091B: "chh", 0x091C: "j", 0x091D: "jh", 0x091E: "ny",
    0x091F: "t", 0x0920: "th", 0x0921: "d", 0x0922: "dh", 0x0923: "n",
    0x0924: "t", 0x0925: "th", 0x0926: "d", 0x0927: "dh", 0x0928: "n",
    0x092A: "p", 0x092B: "ph", 0x092C: "b", 0x092D: "bh", 0x092E: "m",
    0x092F: "y", 0x0930: "r", 0x0931: "r", 0x0932: "l", 0x0933: "l",
    0x0934: "l", 0x0935: "v", 0x0936: "sh", 0x0937: "sh", 0x0938: "s",
    0x0939: "h",
    0x0958: "q", 0x0959: "kh", 0x095A: "g", 0x095B: "z", 0x095C: "r",
    0x095D: "rh", 0x095E: "f", 0x095F: "y",
}

_DEV_DIGITS = {0x0966 + i: str(i) for i in range(10)}

_DEV_MISC = {
    0x0901: "n", 0x0902: "n", 0x0903: "h",
    0x093D: "", 0x0964: " ", 0x0965: " ", 0x0970: ".", 0x0900: "",
}

_DEV_RANGE_RE = re.compile(r"[ऀ-ॿ]")


def _devanagari_to_latin_one(text: str) -> str:
    out: list[str] = []
    pending_a = False  # True if the last emitted item is an un-resolved inherent 'a'
    for ch in text:
        o = ord(ch)
        if o in _DEV_CONSONANTS:
            out.append(_DEV_CONSONANTS[o])
            out.append("a")
            pending_a = True
        elif o in _DEV_MATRAS:
            if pending_a:
                out.pop()
            out.append(_DEV_MATRAS[o])
            pending_a = False
        elif o == 0x094D:  # virama: suppress the inherent vowel just emitted
            if pending_a:
                out.pop()
            pending_a = False
        elif o in _DEV_VOWELS_INDEP:
            out.append(_DEV_VOWELS_INDEP[o])
            pending_a = False
        elif o in _DEV_DIGITS:
            out.append(_DEV_DIGITS[o])
            pending_a = False
        elif o in _DEV_MISC:
            if _DEV_MISC[o]:
                out.append(_DEV_MISC[o])
            pending_a = False
        else:
            out.append(ch)
            pending_a = False
    return "".join(out)


def devanagari_to_latin_series(s: pd.Series) -> pd.Series:
    """Transliterate Devanagari substrings to Latin; passthrough everything else.

    Only rows that actually contain a Devanagari codepoint pay the per-character
    cost; the rest (the vast majority) are returned untouched.
    """
    s = s.astype(str)
    mask = s.str.contains(_DEV_RANGE_RE, regex=True, na=False)
    if not mask.any():
        return s
    out = s.copy()
    out.loc[mask] = s.loc[mask].map(_devanagari_to_latin_one)
    return out


# ---------------------------------------------------------------------------
# Script-family detection (coarse, for feature use only)
# ---------------------------------------------------------------------------

_SCRIPT_BLOCKS = [
    ("devanagari", 0x0900, 0x097F),
    ("bengali", 0x0980, 0x09FF),
    ("gurmukhi", 0x0A00, 0x0A7F),
    ("gujarati", 0x0A80, 0x0AFF),
    ("oriya", 0x0B00, 0x0B7F),
    ("tamil", 0x0B80, 0x0BFF),
    ("telugu", 0x0C00, 0x0C7F),
    ("kannada", 0x0C80, 0x0CFF),
    ("malayalam", 0x0D00, 0x0D7F),
    ("latin_accent", 0x00C0, 0x024F),
]
_NONASCII_RE = re.compile(r"[^\x00-\x7F]")


def _dominant_script_one(text: str) -> str:
    if not _NONASCII_RE.search(text):
        return "latin"
    counts: dict[str, int] = {}
    for ch in text:
        o = ord(ch)
        if o < 128:
            continue
        for name, lo, hi in _SCRIPT_BLOCKS:
            if lo <= o <= hi:
                counts[name] = counts.get(name, 0) + 1
                break
        else:
            counts["other"] = counts.get("other", 0) + 1
    if not counts:
        return "latin"
    return max(counts, key=counts.get)


def dominant_script_series(s: pd.Series) -> pd.Series:
    s = s.astype(str)
    mask = s.str.contains(_NONASCII_RE, regex=True, na=False)
    out = pd.Series("latin", index=s.index, dtype="object")
    if mask.any():
        out.loc[mask] = s.loc[mask].map(_dominant_script_one)
    return out


def strip_accents_series(s: pd.Series) -> pd.Series:
    """Fold accented Latin characters to their base form (NFKD, drop combining marks)."""
    s = s.astype(str)
    mask = s.str.contains(_NONASCII_RE, regex=True, na=False)
    if not mask.any():
        return s
    out = s.copy()

    def _strip(text: str) -> str:
        decomposed = unicodedata.normalize("NFKD", text)
        return "".join(c for c in decomposed if not unicodedata.combining(c))

    out.loc[mask] = s.loc[mask].map(_strip)
    return out


# ---------------------------------------------------------------------------
# Abbreviation canonicalization (long form -> short form; contraction only,
# never expansion, to avoid ambiguous guesses like "co" -> "company").
# ---------------------------------------------------------------------------

NAME_LEGAL_SUFFIX_MAP = {
    "incorporated": "inc",
    "corporation": "corp",
    "limited": "ltd",
    "private": "pvt",
    "company": "co",
    "and": "and",  # keep; '&' is mapped to 'and' separately
}

ADDRESS_ABBREV_MAP = {
    "road": "rd",
    "street": "st",
    "avenue": "ave",
    "drive": "dr",
    "boulevard": "blvd",
    "lane": "ln",
    "court": "ct",
    "place": "pl",
    "square": "sq",
    "highway": "hwy",
    "apartment": "apt",
    "apartments": "apt",
    "building": "bldg",
    "floor": "fl",
    "suite": "ste",
    "north": "n",
    "south": "s",
    "east": "e",
    "west": "w",
    "post": "po",
}

LEGAL_SUFFIX_TOKENS = {
    "inc", "incorporated", "llc", "llp", "lp", "ltd", "limited", "corp",
    "corporation", "co", "company", "pvt", "private", "plc", "pllc",
}

# \w is Unicode-aware in Python 3 (matches Tamil/Telugu/Kannada/... letters
# too, not just ASCII) -- important because Devanagari is transliterated to
# ASCII earlier, but other Indic scripts are intentionally left as-is (see
# module docstring) and must not be stripped as if they were punctuation.
_PUNCT_RE = re.compile(r"[^\w\s]+", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")
_AMPERSAND_RE = re.compile(r"&")
_LEADING_DIGITS_RE = re.compile(r"^\s*(\d+[a-z]?)\b")
_POSTAL5_RE = re.compile(r"(?<!\d)(\d{5})(?:-\d{4})?(?!\d)")
_POSTAL6_RE = re.compile(r"(?<!\d)(\d{6})(?!\d)")


def _build_word_map_regex(mapping: dict) -> re.Pattern:
    keys = sorted(mapping.keys(), key=len, reverse=True)
    pattern = r"\b(" + "|".join(re.escape(k) for k in keys) + r")\b"
    return re.compile(pattern)


_NAME_SUFFIX_RE = _build_word_map_regex(NAME_LEGAL_SUFFIX_MAP)
_ADDR_ABBR_RE = _build_word_map_regex(ADDRESS_ABBREV_MAP)


def _apply_word_map(s: pd.Series, regex: re.Pattern, mapping: dict) -> pd.Series:
    return s.map(lambda text: regex.sub(lambda m: mapping[m.group(0)], text) if text else text)


def basic_clean_series(s: pd.Series) -> pd.Series:
    """NaN-safe -> str, Devanagari transliteration, accent fold, lowercase."""
    s = s.fillna("").astype(str)
    s = devanagari_to_latin_series(s)
    s = strip_accents_series(s)
    s = s.str.lower()
    return s


def normalize_name(raw: pd.Series) -> dict[str, pd.Series]:
    """Return the multiple name representations described in the design doc."""
    original = raw.fillna("").astype(str)
    script = dominant_script_series(original)

    cleaned = basic_clean_series(raw)
    cleaned = cleaned.str.replace(_AMPERSAND_RE, " and ", regex=True)
    cleaned = cleaned.str.replace(_PUNCT_RE, " ", regex=True)
    cleaned = cleaned.str.replace(_WS_RE, " ", regex=True).str.strip()
    cleaned = _apply_word_map(cleaned, _NAME_SUFFIX_RE, NAME_LEGAL_SUFFIX_MAP)
    cleaned = cleaned.str.replace(_WS_RE, " ", regex=True).str.strip()

    tokens = cleaned.str.split()
    compact = cleaned.str.replace(" ", "", regex=False)
    sorted_tokens = tokens.map(lambda t: " ".join(sorted(t)))
    core_tokens = tokens.map(lambda t: [w for w in t if w not in LEGAL_SUFFIX_TOKENS])
    core = core_tokens.map(lambda t: " ".join(t))

    return {
        "name_original": original,
        "name_script": script,
        "name_normalized": cleaned,
        "name_compact": compact,
        "name_tokens": tokens,
        "name_sorted_tokens": sorted_tokens,
        "name_core": core,
    }


def normalize_address(raw: pd.Series) -> dict[str, pd.Series]:
    original = raw.fillna("").astype(str)

    postal5 = original.str.extract(_POSTAL5_RE, expand=False)
    postal6 = original.str.extract(_POSTAL6_RE, expand=False)
    postal_code = postal6.where(postal6.notna(), postal5)
    house_number = original.str.extract(_LEADING_DIGITS_RE, expand=False).str.lower()

    cleaned = basic_clean_series(raw)
    cleaned = cleaned.str.replace(_AMPERSAND_RE, " and ", regex=True)
    cleaned = cleaned.str.replace(_PUNCT_RE, " ", regex=True)
    cleaned = cleaned.str.replace(_WS_RE, " ", regex=True).str.strip()
    cleaned = _apply_word_map(cleaned, _ADDR_ABBR_RE, ADDRESS_ABBREV_MAP)
    cleaned = cleaned.str.replace(_WS_RE, " ", regex=True).str.strip()

    tokens = cleaned.str.split()
    compact = cleaned.str.replace(" ", "", regex=False)
    sorted_tokens = tokens.map(lambda t: " ".join(sorted(t)))

    return {
        "address_original": original,
        "address_normalized": cleaned,
        "address_compact": compact,
        "address_tokens": tokens,
        "address_sorted_tokens": sorted_tokens,
        "postal_code": postal_code,
        "house_number": house_number,
    }
