"""Address structure features: house number and street name compared SEPARATELY.

Motivation (measured on the test predictions): the first-stage matcher accepts ~2% of its French matches with the same
business name and the same house number but a different street ("55 R. Chanoine Larose" vs "55 Boulevard de la
Chauviniere"). Whole-address token overlap stays high for such pairs (number, city, region and "rue/de/la" all match),
so the matcher cannot see that the street -- the part that identifies the place -- differs. Parsing the address into
"<house number> <street>" + the remaining components (locality) makes that contrast explicit:

  as_hn_both        both addresses carry a house number
  as_hn_eq          1 same number, 0 different number (NaN when a side has none); '0042' == '42', '(6)' == '6'
  as_street_sim     token-set similarity of the street names (street-type words / articles removed), NaN if missing
  as_street_jw      Jaro-Winkler similarity of the same strings (typo-tolerant)
  as_num_eq_street_diff  same number but clearly different street (as_street_sim < 0.5)
  as_rest_sim       token-set similarity of the remaining components (city / region / landmarks)

Parsing works on the ORIGINAL address text (the normalized text has lost its commas). It is country-agnostic: the
component holding the house number is found by pattern, whatever its position (component reordering is a known noise).
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pyarrow as pa
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

# street-type words and articles of the three countries' address styles (dropped before comparing street names)
_DROP = set("""rue r av ave avenue bd blvd boul boulevard allee all impasse imp chemin ch che route rte rt place pl cours crs
quai qu square sq passage pass residence res hameau lieu dit faubourg fbg parvis rond point esplanade
street st str road rd drive dr lane ln court ct way circle cir trail trl parkway pkwy pky highway hwy terrace ter terr
boulevard plaza plz point pt pike loop run row path expressway expy freeway fwy north south east west n s e w ne nw se sw
de du des la le les l d et the of and""".split())
_STREET_TYPE = set("""rue r av ave avenue bd blvd boul boulevard allee all impasse imp chemin ch route rte place pl cours quai
square sq passage street st road rd drive dr lane ln court ct way circle cir trail trl parkway pkwy highway hwy terrace""".split())
_NUM_COMP = re.compile(r"^\(?\s*(?:no\.?|n°|nº|n\s?o\.?|#|num(?:ero)?\.?)?\s*0*(\d{1,6})\s*\)?\s*(?:-\s*)?"
                       r"(?:(bis|ter|quater)\b|([a-z])\b)?\s*(.*)$")
_WORD = re.compile(r"[a-z0-9]+")


def _fold(text: str) -> str:
    if not text:
        return ""
    t = unicodedata.normalize("NFKD", text)
    return "".join(c for c in t if not unicodedata.combining(c)).lower()


def parse_address(raw: str):
    """(house number or None, street string, rest string)."""
    comps = [c.strip() for c in _fold(raw).split(",") if c.strip()]
    for i, c in enumerate(comps):
        m = _NUM_COMP.match(c)
        if m and m.group(1) and re.search(r"[a-z]{3,}", m.group(4) or ""):
            street = " ".join(w for w in _WORD.findall(m.group(4)) if w not in _DROP)
            rest = " ".join(comps[:i] + comps[i + 1:])
            return m.group(1).lstrip("0") or "0", street, rest
    for i, c in enumerate(comps):                    # no house number: the first component naming a street type
        words = _WORD.findall(c)
        if any(w in _STREET_TYPE for w in words):
            street = " ".join(w for w in words if w not in _DROP)
            if street:
                return None, street, " ".join(comps[:i] + comps[i + 1:])
    return None, "", " ".join(comps)


def parse_many(texts: list[str]):
    hn, street, rest = [], [], []
    for t in texts:
        a, b, c = parse_address(t or "")
        hn.append(a)
        street.append(b)
        rest.append(c)
    return hn, street, rest


def structure_features(s1_addr: pa.Array, pool_addr: pa.Array, s1_row: np.ndarray, pool_row: np.ndarray,
                       prefix: str = "as_") -> dict:
    """Features for pairs (s1_row, pool_row); s1_addr / pool_addr hold the ORIGINAL address of every partition record.
    Each distinct record is parsed once."""
    u1, inv1 = np.unique(s1_row, return_inverse=True)
    u2, inv2 = np.unique(pool_row, return_inverse=True)
    h1, st1, r1 = parse_many(s1_addr.take(pa.array(u1)).to_pylist())
    h2, st2, r2 = parse_many(pool_addr.take(pa.array(u2)).to_pylist())
    h1 = np.asarray(h1, dtype=object)[inv1]
    h2 = np.asarray(h2, dtype=object)[inv2]
    st1 = np.asarray(st1, dtype=object)[inv1]
    st2 = np.asarray(st2, dtype=object)[inv2]
    r1 = np.asarray(r1, dtype=object)[inv1]
    r2 = np.asarray(r2, dtype=object)[inv2]
    both = np.array([a is not None and b is not None for a, b in zip(h1, h2)])
    eq = np.array([a == b for a, b in zip(h1, h2)])
    hn_eq = np.where(both, eq.astype(np.float32), np.nan).astype(np.float32)
    has_st = np.array([bool(a) and bool(b) for a, b in zip(st1, st2)])
    ss = process.cpdist(st1.tolist(), st2.tolist(), scorer=fuzz.token_set_ratio, workers=-1).astype(np.float32) / 100.0
    sj = process.cpdist(st1.tolist(), st2.tolist(), scorer=JaroWinkler.normalized_similarity, workers=-1).astype(np.float32)
    ss[~has_st], sj[~has_st] = np.nan, np.nan
    has_r = np.array([bool(a) and bool(b) for a, b in zip(r1, r2)])
    rs = process.cpdist(r1.tolist(), r2.tolist(), scorer=fuzz.token_set_ratio, workers=-1).astype(np.float32) / 100.0
    rs[~has_r] = np.nan
    return {
        f"{prefix}hn_both": both.astype(np.int8),
        f"{prefix}hn_eq": hn_eq,
        f"{prefix}street_sim": ss,
        f"{prefix}street_jw": sj,
        f"{prefix}num_eq_street_diff": ((hn_eq == 1) & (np.nan_to_num(ss, nan=1.0) < 0.5)).astype(np.int8),
        f"{prefix}rest_sim": rs,
    }
