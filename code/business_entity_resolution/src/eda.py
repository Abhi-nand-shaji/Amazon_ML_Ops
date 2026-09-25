#!/usr/bin/env python3
"""One-shot exploratory data analysis over the raw train/test files.

Writes a markdown report to reports/data_analysis_report.md and prints a
condensed summary to stdout. Read-only: touches no ground truth at inference
time, this is purely for understanding the data before designing the pipeline.
"""
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]  # student_resource/
DATASET = ROOT / "dataset"
REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"
REPORT_DIR.mkdir(exist_ok=True)

PUNCT_RE = re.compile(r"[^0-9a-z\s]+")
WS_RE = re.compile(r"\s+")
NONASCII_RE = re.compile(r"[^\x00-\x7F]")
DIGIT5_RE = re.compile(r"\b\d{5}\b")
DIGIT6_RE = re.compile(r"\b\d{6}\b")
LEADING_NUM_RE = re.compile(r"^\s*\d+")


def quick_norm(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str).str.lower()
    s = s.str.replace(PUNCT_RE, " ", regex=True)
    s = s.str.replace(WS_RE, " ", regex=True).str.strip()
    return s


def load(path):
    return pd.read_csv(path, sep="\t", engine="pyarrow", dtype_backend="pyarrow")


def basic_profile(name, df, lines):
    lines.append(f"\n### {name}\n")
    lines.append(f"- rows: {len(df):,}")
    lines.append(f"- unique entity_id: {df['entity_id'].nunique():,} (is_unique={df['entity_id'].is_unique})")
    nm = df["business_name"].fillna("")
    ad = df["business_address"].fillna("")
    lines.append(f"- null business_name: {df['business_name'].isna().sum():,}; empty-string: {(nm.str.strip()=='').sum():,}")
    lines.append(f"- null business_address: {df['business_address'].isna().sum():,}; empty-string: {(ad.str.strip()=='').sum():,}")
    lines.append(f"- null country: {df['country'].isna().sum():,}")
    vc = df["country"].value_counts()
    lines.append(f"- country distribution: {dict(vc)}")
    namelen = nm.str.len()
    addrlen = ad.str.len()
    lines.append(f"- name length: mean={namelen.mean():.1f} median={namelen.median():.1f} p90={namelen.quantile(0.9):.1f} max={namelen.max()}")
    lines.append(f"- address length: mean={addrlen.mean():.1f} median={addrlen.median():.1f} p90={addrlen.quantile(0.9):.1f} max={addrlen.max()}")
    nonascii_name = nm.astype(str).str.contains(NONASCII_RE, regex=True)
    lines.append(f"- non-ASCII business_name: {nonascii_name.sum():,} ({nonascii_name.mean()*100:.2f}%)")
    dup = df.duplicated(subset=["business_name", "business_address"]).sum()
    lines.append(f"- exact duplicate (name,address) rows: {dup:,} ({dup/len(df)*100:.2f}%)")
    has_zip5 = ad.astype(str).str.contains(DIGIT5_RE, regex=True)
    has_zip6 = ad.astype(str).str.contains(DIGIT6_RE, regex=True)
    has_leadnum = ad.astype(str).str.contains(LEADING_NUM_RE, regex=True)
    lines.append(f"- address contains a 5-digit token: {has_zip5.mean()*100:.2f}%; 6-digit token: {has_zip6.mean()*100:.2f}%; starts with digits: {has_leadnum.mean()*100:.2f}%")
    return {
        "nonascii_name_mask_sum": int(nonascii_name.sum()),
    }


def main():
    lines = ["# Data Analysis Report\n", f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"]
    t0 = time.time()

    lines.append("\n## 1. Per-source profile (TRAIN)\n")
    s1 = load(DATASET / "train/train_source1.tsv")
    s2 = load(DATASET / "train/train_source2.tsv")
    s3 = load(DATASET / "train/train_source3.tsv")
    basic_profile("train_source1", s1, lines)
    basic_profile("train_source2", s2, lines)
    basic_profile("train_source3", s3, lines)

    lines.append("\n## 2. Per-source profile (TEST)\n")
    t1 = load(DATASET / "test/test_source1.tsv")
    t2 = load(DATASET / "test/test_source2.tsv")
    t3 = load(DATASET / "test/test_source3.tsv")
    basic_profile("test_source1", t1, lines)
    basic_profile("test_source2", t2, lines)
    basic_profile("test_source3", t3, lines)

    lines.append("\n## 3. Train vs test distribution differences\n")
    for label, tr, te in [("source1", s1, t1), ("source2", s2, t2), ("source3", s3, t3)]:
        tr_c = tr["country"].value_counts(normalize=True)
        te_c = te["country"].value_counts(normalize=True)
        lines.append(f"- {label} train country%: {dict((tr_c*100).round(2))}")
        lines.append(f"- {label} test  country%: {dict((te_c*100).round(2))}")

    del t1, t2, t3

    lines.append("\n## 4. Ground truth analysis\n")
    gt = pd.read_csv(DATASET / "train/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
    lines.append(f"- rows: {len(gt):,}; unique source1_entity_id: {gt['source1_entity_id'].nunique():,}")
    empty_mask = gt["matched_entity_ids"].str.strip() == ""
    lines.append(f"- singleton (no match) S1 entities: {empty_mask.sum():,} ({empty_mask.mean()*100:.2f}%)")

    split_ids = gt["matched_entity_ids"].apply(lambda s: [] if not s.strip() else s.split(","))
    n_total = split_ids.apply(len)
    n_s2 = split_ids.apply(lambda lst: sum(1 for x in lst if x.startswith("S2-")))
    n_s3 = split_ids.apply(lambda lst: sum(1 for x in lst if x.startswith("S3-")))
    lines.append(f"- match count distribution:\n```\n{n_total.value_counts().sort_index().to_string()}\n```")
    lines.append(f"- mean matches/entity (incl singletons): {n_total.mean():.3f}; mean given >=1 match: {n_total[n_total>0].mean():.3f}; max: {n_total.max()}")
    lines.append(f"- S1 with S2-only matches: {((n_s2>0)&(n_s3==0)).sum():,}")
    lines.append(f"- S1 with S3-only matches: {((n_s3>0)&(n_s2==0)).sum():,}")
    lines.append(f"- S1 with BOTH S2 and S3 matches: {((n_s2>0)&(n_s3>0)).sum():,}")
    lines.append(f"- S1 with multiple S2 matches (>1): {(n_s2>1).sum():,}; multiple S3 matches (>1): {(n_s3>1).sum():,}")
    lines.append(f"- total positive pairs: {int(n_total.sum()):,} (S2 side: {int(n_s2.sum()):,}, S3 side: {int(n_s3.sum()):,})")

    n_s2_rows, n_s3_rows = len(s2), len(s3)
    matched_s2_ids = set()
    matched_s3_ids = set()
    for lst in split_ids:
        for x in lst:
            if x.startswith("S2-"):
                matched_s2_ids.add(x)
            elif x.startswith("S3-"):
                matched_s3_ids.add(x)
    lines.append(f"- distinct S2 ids used as a true match: {len(matched_s2_ids):,} / {n_s2_rows:,} rows ({len(matched_s2_ids)/n_s2_rows*100:.2f}%)")
    lines.append(f"- distinct S3 ids used as a true match: {len(matched_s3_ids):,} / {n_s3_rows:,} rows ({len(matched_s3_ids)/n_s3_rows*100:.2f}%)")

    lines.append("\n## 5. True-pair similarity analysis (join gt with record fields)\n")
    long_gt = gt.loc[~empty_mask, ["source1_entity_id", "matched_entity_ids"]].copy()
    long_gt["matched_entity_ids"] = split_ids[~empty_mask]
    long_gt = long_gt.explode("matched_entity_ids").rename(columns={"matched_entity_ids": "matched_id"})
    print(f"exploded positive pairs: {len(long_gt):,}  (elapsed {time.time()-t0:.1f}s)")

    s1_idx = s1.set_index("entity_id")[["business_name", "business_address", "country"]]
    s2_idx = s2.set_index("entity_id")[["business_name", "business_address", "country"]]
    s3_idx = s3.set_index("entity_id")[["business_name", "business_address", "country"]]
    both_idx = pd.concat([s2_idx, s3_idx])

    long_gt = long_gt.join(s1_idx.add_suffix("_1"), on="source1_entity_id")
    long_gt = long_gt.join(both_idx.add_suffix("_2"), on="matched_id")
    long_gt["is_s2"] = long_gt["matched_id"].str.startswith("S2-")

    n1 = quick_norm(long_gt["business_name_1"])
    n2 = quick_norm(long_gt["business_name_2"])
    a1 = quick_norm(long_gt["business_address_1"])
    a2 = quick_norm(long_gt["business_address_2"])

    exact_raw_name = (long_gt["business_name_1"].fillna("") == long_gt["business_name_2"].fillna(""))
    exact_norm_name = (n1 == n2)
    exact_norm_addr = (a1 == a2)
    country_match = (long_gt["country_1"].fillna("") == long_gt["country_2"].fillna(""))

    def tok_jaccard(a, b):
        sa = set(a.split()) if a else set()
        sb = set(b.split()) if b else set()
        if not sa and not sb:
            return 1.0
        if not sa or not sb:
            return 0.0
        inter = len(sa & sb)
        union = len(sa | sb)
        return inter / union if union else 0.0

    sample_n = min(200_000, len(long_gt))
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(long_gt), size=sample_n, replace=False)
    name_jac = np.array([tok_jaccard(n1.iloc[i], n2.iloc[i]) for i in sample_idx])
    addr_jac = np.array([tok_jaccard(a1.iloc[i], a2.iloc[i]) for i in sample_idx])

    lines.append(f"- exact RAW name match among true pairs: {exact_raw_name.mean()*100:.2f}%")
    lines.append(f"- exact NORMALIZED name match among true pairs: {exact_norm_name.mean()*100:.2f}%")
    lines.append(f"- exact NORMALIZED address match among true pairs: {exact_norm_addr.mean()*100:.2f}%")
    lines.append(f"- country match among true pairs: {country_match.mean()*100:.4f}% (mismatches: {(~country_match).sum():,})")
    lines.append(f"- (sampled {sample_n:,}) name token-Jaccard quantiles: {np.quantile(name_jac,[0,.1,.25,.5,.75,.9,1]).round(3).tolist()}")
    lines.append(f"- (sampled {sample_n:,}) addr token-Jaccard quantiles: {np.quantile(addr_jac,[0,.1,.25,.5,.75,.9,1]).round(3).tolist()}")
    lines.append(f"- frac true pairs with name_jaccard==0 (sampled): {(name_jac==0).mean()*100:.2f}%")
    lines.append(f"- frac true pairs with addr_jaccard==0 (sampled): {(addr_jac==0).mean()*100:.2f}%")
    both_low = (name_jac < 0.2) & (addr_jac < 0.2)
    lines.append(f"- frac true pairs with BOTH name_jaccard<0.2 AND addr_jaccard<0.2 (sampled): {both_low.mean()*100:.2f}%")

    is_s2 = long_gt["is_s2"].to_numpy()[sample_idx]
    for lbl, mask in [("S2", is_s2), ("S3", ~is_s2)]:
        if mask.sum() > 0:
            lines.append(f"  - [{lbl} true pairs] name_jac median={np.median(name_jac[mask]):.3f}, addr_jac median={np.median(addr_jac[mask]):.3f}, n={mask.sum():,}")

    name1_nonascii = n1.astype(str).str.contains(NONASCII_RE, regex=True) if False else long_gt["business_name_1"].fillna("").astype(str).str.contains(NONASCII_RE, regex=True)
    name2_nonascii = long_gt["business_name_2"].fillna("").astype(str).str.contains(NONASCII_RE, regex=True)
    cross_script = name1_nonascii != name2_nonascii
    lines.append(f"- true pairs where exactly one side's name is non-ASCII (possible transliteration case): {cross_script.sum():,} ({cross_script.mean()*100:.3f}%)")
    both_nonascii = name1_nonascii & name2_nonascii
    lines.append(f"- true pairs where BOTH sides' name are non-ASCII: {both_nonascii.sum():,} ({both_nonascii.mean()*100:.3f}%)")

    lines.append("\n## 6. Source2 vs Source3 noise comparison (independent, not just via true pairs)\n")
    for lbl, df in [("S2", s2), ("S3", s3)]:
        nm = quick_norm(df["business_name"])
        legal_tokens = nm.str.extract(r"\b(inc|llc|ltd|corp|corporation|limited|private|pvt|co|company)\b", expand=False)
        lines.append(f"- {lbl}: legal-suffix-token presence in name: {legal_tokens.notna().mean()*100:.2f}%")

    elapsed = time.time() - t0
    lines.append(f"\n_EDA total runtime: {elapsed:.1f}s_\n")

    report_path = REPORT_DIR / "data_analysis_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote report to {report_path} ({elapsed:.1f}s total)")


if __name__ == "__main__":
    main()
