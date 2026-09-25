#!/usr/bin/env python3
"""Ground-truth edge-case diagnostics on the validation entities.

For every edge case of the brief (singletons, one-to-one, multiple / cross-source matches, name-only and
address-only evidence, collisions, missing fields, normalization / token-order / typo / transliteration
pairs, blocking misses, rejected candidates, ambiguity, ...) this reports: the number of affected S1
entities, candidate recall, final precision / recall / F0.5, the failure count, and representative
examples. Failures are split into

  A. BLOCKING   a true match never became a candidate (candidate-recall failure)
  B. MATCHING   a true candidate was rejected (missed) or a false candidate accepted for an entity
                that does have true matches (false merge)
  C. SINGLETON  an entity with no true match received a prediction
  D. OUTPUT     the output files violate a constraint (checked on output/*.tsv when present)

Input: artifacts/val_predictions<tag>.parquet + val_entities<tag>.json (written by run_train.py) and
the decision config; nothing is recomputed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import Paths  # noqa: E402
from evaluation import FastScorer  # noqa: E402
from ids import load_truth  # noqa: E402
from run_infer import decide  # noqa: E402

TEXT_COLS = ["entity_id", "name_original", "address_original"]


def md_table(df: pd.DataFrame, digits: int = 3) -> str:
    """Dependency-free markdown table (DataFrame.to_markdown would need `tabulate`). Count columns that only became
    float because some cells are missing are printed as integers."""
    int_cols = set()
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().any() and s.notna().sum() == df[c].notna().sum() and (s.dropna() % 1 == 0).all():
            int_cols.add(c)

    def fmt(v, col):
        if isinstance(v, (float, np.floating)):
            if np.isnan(v):
                return ""
            return f"{int(v):,}" if col in int_cols else f"{v:.{digits}f}"
        return str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns)
    return head + "\n" + "\n".join("| " + " | ".join(fmt(v, c) for v, c in zip(row, df.columns)) + " |"
                                    for row in df.itertuples(index=False))


def texts(paths: Paths, ids_by_source: dict) -> dict:
    """entity_id -> 'name | address' for the few ids shown as examples."""
    out = {}
    for source, ids in ids_by_source.items():
        if not ids:
            continue
        t = pq.read_table(paths.normalized_cache("train", source), columns=TEXT_COLS, filters=[("entity_id", "in", sorted(ids))]).to_pylist()
        out.update({r["entity_id"]: f"{r['name_original']} | {r['address_original'] or '(no address)'}" for r in t})
    return out


def output_checks(paths: Paths) -> list[str]:
    """D. OUTPUT/PIPELINE failures on the real submission files, when they exist."""
    m, c = paths.output_dir / "matching_results.tsv", paths.output_dir / "candidate_pairs.tsv"
    if not (m.exists() and c.exists()):
        return ["output files not present yet - run run_infer.py, then re-run this report"]
    md = pd.read_csv(m, sep="\t", dtype=str, keep_default_na=False, engine="pyarrow")
    cd = pd.read_csv(c, sep="\t", dtype=str, keep_default_na=False, engine="pyarrow")
    lines = []
    ml = md["matched_entity_ids"].str.split(",").map(lambda x: [i for i in x if i])
    cl = cd.set_index("source1_entity_id")["candidate_entity_ids"].str.split(",").map(lambda x: [i for i in x if i])
    lines.append(f"rows: matching {len(md):,}, candidate {len(cd):,}; duplicate S1 rows: {int(md['source1_entity_id'].duplicated().sum())} / {int(cd['source1_entity_id'].duplicated().sum())}")
    lines.append(f"S1/other-source ids inside predicted lists: {int(ml.map(lambda x: any(not i.startswith(('S2-', 'S3-')) for i in x)).sum())} entities")
    lines.append(f"duplicate ids inside a predicted list: {int(ml.map(lambda x: len(x) != len(set(x))).sum())} entities; inside a candidate list: {int(cl.map(lambda x: len(x) != len(set(x))).sum())} entities")
    sub_viol = sum(1 for e, x in zip(md["source1_entity_id"], ml) if x and not set(x) <= set(cl.get(e, [])))
    lines.append(f"subset invariant (every match is a candidate) violated for {sub_viol} entities")
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--examples", type=int, default=3)
    args = ap.parse_args()
    paths = Paths(args.root, args.tag)
    sfx = paths.suffix
    val = pd.read_parquet(paths.artifacts_dir / f"val_predictions{sfx}.parquet")
    val_ids = json.loads((paths.artifacts_dir / f"val_entities{sfx}.json").read_text())
    cfg = json.loads(paths.threshold_path().read_text())
    truth = load_truth(paths.ground_truth(), set(val_ids))

    ent_code = pd.factorize(val["s1_entity_id"])[0]          # unique per entity (s1_row repeats across countries)
    keep = decide(val["p"].to_numpy(), val["is_s2"].to_numpy().astype(bool), cfg, ent_code)
    scorer = FastScorer(val["s1_entity_id"].to_numpy(), val["label"].to_numpy(), truth, val_ids)
    st = scorer.stats(keep).set_index("s1_entity_id")
    n_cand_true = val.groupby("s1_entity_id")["label"].sum().reindex(val_ids).fillna(0).astype(int)
    st["true_cands"] = n_cand_true.to_numpy()
    st["n_true_s2"] = [sum(1 for i in truth[e] if i.startswith("S2-")) for e in st.index]
    st["n_true_s3"] = [sum(1 for i in truth[e] if i.startswith("S3-")) for e in st.index]
    st["fn_block"] = st["truth_size"] - st["true_cands"]                    # A: true matches that never became candidates
    st["fn_match"] = st["true_cands"] - st["tp"]                            # B: true candidates rejected
    st["fp_nonsingleton"] = np.where(st["truth_size"] > 0, st["fp"], 0)    # B: false merges on entities that have matches
    st["singleton_fail"] = ((st["truth_size"] == 0) & (st["pred_size"] > 0)).astype(int)   # C

    df = val.assign(keep=keep)
    T, F = df[df["label"] == 1], df[df["label"] == 0]
    high_false = F[F["p"] >= 0.3]["s1_entity_id"].unique()
    has_true = T["s1_entity_id"].unique()

    def ents(mask_rows):
        return set(mask_rows["s1_entity_id"].unique())

    def nan0(col, frame):
        return frame[col].fillna(0)

    cases = {
        "1 true singleton": set(st.index[st["truth_size"] == 0]),
        "2 one-to-one match": set(st.index[st["truth_size"] == 1]),
        "3 multiple matches": set(st.index[st["truth_size"] >= 2]),
        "4 cross-source match (S2 and S3)": set(st.index[(st["n_true_s2"] > 0) & (st["n_true_s3"] > 0)]),
        "5 multiple matches within one source": set(st.index[(st["n_true_s2"] > 1) | (st["n_true_s3"] > 1)]),
        "6 name-only strong (name>=.85, address weak/missing)": ents(T[(nan0("name_ratio", T) >= 0.85) & (nan0("addr_ratio", T) < 0.5)]),
        "7 address-only strong (address>=.85, name weak/missing)": ents(T[(nan0("addr_ratio", T) >= 0.85) & (nan0("name_ratio", T) < 0.5)]),
        "8 name+address both noisy (both < .7)": ents(T[(T["name_ratio"] < 0.7) & (T["addr_ratio"] < 0.7)]),
        "9 name collision (false candidate with name>=.9)": ents(F[nan0("name_ratio", F) >= 0.9]),
        "10 address collision (false candidate with address>=.9)": ents(F[nan0("addr_ratio", F) >= 0.9]),
        "11 missing name (either side)": ents(T[(T["name_missing_1"] == 1) | (T["name_missing_2"] == 1)]),
        "12 missing address (either side)": ents(T[(T["addr_missing_1"] == 1) | (T["addr_missing_2"] == 1)]),
        "13 multiple missing fields": ents(T[((T["name_missing_1"] == 1) | (T["name_missing_2"] == 1)) & ((T["addr_missing_1"] == 1) | (T["addr_missing_2"] == 1))]),
        "14 normalization-only (identical after normalization)": ents(T[T["name_exact_normalized"] == 1]),
        "15 token-order (token-sort>=.95, plain ratio<.9)": ents(T[(nan0("name_token_sort", T) >= 0.95) & (nan0("name_ratio", T) < 0.9)]),
        "16 typo (name JW>=.9 but not identical)": ents(T[(nan0("name_jw", T) >= 0.9) & (T["name_exact_normalized"] == 0)]),
        "17 transliteration / non-Latin name": ents(T[T["name_script_2"] >= 2]),
        "22 ambiguous (true candidate AND a false candidate p>=.3)": set(high_false) & set(has_true),
    }
    rows, details = [], {}
    for name, ent in cases.items():
        e = st.loc[sorted(ent & set(st.index))]
        if len(e) == 0:
            rows.append({"edge case": name, "S1 entities": 0})
            continue
        tp, fp, fn_b, fn_m = int(e["tp"].sum()), int(e["fp"].sum()), int(e["fn_block"].sum()), int(e["fn_match"].sum())
        truth_pairs = int(e["truth_size"].sum())
        rows.append({
            "edge case": name, "S1 entities": len(e),
            "candidate recall": (e["true_cands"].sum() / truth_pairs) if truth_pairs else np.nan,
            "precision": tp / (tp + fp) if tp + fp else np.nan,
            "recall": tp / truth_pairs if truth_pairs else np.nan,
            "macro F0.5": float(e["f05"].mean()),
            "A blocking failures": fn_b, "B matching failures": fn_m + int(e["fp_nonsingleton"].sum()),
            "C singleton failures": int(e["singleton_fail"].sum()),
        })
        details[name] = list(e.sort_values("f05").index[: args.examples])

    # ---- global failure taxonomy
    tax = {
        "A blocking (true match not a candidate)": int(st["fn_block"].sum()),
        "B matching: true candidate rejected": int(st["fn_match"].sum()),
        "B matching: false merge on an entity with matches": int(st["fp_nonsingleton"].sum()),
        "C singleton with a predicted match": int(st["singleton_fail"].sum()),
    }
    n_true_pairs = int(st["truth_size"].sum())

    # ---- examples (worst entities per case)
    need = {"source1": set(), "source2": set(), "source3": set()}
    ex_rows = []
    for name, es in details.items():
        for e in es:
            worst = df[(df["s1_entity_id"] == e)].sort_values("p", ascending=False)
            miss = worst[(worst["label"] == 1) & (~worst["keep"])].head(1)
            fpos = worst[(worst["label"] == 0) & (worst["keep"])].head(1)
            for kind, r in (("missed true match", miss), ("false merge", fpos)):
                if len(r):
                    cid = r["cand_entity_id"].iloc[0]
                    ex_rows.append((name, e, kind, cid, float(r["p"].iloc[0])))
                    need["source1"].add(e)
                    need["source2" if cid.startswith("S2-") else "source3"].add(cid)
    tx = texts(paths, need)

    L = ["# Edge-case diagnostics (validation entities)\n",
         f"Decision config: `{json.dumps(cfg)}`. Validation entities: {len(st):,}; true pairs: {n_true_pairs:,}; "
         f"overall macro F0.5 = {st['f05'].mean():.4f}.\n",
         "## Failure taxonomy (all validation entities)\n",
         "| type | count | share of true pairs |\n|---|---|---|"]
    for k, v in tax.items():
        L.append(f"| {k} | {v:,} | {v / n_true_pairs * 100:.2f}% |")
    L.append("\n## Per edge case\n")
    L.append("Entities can belong to several cases. Precision / recall are pooled over the case's entities; F0.5 is the macro mean.\n")
    tbl = pd.DataFrame(rows)
    L.append(md_table(tbl))
    # ---- Source 2 vs Source 3 (Section 17 of the brief): noise level and matching difficulty per source
    src_rows = []
    for name, m in (("S2", val["is_s2"] == 1), ("S3", val["is_s2"] == 0)):
        d, kp = val[m], keep[m.to_numpy()]
        t = d[d["label"] == 1]
        tp = int(((d["label"] == 1).to_numpy() & kp).sum())
        fp = int(((d["label"] == 0).to_numpy() & kp).sum())
        src_rows.append({
            "source": name, "true pairs in candidates": len(t),
            "mean name similarity (true pairs)": float(t["name_ratio"].mean()),
            "mean address similarity (true pairs)": float(t["addr_ratio"].mean()),
            "address missing": float((t["addr_missing_2"] == 1).mean()),
            "non-Latin name": float((t["name_script_2"] >= 2).mean()),
            "name identical after normalization": float((t["name_exact_normalized"] == 1).mean()),
            "precision": tp / (tp + fp) if tp + fp else np.nan, "recall (of candidate true pairs)": tp / len(t) if len(t) else np.nan,
        })
    L.append("\n## Source 2 vs Source 3\n")
    src_tbl = pd.DataFrame(src_rows)
    L.append(md_table(src_tbl))
    L.append("\n## Not measurable / structural cases\n")
    L.append("- 18 country conflict: 0 of 7,638,365 training true pairs cross a country boundary and candidates are generated per country partition, so no cross-country candidate exists; country mismatch is treated as a hard partition because the data prove it safe.")
    L.append("- 19 unseen country (France): test only; `country` is not a model feature and partitions are formed from whatever labels occur. See `inference_summary.json` for per-country prediction statistics.")
    L.append(f"- 20 true match missed by blocking: {tax['A blocking (true match not a candidate)']:,} pairs (row A above); 21 true candidate rejected: {tax['B matching: true candidate rejected']:,} pairs (row B).")
    dup = int(val.duplicated(["s1_entity_id", "cand_entity_id"]).sum())
    L.append(f"- 23 duplicate candidates: {dup} duplicated (S1, candidate) rows in the validation candidate set (candidates are the union of several rules, deduplicated by construction).")
    L.append(f"- 24 empty ground truth parsed as zero matches: {int((st['truth_size'] == 0).sum()):,} validation singletons.")
    L.append("\n## D. Output / pipeline checks on output/*.tsv (cases 25-27)\n")
    L += [f"- {x}" for x in output_checks(paths)]
    L.append("\n## Representative examples (worst entities of each case)\n")
    for name, e, kind, cid, p in ex_rows[: 60]:
        L.append(f"- [{name}] {kind} p={p:.2f}: S1 `{tx.get(e, e)}`  <->  `{tx.get(cid, cid)}`")
    out = paths.report_path("edge_case_report", "md")
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:60]))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
