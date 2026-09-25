#!/usr/bin/env python3
"""Candidate-generation quality report for the TRAIN split (needs ground truth).

Answers: what is the recall ceiling, how many candidates per entity, which
rules contribute, what each step (retrieval + ranker shortlist, candidate filter)
keeps, how size trades against recall, and *why* true matches get missed (so
blocking can be improved on evidence).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import Paths  # noqa: E402
from ids import id_to_int  # noqa: E402
from run_blocking import candidates_dir, shortlist_dir  # noqa: E402


def rank_within(group_sorted: np.ndarray) -> np.ndarray:
    first = np.flatnonzero(np.r_[True, group_sorted[1:] != group_sorted[:-1]])
    counts = np.diff(np.r_[first, len(group_sorted)])
    return np.arange(len(group_sorted)) - np.repeat(first, counts)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=None)
    ap.add_argument("--tag", default="", help="pipeline-variant suffix (must match the blocking run)")
    ap.add_argument("--miss-detail", type=int, default=40, help="number of example misses to print")
    args = ap.parse_args()
    paths = Paths(args.root, args.tag)
    cdir = candidates_dir(paths, "train")
    meta = json.loads((cdir / "_meta.json").read_text())
    t0 = time.time()

    ents = pq.read_table(cdir / "s1_entities.parquet")
    ent_int = id_to_int(ents["entity_id"])
    ent_country = pd.Series(ents["country"].to_pylist(), index=ent_int)
    n_s1 = len(ent_int)

    frames = []
    evidence = ["n_name_shared", "n_addr_shared", "exact_name", "exact_compact", "postal_eq", "n_strategies",
                "block_score", "pool_row", "n_addr_bigram", "n_name_bigram", "n_prefix", "rank_score"]
    for f in sorted(cdir.glob("country=*.parquet")):
        t = pq.read_table(f, columns=["country", "s1_entity_id", "cand_entity_id", "cand_source"] + evidence)
        frames.append(pd.DataFrame({
            "country": t["country"].to_pylist(),
            "s1": id_to_int(t["s1_entity_id"]), "cand": id_to_int(t["cand_entity_id"]),
            "src": t["cand_source"].to_pylist(),
            **{k: t[k].to_numpy(zero_copy_only=False) for k in evidence},
        }))
    cand = pd.concat(frames, ignore_index=True)
    del frames
    print(f"loaded {len(cand):,} candidate pairs for {n_s1:,} sampled S1 entities ({time.time()-t0:.1f}s)", flush=True)
    # the shortlist the final candidates were cut from (retrieval + ranker, top-k by ranker probability)
    sdir = shortlist_dir(paths, "train")
    short = pd.concat([pd.DataFrame({"s1": id_to_int(t["s1_entity_id"]), "cand": id_to_int(t["cand_entity_id"]),
                                     "pool_row": t["pool_row"].to_numpy(), "rank_score": t["rank_score"].to_numpy()})
                       for t in (pq.read_table(f, columns=["s1_entity_id", "cand_entity_id", "pool_row", "rank_score"])
                                 for f in sorted(sdir.glob("country=*.parquet")))], ignore_index=True)
    order = np.lexsort((short["pool_row"].to_numpy(), -short["rank_score"].to_numpy(), short["s1"].to_numpy()))
    short = short.iloc[order].reset_index(drop=True)
    short["short_rank"] = rank_within(short["s1"].to_numpy())
    fcfg_path = paths.filter_config_path()
    fcfg = json.loads(fcfg_path.read_text()) if fcfg_path.exists() else None

    # rank of each candidate within its entity by the ranking score (same tie-break as truncation)
    order = np.lexsort((cand["pool_row"].to_numpy(), -cand["rank_score"].to_numpy(), cand["s1"].to_numpy()))
    cand = cand.iloc[order].reset_index(drop=True)
    cand["rank"] = rank_within(cand["s1"].to_numpy())

    # ground truth restricted to the sampled entities
    gt = pd.read_csv(paths.ground_truth(), sep="\t", dtype=str, keep_default_na=False)
    gt_int = id_to_int(pa.array(gt["source1_entity_id"].tolist()))
    keep = np.isin(gt_int, ent_int)
    gt = gt.loc[keep].copy()
    gt["s1"] = gt_int[keep]
    gt = gt[gt["matched_entity_ids"].str.strip() != ""]
    gt["cand_id"] = gt["matched_entity_ids"].str.split(",")
    tp = gt[["s1", "cand_id"]].explode("cand_id")
    tp["cand"] = id_to_int(pa.array(tp["cand_id"].tolist()))
    tp["src"] = np.where(tp["cand_id"].str.startswith("S2-"), "S2", "S3")
    tp = tp[["s1", "cand", "cand_id", "src"]]
    n_true = len(tp)
    n_singletons = n_s1 - tp["s1"].nunique()

    m = tp.merge(cand, on=["s1", "cand"], how="left", suffixes=("", "_c"))
    m["found"] = m["rank"].notna()
    m["country"] = ent_country.reindex(m["s1"].to_numpy()).to_numpy()
    m = m.merge(short[["s1", "cand", "short_rank"]], on=["s1", "cand"], how="left")
    m["in_shortlist"] = m["short_rank"].notna()

    L = ["# Blocking (Candidate Generation) Report - TRAIN sample\n",
         f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
         f"Configuration: `{json.dumps(meta['blocking_config'])}`; S1 sample = {meta['n_s1_sample']} (seed {meta['seed']})\n",
         f"Candidate filter: `{json.dumps(meta.get('candidate_filter'))}`\n"]
    per_ent = cand.groupby("s1").size().reindex(ent_int).fillna(0)
    L.append("## Volume\n")
    L.append(f"- S1 entities blocked: {n_s1:,}; true singletons among them: {n_singletons:,} ({n_singletons/n_s1*100:.2f}%)")
    L.append(f"- candidate pairs: {len(cand):,}; per entity mean={per_ent.mean():.2f} median={per_ent.median():.0f} "
             f"p90={per_ent.quantile(.9):.0f} max={per_ent.max():.0f}")
    L.append(f"- entities with ZERO candidates: {(per_ent==0).sum():,} ({(per_ent==0).mean()*100:.3f}%)")
    src_counts = cand["src"].value_counts()
    L.append(f"- candidates by source: {dict(src_counts)}")
    pools = {}
    for s in ("source2", "source3"):
        cs = pq.read_table(paths.normalized_cache("train", s), columns=["country"])["country"].to_pylist()
        for c in set(cs):
            pools[c] = pools.get(c, 0) + cs.count(c)
    denom = sum(int((ent_country == c).sum()) * pools[c] for c in pools if c in set(ent_country))
    L.append(f"- candidate reduction ratio vs. same-country cross product: {1 - len(cand)/denom:.6f} "
             f"({len(cand):,} of {denom:,} possible same-country pairs kept)")

    L.append("\n## Recall (fraction of true (S1, match) pairs present in the candidate set)\n")
    L.append(f"- true pairs among sampled entities: {n_true:,}")
    L.append(f"- **overall candidate recall: {m['found'].mean()*100:.3f}%**")
    for src, g in m.groupby("src"):
        L.append(f"  - {src}: {g['found'].mean()*100:.3f}% (n={len(g):,})")
    for c, g in m.groupby("country"):
        L.append(f"  - country {c}: {g['found'].mean()*100:.3f}% (n={len(g):,})")
    ent_full = m.groupby("s1")["found"].all()
    L.append(f"- entities whose EVERY true match is a candidate: {ent_full.mean()*100:.2f}% of entities with >=1 true match")

    short_per_ent = short.groupby("s1").size().reindex(ent_int).fillna(0)
    L.append("\n### Step by step (share of ALL true pairs of the sampled entities that survive)\n")
    L.append("| step | candidates / entity | recall |\n|---|---|---|")
    L.append(f"| 1. retrieval (inverted indices) + learned ranker, shortlist of the best {meta['blocking_config']['max_candidates']} "
             f"| {short_per_ent.mean():.2f} | {m['in_shortlist'].mean()*100:.3f}% |")
    floor = meta.get("candidate_filter", {}).get("min_score")
    L.append(f"| 2. candidate filter (probability >= {floor}) = **final candidate set** | {per_ent.mean():.2f} | {m['found'].mean()*100:.3f}% |")

    L.append("\n### Shortlist recall if the ranker cut were k\n")
    L.append("| k | recall | mean shortlisted/entity |\n|---|---|---|")
    for k in (5, 10, 15, 20, 30):
        if k > int(short_per_ent.max()) and k != 5:
            break
        L.append(f"| {k} | {(m['short_rank'] < k).mean()*100:.3f}% | {np.minimum(short_per_ent, k).mean():.1f} |")
    if fcfg:
        L.append("\n### Candidate filter: size / recall trade-off (out-of-fold probabilities, all sampled entities)\n")
        L.append("| probability floor | candidates / entity | recall of all true pairs | entities without candidates | true singletons without candidates |")
        L.append("|---|---|---|---|---|")
        for r in fcfg["tradeoff_all_sampled_entities_oof"]:
            mark = " **(chosen)**" if r["floor"] == fcfg["min_score"] else (" (no filter)" if r["floor"] == 0 else "")
            L.append(f"| {r['floor']:g}{mark} | {r['candidates_per_entity']:.2f} | {r['recall_all_true_pairs']*100:.3f}% | "
                     f"{r['entities_without_candidates']*100:.2f}% | {r['true_singletons_without_candidates']*100:.1f}% |")
        L.append(f"\nFloor chosen on the matcher's validation entities: the largest grid value that loses at most "
                 f"{fcfg['recall_budget']:.1%} of their shortlisted true pairs. Filter features by gain: "
                 + ", ".join(f"`{k}` {v*100:.1f}%" for k, v in list(fcfg["feature_gain_share"].items())[:8]) + ".")

    L.append("\n### Contribution of each blocking rule (recall of true pairs carrying that rule's evidence)\n")
    f = m[m["found"]]
    L.append(f"- rare NAME-token rule: {(f['n_name_shared']>0).sum()/n_true*100:.3f}%")
    L.append(f"- rare ADDRESS-token rule: {(f['n_addr_shared']>0).sum()/n_true*100:.3f}%")
    L.append(f"- ADDRESS-bigram rule: {(f['n_addr_bigram']>0).sum()/n_true*100:.3f}%; NAME-bigram rule: "
             f"{(f['n_name_bigram']>0).sum()/n_true*100:.3f}%; name-prefix rule: {(f['n_prefix']>0).sum()/n_true*100:.3f}%")
    L.append(f"- exact normalized name: {(f['exact_name']>0).sum()/n_true*100:.3f}%; exact compact name: {(f['exact_compact']>0).sum()/n_true*100:.3f}%; postal equal: {(f['postal_eq']>0).sum()/n_true*100:.3f}%")
    L.append(f"- found ONLY via name rule: {((f['n_name_shared']>0)&(f['n_addr_shared']==0)&(f['exact_name']==0)&(f['exact_compact']==0)&(f['postal_eq']==0)).sum()/n_true*100:.3f}%")
    L.append(f"- found ONLY via address rule: {((f['n_name_shared']==0)&(f['n_addr_shared']>0)&(f['exact_name']==0)&(f['exact_compact']==0)&(f['postal_eq']==0)).sum()/n_true*100:.3f}%")
    L.append(f"- found by exactly one rule (fragile): {(f['n_strategies']==1).sum()/n_true*100:.3f}%; by >=2 rules: {(f['n_strategies']>=2).sum()/n_true*100:.3f}%")
    dens = m.groupby("s1")["found"].mean()
    fp_rank = m[m["found"]]
    L.append(f"- among FOUND true pairs, rank by block score: median={fp_rank['rank'].median():.0f}, p90={fp_rank['rank'].quantile(.9):.0f}, p99={fp_rank['rank'].quantile(.99):.0f}")

    # ---- why are true pairs missed?
    miss = m[~m["found"]].copy()
    L.append(f"\n## Why true matches are missed ({len(miss):,} pairs = {len(miss)/n_true*100:.3f}%)\n")
    if len(miss):
        s1_id_by_int = gt.drop_duplicates("s1").set_index("s1")["source1_entity_id"]
        s1_ids_needed = s1_id_by_int.reindex(miss["s1"].unique()).tolist()
        s1_txt = pq.read_table(paths.normalized_cache("train", "source1"),
                               columns=["entity_id", "name_original", "address_original", "name_core", "address_normalized"],
                               filters=[("entity_id", "in", s1_ids_needed)]).to_pandas()
        s1_txt = s1_txt.drop_duplicates("entity_id").set_index("entity_id")
        miss_ids = miss["cand_id"].tolist()
        cand_rows = []
        for s in ("source2", "source3"):
            pref = "S2-" if s == "source2" else "S3-"
            ids = [i for i in miss_ids if i.startswith(pref)]
            if ids:
                cand_rows.append(pq.read_table(paths.normalized_cache("train", s),
                                               columns=["entity_id", "name_original", "address_original", "name_core",
                                                        "address_normalized", "name_script"],
                                               filters=[("entity_id", "in", ids)]).to_pandas())
        c_txt = pd.concat(cand_rows).drop_duplicates("entity_id").set_index("entity_id")
        miss = miss.reset_index(drop=True)
        miss["s1_id"] = s1_id_by_int.reindex(miss["s1"].to_numpy()).to_numpy()
        j = miss.join(s1_txt.add_prefix("s1_"), on="s1_id").join(c_txt.add_prefix("c_"), on="cand_id")
        tok = lambda s: set(str(s).split())
        j["name_overlap"] = [len(tok(a) & tok(b)) for a, b in zip(j["s1_name_core"], j["c_name_core"])]
        j["addr_overlap"] = [len(tok(a) & tok(b)) for a, b in zip(j["s1_address_normalized"], j["c_address_normalized"])]
        j["addr_missing"] = j["c_address_original"].fillna("").str.strip() == ""
        cross = j["c_name_script"].fillna("latin") != "latin"
        cat = np.select(
            [cross & j["addr_missing"], cross, j["addr_missing"], (j["name_overlap"] == 0) & (j["addr_overlap"] == 0),
             (j["name_overlap"] == 0), (j["addr_overlap"] == 0)],
            ["non-Latin/accented name AND missing address", "non-Latin/accented name (name tokens can't match)",
             "candidate address missing (only name tokens available)", "no shared token at all in name or address",
             "no shared name token (address tokens shared)", "no shared address token (name tokens shared)"],
            default="shares name+address tokens but dropped (rare-token selection / cap / df_cap)")
        j["category"] = cat
        n_filtered = int(j["in_shortlist"].sum())
        L.append(f"Lost at step 1 (never shortlisted): {len(j) - n_filtered:,} pairs; removed by the candidate filter (step 2): "
                 f"{n_filtered:,} pairs.\n")
        L.append("| category of missed pair | count | share of misses | of which removed by the filter |\n|---|---|---|---|")
        for k, v in j["category"].value_counts().items():
            L.append(f"| {k} | {v:,} | {v/len(j)*100:.1f}% | {int(j.loc[j['category'] == k, 'in_shortlist'].sum()):,} |")
        L.append(f"\nMissed pairs by source: {dict(j['src'].value_counts())}; by country: {dict(j['country'].value_counts())}\n")
        L.append("Examples (S1 name | S1 address  -->  missed candidate name | address):\n")
        for _, r in j.sample(min(args.miss_detail, len(j)), random_state=0).iterrows():
            L.append(f"- [{r['category'][:34]}] `{r['s1_name_original']}` | `{r['s1_address_original']}`  -->  "
                     f"`{r['c_name_original']}` | `{r['c_address_original']}`")
    out = paths.report_path("blocking_report", "md")
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\nwrote {out} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
