#!/usr/bin/env python3
"""Fill Documentation_template.md from the artifacts of the actual run, so the write-up can only state what was measured.

Reads reports/*.json|csv|md and artifacts/*.json; writes the filled document (default: student_resource/Documentation_template.md).
Numbers are taken from the files, never typed: change the pipeline, re-run this script, the document follows.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import Paths  # noqa: E402


def md_table(df: pd.DataFrame, digits: int = 4) -> str:
    """Markdown table; columns holding only whole numbers (counts that became float in a transposed frame) print as integers."""
    int_cols = set()
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().any() and s.notna().sum() == df[c].notna().sum() and (s.dropna() % 1 == 0).all():
            int_cols.add(c)

    def fmt(v, col):
        if isinstance(v, float):
            if v != v:
                return ""
            return f"{int(v):,}" if col in int_cols else f"{v:.{digits}f}"
        return str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns)
    return head + "\n" + "\n".join("| " + " | ".join(fmt(v, c) for v, c in zip(row, df.columns)) + " |"
                                    for row in df.itertuples(index=False))


def grab(text: str, pattern: str, default: str = "n/a") -> str:
    m = re.search(pattern, text)
    return m.group(1) if m else default


# (stack experiment, report row, label in the stage table); rows whose report does not exist are skipped
STAGES = [
    ("ce", "oracle: exactly the true candidates", "ceiling: exactly the true candidates of the v3 candidate sets"),
    ("ce", "first-stage LightGBM p1", "1. first-stage LightGBM matcher (v1)"),
    ("ce", "exclusivity-normalized q(p1)", "2. parameter-free exclusivity rule q(p1)"),
    ("noce", "STACKER", "3. graph stacker: p1 + collective features (v2)"),
    ("ce", "cross-encoder alone", "4. cross-encoder alone (mDeBERTa-v3-base)"),
    ("ce", "mean(p1, p_ce)", "5. mean of p1 and the cross-encoder"),
    ("ce", "STACKER", "6. graph stacker + cross-encoder (v3)"),
    ("v4pre", "oracle: exactly the true candidates", "ceiling: exactly the true candidates of the v4 candidate sets"),
    ("v4pre", "STACKER", "7. v4 candidates, cross-encoder on first-hop pairs only"),
    ("v4", "STACKER", "8. v4 candidates + cross-encoder on all pairs (v4)"),
]


def stage_values(paths: Paths, final_exp: str, final_scores: str) -> dict:
    """Placeholders of the stage-2 sections: stage table, final model numbers and top features, final test statistics."""
    reports = {}
    for exp in {s[0] for s in STAGES} | {final_exp}:
        p = paths.artifacts_dir / f"stack_{exp}" / "stack_report.json"
        if p.exists():
            reports[exp] = json.loads(p.read_text())
    rows = []
    for exp, key, label in STAGES:
        r = reports.get(exp, {}).get(key)
        if r:
            rows.append({"stage": label, "threshold": r["threshold"], "validation F0.5": r["val_macro_f05"],
                         "hold-out F0.5": r["hold_macro_f05"], "hold-out precision": r["hold_precision"],
                         "hold-out recall": r["hold_recall"], "hold-out singleton acc.": r["hold_singleton_acc"]})
    out = {"STAGE_TABLE": md_table(pd.DataFrame(rows)) if rows else "(not available)"}
    fin = reports.get(final_exp, {})
    st = fin.get("STACKER")
    if st:
        out.update({"FINAL_HOLDOUT_F05": f"{st['hold_macro_f05']:.4f}", "FINAL_VAL_F05": f"{st['val_macro_f05']:.4f}",
                    "FINAL_P": f"{st['hold_precision']:.4f}", "FINAL_R": f"{st['hold_recall']:.4f}",
                    "FINAL_SINGLETON": f"{st['hold_singleton_acc']:.4f}", "FINAL_THRESHOLD": f"{st['threshold']:.2f}",
                    "FINAL_TREES": str(st.get("trees", "n/a")),
                    "FINAL_TOP_FEATURES": ", ".join(f"`{k}` ({v * 100:.1f}%)" for k, v in list(fin.get("feature_gain_share_top40", {}).items())[:12])})
    orc = fin.get("oracle: exactly the true candidates")
    if orc:
        out["FINAL_CEILING"] = f"{orc['hold_macro_f05']:.4f}"
    cand = fin.get("candidates_hold")
    if cand:
        out["FINAL_CAND_RECALL"] = f"{cand['candidate_recall']:.2%}"
        out["FINAL_CPE_HOLD"] = f"{cand['candidates_per_entity']:.2f}"
    inf_p = paths.reports_dir / f"inference_summary_{final_scores}.json"
    if inf_p.exists():
        inf = json.loads(inf_p.read_text())
        pc = inf.get("per_country", {})
        out["FINAL_TEST_PAIRS"] = f"{inf.get('pairs', 0):,}"
        out["FINAL_TEST_CPE"] = f"{inf.get('pairs', 0) / max(1, inf.get('n_s1_entities', 1)):.2f}"
        out["FINAL_INFERENCE_PER_COUNTRY"] = md_table(pd.DataFrame(pc).T.reset_index().rename(columns={"index": "country"}))
    npf = paths.artifacts_dir / "new_pair_filter.json"
    if npf.exists():
        nf = json.loads(npf.read_text())
        out["NEWPAIR_FLOOR"] = f"{nf['floor']:g}"
        out["NEWPAIR_KEEP"] = f"{nf['keep_share']:.0%}"
        out["NEWPAIR_GAIN"] = f"{nf['val_recall_gain_at_floor']:.2%}"
        out["NEWPAIR_TABLE"] = md_table(pd.DataFrame(nf["tradeoff_val"]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--template", required=True, help="markdown with {PLACEHOLDERS}")
    ap.add_argument("--out", default=None)
    ap.add_argument("--extra", default=None, help="JSON file with additional {PLACEHOLDER: text} values (prose written by hand)")
    ap.add_argument("--final-exp", default="ce", help="artifacts/stack_<exp> of the submitted stacker")
    ap.add_argument("--final-scores", default="scores_test_stack_ce", help="scores dir whose inference summary describes the submission")
    ap.add_argument("--edge-tag", default=None, help="edge-case report of this tag (stack_<exp>: the submitted stacker, stack_edge_inputs.py)")
    args = ap.parse_args()
    paths = Paths(args.root, args.tag)
    rep = paths.reports_dir
    sfx = paths.suffix

    blocking = (rep / f"blocking_report{sfx}.md").read_text(encoding="utf-8")
    exp = pd.read_csv(rep / f"experiment_log{sfx}.csv")
    val_report = json.loads((rep / f"validation_report{sfx}.json").read_text())
    cfg = json.loads(paths.threshold_path().read_text())
    hold = exp[exp["experiment"].str.startswith("FINAL_holdout")].iloc[0]
    infer_p = rep / f"inference_summary{sfx}.json"
    infer = json.loads(infer_p.read_text()) if infer_p.exists() else {}
    stage2_p = rep / f"stage2_report{sfx}.json"
    stage2 = json.loads(stage2_p.read_text()) if stage2_p.exists() else None
    edge_p = rep / (f"edge_case_report_{args.edge_tag}.md" if args.edge_tag else f"edge_case_report{sfx}.md")
    edge = edge_p.read_text(encoding="utf-8") if edge_p.exists() else ""

    def section(title: str, text: str) -> str:
        m = re.search(rf"## {re.escape(title)}\n(.*?)(?=\n## |\Z)", text, re.S)
        return m.group(1).strip() if m else "(not available)"

    def section_h3(title: str, text: str) -> str:
        m = re.search(rf"### {re.escape(title)}\n(.*?)(?=\n##|\Z)", text, re.S)
        return m.group(1).strip() if m else "(not available)"

    if cfg["mode"] == "global":
        rule = f"a single global threshold ({cfg['global_threshold']:.2f}) was selected"
    elif cfg["mode"] == "per_source":
        rule = f"per-source thresholds (S2 {cfg['s2_threshold']:.2f} / S3 {cfg['s3_threshold']:.2f}) were selected"
    else:
        rule = f"an entity gate ({cfg['gate']:.2f}) with per-source thresholds was selected"
    if stage2:
        rule += ("; the second stage (entity-context re-scoring) was " +
                 (f"ADOPTED (validation {stage2['stage1_validation_f05']:.4f} -> {stage2['stage2_validation_f05']:.4f})" if stage2["adopted"]
                  else f"NOT adopted (validation {stage2['stage1_validation_f05']:.4f} -> {stage2['stage2_validation_f05']:.4f}, "
                       f"needs +{stage2['margin_required']})"))

    keep_cols = ["experiment", "features", "model", "threshold", "candidate_recall", "mean_precision_non_singleton",
                 "mean_recall_non_singleton", "macro_f05", "singleton_accuracy", "avg_candidates_per_entity"]
    log_tbl = exp[keep_cols].rename(columns={"mean_precision_non_singleton": "precision", "mean_recall_non_singleton": "recall",
                                             "macro_f05": "F0.5", "singleton_accuracy": "singleton acc.",
                                             "avg_candidates_per_entity": "avg cand./entity", "candidate_recall": "candidate recall"})
    fps = re.findall(r"\| B matching: false merge on an entity with matches \| ([\d,]+) \| ([\d.]+)%", edge)
    sel_p = rep / "model_selection.json"
    sel = json.loads(sel_p.read_text()) if sel_p.exists() else None
    fcfg = json.loads(paths.filter_config_path().read_text()) if paths.filter_config_path().exists() else {}
    short_row = re.search(r"\| 1\. retrieval[^|]*\| ([\d.]+) \| ([\d.]+%) \|", blocking)
    short_test = paths.shortlist_dir("test")
    n_short_test = sum(pq.ParquetFile(f).metadata.num_rows for f in short_test.glob("country=*.parquet")) if short_test.exists() else 0
    per_country = infer.get("per_country", {})
    lf_rows = [("S1 entities", "s1_entities", "{:,.0f}"), ("candidates per entity", "candidates_per_entity", "{:.2f}"),
               ("entities without any candidate", "share_without_candidates", "{:.1%}"),
               ("entities with a predicted match", "share_with_match", "{:.1%}"),
               ("predicted matches per entity that has one", "predicted_matches_per_entity_that_has_one", "{:.2f}"),
               ("best candidate score >= 0.97", "share_best_score_ge_0.97", "{:.1%}"),
               ("best candidate score in [0.4, 0.9)", "share_best_score_0.4_to_0.9", "{:.1%}"),
               ("best score < 0.2 or no candidate", "share_best_score_below_0.2_or_no_candidate", "{:.1%}")]
    labelfree = ("| | " + " | ".join(per_country) + " |\n|---|" + "---|" * len(per_country) + "\n" + "\n".join(
        f"| {label} | " + " | ".join(fmt.format(v[key]) if v.get(key) is not None else "n/a" for v in per_country.values()) + " |"
        for label, key, fmt in lf_rows)) if per_country else "(not available)"

    def exp_f05(prefix: str) -> str:
        r = exp[exp["experiment"].str.startswith(prefix)]
        return f"{r.iloc[0]['macro_f05']:.3f}" if len(r) else "n/a"

    def baseline_range() -> tuple[float, float]:
        f = exp[exp["experiment"].str.match(r"B(1|2|3|3b)_")]["macro_f05"]
        return (float(f.min()), float(f.max())) if len(f) else (float("nan"), float("nan"))

    def tax(label: str, i: int) -> str:
        """Count (i=1) or share (i=2) of one failure-taxonomy row of the edge-case report."""
        m = re.search(rf"\| {re.escape(label)}[^|]*\| ([\d,]+) \| ([\d.]+%) \|", edge)
        return m.group(i) if m else "n/a"

    def case(prefix: str, col: int) -> str:
        """Column of one per-edge-case row: 1 entities, 2 candidate recall, 3 precision, 4 recall, 5 macro F0.5."""
        m = re.search(rf"\| {re.escape(prefix)}[^|]*\|" + r" ([^|]*) \|" * 5, edge)
        return m.group(col).strip() if m else "n/a"

    def stage_minutes(log_name: str, stage: str) -> str:
        """Minutes of a run_all stage; when the stage was run per country (cut_<C>.log / score_<C>.log), the summed time."""
        logs = paths.artifacts_dir / "logs"
        p = logs / log_name
        m = re.findall(rf"stage '{stage}' finished with exit code 0 in (\d+)s", p.read_text(encoding="utf-8", errors="replace")) if p.exists() else []
        if m:
            return f"{int(m[-1]) / 60:.0f}"
        pattern, prefix = ((r"candidate cut complete in (\d+)s", "cut_") if stage == "cut-test"
                           else (r"scored [\d,]+ pairs in (\d+)s", "score_"))
        secs = [int(x) for f in logs.glob(f"{prefix}*.log") for x in re.findall(pattern, f.read_text(encoding="utf-8", errors="replace"))]
        return f"{sum(secs) / 60:.0f}" if secs else "n/a"
    model_selection = "" if not sel else (
        f"Two matcher variants were trained on identical data: the default feature set (validation {sel['default']['validation']:.4f}, "
        f"hold-out {sel['default']['holdout']:.4f}) and a France-robust set that drops the features whose scale depends on the size of the "
        f"country pool (validation {sel['robust']['validation']:.4f}, hold-out {sel['robust']['holdout']:.4f}). Because France has no labels and a "
        f"3-4x smaller pool, the robust set is promoted whenever it is within {sel['tolerance']} of the default on both validation and hold-out: "
        f"**{sel['promoted']} features were promoted**.")
    values = {
        "MODEL_SELECTION": model_selection,
        "N_TREES": str(val_report.get("best_iteration", "n/a")),
        "CAND_RECALL": grab(blocking, r"overall candidate recall: ([\d.]+%)"),
        "CAND_PER_ENTITY": grab(blocking, r"per entity mean=([\d.]+)"),
        "N_HOLDOUT": f"{int(hold['n_entities']):,}",
        "HOLDOUT_F05": f"{hold['macro_f05']:.4f}",
        "VAL_F05": f"{cfg['validation_macro_f05']:.4f}",
        "CEILING": f"{cfg['oracle_given_blocking']:.4f}",
        "HOLDOUT_DETAILS": (f"On the hold-out: singleton accuracy {hold['singleton_accuracy']:.3f}, mean precision "
                            f"{hold['mean_precision_non_singleton']:.3f}, mean recall {hold['mean_recall_non_singleton']:.3f}, "
                            f"candidate recall {hold['candidate_recall']:.3f}."),
        "DECISION_RULE": rule,
        "TEST_PAIRS": f"{infer.get('pairs', 0):,}",
        "TEST_S1": f"{infer.get('n_s1_entities', 0):,}",
        "TEST_CPE": f"{infer.get('pairs', 0) / max(1, infer.get('n_s1_entities', 1)):.1f}",
        "TEST_SHORTLIST_PAIRS": f"{n_short_test:,}",
        "TEST_SHORTLIST_CPE": f"{n_short_test / max(1, infer.get('n_s1_entities', 1)):.1f}",
        "SHORTLIST_CPE": short_row.group(1) if short_row else "n/a",
        "SHORTLIST_RECALL": short_row.group(2) if short_row else "n/a",
        "FILTER_FLOOR": f"{fcfg['min_score']:g}" if fcfg else "n/a",
        "FILTER_KEEP": f"{1 - fcfg['recall_budget']:.1%}" if fcfg else "n/a",
        "FILTER_TOP_FEATURES": ", ".join(f"`{k}` ({v*100:.0f}%)" for k, v in list(fcfg.get("feature_gain_share", {}).items())[:6]),
        "FILTER_TABLE": section_h3("Candidate filter: size / recall trade-off (out-of-fold probabilities, all sampled entities)", blocking),
        "LABELFREE_TABLE": labelfree,
        **{k: exp_f05(p) for k, p in (("B1_F05", "B1_"), ("B2_F05", "B2_"), ("B3_F05", "B3_"), ("B3B_F05", "B3b_"))},
        "BASELINE_RANGE": "{:.2f} - {:.2f}".format(*baseline_range()),
        "TAX_A_SHARE": tax("A blocking", 2), "TAX_B_REJ_SHARE": tax("B matching: true candidate rejected", 2),
        "TAX_B_FP": tax("B matching: false merge", 1), "TAX_C": tax("C singleton", 1),
        "CASE_MISSING_ADDR_RECALL": case("12 missing address", 4), "CASE_NONLATIN_RECALL": case("17 transliteration", 4),
        "CASE_AMBIG_F05": case("22 ambiguous", 5), "CASE_AMBIG_P": case("22 ambiguous", 3),
        "CUT_TEST_MIN": stage_minutes("run_all_cut.log", "cut-test"),
        "SCORE_TEST_MIN": stage_minutes("run_all_cut.log", "score-test"),
        "TAXONOMY": section("Failure taxonomy (all validation entities)", edge),
        "EDGE_TABLE": section("Per edge case", edge),
        "SOURCE_TABLE": section("Source 2 vs Source 3", edge),
        "EXPERIMENT_LOG": md_table(log_tbl),
        "BLOCKING_LOG": (rep / "experiment_log_blocking.md").read_text(encoding="utf-8") if (rep / "experiment_log_blocking.md").exists() else "",
        "INFERENCE_PER_COUNTRY": md_table(pd.DataFrame(infer.get("per_country", {})).T.reset_index().rename(columns={"index": "country"})) if infer else "",
        "TOP_FEATURES": ", ".join(f"`{k}` ({v*100:.1f}%)" for k, v in list(val_report["top_features_gain_share"].items())[:8]),
    }
    values.update(stage_values(paths, args.final_exp, args.final_scores))
    values.setdefault("FINAL_CEILING", values["CEILING"])   # same candidate sets as the first design
    if args.extra:
        values.update(json.loads(Path(args.extra).read_text(encoding="utf-8")))
    text = Path(args.template).read_text(encoding="utf-8")
    for _ in range(3):                      # hand-written prose may itself contain placeholders: iterate until stable
        before = text
        for k, v in values.items():
            text = text.replace("{" + k + "}", str(v))
        if text == before:
            break
    left = re.findall(r"\{[A-Z_0-9]+\}", text)
    out = Path(args.out) if args.out else paths.root / "Documentation_template.md"
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}; unfilled placeholders: {sorted(set(left))}")


if __name__ == "__main__":
    main()
