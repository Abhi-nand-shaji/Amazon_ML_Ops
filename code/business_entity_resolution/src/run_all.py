#!/usr/bin/env python3
"""Regenerate every artifact and both submission files from the raw data, without manual steps.

    python code/business_entity_resolution/src/run_all.py                # everything
    python code/business_entity_resolution/src/run_all.py --from blocking-test   # resume from a stage
    python code/business_entity_resolution/src/run_all.py --list

Stages are resumable individually (chunked parquet parts, atomic writes) and skip work that is already complete,
so re-running after an interruption continues where it stopped. Run it from the `student_resource/` directory
(or pass --root); all stages use the interpreter that runs this script.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = SRC.parents[2]

# (name, script, arguments)
STAGES = [
    ("cache", "prepare_cache.py", []),
    ("ranker", "train_ranker.py", []),
    ("blocking-train", "run_blocking.py", ["--split", "train", "--n-s1", "50000", "--workers", "3",
                                            "--chunk-s1", "4000", "--expansion-budget", "6000000"]),
    ("filter-train", "train_candidate_filter.py", []),
    ("cut-train", "run_candidate_cut.py", ["--split", "train"]),
    ("features-train", "run_features.py", ["--split", "train"]),
    ("train", "run_train.py", []),
    ("train-robust", "run_train.py", ["--robust", "--exp", "robust", "--no-compare-models"]),
    ("promote", "promote_model.py", []),
    ("blocking-test", "run_blocking.py", ["--split", "test", "--workers", "3", "--chunk-s1", "50000",
                                          "--expansion-budget", "6000000"]),
    ("cut-test", "run_candidate_cut.py", ["--split", "test"]),
    ("score-test", "run_score.py", ["--split", "test"]),
    ("infer", "run_infer.py", []),
    ("validate", str(ROOT / "utils" / "validate_submission.py"),
     ["--matching", "output/matching_results.tsv", "--candidate", "output/candidate_pairs.tsv", "--test-dir", "dataset/test"]),
    ("reports", "run_blocking_report.py", []),
    ("edge-cases", "run_edge_cases.py", []),
    ("documentation", "build_documentation.py", ["--template", "code/business_entity_resolution/reports/documentation_template.md.in",
                                                 "--extra", "code/business_entity_resolution/reports/documentation_prose.json"]),
]
# Implemented and tested but NOT part of the submitted pipeline (it must beat stage 1 on validation to be adopted, and was not run):
#   python code/business_entity_resolution/src/run_stage2.py   (run it before the `infer` stage if you want to try it)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="start", default=None, help="first stage to run (see --list)")
    ap.add_argument("--only", default=None, help="run a single stage")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    names = [s[0] for s in STAGES]
    if args.list:
        print("\n".join(names))
        return
    todo = STAGES
    if args.only:
        todo = [s for s in STAGES if s[0] == args.only]
    elif args.start:
        todo = STAGES[names.index(args.start):]
    for name, script, extra in todo:
        path = script if Path(script).is_absolute() else str(SRC / script)
        cmd = [sys.executable, "-u", path, *extra]
        print(f"\n===== stage '{name}': {' '.join(cmd[1:])}", flush=True)
        t0 = time.time()
        rc = subprocess.call(cmd, cwd=str(ROOT))
        print(f"===== stage '{name}' finished with exit code {rc} in {time.time() - t0:.0f}s", flush=True)
        if rc != 0:
            sys.exit(f"stage '{name}' failed (exit code {rc}); fix and re-run with --from {name}")


if __name__ == "__main__":
    main()
