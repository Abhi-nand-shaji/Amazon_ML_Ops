#!/usr/bin/env python3
"""Assemble the final submission archive in the layout the challenge asks for:

    <name>_submission.zip
    |-- output/matching_results.tsv, output/candidate_pairs.tsv
    |-- code/business_entity_resolution/{src, tests, reports, README.md, requirements.txt, artifacts (models + configs only)}
    `-- Documentation_template.md   (the filled-in methodology document)

The multi-GB normalized caches, candidate / feature / score parquet files and logs are NOT included: everything can be
regenerated from the raw data with `python code/business_entity_resolution/src/run_all.py`.
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CODE = ROOT / "code" / "business_entity_resolution"
SKIP_DIRS = {"__pycache__", "cache", "logs"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="business_entity_resolution", help="team name used as the archive prefix")
    ap.add_argument("--out-dir", default=str(ROOT / "submission"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{args.name}_submission.zip"

    members: list[tuple[Path, str]] = []
    for f in ("matching_results.tsv", "candidate_pairs.tsv"):
        p = ROOT / "output" / f
        if not p.exists():
            sys.exit(f"missing {p}: run the pipeline first")
        members.append((p, f"output/{f}"))
    doc = ROOT / "Documentation_template.md"
    if doc.exists():
        members.append((doc, "Documentation_template.md"))
    for p in sorted(CODE.rglob("*")):
        rel = p.relative_to(CODE)
        if p.is_dir() or any(part in SKIP_DIRS for part in rel.parts) or p.suffix in {".pyc", ".tmp"}:
            continue
        if rel.parts[0] == "artifacts" and p.suffix not in {".txt", ".json"}:
            continue                                           # only models / configs / feature lists from artifacts/
        members.append((p, f"code/business_entity_resolution/{rel.as_posix()}"))

    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p, arc in members:
            z.write(p, arc)
            print(f"  + {arc}  ({p.stat().st_size / 1e6:.1f} MB)")
    print(f"\nwrote {target} ({target.stat().st_size / 1e6:.0f} MB, {len(members)} files)")


if __name__ == "__main__":
    main()
