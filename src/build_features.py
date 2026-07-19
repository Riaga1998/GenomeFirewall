"""
Module 01 batch runner — genomes -> AMRFinderPlus -> feature matrix
===================================================================
Runs AMRFinderPlus on every .fna in --genomes-dir (caching each TSV so re-runs
are free), parses the results, and writes the binary genome x determinant matrix
the predictor consumes.

    python src/build_features.py --genomes-dir data/genomes \
        --cache data/amrfinder --organism Staphylococcus_aureus \
        --jobs 4 --out data/features.parquet

INSTALL AMRFinderPlus (the one non-pip dependency — use conda/bioconda):
    conda install -c bioconda -c conda-forge ncbi-amrfinderplus
    amrfinder -u          # downloads the reference database (required, once)

Why --organism matters: passing Staphylococcus_aureus enables AMRFinderPlus's
curated POINT-mutation screen (gyrA/grlA/rpoB etc.) and organism-specific
rules. Without it you only get acquired genes and miss the mutation features.
Check valid names with:  amrfinder -l
"""
from __future__ import annotations
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from features import run_amrfinder, parse_amrfinder_tsv, build_feature_matrix  # noqa: E402


def _process(fasta: Path, cache: Path, organism: str | None):
    """Run (or reuse cached) AMRFinderPlus for one genome; return (id, features|None)."""
    gid = fasta.stem
    out_tsv = cache / f"{gid}.tsv"
    try:
        run_amrfinder(fasta, out_tsv, organism=organism)
        feats = parse_amrfinder_tsv(out_tsv)
        return gid, feats, None
    except Exception as e:
        return gid, None, str(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genomes-dir", required=True, type=Path)
    ap.add_argument("--cache", type=Path, default=Path("data/amrfinder"))
    ap.add_argument("--organism", default=None,
                    help="e.g. Staphylococcus_aureus (enables point mutations)")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("data/features.parquet"))
    args = ap.parse_args()

    args.cache.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fastas = sorted(args.genomes_dir.glob("*.fna"))
    if not fastas:
        sys.exit(f"No .fna files in {args.genomes_dir}. Run download_genomes.py first.")
    print(f"{len(fastas)} genomes -> AMRFinderPlus (organism={args.organism}, "
          f"jobs={args.jobs})")

    per_genome, failed = {}, []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(_process, f, args.cache, args.organism): f for f in fastas}
        for i, fut in enumerate(as_completed(futs), 1):
            gid, feats, err = fut.result()
            if err is None:
                per_genome[gid] = feats
            else:
                failed.append((gid, err))
            if i % 100 == 0:
                print(f"  {i}/{len(fastas)} ({len(failed)} failed)")

    if not per_genome:
        sys.exit("AMRFinderPlus produced no results. Is it installed + `amrfinder -u` run?")

    # build matrix from the cached TSVs we succeeded on
    ok_ids = sorted(per_genome)
    tsvs = {gid: args.cache / f"{gid}.tsv" for gid in ok_ids}
    mat = build_feature_matrix(ok_ids, tsvs)
    mat.to_parquet(args.out)

    print(f"\nWrote {args.out}: {mat.shape[0]} genomes x {mat.shape[1]} determinants")
    if failed:
        log = args.cache / "amrfinder_failures.log"
        log.write_text("\n".join(f"{g}\t{e}" for g, e in failed) + "\n")
        print(f"{len(failed)} genomes failed (see {log})")
    print("Next: align features with labels.csv and run run_pipeline.py")


if __name__ == "__main__":
    main()
