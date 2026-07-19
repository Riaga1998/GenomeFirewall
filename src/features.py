"""
Module 01 — The Genome Reader
=============================
Reconstructed FASTA  ->  AMRFinderPlus  ->  binary feature matrix.

We keep AMRFinderPlus as the default annotation tool (public-domain, the brief's
gold standard). Each genome becomes a row; each known AMR gene / point mutation
becomes a column (1 = detected, 0 = absent).

Every feature carries a TIER so downstream reports can honestly separate
"known resistance mechanism" from "statistical association":
    tier 1 = curated known AMR determinant (AMRFinderPlus hit)
    tier 2 = anything the model leans on that is NOT a curated hit (handled in
             decision.py, not here)
"""
from __future__ import annotations
import subprocess
import shutil
from pathlib import Path
import pandas as pd


# AMRFinderPlus renamed columns across versions. Be defensive.
_SYMBOL_COLS = ["Element symbol", "Gene symbol"]
_TYPE_COLS = ["Element type"]
_SUBTYPE_COLS = ["Element subtype"]
_CLASS_COLS = ["Class"]


def _first_present(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def run_amrfinder(fasta: Path, out_tsv: Path, organism: str | None = None) -> Path:
    """Run AMRFinderPlus on a single assembled nucleotide FASTA.

    Requires `amrfinder` on PATH. Results are cached to out_tsv so re-runs are
    free. If the cache exists we skip the call.
    """
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    if out_tsv.exists():
        return out_tsv
    if shutil.which("amrfinder") is None:
        raise RuntimeError(
            "amrfinder not found on PATH. Install AMRFinderPlus, or use the "
            "precomputed TSV cache / synthetic path (make_synthetic.py)."
        )
    cmd = ["amrfinder", "--nucleotide", str(fasta), "--plus", "-o", str(out_tsv)]
    if organism:
        cmd += ["--organism", organism]
    subprocess.run(cmd, check=True)
    return out_tsv


def parse_amrfinder_tsv(tsv: Path) -> set[str]:
    """Return the set of AMR determinant symbols detected in one genome.

    We keep only Element type == AMR (drop STRESS / VIRULENCE). Point mutations
    are kept and namespaced so 'gyrA_S83L' is distinct from gene presence.
    """
    df = pd.read_csv(tsv, sep="\t")
    if df.empty:
        return set()

    sym_col = _first_present(df, _SYMBOL_COLS)
    type_col = _first_present(df, _TYPE_COLS)
    subtype_col = _first_present(df, _SUBTYPE_COLS)
    if sym_col is None:
        raise ValueError(f"No symbol column in {tsv}; columns={list(df.columns)}")

    if type_col is not None:
        df = df[df[type_col].astype(str).str.upper() == "AMR"]

    feats: set[str] = set()
    for _, row in df.iterrows():
        symbol = str(row[sym_col]).strip()
        if not symbol or symbol.lower() == "nan":
            continue
        subtype = str(row[subtype_col]).upper() if subtype_col else ""
        prefix = "MUT:" if subtype == "POINT" else "GENE:"
        feats.add(prefix + symbol)
    return feats


def build_feature_matrix(
    genome_ids: list[str],
    tsv_paths: dict[str, Path],
) -> pd.DataFrame:
    """Aggregate per-genome determinant sets into a binary genome x feature matrix."""
    per_genome = {gid: parse_amrfinder_tsv(tsv_paths[gid]) for gid in genome_ids}
    all_feats = sorted({f for feats in per_genome.values() for f in feats})

    rows = []
    for gid in genome_ids:
        present = per_genome[gid]
        rows.append({f: int(f in present) for f in all_feats})
    mat = pd.DataFrame(rows, index=genome_ids).fillna(0).astype(int)
    mat.index.name = "genome_id"
    return mat


def curated_determinant_columns(mat: pd.DataFrame) -> list[str]:
    """All AMRFinderPlus-derived columns are tier-1 (curated known determinants)."""
    return [c for c in mat.columns if c.startswith(("GENE:", "MUT:"))]


if __name__ == "__main__":
    # Example: build features from a directory of cached AMRFinder TSVs.
    import sys
    import yaml

    cfg = yaml.safe_load(open("config/config.yaml"))
    cache = Path(cfg["paths"]["amrfinder_cache"])
    tsvs = {p.stem: p for p in cache.glob("*.tsv")}
    if not tsvs:
        sys.exit(f"No TSVs in {cache}. Run AMRFinderPlus first, or use synthetic data.")
    mat = build_feature_matrix(sorted(tsvs), tsvs)
    Path(cfg["paths"]["features"]).parent.mkdir(parents=True, exist_ok=True)
    mat.to_parquet(cfg["paths"]["features"])
    print(f"Wrote features: {mat.shape[0]} genomes x {mat.shape[1]} determinants")
