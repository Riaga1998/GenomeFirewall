"""
BV-BRC AMR phenotype -> clean labels
====================================
Turns a BV-BRC AMR table (from ANY of: the website "AMR Phenotypes" Download
button, the FTP bulk file PATRIC_genome_AMR.txt, or the data API) into the two
files the Genome Firewall pipeline needs:

    data/labels.csv       genome_id, antibiotic, label(R/S)
    data/genome_list.txt  one genome_id per line (feed to the FASTA downloader)

Key safety filter (per the brief): keep ONLY laboratory-measured results, drop
model-generated / predicted phenotypes. We do that by requiring a real
laboratory typing method and an evidence value that is not a prediction.

Usage:
    python src/download_bvbrc.py --amr-tsv path/to/bvbrc_amr.tsv \
        --species "Klebsiella pneumoniae" \
        --antibiotics meropenem ciprofloxacin gentamicin ceftriaxone

Then download the genomes (FTPS one-liner from the BV-BRC docs):
    while read i; do wget -qN "ftps://ftp.bv-brc.org/genomes/$i/$i.fna" \
        -P data/genomes; done < data/genome_list.txt
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


# BV-BRC column names differ between the website export, the FTP bulk file, and
# the API. Normalise, then resolve each field from a list of candidates.
def _norm(col: str) -> str:
    return col.strip().lower().replace(" ", "_")


_FIELDS = {
    "genome_id": ["genome_id"],
    "genome_name": ["genome_name"],
    "antibiotic": ["antibiotic"],
    "phenotype": ["resistant_phenotype", "phenotype"],
    "lab_method": ["laboratory_typing_method", "laboratory_typing_method_version"],
    "evidence": ["evidence"],
    "measurement": ["measurement", "measurement_value"],
}


def _resolve(cols: list[str]) -> dict[str, str]:
    norm_to_orig = { _norm(c): c for c in cols }
    resolved = {}
    for canon, candidates in _FIELDS.items():
        for cand in candidates:
            if cand in norm_to_orig:
                resolved[canon] = norm_to_orig[cand]
                break
    return resolved


def load_and_filter(
    amr_tsv: Path,
    species: str,
    antibiotics: list[str],
    keep_intermediate: bool = False,
) -> pd.DataFrame:
    df = pd.read_csv(amr_tsv, sep="\t", dtype=str, low_memory=False)
    cols = _resolve(list(df.columns))
    missing = {"genome_id", "antibiotic", "phenotype"} - set(cols)
    if missing:
        raise ValueError(
            f"Could not find required columns {missing}. "
            f"Available: {list(df.columns)}"
        )

    g = df.rename(columns={v: k for k, v in cols.items()})

    # 1) species filter (substring match on genome_name if present)
    if "genome_name" in g.columns and species:
        g = g[g["genome_name"].str.contains(species, case=False, na=False)]

    # 2) antibiotic filter (case-insensitive)
    abx_lower = {a.lower() for a in antibiotics}
    g["antibiotic"] = g["antibiotic"].str.strip().str.lower()
    g = g[g["antibiotic"].isin(abx_lower)]

    # 3) LAB-MEASURED ONLY — the crucial honesty filter.
    #    Require a real laboratory typing method; drop predicted/model rows.
    if "lab_method" in g.columns:
        g = g[g["lab_method"].notna() & (g["lab_method"].str.strip() != "")]
    if "evidence" in g.columns:
        # keep laboratory/panel evidence; drop anything that looks predicted
        ev = g["evidence"].str.lower().fillna("")
        g = g[~ev.str.contains("predict|computational|in silico|model", regex=True)]

    # 4) phenotype -> binary label
    pheno = g["phenotype"].str.strip().str.lower()
    mapping = {"resistant": "R", "susceptible": "S"}
    if keep_intermediate:
        mapping["intermediate"] = "R"  # conservative: treat as failure
    g["label"] = pheno.map(mapping)
    g = g.dropna(subset=["label"])

    return g[["genome_id", "antibiotic", "label"]]


def collapse_conflicts(labels: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """A (genome, antibiotic) pair can appear in multiple tests. If they all
    agree, keep the label; if they conflict, DROP the pair (safer than guessing)
    and report how many were dropped so you can justify it.
    """
    agg = (labels.groupby(["genome_id", "antibiotic"])["label"]
           .agg(lambda s: s.iloc[0] if s.nunique() == 1 else "CONFLICT")
           .reset_index())
    n_conflict = int((agg["label"] == "CONFLICT").sum())
    clean = agg[agg["label"] != "CONFLICT"].reset_index(drop=True)
    return clean, n_conflict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amr-tsv", required=True, type=Path)
    ap.add_argument("--species", required=True)
    ap.add_argument("--antibiotics", nargs="+", required=True)
    ap.add_argument("--outdir", type=Path, default=Path("data"))
    ap.add_argument("--keep-intermediate", action="store_true")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    raw = load_and_filter(args.amr_tsv, args.species, args.antibiotics,
                          args.keep_intermediate)
    clean, n_conflict = collapse_conflicts(raw)

    clean.to_csv(args.outdir / "labels.csv", index=False)
    ids = sorted(clean["genome_id"].unique())
    (args.outdir / "genome_list.txt").write_text("\n".join(ids) + "\n")

    print(f"Species: {args.species}")
    print(f"Antibiotics kept: {sorted(clean['antibiotic'].unique())}")
    print(f"Lab-measured labels: {len(clean)} "
          f"across {len(ids)} genomes "
          f"(dropped {n_conflict} conflicting genome-antibiotic pairs)")
    print("Class balance per drug:")
    print(clean.groupby(["antibiotic", "label"]).size().unstack(fill_value=0))
    print(f"\nWrote {args.outdir/'labels.csv'} and {args.outdir/'genome_list.txt'}")
    print("Next: download FASTAs, then run src/features.py and src/dedup.py.")


if __name__ == "__main__":
    main()
