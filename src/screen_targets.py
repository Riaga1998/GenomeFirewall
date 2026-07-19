"""
Target screening — pick species + antibiotics from the DATA, not from vibes
==========================================================================
Given a BV-BRC AMR table (website "AMR Phenotypes" download, FTP bulk file, or
API export), this ranks every species-antibiotic combination by how usable it is
for a trustworthy classifier:

    n_genomes     enough samples to learn + hold out a grouped test set
    pct_resistant class balance (avoid 95/5 combos that reward a lazy majority
                  classifier and destroy PR-AUC)

Run this FIRST, before download_bvbrc.py, to choose your target. It only counts
LAB-MEASURED results (drops predicted / computational phenotypes).

Usage:
    python src/screen_targets.py --amr-tsv path/to/bvbrc_amr.tsv --min-n 450
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def _norm(c: str) -> str:
    return c.strip().lower().replace(" ", "_")


def _resolve(cols):
    m = {_norm(c): c for c in cols}
    def pick(*cands):
        for c in cands:
            if c in m:
                return m[c]
        return None
    return {
        "genome_id": pick("genome_id"),
        "genome_name": pick("genome_name"),
        "antibiotic": pick("antibiotic"),
        "phenotype": pick("resistant_phenotype", "phenotype"),
        "lab_method": pick("laboratory_typing_method"),
        "evidence": pick("evidence"),
    }


def _species_of(name: str) -> str:
    """First two tokens of genome_name ~= 'Genus species'. Good enough to screen."""
    parts = str(name).split()
    return " ".join(parts[:2]) if len(parts) >= 2 else str(name)


def screen(amr_tsv: Path, min_n: int, balance_lo: float, balance_hi: float):
    df = pd.read_csv(amr_tsv, sep="\t", dtype=str, low_memory=False)
    col = _resolve(df.columns)
    for req in ("genome_id", "antibiotic", "phenotype"):
        if col[req] is None:
            raise ValueError(f"Missing column '{req}'. Have: {list(df.columns)}")

    g = df.rename(columns={v: k for k, v in col.items() if v})

    # lab-measured only
    if "lab_method" in g:
        g = g[g["lab_method"].notna() & (g["lab_method"].str.strip() != "")]
    if "evidence" in g:
        ev = g["evidence"].str.lower().fillna("")
        g = g[~ev.str.contains("predict|computational|in silico|model", regex=True)]

    g["label"] = g["phenotype"].str.strip().str.lower().map(
        {"resistant": 1, "susceptible": 0})
    g = g.dropna(subset=["label"])
    g["species"] = g["genome_name"].map(_species_of) if "genome_name" in g else "?"
    g["antibiotic"] = g["antibiotic"].str.strip().str.lower()

    # one label per genome-antibiotic (drop conflicts) before counting
    pair = (g.groupby(["species", "antibiotic", "genome_id"])["label"]
            .agg(lambda s: s.iloc[0] if s.nunique() == 1 else -1)
            .reset_index())
    pair = pair[pair["label"] != -1]

    rows = []
    for (sp, ab), sub in pair.groupby(["species", "antibiotic"]):
        n = len(sub)
        pct_r = float(sub["label"].mean())
        usable = (n >= min_n) and (balance_lo <= pct_r <= balance_hi)
        rows.append({"species": sp, "antibiotic": ab, "n_genomes": n,
                     "pct_resistant": round(pct_r, 3), "usable": usable})

    out = pd.DataFrame(rows).sort_values(
        ["usable", "n_genomes"], ascending=[False, False]).reset_index(drop=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amr-tsv", required=True, type=Path)
    ap.add_argument("--min-n", type=int, default=450)
    ap.add_argument("--balance-lo", type=float, default=0.15)
    ap.add_argument("--balance-hi", type=float, default=0.85)
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    out = screen(args.amr_tsv, args.min_n, args.balance_lo, args.balance_hi)
    pd.set_option("display.max_rows", None)
    print(out.head(args.top).to_string(index=False))

    usable = out[out["usable"]]
    print(f"\n{len(usable)} usable combos "
          f"(n>={args.min_n}, {args.balance_lo:.0%}-{args.balance_hi:.0%} resistant).")
    if len(usable):
        best_sp = (usable.groupby("species")["antibiotic"].count()
                   .sort_values(ascending=False))
        print("\nSpecies with the most usable antibiotics (your best target):")
        print(best_sp.head(5).to_string())
        print("\nPick the top species, then take its usable antibiotics as your 3-5.")


if __name__ == "__main__":
    main()
