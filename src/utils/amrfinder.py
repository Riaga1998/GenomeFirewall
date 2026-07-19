"""AMRFinderPlus wrapper and output parsing.

AMRFinderPlus (NCBI) is the default annotation tool for this challenge. It takes an
assembled nucleotide FASTA and reports antimicrobial-resistance genes and
resistance-associated point mutations.

If AMRFinderPlus is not installed, `is_available()` returns False and callers should
fall back to precomputed results supplied with the challenge dataset.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

AMRFINDER_BIN = "amrfinder"

# Columns we need, and every header AMRFinderPlus has used for them.
#
# Version 4 renamed most of these ("Gene symbol" -> "Element symbol", "Element type" ->
# "Type", and the "% ... reference sequence" pair lost its trailing word). Accepting both
# spellings keeps this working against a live v4 install and against precomputed v3
# output, which the challenge dataset may ship.
_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "protein_id": ("Protein id", "Protein identifier"),
    "contig_id": ("Contig id",),
    "gene_symbol": ("Element symbol", "Gene symbol"),
    "sequence_name": ("Element name", "Sequence name"),
    "element_type": ("Type", "Element type"),
    "element_subtype": ("Subtype", "Element subtype"),
    "drug_class": ("Class",),
    "drug_subclass": ("Subclass",),
    "coverage": ("% Coverage of reference", "% Coverage of reference sequence"),
    "identity": ("% Identity to reference", "% Identity to reference sequence"),
}


@dataclass(frozen=True)
class AMRHit:
    """One resistance determinant detected in a genome."""

    gene_symbol: str
    element_type: str
    element_subtype: str
    drug_class: str
    drug_subclass: str
    identity: float
    coverage: float
    contig_id: str

    @property
    def is_point_mutation(self) -> bool:
        return self.element_subtype.upper() == "POINT"


def find_binary() -> str | None:
    """Locate the amrfinder executable.

    Checks PATH first, then common conda environment locations. The app is normally
    launched from the project virtualenv, which does not inherit a conda env's PATH, so
    without this the FASTA path would appear unavailable on a machine where the tool is
    installed and working.
    """
    on_path = shutil.which(AMRFINDER_BIN)
    if on_path:
        return on_path

    if env_override := os.environ.get("AMRFINDER_PATH"):
        candidate = Path(env_override)
        if candidate.is_file():
            return str(candidate)

    roots = [
        Path.home() / "miniconda3" / "envs",
        Path.home() / "anaconda3" / "envs",
        Path.home() / "miniforge3" / "envs",
        Path("/opt/anaconda3/envs"),
        Path("/opt/miniconda3/envs"),
        Path("/opt/homebrew/Caskroom/miniconda/base/envs"),
    ]
    for root in roots:
        if not root.is_dir():
            continue
        for env_dir in sorted(root.iterdir()):
            candidate = env_dir / "bin" / AMRFINDER_BIN
            if candidate.is_file():
                return str(candidate)
    return None


def is_available() -> bool:
    """True if the amrfinder binary can be located."""
    return find_binary() is not None


def run_amrfinder(
    fasta_path: str | Path,
    organism: str | None = None,
    threads: int = 4,
    timeout: int = 900,
) -> pd.DataFrame:
    """Run AMRFinderPlus on one assembled genome and return its report as a DataFrame.

    `organism` enables species-specific point-mutation screening (e.g. "Escherichia",
    "Klebsiella_pneumoniae"). Without it AMRFinderPlus only reports acquired genes,
    which silently drops every resistance mechanism that comes from a chromosomal SNP.
    """
    fasta_path = Path(fasta_path)
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA not found: {fasta_path}")

    binary = find_binary()
    if binary is None:
        raise RuntimeError(
            "amrfinder not found. Install it (conda install -c bioconda "
            "ncbi-amrfinderplus && amrfinder -u), set AMRFINDER_PATH to the binary, or "
            "use precomputed results."
        )

    with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False) as tmp:
        out_path = Path(tmp.name)

    cmd = [
        binary,
        "--nucleotide", str(fasta_path),
        "--output", str(out_path),
        "--threads", str(threads),
        "--plus",
    ]
    if organism:
        cmd += ["--organism", organism]

    # AMRFinderPlus resolves its database relative to CONDA_PREFIX. When invoked from
    # another virtualenv that variable points elsewhere, so pin it to the env the
    # binary actually lives in.
    env = dict(os.environ)
    env["CONDA_PREFIX"] = str(Path(binary).parent.parent)

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout, env=env)
        return _read_report(out_path)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"AMRFinderPlus failed on {fasta_path.name}: {exc.stderr}") from exc
    finally:
        out_path.unlink(missing_ok=True)


def _read_report(path: Path) -> pd.DataFrame:
    return normalize_report(pd.read_csv(path, sep="\t", dtype=str).fillna(""))


def normalize_report(df: pd.DataFrame) -> pd.DataFrame:
    """Rename AMRFinderPlus columns to our short names and coerce numeric fields.

    Accepts a report read from disk or a slice of a pooled multi-genome TSV, so the
    live and precomputed paths share one parser. Raises if the essential columns are
    missing under any known spelling — a silently empty result would read as
    "no resistance detected", which is the most dangerous way this could fail.
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    rename: dict[str, str] = {}
    for canonical, candidates in _COLUMN_ALIASES.items():
        match = next((c for c in candidates if c in df.columns), None)
        if match is not None:
            rename[match] = canonical

    df = df[list(rename)].rename(columns=rename)

    missing = {"gene_symbol", "element_type"} - set(df.columns)
    if missing:
        raise ValueError(
            f"AMRFinderPlus report is missing required column(s): {sorted(missing)}. "
            f"Known headers: {sorted(c for v in _COLUMN_ALIASES.values() for c in v)}. "
            "An unrecognized schema would drop every hit and look like a clean genome."
        )

    for numeric in ("coverage", "identity"):
        if numeric in df.columns:
            df[numeric] = pd.to_numeric(df[numeric], errors="coerce")
    return df


def parse_report(df: pd.DataFrame, min_identity: float = 90.0, min_coverage: float = 90.0) -> list[AMRHit]:
    """Convert an AMRFinderPlus report into filtered AMRHit records.

    Low-identity partial matches are the main source of false-positive gene calls, so
    they are dropped by default. Point mutations are exempt from the coverage filter
    because AMRFinderPlus reports them against a short reference window.
    """
    has_identity = "identity" in df.columns
    has_coverage = "coverage" in df.columns

    hits: list[AMRHit] = []
    for row in df.to_dict("records"):
        element_type = str(row.get("element_type", ""))
        if element_type.upper() != "AMR":
            continue

        subtype = str(row.get("element_subtype", ""))
        is_point = subtype.upper() == "POINT"

        # When a report omits identity/coverage we keep the hit rather than treat the
        # missing value as zero, which would silently discard every call.
        identity = float(row.get("identity") or 0.0) if has_identity else float("nan")
        coverage = float(row.get("coverage") or 0.0) if has_coverage else float("nan")

        if has_identity and identity < min_identity:
            continue
        if has_coverage and not is_point and coverage < min_coverage:
            continue

        hits.append(
            AMRHit(
                gene_symbol=str(row.get("gene_symbol", "")).strip(),
                element_type=element_type,
                element_subtype=subtype,
                drug_class=str(row.get("drug_class", "")).strip().upper(),
                drug_subclass=str(row.get("drug_subclass", "")).strip().upper(),
                identity=identity,
                coverage=coverage,
                contig_id=str(row.get("contig_id", "")).strip(),
            )
        )
    return hits


# AMRFinderPlus --organism values, keyed by the species names this project uses.
#
# Supplying this flag is what enables point-mutation detection. Without it the tool
# reports acquired genes only — and for S. aureus ciprofloxacin, where resistance is
# stepwise gyrA/grlA mutation rather than an acquired gene, that means the real
# mechanism is invisible and every isolate looks like it has no quinolone signal.
#
# The names are not derivable from the species string: E. coli is "Escherichia", and
# Salmonella and Campylobacter are genus-level. Verified against `amrfinder -l`
# (database 2026-05-15.1).
ORGANISM_FLAGS: dict[str, str] = {
    "acinetobacter baumannii": "Acinetobacter_baumannii",
    "campylobacter": "Campylobacter",
    "enterococcus faecalis": "Enterococcus_faecalis",
    "enterococcus faecium": "Enterococcus_faecium",
    "escherichia coli": "Escherichia",
    "klebsiella pneumoniae": "Klebsiella_pneumoniae",
    "neisseria gonorrhoeae": "Neisseria_gonorrhoeae",
    "pseudomonas aeruginosa": "Pseudomonas_aeruginosa",
    "salmonella enterica": "Salmonella",
    "staphylococcus aureus": "Staphylococcus_aureus",
    "staphylococcus epidermidis": "Staphylococcus_epidermidis",
    "streptococcus agalactiae": "Streptococcus_agalactiae",
    "streptococcus pneumoniae": "Streptococcus_pneumoniae",
    "streptococcus pyogenes": "Streptococcus_pyogenes",
}


def organism_flag(species: str | None) -> str | None:
    """Map a species name to its --organism value, or None if unsupported.

    An unsupported species is not an error: annotation still runs and finds acquired
    genes. Only point mutations are unavailable, and the report says so rather than
    presenting their absence as evidence of susceptibility.
    """
    if not species:
        return None
    return ORGANISM_FLAGS.get(species.strip().lower())
