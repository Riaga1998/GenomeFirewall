"""
De-duplication & grouped split (the honesty engine)
===================================================
Near-identical genomes leaking across train/test is THE classic way to inflate
AMR-prediction scores. We sketch each genome with sourmash MinHash, cluster by
sequence similarity, and hand the cluster ids to GroupKFold so related strains
stay on the same side of the split.

The same signatures power the OOD no-call in decision.py (distance-to-training).
"""
from __future__ import annotations
import pickle
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


def sketch_genomes(fasta_dir: Path, ksize: int, scaled: int) -> dict[str, "MinHash"]:
    """One MinHash signature per FASTA. Returns {genome_id: MinHash}."""
    import sourmash

    sigs: dict = {}
    for fasta in sorted(Path(fasta_dir).glob("*.fna")):
        mh = sourmash.MinHash(n=0, ksize=ksize, scaled=scaled)
        for line in open(fasta):
            if not line.startswith(">"):
                mh.add_sequence(line.strip(), force=True)
        sigs[fasta.stem] = mh
    return sigs


def save_signatures(sigs: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(sigs, fh)


def load_signatures(path: Path) -> dict:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def jaccard_matrix(sigs: dict) -> tuple[list[str], np.ndarray]:
    """Symmetric pairwise Jaccard similarity matrix over genomes."""
    ids = list(sigs)
    n = len(ids)
    J = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            s = sigs[ids[i]].jaccard(sigs[ids[j]])
            J[i, j] = J[j, i] = s
    return ids, J


def cluster_into_groups(
    ids: list[str], J: np.ndarray, jaccard_threshold: float
) -> dict[str, int]:
    """Single-linkage clustering. Genomes with Jaccard >= threshold merge.

    Returns {genome_id: group_id}. Group ids are what GroupKFold consumes.
    """
    if len(ids) == 1:
        return {ids[0]: 0}
    dist = 1.0 - J
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="single")
    cut = 1.0 - jaccard_threshold
    labels = fcluster(Z, t=cut, criterion="distance")
    return {gid: int(lbl) for gid, lbl in zip(ids, labels)}


def max_containment_to_reference(query, reference_sigs: dict) -> float:
    """Best containment of a query MinHash within any reference signature.

    Used for the OOD no-call: low value => genome is unlike the training set.
    """
    if not reference_sigs:
        return 0.0
    return max(query.contained_by(ref) for ref in reference_sigs.values())


if __name__ == "__main__":
    import yaml

    cfg = yaml.safe_load(open("config/config.yaml"))
    d = cfg["dedup"]
    sigs = sketch_genomes(Path(cfg["paths"]["fasta_dir"]), d["ksize"], d["scaled"])
    save_signatures(sigs, Path(cfg["paths"]["signatures"]))
    ids, J = jaccard_matrix(sigs)
    groups = cluster_into_groups(ids, J, d["jaccard_threshold"])
    import pandas as pd

    out = pd.DataFrame(
        {"genome_id": list(groups), "group_id": list(groups.values())}
    )
    out.to_csv(cfg["paths"]["groups"], index=False)
    n_groups = out["group_id"].nunique()
    print(f"{len(ids)} genomes collapsed into {n_groups} groups "
          f"(Jaccard>={d['jaccard_threshold']})")
