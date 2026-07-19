"""Synthetic AMR cohort for pipeline verification.

The organizer dataset is not published until the event, so this generator produces data
with the same shape and the same statistical hazards: lineage structure (clonal
clusters), class imbalance, missing labels, and resistance driven by a few real
determinants plus noise.

This is scaffolding for testing the pipeline end-to-end. Every reported result in the
submission must come from the real challenge dataset — synthetic numbers prove the code
runs, not that the model works.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Determinants named after real AMR genes so the evidence output reads sensibly.
# Staphylococcus aureus resistance determinants.
#
# These are staphylococcal, not enterobacterial: no blaCTX-M, no qnr, no sul. Using an
# E. coli gene pool here would train the model on determinants that do not occur in the
# organism, and the resulting feature names would never match a real AMRFinderPlus report.
GENE_POOL = {
    "BETA-LACTAM": ["mecA", "mecC", "blaZ"],
    "LINCOSAMIDE/MACROLIDE/STREPTOGRAMIN": ["erm(A)", "erm(B)", "erm(C)", "msr(A)", "mph(C)"],
    "TETRACYCLINE": ["tet(K)", "tet(L)", "tet(M)"],
    "AMINOGLYCOSIDE": ["aac(6')-aph(2'')", "aph(3')-III", "ant(4')-Ia"],
    "TRIMETHOPRIM": ["dfrG", "dfrK"],
    "QUINOLONE": ["norA"],
}

POINT_POOL = {
    # S. aureus fluoroquinolone resistance is stepwise chromosomal mutation, not an
    # acquired gene. Topoisomerase IV here is grlA/grlB — the parC/parE homologue — so
    # the E. coli parC_S80I is the wrong name for this organism.
    "QUINOLONE": ["gyrA_S84L", "gyrA_E88K", "grlA_S80F", "grlA_S80Y"],
}

# Which determinants actually drive resistance for each drug, and how strongly.
#
# The four challenge drugs. Two deliberate pieces of real biology are encoded here,
# because they are exactly what a model should be able to recover:
#
#   * mecA (and mecC) drive cefoxitin. mecA encodes PBP2a, an alternative
#     penicillin-binding protein with low beta-lactam affinity — this is MRSA, and
#     cefoxitin is the surrogate the laboratory uses to call it.
#   * blaZ does NOT drive cefoxitin. It is a penicillinase: it hydrolyses penicillin
#     but leaves cephamycins intact. It is common in S. aureus, so it co-occurs with
#     resistance without causing it — a genuine confounder, and the sort of thing that
#     separates "known determinant" from "statistical association only".
DRUG_MECHANISMS = {
    "cefoxitin": {"class": "BETA-LACTAM", "drivers": {"mecA": 0.97, "mecC": 0.93}},
    "ciprofloxacin": {
        "class": "QUINOLONE",
        "drivers": {"gyrA_S84L": 0.82, "gyrA_E88K": 0.74, "grlA_S80F": 0.78,
                    "grlA_S80Y": 0.71, "norA": 0.35},
    },
    "erythromycin": {
        "class": "LINCOSAMIDE/MACROLIDE/STREPTOGRAMIN",
        "drivers": {"erm(A)": 0.93, "erm(B)": 0.90, "erm(C)": 0.92, "msr(A)": 0.76, "mph(C)": 0.55},
    },
    "tetracycline": {
        "class": "TETRACYCLINE",
        "drivers": {"tet(K)": 0.86, "tet(L)": 0.80, "tet(M)": 0.91},
    },
}


def generate_cohort(
    n_genomes: int = 1200,
    n_clusters: int = 40,
    drugs: list[str] | None = None,
    missing_label_rate: float = 0.15,
    background_resistance: float = 0.04,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Generate (features, labels, clusters).

    Determinant carriage is drawn per *cluster*, not per genome, which is what creates
    the leakage hazard the pipeline must defend against: genomes within a cluster look
    nearly identical, so a random split would let the model memorize them.

    Returns:
        features: genomes x binary determinant features (gene:/point:/class: prefixed)
        labels:   genomes x drugs, 1 = resistant, 0 = susceptible, NaN = not tested
        clusters: genome_id -> cluster_id
    """
    rng = np.random.default_rng(seed)
    drugs = drugs or list(DRUG_MECHANISMS)

    all_genes = [g for genes in GENE_POOL.values() for g in genes]
    all_points = [p for points in POINT_POOL.values() for p in points]

    gene_to_class = {g: cls for cls, genes in GENE_POOL.items() for g in genes}
    gene_to_class.update({p: cls for cls, points in POINT_POOL.items() for p in points})

    # Each cluster gets a carriage probability per determinant: mostly near 0 or near 1,
    # so members of a cluster share a profile the way a real clonal lineage does.
    cluster_profiles = {
        cid: {
            det: float(rng.beta(0.25, 0.25))
            for det in all_genes + all_points
        }
        for cid in range(n_clusters)
    }

    cluster_assignment = rng.integers(0, n_clusters, size=n_genomes)
    genome_ids = [f"GEN{idx:05d}" for idx in range(n_genomes)]

    feature_rows, label_rows = [], []
    for genome_id, cluster_id in zip(genome_ids, cluster_assignment):
        profile = cluster_profiles[int(cluster_id)]
        carried = {det: int(rng.random() < prob) for det, prob in profile.items()}

        features: dict[str, int] = {}
        for det, present in carried.items():
            if not present:
                continue
            prefix = "point" if det in all_points else "gene"
            features[f"{prefix}:{det}"] = 1
            features[f"class:{gene_to_class[det]}"] = 1

        labels: dict[str, float] = {}
        for drug in drugs:
            mechanism = DRUG_MECHANISMS[drug]
            # Resistance probability is the noisy-OR of the carried drivers, plus a
            # small background rate for mechanisms outside the annotated feature set.
            p_susceptible = 1.0 - background_resistance
            for driver, strength in mechanism["drivers"].items():
                if carried.get(driver, 0):
                    p_susceptible *= 1.0 - strength

            resistant = int(rng.random() < (1.0 - p_susceptible))
            labels[drug] = np.nan if rng.random() < missing_label_rate else float(resistant)

        feature_rows.append(features)
        label_rows.append(labels)

    features_df = pd.DataFrame(feature_rows, index=genome_ids).fillna(0).astype(int)
    features_df.index.name = "genome_id"
    features_df = features_df.sort_index(axis=1)

    labels_df = pd.DataFrame(label_rows, index=genome_ids)
    labels_df.index.name = "genome_id"

    clusters = pd.Series(cluster_assignment, index=genome_ids, name="cluster_id")
    clusters.index.name = "genome_id"

    return features_df, labels_df, clusters
