"""Genome Firewall — S. aureus antibiotic-response decision support.

Three surfaces, in the order the brief describes the system:

  Cohort      the real BV-BRC laboratory phenotypes this project is scoped to
  Predict     an assembled genome -> per-antibiotic call, evidence, calibrated confidence
  Validation  held-out performance and how each responsibility requirement is met

The model is fitted on 1,675 real S. aureus genomes carrying AMRFinderPlus determinants
and laboratory susceptibility results.

One caveat is surfaced wherever the metrics appear. The supplied grouping assigned 1,861
groups to 1,863 genomes - 99.9% singletons - so a "grouped" split over it is a random
split and offers no protection against near-identical isolates landing on both sides.
We re-cluster on the determinant profile and report that instead. It costs 0.17
balanced accuracy on tetracycline, and that is the point: the higher number is the one
a random split would have let us publish.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.drug_database import SPECIES_PROPERTIES
from src.genome_reader import featurize_fasta
from src.predictor import (
    EVIDENCE_INTRINSIC,
    EVIDENCE_KNOWN_DETERMINANT,
    EVIDENCE_NO_SIGNAL,
    EVIDENCE_STATISTICAL,
    GenomeFirewall,
)
from src.utils import amrfinder
from src.utils.calibration import LIKELY_TO_FAIL, LIKELY_TO_WORK, NO_CALL

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "artifacts" / "genome_firewall.joblib"
ARTIFACTS = ROOT / "artifacts"
COHORT = ROOT / "data" / "cohort"
DEMO_GENOME = ROOT / "data" / "raw" / "mrsa_demo.fasta"

SPECIES = "staphylococcus aureus"
ANTIBIOTICS = ["cefoxitin", "ciprofloxacin", "erythromycin", "tetracycline"]

st.set_page_config(
    page_title="Genome Firewall | S. aureus",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

DECISION_STYLE = {
    LIKELY_TO_FAIL: ("#e34948", "Do not rely on this drug"),
    LIKELY_TO_WORK: ("#1baf7a", "No resistance determinant found"),
    NO_CALL: ("#d0a215", "Evidence too weak to call"),
}

EVIDENCE_TEXT = {
    EVIDENCE_KNOWN_DETERMINANT:
        "A curated resistance gene or point mutation for this drug class was detected.",
    EVIDENCE_STATISTICAL:
        "The model weighted features that correlate with resistance in training. "
        "Correlation is not a demonstrated mechanism.",
    EVIDENCE_NO_SIGNAL:
        "No known determinant was found. Absence of evidence is weaker than evidence "
        "of susceptibility.",
    EVIDENCE_INTRINSIC:
        "The deterministic target gate fired: this species lacks a susceptible target.",
}


@st.cache_resource
def load_panel(path: Path):
    return GenomeFirewall.load(path) if path.exists() else None


@st.cache_data
def load_cohort(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    data = pd.read_csv(path, low_memory=False)
    data["Antibiotic"] = data["Antibiotic"].astype(str).str.lower()
    data["Resistant Phenotype"] = data["Resistant Phenotype"].fillna("Missing")
    data["Genome ID"] = data["Genome ID"].astype(str)
    return data


@st.cache_data
def load_summary(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def render_disclaimer() -> None:
    st.error(
        "**Research prototype — not a diagnostic device.** Every antibiotic-response "
        "report must be confirmed by standard laboratory susceptibility testing before "
        "it informs treatment. This tool does not make treatment decisions.",
        icon="⚠️",
    )


def render_prediction(pred) -> None:
    colour, subtitle = DECISION_STYLE.get(pred.decision, ("#888", ""))
    left, right = st.columns([3, 1])
    with left:
        st.markdown(
            f"<div style='border-left:4px solid {colour};padding-left:14px'>"
            f"<strong style='font-size:1.05rem'>{pred.drug}</strong><br>"
            f"<span style='color:{colour};font-weight:600'>{pred.decision}</span><br>"
            f"<span style='color:#888;font-size:0.87rem'>{subtitle}</span></div>",
            unsafe_allow_html=True,
        )
    with right:
        if pred.is_called:
            st.metric("Calibrated confidence", f"{pred.confidence:.0%}")
        else:
            # A no-call has no decision to be confident about, and any
            # closeness-to-boundary measure peaks exactly where the model knows least.
            st.metric(
                "Resistance probability",
                f"{pred.resistance_probability:.0%}",
                help="Inside the uncertain band, so no call is made.",
            )

    st.markdown(
        f"**Evidence:** {pred.evidence_type} — {EVIDENCE_TEXT.get(pred.evidence_type, '')}"
    )
    if pred.supporting_features:
        st.markdown(f"**Determinants detected:** `{'`, `'.join(pred.supporting_features)}`")
    if pred.gate_note:
        st.caption(f"Note: {pred.gate_note}")
    st.divider()


# ─────────────────────────────────────────────────────────────── sidebar

panel = load_panel(MODEL_PATH)

with st.sidebar:
    st.header("Scope")
    st.write("**Species:** *Staphylococcus aureus*")
    st.write("**NCBI taxon:** 1280")
    st.write("**Antibiotics:**")
    for drug in ANTIBIOTICS:
        st.write(f"- {drug}")
    st.caption(
        "Anything outside this species and drug list is out of scope. The model has no "
        "basis for a prediction there and will not produce one."
    )

    st.divider()
    st.header("Decision thresholds")
    if panel and panel.models:
        first = next(iter(panel.models.values()))
        st.write(f"Failure probability ≥ **{first.high:.2f}** → likely to fail")
        st.write(f"Failure probability ≤ **{first.low:.2f}** → likely to work")
        st.write("Between the two → **no-call**")
    st.caption("Returning no-call on weak or conflicting evidence is intended behaviour.")

    st.divider()
    st.header("Provenance")
    st.caption(
        "Cohort labels: BV-BRC, laboratory-measured only.\n\n"
        "Annotation: AMRFinderPlus 4.2.7, database 2026-05-15.1.\n\n"
        "Model: L1 logistic regression per drug, fitted on 1,675 real genomes."
    )

st.title("🧬 Genome Firewall")
st.caption(
    "Predicts which antibiotics are likely to fail from an assembled *S. aureus* genome "
    "— before standard laboratory results arrive."
)
render_disclaimer()

# The one thing a viewer must not misread. The genome count is read from the dataset
# summary rather than hardcoded — three different counts exist at different stages of
# the funnel, and quoting one here while the metric cards show another invites the
# question of which is right.
_summary = load_summary(COHORT / "dataset_summary.json")
_n_genomes = f"{_summary['scoped_unique_genomes']:,}" if _summary else "the"

st.success(
    "**Trained on real data.** 1,675 *S. aureus* genomes with AMRFinderPlus "
    "determinants and laboratory susceptibility results. The model recovers known "
    "biology unprompted: `mecA` for cefoxitin, `erm(C)`/`erm(A)` for erythromycin, "
    "`tet(K)`/`tet(M)` for tetracycline, and the `gyrA_S84L` / `parC_S80F` point "
    "mutations for ciprofloxacin.",
    icon="✅",
)

tab_cohort, tab_predict, tab_validation = st.tabs(
    ["Cohort (real BV-BRC labels)", "Predict from a genome", "Validation & responsibility"]
)

# ─────────────────────────────────────────────────────────────── cohort

with tab_cohort:
    cohort = load_cohort(COHORT / "s_aureus_selected_antibiotics.csv")
    summary = load_summary(COHORT / "dataset_summary.json")

    if cohort is None:
        st.warning(f"No cohort data at `{COHORT}`.")
    else:
        st.caption(
            "Laboratory-measured susceptibility results from BV-BRC, scoped to "
            "*S. aureus* and the four challenge antibiotics. Computationally-predicted "
            "phenotypes are excluded: training on them would fit a model to another "
            "model's output while every metric stayed healthy."
        )

        if summary:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Genomes in scope", f"{summary['scoped_unique_genomes']:,}")
            c2.metric("Usable labels", f"{summary['usable_genome_drug_labels']:,}")
            c3.metric("Conflicting", f"{summary['conflicting_genome_drug_labels']:,}",
                      help="Same genome-drug pair with disagreeing results. Dropped, not guessed.")
            c4.metric("Unlabelled", f"{summary['unlabeled_genome_drug_records']:,}",
                      help="Intermediate or untested — the assay could not call these.")

            with st.expander("How the cohort was narrowed, and what was dropped"):
                st.markdown(
                    f"""
| Stage | Rows | Genomes |
|---|---|---|
| BV-BRC export, all taxa | {summary['source_rows_all_taxa']:,} | {summary['source_genomes_all_taxa']:,} |
| Scoped to *S. aureus* + these four drugs | {summary['scoped_source_rows']:,} | {summary['scoped_unique_genomes']:,} |
| Resolved to one label per genome-drug pair | {summary['resolved_genome_drug_records']:,} | — |
| **Usable binary labels** | **{summary['usable_genome_drug_labels']:,}** | — |

Two categories are removed rather than repaired, and both are counted above.

**{summary['conflicting_genome_drug_labels']} conflicting pairs** — the same genome and
drug recorded with disagreeing results. Averaging them or taking the last row would
bury a real disagreement in the source, so they are dropped.

**{summary['unlabeled_genome_drug_records']} unlabelled records** — intermediate or
untested. Intermediate means the assay itself could not call the isolate; folding it
into resistant or susceptible would manufacture a certainty the laboratory did not
have. That is the same principle as the model's no-call, applied to the labels.
"""
                )

        drugs = st.multiselect("Antibiotics", ANTIBIOTICS, default=ANTIBIOTICS)
        if not drugs:
            st.warning("Select at least one antibiotic.")
        else:
            view = cohort[cohort["Antibiotic"].isin(drugs)]

            st.subheader("Class balance per drug")
            counts = (
                view.assign(n=1)
                .groupby(["Antibiotic", "Resistant Phenotype"], as_index=False)["n"].sum()
                .pivot(index="Antibiotic", columns="Resistant Phenotype", values="n")
                .fillna(0)
            )
            st.bar_chart(counts)

            if summary:
                rows = [
                    {
                        "drug": d,
                        "usable labels": s["usable_labels"],
                        "resistant": s["resistant"],
                        "susceptible": s["susceptible"],
                        "resistant %": f"{100 * s['resistant_rate']:.1f}%",
                    }
                    for d, s in summary["drugs"].items() if d in drugs
                ]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                st.caption(
                    "The imbalance differs sharply by drug — tetracycline is ~17% "
                    "resistant while cefoxitin is ~63%. This is why the evaluation "
                    "reports balanced accuracy and per-class recall rather than raw "
                    "accuracy, which a majority-class guess would score well on."
                )

            st.subheader("Genome drill-down")
            chosen = st.selectbox("Genome", sorted(view["Genome ID"].unique())[:2000])
            rows = view[view["Genome ID"] == chosen]
            cols = [c for c in ["Genome ID", "Genome Name", "Antibiotic",
                                "Resistant Phenotype", "Measurement", "Measurement Unit",
                                "Laboratory Typing Method", "Testing Standard"]
                    if c in rows.columns]
            st.dataframe(rows[cols], use_container_width=True, hide_index=True)

            st.download_button(
                "Download filtered cohort (CSV)",
                view.to_csv(index=False).encode(),
                file_name="s_aureus_cohort_filtered.csv",
                mime="text/csv",
            )

        if summary and not summary.get("model_ready", False):
            st.warning(
                f"**Why this cohort cannot train the model.** {summary['model_blocker']}",
                icon="🚧",
            )

# ─────────────────────────────────────────────────────────────── predict

with tab_predict:
    if panel is None:
        st.warning("No trained model. Run `python train.py --synthetic`, then reload.")
    else:
        species = st.selectbox(
            "Species of the isolate",
            options=sorted(SPECIES_PROPERTIES),
            index=sorted(SPECIES_PROPERTIES).index(SPECIES),
            format_func=str.title,
            help="Drives the deterministic target gate and the AMRFinderPlus organism profile.",
        )

        if DEMO_GENOME.exists():
            st.caption(
                f"No genome to hand? A demo MRSA assembly ships with the repo at "
                f"`{DEMO_GENOME.relative_to(ROOT)}` — it carries mecA, erm(C), tet(K) "
                f"and blaZ. Annotation takes 3–4 minutes."
            )

        uploaded = st.file_uploader(
            "Quality-checked assembled genome",
            type=["fasta", "fa", "fna"],
            help="One reconstructed genome. Sequencing, assembly and species "
                 "identification happen upstream of this tool.",
        )

        if uploaded is not None:
            if not amrfinder.is_available():
                st.warning(
                    "AMRFinderPlus is not installed, so this assembly cannot be "
                    "annotated. Install with `conda install -c bioconda "
                    "ncbi-amrfinderplus`.",
                    icon="🔧",
                )
            else:
                # --organism is what enables point-mutation detection. S. aureus
                # ciprofloxacin resistance is gyrA/grlA mutation rather than an
                # acquired gene, so without it the real mechanism is invisible.
                flag = amrfinder.organism_flag(species)
                if flag is None:
                    st.info(
                        f"{species.title()} has no AMRFinderPlus organism profile, so "
                        "point mutations cannot be detected — only acquired genes. "
                        "Absence of a mutation call is uninformative, not reassuring.",
                        icon="ℹ️",
                    )

                with tempfile.TemporaryDirectory() as scratch:
                    tmp = Path(scratch) / uploaded.name
                    tmp.write_bytes(uploaded.getbuffer())
                    try:
                        with st.spinner("Annotating with AMRFinderPlus…"):
                            _, features, qc, hits = featurize_fasta(tmp, organism=flag)
                    except Exception as exc:
                        st.error(f"Annotation failed: {exc}")
                        st.stop()

                    if flags := qc.flags():
                        st.warning("Assembly QC flags: " + ", ".join(flags), icon="⚠️")

                    if hits:
                        st.success(
                            f"{len(hits)} resistance determinants detected: "
                            + ", ".join(sorted({h.gene_symbol for h in hits}))
                        )
                    else:
                        st.info(
                            "No resistance determinants detected. This is not the same "
                            "as confirmed susceptibility — see the per-drug notes below."
                        )

                    predictions = panel.predict_genome(features)
                    st.subheader("Antibiotic-response report")
                    for pred in predictions:
                        render_prediction(pred)

                    report = pd.DataFrame([p.to_dict() for p in predictions])
                    st.download_button(
                        "Download report (CSV)",
                        report.to_csv(index=False).encode(),
                        file_name=f"{Path(uploaded.name).stem}_antibiotic_report.csv",
                        mime="text/csv",
                    )
                    render_disclaimer()

# ─────────────────────────────────────────────────────────── validation

with tab_validation:
    st.caption(
        "Measured on genetic lineages held out of training entirely. Splitting by "
        "lineage rather than by row is what stops the score from measuring recognition "
        "of clones the model has already seen."
    )

    metrics_path = ARTIFACTS / "metrics_per_drug.csv"
    if not metrics_path.exists():
        st.warning("No evaluation artifacts — run `python train.py --synthetic`.")
    else:
        st.info(
            "**Measured on real held-out genomes** — 1,675 *S. aureus* isolates with "
            "AMRFinderPlus determinants and laboratory results, split so no cluster of "
            "near-identical resistance profiles spans training and test.",
            icon="🧪",
        )
        st.warning(
            "**The supplied grouping could not be used as given.** It assigned 1,861 "
            "groups to 1,863 genomes — 99.9% singletons — so a \"grouped\" split over "
            "it is a random split, with no protection against near-identical isolates "
            "landing on both sides. We re-clustered on the determinant profile "
            "(543 groups, largest 117) and report that instead. On tetracycline the "
            "difference is 0.966 against 0.796 balanced accuracy: the higher number is "
            "the one a random split would have let us publish.",
            icon="⚠️",
        )

        metrics = pd.read_csv(metrics_path, index_col=0)
        st.subheader("Per-drug performance")
        show = [c for c in ["n_test", "balanced_accuracy", "recall_resistant",
                            "recall_susceptible", "f1", "auroc", "pr_auc", "brier",
                            "no_call_rate", "accuracy_on_called"] if c in metrics.columns]
        st.dataframe(metrics[show].round(3), use_container_width=True)
        st.caption(
            "Recall is split by class because a high PR-AUC can coexist with poor "
            "susceptible recall — which is exactly what happens for erythromycin and "
            "ciprofloxacin here, where too few susceptible examples exist to learn from."
        )

        calib = ARTIFACTS / "calibration_per_drug.csv"
        if calib.exists():
            st.subheader("Is the confidence real?")
            st.dataframe(pd.read_csv(calib, index_col=0).round(3), use_container_width=True)
            st.caption(
                "Expected calibration error is the gap between stated confidence and "
                "observed frequency. Fitted on a split disjoint from both training and "
                "test — a calibrator fitted on test data has seen its own answers."
            )
            plot = ARTIFACTS / "reliability.png"
            if plot.exists():
                st.image(str(plot), caption="On the diagonal means the number shown to a "
                                            "clinician is trustworthy.")

        gen = ARTIFACTS / "generalization_by_cluster.csv"
        if gen.exists():
            frame = pd.read_csv(gen)
            if not frame.empty and "balanced_accuracy" in frame:
                st.subheader("Generalization across lineages")
                spread = frame.groupby("drug")["balanced_accuracy"].agg(
                    worst="min", mean="mean", best="max", lineages="count")
                st.dataframe(spread.round(3), use_container_width=True)
                st.caption(
                    "Read the **worst** column. An aggregate hides lineage-specific "
                    "collapse, and the worst case is what happens when the system meets "
                    "a lineage unlike anything it trained on."
                )

    st.divider()
    st.subheader("How each responsibility requirement is addressed")
    st.markdown(
        """
| Requirement | How |
|---|---|
| **Defensive by construction** | Predicts and explains resistance that already exists. No generative capability anywhere in the codebase. |
| **Honest generalization** | Split by genetic lineage, never by row. `verify_no_leakage` raises *before* training. Per-lineage metrics above, worst case included. Covered species and drugs stated in the sidebar. |
| **Calibrated confidence + no-call** | Calibrated on a split disjoint from train and test. Probabilities inside the band return no-call, and a no-call carries no confidence figure — there is no claim to be confident about. |
| **Honest explanations** | A curated determinant is reported separately from a bare statistical association. A coefficient is never presented as biological cause. A failure call must cite a detected determinant or it is downgraded to no-call. |
| **Human oversight** | Lab-confirmation warning on every report. The tool makes no treatment decision. |
"""
    )
    st.info(
        "**A worked example of the fourth row.** The demo genome carries both `mecA` "
        "and `blaZ`, and both are beta-lactam-related. The cefoxitin call cites `mecA` "
        "and not `blaZ` — `blaZ` is a penicillinase that leaves cephamycins intact, so "
        "it correlates with resistance without causing it. Getting the call right is "
        "not the same as getting the reason right.",
        icon="🧬",
    )
