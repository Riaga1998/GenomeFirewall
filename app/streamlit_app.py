"""Genome Firewall — antibiotic-response report.

Decision support for a clinician or lab professional: upload an assembled genome, get a
per-antibiotic call with calibrated confidence and the evidence behind it. Every result
carries the requirement that it be confirmed by standard laboratory testing.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.drug_database import SPECIES_PROPERTIES
from src.genome_reader import featurize_fasta, hits_to_features
from src.predictor import (
    EVIDENCE_INTRINSIC,
    EVIDENCE_KNOWN_DETERMINANT,
    EVIDENCE_NO_SIGNAL,
    EVIDENCE_STATISTICAL,
    GenomeFirewall,
)
from src.utils import amrfinder
from src.utils.calibration import LIKELY_TO_FAIL, LIKELY_TO_WORK, NO_CALL

MODEL_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "genome_firewall.joblib"

DECISION_STYLE = {
    LIKELY_TO_FAIL: ("#d03b3b", "Do not rely on this drug"),
    LIKELY_TO_WORK: ("#0ca30c", "No resistance determinant found"),
    NO_CALL: ("#fab219", "Evidence too weak to call"),
}

EVIDENCE_HELP = {
    EVIDENCE_KNOWN_DETERMINANT: "A curated resistance gene or point mutation for this drug class was detected in the assembly.",
    EVIDENCE_STATISTICAL: "The model weighted features that correlate with resistance in training data. Correlation is not a demonstrated biological mechanism.",
    EVIDENCE_NO_SIGNAL: "No known determinant was found. Absence of evidence is weaker than evidence of susceptibility.",
    EVIDENCE_INTRINSIC: "The species lacks a susceptible target for this drug, independent of any acquired resistance gene.",
}

st.set_page_config(page_title="Genome Firewall", page_icon="🧬", layout="wide")


@st.cache_resource
def load_panel(path: Path) -> GenomeFirewall | None:
    return GenomeFirewall.load(path) if path.exists() else None


def render_disclaimer() -> None:
    st.error(
        "**Research prototype — not a diagnostic device.** Every antibiotic-response "
        "report below must be confirmed by standard laboratory susceptibility testing "
        "before it informs treatment. This tool does not make treatment decisions.",
        icon="⚠️",
    )


def render_prediction(pred) -> None:
    colour, plain_english = DECISION_STYLE[pred.decision]
    left, right = st.columns([3, 2])

    with left:
        st.markdown(
            f"<div style='border-left:4px solid {colour};padding:0.4rem 0 0.4rem 0.9rem;'>"
            f"<div style='font-size:1.05rem;font-weight:600;'>{pred.drug}</div>"
            f"<div style='color:{colour};font-weight:600;'>{pred.decision}</div>"
            f"<div style='color:#666;font-size:0.85rem;'>{plain_english}</div></div>",
            unsafe_allow_html=True,
        )

    with right:
        if pred.is_called:
            st.metric("Calibrated confidence", f"{pred.confidence:.0%}")
        else:
            # A no-call has no decision to be confident about. Showing the raw
            # probability and the band it fell into is the honest substitute.
            st.metric(
                "Resistance probability",
                f"{pred.resistance_probability:.0%}",
                help="Fell inside the uncertain band, so no call is made.",
            )

    st.caption(f"**Evidence:** {pred.evidence_type} — {EVIDENCE_HELP[pred.evidence_type]}")
    if pred.supporting_features:
        st.caption("**Determinants detected:** " + ", ".join(pred.supporting_features))
    if pred.gate_note:
        st.caption(f"**Note:** {pred.gate_note}")
    st.divider()


st.title("🧬 Genome Firewall")
st.caption(
    "Predicts which antibiotics are likely to fail from an assembled bacterial genome — "
    "before standard laboratory results arrive."
)
render_disclaimer()

panel = load_panel(MODEL_PATH)
if panel is None:
    st.warning(
        f"No trained model at `{MODEL_PATH.relative_to(MODEL_PATH.parent.parent)}`. "
        "Run `python train.py --synthetic` to build one, then reload this page."
    )
    st.stop()

with st.sidebar:
    st.header("Coverage")
    st.write(f"**Species trained on:** {panel.species.title()}")
    st.write(f"**Antibiotics covered:** {len(panel.models)}")
    for drug in panel.models:
        st.write(f"- {drug}")
    st.divider()
    st.caption(
        "Anything outside this species and drug list is out of scope. The model has no "
        "basis for a prediction there and will not produce one."
    )
    st.divider()
    st.header("Decision thresholds")
    first = next(iter(panel.models.values()))
    st.write(f"Failure probability ≥ **{first.high:.2f}** → likely to fail")
    st.write(f"Failure probability ≤ **{first.low:.2f}** → likely to work")
    st.write("Between the two → **no-call**")
    st.caption("Returning no-call on weak or conflicting evidence is intended behaviour.")

species = st.selectbox(
    "Species of the isolate",
    options=sorted(SPECIES_PROPERTIES),
    index=sorted(SPECIES_PROPERTIES).index(panel.species) if panel.species in SPECIES_PROPERTIES else 0,
    format_func=str.title,
    help="Used by the deterministic target gate, which rules out drugs the species is intrinsically resistant to.",
)

tab_upload, tab_manual, tab_performance = st.tabs([
    "Upload assembly (FASTA)",
    "Enter determinants manually",
    "Held-out performance",
])

# The closing disclaimer refers to "every report below", so it may only appear once a
# report actually exists. Rendered unconditionally it stacked directly under the opening
# disclaimer, pointing at nothing.
report_rendered = False

with tab_upload:
    uploaded = st.file_uploader(
        "Quality-checked assembled genome",
        type=["fasta", "fa", "fna"],
        help="One reconstructed genome. Sequencing, assembly, and species identification happen upstream of this tool.",
    )

    if uploaded is not None:
        if not amrfinder.is_available():
            st.warning(
                "AMRFinderPlus is not installed, so this assembly cannot be annotated here. "
                "Install it with `conda install -c bioconda ncbi-amrfinderplus`, or use the "
                "manual tab to enter determinants that were annotated elsewhere.",
                icon="🔧",
            )
        else:
            # Uploaded genomes go to a scratch directory that is removed on exit, not
            # into the working tree.
            with tempfile.TemporaryDirectory() as scratch:
                tmp_path = Path(scratch) / uploaded.name
                tmp_path.write_bytes(uploaded.getbuffer())

                # Passing --organism is what turns on point-mutation detection. For
                # S. aureus ciprofloxacin the mechanism is gyrA/grlA mutation rather
                # than an acquired gene, so without this the real cause is invisible
                # and the isolate looks like it carries no quinolone signal at all.
                flag = amrfinder.organism_flag(species)
                if flag is None:
                    st.info(
                        f"{species.title()} has no AMRFinderPlus organism profile, so "
                        "point mutations cannot be detected — only acquired genes. "
                        "Absence of a mutation call below is uninformative, not "
                        "reassuring.",
                        icon="ℹ️",
                    )

                try:
                    with st.spinner("Annotating with AMRFinderPlus…"):
                        _, features, qc, hits = featurize_fasta(tmp_path, organism=flag)
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
                        "No resistance determinants detected. This is not the same as "
                        "confirmed susceptibility — see the per-drug notes below."
                    )

                predictions = panel.predict_genome(features)
                st.subheader("Antibiotic-response report")
                for pred in predictions:
                    render_prediction(pred)
                report_rendered = True

                report = pd.DataFrame([p.to_dict() for p in predictions])
                st.download_button(
                    "Download report (CSV)",
                    report.to_csv(index=False).encode(),
                    file_name=f"{Path(uploaded.name).stem}_antibiotic_report.csv",
                    mime="text/csv",
                )

with tab_manual:
    st.caption(
        "For genomes already annotated elsewhere. Select the determinants AMRFinderPlus "
        "reported and the panel will score them."
    )
    determinants = [f for f in panel.feature_names_ if f.startswith(("gene:", "point:"))]
    classes = [f for f in panel.feature_names_ if f.startswith("class:")]

    chosen = st.multiselect(
        "Resistance determinants present",
        options=determinants,
        format_func=lambda f: f.split(":", 1)[1] + (" (point mutation)" if f.startswith("point:") else ""),
    )
    chosen_classes = st.multiselect(
        "Drug classes those determinants act against",
        options=classes,
        format_func=lambda f: f.split(":", 1)[1].title(),
        help=(
            "AMRFinderPlus reports a drug class alongside each determinant. The class "
            "rollup is what separates a known mechanism from a bare statistical "
            "association, so set it to match the annotation."
        ),
    )

    if st.button("Generate report", type="primary"):
        hit_features = {f: 1 for f in chosen + chosen_classes}
        predictions = panel.predict_genome(hit_features)
        st.subheader("Antibiotic-response report")
        for pred in predictions:
            render_prediction(pred)
        report_rendered = True

        report = pd.DataFrame([p.to_dict() for p in predictions])
        st.download_button(
            "Download report (CSV)",
            report.to_csv(index=False).encode(),
            file_name="antibiotic_report.csv",
            mime="text/csv",
        )

with tab_performance:
    # The brief asks that the responsibility requirements be shown on held-out data in
    # the demo, not only in the repository. A judge should be able to see how the model
    # was validated without reading the code.
    ARTIFACTS = MODEL_PATH.parent

    st.caption(
        "Measured on genetic lineages held out of training entirely. Splitting by "
        "lineage rather than by row is what stops the score from measuring "
        "recognition of clones the model has already seen."
    )

    metrics_path = ARTIFACTS / "metrics_per_drug.csv"
    if not metrics_path.exists():
        st.warning("No evaluation artifacts yet — run `python train.py --synthetic`.")
    else:
        metrics = pd.read_csv(metrics_path, index_col=0)

        st.warning(
            "**These numbers come from a synthetic cohort.** They verify the pipeline "
            "and its honesty properties. They say nothing about real-world "
            "performance: the model has not been fitted on real genomes.",
            icon="⚠️",
        )

        st.subheader("Per-drug performance")
        show = [c for c in [
            "n_test", "balanced_accuracy", "recall_resistant", "recall_susceptible",
            "f1", "auroc", "pr_auc", "brier", "no_call_rate", "accuracy_on_called",
        ] if c in metrics.columns]
        st.dataframe(metrics[show].round(3), use_container_width=True)
        st.caption(
            "Balanced accuracy and PR-AUC rather than raw accuracy: the cohort is "
            "heavily skewed toward resistance, so a model answering \"resistant\" every "
            "time would post a strong accuracy having learned nothing. Recall is shown "
            "separately for each class because a high PR-AUC can coexist with poor "
            "susceptible recall — which is exactly what happens here for erythromycin "
            "and ciprofloxacin, where too few susceptible examples exist to learn from."
        )

        calib_path = ARTIFACTS / "calibration_per_drug.csv"
        if calib_path.exists():
            st.subheader("Is the confidence real?")
            st.dataframe(pd.read_csv(calib_path, index_col=0).round(3),
                         use_container_width=True)
            st.caption(
                "Expected calibration error is the gap between stated confidence and "
                "observed frequency. Fitted on a split disjoint from both training and "
                "test — a calibrator fitted on test data has seen its own answers."
            )
            reliability = ARTIFACTS / "reliability.png"
            if reliability.exists():
                st.image(str(reliability),
                         caption="Reliability curves — on the diagonal means the "
                                 "number shown to a clinician is trustworthy.")

        gen_path = ARTIFACTS / "generalization_by_cluster.csv"
        if gen_path.exists():
            gen = pd.read_csv(gen_path)
            if not gen.empty and "balanced_accuracy" in gen:
                st.subheader("Generalization across lineages")
                spread = gen.groupby("drug")["balanced_accuracy"].agg(
                    worst="min", mean="mean", best="max", lineages="count")
                st.dataframe(spread.round(3), use_container_width=True)
                st.caption(
                    "Read the **worst** column. An aggregate score hides "
                    "lineage-specific collapse, and the worst case is what happens "
                    "when the system meets a lineage unlike anything it trained on — "
                    "the situation it actually faces in a hospital."
                )

    st.divider()
    st.subheader("How each responsibility requirement is addressed")
    st.markdown(
        """
| Requirement | How |
|---|---|
| **Defensive by construction** | Predicts and explains resistance that already exists. No generative capability anywhere in the codebase. |
| **Honest generalization** | Split by genetic lineage, never by row. `verify_no_leakage` raises *before* training. Per-lineage metrics above, worst case included. Covered species and drugs stated in the sidebar; anything else is refused. |
| **Calibrated confidence + no-call** | Calibrated on a split disjoint from train and test. Probabilities between the thresholds return no-call, and a no-call carries no confidence figure — there is no claim to be confident about. |
| **Honest explanations** | A known curated determinant is reported separately from a bare statistical association. A coefficient is never presented as biological cause. A failure call must cite a detected determinant, or it is downgraded to no-call. |
| **Human oversight** | Lab-confirmation warning on every report. The tool makes no treatment decision. |
"""
    )
    st.info(
        "**A worked example of the third row.** The demo genome carries both `mecA` and "
        "`blaZ`, and both are beta-lactamase-related. The cefoxitin call cites `mecA` "
        "and not `blaZ` — `blaZ` is a penicillinase that leaves cephamycins intact, so "
        "it correlates with resistance without causing it. Getting the call right is "
        "not the same as getting the reason right.",
        icon="🧬",
    )

if report_rendered:
    st.divider()
    render_disclaimer()
