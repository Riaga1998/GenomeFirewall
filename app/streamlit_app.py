"""
GENOME FIREWALL — demo app
==========================
    streamlit run app/streamlit_app.py

Loads trained models, lets you pick a test genome (or upload a features row),
and shows the honest antibiotic-response report: label, calibrated confidence,
evidence tier, reason — plus the mandatory "confirm with lab testing" banner.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from train import load_models, predict_raw          # noqa: E402
from decision import decide, LIKELY_FAIL, LIKELY_WORK, NO_CALL  # noqa: E402
from features import curated_determinant_columns    # noqa: E402
from target_gate import load_gate                    # noqa: E402
from report import render_report, MANDATORY_DISCLAIMER  # noqa: E402

st.set_page_config(page_title="Genome Firewall", page_icon="🧬", layout="centered")


@st.cache_resource
def _load():
    cfg = yaml.safe_load(open(ROOT / "config/config.yaml"))
    feats = pd.read_parquet(ROOT / cfg["paths"]["features"])
    models = load_models(ROOT / cfg["paths"]["models"])
    gate = load_gate(str(ROOT / "config/target_gate.yaml"))
    curated = set(curated_determinant_columns(feats))
    return cfg, feats, models, gate, curated


cfg, feats, models, gate, curated = _load()

st.title("🧬 Genome Firewall")
st.caption(f"Defensive antibiotic-response prediction · {cfg['species']} · research prototype")

st.warning(MANDATORY_DISCLAIMER)

genome_id = st.selectbox("Choose a genome to analyse", feats.index.tolist())
containment = st.slider(
    "Simulated similarity to training set (OOD check)", 0.0, 1.0, 1.0, 0.01,
    help="In production this comes from sourmash containment vs the training set. "
         "Drag low to see the OOD no-call fire.",
)

if st.button("Run Genome Firewall", type="primary"):
    row = feats.loc[genome_id]
    present = {c for c in feats.columns if row.get(c, 0) == 1}
    detected = sorted(present & curated)

    st.subheader("Detected AMR determinants")
    st.write(", ".join(detected) if detected else "None detected")

    verdicts = []
    for abx in cfg["antibiotics"]:
        dm = models.get(abx)
        pr = predict_raw(dm, row) if dm else None
        v = decide(abx, pr, present, curated, gate, containment, cfg)
        verdicts.append(v)

    st.subheader("Antibiotic-response report")
    color = {LIKELY_FAIL: "🔴", LIKELY_WORK: "🟢", NO_CALL: "🟡"}
    for v in verdicts:
        with st.container(border=True):
            c1, c2 = st.columns([3, 1])
            c1.markdown(f"**{v.antibiotic}** — {color[v.label]} {v.label.upper()}")
            c1.caption(v.reason)
            c1.caption(f"Evidence: {v.evidence_tier.replace('_', ' ')}"
                       + (f" · {', '.join(v.supporting_features)}"
                          if v.supporting_features else ""))
            c2.metric("confidence", f"{v.confidence:.0%}")

    st.subheader("Clinician summary")
    st.text(render_report([v.to_dict() for v in verdicts], cfg["species"]))
