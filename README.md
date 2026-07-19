# 🧬 Genome Firewall

A **strictly defensive** research prototype: reconstructed bacterial genome (FASTA)
→ per-antibiotic prediction of **likely to fail / likely to work / no-call**, with a
calibrated confidence, an evidence tier, and the supporting genes. It never designs,
modifies, or suggests changes to an organism.

It runs end-to-end **on synthetic data out of the box** so nobody is blocked, with
clean seams to swap in real BV-BRC genomes.

---

## Pipeline at a glance

```
                 ┌──────────────────────── real-data path ────────────────────────┐
 FASTA genome ──▶ features.py (AMRFinderPlus) ──▶ genome × determinant matrix ─┐
      │                                                                        │
      └────────▶ dedup.py (sourmash MinHash) ──▶ groups + OOD signatures ──────┤
                                                                               ▼
 make_synthetic.py ──(shortcut for the demo)──▶ features / labels / groups ──▶ train.py
                                                                               │
                                        per-antibiotic logistic regression      │
                                        + GroupKFold OOF calibration (Venn-Abers)│
                                                                               ▼
                              decision.py:  GATE ▶ OOD ▶ MODEL ▶ CONFLICT ▶ BAND
                                                                               │
                                            Verdict{label, confidence, tier}    │
                                          ┌────────────────────┬───────────────┘
                                          ▼                    ▼
                                   evaluate.py           report.py (OpenAI writes
                              (honest grouped metrics)    the clinician summary)
                                          │                    │
                                          └────────▶ app/streamlit_app.py ◀─────┘
```

## Three modules (mapped to the brief)

| Brief module | Files | What it does |
|---|---|---|
| **01 Genome Reader** | `features.py` | Runs/parses **AMRFinderPlus** → binary determinant matrix. All columns are tier-1 curated determinants. |
| **02 Predictor** | `dedup.py`, `train.py`, `target_gate.py` | Sequence-homology **de-dup → grouped split**; one calibrated logistic-regression per drug; **deterministic target gate**. |
| **03 Decision Report** | `decision.py`, `evaluate.py`, `report.py`, `app/` | Gate + OOD + model + conflict + confidence band → verdict; honest metrics; Streamlit demo with the mandatory lab-testing banner. |

---

## Quickstart (synthetic — ~10 seconds)

```bash
pip install -r requirements.txt          # core deps are enough for the synthetic run
python src/make_synthetic.py             # writes data/{features,labels,groups}
python run_pipeline.py                   # grouped split → train → evaluate
streamlit run app/streamlit_app.py       # interactive demo
```

`run_pipeline.py` prints balanced accuracy, PR-AUC, Brier, no-call rate, and the
accuracy of the calls actually made — and writes `artifacts/metrics.json`.

## Swapping in real BV-BRC data

Nothing downstream changes; you only replace the three data files.

1. **Genomes.** Put one `*.fasta` per genome in `data/genomes/`.
2. **Labels.** Use the **organizer-pinned, lab-measured** SIR outcomes (not general
   phenotype fields, which may be model-generated). Write `data/labels.csv` with
   columns `genome_id, antibiotic, label` where label ∈ {R, S}.
3. **Features.** `python src/features.py` runs AMRFinderPlus per genome (cached) and
   writes `data/features.parquet`.
4. **Groups + OOD.** `python src/dedup.py` sketches genomes with sourmash, clusters
   near-identical strains, and writes `data/groups.csv` + `data/signatures.pkl`.
5. Run `python run_pipeline.py`. In `run_pipeline.py`, wire `containment_to_train`
   to `dedup.max_containment_to_reference(query_sig, train_sigs)` so the OOD
   no-call fires on genuinely novel genomes.

Set `OPENAI_API_KEY` to have `report.py` write the clinician summary via the API;
without it, a deterministic template is used (both always append the safety line).

---

## How this hits the "strong submission" rubric

- **Grouped, honest split.** Near-identical genomes can't straddle train/test
  (`dedup.py` → GroupKFold). Report the random-split *and* grouped-split numbers and
  explain the gap.
- **Calibrated confidence + real no-call.** Venn-Abers out-of-fold calibration;
  abstention has **three distinct triggers** — weak evidence, conflicting evidence,
  and out-of-distribution — not one arbitrary band.
- **Target gate.** `target_gate.py` forces "fail" when the drug's target is absent,
  so "likely to work" is never concluded from mere absence of resistance genes.
- **Honest evidence tiers.** Verdicts separate a *known determinant* from a
  *statistical association only* — a SHAP/weight is never presented as biological cause.
- **Human oversight.** Every report carries the mandatory
  "confirm with standard laboratory testing" banner. The tool is decision support only.
- **The plot that sells it.** `evaluate.risk_coverage_curve` — selective accuracy
  rises as coverage drops, proving the no-call system works. Lead with this + per-drug
  PR-AUC + a reliability diagram, **not** a single accuracy number.

## Tuning knobs (all in `config/config.yaml`)

- `dedup.jaccard_threshold` — how aggressively to collapse near-identical genomes.
- `ood.min_containment_to_train` — how novel a genome must be to trigger an OOD no-call.
- `decision.{resistant,susceptible}_threshold` — width of the weak-evidence no-call band.

## Scope & safety

Limited to **predicting and explaining resistance that already exists** from
reconstructed, openly available genomes. Excludes sample-to-genome processing and any
design, synthesis, or enhancement of organisms. Predictions from historical data do
not prove the system is safe or approved for real clinical use. **Every result must be
confirmed by standard laboratory testing.**
