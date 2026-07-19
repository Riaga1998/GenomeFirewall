# 🧬 Genome Firewall

**🔗 Live demo:** https://genomefirewall-fn6vvubbx9k2lkn6guvdsr.streamlit.app/

### Team

| Member | Programme / Role | Institution | Location |
|---|---|---|---|
| **Ritika Agarwal** | MSE, Computer and Information Sciences · Research Assistant, Perelman School of Medicine | University of Pennsylvania | Philadelphia, PA, USA |
| **Deepak Kushwaha** | MSc Data Science | University of Naples Federico II | Naples, Italy |
| **Hirak Jain** | B.Tech, Artificial Intelligence | Guru Gobind Singh Indraprastha University | Delhi, India |

Built for the Hack-Nation Global AI Hackathon, Challenge 06 — Genome Firewall.

---


**Scope: *Staphylococcus aureus*, four antibiotics — cefoxitin, ciprofloxacin,
erythromycin, tetracycline.** Cefoxitin is the laboratory surrogate for methicillin
resistance, so a cefoxitin call is an MRSA call.


Predicts which antibiotics are likely to fail from an assembled bacterial genome —
before standard laboratory susceptibility results arrive.

Standard testing takes one to three days. During that window treatment is a best guess,
and every ineffective course costs the patient time and gives resistant bacteria another
opportunity. This tool turns a reconstructed genome into an earlier, evidence-backed
prediction for each drug, with calibrated confidence and an explicit refusal to answer
when the evidence is weak.

**Scope: strictly defensive.** The system predicts and explains resistance that already
exists in a sequenced isolate, to support treatment choices and public-health tracking.
It does not design, modify, or optimize organisms.

**Research prototype — not a diagnostic device.** Every report must be confirmed by
standard laboratory susceptibility testing before it informs treatment.

---

## Documentation

| Document | Covers |
|---|---|
| [`docs/model_card.md`](docs/model_card.md) | Scope, metrics, limitations, intended use |
| [`docs/feature_schema.md`](docs/feature_schema.md) | Module 01 output specification (v1.0.0) |
| [`docs/thresholds.md`](docs/thresholds.md) | De-duplication threshold: how it was chosen |

---

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

conda create -n amrfinder -c conda-forge -c bioconda ncbi-amrfinderplus -y
conda run -n amrfinder amrfinder -u     # download the reference database

python train.py --synthetic      # verify the pipeline end to end
pytest tests/ -q                 # 47 tests
streamlit run app/streamlit_app.py
```

The app finds the `amrfinder` binary automatically — PATH first, then common conda
env locations, or set `AMRFINDER_PATH`. It does not need the conda env activated, and it
pins `CONDA_PREFIX` for the subprocess so AMRFinderPlus resolves its database correctly
when called from the project virtualenv.

Verified against **AMRFinderPlus 4.2.7**, database **2026-05-15.1**, native arm64.

`--synthetic` generates a cohort with the same statistical hazards as the real data
(clonal lineage structure, class imbalance, missing labels). It proves the pipeline runs
correctly and **nothing about real-world performance**.

Train on the real cohort (1,675 genomes, shipped in `data/real/`):

```bash
python train_real.py
```

---

## What it does

```
assembled FASTA
      │
      ▼
[01] Genome Reader ── AMRFinderPlus ──▶ binary determinant features
      │                                  gene:mecA, point:gyrA_S84L, class:BETA-LACTAM
      ▼
[02] Predictor ── target gate ──▶ L1 logistic regression ──▶ calibration
      │           (deterministic)      (one per drug)        (held-out split)
      ▼
[03] Decision Report ── likely to fail / likely to work / no-call
                        + calibrated confidence + the determinant behind it
```

### Module 01 — Genome Reader (`src/genome_reader.py`)

Runs AMRFinderPlus over an assembly and encodes every hit as a named binary feature.
Three interpretable families — `gene:` (acquired), `point:` (resistance mutation), and
`class:` (drug-class rollup) — so any prediction traces back to a specific determinant.
Determinants appearing in fewer than 3 genomes are dropped: singletons cannot generalize,
they only let the model memorize individual isolates.

### Module 02 — Predictor (`src/predictor.py`)

One L1-penalized logistic regression per antibiotic. L1 is deliberate — it zeroes most
coefficients, leaving a short list of determinants per drug that can be shown as
evidence.

Two layers wrap the model:

**A deterministic target gate** (`src/drug_database.py`) runs *before* the model. A drug
whose molecular target the species does not possess is intrinsically ineffective — that
is biology, not something to learn from data. Vancomycin cannot cross a gram-negative
outer membrane; ampicillin is futile against *K. pneumoniae*. The gate also prevents the
brief's stated failure mode: reporting "likely to work" merely because no resistance
marker was found.

**Probability calibration** (`src/utils/calibration.py`) fitted on a *separate* held-out
split. If the app says 85% confident, then among all calls it labels 85%, about 85% must
be right. Isotonic regression by default, falling back to Platt scaling on small
calibration sets where isotonic would fit noise.

### Module 03 — Decision Report (`src/decision_report.py`, `app/streamlit_app.py`)

Per-drug metrics, reliability curves, and a Streamlit app that shows for every result:
the drug, the call, calibrated confidence, and which of three evidence categories it
rests on.

---

## The design decisions that matter

### Splitting by genetic cluster, never by row

The most common way a genomic AMR model produces an inflated score is near-identical
isolates from the same outbreak landing in both train and test. A random row split then
measures memorization, not resistance biology.

`src/utils/clustering.py` sketches each genome with MinHash over k-mers, clusters by
Jaccard similarity, and splits at the *cluster* level. `verify_no_leakage` runs before
training and raises if any cluster spans two sets.

The calibration set is held out separately from test — fitting the calibrator on test
data would make the reported Brier score and reliability curve optimistic, because the
calibrator would have seen its own answers.

### A failure call must cite evidence

Testing surfaced a real bug worth describing, because it is the exact failure the
challenge penalizes.

With a training cohort that is ~80% resistant, the logistic regression learns a large
positive intercept. A genome with **zero** resistance determinants was therefore scored
at 88% probability of failure for ciprofloxacin — a confident assertion resting on
nothing but the base rate of the training set.

The fix is a coherence rule in `DrugModel.predict_one`: a `likely to fail` decision must
be backed by at least one detected determinant. Without one, the score reflects the
model's prior rather than evidence from *this* isolate, and the call is downgraded to
`no-call` with that stated plainly. Verified by
`test_empty_genome_never_asserts_failure`, alongside
`test_detected_determinant_still_drives_a_failure_call`, which confirms the downgrade
does not suppress genuine evidence-backed resistance.

```
clean genome, no determinants          mecA present (MRSA)
Cefoxitin       no-call                Cefoxitin      likely to fail  ← gene:mecA
Ciprofloxacin   no-call                Ciprofloxacin  no-call
Erythromycin    no-call                Erythromycin   no-call
Tetracycline    no-call                Tetracycline   no-call
```

### Two schema bugs found by running the real tool

Both were invisible until AMRFinderPlus was actually installed and pointed at a genome
with known resistance genes. Both are now regression-tested.

**The parser read the wrong column names.** AMRFinderPlus v4 renamed most output headers
— `Gene symbol` → `Element symbol`, `Element type` → `Type`, and the
`% ... reference sequence` pair lost its trailing word. The parser only knew the v3
names, so nothing matched and every hit was dropped. A genome carrying five resistance
genes came back completely clean, which is the most dangerous possible failure: silence
that reads as good news. `normalize_report` now accepts both spellings and *raises* on an
unrecognized schema rather than returning an empty result.

**The drug-class vocabulary did not match the tool's.** AMRFinderPlus reports `sul1`
under class `SULFONAMIDE`; the drug database called the same thing
`FOLATE-PATHWAY-ANTAGONIST`. The tags never matched, so a genuine known determinant for
TMP-SMX was demoted to "statistical association only" — understating evidence the system
actually had. Each drug now declares the set of AMRFinderPlus class tags that count as a
known mechanism, which also handles combination drugs whose components are reported
separately.

### A no-call has no confidence

Found by driving the actual app. Ciprofloxacin came back as `no-call` displayed beside
**"Calibrated confidence: 94%"** — a refusal to answer, presented as a near-certain
result.

The cause was scoring a no-call by its closeness to the decision boundary
(`1 - |p - 0.5| × 2`), which peaks at exactly the point where the model knows *least*.
A genome the system could say nothing useful about produced the highest number on the
page.

`Prediction` now carries `resistance_probability` (always meaningful) separately from
`confidence`, which is `None` unless a decision was actually made — there is no claim to
be confident about otherwise. The app shows the raw probability for a no-call instead.
Ciprofloxacin now reads **"Resistance probability: 53%"**, which sits inside the
uncertain band and explains itself. Locked in by `test_no_call_reports_no_confidence`
and `test_called_confidence_agrees_with_the_decision`.

### Three evidence categories, kept distinct

| Category | Meaning |
|---|---|
| **known resistance determinant detected** | A curated gene or point mutation for this drug class was found. |
| **statistical association only** | The model weighted features correlating with resistance in training. Correlation is not a demonstrated mechanism. |
| **no known resistance signal found** | Nothing detected. Absence of evidence is weaker than evidence of susceptibility. |
| **intrinsic resistance** | The deterministic gate fired; the species lacks a susceptible target. |

A non-zero coefficient is not proof of biological causation, and the app never presents
one as such.

### No-call as a feature

Probabilities between 0.35 and 0.65 return `no-call`. Forcing every sample into a yes/no
answer manufactures false confidence; a confident wrong result points a care team toward
the wrong drug.

---

## Evaluation

`train.py` reports what the challenge grades on:

- **Balanced accuracy**, with recall for resistant and susceptible cases separately
- **F1, AUROC, PR-AUC** per drug — PR-AUC because class imbalance makes AUROC flattering
- **Brier score** and **expected calibration error**, plus a reliability plot
- **No-call rate** and accuracy on the called subset
- **Per-cluster breakdown** — a model that scores well overall but collapses on one
  lineage is memorizing that lineage's background

That last table is the one worth reading closely. On synthetic data, drugs averaging
0.93 balanced accuracy still drop to 0.50 — chance — on individual held-out clusters. A
headline average hides that; the per-cluster spread is what tells you whether the model
would survive contact with a new isolate.

Artifacts land in `artifacts/`: `metrics_per_drug.csv`,
`generalization_by_cluster.csv`, `selected_determinants.csv`, `reliability.png`.

---

## Responsibility requirements

| Requirement | How it is met |
|---|---|
| Defensive by construction | Predicts existing resistance only. No generative capability anywhere in the codebase. |
| Honest generalization | Cluster-level splits, `verify_no_leakage` enforced pre-training, per-cluster metrics reported. |
| Calibrated confidence + no-call | Calibration on a disjoint split; ECE and reliability curves reported; uncertain band returns no-call. |
| Honest explanations | Known determinants separated from statistical associations; coefficients never presented as causation. |
| Human oversight | Lab-confirmation disclaimer above and below every report. The tool makes no treatment decision. |
| Stated coverage | Sidebar lists the exact species and antibiotics covered; anything else is out of scope. |

**Out of scope by design:** sample collection, reading DNA from blood, species
identification, genome reconstruction, and separating multiple organisms in one sample.
The system starts after isolation, sequencing, and assembly are complete.

---

## Layout

```
src/
  genome_reader.py      Module 01 — FASTA → features
  predictor.py          Module 02 — per-drug models, gate, coherence rule
  decision_report.py    Module 03 — metrics, calibration, generalization
  drug_database.py      drug properties + deterministic target gate
  synthetic_data.py     test cohort generator
  utils/
    amrfinder.py        AMRFinderPlus wrapper and parsing
    clustering.py       MinHash sketching, cluster splits, leakage check
    calibration.py      calibrators, metrics, reliability curves
app/streamlit_app.py    the demo
tests/test_pipeline.py  47 tests
train.py                end-to-end pipeline
```

## Installing AMRFinderPlus

```bash
conda install -c bioconda ncbi-amrfinderplus
amrfinder -u        # download the reference database
```

Without it the FASTA upload path is unavailable and the app says so; the manual-entry
tab still works for genomes annotated elsewhere, and training runs from a precomputed
AMRFinderPlus TSV via `build_feature_matrix(precomputed_tsv=...)`.

## Data

- **BV-BRC** (bv-brc.org) — genomes with laboratory-measured susceptibility results.
  Use organizer-pinned lab measurements, not general phenotype fields, which may contain
  model-generated predictions.
- **AMRFinderPlus** (github.com/ncbi/amr) — NCBI, public domain.
