# Genome Firewall — How It Works

**In one sentence:** real lab-measured *Staphylococcus aureus* genomes → AMRFinderPlus
resistance features → one calibrated logistic-regression per antibiotic → a decision
layer that abstains when unsure → evaluated on a leakage-checked grouped split.

Strictly defensive: the system **predicts and explains** existing resistance to support
treatment choices. It never designs, modifies, or suggests changes to an organism.

---

## 1. The data

We needed pairs of *(bacterial genome, did antibiotic X work in a real lab test)*.
BV-BRC provides them — but with a critical catch.

**The trap we avoided.** Most of BV-BRC's phenotype rows are *another model's
predictions* (labelled "Computational Method / AdaBoost Classifier"), not real lab
results. Training on those would mean building a model that imitates a model — circular
and disqualifying. We filtered to **Evidence = Laboratory Method** only, keeping just
genome–antibiotic pairs backed by an actual susceptibility test (broth dilution, disk
diffusion, etc.).

**What we used:**
- **Source:** BV-BRC AMR Phenotypes, laboratory-measured rows only.
- **Species:** *Staphylococcus aureus* (chosen because it had by far the most
  lab-measured data — selected from the data, not by guessing, via `screen_targets.py`).
- **Antibiotics (4):** cefoxitin, ciprofloxacin, erythromycin, tetracycline — each
  well-populated *and* class-balanced.
- **Scale:** 1,863 genomes, 8,116 lab-measured labels (above the 1,000–3,000 target).

| Antibiotic | Mechanism | Resistant | Susceptible |
|---|---|---|---|
| cefoxitin | `mecA` (methicillin resistance) | 816 | 220 |
| ciprofloxacin | `gyrA`/`grlA` point mutations | 910 | 577 |
| erythromycin | `erm`/`msr` genes | 666 | 578 |
| tetracycline | `tet` genes | 303 | 959 |

---

## 2. The model

**Four independent logistic-regression classifiers — one per antibiotic.** Not a neural
network, not one joint model. For each drug, a plain linear model reads the genome's
resistance determinants (present/absent) and outputs a probability of resistance.

This is deliberate, not a shortcut: the challenge recommends this exact baseline because
it runs on a CPU in seconds and — crucially — is **explainable and calibratable**, which
is what the challenge scores hardest. The simple model *is* the right answer here.

---

## 3. The four transformations (genome → verdict)

A genome is ~2.8 million letters of DNA; nothing learns from raw letters. So:

1. **Genome → features** (`features.py`, via **AMRFinderPlus 4.2.7**). Scans each genome
   and reports which known resistance determinants are present. Output: a row of **137**
   binary features per genome (49 acquired genes like `GENE:mecA`, 88 point mutations
   like `MUT:gyrA_S84L`).
2. **Features → learned weights** (`train.py`). Each per-drug logistic regression learns
   which determinants predict failure, from the real lab labels.
3. **Score → trustworthy verdict** (`decision.py`). The raw score is turned into a
   *calibrated* probability (Venn-Abers), then five ordered checks decide the outcome:
   **GATE → OOD → MODEL → CONFLICT → BAND**. Weak, conflicting, or unfamiliar evidence
   returns **no-call** rather than a guess.
4. **Honest evaluation** (`evaluate.py`). Scored on a **grouped split** so near-identical
   strains cannot appear in both train and test.

Every verdict carries: label (likely to fail / likely to work / no-call), a calibrated
confidence, an evidence tier (known determinant vs statistical-only), and the supporting
genes. A mandatory "confirm with standard laboratory testing" notice appears on every report.

---

## 4. Why this is trustworthy (the actual point)

- **Leakage-checked grouped split.** Genomes are clustered by sequence similarity
  (sourmash MinHash) and split by group, so the model can't score high by memorising
  near-duplicate strains. We verified duplication was negligible in this dataset
  (max pairwise Jaccard ≈ 0.62 — normal *S. aureus* diversity, no near-duplicates).
- **Calibrated confidence.** Venn-Abers calibration means a stated 90% actually behaves
  like 90%.
- **A real no-call.** The system abstains on weak, conflicting, or out-of-distribution
  evidence instead of forcing a yes/no — a confident wrong answer could point a care team
  at the wrong drug.
- **Honest explanations.** A curated known determinant is separated from a merely
  statistical association; a model weight is never presented as biological cause.
- **Human oversight.** Decision support only — every result must be confirmed by a lab.

---

## 5. Results (held-out grouped test set)

> Metrics from the current run. Re-run `run_pipeline.py` after tuning to refresh.

| Antibiotic | Balanced acc | PR-AUC | Brier | No-call rate | Accuracy on calls made |
|---|---|---|---|---|---|
| cefoxitin | 0.928 | 0.981 | 0.040 | 21% | 0.982 |
| ciprofloxacin | 0.998 | 0.999 | 0.003 | 42% | 1.000 |
| erythromycin | 0.972 | 0.986 | 0.026 | 51% | 0.994 |
| tetracycline | 0.997 | 0.992 | 0.006 | 80% | 0.971 |

**Read these honestly, and present them that way:** the accuracy-on-calls-made is measured
only on the samples the system *chose* to answer. Tetracycline, for example, abstains on
80% and is 97% accurate on the remaining 20% — that is the no-call system working as
designed, and it should be stated as "abstains on 80%, 97% accurate on the rest," never as
a flat 97%.

---

## 6. Scope & limitations

- One species (*S. aureus*), four antibiotics. Does **not** cover other species/drugs.
- Predicts resistance that already exists from a reconstructed genome. Excludes
  sample-to-genome processing and any organism design or modification.
- A research prototype trained on historical data — not validated, approved, or safe for
  real clinical use. Every result must be confirmed by standard laboratory testing.

---

## 7. Reproduce it

Runnable from a fresh clone (the small artifacts are committed):

```bash
pip install -r requirements.txt
python run_pipeline.py          # trains + evaluates on committed features/labels/groups
streamlit run app/streamlit_app.py
```

Rebuild from raw genomes (needs the ~5 GB FASTA download + AMRFinderPlus — see RUNBOOK.md):

```bash
python src/download_genomes.py --genome-list data/genome_list.txt --outdir data/genomes
python src/build_features.py --genomes-dir data/genomes --organism Staphylococcus_aureus --out data/features.parquet
python src/dedup.py
python run_pipeline.py
```
