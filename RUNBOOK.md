# RUNBOOK — Genome Firewall (Staphylococcus aureus)

Everything is preconfigured. `data/labels.csv` (8,116 labels) and
`data/genome_list.txt` (2,528 genome IDs) are already in place.

## 0. One-time setup

```bash
# Python deps
pip install -r requirements.txt

# AMRFinderPlus is NOT pip — install via conda/bioconda:
conda install -c bioconda -c conda-forge ncbi-amrfinderplus
amrfinder -u                 # download the reference DB (required, once)
amrfinder --version          # confirm it works before going further
```

Checkpoint: `amrfinder --version` prints a version. If it errors, fix this
before anything else — nothing downstream works without it.

## 1. Download the genome FASTAs  (START THIS FIRST — it's the long pole)

```bash
python src/download_genomes.py --genome-list data/genome_list.txt \
    --outdir data/genomes --jobs 8
```

* Downloads 2,528 `.fna` files into `data/genomes/`.
* Resumable — re-run if it dies; it skips what's already there.
* Tight on time? Add `--max-genomes 1200` for a faster, still-solid run.

Checkpoint: `ls data/genomes/*.fna | wc -l` shows a few thousand files.

## 2. Genome → features (AMRFinderPlus)

```bash
python src/build_features.py --genomes-dir data/genomes \
    --cache data/amrfinder --organism Staphylococcus_aureus \
    --jobs 4 --out data/features.parquet
```

* ~10–60 s per genome. Parallelised, cached. Runs while you build slides.
* `--organism Staphylococcus_aureus` is REQUIRED for ciprofloxacin — it turns
  on gyrA/grlA point-mutation detection. Drop it and that drug tanks.

Checkpoint:
```bash
python -c "import pandas as pd; d=pd.read_parquet('data/features.parquet'); print(d.shape); print(d.columns[:20].tolist())"
```
You should see ~thousands of genomes × tens–hundreds of determinants, with
column names like `GENE:mecA`, `MUT:gyrA_S84L`.

## 3. Group genomes by similarity (the honest split)

```bash
python src/dedup.py
```

Sketches each genome (sourmash), clusters near-identical strains, writes
`data/groups.csv` + `data/signatures.pkl`.

Checkpoint: `data/groups.csv` exists; group count << genome count.

## 4. Train + evaluate (grouped split, calibrated, honest metrics)

```bash
python run_pipeline.py
```

Prints per-drug balanced accuracy, PR-AUC, Brier, no-call rate, and accuracy of
the calls actually made. Writes `artifacts/models/` and `artifacts/metrics.json`.

## 5. Demo

```bash
streamlit run app/streamlit_app.py
```

Optional: `export OPENAI_API_KEY=...` first so the clinician report is written
by the API (falls back to a template otherwise).

---

## If AMRFinderPlus install/compute blocks you on the day
BV-BRC ships per-genome `.PATRIC.spgene.tab` AMR calls, and NCBI's MicroBIGG-E
has precomputed AMRFinderPlus results for public assemblies. Either is a
fallback to keep you unblocked — but running AMRFinderPlus yourself is the
scoreable Module-01 path, so prefer it.

## Order of operations that saves you
Kick off **step 1 immediately**, let steps 1–2 run in the background, and build
your demo/slides meanwhile. The download + AMRFinderPlus is the only slow part;
everything after it takes seconds.
