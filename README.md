# Fake News & Misinformation Detector

**Status: MSE1 deliverables complete (2026-08-28)** — datasets downloaded and unified, `make data` preprocessing with artefact/leakage removal (byte-identical across runs), EDA (18 figures + `docs/eda_summary.md`), TF-IDF + NB/LR/SVM baselines on WELFake and ISOT (`docs/results/classifier_table.md`, `docs/model_identification.md`), paper §I–II, 102 passing tests. Not yet (MSE2/ESE): DistilBERT, retrieval, stance, fusion, API, UI, deployment. See `docs/milestones/MSE1_ProjectDetails.md` for the ticked Definition-of-Done. B.Tech semester project (NLP, Dept. of CSE-AI/ML, KIET Group of Institutions, Aug–Dec 2026).

A locally runnable web app that takes a **headline, article or URL**, extracts the check-worthy **claims**, retrieves **evidence** (free online sources + an offline FAISS fact-check index), runs **NLI stance detection**, checks **dates/context** ("old news presented as new") and fuses everything into one of five verdicts — **REAL / FAKE / PARTIALLY TRUE / MISLEADING / UNVERIFIABLE** — with a confidence score, evidence cards (source + date) and a plain-language explanation. Evidence decides the verdict; a fine-tuned DistilBERT content classifier only adjusts confidence and breaks ties. Everything runs on a student laptop with free tools (one RTX 2050/3050-class 4 GB GPU for training; CPU inference).

**Team:** Navishka Sharma (Data & EDA), Naitik Kukreja (Modelling), Naveen (Claims / Evidence / Stance), Prateek Srivastava (Backend / API / Deployment), Nikhil (UI / XAI / Paper).

## Quickstart

```bash
make setup             # .venv (Python 3.11 via uv, pip fallback) + requirements.txt + spaCy en_core_web_sm (wheel URL) + ipykernel "fakenews"
make download          # raw datasets -> data/raw/   (no Kaggle account needed; ~330 MB; see data/README.md)  <- the only manual prerequisite for everything below
make data              # unify + strip artefacts + clean + de-dup + split -> data/processed/*.parquet, data/splits/*.csv  (≈ 7 min CPU, 1.8 GB RAM; log in docs/mse1_make_data.log)
make test              # pytest: processed data, artefact regexes, EDA outputs, baselines (≈ 1 min)
make nb-run            # execute notebooks/00_datasets.ipynb in place (outputs saved for the viva)
make eda               # execute notebooks/01_eda.ipynb in place -> docs/figures/eda_*.png + docs/eda_summary.md (≈ 4 min CPU, 2.5 GB RAM)
make train-baselines   # TF-IDF + NB/LR/SVM on WELFake + ISOT (train -> val) -> data/models/tfidf_*_v0.joblib, docs/results/ (≈ 2 min CPU)
make nb-run-baselines  # build + execute notebooks/02_baselines.ipynb in place (≈ 3 min CPU)
make paper             # build paper/main.pdf (latexmk if installed, otherwise the tectonic binary in ~/.local/bin)
```

Fresh-clone recipe (verified 2026-08-28): `git clone … && cd fake-news-detector && make setup && make download && make data && make test && make paper` — only `data/raw/` (from `make download`, or copied from a teammate) is needed beyond the repo; everything else is regenerated. Other Makefile targets (`make help`): `index`, `train-distilbert`, `train-liar`, `train`, `models`, `eval-models`, `eval-e2e`, `eval`, `run`, `docker-build`, `docker-run` — these print "not implemented until MSE2/ESE" for now.

## Repository layout

```
Makefile, requirements.txt, pyproject.toml, .env.example
scripts/download_data.py      # make download (Zenodo / UVic / UCSB / fever.ai / GitHub sources)
scripts/build_notebook_0{0,1,2}.py   # the notebooks are generated from these builders, then executed in place
src/config.py                 # paths, seed 42, split ratios, label mappings, canonical file names
src/preprocess/               # load.py, artefacts.py (25 text + 2 title leakage patterns), clean.py, dedupe.py, split.py, run_all.py  (make data)
src/models/baselines.py       # TF-IDF + NB/LR/SVM, split join, leakage-token check, classifier table (make train-baselines)
src/{ingest,models,claims,evidence,stance,temporal,fusion,explain,api,eval}/   # later milestones (stubs)
src/common/seed.py            # fixed seeds
data/raw/                     # downloads (git-ignored)   data/processed/*.parquet   data/splits/*.csv
data/README.md                # dataset table: source URL, licence, download date, SHA-256, rows
notebooks/00_datasets.ipynb   # counts, schema, label distributions (executed, outputs saved)
notebooks/01_eda.ipynb        # EDA: 18 figures -> docs/figures/eda_*.png, insights -> docs/eda_summary.md (make eda)
notebooks/02_baselines.ipynb  # TF-IDF baselines, top-25 coefficients + leakage check, ISOT ⊂ WELFake hash join (make nb-run-baselines)
tests/                        # test_data.py, test_artefacts.py, test_eda.py, test_models.py  (make test)
docs/results/                 # classifier_table.md (generated), train_logs/*.json
docs/model_identification.md, docs/eda_summary.md, docs/problem_identification.md, docs/literature_table.md   # MSE1 viva artefacts
paper/                        # main.tex (IEEEtran), refs.bib (33 entries), Makefile  (make paper)
docs/FakeNewsDetector_ProjectDetails.md        # master project document (architecture, datasets, models, plan)
docs/milestones/{MSE1,MSE2,ESE}_ProjectDetails.md   # per-milestone checklists + viva Q&A
docs/mse1_make_data.log       # last `make data` run log
```

## Datasets (see `data/README.md`)

WELFake (Zenodo, CC BY 4.0) · ISOT (UVic ISOT lab) · LIAR (UCSB) · FEVER (fever.ai, CC BY-SA 3.0) · FakeNewsNet PolitiFact titles (GitHub). All downloaded by `make download` without Kaggle.

## Documents

- Master doc: `docs/FakeNewsDetector_ProjectDetails.md` (+ `.docx`)
- Milestones: `docs/milestones/MSE1_ProjectDetails.md`, `MSE2_ProjectDetails.md`, `ESE_ProjectDetails.md`
- AI-agent conventions: `CLAUDE.md`
