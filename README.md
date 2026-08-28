# Fake News & Misinformation Detector

**Status: MSE1 (dataset + preprocessing foundation).** B.Tech semester project (NLP, Dept. of CSE-AI/ML, KIET Group of Institutions, Aug–Dec 2026).

A locally runnable web app that takes a **headline, article or URL**, extracts the check-worthy **claims**, retrieves **evidence** (free online sources + an offline FAISS fact-check index), runs **NLI stance detection**, checks **dates/context** ("old news presented as new") and fuses everything into one of five verdicts — **REAL / FAKE / PARTIALLY TRUE / MISLEADING / UNVERIFIABLE** — with a confidence score, evidence cards (source + date) and a plain-language explanation. Evidence decides the verdict; a fine-tuned DistilBERT content classifier only adjusts confidence and breaks ties. Everything runs on a student laptop with free tools (one RTX 2050/3050-class 4 GB GPU for training; CPU inference).

**Team:** Navishka Sharma (Data & EDA), Naitik Kukreja (Modelling), Naveen (Claims / Evidence / Stance), Prateek Srivastava (Backend / API / Deployment), Nikhil (UI / XAI / Paper).

## Quickstart

```bash
make setup      # .venv (Python 3.11 via uv, pip fallback) + requirements.txt + spaCy en_core_web_sm + ipykernel "fakenews"
make download   # raw datasets -> data/raw/   (no Kaggle account needed; ~330 MB; see data/README.md)
make data       # unify + clean + de-dup + split -> data/processed/*.parquet, data/splits/*.csv  (≈ 5 min CPU; log in docs/mse1_make_data.log)
make test       # pytest checks on the processed data (schema, split ratios, no leakage, LIAR splits unchanged)
make nb-run     # execute notebooks/00_datasets.ipynb in place (outputs saved for the viva)
```

Other Makefile targets (`make help`): `index`, `train-baselines`, `train-distilbert`, `train-liar`, `train`, `models`, `eval-models`, `eval-e2e`, `eval`, `run`, `paper`, `docker-build`, `docker-run` — these print "not implemented until MSE2/ESE" for now.

## Repository layout

```
Makefile, requirements.txt, pyproject.toml, .env.example
scripts/download_data.py      # make download (Zenodo / UVic / UCSB / fever.ai / GitHub sources)
src/config.py                 # paths, seed 42, split ratios, label mappings, canonical file names
src/preprocess/               # load.py, artefacts.py, clean.py, dedupe.py, split.py, run_all.py  (make data)
src/{ingest,models,claims,evidence,stance,temporal,fusion,explain,api,eval}/   # later milestones (stubs)
src/common/seed.py            # fixed seeds
data/raw/                     # downloads (git-ignored)   data/processed/*.parquet   data/splits/*.csv
data/README.md                # dataset table: source URL, licence, download date, SHA-256, rows
notebooks/00_datasets.ipynb   # counts, schema, label distributions (executed, outputs saved)
tests/test_data.py            # make test
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
