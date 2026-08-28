# CLAUDE.md — Fake News & Misinformation Detector

## Project

B.Tech semester project (NLP, Dept. of CSE-AI/ML, KIET Group of Institutions, Aug–Dec 2026). A locally runnable web app that takes a headline / article / URL, extracts claims, retrieves evidence (online free sources + an offline FAISS fact-check index), runs NLI stance detection, checks dates ("old news presented as new"), and fuses everything into one of five verdicts — **REAL / FAKE / PARTIALLY TRUE / MISLEADING / UNVERIFIABLE** — with confidence, evidence cards (source + date) and a plain-language explanation. Team of five: Navishka Sharma, Naitik Kukreja, Naveen, Prateek Srivastava, Nikhil.

**Current status: MSE1 foundation built (2026-08-28)** — env, `make download`, `make data`, `notebooks/00_datasets.ipynb`, `tests/test_data.py`. Not yet: EDA notebook, LR baseline, models, retrieval, API, UI.

## Where things are

- `docs/FakeNewsDetector_ProjectDetails.md` — **master project document** (architecture, datasets, models, fusion rules, evaluation, milestone plan mapped to the marking scheme, week-by-week schedule). The `.docx` next to it is generated: `pandoc docs/FakeNewsDetector_ProjectDetails.md -o docs/FakeNewsDetector_ProjectDetails.docx --toc`.
- `docs/milestones/{MSE1,MSE2,ESE}_ProjectDetails.md` (+ `.docx`) — per-milestone verification checklists (artefacts, DoD, verification commands, viva Q&A). Artefact names there are canonical and mirrored in master §18/§21/§24.
- `agent-docs/README.md` — shared knowledge base for AI agents: decisions, findings, change log. **Read it first, keep it updated.**
- `data/README.md` — dataset provenance table (URL, licence, SHA-256, download date, rows) + label conventions + processed layout.
- Code: `src/config.py` (paths, seed 42, split ratios, label maps, `SCHEMA`), `src/preprocess/{load,artefacts,clean,dedupe,split,run_all}.py`, `scripts/download_data.py`, `src/common/seed.py`; stubs with docstrings for `src/{ingest,models,claims,evidence,stance,temporal,fusion,explain,api,eval}`.
- `notebooks/00_datasets.ipynb` is generated from `scripts/build_notebook_00.py` and executed in place with `make nb-run` (outputs are saved on purpose — examiners need them; never `nbstripout` before a viva).

## How to run

```bash
make setup       # uv venv .venv --python 3.11 + requirements.txt + spaCy en_core_web_sm + ipykernel "fakenews"
make download    # data/raw/ (WELFake via Zenodo API, ISOT via UVic zip, LIAR via UCSB, FEVER via fever.ai, FakeNewsNet via GitHub) — NO Kaggle account
make data        # src.preprocess.run_all -> data/processed/*.parquet + data/splits/*.csv ; log tee'd to docs/mse1_make_data.log (~5 min)
make test        # pytest tests/
make nb-run      # execute notebooks/00_datasets.ipynb in place
```

Always use the venv interpreter (`.venv/bin/python`, `PY=.venv/bin/python` in the Makefile); the system `python3` (3.12) has no pandas.

## Environment facts (this laptop)

- Python 3.11.11 in `.venv` (uv at `~/.local/bin/uv`); torch 2.13 + CUDA 13 wheel from PyPI, `torch.cuda.is_available() == True`.
- GPU = **NVIDIA GeForce RTX 2050, 4 GB** (docs say "RTX 3050" — treat as the same "RTX 2050/3050-class 4 GB" budget). 8 CPU threads, 13 GB RAM (often only ~4 GB free — keep preprocessing streaming/vectorised), ~40 GB free disk.
- No Kaggle credentials on this machine — never depend on them; `scripts/download_data.py` avoids Kaggle entirely.
- pandoc available for the DOCX regeneration.

## Data conventions (decided at MSE1; see data/README.md)

- WELFake CSV `label`: **0 = real, 1 = fake** (the Zenodo/Kaggle blurb is wrong; verified by artefact rates and the paper's counts).
- Unified parquet schema: `id, title, text, label, label5, label3, source_dataset, date, split, meta` for every dataset. Text is cased; artefacts (Reuters datelines, "Featured image via", URLs, handles…) already stripped for WELFake/ISOT.
- Splits: stratified 80/10/10 seed 42 for WELFake/ISOT (`data/splits/*.csv`, `id,label`, sorted by id — byte-identical across runs); LIAR official; FEVER balanced 20k/3k.
- Dedup is greedy first-wins (exact SHA-1, then MinHash Jaccard ≥ 0.9); nothing is dropped from LIAR/FEVER-dev/FakeNewsNet.

## Locked stack (do not swap; refine only)

Python 3.11 · scikit-learn · PyTorch + HuggingFace transformers/datasets · spaCy `en_core_web_sm` · sentence-transformers `all-MiniLM-L6-v2` · faiss-cpu · trafilatura (+ newspaper3k fallback) · htmldate + dateparser · evidence via `ddgs` (DuckDuckGo), Wikipedia API, Google Fact Check Tools API (free key) + offline FAISS index (LIAR + FakeNewsNet PolitiFact) · FastAPI · HTML/CSS/JS front-end (Streamlit fallback) · Docker · Hugging Face Spaces (primary demo) / Render free tier `MODE=lite` (backup) · LIME (SHAP optional) · Ollama paraphrase = stretch only.

Models: TF-IDF + NB/LR/LinearSVC baselines → optional Bi-LSTM+GloVe → **fine-tuned DistilBERT** (fp16, max_len 256, batch 16 on the 4 GB GPU; CPU path = 10k subset / Colab free T4) · stance = `cross-encoder/nli-deberta-v3-small`. Datasets: WELFake (primary), ISOT (cross-dataset/leakage), LIAR (fine-grained + offline index), FEVER subset (stance), Live Claims Set (~100 hand-labelled recent claims).

## Hard rules

- **Evidence decides the verdict; the classifier only adjusts confidence / breaks ties.** Never let a text classifier alone output FAKE/REAL when evidence exists.
- **Never fabricate evidence, sources, quotes or dates.** UNVERIFIABLE is a valid result.
- Free tools only; everything must run on a student laptop (CPU inference ≤ 4 GB RAM).
- Reproducibility: fixed seeds, pinned `requirements.txt`, `make data / train / eval` regenerate all artefacts. Makefile targets (master §24): `setup`, `download`, `data`, `index`, `train-baselines`, `train-distilbert`, `train-liar`, `train` (= the three train-* targets), `models` (= `train` + `index`), `eval-models`, `eval-e2e`, `eval`, `run`, `test`, `paper`, `docker-build`, `docker-run` (+ helper `nb-run`, `lint`, `clean-data`). Unimplemented targets print "not implemented until MSE2/ESE" and exit 0.
- `agent-docs/`, scratch files and helper notes are **local only — never `git add`, stage, commit or push them**. Do not run git commands that stage/commit/push unless the user explicitly asks. Nothing has been committed yet (repo was `git init`ed on 2026-08-28; a stray empty `~/.git` exists one level up — ignore it, do not delete it).
- Document every change and decision in `agent-docs/README.md` (and keep this file current as conventions evolve). Regenerate the DOCX after editing any doc `.md`.
