# CLAUDE.md — Fake News & Misinformation Detector

## Project

B.Tech semester project (NLP, Dept. of CSE-AI/ML, KIET Group of Institutions, Aug–Dec 2026). A locally runnable web app that takes a headline / article / URL, extracts claims, retrieves evidence (online free sources + an offline FAISS fact-check index), runs NLI stance detection, checks dates ("old news presented as new"), and fuses everything into one of five verdicts — **REAL / FAKE / PARTIALLY TRUE / MISLEADING / UNVERIFIABLE** — with confidence, evidence cards (source + date) and a plain-language explanation. Team of five: Navishka Sharma, Naitik Kukreja, Naveen, Prateek Srivastava, Nikhil.

**Current status: MSE1 complete and pushed (2026-08-28 evening)** — env, `make download`, `make data` (with the residual-artefact fix: 26 text + 2 title patterns in `src/preprocess/artefacts.py`, byte-identical across runs, 6.8 min), `notebooks/00_datasets.ipynb`, `tests/test_data.py` + `tests/test_artefacts.py`; `make train-baselines` (NB/LR/SVM on WELFake + ISOT → `data/models/tfidf_*_v0.joblib`, `docs/results/classifier_table.md`), `notebooks/02_baselines.ipynb`, `docs/model_identification.md`, `tests/test_models.py`; EDA: `notebooks/01_eda.ipynb` (`make eda` → 18 figures `docs/figures/eda_*.png`, `docs/eda_summary.md`, `tests/test_eda.py`); `make test` = 102 passed; `make paper` builds; fresh-clone verified. The MSE1 Definition-of-Done is ticked in `docs/milestones/MSE1_ProjectDetails.md` (status line at the top). Not yet: DistilBERT, retrieval, stance, API, UI (MSE2/ESE); team-only items: slides, dry-run viva.

**Findings every agent must respect** (numbers in `docs/eda_summary.md` / `docs/model_identification.md`, details in `agent-docs/README.md`): **ISOT ⊂ WELFake** — 99.9 % of processed ISOT texts occur verbatim in WELFake (≈ 80 % of ISOT val/test rows sit in WELFake *train*), so the cross-dataset protocol is **WELFake∖ISOT (23,257 rows, hash join on `re.sub(r"\W+"," ",text.lower())` + MinHash near-dups) ↔ ISOT** — written into master §10/§18/§21.2 and the MSE2 doc; plain WELFake → ISOT is in-domain (0.978 preview) and must never be reported as cross-dataset. DistilBERT sweep grid = lr {2e-5, 3e-5, 5e-5} × max_len **{256, 512}** (only 19 % of WELFake inputs fit in 256 tokens, 51 % in 512; batch 16 at 256, batch 8 × grad-accum 2 at 512). ISOT `subject`, publication date and LIAR speaker/party are label proxies — never features. LIAR's official splits contain 5 train∩val / 4 train∩test duplicate statements (kept, disclosed). The FEVER parquet stores evidence *pointers* (page + sentence id), not sentence text — the MSE2 stance evaluation must fetch the sentences.

## Where things are

- `docs/FakeNewsDetector_ProjectDetails.md` — **master project document** (architecture, datasets, models, fusion rules, evaluation, milestone plan mapped to the marking scheme, week-by-week schedule). The `.docx` next to it is generated: `pandoc docs/FakeNewsDetector_ProjectDetails.md -o docs/FakeNewsDetector_ProjectDetails.docx --toc`.
- `docs/milestones/{MSE1,MSE2,ESE}_ProjectDetails.md` (+ `.docx`) — per-milestone verification checklists (artefacts, DoD, verification commands, viva Q&A). Artefact names there are canonical and mirrored in master §18/§21/§24.
- `agent-docs/README.md` — shared knowledge base for AI agents: decisions, findings, change log. **Read it first, keep it updated.**
- `data/README.md` — dataset provenance table (URL, licence, SHA-256, download date, rows) + label conventions + processed layout.
- Code: `src/config.py` (paths, seed 42, split ratios, label maps, `SCHEMA`), `src/preprocess/{load,artefacts,clean,dedupe,split,run_all}.py`, `scripts/download_data.py`, `src/common/seed.py`; stubs with docstrings for `src/{ingest,models,claims,evidence,stance,temporal,fusion,explain,api,eval}`.
- `notebooks/00_datasets.ipynb` is generated from `scripts/build_notebook_00.py` and executed in place with `make nb-run` (outputs are saved on purpose — examiners need them; never `nbstripout` before a viva).
- `notebooks/01_eda.ipynb` is generated from `scripts/build_notebook_01.py` (`--py` dumps the code cells for a dry run; **`make eda` does not rebuild it — run the builder first after editing it**) and executed with `make eda` (≈ 10 min CPU since the artefact-pattern list grew to 26, peak ≈ 2.5 GB RAM); its last cell writes `docs/eda_summary.md` and `data/processed/eda_numbers.json` — edit the template in the builder, never the .md by hand. RAM rules learned there: never `Series.str.split().str.len()` on the article corpora (+2 GB; use `str.count(r"\S+")`), prefer streaming `Counter`s over `CountVectorizer` for full-corpus counts, and use `tokenizers.Tokenizer.from_pretrained` instead of importing transformers/torch just for a tokenizer.
- `python -m spacy download en_core_web_sm` silently installs nowhere under the uv venv ("Audited 1 package"); `make setup` therefore installs the wheel URL (`SPACY_MODEL_URL` in the Makefile) and asserts `spacy.load('en_core_web_sm')`.
- `src/preprocess/artefacts.py` — the leakage-removal module: `PATTERNS` (26 named text regexes, strip order = list order; also the features of `artefact_only_accuracy`) and `TITLE_PATTERNS` (format tags `[VIDEO]`, trailing outlet suffixes from a closed `OUTLETS` list). Principle: strip publisher/format boiler-plate (credits, captions, bylines, title tags, URL debris), never content words — bare source names in prose ("told Reuters", "Breitbart News reported", "via Twitter") stay. Every pattern has a comment with the raw-WELFake document count; `tests/test_artefacts.py` (44 tests) pins the behaviour. Adding a pattern means: unit test → `make data` twice (check `sha256sum` identity) → `make train-baselines` → `make nb-run-baselines` → `make eda` (rebuild the notebook from the builder first if the builder changed) → update `docs/model_identification.md` §4.1.
- `src/models/baselines.py` — TF-IDF + NB/LR/SVM: `load_split` (joins `data/splits/*.csv` by id — never re-split), `fit_eval_all` (one TF-IDF per dataset shared by the three classifiers, val metrics), `top_coefficients` / `find_leak_tokens`, `render_classifier_table` (writes `docs/results/classifier_table.md` from `docs/results/train_logs/baselines_*.json`); CLI `python -m src.models.baselines --dataset welfake --model lr`. `src/models/tfidf_baselines.py` is only an alias (the docs use that name). `notebooks/02_baselines.ipynb` is generated from `scripts/build_notebook_02.py` and executed with `make nb-run-baselines`. Viva artefact: `docs/model_identification.md`.

## How to run

```bash
make setup       # uv venv .venv --python 3.11 + requirements.txt + spaCy en_core_web_sm + ipykernel "fakenews"
make download    # data/raw/ (WELFake via Zenodo API, ISOT via UVic zip, LIAR via UCSB, FEVER via fever.ai, FakeNewsNet via GitHub) — NO Kaggle account
make data        # src.preprocess.run_all -> data/processed/*.parquet + data/splits/*.csv ; log tee'd to docs/mse1_make_data.log (≈ 7 min, 1.8 GB RAM; byte-identical across runs)
make test        # pytest tests/  (102 tests: data 27, artefacts 44, eda 22, models 9; ≈ 40 s)
make nb-run      # execute notebooks/00_datasets.ipynb in place (11 s)
make eda         # execute notebooks/01_eda.ipynb in place -> docs/figures/eda_*.png + docs/eda_summary.md (≈ 10 min)
make train-baselines   # TF-IDF + NB/LR/SVM on WELFake + ISOT (train -> val, ~2 min CPU, peak RSS ~2.6 GB) -> data/models/, docs/results/
make nb-run-baselines  # build + execute notebooks/02_baselines.ipynb in place (~2.5 min)
make paper       # paper/main.pdf via paper/Makefile (latexmk if installed, else ~/.local/bin/tectonic)
```

Always use the venv interpreter (`.venv/bin/python`, `PY=.venv/bin/python` in the Makefile); the system `python3` (3.12) has no pandas.

## Environment facts (this laptop)

- Python 3.11.11 in `.venv` (uv at `~/.local/bin/uv`); torch 2.13 + CUDA 13 wheel from PyPI, `torch.cuda.is_available() == True`.
- GPU = **NVIDIA GeForce RTX 2050, 4 GB** (docs say "RTX 3050" — treat as the same "RTX 2050/3050-class 4 GB" budget). 8 CPU threads, 13 GB RAM (often only ~4 GB free — keep preprocessing streaming/vectorised), ~40 GB free disk.
- No Kaggle credentials on this machine — never depend on them; `scripts/download_data.py` avoids Kaggle entirely.
- pandoc available for the DOCX regeneration.

## Data conventions (decided at MSE1; see data/README.md)

- **ISOT ⊂ WELFake**: 99.9 % of processed ISOT texts (37,542 / 37,567) occur verbatim in processed WELFake (WELFake = Kaggle + McIntire + Reuters/ISOT + BuzzFeed). "Train WELFake → test ISOT" is therefore in-domain, not cross-dataset — use the hash-join-defined WELFake∖ISOT subset (23,257 rows) as the second domain (master §18, `docs/model_identification.md` §4.2).
- Baseline evaluation at MSE1 is on **val** only; the test splits are untouched until MSE2. The classifier is a style signal: after the residual-artefact fix LR val macro-F1 = **0.932** WELFake (was 0.946 with the boiler-plate), SVM 0.948, NB 0.862; ISOT LR 0.982 / SVM 0.990; ISOT → WELFake 0.83. Top-25 LR coefficients contain no publisher/format tokens any more; `reuters`, `breitbart`, `via` inside sentences remain by policy (content) and are disclosed in `docs/model_identification.md` §4.1. Keep the pre-fix numbers (LR 0.9462 / SVM 0.9610) as the "with residual artefacts" arm of the MSE2 ablation.
- WELFake CSV `label`: **0 = real, 1 = fake** (the Zenodo/Kaggle blurb is wrong; verified by artefact rates and the paper's counts).
- Unified parquet schema: `id, title, text, label, label5, label3, source_dataset, date, split, meta` for every dataset. Text is cased; artefacts (Reuters datelines, credits, captions, bylines, title tags, outlet suffixes, URL debris …) already stripped from text **and title** for WELFake/ISOT.
- Processed counts (2026-08-28 evening): WELFake 72,134 → 60,799 (34,295 real / 26,504 fake; 48,639 / 6,080 / 6,080), ISOT 44,898 → 37,567 (20,905 / 16,662; 30,053 / 3,757 / 3,757), LIAR 12,836, FEVER 23,000, FNN 1,056; artefact-only classifier on raw ISOT 0.9973.
- Splits: stratified 80/10/10 seed 42 for WELFake/ISOT (`data/splits/*.csv`, `id,label`, sorted by id — byte-identical across runs, parquets too); LIAR official; FEVER balanced 20k/3k.
- Dedup is greedy first-wins (exact SHA-1, then MinHash Jaccard ≥ 0.9); nothing is dropped from LIAR/FEVER-dev/FakeNewsNet.

## Locked stack (do not swap; refine only)

Python 3.11 · scikit-learn · PyTorch + HuggingFace transformers/datasets · spaCy `en_core_web_sm` · sentence-transformers `all-MiniLM-L6-v2` · faiss-cpu · trafilatura (+ newspaper3k fallback) · htmldate + dateparser · evidence via `ddgs` (DuckDuckGo), Wikipedia API, Google Fact Check Tools API (free key) + offline FAISS index (LIAR + FakeNewsNet PolitiFact) · FastAPI · HTML/CSS/JS front-end (Streamlit fallback) · Docker · Hugging Face Spaces (primary demo) / Render free tier `MODE=lite` (backup) · LIME (SHAP optional) · Ollama paraphrase = stretch only.

Models: TF-IDF + NB/LR/LinearSVC baselines → optional Bi-LSTM+GloVe → **fine-tuned DistilBERT** (fp16, max_len 256, batch 16 on the 4 GB GPU; CPU path = 10k subset / Colab free T4) · stance = `cross-encoder/nli-deberta-v3-small`. Datasets: WELFake (primary), ISOT (cross-dataset/leakage), LIAR (fine-grained + offline index), FEVER subset (stance), Live Claims Set (~100 hand-labelled recent claims).

## Hard rules

- **Evidence decides the verdict; the classifier only adjusts confidence / breaks ties.** Never let a text classifier alone output FAKE/REAL when evidence exists.
- **Never fabricate evidence, sources, quotes or dates.** UNVERIFIABLE is a valid result.
- Free tools only; everything must run on a student laptop (CPU inference ≤ 4 GB RAM).
- Reproducibility: fixed seeds, pinned `requirements.txt`, `make data / train / eval` regenerate all artefacts. Makefile targets (master §24): `setup`, `download`, `data`, `index`, `train-baselines`, `train-distilbert`, `train-liar`, `train` (= the three train-* targets), `models` (= `train` + `index`), `eval-models`, `eval-e2e`, `eval`, `run`, `test`, `paper`, `docker-build`, `docker-run` (+ helper `nb-run`, `lint`, `clean-data`). Unimplemented targets print "not implemented until MSE2/ESE" and exit 0.
- `agent-docs/`, scratch files and helper notes are **local only — never `git add`, stage, commit or push them**. Do not run git commands that stage/commit/push unless the user explicitly asks. Remote: `origin` = https://github.com/badnikhil/fake-news-detector (`main`); commit messages end with the `Co-Authored-By` / `Claude-Session` trailers the user asked for. A stray empty `~/.git` exists one level up — ignore it, do not delete it.
- Notebooks are generated from `scripts/build_notebook_0{0,1,2}.py`; edit the builder, regenerate, then execute in place (`make nb-run` / `make eda` / `make nb-run-baselines` — only the last one rebuilds automatically). Commit the executed notebooks with outputs.
- Document every change and decision in `agent-docs/README.md` (and keep this file current as conventions evolve). Regenerate the DOCX after editing any doc `.md`.
