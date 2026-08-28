---
title: "MSE 1 — Project Details & Verification"
subtitle: "Fake News & Misinformation Detector (evidence-grounded, explainable 5-class news verification)"
date: "v1.0 — 27 August 2026"
---

# 1. Title block

| Field | Value |
|---|---|
| **Project** | Fake News & Misinformation Detector: an evidence-grounded, explainable news verification system |
| **Milestone** | **MSE 1** (Mid-Semester Evaluation 1) — 20 marks, viva-based |
| **Course** | B.Tech semester project (NLP / ML), Dept. of CSE (AI/ML), KIET Group of Institutions, Ghaziabad; odd semester Aug–Dec 2026 |
| **Team** | Navishka Sharma (Data & EDA lead), Naitik Kukreja (Modelling lead), Naveen (Claim / Evidence / Stance lead), Prateek Srivastava (Backend / API / Deployment lead), Nikhil (UI / XAI / Paper lead) |
| **Guide** | `<Guide name, designation>` |
| **Assumed MSE1 window** | Weeks 6–7, **21 Sep – 4 Oct 2026** (week 6 = buffer + dry-run viva, week 7 = exam window). Adjust to the official academic calendar. |
| **Document** | v1.0, 27 Aug 2026. Companion to the master document `docs/FakeNewsDetector_ProjectDetails.md` (v2.0); all file names, dataset IDs and objective numbers follow it. |
| **Repository** | `/home/nikhil/Desktop/nlp` (local); GitHub remote `<to be added>` |

## MSE 1 marking scheme (as on the evaluation form)

| Component | Criterion | Marks |
|---|---|---|
| **Project (15 marks, viva)** | Problem Identification | 3 |
| | Dataset | 3 |
| | Data Preprocessing | 3 |
| | EDA | 3 |
| | Model Identification | 3 |
| **Research Paper (5 marks, viva)** | Introduction | 2.5 |
| | Literature Survey | 2.5 |
| **Total** | | **20** |

# 2. Milestone summary

**What exists by MSE1.** The problem, 5-class output scheme (REAL / FAKE / PARTIALLY TRUE / MISLEADING / UNVERIFIABLE), objectives O1–O8 and scope are fixed in the master document. The four core datasets (D1 WELFake, D2 ISOT, D3 LIAR, D4 FEVER subset) plus D5 FakeNewsNet-PolitiFact titles are downloaded, schema-unified and documented (`data/README.md`, `notebooks/00_datasets.ipynb`). A reproducible preprocessing pipeline (`make data` → `src/preprocess/`) performs cleaning, exact + near-duplicate removal, artefact/leakage removal (e.g. the ISOT "(Reuters)" dateline), spaCy sentence segmentation and stratified 80/10/10 splits with seed 42. `notebooks/01_eda.ipynb` produces ≥ 10 figures in `docs/figures/` with a one-line insight each (`docs/eda_summary.md`), including a leakage-token table. The model ladder (TF-IDF baselines → DistilBERT; NLI cross-encoder for stance; MiniLM for retrieval) is justified with hyperparameter spaces and a hardware budget, and a **TF-IDF + Logistic Regression baseline is already trained** (`notebooks/02_baselines.ipynb`) as proof of feasibility. Paper §I (Introduction) and §II (Literature Survey) are drafted in `paper/main.tex` with ≥ 15 BibTeX entries and a comparison table.

**What deliberately does NOT exist yet** (planned for MSE2/ESE, so the examiner is not surprised): no fine-tuned DistilBERT, no GridSearchCV results, no evidence retrieval (online or offline FAISS index), no stance/NLI evaluation, no temporal check, no fusion rules in code, no FastAPI backend, no UI, no Docker image or deployment, no Live Claims Set (D7), no `tests/`. A URL → text + date `ingest` prototype may exist (week 5) but is not evaluated at MSE1.

# 3. Per-criterion sections

## 3.1 Problem Identification (3 marks)

**What we built / delivered.** A precise problem statement (given a headline / article / URL, decide with evidence, dates and an explanation whether its central claims are real, fake, partially true, misleading or unverifiable, using free tools on student hardware); motivation with cited statistics (Vosoughi et al. 2018; WEF Global Risks Report 2024; Islam et al. 2020 infodemic; Indian "old footage as breaking news" pattern); the gap (existing detectors are binary style classifiers that cannot explain, cannot handle partial truth/context, cannot say "I don't know"); eight measurable objectives O1–O8; explicit in/out-of-scope table; seven user stories; the 5-class output scheme with definitions and examples; the system architecture (§9 of the master doc).

**Artefacts**

| Artefact | What it shows |
|---|---|
| `docs/FakeNewsDetector_ProjectDetails.md` §2–§9 (+ `.docx`, printed) | Problem statement, motivation, objectives, scope, user stories, 5-class scheme, architecture diagram + component table |
| `docs/mse1_slides.pdf` (slides 1–6) | 5-minute version of the above for the viva |
| `paper/main.tex` §I | Same problem statement in paper form with 3 contribution bullets |

**Acceptance criteria / Definition of Done**

- [ ] Problem statement is one sentence, appears identically in the master doc §3, the slides and `paper/main.tex` §I.
- [ ] Every motivating statistic has a citation; items marked `[verify]` in the master doc (Indian lynching figure) are either verified or dropped from the paper.
- [ ] Objectives O1–O8 each have a numeric target (master doc §4).
- [ ] The five classes have a one-line definition and one example each (master doc §8); UNVERIFIABLE is explained as a first-class output.
- [ ] Out-of-scope table (§5.2) is present — no video/image forensics, no multilingual, no crawlers, no paid APIs/LLMs.
- [ ] Every team member can draw the pipeline (input → ingest → preprocess → claims → evidence → stance → temporal → fusion → explanation) on the whiteboard in under 2 minutes.

**How the examiner can verify it in ≤ 3 minutes.** Open the printed master doc at §3 (problem statement) and §8 (class table); ask any member to define the five classes and give one example each; ask "what happens when no evidence is found?" (answer: UNVERIFIABLE, by design); check that the motivation statistics carry references (§29).

**Likely viva questions**

1. *Why five classes instead of fake/real?* Real misinformation is mostly partial truth or true-but-out-of-context; a binary label cannot express "old news as new" (MISLEADING) or "right event, wrong number" (PARTIALLY TRUE), and forcing FAKE/REAL when evidence is missing is itself misinformation — hence UNVERIFIABLE.
2. *Why is UNVERIFIABLE a valid output?* Hyper-local or very recent claims have no coverage yet; saying "no reliable evidence either way" is more honest and safer than guessing. O6 targets UNVERIFIABLE precision ≥ 0.7.
3. *How is this different from a text classifier?* A classifier learns writing style and dataset artefacts; it cannot say *why*. Here **evidence decides the verdict**; the classifier only adjusts confidence (±0.10) or breaks a single-source tie (fusion rule R8).
4. *Why not use ChatGPT / a large LLM?* Cost, hardware (must run on a 4 GB GPU / CPU laptop), non-reproducibility and hallucination — an LLM could invent evidence, which violates requirement NFR-3 ("no fabricated evidence").
5. *Who are the users?* Ordinary news consumers who receive items on WhatsApp/Instagram and want a fast second opinion with sources; secondary users are examiners (offline mode) and ourselves (research testbed).
6. *What is the measurable goal?* O1 macro-F1 ≥ 0.95 on WELFake test (with honest cross-dataset numbers); O6 macro-F1 ≥ 0.55 on the 5-class Live Claims Set; O7 a public free-tier demo with ≤ 15 s median latency.
7. *What is explicitly out of scope and why?* Deepfake/image forensics (separate research problem), multilingual (English only; Hindi is a stretch), social-platform crawlers (ToS/brittleness), guarantees of truth.

## 3.2 Dataset (3 marks)

**What we built / delivered.** All core datasets downloaded into `data/raw/` (git-ignored), converted to a unified schema `id, title, text, label, label5, label3, source_dataset, date, split, meta` (identical for every dataset; `meta` = JSON extras), and documented with size, label set, source URL, licence and role. Label mappings are fixed: LIAR 6 → project 5 (true, mostly-true → REAL; half-true → PARTIALLY TRUE; barely-true → MISLEADING; false, pants-fire → FAKE) plus a 3-way collapse (TRUE / MIXED / FALSE); FEVER SUPPORTS / REFUTES / NEI → SUPPORTS / REFUTES / NEUTRAL stance.

| ID | Dataset | Size (must match on disk) | Role at MSE1 |
|---|---|---|---|
| D1 | WELFake (Verma et al., 2021) | 72,134 articles (35,028 real / 37,106 fake; CSV `label` 0 = real, 1 = fake — the Zenodo blurb says the reverse, see `data/README.md`) | Primary classifier train/val/test |
| D2 | ISOT (Ahmed et al., 2017) | 44,898 (21,417 true / 23,481 fake) | Cross-dataset + leakage study |
| D3 | LIAR (Wang, 2017) | 12,836 (train 10,269 / val 1,284 / test 1,283) | Fine-grained labels; offline index later |
| D4 | FEVER (Thorne et al., 2018) | 185,445 total → 20k train / 3k dev sampled, balanced | Stance evaluation (MSE2) |
| D5 | FakeNewsNet – PolitiFact titles (Shu et al.) | ~1,056 (432 fake / 624 real) | Offline index (MSE2) |
| D6 | ClaimBuster (Arslan et al., 2020) | 23,533 sentences | Optional; download only |
| D7 | Live Claims Set (ours) | ~100, 5-class | **Not started** (weeks 9–13) |

**Artefacts**

| Artefact | What it shows |
|---|---|
| `data/README.md` | Per-dataset table: URL, download date, file checksum, licence, size, label set, our mapping |
| `notebooks/00_datasets.ipynb` | Loads each raw file, prints row counts + label distributions, applies the unified schema, shows 3 sample rows per dataset |
| `data/raw/` (local only) | Original files; `.gitignore`d |
| Master doc §10, §10.1, §10.2 | Dataset table and label mappings |

**Acceptance criteria / Definition of Done**

- [ ] `notebooks/00_datasets.ipynb` runs "Run All" without error and prints **raw** counts equal to the table above (72,134 / 44,898 / 12,836 / FEVER subset 20k + 3k / ~1,056); processed counts for WELFake/ISOT are lower after empty/short-row drop and exact + near de-duplication (numbers in `docs/mse1_make_data.log`).
- [ ] Every dataset row in `data/README.md` has: source URL, licence line, download date, SHA-256 of the raw archive.
- [ ] Unified schema is identical across datasets; `label` uses the project vocabulary (`fake`/`real`, or LIAR 6-way + mapped columns `label5`, `label3`).
- [ ] LIAR official train/val/test are kept unchanged; FEVER subset sampling uses a fixed seed and is balanced across the three labels.
- [ ] Licence notes marked `[verify]` in the master doc (WELFake CC BY 4.0 on Zenodo, ClaimBuster CC BY 4.0) are confirmed and recorded.
- [ ] Nothing in `data/raw/` or `data/processed/` is committed to git.

**How the examiner can verify it in ≤ 3 minutes.**

```bash
cat data/README.md                                   # sizes, licences, mappings
python -c "import pandas as pd; \
  [print(n, len(pd.read_parquet(f'data/processed/{n}.parquet'))) \
   for n in ['welfake','isot','liar','fever_subset','fnn_politifact']]"
jupyter notebook notebooks/00_datasets.ipynb         # scroll to the count cells
```

Raw counts must equal the table (processed WELFake/ISOT are smaller after de-duplication — see `docs/mse1_make_data.log`); ask for the LIAR → 5-class mapping from memory.

**Likely viva questions**

1. *Why four datasets instead of one?* Each has a job: WELFake (largest, article-level, near-balanced) trains the classifier; ISOT tests cross-dataset generalisation and exposes leakage; LIAR gives fine-grained truth shades for PARTIALLY TRUE / MISLEADING and seeds the offline index; FEVER provides labelled (claim, evidence) pairs to evaluate stance detection.
2. *Why WELFake as primary?* 72k articles merged from four sources by its authors, so less single-source style bias than ISOT; near-balanced (35k/37k); free (Zenodo, CC BY 4.0 — `make download` fetches it without a Kaggle account).
3. *What is wrong with ISOT?* All "true" articles come from Reuters and carry a "WASHINGTON (Reuters) –" dateline; a model can reach ~99% by learning that token. We strip it and report the drop.
4. *Why map LIAR 6 → 5 that way?* PolitiFact's own definitions: mostly-true is "accurate with clarification" (REAL); half-true is "leaves out important details" (PARTIALLY TRUE); barely-true "ignores critical facts" (MISLEADING); false and pants-fire are both wrong (FAKE). UNVERIFIABLE has no LIAR counterpart — it arises only from retrieval failure.
5. *Why also a 3-way LIAR collapse?* 6-way text-only accuracy is ~27% (Wang 2017); TRUE/MIXED/FALSE is what the fusion step actually consumes.
6. *Why only 20–30k of FEVER?* It is used to evaluate (and optionally fine-tune) a pre-trained NLI model, not to train from scratch; 20k pairs fit one epoch in ~30–40 min on an RTX 2050/3050-class 4 GB GPU.
7. *Are the datasets recent?* No — ISOT is 2015–2018, LIAR ≤ 2017. That is why the end-to-end evaluation uses our own Live Claims Set of ~100 claims from Aug–Nov 2026. (Licences: see the DoD list above and `data/README.md`.)

## 3.3 Data Preprocessing (3 marks)

**What we built / delivered.** `make data` runs `src/preprocess/{clean,dedupe,artefacts,split}.py` in order: (1) load + unify schema; (2) drop empty / < 20-token texts and log counts; (3) exact duplicates via SHA-1 of normalised text, near-duplicates via MinHash (`datasketch`) — and assert zero duplicate pairs across train/test; (4) **artefact / leakage removal**: ISOT "`WASHINGTON (Reuters) - `" datelines and "`(Reuters)`" tokens, "Featured image via…", "Read more:", "21st Century Wire says…" trailers, URLs, e-mails, Twitter handles; an "artefact-only" classifier is trained to quantify the leakage; (5) NFKC normalisation, whitespace/quote normalisation (lower-casing only for the TF-IDF path — cased text kept for transformers); (6) spaCy sentence segmentation with offsets; (7) tokenisation choices recorded (TF-IDF word 1–2-grams; `DistilBertTokenizerFast` max_len 256); (8) stratified **80/10/10** splits per dataset with `random_state=42`, saved as ID lists; LIAR keeps official splits; (9) class weights (not resampling) noted for LIAR.

**Artefacts**

| Artefact | What it shows |
|---|---|
| `Makefile` target `data` | One command regenerates everything below |
| `src/preprocess/clean.py`, `dedupe.py`, `artefacts.py`, `split.py` | The pipeline steps, each with a `main()` and logged counts |
| `data/processed/{welfake,isot,liar,fever_subset,fnn_politifact}.parquet` | Unified, cleaned tables |
| `data/splits/{welfake,isot}_{train,val,test}.csv`, `data/splits/liar_{train,val,test}.csv` | ID lists so every model uses identical splits |
| `notebooks/01_eda.ipynb` (section "Before/after") | 5 raw vs. cleaned examples incl. a Reuters-prefix removal; leakage-token table; artefact-only classifier accuracy |
| `src/common/seed.py` | Fixed seeds for `random`, `numpy`, `torch` |
| Master doc §11.1 | The step list with rationale |

**Acceptance criteria / Definition of Done**

- [ ] `make data` runs end-to-end from the raw files in ≤ 15 min on a 4-core laptop CPU (dataset download is a documented prerequisite, not part of the target) and writes every parquet/CSV listed above.
- [ ] Running `make data` twice yields byte-identical split CSVs (seed 42).
- [ ] Log output reports, per dataset: rows in, rows dropped (empty/short), exact duplicates removed, near-duplicates removed, rows out.
- [ ] Cross-split overlap check prints `0` duplicate pairs between train and test for WELFake and ISOT.
- [ ] Split proportions are 80/10/10 ± 0.5% and class proportions in each split match the full dataset ± 1% (stratification).
- [ ] `grep -c "(Reuters)" ` on the processed ISOT text returns 0; the artefact-only classifier's ISOT accuracy is reported (expected: very high, ≥ 0.9 — measured 0.996 on 2026-08-28, logged in `docs/mse1_make_data.log`).
- [ ] Cased text is preserved in the parquet; lower-casing happens inside the TF-IDF vectoriser only.
- [ ] `data/processed/` and `data/splits/` are reproducible from a fresh clone + raw files by following `data/README.md`.

**How the examiner can verify it in ≤ 3 minutes.**

```bash
make data 2>&1 | tail -30          # or, if already run: ls -la data/processed data/splits
python -c "import pandas as pd; d=pd.read_parquet('data/processed/isot.parquet'); \
  print(d.text.str.contains('\(Reuters\)').sum())"     # expect 0
python -c "import pandas as pd; s=pd.read_csv('data/splits/welfake_train.csv'); print(len(s))"  # 48,677 = 80% of the 60,847 de-duplicated rows
```

Then open `notebooks/01_eda.ipynb` at the "Before/after" cell and the leakage-token table.

**Likely viva questions**

1. *Why remove "Reuters"?* It appears in essentially every ISOT true article and no fake one — a label shortcut that gives ~99% in-domain accuracy and near-zero transfer. Removing it makes the numbers honest; we report the artefact-only accuracy as evidence.
2. *Why stratified 80/10/10 with a fixed seed?* Validation is for model/hyperparameter selection, test is touched once; stratification keeps class ratios equal in every split; seed 42 + saved ID lists make every model comparable and the results reproducible.
3. *Why near-duplicate removal, not just exact?* Syndicated articles differ by a few tokens; a near-duplicate in both train and test inflates accuracy. We use MinHash (Jaccard on shingles) and verify zero cross-split duplicates.
4. *Why keep cased text?* Transformers use their own tokeniser (DistilBERT-uncased lower-cases internally); lower-casing beforehand would only lose information for the TF-IDF char-gram option and for spaCy NER, which is case-sensitive.
5. *Why not stemming / stop-word removal?* Sub-word tokenisers handle morphology; TF-IDF with sublinear tf already down-weights frequent words; stop words like "not" matter for claims.
6. *Why max_len 256?* The length histogram shows most article heads (title + first ~200 tokens) carry the signal; 256 fits fp16 batch 16 on a 4 GB GPU.
7. *Why class weights rather than oversampling for LIAR?* Imbalance is mild; weights avoid duplicating short statements and leaking them across folds.

## 3.4 EDA (3 marks)

**What we built / delivered.** `notebooks/01_eda.ipynb` produces every plot/table listed in master doc §11.2, saves each to `docs/figures/eda_*.png`, and `docs/eda_summary.md` lists each figure with **one-line insight → modelling consequence**.

**Artefacts**

| Figure / table (`docs/figures/`) | What it shows → consequence |
|---|---|
| `eda_class_balance.png` | Class bars per dataset → WELFake/ISOT near-balanced; accuracy acceptable but we report macro-F1 |
| `eda_length_hist.png` | Token-length histograms per class → max_len 256; fake articles' length pattern |
| `eda_top_ngrams_before.png` / `eda_top_ngrams_after.png` | Top 30 uni/bigrams per class before and after artefact removal → "Reuters", "said", "video", "Hillary" leakage tokens |
| `eda_wordclouds.png` | Per-class word clouds → presentation aid |
| `eda_ner_types.png` | spaCy entity-type frequency per class → motivates claim heuristics (PERSON/ORG/GPE/DATE) |
| `eda_isot_dates.png` | ISOT publication-date distribution (2015–2018) → dataset age; motivates temporal check and the Live Claims Set |
| `eda_duplicates.png` | Exact / near-duplicate counts and cross-split overlaps → leakage audit |
| `eda_liar_labels.png` | LIAR 6- and 3-label distributions; label vs. speaker party → fine-grained difficulty |
| `eda_fever_balance.png` | FEVER subset label balance; evidence-sentence lengths → stance model input sizing |
| `eda_style_features.png` | Readability, punctuation, ALL-CAPS ratio per class → style features are real but shallow |
| `docs/eda_summary.md` | One insight per figure; leakage-token table with artefact-only classifier accuracy |

**Acceptance criteria / Definition of Done**

- [ ] ≥ 10 figures exist in `docs/figures/` with the names above, each ≥ 150 dpi, titled, axis-labelled, legend where needed.
- [ ] `notebooks/01_eda.ipynb` runs "Run All" in ≤ 20 min on CPU and regenerates every figure.
- [ ] The before/after n-gram figures visibly differ (leakage tokens disappear after artefact removal).
- [ ] `docs/eda_summary.md` has exactly one insight line per figure and states the artefact-only classifier accuracy on ISOT.
- [ ] Duplicate audit reports numbers (not just a plot): exact, near, cross-split — cross-split = 0 after preprocessing.
- [ ] At least one insight per figure is turned into a design decision (max_len, macro-F1, class weights, claim heuristics, temporal check).

**How the examiner can verify it in ≤ 3 minutes.** `ls docs/figures/eda_*.png | wc -l` (≥ 10); open `docs/eda_summary.md`; open `eda_top_ngrams_before.png` next to `eda_top_ngrams_after.png`; ask "what does this figure change in your model?" for any two figures.

**Likely viva questions**

1. *What did the EDA change in your design?* max_len 256 (length histogram), macro-F1 as headline metric (balance), artefact removal (n-gram leakage), entity-based claim heuristics (NER plot), temporal check + Live Claims Set (date plot).
2. *Which tokens leak the label?* Before removal: "Reuters", "WASHINGTON", "said" for real; "video", "image", "Hillary", "via" for fake. After removal the top lists look topical rather than source-specific.
3. *Are fake articles longer or shorter?* Report the actual histogram; typical finding: fake articles have a heavier tail of very short and very long texts, more ALL-CAPS and exclamation marks.
4. *What does the LIAR label-vs-party plot tell you?* Labels are politically skewed by speaker; a text-only model can exploit speaker cues, which is why we use LIAR for fine-grained shading rather than as the main classifier.
5. *Why is the ISOT date range relevant?* All data are 2015–2018 US politics; a model trained on it is stale — motivates cross-dataset testing and the 2026 Live Claims Set.
6. *How did you measure duplication?* SHA-1 on normalised text (exact) and MinHash Jaccard ≥ 0.9 (near); counts are in `eda_duplicates.png` and the notebook.

## 3.5 Model Identification (3 marks)

**What we built / delivered.** The model ladder with rationale, hyperparameter search spaces and hardware/time budget (master doc §12), and a **trained TF-IDF + Logistic Regression baseline** on WELFake with accuracy / macro-F1 on the validation split as proof of feasibility.

| Tier | Model | Role |
|---|---|---|
| Baseline | TF-IDF + Multinomial NB; TF-IDF + LR (**trained at MSE1**); TF-IDF + LinearSVC | Reference points; LR coefficients interpretable; SVM = CPU fallback for `MODE=lite` |
| Middle (optional) | Bi-LSTM + GloVe 100d | Only if time permits (week 9) |
| Final | **DistilBERT-base-uncased fine-tuned** (fp16, max_len 256, batch 16 on an RTX 3050 4 GB; CPU path 10k subset / Colab T4) | Content classifier |
| Fine-grained | DistilBERT (or TF-IDF + LR) on LIAR 3-way | Truth-shade probability for fusion |
| Stance | `cross-encoder/nli-deberta-v3-small` (pre-trained NLI; optional FEVER fine-tune) | SUPPORTS / REFUTES / NEUTRAL per (claim, evidence) |
| Embedding | `all-MiniLM-L6-v2` | Offline FAISS index + passage re-ranking |
| Claims (optional) | TF-IDF + LR on ClaimBuster | Check-worthiness, combined with heuristics |

**Artefacts**

| Artefact | What it shows |
|---|---|
| Master doc §12.1–§12.3 | Ladder + rationale, search spaces (`GridSearchCV` grid; DistilBERT lr × epochs × max_len), time per model on the RTX 3050 / CPU |
| `notebooks/02_baselines.ipynb` | Loads `data/splits/welfake_*`, fits TF-IDF + LR, prints accuracy, macro-F1, confusion matrix on val; top ±20 coefficients |
| `src/models/tfidf_baselines.py` | Reusable training code (`make train-baselines` later runs NB/LR/SVM + GridSearch) |
| `docs/results/classifier_table.md` | First row (LR, val) filled at MSE1; rest at MSE2 |
| `docs/mse1_slides.pdf` (model slides) | Ladder, why DistilBERT, why NLI, why not an LLM |

**Acceptance criteria / Definition of Done**

- [ ] `notebooks/02_baselines.ipynb` runs "Run All" in ≤ 10 min on CPU using the saved split IDs (no re-splitting inside the notebook).
- [ ] Reported on WELFake **val**: accuracy, precision, recall, macro-F1, confusion matrix; LR macro-F1 ≥ 0.90 (sanity check; the O1 target of ≥ 0.95 is for the final model on test).
- [ ] The fitted vectoriser + LR are saved to `data/models/tfidf_lr_welfake_v0.joblib` and reload in one line.
- [ ] Top positive/negative LR coefficients are listed and contain no leakage tokens (otherwise preprocessing is incomplete).
- [ ] The DistilBERT configuration (lr grid, epochs, max_len, batch, fp16, grad-accum) and the CPU/Colab fallback are written down; every member can state the hardware budget.
- [ ] Each model in the ladder has a one-line "why included" and the alternatives rejected (BERT-base, RoBERTa, LLMs) are named with reasons.

**How the examiner can verify it in ≤ 3 minutes.** Open `notebooks/02_baselines.ipynb` — scroll to the metrics cell and coefficient table; `cat docs/results/classifier_table.md`; ask the "why" questions below.

**Likely viva questions**

1. *Why baselines first?* They take minutes, give a floor (LR usually ≥ 0.90 F1 here), expose leakage through interpretable coefficients, and become the CPU fallback (`MODE=lite`); a transformer that cannot beat TF-IDF + SVM is not worth its cost.
2. *Why DistilBERT rather than BERT-base?* 66M vs. 110M parameters, ~40% smaller, ~60% faster, retains ~97% of BERT's GLUE performance (Sanh et al. 2019); fits fp16 batch 16 × 256 tokens on a 4 GB RTX 3050 (~15–25 min/epoch on 58k articles) and CPU inference ≤ 1 s.
3. *Why not RoBERTa / DeBERTa for the classifier?* Larger memory footprint on 4 GB; the classifier's job is only to adjust confidence, so the accuracy gain is not worth 2–3× training time. DeBERTa-v3-small *is* used where it matters — NLI stance.
4. *Why NLI for stance?* Stance = "does the evidence entail / contradict / say nothing about the claim" — exactly the three NLI classes, so a pre-trained cross-encoder (`nli-deberta-v3-small`, trained on SNLI/MNLI/FEVER-NLI) works with zero training; FEVER gives a labelled dev set to measure it.
5. *Why a cross-encoder rather than bi-encoder embeddings for stance?* A cross-encoder reads claim and evidence jointly and models contradiction; cosine similarity of embeddings cannot distinguish "X happened" from "X did not happen". Bi-encoder MiniLM is used only for retrieval, where speed matters.
6. *Which hyperparameters will you tune and how?* Classical: `GridSearchCV` 5-fold over ngram_range, max_features, sublinear_tf, C / alpha, selected by val macro-F1. DistilBERT: lr {2e-5, 3e-5, 5e-5} × epochs {2, 3} × max_len {128, 256}, warmup {0.06, 0.1}, on a 20k subset, best config retrained on full train.
7. *What if the GPU is unavailable?* CPU path: 10k subset, max_len 128, 2 epochs (~1.5–3 h), or Google Colab free T4; checkpoints shared via the HF Hub so others only run inference. (For "why not an LLM", see §3.1 Q4.)

## 3.6 Research Paper — Introduction (2.5 marks)

**What we built / delivered.** `paper/main.tex` §I (≈ 1 page, IEEE two-column, IEEEtran): motivation with cited statistics, the gap in style-based detectors, the problem statement, and three contribution bullets: (i) an evidence-grounded 5-class verdict scheme with an explicit UNVERIFIABLE class; (ii) a laptop-scale hybrid pipeline (claim identification → free-source retrieval with an offline index → NLI stance → temporal check → rule fusion) with explanations; (iii) an honest evaluation including a cross-dataset leakage audit and a 2026 Live Claims Set.

**Artefacts**

| Artefact | What it shows |
|---|---|
| `paper/main.tex` §I; `paper/main.pdf` (built via `make paper`) | The introduction, compiling without errors |
| `paper/refs.bib` | Entries cited in §I (Vosoughi 2018, WEF 2024, Islam 2020, etc.) |

**Acceptance criteria / Definition of Done**

- [ ] §I is 0.75–1.0 page in the IEEE template, compiles with `make paper` (latexmk), no `??` citations.
- [ ] Problem statement in §I matches the master doc §3 word-for-word in meaning.
- [ ] Every statistic in §I is cited; the Indian lynching figure is verified or omitted.
- [ ] Exactly three contribution bullets, each traceable to an objective (O6, O3/O4/O5, O1).
- [ ] Written from notes, not pasted — will pass the week-14 Turnitin check (< 10%).

**How the examiner can verify it in ≤ 3 minutes.** Open `paper/main.pdf` page 1; check the contribution bullets against §4 objectives of the master doc; ask a member to state the gap in one sentence.

**Likely viva questions**

1. *What are your contributions?* The three bullets above — scheme, pipeline, honest evaluation.
2. *What is the gap you address?* Existing detectors classify style, not facts; they cannot explain, cannot handle partial truth or old-news-as-new, and cannot abstain.
3. *Why does misinformation matter enough for a paper?* Vosoughi et al. 2018: false news is 70% more likely to be retweeted and reaches 1,500 people ~6× faster; WEF 2024 ranks mis/disinformation the top short-term global risk; Islam et al. 2020 count ≥ 800 deaths from one COVID rumour.
4. *Which venue?* arXiv (cs.CL) in any case (week 15) plus an IEEE-sponsored Indian conference with a Nov–Dec deadline, chosen with the guide by week 10 (master doc §20).
5. *Who wrote it?* Nikhil assembles; each member writes their module's section and reviews another's; §I drafted in week 4.

## 3.7 Research Paper — Literature Survey (2.5 marks)

**What we built / delivered.** `paper/main.tex` §II (≈ 1–1.5 pages) grouped into five themes — (a) style-based detectors and their datasets, (b) evidence-based fact verification, (c) claim detection / check-worthiness, (d) spread and temporal misinformation, (e) explainability and efficient models — ending with a comparison table and a "what is missing" paragraph that leads into our method. `docs/literature_table.md` holds the full table; `paper/refs.bib` holds ≥ 15 entries.

**Literature-survey table skeleton** (`docs/literature_table.md`; fill `[fill]` cells from the papers, never from memory)

| # | Reference | Dataset | Method | Reported result | Limitation | What we take from it |
|---|---|---|---|---|---|---|
| 1 | Wang 2017, "Liar, Liar Pants on Fire" (ACL) | LIAR, 12.8k statements, 6 labels | CNN on text + speaker metadata | ~27% 6-way test acc. (text), 27.4% hybrid | Short statements; no evidence; 6-way barely above chance | LIAR data, 6→5 and 3-way mapping, honest baseline |
| 2 | Shu et al. 2020, FakeNewsNet (Big Data) | PolitiFact + GossipCop news with social context | Repository + content/social baselines | [fill] | Content requires re-crawling; social features unavailable to us | PolitiFact titles for the offline index |
| 3 | Ahmed, Traore & Saad 2017 (ISDDC) | ISOT, 44.9k articles | TF-IDF n-grams + LinearSVC etc. | up to 92% acc. (LSVM) | Reuters-only "true" class → source leakage | ISOT for cross-dataset + leakage audit; classical baselines |
| 4 | Verma et al. 2021, WELFake (IEEE TCSS) | WELFake, 72k articles | Linguistic features + word embeddings | 96.73% acc. | Binary; in-domain only | Primary training set; O1 target |
| 5 | Thorne et al. 2018, FEVER (NAACL) | 185k claims + Wikipedia evidence | Retrieval + NLI pipeline | 31.87 FEVER score / 50.91% label acc. (baseline) | Wikipedia-only; synthetic claims | Stance labels, evaluation set, pipeline shape |
| 6 | Kaliyar et al. 2021, FakeBERT (MTAP) | Kaggle fake-news | BERT + parallel CNN | 98.90% acc. | Binary; no evidence; leakage not audited | Transformer classifier upper bound; comparison row |
| 7 | Devlin et al. 2019, BERT (NAACL) | GLUE etc. | Bidirectional transformer pre-training | SOTA on 11 tasks | 110M params; heavy for 4 GB GPU | Fine-tuning recipe |
| 8 | Sanh et al. 2019, DistilBERT (arXiv) | GLUE | Knowledge distillation of BERT | 40% smaller, 60% faster, ~97% of BERT | Slight accuracy loss | Our content classifier |
| 9 | Reimers & Gurevych 2019, Sentence-BERT (EMNLP) | STS, NLI | Siamese BERT sentence embeddings | Similarity search 65 h → 5 s | Bi-encoder cannot model contradiction | MiniLM embeddings for retrieval; cross-encoder for stance |
| 10 | He et al. 2023, DeBERTaV3 (ICLR) | GLUE, MNLI | ELECTRA-style pre-training + disentangled attention | [fill] MNLI acc. | Base/large too big for 4 GB | `nli-deberta-v3-small` stance model |
| 11 | Ribeiro et al. 2016, LIME (KDD) | Any classifier | Local surrogate explanations | User-study gains in trust | Unstable across samples; cost | Token highlights labelled "style only" |
| 12 | Lundberg & Lee 2017, SHAP (NeurIPS) | Any model | Shapley-value attributions | Consistent attributions | Slow for text | Optional report figures |
| 13 | Vosoughi, Roy & Aral 2018 (Science) | 126k Twitter cascades | Empirical spread analysis | Falsehood 70% more likely retweeted; 6× faster to 1,500 people | Not a detector | Motivation statistics |
| 14 | Hassan et al. 2017, ClaimBuster (KDD) | Debate sentences | Check-worthiness classifier (SVM/NB) | [fill] | Political debates only | Claim heuristics; optional classifier |
| 15 | Arslan et al. 2020, ClaimBuster benchmark (ICWSM) | 23.5k sentences, CFS/UFS/NFS | Benchmark + classifiers | [fill] | Same | D6 dataset |
| 16 | Augenstein et al. 2019, MultiFC (EMNLP) | 34.9k claims, 26 fact-checkers | Evidence-based claim verification | [fill] | Heterogeneous labels | Real-world fact-checker labels as ground truth |
| 17 | Liu et al. 2020, KGAT (ACL) | FEVER | Kernel graph attention over evidence | ~70 FEVER score | Large model; Wikipedia only | Ceiling for evidence-based verification |
| 18 | Guo, Schlichtkrull & Vlachos 2022, survey (TACL) | — | Claim detection → retrieval → verdict → justification framework | — | — | Our pipeline follows this framework |
| 19 | Zhou & Zafarani 2020, survey (ACM CSUR) | — | Taxonomy: knowledge / style / propagation / source | — | — | Positions us as knowledge + style hybrid |
| 20 | Islam et al. 2020, infodemic (AJTMH) | Social media Jan–Apr 2020 | Content analysis | ≥ 800 deaths, ~5,800 hospitalisations from one rumour | Not a detector | Motivation |

**Artefacts**

| Artefact | What it shows |
|---|---|
| `paper/main.tex` §II | Themed survey with the comparison table and gap paragraph |
| `paper/refs.bib` | ≥ 15 BibTeX entries, IEEE style via `IEEEtran` bibliography |
| `docs/literature_table.md` | The table above, fully filled |

**Acceptance criteria / Definition of Done**

- [ ] ≥ 15 references in `refs.bib`, each cited at least once in §I–§II; no `[fill]` cells left in the paper's table.
- [ ] Five themes, each with ≥ 2 papers; the survey ends with an explicit gap statement matching our contributions.
- [ ] Every "Reported result" cell is copied from the paper with page/table reference in the team's notes.
- [ ] Each member can explain two papers in 1 minute each (assign: Navishka — LIAR, WELFake; Naitik — FakeBERT, DistilBERT; Naveen — FEVER, KGAT; Prateek — ISOT, Sentence-BERT; Nikhil — LIME, Vosoughi).
- [ ] No sentence pasted from a source (paraphrased from notes).

**How the examiner can verify it in ≤ 3 minutes.** Open `paper/main.pdf` §II and `docs/literature_table.md`; pick any two rows and ask the questions below; `grep -c "@" paper/refs.bib` ≥ 15.

**Likely viva questions**

1. *What does LIAR show?* That 6-way truthfulness from text alone is ~27% — fine-grained truth needs evidence, not style.
2. *What is FEVER and why is it relevant?* 185k Wikipedia-grounded claims labelled SUPPORTS/REFUTES/NEI with a retrieve-then-verify pipeline; it defines the stance task and gives us an evaluation set.
3. *FakeBERT reports 98.9% — why do you not just use that?* It is binary, in-domain, and does not audit leakage; our cross-dataset experiment shows such numbers collapse across datasets.
4. *What is the difference between style-based and knowledge-based detection (Zhou & Zafarani)?* Style uses how it is written; knowledge compares claims to external facts. We use knowledge to decide and style only to adjust confidence.
5. *Which paper is closest to your system?* The Guo et al. 2022 framework (claim detection → evidence retrieval → verdict → justification) and FEVER-style pipelines; our additions are the news domain, temporal check, the 5-class scheme and laptop-scale deployment.
6. *Why LIME rather than attention weights?* LIME is model-agnostic and gives token-level attributions that users understand; attention is not a faithful explanation.

# 4. Research paper at MSE1 — expected state

| Item | Expectation at MSE1 |
|---|---|
| Location | `paper/main.tex`, `paper/refs.bib`, `paper/figures/` (empty or pipeline sketch); built by `make paper` → `paper/main.pdf` |
| Template | IEEEtran conference, two-column (Overleaf "IEEE Conference Template"); final paper 6–8 pages |
| §I Introduction | ≈ 1 page: motivation with 3 cited statistics, gap, problem statement, 3 contribution bullets |
| §II Literature Survey | ≈ 1–1.5 pages: five themes, 15–25 references, comparison table (from `docs/literature_table.md`), gap paragraph |
| §III–VIII | Section headings present with a one-line placeholder each (due MSE2/ESE per master doc §20) |
| Abstract | Placeholder only (ESE draft, week 14) |
| Plagiarism | Written from notes; first Turnitin check is week 14 (< 10% target) |

# 5. Viva demo script (10 minutes)

| Min | Criterion | Presenter | Show | Say |
|---|---|---|---|---|
| 0–1.5 | Problem Identification | **Nikhil** | Master doc §3, §8 (printed) / slides 1–4 | Problem sentence; the 5 classes with one example; "evidence decides, style advises"; UNVERIFIABLE is intended |
| 1.5–2.5 | Architecture | **Prateek Srivastava** | Slide with the §9 pipeline; `Makefile` targets | Input → claims → evidence → stance → temporal → fusion → explanation; `make data / train / eval` reproducibility |
| 2.5–4 | Dataset | **Navishka Sharma** | `data/README.md`; `notebooks/00_datasets.ipynb` count cells | Why four datasets; sizes match; LIAR 6→5 mapping; licences |
| 4–5.5 | Data Preprocessing | **Navishka Sharma** | `make data` log (pre-run) ; `01_eda.ipynb` before/after cell | Steps; Reuters-prefix removal; zero cross-split duplicates; stratified 80/10/10, seed 42 |
| 5.5–7 | EDA | **Navishka Sharma** (figures) + **Naveen** (NER/date figures → claims & temporal) | `eda_top_ngrams_before/after.png`, `eda_length_hist.png`, `eda_ner_types.png`, `eda_isot_dates.png`; `docs/eda_summary.md` | One insight per figure and the design decision it caused |
| 7–8.5 | Model Identification | **Naitik Kukreja** (ladder, DistilBERT, LR baseline) + **Naveen** (NLI stance, MiniLM) | `02_baselines.ipynb` metrics + coefficients; `docs/results/classifier_table.md`; §12 tables | Why baselines first; LR val macro-F1; why DistilBERT on 4 GB; why NLI cross-encoder; not an LLM |
| 8.5–10 | Paper §I–II | **Nikhil** | `paper/main.pdf` p.1–2; `docs/literature_table.md` | Contributions; five survey themes; two papers in depth (LIAR, FEVER); the gap |

Rules: laptop pre-opened with all notebooks executed and figures visible; `make data` log saved to `docs/mse1_make_data.log` in case the live run is not possible; every member answers questions on any module (backup owners per master doc §23).

# 6. Pre-MSE1 checklist (the day before)

- [ ] Fresh-clone test on one member's laptop: `make setup` → copy raw data → `make data` → three notebooks "Run All" without error.
- [ ] `ls docs/figures/eda_*.png | wc -l` ≥ 10; `docs/eda_summary.md` complete.
- [ ] `notebooks/02_baselines.ipynb` shows LR val accuracy/macro-F1; `docs/results/classifier_table.md` has the LR row.
- [ ] `paper/main.pdf` builds; §I–II present; `refs.bib` ≥ 15 entries; `docs/literature_table.md` has no `[fill]` cells.
- [ ] Master doc v2.0 printed (or PDF on a second laptop); `docs/mse1_slides.pdf` exported.
- [ ] Notebook outputs are **saved** (do not enable `nbstripout` before the viva).
- [ ] `data/raw/`, `data/processed/`, `data/models/` are git-ignored; nothing large is committed.
- [ ] `[verify]` items resolved: dataset licences, lynching statistic, satire-domain list (not needed at MSE1).
- [ ] Dry-run viva done (week 6) with the demo script above; each member rehearsed their "why" answers and two literature papers.
- [ ] Evidence-of-work log (§7) filled and printed.
- [ ] Laptop charged; offline copy of everything (no dependency on college Wi-Fi at MSE1).

**Known gaps & honest answers**

| Gap at MSE1 | Honest answer if asked |
|---|---|
| DistilBERT not yet trained | "Identified and budgeted (§12.3: ~1–1.25 h for 3 epochs on the RTX 3050); training is the MSE2 criterion. The LR baseline proves the data pipeline." |
| No GridSearchCV / sweep results | "Search spaces are fixed (§12.2); runs start in week 7 and are reported at MSE2 (Hyperparameter Tuning)." |
| No retrieval / stance / temporal / fusion code | "Designed in §13–14 with decision table R0–R9; implementation is weeks 7–10. The NLI model is pre-trained, so stance needs evaluation, not training." |
| No UI / API / Docker / deployment | "ESE deliverables; architecture and endpoints are specified (§16–17)." |
| Live Claims Set (D7) not started | "Starts week 9 to keep claims recent (Aug–Nov 2026); ≥ 60 by MSE2, ~100 by ESE." |
| Only WELFake baseline, not ISOT | "ISOT baseline + cross-dataset results are the MSE2 Result Analysis item; the leakage audit is already in the EDA." |
| Optional items (Bi-LSTM, ClaimBuster classifier, SHAP, Ollama) absent | "Marked optional/stretch in scope; only after the core Definition of Done." |
| Numbers may still shift | "All numbers are regenerated by `make data` / notebooks with seed 42; values quoted today are from `docs/results/classifier_table.md`." |

# 7. Evidence-of-work log (fill in; one row per artefact or decision)

| Date | Member | Item (what was done) | Link / path | Reviewed by |
|---|---|---|---|---|
| 2026-08-27 | Nikhil | Master doc v2.0 + this MSE1 doc | `docs/FakeNewsDetector_ProjectDetails.md`, `docs/milestones/MSE1_ProjectDetails.md` | — |
| | Prateek Srivastava | Repo skeleton, `requirements.txt`, `Makefile`, `.env.example` | | |
| | Navishka Sharma | Datasets downloaded, `data/README.md`, `00_datasets.ipynb` | | |
| | Navishka Sharma | `make data` pipeline, splits, leakage audit | | |
| | Navishka Sharma | `01_eda.ipynb`, `docs/figures/eda_*.png`, `docs/eda_summary.md` | | |
| | Naitik Kukreja | `02_baselines.ipynb`, LR baseline, `classifier_table.md` row | | |
| | Naveen | Model-identification write-up for stance/retrieval; NER/date EDA insights | | |
| | Nikhil | `paper/main.tex` §I–II, `refs.bib`, `docs/literature_table.md` | | |
| | All | `docs/mse1_slides.pdf`; dry-run viva (week 6) | | |
| | | | | |
