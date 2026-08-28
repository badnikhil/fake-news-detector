---
title: "MSE 2 — Project Details & Verification"
subtitle: "Fake News & Misinformation Detector (evidence-grounded, explainable 5-class news verification)"
date: "v1.0 — 27 August 2026 (prepared for the MSE 2 window, weeks 12–13, 2–15 Nov 2026)"
---

# 1. Title Block

| Field | Value |
|---|---|
| **Project** | Fake News & Misinformation Detector: an evidence-grounded, explainable news verification system |
| **Milestone** | **MSE 2** (Mid-Semester Evaluation 2) — Model Training, Hyperparameter Tuning, Result Analysis; Paper Methodology & Results |
| **Department / course** | CSE (AI/ML), KIET Group of Institutions — B.Tech semester project (NLP), Aug–Dec 2026 |
| **Team** | Navishka Sharma, Naitik Kukreja, Naveen, Prateek Srivastava, Nikhil |
| **Guide** | `<Guide name, designation>` |
| **Assumed window** | Weeks 12–13, **2–15 Nov 2026** (week 12 = buffer/freeze, week 13 = viva) — *adjust to the official calendar* |
| **Master document** | `docs/FakeNewsDetector_ProjectDetails.md` v2.0 (§12–14, 18, 20–24 are the source for everything below) |
| **This document** | `docs/milestones/MSE2_ProjectDetails.md` v1.0 — bring a printed copy to the viva |

## MSE 2 marking scheme (20 marks)

| Component | Criterion | Marks | Mode |
|---|---|---|---|
| **Project (15)** | Model Training | 5 | viva |
| | Hyperparameter Tuning | 5 | viva |
| | Result Analysis | 5 | viva |
| **Research Paper (5)** | Methodology | 2.5 | viva |
| | Results | 2.5 | viva |

# 2. Milestone Summary

**Recap of MSE 1 (weeks 1–7).** Problem, 5-class scheme (REAL / FAKE / PARTIALLY TRUE / MISLEADING / UNVERIFIABLE) and objectives fixed; WELFake, ISOT, LIAR, FEVER-subset downloaded and schema-unified; `make data` produces cleaned, de-duplicated, artefact-stripped, stratified 80/10/10 splits (seed 42); EDA + leakage audit in `notebooks/01_eda.ipynb`; TF-IDF + LR baseline trained; paper §I–II drafted.

**What exists by MSE 2 (weeks 7–12).** (1) All content classifiers trained on WELFake with pinned configs: TF-IDF + Naive Bayes / Logistic Regression / LinearSVC, (optional Bi-LSTM + GloVe), and the final **fine-tuned DistilBERT** (`data/models/distilbert-welfake-v1`); (2) the **LIAR 3-way head** (TRUE / MIXED / FALSE); (3) the stance model `cross-encoder/nli-deberta-v3-small` **evaluated on a FEVER dev subset** (fine-tune optional); (4) the **offline FAISS index** (LIAR statements + FakeNewsNet PolitiFact titles, MiniLM embeddings); (5) GridSearchCV tables for the classical models and the DistilBERT sweep table; (6) test metrics, confusion matrices, ROC curves, the **cross-dataset WELFake↔ISOT table** with leakage discussion, and an error analysis; (7) online retrieval (DuckDuckGo + Wikipedia, cached), claim identifier v1, temporal check v1 and **fusion v1 (rules R0–R9 + confidence)**; (8) **`POST /analyze` working end-to-end in `MODE=offline`** (claims → offline evidence → stance → temporal → fusion → JSON); (9) paper §III–VI drafted (≈ 5 pages); (10) Live Claims Set ≥ 60 items labelled.

**Deliberately not yet built (scheduled for weeks 13–16, ESE).** Front-end polish, LIME highlights + explanation templates + uncertainty text, Docker image and HF Spaces / Render deployment, full end-to-end 5-class evaluation on the ~100-item Live Claims Set, ablation, temporal test set, latency, case studies, comparison table, Turnitin check and venue submission. Saying this up front is expected and correct.

# 3. Criterion 1 — Model Training (5 marks)

**What we built.** Every model in the §12 ladder trained on identical splits (`data/splits/*.csv`) with fixed seeds (`src/common/seed.py`): TF-IDF + NB / LR / LinearSVC (`src/models/tfidf_baselines.py`); DistilBERT-base-uncased fine-tuned with fp16, max_len 256, batch 16 on the RTX 3050 4 GB (`src/models/distilbert.py`; CPU members use the 10k-subset / max_len 128 path or Colab T4); LIAR 3-way head with class weights (`src/models/liar_head.py`); stance model evaluated on 3k balanced FEVER dev pairs with gold evidence (`src/stance/fever_eval.py`); offline FAISS `IndexFlatIP` over ~14k MiniLM-embedded statements (`src/evidence/offline_index.py`, `make index`). Checkpoints in `data/models/` (git-ignored, mirrored to the team HF Hub account).

**Artefacts**

| Path | What it shows |
|---|---|
| `src/models/tfidf_baselines.py`, `distilbert.py`, `liar_head.py`, `bilstm.py` (opt.) | Training scripts; `make train-baselines`, `make train-distilbert`, `make train-liar` |
| `data/models/tfidf_{nb,lr,svm}.joblib`, `data/models/distilbert-welfake-v1/`, `data/models/distilbert-liar3-v1/` | Saved models; DistilBERT folder has `config.json`, `trainer_state.json` (loss/F1 per epoch), `training_args.json` |
| `docs/results/train_logs/*.json` | Copy of every run's log: seed, config, wall-clock, hardware, per-epoch loss & val macro-F1 |
| `docs/figures/train_*.png` | Training curves (loss and val macro-F1 vs. epoch) for DistilBERT and LIAR head |
| `docs/results/classifier_table.md` | Val + test accuracy / P / R / macro-F1 / ROC-AUC per model |
| `docs/results/stance_fever.md`, `docs/figures/cm_stance_fever.png` | Stance accuracy / macro-F1 / CM on the FEVER dev subset |
| `data/index/` (`*.faiss`, `meta.parquet`) | Offline index (~14k vectors, < 30 MB) built by `make index` |
| `notebooks/03_transformers.ipynb`, `05_stance.ipynb` | Narrated training / stance runs |

**Definition of Done**

- [ ] `make train-baselines` runs on a fresh clone in < 10 min and writes the three `.joblib` files.
- [ ] `data/models/distilbert-welfake-v1/` exists, loads with `AutoModelForSequenceClassification`, and is the model served by `POST /classify`.
- [ ] A training log for **every** model (incl. the LIAR head) is saved under `docs/results/train_logs/` with seed, full config, wall-clock, hardware (GPU/CPU/Colab) and per-epoch metrics.
- [ ] `docs/figures/train_distilbert.png` shows train loss decreasing and val macro-F1 per epoch; best epoch marked.
- [ ] Stance model evaluated on the 3k FEVER dev subset; table + CM saved; accuracy reported against the O4 target (≥ 85 %).
- [ ] `make index` builds `data/index/` in < 5 min; a query returns top-5 hits with cosine scores and record IDs.
- [ ] All models use the same `data/splits/` IDs (assert in scripts; no train/test overlap).

**Verify in ≤ 3 minutes.** (1) `cat docs/results/classifier_table.md` — five rows, all metrics filled. (2) `cat data/models/distilbert-welfake-v1/trainer_state.json | head -60` — epochs, loss, `eval_macro_f1`. (3) Open `docs/figures/train_distilbert.png`. (4) `MODE=offline make run`, then `curl -s -X POST localhost:8000/classify -H 'content-type: application/json' -d '{"text":"<paste a headline>"}'` — returns `p_fake` from the checkpoint.

**Likely viva questions**

1. *Why DistilBERT and not BERT-base / RoBERTa?* 66 M params (40 % smaller, ~60 % faster than BERT-base, ~97 % of its accuracy); fits a 4 GB GPU at max_len 256 / batch 16 in fp16; ≤ 1 s CPU inference for the app.
2. *Why fp16, max_len 256, batch 16?* fp16 halves activation memory and uses tensor cores; the EDA length histogram shows the title + first ~200 word-pieces carry the signal, and 512 would double memory and time; 16 is the largest batch that fits — `gradient_accumulation_steps=2` (effective 32) or gradient checkpointing if OOM.
3. *What do warmup and weight decay do?* Warmup (6–10 % of steps) ramps the LR from 0 so large early Adam updates do not wreck the pretrained weights; weight decay 0.01 is L2-style regularisation (excluding bias/LayerNorm).
4. *Early stopping?* Val macro-F1 evaluated every epoch, best checkpoint kept (`load_best_model_at_end`), stop when no improvement — 2–3 epochs suffice for 58k examples; more epochs overfit.
5. *How do you guarantee identical splits across models?* Stratified 80/10/10 with `random_state=42`, saved as ID lists in `data/splits/`; every script loads those IDs and asserts zero overlap.
6. *How did CPU-only members train?* 10k subset, max_len 128, 2 epochs (1.5–3 h), or Colab free T4; checkpoints shared via HF Hub — others only need inference.
7. *Was the stance model trained?* No — `nli-deberta-v3-small` is pre-trained on SNLI/MNLI/FEVER-NLI; we evaluate it (premise = evidence, hypothesis = claim; entailment/contradiction/neutral → SUPPORTS/REFUTES/NEUTRAL). A 1-epoch fine-tune on 20k FEVER pairs is optional.
8. *What is in the offline index and why?* 12,836 LIAR statements with PolitiFact rulings + ~1,056 FakeNewsNet PolitiFact titles, 384-d `all-MiniLM-L6-v2` vectors, FAISS `IndexFlatIP` on normalised vectors (= cosine), top-5 with cosine ≥ 0.55 — so the demo works with no internet.

# 4. Criterion 2 — Hyperparameter Tuning (5 marks)

**What we built.** `GridSearchCV` (5-fold, on train) for NB / LR / SVM over `ngram_range ∈ {(1,1),(1,2)}`, `max_features ∈ {50k,100k,200k}`, `sublinear_tf ∈ {T,F}`, `C ∈ {0.1,1,10}` (LR/SVM), `alpha ∈ {0.1,0.5,1.0}` (NB); a DistilBERT sweep on a 20k stratified subset over `lr ∈ {2e-5,3e-5,5e-5} × epochs ∈ {2,3} × max_len ∈ {128,256}` (warmup ratio {0.06, 0.1}, weight decay 0.01, batch 16; ≤ 6 Optuna trials or a reduced manual grid), winner retrained on full train; LIAR head over `lr ∈ {2e-5,3e-5} × epochs ∈ {3,5}`; fusion thresholds (τ, min evidence count, N months) grid-searched on a 30-item dev slice of the Live Claims Set. Selection metric everywhere: **validation macro-F1**.

**Artefacts**

| Path | What it shows |
|---|---|
| `docs/results/gridsearch_{nb,lr,svm}.csv` | Every configuration tried with mean/std CV macro-F1, fit time; best row flagged |
| `docs/results/distilbert_sweep.md` | Every DistilBERT run: lr, epochs, max_len, warmup, seed, val macro-F1, minutes, hardware; winner + full-train retrain row |
| `docs/figures/sweep_*.png` | Val macro-F1 vs. lr (grouped by max_len / epochs); GridSearch heat-map for SVM |
| `src/fusion/fusion.yaml` + `docs/results/fusion_thresholds.md` | Chosen τ / min-evidence / N-months and the dev-slice grid results |
| `notebooks/04_tuning.ipynb` | Narrated sweep, plots, val-vs-test gap table |

**Definition of Done**

- [ ] Tuning tables list **every** configuration tried (no cherry-picking) with val macro-F1; the winner is marked and its val-vs-test gap is reported.
- [ ] `gridsearch_*.csv` are the raw `cv_results_` exports (36 rows per LR/SVM grid, 18 for NB).
- [ ] `distilbert_sweep.md` includes the lr = 5e-5 runs and a one-line explanation of their behaviour.
- [ ] The config used for `distilbert-welfake-v1` equals the sweep winner (stated in `train_logs`).
- [ ] `fusion.yaml` values match `docs/results/fusion_thresholds.md`; dev slice IDs are disjoint from evaluation items.
- [ ] `make eval-models` regenerates the sweep figures from the CSV/MD tables.

**Verify in ≤ 3 minutes.** `column -s, -t docs/results/gridsearch_svm.csv | sort -k<macroF1 col> -r | head`; `cat docs/results/distilbert_sweep.md`; open `docs/figures/sweep_distilbert_lr.png`; `cat src/fusion/fusion.yaml`.

**Likely viva questions**

1. *Why GridSearchCV for classical models but a small sweep for DistilBERT?* A classical fit costs 1–5 min, so a 5-fold grid (36 configs) finishes in 30–60 min; a DistilBERT run costs ~20 min even on the 20k subset, so we sweep only the 3 hyperparameters that matter most and retrain the winner on full data.
2. *Which config won and why?* `<fill from distilbert_sweep.md>` — explain via the curve: e.g., lr 3e-5 / 3 epochs / 256 gives the best val macro-F1 with a small val-test gap.
3. *What happened at lr = 5e-5?* `<fill>` — typically noisier or lower val F1 / occasional divergence under fp16; that is why 2e-5–3e-5 is the safe band for fine-tuning.
4. *Why select on validation, never on test?* Test is touched once, at the end; selecting on it leaks hyperparameters and inflates the reported number. The val-vs-test gap is our over-fitting check.
5. *Why sweep on a 20k subset?* Ranking of configs on a stratified subset is a good proxy; it cuts a 12-run sweep from ~12 h to ~4 h; the winner is retrained on all 58k.
6. *What does `sublinear_tf` / `C` / `alpha` do?* `1+log(tf)` damps very frequent terms in long articles; `C` is the inverse regularisation strength; `alpha` is Laplace smoothing for unseen words.
7. *How were fusion thresholds tuned?* Grid over τ, min evidence count and N months on a 30-item dev slice, maximising 5-class macro-F1; small slice, so we report it as preliminary and freeze the values in `fusion.yaml`.

# 5. Criterion 3 — Result Analysis (5 marks)

**What we built.** Held-out test metrics for NB, LR, SVM, DistilBERT (and Bi-LSTM if trained) on WELFake and ISOT; confusion matrices and ROC curves; the **cross-dataset table** (train WELFake → test ISOT and vice versa, with and without artefact removal) and a **leakage audit** (an "artefact-only" classifier on ISOT datelines / source tokens); LIAR 3-way head vs. Wang (2017); stance metrics on FEVER; error analysis of ≥ 20 misclassified WELFake test items categorised (truncation, satire, opinion, label noise, artefact-driven, near-duplicate).

**Artefacts**

| Path | What it shows |
|---|---|
| `docs/results/classifier_table.md` | Accuracy / precision / recall / macro-F1 / ROC-AUC, test split, all models, WELFake and ISOT |
| `docs/results/cross_dataset.md` | WELFake↔ISOT F1 matrix, with/without artefact removal; artefact-only classifier accuracy; discussion |
| `docs/figures/cm_*.png`, `docs/figures/roc_*.png` | Confusion matrices and ROC curves per model/dataset |
| `docs/results/liar_head.md` | 3-way (and mapped 6-way) accuracy / macro-F1 vs. Wang (2017) |
| `docs/results/stance_fever.md` | Stance accuracy / macro-F1 / CM |
| `docs/results/error_analysis.md` | ≥ 20 misclassified items: id, truncated text, true, predicted, p_fake, category, comment |
| `src/eval/eval_models.py`, `src/eval/eval_stance.py` | Scripts behind `make eval-models` |

**Definition of Done**

- [ ] Test-set table complete for NB, LR, SVM, DistilBERT on **both** WELFake and ISOT (in-domain).
- [ ] Cross-dataset table has all four cells (W→W, W→I, I→W, I→I) × {raw, artefact-removed}; F1 drop quantified in one sentence.
- [ ] Artefact-only classifier accuracy on ISOT reported (expected very high) and discussed as leakage.
- [ ] CM + ROC figures exist for every model; DistilBERT CM annotated with counts and percentages.
- [ ] `error_analysis.md` has ≥ 20 categorised examples and a category-frequency table.
- [ ] Objective O1 checked: DistilBERT macro-F1 ≥ 0.95 on WELFake test (or the shortfall explained).
- [ ] `make eval-models` regenerates every number and figure above from the saved checkpoints.

**Verify in ≤ 3 minutes.** `cat docs/results/classifier_table.md docs/results/cross_dataset.md`; open `docs/figures/cm_distilbert_welfake.png`; `head -40 docs/results/error_analysis.md`; optionally `make eval-models` (start before the viva; ≈ 10 min on CPU).

**Likely viva questions**

1. *Why does ISOT give ~99 % and why is that suspicious?* ISOT's real articles are all Reuters ("WASHINGTON (Reuters) –" datelines) and its fakes come from a few sites with fixed trailers; a classifier trained only on those artefacts already scores very high, so the number measures source style, not truth.
2. *What does macro-F1 tell you that accuracy does not?* It averages per-class F1 equally, so ignoring a minority class is punished; on balanced WELFake they agree, on LIAR 3-way and the 5-class Live Claims Set (UNVERIFIABLE is rare) they diverge.
3. *What does the cross-dataset drop mean for your project?* Style-based classifiers learn dataset-specific cues; the drop (larger after artefact removal is undone) is the empirical reason our verdict comes from evidence, not the classifier.
4. *Interpret this confusion matrix.* Rows = true, columns = predicted; the fake→real cell is the dangerous one for users (missed fakes); we report it separately as fake-class recall.
5. *How do you get ROC-AUC for LinearSVC without probabilities?* From `decision_function` margins (AUC is threshold-free), or `CalibratedClassifierCV` if calibrated probabilities are needed.
6. *How was the stance model evaluated on FEVER?* 3k balanced dev claims with gold Wikipedia evidence; NEI claims paired with a random sentence from the same page; accuracy / macro-F1 / CM. Caveat stated openly: FEVER-NLI is in the model's pre-training mix, so the score is optimistic for news-domain evidence.
7. *Most common error category and fix?* `<fill from error_analysis.md>` — e.g., long articles truncated at 256 tokens → use head + tail truncation; satire → satire-domain list (rule R0).

# 6. Criterion 4 — Research Paper: Methodology (2.5 marks)

**What we deliver.** `paper/main.tex` §III Datasets & Preprocessing (condensed §10–11, leakage audit), §IV Methodology (pipeline figure; claim identification score `h`; retrieval — ddgs / Wikipedia / Google Fact Check / offline FAISS; stance via NLI cross-encoder; temporal check with N = 6 months; fusion decision table R0–R9; confidence formula), §V Experimental Setup (splits, hyperparameters, hardware, seeds). Owners: Navishka §III, Naitik §IV-models, Naveen §IV-retrieval/stance, Prateek §V; Nikhil assembles.

**Artefacts**

| Path | What it shows |
|---|---|
| `paper/main.tex` §III–V, `paper/refs.bib` | IEEEtran two-column text, ≈ 2.5 pages, ≥ 20 references |
| `paper/figures/pipeline.pdf` | Architecture figure (matches §9 of the master doc and the code) |
| Table: datasets (name, size, labels, role); Table: fusion rules R0–R9 | Reproduced from master §10 and §14.2 |
| `make paper` (latexmk) → `paper/main.pdf` | Builds without errors |

**Definition of Done**

- [ ] Every module named in §IV exists as code under `src/` (claims, evidence, stance, temporal, fusion) — an examiner can cross-check.
- [ ] Rule table in the paper equals `src/fusion/rules.py` and `fusion.yaml` (τ = 0.6, N = 6, k = 5).
- [ ] Pipeline figure has the same boxes as §9 of the master doc; dataset table numbers equal `notebooks/00_datasets.ipynb` output.
- [ ] §V lists seed 42, 80/10/10 split, the winning hyperparameters, RTX 3050 4 GB / Colab T4, library versions.

**Verify in ≤ 3 minutes.** Open `paper/main.pdf` pages 2–4; compare the rule table with `cat src/fusion/fusion.yaml`; check the figure against `docs/FakeNewsDetector_ProjectDetails.md` §9.

**Likely viva questions**

1. *Why does the fusion rule let evidence override the classifier?* The classifier learns writing style and dataset artefacts (see cross-dataset results); evidence + stance checks facts. So the invariant is: evidence decides, the classifier only adjusts confidence ±0.10 and breaks the single-source tie (R8).
2. *What happens when no evidence is found?* R1 fires → UNVERIFIABLE with confidence `1 − max(agreement, 0.2)`, plus the statement "this does not mean the claim is false". The classifier never turns that into FAKE/REAL.
3. *Why rules instead of a learned fusion?* ~100 labelled end-to-end items are too few to learn a 5-class combiner; rules are auditable (`rule_fired` is returned) and match how fact-checkers reason. Learned fusion is a listed stretch goal.
4. *Why an NLI cross-encoder rather than a bi-encoder or LLM?* Cross-encoders read claim and evidence jointly (best accuracy for 3-way stance), run on CPU at ~0.15 s/pair, and cannot hallucinate evidence (NFR-3).
5. *How does the temporal check work?* Old-news flag = recency cue in the text AND (post date − median date of supporting evidence) ≥ 6 months; out-of-context = GPE/DATE entity conflict with ≥ 2 supporting passages; headline–body mismatch via NLI with p ≥ 0.7.
6. *How is PARTIALLY TRUE distinguished from MISLEADING?* PARTIALLY TRUE (R5/R6): main claim supported, another important claim refuted. MISLEADING (R4): main claim supported but a temporal/context flag is set.

# 7. Criterion 5 — Research Paper: Results (2.5 marks)

**What we deliver.** `paper/main.tex` §VI Results with (a) the model-comparison table (test accuracy / P / R / macro-F1 / ROC-AUC for NB, LR, SVM, DistilBERT on WELFake and ISOT), (b) the tuning table (DistilBERT sweep summary; best classical config per model), (c) the cross-dataset table with artefact-removal ablation and the artefact-only baseline, (d) stance results on FEVER, (e) LIAR 3-way results, (f) DistilBERT confusion matrices as figures, (g) preliminary end-to-end results on the ≥ 60 labelled Live Claims items. Every table is generated from `docs/results/*.md`.

**Artefacts**

| Path | What it shows |
|---|---|
| `paper/main.tex` §VI | Tables I–V + Figs (CM, ROC, training curves) |
| `paper/figures/cm_distilbert_welfake.pdf`, `roc_all.pdf`, `sweep_distilbert.pdf` | Exported from `docs/figures/` |
| `docs/results/*.md` | Source of truth for every number in the paper |

**Definition of Done**

- [ ] Every number in §VI equals the corresponding cell in `docs/results/` (spot-check three cells at random).
- [ ] Tables: model comparison, tuning, cross-dataset, stance; Figures: pipeline, CM, ROC, sweep curve.
- [ ] Each table has one paragraph of interpretation (what it shows, what it implies).
- [ ] Preliminary e2e numbers labelled "preliminary (n = 60)"; ESE additions listed in one sentence.
- [ ] Draft ≈ 5 pages total (§I–VI), compiles with `make paper`.

**Verify in ≤ 3 minutes.** Open `paper/main.pdf` §VI; pick any three numbers and `grep` them in `docs/results/classifier_table.md` / `cross_dataset.md`.

**Likely viva questions**

1. *Which of your results are final and which are preliminary?* Classifier, tuning, cross-dataset, LIAR and stance tables are final; e2e / ablation / latency arrive at ESE.
2. *How do you make sure paper numbers match the code?* All tables originate from `make eval-models` output in `docs/results/`; the paper cites the commit hash and the seed.
3. *What is the single most important result so far?* The cross-dataset drop versus in-domain ~99 %: it justifies the evidence-based architecture.
4. *Why report ROC-AUC alongside macro-F1?* AUC is threshold-free and shows ranking quality; macro-F1 shows performance at the operating threshold used in the app.
5. *What will change in §VI by ESE?* e2e 5-class confusion matrix, UNVERIFIABLE precision, ablation (classifier-only / +evidence / +temporal / offline-only), latency, 5 case studies, comparison table.

# 8. Demo Script for the Viva (10 minutes)

Pre-start (before entering): `MODE=offline make run` already up on port 8000; `docs/results/` and figures open in tabs; `paper/main.pdf` open; phone hotspot as backup.

| Time | Presenter | Action |
|---|---|---|
| 0:00–1:00 | Naitik | One slide: MSE 1 recap (3 lines) → what is new for MSE 2 → what is deferred to ESE (§2 above). |
| 1:00–3:00 | Naitik | **Model Training**: `classifier_table.md`; `train_distilbert.png` loss/F1 per epoch; `trainer_state.json`; state config (fp16, 256, 16, seed 42, wall-clock, GPU). |
| 3:00–4:30 | Naitik | **Tuning**: `gridsearch_svm.csv` top rows; `distilbert_sweep.md` — which config won, what lr 5e-5 did, val-vs-test gap; `fusion.yaml`. |
| 4:30–6:00 | Navishka | **Result Analysis**: `cross_dataset.md` and the artefact-only ISOT classifier (leakage story); `cm_distilbert_welfake.png`; two entries from `error_analysis.md`. |
| 6:00–8:00 | Prateek + Naveen | **Live `/analyze` (offline)**: `curl localhost:8000/health` → then `curl -s -X POST localhost:8000/analyze -H 'content-type: application/json' -d '{"input_type":"headline","text":"<LIAR-style claim>","mode":"offline","k":3}'`. Walk the JSON: claims → evidence cards (source, date, cosine, stance p) → `rule_fired` → verdict + confidence. Second call with a hyper-local claim → R1 → UNVERIFIABLE. |
| 8:00–9:00 | Naveen | **Stance + index**: `stance_fever.md` (accuracy vs. O4 target, caveat), `data/index/` stats (14k vectors, < 30 MB, < 2 s query). |
| 9:00–10:00 | Nikhil | **Paper**: `paper/main.pdf` §III–VI — pipeline figure, rule table, results tables; state that every number is traceable to `docs/results/`. Open for questions. |

Everyone answers questions on every module.

# 9. Pre-MSE 2 Checklist (day before) and Known Gaps

**Day-before checklist**

- [ ] Fresh clone → `make setup && make index` succeeds; `MODE=offline make run` answers `/health` on the demo laptop **with Wi-Fi off**.
- [ ] `make eval-models` run once more; `docs/results/` and `docs/figures/` timestamps newer than the last model change.
- [ ] Two demo inputs rehearsed (one that returns evidence from the offline index, one that returns UNVERIFIABLE); expected `rule_fired` written on the slide.
- [ ] `paper/main.pdf` compiled; three numbers spot-checked against `docs/results/`.
- [ ] Printed: this document, `classifier_table.md`, `cross_dataset.md`, `distilbert_sweep.md`, the CM figure.
- [ ] Checkpoints uploaded to HF Hub; `data/models/` present on the demo laptop; `requirements.txt` pinned.
- [ ] Every member can state the five classes, the fusion invariant, and their own module in 30 s.
- [ ] Evidence-of-work log (§10) filled and printed.

**Known gaps and honest answers**

| Likely challenge | Honest answer |
|---|---|
| "The UI is basic / no LIME yet." | Scheduled for weeks 13–14 per the plan; MSE 2 is the modelling milestone. The JSON already contains everything the UI will render. |
| "Only 60 Live Claims items." | Labelling reaches ~100 by week 14; e2e numbers are marked preliminary. |
| "Stance model not fine-tuned." | Pre-trained NLI meets the O4 target on FEVER; fine-tuning is optional and listed as a stretch goal. |
| "WELFake 95 %+ is easy." | Agreed — that is why we report cross-dataset and artefact-only numbers, and why evidence, not the classifier, decides. |
| "Online retrieval untested at scale." | ddgs + Wikipedia work with caching; Google Fact Check key obtained; rate limits mitigated by the SQLite cache and offline mode. |
| "Fusion thresholds tuned on 30 items." | Yes — reported as preliminary; frozen in `fusion.yaml`; revisited with the full Live Claims Set at ESE. |

# 10. Evidence-of-Work Log (template)

| Date | Member | Item | Link / path |
|---|---|---|---|
| 2026-09-30 | Naitik Kukreja | GridSearchCV NB/LR/SVM complete | `docs/results/gridsearch_*.csv` |
| 2026-10-02 | Naveen | Offline FAISS index built | `data/index/`, `src/evidence/offline_index.py` |
| 2026-10-09 | Naitik Kukreja | DistilBERT sweep + full retrain | `docs/results/distilbert_sweep.md`, `data/models/distilbert-welfake-v1/` |
| 2026-10-25 | Prateek Srivastava | Fusion v1 + `/analyze` offline e2e | `src/fusion/`, `src/api/` |
| 2026-11-01 | Nikhil | Paper §III–VI draft; figures | `paper/main.tex`, `paper/figures/` |
| `<date>` | `<member>` | `<item>` | `<path / commit / URL>` |
