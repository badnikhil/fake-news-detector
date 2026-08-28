# Model Identification (MSE1 criterion, 3 marks) — `docs/model_identification.md`

Fake News & Misinformation Detector · MSE1 artefact for master doc §12 / MSE1 doc §3.5. Companion files:
`notebooks/02_baselines.ipynb` (executed, outputs saved), `src/models/baselines.py` (reusable code + CLI behind
`make train-baselines`), `docs/results/classifier_table.md` (numbers), `data/models/tfidf_lr_welfake_v0.joblib`
(the MSE1 checkpoint), `docs/figures/cm_lr_welfake_val.png`. Written 2026-08-28 from an actual run on the dev laptop; **numbers updated the same evening after the residual-artefact fix in `src/preprocess/artefacts.py`** (the first run's numbers are kept in §4.1 as the "with residual artefacts" reference).

## 1. The model ladder

The system is *evidence-first* (master §14: evidence decides the verdict; the content classifier only moves confidence by
±0.10 and breaks the single-source tie, rule R8). The ladder therefore climbs from the cheapest model that gives a
floor to the strongest model that still fits a 4 GB student GPU — nothing bigger is justified for a component that is
not allowed to decide the verdict on its own.

| # | Tier | Model | Why included (one line) | Role in the system | Status |
|---|---|---|---|---|---|
| 1 | Baseline | **TF-IDF + Multinomial NB** | Trains in 0.2 s; the classic bag-of-words floor every examiner knows | Reference point | **trained (MSE1, val)** |
| 2 | Baseline | **TF-IDF + Logistic Regression** | Strong linear baseline; coefficients are readable → leakage audit; `predict_proba` gives a calibrated-ish confidence for fusion | Reference + interpretable feature analysis; the MSE1 proof-of-feasibility checkpoint | **trained (MSE1, val) → `tfidf_lr_welfake_v0.joblib`** |
| 3 | Baseline | **TF-IDF + LinearSVC** | Usually the best classical text model; 1.8 s fit; CPU-only inference | Best classical; CPU fallback for `MODE=lite` deployment | **trained (MSE1, val)** |
| 4 | Middle (optional) | Bi-LSTM + GloVe 100d | Shows sequence modelling for the comparison table; cheap on the GPU | Only if time permits (week 9); dropped first if behind | MSE2 (optional) |
| 5 | **Final** | **DistilBERT-base-uncased, fine-tuned** | Best accuracy/size trade-off: 66 M params, ~97 % of BERT-base's GLUE score at 60 % of the cost (Sanh et al. 2019); fp16 · max_len 256 · batch 16 fits 4 GB | Content classifier in production (confidence signal in fusion) | MSE2 — configuration fixed in §3 |
| 6 | Fine-grained | DistilBERT (or TF-IDF + LR) on LIAR 3-way (TRUE / MIXED / FALSE) | Gives a truth-*shade* probability for short claims, which the binary WELFake model cannot | Confidence adjustment in fusion for PARTIALLY TRUE / MISLEADING | MSE2 |
| 7 | Stance | **`cross-encoder/nli-deberta-v3-small`** (pre-trained on SNLI / MNLI / FEVER-NLI / ANLI …) | Stance = NLI: entail / contradict / neutral ≡ supports / refutes / neutral, so a ready-made cross-encoder needs **zero training**; ~140 M params but CPU-usable on (claim, passage) pairs | Stance detection; optional fine-tune on the 20k FEVER subset | MSE2 |
| 8 | Retrieval | **`all-MiniLM-L6-v2` + FAISS `IndexFlatIP`** | 22 M-param sentence encoder, ~2 min to embed the ~14k offline statements; cosine search in milliseconds | Offline fact-check index (LIAR + FakeNewsNet-PolitiFact) and passage re-ranking | MSE2 |
| 9 | Claims (optional) | TF-IDF + LR on ClaimBuster | Cheap check-worthiness score to combine with the §13.1 heuristics | Claim ranking | MSE2 (optional) |

## 2. Alternatives rejected (and why)

| Alternative | Why not |
|---|---|
| **BERT-base-uncased** (110 M) | 1.7× the parameters and ~1.6× the step time of DistilBERT for ≈ +1–2 points on news-style classification; at fp16 × batch 16 × 256 tokens it needs ~4.5–5 GB with AdamW, i.e. it only runs on our 4 GB card with batch 8 + gradient accumulation and checkpointing (≈ 2× wall time). The classifier is a *confidence* signal, not the verdict — the extra cost buys nothing the fusion can use. |
| **RoBERTa-large / DeBERTa-v3-large** (355 M / 435 M) | Do not fit 4 GB for fine-tuning at any useful batch size; 3–5× slower CPU inference breaks the ≤ 4 GB / laptop-CPU rule (NFR-6). DeBERTa-v3-**small** *is* used where it matters — as the NLI stance cross-encoder — because that is inference-only. |
| **GPT-style LLMs (API or local 7B+)** as the classifier or fact-checker | Cost, hardware (a 7B model in 4-bit already needs the whole GPU), non-reproducible outputs, and — decisive — hallucination: an LLM will invent "evidence", which directly violates NFR-3 / the hard rule "never fabricate evidence, sources, quotes or dates". A small local LLM (Ollama) may only *rephrase* an already-templated explanation (stretch goal, never on the critical path). |
| **Zero-shot prompting / zero-shot NLI as the classifier** | No training data used, so no measurable, seed-fixed number for the paper; sensitive to prompt wording; slower than a fine-tuned 66 M model. Zero-shot NLI is kept **only** for stance, where the three NLI classes *are* the task and FEVER gives a labelled dev set to measure it. |
| **Bi-encoder embeddings (MiniLM) for stance** | Cosine similarity cannot distinguish "X happened" from "X did not happen"; a cross-encoder reads claim and evidence jointly and models contradiction. MiniLM is used only for retrieval, where speed matters. |
| **Char-n-gram TF-IDF, fastText** | Considered as extra baselines; not added at MSE1 because word 1–2-grams already exceed the sanity bar and the GridSearch at MSE2 (§12.2) is the place to add them if they help. |

## 3. DistilBERT configuration (fixed now, run at MSE2) — master §12.2 / §12.3

| Item | Value |
|---|---|
| Checkpoint | `distilbert-base-uncased` (66 M params); tokenizer lower-cases internally, so the cased parquet text is used as-is (head-truncation of `title + text`) |
| Input | `title + [SEP] + text`, **max_len ∈ {256, 512}** (swept; the EDA showed only 19 % of WELFake inputs fit in 256 tokens and 50 % in 512, so 128 — which keeps ~25 % of an article — was dropped from the grid in favour of 512) |
| Learning rate | **lr ∈ {2e-5, 3e-5, 5e-5}** (swept), AdamW, weight decay 0.01, linear schedule, **warmup ratio 0.1** |
| Epochs | **fixed 3**, evaluate every epoch on val, **keep the best epoch by val macro-F1** (subsumes the epochs ∈ {2, 3} choice) |
| Batch | **16** (fp16) at max_len 256; **batch 8 × gradient accumulation 2** (effective 16) at max_len 512; `gradient_checkpointing=True` as the fallback if the 4 GB card still OOMs |
| Precision | **fp16 = True** (`torch.cuda.amp`); the RTX 2050 is Ampere (compute capability 8.6) so tensor-core fp16 is available |
| Sweep | **reduced manual grid of 6 configurations** = 3 lr × 2 max_len, each on a **20 k stratified subset** of WELFake train, seed 42 → `docs/results/distilbert_sweep.md`, `docs/figures/sweep_distilbert_lr.png`. Optuna over the same 6 points is optional, never more. |
| Final model | best config **retrained on the full train split** (48,639 rows after dedup), best epoch kept → `data/models/distilbert-welfake-v1/`; evaluated once on **test** (6,080) → `docs/results/classifier_table.md`, `docs/figures/cm_distilbert_welfake.png`, `roc_*.png` |
| Selection metric | val macro-F1 (accuracy, P, R, ROC-AUC reported alongside) |
| LIAR 3-way head | same recipe with class weights; lr ∈ {2e-5, 3e-5}, epochs ∈ {3, 5}; ~3 min/epoch → `data/models/distilbert-liar3-v1/` |

**Hardware budget (measured on the dev laptop, 2026-08-28):** GPU = **NVIDIA GeForce RTX 2050, 4096 MiB** (torch reports 3.68 GiB usable; compute capability 8.6; driver 595.84), CPU = AMD Ryzen 5 5500H, 8 threads, 13.5 GB RAM (typically only ~4–5 GB free next to a browser), Python 3.11.11, **torch 2.13.0+cu130, `torch.cuda.is_available() == True`**. The docs' "RTX 3050 4 GB" and this RTX 2050 share the same 4 GB Ampere budget, so the §12.3 timings apply unchanged.

| Run | Hardware | Time (master §12.3 estimate) |
|---|---|---|
| One sweep config (20 k subset, 3 epochs, fp16) | RTX 2050 4 GB | ≈ 20 min at max_len 256, ≈ 40 min at 512 → **6 configs ≈ 3 h** |
| Best config retrained on full train (48.7 k × 3 epochs) | RTX 2050 4 GB | ≈ 15–25 min/epoch → **≈ 1–1.25 h** |
| Total DistilBERT budget | RTX 2050 4 GB | **≈ 4–5.5 h GPU** (3 h sweep + 1–2.5 h full retrain depending on the winning max_len), spread over two or three evenings |
| Memory estimate at max_len 256, batch 16, fp16 + AdamW | | weights + grads + Adam states ≈ 1.1 GB, activations ≈ 1.5–2 GB → fits 3.68 GiB; if not, grad-accum 2 (batch 8) then gradient checkpointing |
| CPU-only member fallback | laptop CPU (8 threads) | 10 k subset, max_len 128, 2 epochs ≈ 1.5–3 h; or |
| Colab fallback | Google Colab free T4 (16 GB) | same script, batch 32 works; download the checkpoint to `data/models/` and share via the HF Hub — the other members only run inference |
| TF-IDF baselines (this document) | laptop CPU | WELFake: TF-IDF 47–51 s, NB 0.2 s / LR 4.3 s / SVM 1.7 s (≈ 60 s total, peak RSS 2.6 GB); ISOT: ≈ 25 s total. Full `GridSearchCV` at MSE2: 30–60 min |

## 4. MSE1 baseline results (WELFake and ISOT, **validation** split; test untouched until MSE2)

Configuration = one point of the §12.2 grid: word 1–2-grams, `sublinear_tf=True`, `max_features=100 000`, `min_df=2`,
`max_df=0.95`, lower-casing inside the vectoriser only; NB `alpha=1`, LR `C=1` (liblinear), LinearSVC `C=1`; input =
`title + "\n" + text`; one TF-IDF matrix per dataset shared by the three classifiers; seed 42; split IDs read from
`data/splits/` (never re-split). Full table with confusion matrices: `docs/results/classifier_table.md`.

| Model | Dataset (train / val rows) | Acc | P (macro) | R (macro) | **Macro-F1** | Confusion [[TN, FP], [FN, TP]] (fake = positive) | Train time |
|---|---|---|---|---|---|---|---|
| TF-IDF + Multinomial NB | WELFake (48,639 / 6,080) | 0.8633 | 0.8605 | 0.8632 | **0.8616** | [[2965, 465], [366, 2284]] | 47 s (tfidf 46.6 + 0.2) |
| **TF-IDF + Logistic Regression** | WELFake | **0.9327** | 0.9315 | 0.9317 | **0.9316** | [[3223, 207], [202, 2448]] | 51 s (tfidf 46.6 + 4.1) |
| TF-IDF + LinearSVC | WELFake | 0.9492 | 0.9480 | 0.9488 | **0.9484** | [[3265, 165], [144, 2506]] | 48 s (tfidf 46.6 + 1.7) |
| TF-IDF + Multinomial NB | ISOT (30,053 / 3,757) | 0.9542 | 0.9539 | 0.9533 | **0.9536** | [[2011, 80], [92, 1574]] | 20 s |
| TF-IDF + Logistic Regression | ISOT | 0.9824 | 0.9830 | 0.9815 | **0.9822** | [[2070, 21], [45, 1621]] | 22 s |
| TF-IDF + LinearSVC | ISOT | 0.9896 | 0.9896 | 0.9894 | **0.9895** | [[2073, 18], [21, 1645]] | 21 s |
| Cross-dataset preview (LR, no adaptation): WELFake → ISOT val | | 0.9782 | | | **0.9778 (see §4.2 — not a real cross-dataset number)** | | — |
| Cross-dataset preview (LR, no adaptation): ISOT → WELFake val | | 0.8299 | | | **0.8299** | | — |
| *First run, before the residual-artefact fix (v0-morning, for the record)* | WELFake (48,677 / 6,085) | NB 0.8639 · LR 0.9471 · SVM 0.9615 | | | NB 0.8620 · **LR 0.9462** · SVM 0.9610 | | |

**MSE1 sanity bar: LR macro-F1 on WELFake val = 0.9316 ≥ 0.90 → met.** Removing the residual publisher/format artefacts cost the
LR 1.5 points (0.9462 → 0.9316) and the SVM 1.3 points (0.9610 → 0.9484) — that difference *is* the amount of in-domain accuracy that came
from boiler-plate rather than content, and it is the first data point of the MSE2 artefact-removal ablation. Learning curve (LR, same
TF-IDF): val macro-F1 0.854 (1k) → 0.862 (2k) → 0.888 (5k) → 0.905 (10k) → 0.917 (20k) → 0.932 (48.6k) — still rising slowly at the full
train size, i.e. the linear model is not yet data-saturated but the last doubling buys ≈ 1.5 points.

### 4.1 What a ≥ 0.9 macro-F1 from TF-IDF + LR tells us — and why it is *not* the answer

1. **The task, as posed by WELFake/ISOT, is largely a *style* task.** The top LR coefficients (notebook §2.2) are register
   markers, not facts: towards *fake* — `the us`, `hillary`, `2016`, `this`, `just`, `october`, `breaking`, `obama`, `anti`, `you`,
   `watch`, `gop` …; towards *real* — `said` (−13.3), `president donald` (−9.8), `said on`, `said in`, weekday tokens (`on tuesday`,
   `on thursday` …), `mr`, `ms`, `percent`, `he said`. A model that keys on "said on Tuesday" vs. "BREAKING: … just …" separates
   wire-service prose from hyper-partisan blog prose; it does not know whether the *claim* is true. A well-written false story in
   wire style would pass, a sloppy true story would fail.
2. **Leakage-token check — before and after the fix (reported, not hidden).** The *first* MSE1 run (morning of 2026-08-28)
   was **not** clean: of the top-25 LR tokens, `via` (#1, 12.1), `video` (#2, 10.5), `image` (→ fake) and `breitbart` (#1, −14.0),
   `on twitter`, `follow`, `twitter`, `york times` (→ real) were publisher/format boiler-plate; LinearSVC additionally surfaced
   `pic twitter`, `https`, `pic`, `told reuters`, `via breitbart`. Where they came from in the then-processed train text:
   * `[VIDEO]` / `(VIDEO)` / `(IMAGE)` title tags: 1,049 fake titles (4.9 %) vs 0 real — a source-site template, not stripped by `artefacts.py`;
   * photo-credit lines "Photo: X **via** Getty Images", "**Via:** Vocativ.com", "Featured **image**: Flickr" (the regex strips only "Featured image via …"): `via` 18.3 % of fake docs vs 2.8 % real, `getty` 3.3 % vs 0.04 %, "featured image" still in 951 fake docs;
   * "Watch it **below**:" / "listen in the player below" embed captions: 12.3 % fake vs 3.2 % real;
   * real-class residue from WELFake's Breitbart share: the title suffix " - **Breitbart**" (2,683 real docs, 9.8 % vs 3.2 %), the byline "**Follow** Trent Baker **on Twitter**", and embedded-tweet stubs left after URL removal — "`pic. twitter. - Name ( ) May 20, 2017`" (341 real docs) and a dangling "`https: .`" (238 real docs);
   * "told **Reuters**" in prose (7.8 % real vs 0.2 % fake) — kept deliberately (`data/README.md`), but it *is* a source cue.
   **Fix applied the same day** (`src/preprocess/artefacts.py`, 26 text + 2 title patterns, 44 unit tests in `tests/test_artefacts.py`;
   `make data` re-run twice — byte-identical — then `make train-baselines` and this notebook re-executed): title tags `[VIDEO]`/`(IMAGES)`/`[TWEETS]`,
   trailing outlet suffixes (" - Breitbart", " - The New York Times", closed list), photo/image credits ("Photo: … via Getty Images", "(Photo by AFP)",
   "Image credit: …"), "Via:" / "Source:" / "For entire story:" source lines, embed captions ("Watch it below:", "Here's the video via YouTube"),
   "Follow … on Twitter" / "… is a reporter for Breitbart" bylines, 21WIRE "SUPPORT / Continue this story" stubs, and the `pic. twitter.` /
   `https: .` / `( )` / tweet-time-stamp debris.
   **Result after the fix (top-25 LR, WELFake val):** every *format / boiler-plate* token is gone — no `video`, `image`, `getty`, `photo`,
   `featured`, `on twitter`, `follow`, `twitter`, `york times`, `pic`, `https`. Three tokens on the leakage watch-list still appear, and we
   keep them **on purpose** because they are source names used *inside sentences*, i.e. content, not boiler-plate: `reuters` (−4.6; "…
   told Reuters", "surveyed by Reuters" — 14.8 % of real vs 1.4 % of fake train docs), `breitbart` (−6.1) / `breitbart news` (−4.0)
   ("Breitbart News reported", "Steve Bannon, executive chairman of Breitbart News" — 4.3 % vs 2.2 %; WELFake's real class contains Breitbart's
   own articles, which cite themselves), and `via` (#22, 3.3; "announced via Twitter", "via the National Organic Program" — 7.5 % vs 2.8 %).
   Stripping those would remove ordinary words from ordinary sentences, so the honest statement is: *the classifier still knows which
   outlets an article talks about, and that correlates with the label in this corpus.* The MSE2 cross-dataset study (WELFake∖ISOT ↔ ISOT,
   §4.2) and the ablation "with vs. without artefact removal" quantify what is left; the first-run numbers above (LR 0.9462 / SVM 0.9610)
   are the "with residual artefacts" arm, the current ones (0.9316 / 0.9484) the "without" arm.
3. **The cross-dataset numbers need care — ISOT is inside WELFake.** In-domain ISOT stays at 0.98–0.99 macro-F1 after
   dateline removal because its two classes come from disjoint outlets. But the WELFake-trained LR applied to ISOT val scores
   **0.9778** — *higher* than on its own val — while ISOT → WELFake val drops to **0.8299**. The asymmetry is explained by a
   hash join done for this document (notebook §5.1): **37,542 of the 37,567 processed ISOT articles (99.93 %) occur verbatim in
   processed WELFake** (WELFake = Kaggle + McIntire + *Reuters/ISOT* + BuzzFeed Political, Verma et al. 2021), and 2,997 of the
   3,757 ISOT-val rows are in WELFake *train*. So "WELFake → ISOT" is in-domain (even train → train) and is **not** a valid
   cross-dataset test; "ISOT → WELFake" is the only informative direction and its 0.83 is the honest out-of-domain number for a
   bag-of-n-grams model. See §4.2 for the change to the evaluation plan (now written into master §18 / §21.2 and the MSE2 doc).
4. **Consequence for the design (unchanged):** the classifier is one signal with a ±0.10 say in the confidence (§14); the verdict
   comes from retrieved evidence + NLI stance + the temporal check. DistilBERT is expected to add a few points in-domain and to
   degrade more gracefully across datasets (sub-word context instead of n-gram identity) — that is what the MSE2 comparison
   table must show; if it cannot beat TF-IDF + SVM out-of-domain, the SVM stays as the `MODE=lite` classifier.

### 4.2 Finding: ISOT ⊂ WELFake — consequence for the MSE2 cross-dataset plan (master §18 / §21.2)

| Check (processed parquets, normalised text SHA-1) | Result |
|---|---|
| ISOT rows whose text also appears in WELFake | **37,542 / 37,567 = 99.93 %** |
| WELFake rows involved | 37,542 / 60,799 = 61.8 % |
| ISOT **val** rows present in WELFake **train** | 2,997 / 3,757 (79.8 %) |
| ISOT **test** rows present in WELFake **train** | 2,998 / 3,757 (79.8 %) |
| WELFake rows **not** in ISOT (= the second domain, WELFake∖ISOT) | **23,257 / 60,799 = 38.3 %** |

Consequences: (a) the planned "train WELFake → test ISOT" row is *not* cross-dataset and must not be reported as such;
(b) the informative direction is ISOT → WELFake, and even WELFake val/test are ~62 % ISOT-derived; (c) a clean design for
`docs/results/cross_dataset.md` is **WELFake∖ISOT (23,257 rows: Kaggle + McIntire + BuzzFeed) ↔ ISOT**, using the hash join
(SHA-1 of `re.sub(r"\W+", " ", text.lower())`, plus MinHash near-duplicates) to define the two disjoint domains, with the
artefact-removal ablation on top; (d) the "ISOT ~99 % is a leakage story" narrative (§18) still holds — but the leakage is *source
style*, and now also *dataset overlap*; (e) the paper's dataset section states the overlap (§II already does; §III at MSE2).
`src/models/baselines.py` marks the pending row accordingly; master §10/§18/§21.2, the MSE2 doc and the ESE doc were updated on
2026-08-28.

## 5. Artefacts produced at MSE1 (all reproducible with `make train-baselines` / `make nb-run-baselines`)

| Path | Content |
|---|---|
| `src/models/baselines.py` (+ alias `tfidf_baselines.py`) | `build_pipeline`, `load_split` (joins saved IDs), `fit_eval` / `fit_eval_all` (one TF-IDF, three classifiers, val metrics + confusion), `top_coefficients`, `find_leak_tokens`, `save_pipeline` / `load_pipeline`, `render_classifier_table`; CLI `python -m src.models.baselines --dataset welfake --model lr` |
| `data/models/tfidf_{nb,lr,svm}_{welfake,isot}_v0.joblib` | fitted `Pipeline(tfidf → clf)`; `tfidf_lr_welfake_v0.joblib` is the canonical MSE1 checkpoint (1.9 MB); reload = `joblib.load(path).predict([text])` |
| `docs/results/train_logs/baselines_{dataset}_{model}_val.json` | metrics, confusion, timings, parameters per run (source of the table) |
| `docs/results/classifier_table.md` | MSE1 rows filled; MSE2 rows marked pending |
| `docs/figures/cm_lr_welfake_val.png`, `cm_baselines_welfake_val.png`, `train_lr_welfake_learning_curve.png` | confusion matrices; learning curve |
| `notebooks/02_baselines.ipynb` | executed end-to-end (133 s in-kernel; budget ≤ 10 min) |
| `tests/test_models.py` | 2k-sample train/predict for NB/LR/SVM, coefficient + leak-check helpers, joblib round-trip, saved-artefact reload |
| `tests/test_artefacts.py` | 44 unit tests for the residual-artefact regexes (credits, captions, bylines, title tags, outlet suffixes, tweet debris; prose uses of *via* / *photo* / *source* must survive) |

## 6. Viva one-liners

* *Why baselines first?* Minutes to train, a floor (0.932 LR / 0.948 SVM macro-F1 after artefact removal; 0.946 / 0.961 before), interpretable coefficients that exposed the residual artefacts above — and led to the same-day fix — and a CPU fallback for deployment.
* *Why DistilBERT and not BERT-base?* 66 M vs 110 M params, ~60 % faster, ~97 % of the accuracy; fits fp16 × batch 16 × 256 tokens on our 4 GB RTX 2050; CPU inference < 1 s per article.
* *Why NLI for stance and a cross-encoder?* Stance *is* NLI (supports / refutes / neutral); a cross-encoder reads claim and evidence jointly and can model negation, a bi-encoder cannot.
* *Why not an LLM?* Cost, hardware, non-reproducibility and hallucinated evidence (NFR-3). Allowed only to paraphrase a template explanation, as a stretch.
* *What does 0.93–0.95 on WELFake mean?* That style separates the two sources in this corpus — not that the model knows the truth; 1.5 of those points were pure boiler-plate (gone after the fix), and source names inside sentences (`reuters`, `breitbart`) still carry label information; hence evidence-first fusion and the cross-dataset / artefact ablation at MSE2.
* *Which leakage tokens are left and why?* `reuters`, `breitbart` and `via` inside ordinary sentences ("told Reuters", "Breitbart News reported", "announced via Twitter"). They are content words that happen to correlate with the label in this corpus; stripping them would mean deleting words from sentences, so they stay and are disclosed.
