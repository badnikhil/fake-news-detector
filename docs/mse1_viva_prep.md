# MSE1 viva prep — one-night cram sheet

Fake News & Misinformation Detector · KIET CSE(AI/ML) · Navishka Sharma, Naitik Kukreja, Naveen, Prateek Srivastava, Nikhil.
**Standalone: everything you need is on this page.** Every number is the measured value from the 28 Aug 2026 run; the source file is named so you can double-check. Print this.

---

## 1. One-minute pitch

**30 seconds.** "We built an evidence-grounded fake-news detector. You paste a headline, an article or a URL; it pulls out the checkable claims, searches free sources plus an offline fact-check index, decides with an NLI model whether each piece of evidence supports or refutes the claim, checks whether old news is being sold as new, and returns one of five verdicts — REAL, FAKE, PARTIALLY TRUE, MISLEADING, UNVERIFIABLE — with a confidence score, evidence cards showing source and date, and a plain-language explanation. Everything runs free on a student laptop."

**2 minutes.** Add: "Most published detectors are binary text classifiers. They learn *writing style* and dataset artefacts, not facts. We proved that on our own data: a classifier that sees *nothing but counts of publisher boiler-plate patterns* — datelines, photo credits, `[VIDEO]` tags — gets **99.73 %** on raw ISOT. So a 99 % number means the model found the newsroom's template, not the truth. Three things follow. (1) Binary is the wrong output space: real misinformation is mostly partial truth or true-but-out-of-context, so we use five classes. (2) The system must be able to say **UNVERIFIABLE** — forcing FAKE/REAL with no evidence is itself misinformation. (3) **Evidence decides the verdict; the classifier only moves confidence by ±0.10 and breaks a single-source tie (rule R8).** At MSE1 we have the data pipeline, the EDA, the leakage audit and the TF-IDF baselines; DistilBERT, retrieval, stance and the app are MSE2/ESE."

---

## 2. Numbers to memorise

| # | Thing | Value | Source file |
|---|---|---|---|
| 1 | WELFake raw → processed | **72,134 → 60,799** (dropped: 1,591 empty, 1,047 short, 8,217 exact dups, 480 near dups) | `docs/mse1_make_data.log`, `data/README.md` |
| 2 | WELFake raw labels | 35,028 real / 37,106 fake; CSV `label` **0 = real, 1 = fake** (blurb is wrong) | `data/README.md` |
| 3 | WELFake processed balance | 34,295 real / 26,504 fake = **56.4 % real** | `docs/eda_summary.md` |
| 4 | WELFake splits | train **48,639** / val **6,080** / test **6,080** (80/10/10, seed 42) | `data/README.md` |
| 5 | ISOT raw → processed | **44,898 → 37,567** (1,439 / 478 / 5,380 / 34) | `docs/mse1_make_data.log` |
| 6 | ISOT raw / processed labels | raw 21,417 true / 23,481 fake; processed 20,905 real / 16,662 fake (55.6 % real) | `data/README.md` |
| 7 | ISOT splits | 30,053 / 3,757 / 3,757 | `data/README.md` |
| 8 | LIAR | **12,836** (train 10,269 / val 1,284 / test 1,283); label5: REAL 4,529, PARTIALLY TRUE 2,638, MISLEADING 2,108, FAKE 3,561; majority class 20.6 % | `docs/mse1_make_data.log` |
| 9 | FEVER subset | **23,000** = 20,000 train (6,667/6,667/6,666) + 3,000 dev (1,000 each); from 145,449 train / 19,998 dev | `docs/mse1_make_data.log` |
| 10 | FakeNewsNet PolitiFact | **1,056** (432 fake / 624 real) | `docs/mse1_make_data.log` |
| 11 | **WELFake val — NB** | acc 0.8633 · macro-F1 **0.8616** | `docs/results/classifier_table.md` |
| 12 | **WELFake val — LR** | acc 0.9327 · macro-F1 **0.9316** · CM [[3223,207],[202,2448]] | same |
| 13 | **WELFake val — SVM** | acc 0.9492 · macro-F1 **0.9484** | same |
| 14 | Pre-artefact-fix (same day, "with residual artefacts") | NB 0.8620 · **LR 0.9462** · SVM 0.9610 | `docs/model_identification.md` §4 |
| 15 | ISOT val | NB **0.9536** · LR **0.9822** · SVM **0.9895** (macro-F1) | `docs/results/classifier_table.md` |
| 16 | Artefact-only classifier, raw ISOT | **0.9973 ± 0.0006** (LR on 26 pattern counts, 5-fold CV; dummy 0.523) | `docs/mse1_make_data.log`, `docs/eda_summary.md` |
| 17 | Artefact-only, raw WELFake / processed | 0.840 (dummy 0.514) / processed 0.557 ISOT, 0.564 WELFake = majority rate | `docs/eda_summary.md` |
| 18 | **ISOT ⊂ WELFake** | **37,542 / 37,567 = 99.93 %** on normalised-text hash join (byte-identical variant: 37,408 = 99.6 %) | `docs/model_identification.md` §4.2 / `docs/eda_summary.md` |
| 19 | ISOT val/test rows sitting in WELFake **train** | 2,997 and 2,998 of 3,757 = **79.8 %** | `docs/model_identification.md` §4.2 |
| 20 | **WELFake∖ISOT** (second domain) | **23,257 rows = 38.3 % of WELFake** | same |
| 21 | Cross-dataset preview | WELFake→ISOT **0.9778** (*in-domain, not reportable*); **ISOT→WELFake 0.8299** (the honest one) | `docs/model_identification.md` §4 |
| 22 | DistilBERT max_len coverage (WELFake) | **8.4 % / 19.1 % / 50.9 %** of inputs fit in 128 / 256 / 512 tokens (ISOT 9.8 / 21.4 / 59.0); median ≈ 506 sub-words | `docs/eda_summary.md` |
| 23 | Learning curve (LR, WELFake val macro-F1) | 0.854 (1k) → 0.888 (5k) → 0.905 (10k) → 0.917 (20k) → **0.932** (48.6k) | `docs/model_identification.md` §4 |
| 24 | `make data` runtime | **6.8 min** (410.7 s), peak RSS ≈ 1.8 GB, byte-identical across runs | `docs/mse1_make_data.log` |
| 25 | `make eda` / `make train-baselines` / notebook 02 | ≈ 10.5 min / 1 min 46 s / 133 s | `agent-docs`, MSE1 doc §3.4–3.5 |
| 26 | `make test` | **102 passed** (data 27, artefacts 44, eda 22, models 9), ≈ 36–40 s | MSE1 status line |
| 27 | Figures / references | **18** EDA figures at 200 dpi; **33** BibTeX entries (32 rows in the literature table) | `ls docs/figures/eda_*.png`, `paper/refs.bib` |
| 28 | Artefact patterns | **26 text + 2 title** regexes | `src/preprocess/artefacts.py` |
| 29 | Style (WELFake fake vs real) | exclamations 0.48 vs 0.05 /1k chars; ALL-CAPS titles 4.2 % vs 0.0 %; median length 426 (real) vs 376 (fake) words | `docs/eda_summary.md` |
| 30 | Cross-split overlap | **0 exact, 0 near** (WELFake, ISOT); LIAR official splits: 5 train∩val, 4 train∩test (kept) | `docs/mse1_make_data.log`, `docs/eda_summary.md` |
| 31 | Hardware | RTX 2050 **4 GB** (3.68 GiB usable), 8 threads, 13 GB RAM, torch 2.13+cu130 | `docs/model_identification.md` §3 |

---

## 3. The five verdicts

| Verdict | One line | Example |
|---|---|---|
| **REAL** | Main claims supported by reliable evidence, no temporal/context problem. | "Chandrayaan-3 landed near the lunar south pole on 23 Aug 2023" — ISRO + wires confirm. |
| **FAKE** | Reliable evidence (especially a fact-checker) refutes the main claim. | "WHO said 5G towers spread COVID-19" — refuted by WHO and multiple fact-checkers. |
| **PARTIALLY TRUE** | True core, materially wrong detail; or some claims supported and some refuted. | "India overtook China in population in 2023, reaching 2 billion" — first half true, figure wrong (~1.43 bn). |
| **MISLEADING** | Individually true content in the wrong context: old event as current, headline contradicting the body. | A 2018 Kerala-flood video posted in 2026 as "BREAKING: Kerala under water today". |
| **UNVERIFIABLE** | No reliable evidence, or evidence neutral/contradictory — no trustworthy conclusion. | "A shop in Ghaziabad sold 500 phones in one hour" — no indexed coverage. |

UNVERIFIABLE is **first-class, not a failure**. Target: precision ≥ 0.7 (O6).

---

## 4. Pipeline in 8 steps (draw this in < 2 min)

1. **Ingest** — URL → text + publication date (trafilatura / newspaper3k, htmldate).
2. **Preprocess** — clean, de-boilerplate, sentence-split; non-English → UNVERIFIABLE.
3. **Claims** — score sentences by entities + numbers + reporting verbs → top-k check-worthy claims.
4. **Evidence** — online (DuckDuckGo `ddgs`, Wikipedia API, Google Fact Check Tools) + offline FAISS index (LIAR + PolitiFact titles, MiniLM embeddings) → passages with source, URL, date, reliability weight.
5. **Stance** — NLI cross-encoder `nli-deberta-v3-small`: premise = evidence, hypothesis = claim → SUPPORTS / REFUTES / NEUTRAL.
6. **Temporal / context** — old-news flag (recency cue AND post-date − evidence date ≥ 6 months), out-of-context entity conflict, headline–body mismatch.
7. **Fusion** — rules R0–R9, first match wins; evidence decides, classifier adjusts confidence ±0.10 and breaks the single-source tie (R8).
8. **Explanation** — rule fired, evidence cards, LIME token highlights marked "style only", explicit uncertainty → FastAPI `/analyze` → web UI.

*Side branch:* **content classifier** (DistilBERT, p_fake) feeds only into step 7.

```
INPUT ─ INGEST ─ PREPROCESS ─┬─ CLAIMS ─ EVIDENCE ─ STANCE ─ TEMPORAL ─┐
                             │                                        ▼
                             └─ CLASSIFIER (DistilBERT, p_fake) ───► FUSION (R0–R9)
                                                                        │
                                          5 verdicts + confidence ◄─ EXPLANATION
```

---

## 5. The three findings that make us look good

**A. ISOT ⊂ WELFake.** Our own SHA-1 hash join on normalised text shows **37,542 of 37,567 processed ISOT articles (99.93 %) appear verbatim inside WELFake**, and **79.8 % of ISOT's val/test rows are in WELFake's *train* split**. WELFake's authors list Reuters/ISOT as one of their four source corpora. *So what:* the classic "train on WELFake, test on ISOT → 0.978" result that many papers would call cross-dataset is **training on the test set**. The only honest direction is ISOT → WELFake = **0.83**. We redesigned the MSE2 protocol as **WELFake∖ISOT (23,257 rows) ↔ ISOT**. This is the kind of check a reviewer expects and almost nobody does.

**B. Boiler-plate leakage.** A logistic regression that sees **only counts of 26 publisher/format patterns** — Reuters datelines, "Featured image via", `[VIDEO]` title tags, photo credits, "Follow X on Twitter" bylines, tweet debris — reaches **0.9973** on raw ISOT and 0.840 on raw WELFake; on our processed text both collapse to the majority rate (0.557 / 0.564). We stripped all of it (26 text + 2 title patterns, 44 unit tests) and **our LR macro-F1 fell 0.9462 → 0.9316 and SVM 0.9610 → 0.9484**. *So what:* **the drop is the result.** Those 1.5 points were accuracy that came from the newsroom's template, not from the content. We can now quantify exactly how much of a published "99 %" is leakage, and the before/after pair is the first row of the MSE2 ablation.

**C. max_len coverage.** With the DistilBERT tokenizer only **8.4 % / 19.1 % / 50.9 %** of WELFake inputs fit entirely in 128 / 256 / 512 tokens (median ≈ 506 sub-words). *So what:* max_len 256 is a **GPU-budget** decision (fp16, batch 16 on 4 GB), not a coverage decision — head truncation at 256 keeps about half a typical article. 128 would keep a quarter, so we dropped it and the MSE2 sweep compares **256 vs 512** (batch 8 × grad-accum 2). A measured plot replaced a guessed hyperparameter.

---

## 6. Likely viva questions (in likelihood order)

1. **Why five classes, not two?** Most misinformation is partial truth or true-content-wrong-context. Binary cannot express "old news as new" (MISLEADING) or "right event, wrong number" (PARTIALLY TRUE), and forcing a label without evidence is itself misinformation — hence UNVERIFIABLE.
2. **How is this different from a text classifier?** A classifier learns style and artefacts and cannot say *why*. Here evidence decides; the classifier only shifts confidence ±0.10 and breaks single-source ties (R8).
3. **Why not just use ChatGPT / an LLM?** Cost, hardware (must fit a 4 GB GPU / CPU laptop), non-reproducible outputs, and decisively: an LLM will *invent* evidence, sources and dates. That violates our hard rule NFR-3. A local small model may only paraphrase an already-templated explanation — stretch goal, never on the critical path.
4. **Your accuracy is lower than published papers — why?** Because we removed what made theirs high. FakeBERT reports 98.9 %, WELFake 96.7 %, ISOT 92 % — all in-domain, none leakage-audited. We de-duplicated (8,217 exact + 480 near in WELFake), stripped publisher boiler-plate (cost us 1.5 points) and discovered ISOT sits inside WELFake. Our 0.9316 on clean de-duplicated data is a smaller but honest number.
5. **What if there is no evidence?** UNVERIFIABLE — by design, rule R1 (and R8/R9). Precision target ≥ 0.7. We never guess.
6. **How do you know your model isn't just learning style?** We measured it. Artefact-only classifier 0.9973 on raw ISOT; top-25 LR coefficients after the fix are `said`, `said on`, weekdays, `mr`, `percent` for real and `breaking`, `just`, `watch`, `you` for fake — that is register, not facts. That is exactly why style is not allowed to decide the verdict.
7. **What is your novelty?** (i) A five-class evidence-grounded scheme with an explicit UNVERIFIABLE output; (ii) a laptop-scale hybrid pipeline (claims → free retrieval with offline fallback → NLI stance → temporal check → rule fusion) with explanations; (iii) an honest evaluation: leakage audit, corpus-overlap discovery, and a 2026 Live Claims Set instead of only 2016–18 benchmarks.
8. **Did you write the code yourself?** We used AI assistance for drafting, the way one uses an IDE or Stack Overflow — every design decision is ours, every number was produced by running our own code on this laptop, and we can explain and defend any file, regex or metric on this page. Ask us to open any of them.
9. **Why remove "Reuters"?** The dateline appears in ~98 % of ISOT real articles and 0 % of fakes — a pure label shortcut. Removing it makes the number honest; we report the artefact-only accuracy as the proof.
10. **Why four datasets?** WELFake trains (largest, 4 merged sources, near-balanced); ISOT gives the leakage study; LIAR gives fine-grained truth shades and seeds the offline index; FEVER gives labelled (claim, evidence) pairs for stance.
11. **Why WELFake as primary?** 72,134 articles merged from four corpora → less single-source style bias than ISOT; near-balanced; CC BY 4.0 on Zenodo, so `make download` needs no Kaggle account.
12. **Why stratified 80/10/10 with seed 42?** Val selects models, test is touched once (at MSE2); stratification keeps class ratios; saved ID lists make every model comparable. Two `make data` runs produce byte-identical CSVs.
13. **Why near-duplicate removal, not just exact?** Syndicated copies differ by a few tokens; a near-dup spanning train and test inflates accuracy. MinHash, 128 perms, Jaccard ≥ 0.9 → 0 cross-split overlap.
14. **Why DistilBERT, not BERT-base?** 66 M vs 110 M params, ~60 % faster, retains ~97 % of BERT's GLUE score; fits fp16 × batch 16 × 256 tokens on 4 GB. The classifier only adjusts confidence, so extra size buys nothing.
15. **Why max_len 256?** GPU budget, not coverage — only 19.1 % of inputs fit in 256, 50.9 % in 512. Title + lede carry most of the style signal. MSE2 sweeps 256 vs 512.
16. **Why NLI for stance?** Stance *is* NLI: entail / contradict / neutral ≡ supports / refutes / neutral, so a pre-trained cross-encoder needs zero training and FEVER gives us a labelled dev set.
17. **Why a cross-encoder, not embeddings?** Cosine similarity cannot separate "X happened" from "X did not happen". A cross-encoder reads claim and evidence jointly and models negation. MiniLM bi-encoder is used only for retrieval, where speed matters.
18. **Which tokens leaked, and what is left?** Gone: `video`, `image`, `getty`, `photo`, `featured`, `on twitter`, `follow`, `pic`, `https`, `york times`. Left by policy: `reuters`, `breitbart`, `via` — used inside ordinary sentences ("told Reuters", "Breitbart News reported"). They are content words; deleting them would mean editing sentences.
19. **What did the EDA change in your design?** max_len sweep 256 vs 512, macro-F1 as headline metric, class weights for LIAR, the 26-pattern artefact list, entity-based claim heuristics, the temporal check + Live Claims Set.
20. **Are fake articles longer or shorter?** Measured, not assumed: median 426 (real) vs 376 (fake) words; the *real* class has the wider spread (short wire briefs, long features). What fakes do have: ~10× more exclamation marks and 4.2 % ALL-CAPS titles vs 0.0 %.
21. **How did you measure duplication?** SHA-1 of normalised text (exact) + MinHash Jaccard ≥ 0.9 (near). Counts in `eda_duplicates.png` and the log.
22. **Why keep cased text?** DistilBERT-uncased lower-cases internally; spaCy NER is case-sensitive; lower-casing early would only lose information. TF-IDF lower-cases inside the vectoriser.
23. **Why no stemming / stop-word removal?** Sub-word tokenisers handle morphology, sublinear TF already down-weights frequent words, and stop words like "not" carry the claim.
24. **Why class weights, not oversampling, for LIAR?** Imbalance is mild (max/min 2.5); weights avoid duplicating short statements across folds.
25. **Why map LIAR 6 → 5 that way?** PolitiFact's own definitions: true/mostly-true → REAL; half-true ("leaves out important details") → PARTIALLY TRUE; barely-true ("ignores critical facts") → MISLEADING; false/pants-fire → FAKE. UNVERIFIABLE has no LIAR counterpart — it only comes from retrieval failure.
26. **Why only 20k of FEVER?** It evaluates (and optionally fine-tunes) a pre-trained NLI model, not one trained from scratch; 20k pairs run one epoch in ~30–40 min on the 4 GB GPU.
27. **Are your datasets recent?** No — ISOT is 2015–2018, LIAR ≤ 2017. That is why the end-to-end evaluation uses our own ~100-item 2026 Live Claims Set.
28. **What is the fusion rule that decides FAKE?** R2 (a Tier-1 fact-checker rates the matched claim False/Pants-on-Fire) or R3 (main stance REFUTES with ≥ 2 independent evidence items).
29. **How is confidence computed?** 0.5×agreement + 0.3×coverage + 0.2×source quality, ±0.10 from the classifier, clipped to [0.05, 0.98]; shown as Low / Medium / High bands.
30. **What are your measurable objectives?** O1 macro-F1 ≥ 0.95 on WELFake test; O4 stance ≥ 85 % on FEVER dev; O6 5-class macro-F1 ≥ 0.55 with UNVERIFIABLE precision ≥ 0.7; O7 public demo with ≤ 15 s median latency.
31. **What is reproducible right now?** `make setup → download → data → test → train-baselines → nb-run-baselines → eda → paper`, verified end-to-end in a fresh clone; seeds fixed, outputs byte-identical.
32. **Why baselines before a transformer?** Minutes to train, they set a floor, their coefficients exposed the leakage (and triggered the same-day fix), and LinearSVC is our CPU fallback for `MODE=lite`. A transformer that cannot beat TF-IDF + SVM is not worth its cost.
33. **What does LIAR's ~27 % 6-way accuracy tell you?** That fine-grained truthfulness cannot come from text style — it needs evidence. Hence LIAR is used only for a 3-way shading head.
34. **Which paper is closest to yours?** Guo et al. 2022's framework (claim detection → evidence retrieval → verdict → justification). Our additions: news domain, temporal check, 5-class scheme, laptop-scale deployment.
35. **Why LIME rather than attention weights?** LIME is model-agnostic and token-level; attention is not a faithful explanation. We label LIME output "style only" so users do not mistake it for evidence.
36. **What is explicitly out of scope?** Deepfake/image forensics, multilingual (English only), platform crawlers, paid APIs/LLMs, guarantees of truth.

---

## 7. Per-member cheat cards

**Navishka Sharma — Data & EDA lead** (`src/preprocess/`, `src/temporal/`, datasets, splits, EDA figures).
Likely questions: (1) Walk me through `make data` step by step. (2) How did you prove there is no train/test leakage? (0 exact, 0 near; MinHash Jaccard ≥ 0.9.) (3) Which EDA figure changed a modelling decision, and how?
Papers: **Wang 2017, LIAR** — 12,836 PolitiFact statements, 6 labels; text-only CNN 27.0 %, hybrid 27.4 %, majority 20.8 % → fine-grained truth needs evidence. **Verma et al. 2021, WELFake** — 72,134 articles merged from four corpora, linguistic features + embeddings, 96.73 % accuracy; binary and in-domain only.

**Naitik Kukreja — Modelling lead** (`src/models/`, training, sweeps, LIAR head).
Likely questions: (1) Why does LR score 0.9316 and SVM 0.9484 — and why did both go *down*? (2) What exactly will you sweep at MSE2? (3) Will DistilBERT fit 4 GB, and what if it OOMs? (batch 8 × grad-accum 2, then gradient checkpointing.)
Papers: **Kaliyar et al. 2021, FakeBERT** — BERT + parallel 1-D CNN, 98.90 % on a 20,800-article Kaggle corpus; binary, single-domain, leakage not audited. **Sanh et al. 2019, DistilBERT** — distillation of BERT: 40 % smaller, 60 % faster, ~97 % of its language-understanding performance → our content classifier.

**Naveen — Claims / Evidence / Stance lead** (`src/claims/`, `src/evidence/`, `src/stance/`).
Likely questions: (1) How do you pick check-worthy claims? ((PERSON|ORG|GPE) ∧ (DATE|CARDINAL|PERCENT|MONEY) — 99 % of real vs 89 % of fake docs have both.) (2) Why NLI and why a cross-encoder? (3) What happens when the internet is down? (Offline FAISS index over LIAR + PolitiFact titles, answers in < 2 s.)
Papers: **Thorne et al. 2018, FEVER** — 185,445 Wikipedia-grounded claims (SUPPORTS/REFUTES/NEI); baseline 31.87 FEVER score, 50.91 % label accuracy → defines the retrieve-then-verify shape. **Liu et al. 2020, KGAT** — kernel graph attention over evidence: RoBERTa-large 74.07 % label accuracy / 70.38 FEVER score (BERT-base 72.81 / 69.40) → the ceiling we do not need to reach.

**Prateek Srivastava — Backend / API / Deployment lead** (`src/ingest/`, `src/fusion/`, `src/api/`, `docker/`, `tests/`, Makefile).
Likely questions: (1) Draw the architecture and name the Makefile target for each stage. (2) Explain fusion rules R1, R3, R7, R8. (3) How do you guarantee reproducibility? (Pinned requirements, seed 42, byte-identical `make data`, 102 tests, fresh-clone verified.)
Papers: **Ahmed, Traore & Saad 2017, ISOT** — 44,898 articles (21,417 Reuters-true / 23,481 fake), TF-IDF n-grams + classical ML, best 92 % with LinearSVM; all true articles from one source → source leakage. **Reimers & Gurevych 2019, Sentence-BERT** — siamese BERT embeddings turn a 65-hour similarity search into ~5 s; bi-encoders cannot model contradiction, so we use it for retrieval only.

**Nikhil — UI / XAI / Paper lead** (`app/`, `src/explain/`, `paper/`, `docs/`).
Likely questions: (1) State the problem in one sentence and list the five classes with examples. (2) What are your three contributions and which objective does each map to? (O6; O3–O5; O1.) (3) Why LIME and how is it presented to the user?
Papers: **Ribeiro et al. 2016, LIME** — model-agnostic local sparse surrogate; their user study showed non-experts pick the better-generalising classifier with it; explanations can be unstable. **Vosoughi, Roy & Aral 2018 (Science)** — ~126,000 Twitter cascades: falsehood 70 % more likely to be retweeted, truth ~6× slower to reach 1,500 people → our motivation statistic.

---

## 8. Demo runbook (10 minutes)

Pre-open the laptop with all three notebooks executed, figures visible, terminal in `/home/nikhil/Desktop/nlp`. **Do not run anything long live.**

| Min | Who | Command / file to show | Point at |
|---|---|---|---|
| 0–1.5 | Nikhil | Printed master doc §3 + §8; `docs/problem_identification.md` | Problem sentence; five classes with one example each; "evidence decides, style advises" |
| 1.5–2.5 | Prateek | `make help`; the §9 pipeline diagram | 8 pipeline stages; one Makefile target per stage; reproducibility |
| 2.5–4 | Navishka | `cat data/README.md`; `notebooks/00_datasets.ipynb` count cells (already executed, 11 s if rerun) | Why four datasets; raw counts 72,134 / 44,898 / 12,836; licences; LIAR 6→5 mapping |
| 4–5.5 | Navishka | `tail -30 docs/mse1_make_data.log`; `01_eda.ipynb` before/after cell | 6.8 min run; the summary table; artefact-only 0.9973; cross-split overlap 0 |
| 5.5–7 | Navishka + Naveen | `docs/figures/eda_top_ngrams_before.png` vs `_after.png`, `eda_maxlen_coverage.png`, `eda_ner_types.png`, `eda_isot_dates.png`; `docs/eda_summary.md` | One insight → one design decision per figure |
| 7–8.5 | Naitik + Naveen | `notebooks/02_baselines.ipynb` metrics + coefficient cells; `cat docs/results/classifier_table.md` | LR 0.9316 / SVM 0.9484; top ±25 coefficients; ISOT ⊂ WELFake hash join (§5.1); why DistilBERT; why NLI |
| 8.5–10 | Nikhil | `paper/main.pdf` p.1–2; `docs/literature_table.md` | Three contributions; five survey themes; two papers in depth; the gap |

Quick verification commands if the examiner asks: `ls docs/figures/eda_*.png | wc -l` → 18 · `grep -c "^@" paper/refs.bib` → 33 · `make test` → 102 passed (~40 s) · `python -c "import pandas as pd; print(pd.read_parquet('data/processed/isot.parquet').text.str.contains(r'\(Reuters\)').sum())"` → 0.

**Fallback if anything fails:** every notebook is committed **with its outputs** — open the `.ipynb` and scroll instead of executing. `docs/mse1_make_data.log` is the saved `make data` run. `docs/results/classifier_table.md`, `docs/eda_summary.md` and `docs/model_identification.md` carry every number without running code. Never enable `nbstripout` before a viva. `make paper` is a no-op if `paper/main.pdf` is newer than `main.tex` — just open the committed PDF.

---

## 9. Known weaknesses & honest answers

| Weakness | One-sentence reply |
|---|---|
| `reuters`, `breitbart`, `via` still in the top LR coefficients | "Publisher boiler-plate is gone; these are source names inside ordinary sentences — content words — so we keep them and disclose that the classifier still knows which outlets an article talks about." |
| ISOT ⊂ WELFake (99.93 %) | "We found it ourselves with a hash join; it means WELFake→ISOT is in-domain, so MSE2 uses WELFake∖ISOT (23,257 rows) ↔ ISOT, and the raw-vs-processed leakage audit is unaffected." |
| LIAR official splits share 5 train∩val + 4 train∩test statements | "Kept deliberately for comparability with Wang 2017 and every later LIAR paper; < 0.1 % of the data, disclosed in `docs/eda_summary.md`. Our own splits have zero." |
| FEVER parquet stores evidence *pointers*, not sentence text | "The MSE2 stance evaluation fetches the sentences from the FEVER Wikipedia dump; that is why the EDA reports evidence pointers per claim (29 % need more than one)." |
| `make paper` uses tectonic, not latexmk | "The local TeX Live has no latexmk, IEEEtran or Times fonts; `paper/Makefile` uses latexmk when present and the tectonic binary otherwise, and Overleaf compiles it unchanged." |
| No slide deck yet | "Team action for the buffer week; every number on the slides will be lifted from `classifier_table.md`, `eda_summary.md` and `mse1_make_data.log`." |
| DistilBERT / retrieval / stance / fusion / UI not built | "Deliberate scope: MSE1 is problem, data, preprocessing, EDA and model *identification*. Training is the MSE2 criterion and is already budgeted at ≈ 3 h for the 6-config sweep plus 1–1.25 h for the full retrain." |
| `make eda` takes ≈ 10 min, not the 4 min originally planned | "The artefact-only classifier is now recomputed on raw and processed WELFake and ISOT with 26 regex features; budget is 20 min and all figures are committed." |
| Datasets are 2015–2018 | "Known and is exactly why the end-to-end evaluation uses our own 2026 Live Claims Set instead of only these benchmarks." |

---

## 10. Night-before checklist (still open)

- [ ] Export `docs/mse1_slides.pdf`; problem statement must match master §3 word-for-word in meaning.
- [ ] Print master doc v2.1 (or have the PDF on a second laptop) and this sheet, one copy per member.
- [ ] Dry-run viva with the §8 script, timed.
- [ ] Every member rehearses their two papers, 1 minute each (§7 above).
- [ ] Every member draws the 8-step pipeline on paper in under 2 minutes, from memory.
- [ ] Every member can state the hardware budget (4 GB GPU, fp16, batch 16 at max_len 256).
- [ ] Fill in and print the evidence-of-work log (MSE1 doc §7), including the "Reviewed by" column.
- [ ] Repeat the fresh-clone recipe on a second member's laptop.
- [ ] Laptops charged; full offline copy of the repo, notebooks with outputs, figures and PDFs — assume no Wi-Fi.
- [ ] Do **not** run `nbstripout`; confirm notebook outputs are still saved.
