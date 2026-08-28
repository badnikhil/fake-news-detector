# Problem Identification — Fake News & Misinformation Detector

MSE1 artefact for criterion **3.1 Problem Identification (3 marks)**. Everything here is a condensed, viva-ready view of the master document (`docs/FakeNewsDetector_ProjectDetails.md` §3–§9) and of `paper/main.tex` §I; when in doubt the master document wins. Last updated 28 Aug 2026.

## 1. Problem statement (one sentence — identical in meaning in master §3, paper §I and the slides)

> Given a news item — a headline, an article text or a URL — automatically determine, with supporting evidence, dates and a human-readable explanation, whether its central claims are **real, fake, partially true, misleading or unverifiable**, using only free tools and hardware available to students.

## 2. Motivation (every number carries a reference; bibkeys are in `paper/refs.bib`)

| Fact | Figure | Source |
|---|---|---|
| False news travels further and faster | In ~126,000 Twitter story cascades (2006–2017), falsehoods were **70 % more likely to be retweeted** and the truth took **about six times longer to reach 1,500 people** | Vosoughi, Roy & Aral, *Science* 2018 (`vosoughi2018spread`) |
| Misinformation is the top-ranked global risk | *Global Risks Report 2024* ranks **misinformation and disinformation #1** over the two-year horizon, ahead of extreme weather and armed conflict | World Economic Forum 2024 (`wef2024risks`) |
| Misinformation kills | One COVID-19 rumour (methanol/alcohol "cure") linked to **~800 deaths, 5,876 hospitalisations, 60 cases of blindness** in Jan–Apr 2020 | Islam et al., *AJTMH* 2020 (`islam2020infodemic`) |
| Indian context: rumours → violence | WhatsApp child-kidnapping rumours linked to **at least 17 mob killings across India between April and July 2018** (BBC); "more than two dozen" (Al Jazeera) | BBC News 18 Jul 2018 (`bbc2018whatsapp`); Al Jazeera 17 Jul 2018 (`aljazeera2018whatsapp`) |
| Indian context: old footage as "breaking" news | Indian fact-checkers (AltNews, BOOM, Factly, PIB Fact Check) repeatedly debunk old flood/riot/accident footage re-circulated during new events; an analysis of 125 fact-checked Indian COVID-19 items found 67.2 % health-related, text + video the most common format (47.2 %), and 94.4 % originating on social/online media | Al-Zaman, *Journalism and Media* 2021 (`alzaman2021india`); fact-checker sites listed in master §29 |

**The gap.** Manual fact-checking is accurate but slow. Most automatic detectors are *binary text classifiers* that learn writing style and dataset artefacts rather than facts, so they (i) cannot say *why* an item is false, (ii) cannot represent partial truth or true-but-out-of-context items, and (iii) cannot abstain when evidence is missing. Numbers like 96–99 % accuracy (WELFake, FakeBERT) are in-domain and collapse across corpora; ISOT's "true" class is all Reuters, so boilerplate alone separates it.

## 3. Objectives with measurable targets (copied from master §4)

| # | Objective | Measurable target |
|---|---|---|
| O1 | Build a content classifier for fake vs. real news articles. | Macro-F1 ≥ 0.95 on WELFake held-out test; report cross-dataset F1 (WELFake→ISOT) honestly. |
| O2 | Identify check-worthy claims in an article. | Top-3 claims judged relevant by annotators in ≥ 80 % of 50 sampled articles. |
| O3 | Retrieve evidence from free sources with an offline fallback. | ≥ 1 evidence passage for ≥ 85 % of Live Claims Set items online; offline index answers in < 2 s. |
| O4 | Detect stance of evidence toward a claim. | Accuracy ≥ 85 % (3-way) on a FEVER dev subset with gold evidence. |
| O5 | Detect "old news presented as new". | Correct flag on ≥ 80 % of a 30-item hand-built temporal test set. |
| O6 | Produce a 5-class verdict with confidence and explanation. | Macro-F1 ≥ 0.55 on the ~100-item Live Claims Set (5 classes); UNVERIFIABLE precision ≥ 0.7. |
| O7 | Deliver a usable web app, containerised and deployed on a free tier. | Docker image builds; public HF Space URL responds; median latency ≤ 15 s online, ≤ 5 s offline. |
| O8 | Write and communicate an IEEE-format research paper. | Turnitin similarity < 10 %; submission receipt from a conference/arXiv before ESE. |

Paper §I contribution bullets ↔ objectives: bullet 1 (five-class evidence-grounded scheme with UNVERIFIABLE) → **O6**; bullet 2 (claims → retrieval with offline fallback → NLI stance → temporal check) → **O3, O4, O5**; bullet 3 (DistilBERT + LIME, cross-dataset audit) → **O1**.

## 4. The five output classes (master §8) — one-line definition and one example each

| Class | Definition | Example |
|---|---|---|
| **REAL** | Main claim(s) supported by reliable evidence, no temporal/context inconsistency. | "ISRO's Chandrayaan-3 landed near the lunar south pole on 23 Aug 2023." → supported by ISRO, Reuters, Wikipedia. |
| **FAKE** | Reliable evidence (especially fact-checkers) refutes the main claim, or it is shown to be fabricated. | "WHO declared that 5G towers spread COVID-19." → refuted by WHO and multiple fact-checkers. |
| **PARTIALLY TRUE** | Some important claims supported, others refuted; or a true core with materially wrong details (numbers, names, dates). | "India's population overtook China's in 2023, reaching 2 billion." → first part supported; the figure is wrong (~1.43 bn). |
| **MISLEADING** | Individually supportable claims presented in the wrong context: old event as current, real quote/statistic attached to the wrong event/place, or a headline the body contradicts. | A 2018 Kerala-flood video shared in 2026 as "BREAKING: Kerala under water today". |
| **UNVERIFIABLE** | No reliable evidence, or evidence neutral/contradictory with low weight; no trustworthy conclusion possible. | "A local shop in Ghaziabad sold 500 phones in one hour." → no coverage in any indexed source. |

UNVERIFIABLE is a **first-class, expected** output: for hyper-local, very recent or opinion-like inputs it is the correct answer, and forcing FAKE/REAL there would itself be misinformation (O6 targets UNVERIFIABLE precision ≥ 0.7).

## 5. Out of scope (master §5.2)

| Removed feature | Reason |
|---|---|
| Video manipulation / deepfake detection | Separate, compute-heavy research problem. |
| Image forensics (edited/manipulated images) | Separate advanced problem; only text input is supported. Screenshot OCR is a stretch goal only. |
| Large-scale multilingual support | English only. Hindi/Hinglish is a stretch goal via translation, not core. Non-English input → UNVERIFIABLE with "English-only in this version" (FR-13). |
| Platform-specific crawlers (Instagram, TV, Twitter/X, WhatsApp) | Users paste text/URL; no scraping of walled platforms (ToS, keys, brittleness). |
| Full story-history reconstruction | Only the dates that matter for the "old vs. new" decision are checked. |
| Automatic reverse-search of the original source of every image/video | Future enhancement. |
| Absolute guarantees of truth | Evidence-based conclusions with stated uncertainty. |
| Paid APIs / cloud GPUs / large LLMs (GPT-class) | Must run on a student laptop with free services. A local small LLM (Ollama) may *paraphrase* explanations as a stretch goal but is never on the critical path. |
| User accounts, history, moderation dashboards | Not needed for the evaluation criteria. |

## 6. Pipeline sketch — every member must be able to draw this in under 2 minutes

```
INPUT (headline | article text | URL)
  │
  ▼
INGEST  trafilatura / newspaper3k → title, body, publication date (htmldate)
  │
  ▼
PREPROCESS  clean, de-boilerplate, spaCy sentence split; language check (non-English → UNVERIFIABLE)
  │
  ├──────────────────────────────────────────────┐
  ▼                                              ▼
CLAIMS  heuristic score (entities, numbers,     CONTENT CLASSIFIER  fine-tuned DistilBERT
        reporting verbs, position) + optional    → p_fake  (+ LIME token highlights)
        ClaimBuster LR → top-k claims             (only adjusts confidence / breaks R8 tie)
  │                                              │
  ▼                                              │
EVIDENCE  online: ddgs search, Wikipedia API,    │
          Google Fact Check Tools API            │
          offline: FAISS index (LIAR + PolitiFact│
          titles, MiniLM embeddings)             │
          → passages with source, URL, date,     │
            reliability weight w_s               │
  │                                              │
  ▼                                              │
STANCE  NLI cross-encoder (nli-deberta-v3-small):│
        premise = evidence, hypothesis = claim   │
        → SUPPORTS / REFUTES / NEUTRAL + p       │
  │                                              │
  ▼                                              │
TEMPORAL / CONTEXT  old-news flag (recency cue AND post date − evidence date ≥ 6 months),
                    out-of-context flag (GPE/DATE entity conflict), headline–body mismatch
  │                                              │
  ▼                                              ▼
FUSION  rules R0–R9 (evidence decides; classifier ±0.10 confidence, R8 tie-break)
        → REAL | FAKE | PARTIALLY TRUE | MISLEADING | UNVERIFIABLE + confidence 0–1
  │
  ▼
EXPLANATION  rule fired, key claims, evidence cards (source, date, stance, snippet, URL),
             LIME highlights ("style only"), explicit uncertainty statement
  │
  ▼
FastAPI  POST /analyze → JSON → HTML/JS UI   (Docker → Hugging Face Spaces; MODE=offline for the demo)
```

Whiteboard version: **input → ingest → preprocess → claims → evidence → stance → temporal → fusion → explanation**, with the classifier as a side branch into fusion.

## 7. Verified facts (resolutions of the `[verify]` markers in the master doc, 28 Aug 2026)

| # | Item | Outcome | Evidence | Applied to |
|---|---|---|---|---|
| V1 | WELFake licence (Zenodo record 4561253) | **CC BY 4.0** confirmed (`license.id = cc-by-4.0`; record "WELFake dataset for fake news detection in text data", Verma, Agrawal, Prodan, published 25 Feb 2021, DOI 10.5281/zenodo.4561253; 72,134 articles = 35,028 real + 37,106 fake). The Kaggle mirror `saurabhshahane/fake-news-classification` shows no separately readable licence (JS-rendered page) — treat the Zenodo licence as authoritative and cite Zenodo. | https://zenodo.org/api/records/4561253 ; https://zenodo.org/records/4561253 | master §10 row D1 |
| V2 | ClaimBuster licence (Zenodo record 3609356) | **CC BY 4.0** confirmed (`license.id = cc-by-4.0`; "ClaimBuster: A Benchmark Dataset of Check-worthy Factual Claims", Arslan, Hassan, Li, Tremayne, published 15 Jan 2020, DOI 10.5281/zenodo.3609356; 23,533 sentences confirmed). | https://zenodo.org/api/records/3609356 | master §10 row D6 |
| V3 | 2018 India WhatsApp-rumour lynching figure | **Verified and kept.** BBC News, "How WhatsApp helped turn an Indian village into a lynch mob" (Deepthi Bathini, 18 Jul 2018): "At least 17 others have allegedly been killed in India over child kidnapping rumours since April 2018." Al Jazeera, "In India, WhatsApp stirs up deadly rumours" (17 Jul 2018): "more than two dozen people have been beaten to death by mob vigilantes across India over suspicions of child abduction". IndiaSpend tracker (9 Jul 2018, now mirrored on isignal.in): 33 killed, ≥ 99 injured in 69 child-lifting-rumour attacks, 1 Jan 2017 – 5 Jul 2018 (28 % traced to WhatsApp). No stand-alone Reuters article could be fetched. **Paper wording:** "linked to at least 17 mob killings across the country between April and July 2018" citing BBC + Al Jazeera (the conservative figure). Note: bbc.com blocked direct automated fetch; text was read through a text-extraction proxy and the URL confirmed live via the Wayback Machine — open the URL once in a browser before the viva. | https://www.bbc.com/news/world-asia-india-44856910 ; https://www.aljazeera.com/features/2018/7/17/in-india-whatsapp-stirs-up-deadly-rumours ; https://www.isignal.in/child-lifting-rumours-33-killed-in-69-mob-attacks-since-jan-2017-before-that-only-1-attack-in-2012-2012 | master §3 item 4; paper §I; `refs.bib` `bbc2018whatsapp`, `aljazeera2018whatsapp` |
| V4 | IEEE INDICON 2026 deadline / realistic Nov–Dec 2026 venue | **INDICON 2026** (23rd; IEEE India Council + IEEE Madras Section; Sri Sairam Engineering College, Chennai; 18–20 Dec 2026): paper submission deadline **31 Aug 2026** — three days from today, i.e. *not* usable. **Chosen target: 17th IEEE International Conference on Cloud Computing, Data Science & Engineering (Confluence 2027)**, Amity University, Noida (≈ 30 km from KIET): conference 21 Jan 2027, **submission deadline 4 Dec 2026**, notification 25 Dec 2026; past editions are in IEEE Xplore (12th, 14th confirmed). Confluence 2027 dates come from a conference-aggregator page because Amity's own pages returned errors — confirm on the official CFP with the guide by week 10. Also checked: IEEE CSNT 2027 (Gwalior; deadline 25 Sep 2026 — too early unless the paper is ready by week 6), IEEE ICPCSN 2027 (deadline 1 Feb 2027 — after ESE), DELCON 2026 and INDISCON 2026 (deadlines passed 15 Jun 2026), ICCCNT 2027 (not announced), IEEE ESCI 2027 (Pune, Mar 2027; site still shows 2026 dates — by extrapolation a Nov 2026 deadline is plausible, unconfirmed). arXiv (cs.CL) preprint remains the guaranteed "communicated" fallback. | https://ieeeindiacouncil.org/ieee-india-council-international-conference-indicon-2026/ ; https://www.indicon2026.com/ ; https://www.myhuiban.com/conference/5783 ; https://www.csnt.in/Important_Dates.html ; https://icpcsn.com/2027/ | master §20 venue table; §22 week-15/16 plan |
| V5 | WEF Global Risks Report 2024 ranking | Confirmed via WEF press release "Disinformation Tops Global Risks 2024" (Jan 2024): misinformation and disinformation is the most severe risk over the two-year horizon. The PDF itself blocked automated download — lift a verbatim sentence from pp. 8–9 if a quotation is needed. | https://www.weforum.org/press/2024/01/global-risks-report-2024-press-release/ | paper §I (`wef2024risks`) |
| V6 | Devlin et al. 2019 (BERT) missing from master §29 | Added to §29 together with the other new paper references (Williams 2018 MNLI, Nie 2019 NSMN, Shu 2019 dEFEND, Hanselowski 2018, Popat 2018 DeClarE, Pérez-Rosas 2018, Zubiaga 2018, Al-Zaman 2021, Wang 2020 MiniLM, Ahmed 2018, BBC/Al Jazeera 2018). | `paper/refs.bib` (33 entries, all metadata checked against ACL Anthology / publisher pages) | master §29 |

Still open in the master doc (not in this brief): the satire/hoax domain list (§14.3, `[verify list before use]`) and "any IEEE conference hosted by KIET" (§20, `[verify with guide]`).

## 8. MSE1 Definition-of-Done check (criterion 3.1)

- [x] Problem statement is one sentence and identical in meaning in master §3, paper §I and this document (slides: copy §1 above).
- [x] Every motivating statistic has a citation; the Indian lynching figure is verified (V3) and kept.
- [x] Objectives O1–O8 each have a numeric target.
- [x] Five classes with a one-line definition and one example each; UNVERIFIABLE explained as first-class.
- [x] Out-of-scope table present.
- [ ] Every team member can draw the pipeline in < 2 minutes — practise with §6 (to be ticked by the team, not by a document).
