# Literature survey table — Fake News & Misinformation Detector

Companion to `paper/main.tex` §II (Related Work) and `paper/refs.bib`. Bibkeys in the **Ref** column are the BibTeX keys; keep them stable. "Reported result" is the figure the paper itself reports (abstract or main results table), checked against the ACL Anthology / publisher / arXiv page on 28 Aug 2026; confidence and the page consulted are in the last section. Nothing here is written from memory alone.

Themes (paper §II): **A** datasets & benchmarks · **B** content-based classifiers · **C** claim detection, evidence retrieval, fact verification & stance · **D** explainability · **E** spread, temporal & contextual misinformation (incl. surveys and motivation sources).

| # | Ref (bibkey) | Year | Dataset | Method | Reported result | Limitation | What we take from it |
|---|---|---|---|---|---|---|---|
| 1 | `wang2017liar` — Wang, "Liar, Liar Pants on Fire" (ACL) | 2017 | LIAR: 12,836 PolitiFact statements, 6 labels | CNN over text, plus speaker metadata (hybrid) | 6-way test acc.: majority 20.8 %, text-only CNN 27.0 %, hybrid CNN + metadata 27.4 % (Table 2) | Short statements, no evidence; 6-way barely above chance | LIAR data; 6→5 and 3-way label mapping; statements for the offline index; honest fine-grained baseline (theme A) |
| 2 | `shu2020fakenewsnet` — Shu et al., FakeNewsNet (Big Data) | 2020 | PolitiFact + GossipCop news with social context | Repository + content/social baselines | Content-only baselines (Table 3): CNN acc. 0.629 PolitiFact / 0.723 GossipCop; SAF/S 0.654 / 0.689 | Content must be re-crawled; social features unavailable to us | PolitiFact titles for the offline FAISS index (A) |
| 3 | `ahmed2017isot` — Ahmed, Traore & Saad (ISDDC, LNCS 10618) | 2017 | ISOT: 44,898 articles (21,417 Reuters true / 23,481 fake) | TF-IDF n-grams + NB / LR / SVM / LSVM etc. | Best 92 % acc. with Linear SVM on unigram TF-IDF | All genuine articles from one source (Reuters) → style/source leakage | ISOT for cross-dataset transfer and the leakage audit; classical baselines (A, B) |
| 4 | `ahmed2018opinion` — Ahmed, Traore & Saad (Security and Privacy) | 2018 | ISOT + opinion-spam corpora | Same n-gram + classical ML pipeline, journal extension | Same ISOT figure (92 % LSVM); extends to opinion spam | As above | Confirms that n-gram models saturate on ISOT — motivates the artefact-removal ablation (B) |
| 5 | `verma2021welfake` — Verma et al., WELFake (IEEE TCSS) | 2021 | WELFake: 72,134 articles merged from four corpora | Linguistic features + word embeddings, classical classifiers | 96.73 % acc., 96.56 % F1 | Binary; in-domain only; no cross-corpus test | Primary training set; O1 target; the 4-corpus merge reduces (but does not remove) source leakage (A, B) |
| 6 | `thorne2018fever` — Thorne et al., FEVER (NAACL) | 2018 | FEVER: 185,445 Wikipedia-grounded claims, SUPPORTS / REFUTES / NEI | Document retrieval → sentence selection → NLI (decomposable attention) | 31.87 % FEVER score (label + correct evidence); 50.91 % label acc. ignoring evidence | Wikipedia-only; claims written by annotators, not real news | Stance label set and mapping; evaluation subset for O4; retrieve-then-verify pipeline shape (A, C) |
| 7 | `augenstein2019multifc` — Augenstein et al., MultiFC (EMNLP) | 2019 | 34,918 claims from 26 fact-checking sites with retrieved evidence | Multi-task learning with label embeddings over claim + evidence + metadata | Best model (MTL + LEL): micro-F1 0.625, macro-F1 0.492 | Heterogeneous per-site label sets; hard to compare | Real fact-checker ratings as ground truth; mapping fact-check ratings to our five classes (A) |
| 8 | `arslan2020claimbuster` — Arslan et al., ClaimBuster benchmark (ICWSM) | 2020 | 23,533 US-debate sentences: CFS / UFS / NFS | Dataset-release paper (crowd + expert annotation) | Dataset paper — no classifier table; 23,533 sentences (22,281 met the annotation stopping rule) | Political debates only; US English | D6 dataset for the optional check-worthiness classifier (A, C) |
| 9 | `hassan2017claimbuster` — Hassan et al., ClaimBuster (KDD) | 2017 | Debate sentences (earlier ClaimBuster corpus) | Sentence features (POS, entities, sentiment, length) + SVM | Check-worthy class: precision 79 %, recall 74 % | Domain-specific; needs hand-labelled data | Heuristic cue set (entities, numbers, reporting verbs) for `src/claims` (C) |
| 10 | `devlin2019bert` — Devlin et al., BERT (NAACL) | 2019 | GLUE, MNLI, SQuAD | Bidirectional transformer pre-training + fine-tuning | GLUE 80.5 (+7.7), MNLI 86.7 % (+4.6), SQuAD 1.1 F1 93.2; new SOTA on 11 tasks | 110M params; too heavy for a 4 GB GPU at max_len 256 | Fine-tuning recipe (lr, warm-up, epochs) for the content classifier (B) |
| 11 | `sanh2019distilbert` — Sanh et al., DistilBERT (arXiv / NeurIPS-W) | 2019 | GLUE | Knowledge distillation of BERT (6 layers) | 40 % smaller, 60 % faster, retains 97 % of BERT's language-understanding performance | Small accuracy loss vs. BERT | Our content classifier (fp16, batch 16 fits an RTX 3050) (B) |
| 12 | `kaliyar2021fakebert` — Kaliyar et al., FakeBERT (MTAP) | 2021 | Kaggle "fake news" corpus (20,800 articles, 2016 US election) | BERT embeddings + parallel 1-D CNN blocks | 98.90 % acc. | Binary; single in-domain corpus; leakage not audited | Transformer upper bound and comparison row for §VII (B) |
| 13 | `perezrosas2018automatic` — Pérez-Rosas et al. (COLING) | 2018 | FakeNewsAMT (crowd-written) + Celebrity | Linear SVM on lexical, syntactic, readability features | Best acc. 0.78 on FakeNewsAMT (readability features), 0.73 on Celebrity (all features) — arXiv version tables; camera-ready may differ slightly | Small corpora; crowd-written fakes | Style features as a cheap secondary signal; caution about small-corpus numbers (B) |
| 14 | `nie2019nsmn` — Nie, Chen & Bansal, NSMN (AAAI) | 2019 | FEVER | Neural semantic matching networks shared across retrieval, selection, verification | FEVER shared-task winner: FEVER score 64.21 %, label acc. 68.21 % (test) | FEVER-specific; heavy | Reference point for what a unified retrieve-verify model achieves (C) |
| 15 | `liu2020kgat` — Liu et al., KGAT (ACL) | 2020 | FEVER | Kernel graph attention over evidence sentences (BERT / RoBERTa) | RoBERTa-large: 74.07 % label acc., 70.38 FEVER score (test, Table 2); BERT-base 72.81 / 69.40 | Large model; Wikipedia-only | Ceiling for evidence-based verification; we stay at cross-encoder scale (C) |
| 16 | `popat2018declare` — Popat et al., DeClarE (EMNLP) | 2018 | Snopes, PolitiFact, NewsTrust, SemEval-2017 claims + web search results | Bi-LSTM over retrieved articles with claim-guided attention + source embedding | AUC 0.86 Snopes, 0.75 PolitiFact; macro-acc. 0.57 SemEval; MSE 0.29 NewsTrust | Needs a search engine; binary credibility | Web-search evidence path with source-credibility weights; attention words as explanation (C, D) |
| 17 | `reimers2019sbert` — Reimers & Gurevych, Sentence-BERT (EMNLP) | 2019 | STS benchmark, NLI corpora | Siamese/bi-encoder BERT with pooling | Most-similar-pair search over 10k sentences: ~65 h with BERT → ~5 s with SBERT | Bi-encoders cannot model contradiction between two sentences | MiniLM bi-encoder for the offline index and passage re-ranking; cross-encoder kept for stance (C) |
| 18 | `wang2020minilm` — Wang et al., MiniLM (NeurIPS) | 2020 | GLUE, SQuAD | Deep self-attention distillation | `all-MiniLM-L6-v2`: ~22M params, 384-d embeddings (model card) | Lower quality than large bi-encoders | Embedding model for FAISS index (~14k statements embed in ~2 min on CPU) (C) |
| 19 | `williams2018mnli` — Williams, Nangia & Bowman, MultiNLI (NAACL) | 2018 | MultiNLI: 433k premise–hypothesis pairs, 10 genres | Crowd-sourced NLI corpus; ESIM baseline | ESIM 72.3 % matched / 72.1 % mismatched acc. | Sentence pairs, not claim–document; genre shift hurts | Training basis of the NLI cross-encoder we use for stance (C) |
| 20 | `he2023debertav3` — He, Gao & Chen, DeBERTaV3 (ICLR) | 2023 | GLUE, MNLI | ELECTRA-style pre-training + gradient-disentangled embedding sharing | DeBERTaV3-base MNLI-m/mm 90.6 / 90.7; `cross-encoder/nli-deberta-v3-small` card: SNLI-test 91.65 %, MNLI-mismatched 87.55 % | Base/large too big for 4 GB; small variant loses some accuracy | Stance model `nli-deberta-v3-small` (inference on CPU) (C) |
| 21 | `hanselowski2018stance` — Hanselowski et al., FNC-1 retrospective (COLING) | 2018 | Fake News Challenge-1 headline–body pairs | Re-evaluation of top systems + feature-rich stackLSTM | Best original system (Athene) macro-F1 0.604; their stackLSTM 0.609 — far below the FNC-1 score | Headline–body stance ≠ claim truth; minority classes hard | Report macro-F1, not accuracy; headline–body mismatch flag in the temporal/context module (C) |
| 22 | `ribeiro2016lime` — Ribeiro, Singh & Guestrin, LIME (KDD) | 2016 | Any classifier (text, image) | Local sparse linear surrogate on perturbed inputs | User study: non-experts using LIME pick the better-generalising classifier and improve a classifier by removing spurious features (~89 % correct picks) | Explanations unstable across samples; perturbation cost | Token highlights for the DistilBERT signal, labelled "style only" (D) |
| 23 | `lundberg2017shap` — Lundberg & Lee, SHAP (NeurIPS) | 2017 | Any model | Shapley-value attributions unifying six methods | Unifies six additive attribution methods; better agreement with human intuition in their studies | Slow for long text | Optional report figures (D) |
| 24 | `shu2019defend` — Shu et al., dEFEND (KDD) | 2019 | FakeNewsNet PolitiFact + GossipCop with user comments | Sentence–comment co-attention; explanation = top sentences and comments | Not verified here — see paper Table 3 (accuracy on PolitiFact / GossipCop) | Needs user comments, which we do not have | Explanation = the evidence sentences that drove the decision, shown to the user (D) |
| 25 | `vosoughi2018spread` — Vosoughi, Roy & Aral (Science) | 2018 | ~126,000 Twitter cascades, ~3M people, 2006–2017 | Empirical diffusion analysis | Falsehoods 70 % more likely to be retweeted; truth took ~6× longer to reach 1,500 people | Descriptive, not a detector | Motivation statistics for §I (E) |
| 26 | `zubiaga2018rumours` — Zubiaga et al., rumour survey (ACM CSUR) | 2018 | — (survey) | Rumour detection → tracking → stance → veracity framework | Survey (no single number) | Social-media focus | Veracity resolved over time; stance as a sub-task; temporal framing (E) |
| 27 | `alzaman2021india` — Al-Zaman (Journalism and Media) | 2021 | 125 fact-checked COVID-19 fake-news items from India | Content analysis | 125 items; 67.2 % health-related (seven themes); text + video the most common format (47.2 %); 94.4 % originated in online/social media (Twitter, Facebook, WhatsApp, YouTube) vs. 5.6 % mainstream media | Small sample; one period; no truth-type breakdown (e.g. false context vs. fabricated) | Indian context for the motivation; video-plus-caption items are exactly the re-contextualised material our temporal check targets (E) |
| 28 | `zhou2020survey` — Zhou & Zafarani (ACM CSUR) | 2020 | — (survey) | Taxonomy: knowledge-, style-, propagation-, source-based detection | Survey | — | Positions us as knowledge-based (evidence) + style-based (classifier) hybrid (E) |
| 29 | `guo2022survey` — Guo, Schlichtkrull & Vlachos (TACL) | 2022 | — (survey) | Framework: claim detection → evidence retrieval → verdict prediction → justification production | Survey | — | Our pipeline follows these four stages; "justification production" = our explanation module (E) |
| 30 | `islam2020infodemic` — Islam et al. (AJTMH) | 2020 | 2,300+ rumour/stigma/conspiracy reports, 87 countries, Jan–Apr 2020 | Content analysis of social-media and news reports | ~800 deaths, 5,876 hospitalisations, 60 cases of blindness linked to the methanol "cure" rumour | Not a detector | Motivation statistic (E) |
| 31 | `wef2024risks` — World Economic Forum, Global Risks Report 2024 | 2024 | Global Risks Perception Survey | Expert survey | Misinformation and disinformation ranked #1 global risk over the two-year horizon | Not a detector | Motivation (E) |
| 32 | `bbc2018whatsapp` / `aljazeera2018whatsapp` — BBC News, Al Jazeera | 2018 | Reporting on WhatsApp child-kidnapping rumours in India | Journalism | BBC (18 Jul 2018): at least 17 killed across India since April 2018; Al Jazeera (17 Jul 2018): "more than two dozen" | Counts vary by outlet and period | Indian motivation example in §I; the paper quotes the conservative BBC figure (E) |

## Who explains which papers in the viva (MSE1 doc §3.7)

Navishka — LIAR (#1), WELFake (#5) · Naitik — FakeBERT (#12), DistilBERT (#11) · Naveen — FEVER (#6), KGAT (#15) · Prateek — ISOT (#3), Sentence-BERT (#17) · Nikhil — LIME (#22), Vosoughi (#25).

## Verification notes (where each "reported result" was checked, 28 Aug 2026)

| # | Checked at | Confidence |
|---|---|---|
| 1 | aclanthology.org/P17-2067 (Table 2) | high |
| 2 | ar5iv arXiv:1809.01286 (Table 3) | medium — re-check against the Big Data version before quoting in §VII |
| 3, 4 | Springer chapter page + Wiley abstract | medium-high (the "50k features" detail was not re-verified) |
| 5 | ResearchGate / IEEE Xplore abstract | high |
| 6 | aclanthology.org/N18-1074 abstract | high |
| 7 | arXiv:1909.03242 | medium-high (claim count from the paper title/abstract) |
| 8 | arXiv:2004.14425 | high (size); the paper has no classifier benchmark table |
| 9 | dl.acm.org/doi/10.1145/3097983.3098131 | medium-high |
| 10 | aclanthology.org/N19-1423 abstract | high |
| 11 | arXiv:1910.01108 abstract | high |
| 12 | PMC7788551 | high |
| 13 | ar5iv arXiv:1708.07104 tables | medium — preprint numbers; confirm with the COLING PDF |
| 14 | ojs.aaai.org AAAI 4662 | high |
| 15 | aclanthology.org/2020.acl-main.655 (Table 2) | high |
| 16 | arXiv:1809.06416 | medium |
| 17 | aclanthology.org/D19-1410 abstract | high |
| 18 | HF model card sentence-transformers/all-MiniLM-L6-v2 | high |
| 19 | ar5iv arXiv:1704.05426 | high |
| 20 | arXiv:2111.09543 + HF model card cross-encoder/nli-deberta-v3-small | high |
| 21 | aclanthology.org/C18-1158 | high |
| 22 | arXiv:1602.04938 | medium-high |
| 23 | papers.nips.cc 7062 | high |
| 24 | primary PDF not machine-readable in our tooling — number deliberately left out | — |
| 25 | PubMed 29590045 abstract | high |
| 26 | arXiv:1704.00656 | high |
| 27 | doi.org/10.3390/journalmedia2010007 | high |
| 28 | dl.acm.org/doi/10.1145/3395046 | high |
| 29 | aclanthology.org/2022.tacl-1.11 | high |
| 30 | ajtmh.org 103/4/1621 | high |
| 31 | weforum.org press release (the PDF itself blocked automated fetch) | high on the ranking; lift the verbatim sentence from p. 8–9 of the PDF if a quote is needed |
| 32 | bbc.com/news/world-asia-india-44856910 (via text proxy + Wayback), aljazeera.com features 2018/7/17 | medium-high / high |
