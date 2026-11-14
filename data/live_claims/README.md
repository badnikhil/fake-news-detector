# Live Claims Set (D7) — `live_claims.csv`

95 recent claims (fact-checks published 2025-02 → 2026-10-25, 93 of them in 2026), each taken from a published professional fact-check and labelled with the project's 5-class scheme (master §6, §10 D7). Used for end-to-end evaluation (`docs/results/e2e.md`, ESE) and fusion-threshold tuning on the 30-item dev slice (`docs/results/fusion_thresholds.md`, MSE2). Built 2026-10-26; load with `src.eval.live_claims.load_live_claims(split=...)`, check with `python -m src.eval.live_claims` / `tests/test_live_claims.py`.

## Columns

| column | meaning |
|---|---|
| `id` | `LC001…`, ordered by fact-check date |
| `claim` | the claim as the fact-checker states it (PolitiFact: the quoted statement; ClaimReview sites: `claimReviewed`). This is the pipeline input. |
| `label` | REAL / FAKE / PARTIALLY TRUE / MISLEADING / UNVERIFIABLE (mapping below) |
| `original_rating` | the fact-checker's rating, verbatim (Full Fact: its free-text verdict) |
| `publisher`, `factcheck_url`, `factcheck_date` | where the rating comes from; date = publication date of the fact-check |
| `claim_date` | when the claim was made, if the fact-check says so (PolitiFact "stated on …", Lead Stories "the claim appeared … on …"); empty otherwise |
| `claimant`, `context` | PolitiFact speaker and venue ("in an X post"); empty for other publishers |
| `input_type` | `headline` for all items (short claims) |
| `modality` | `media` if the claim is about an image/video/photo/screenshot (12 items; a text-only pipeline can at best reach the fact-check itself), else `text` |
| `split` | `dev` (30 items, fusion tuning) / `eval` (65 items, held out) |
| `notes` | labelling caveats |

## Rating → 5-class mapping

| Publisher | Ratings → class |
|---|---|
| PolitiFact (same as LIAR, master §10.1) | True, Mostly True → **REAL**; Half True → **PARTIALLY TRUE**; Mostly False (= LIAR barely-true) → **MISLEADING**; False, Pants on Fire → **FAKE** |
| Snopes | True → **REAL**; Mixture → **PARTIALLY TRUE**; Miscaptioned, Outdated (once true, re-shared as current = old-as-new) → **MISLEADING**; False, Fake → **FAKE** |
| BOOM, Alt News (ClaimReview) | False → **FAKE**; Misleading / Mostly False → **MISLEADING** |
| Full Fact | first word of the verdict: "False." → **FAKE**; "Missing context." → **MISLEADING** |
| Lead Stories | evidence-absence ratings — Unverified, No Proof, No Reports, Not Official, and "What We Know" explainers that leave the question open → **UNVERIFIABLE** |

Snopes retired its Unproven / Unfounded ratings (snopes.com/fact-check-ratings, checked 2026-10-26), and PolitiFact has no such rating, so all 9 UNVERIFIABLE items come from Lead Stories. Lead Stories' article text still says "No, that's not true" for those, but every one rests only on missing evidence (no source given, no credible reports, nobody could confirm) and none cites contrary evidence. That is exactly where our system is supposed to return UNVERIFIABLE. Lead Stories items whose verdict did cite contrary evidence (police denial, AI detector, "he is alive", "no runway") were left out instead of being labelled FAKE by us. **UNVERIFIABLE is therefore the noisiest and smallest class (9); report it with that caveat.**

## Counts

| label | dev | eval | total |
|---|---|---|---|
| FAKE | 8 | 19 | 27 |
| REAL | 7 | 14 | 21 |
| PARTIALLY TRUE | 6 | 13 | 19 |
| MISLEADING | 6 | 13 | 19 |
| UNVERIFIABLE | 3 | 6 | 9 |
| **total** | **30** | **65** | **95** |

Publishers: PolitiFact 46, Snopes 31, Lead Stories 9, BOOM 4, Alt News 3, Full Fact 2.

## Provenance and verification

Every row was checked against the live fact-check page on 2026-10-26: the page was downloaded and the rating, the claim and the date were read from the page itself. For PolitiFact that means the statement block (`pf-statement-quote` plus the meter image's `alt`, e.g. "Mostly False"). For Snopes, Lead Stories, BOOM, Alt News and Full Fact it means the schema.org `ClaimReview` JSON-LD (`claimReviewed`, `reviewRating.alternateName`, `datePublished`). Nothing was typed in by hand except the Lead Stories claim dates, which were read from the article text. Candidates came from PolitiFact's per-rating list pages, Snopes' rating pages, the Lead Stories monthly archives and the BOOM / Alt News / Full Fact RSS feeds. A claim covered by two fact-checkers appears once. The PolitiFact copy was kept for the poll-worker ballot claim and the Talarico "mental hospital" claim. The "Biden cut 440 reporters' credentials" claim was dropped because PolitiFact (Mostly False) and Snopes (Mixture) disagree. Spanish-language duplicates and items targeting private individuals were not used. AFP and Reuters Fact Check block scripted access (HTTP 403/401) and FactCheck.org's feed had no machine-readable ratings, so none of their items are included.

Selection: dev = 30 items stratified by label (proportional quotas, largest remainder), sampled with `random.Random(42)` from the id-sorted rows of each label.

**Double labelling (done 2026-11-14):** Second labelling = AI-assisted blind pass (not a human annotator), accepted by the team on 2026-11-14; Cohen's κ = 0.9865 (n = 95); 1 disagreement (LC023) adjudicated (`docs/results/agreement.md`).

**Not yet done (master §15, R7):** growing the set to about 100 items with at least 15 per class (UNVERIFIABLE is the gap). Fact-check pages can be updated after publication, so re-check `original_rating` before the ESE run.

## Second labeller — instructions (blind double labelling, ESE DoD)

Fill **`second_labeller_sheet.csv`** (generated by `python -m src.eval.agreement sheet`; columns `id, claim, factcheck_url, label2, labeller, notes`). It deliberately shows no first label or rating. **Do not open `live_claims.csv` (its `label` / `original_rating` columns) or the rating table above until you have finished.**

1. For each row, open `factcheck_url`, read the claim and the fact-check article, and find the fact-checker's verdict.
2. Write one of the five classes in `label2` (exact spelling), your name in `labeller`, and anything uncertain in `notes`:
   - **REAL**: the claim is accurate (PolitiFact True / Mostly True, Snopes True).
   - **FAKE**: the claim is false or fabricated, and the article gives positive contrary evidence (False, Pants on Fire, Fake).
   - **PARTIALLY TRUE**: some parts are accurate and some are not (Half True, Mixture).
   - **MISLEADING**: true in a narrow or old sense but distorted: out of date, miscaptioned, missing context, or Mostly False.
   - **UNVERIFIABLE**: the article finds no evidence either way. It rests only on missing proof, such as no source, no credible reports or no confirmation, and cites no contrary evidence.
3. Label from the claim + the article's verdict, not from what you believe about the topic. If a page will not open, write `UNREADABLE`. Those rows are left out of κ and the number left out is reported.
4. Run `make agreement`. It writes Cohen's κ against `label`, a confusion table and per-class agreement to `docs/results/agreement.md`. The team adjudicates disagreements, and `live_claims.csv` changes only after the team agrees.

`second_pass_ai.csv` is an AI-assisted blind re-read of every URL (Claude sub-agents, 2026-11-12; given only id / claim / URL). **The team accepted it as the second labeller on 2026-11-14** (it is not a human annotator); the blind human sheet above stays optional.

## Adjudication log

| id | label | second pass | decision (2026-11-14) |
|---|---|---|---|
| LC023 | UNVERIFIABLE | FAKE | **UNVERIFIABLE kept, `live_claims.csv` unchanged.** Lead Stories rating "What We Know"; the article leaves the question open: "Lead Stories repeatedly tried to contact Aldi UK to get a definitive answer, but did not receive a response." The mapping above sends open "What We Know" explainers to UNVERIFIABLE. (Split: eval.) |
