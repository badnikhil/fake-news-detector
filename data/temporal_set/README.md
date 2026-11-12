# Temporal Set — `temporal_set.csv`

30 claims for testing the "old news presented as new" detector (objective O5, `src/temporal/checks.py`). There are **15 positives** (`old_as_new=1`), where old media or an old event was re-shared as if it were current, and **15 controls** (`old_as_new=0`), which are claims about events that were current when they were shared. Every value was taken from a published professional fact-check that was opened and read. Built 2026-11-12.

## Columns

| column | meaning |
|---|---|
| `id` | `TS001…TS015` = positives, `TS016…TS030` = controls |
| `claim` | the claim as shared, taken from the page's schema.org ClaimReview `claimReviewed`. TS012 is the exception: it uses BOOM's "Claim 2" heading verbatim, because that article's ClaimReview covers two claims. English only. |
| `old_as_new` | 1 = the page says the media or event is from an earlier date than the post implies; 0 = the page says the media or event is current, and the timing was not the issue |
| `t_post` | the date the claim or post was shared, YYYY-MM-DD. When the page gives only a month, the first day of that month is used and the notes say so. **Left empty when the page gives no date** (for example "just days before Halloween" or "recently shared"). |
| `t_event` | the original date of the media or event as the page states it. A month only becomes the first day of the month, and a year only becomes YYYY-01-01. The notes say which. |
| `rating` | the publisher's rating, verbatim (ClaimReview `alternateName`) |
| `publisher`, `factcheck_url`, `factcheck_date` | where the rating comes from. The date is the ClaimReview `datePublished`, date part only (Snopes UTC, BOOM/Alt News IST). |
| `date_quote` | a short verbatim quote from the page that states the original date (positives) or the timing (controls) |
| `notes` | caveats: approximated dates, mixed issues (old + cropped, old + wrong person), short gaps |

## Counts

| | Snopes | BOOM | Alt News | total |
|---|---|---|---|---|
| positives (1) | 11 | 3 | 1 | 15 |
| controls (0) | 13 | 1 | 1 | 15 |

Positive ratings: Outdated 7, Miscaptioned 4, False 4 (BOOM/Alt News "old video" items). Control ratings: True 8, Miscaptioned 5, False 2. The Miscaptioned and False controls are wrong about content, place or person, and their pages state that the footage is from the time it was posted. TS030 is a deliberate hard negative: the claim says an image is old, and Alt News rates that claim False because the image is current.

## How candidates were found and verified

- Candidate lists came from the Snopes rating listings `/fact-check/rating/{outdated,miscaptioned,true}/`, the BOOM `/fact-check` listing and the Alt News home page, all fetched on 2026-11-12. AFP and Reuters were not used because they block scripted fetches.
- For each candidate, the page HTML was downloaded. The claim, rating and publication date were read from its ClaimReview JSON-LD, and the article text was read for t_post and t_event. A build script checked that every `date_quote` occurs verbatim in the page text (whitespace-normalised) and that no URL or claim is already in `data/live_claims/live_claims.csv`. Nine Snopes candidates were already in that set and were excluded.
- No date was inferred across pages or from a calendar. When a page gives a weekday ("on Tuesday"), a relative time ("late October"), or no date at all, the date was approximated as described above or left empty.

## Limitations and items not used

- **15 of the 30 rows have an empty `t_post`** (4 positives — TS008, TS011, TS014, TS015 — and 11 controls; count corrected 2026-11-12 by the evaluation agent; Snopes "True" pages usually say only "in early November"). Approximated dates appear in the notes of 7 `t_post` and 8 `t_event` values. Any metric that needs both dates should either skip the empty rows or fall back to `factcheck_date` as an upper bound on `t_post`, and should say which it did.
- Two positives have short gaps, about 7 weeks (TS008) and about 10 days (TS014). Several positives are also wrong-person or cropped (TS007, TS011, TS015), so the old date is not the only problem.
- TS019 (Time cover) calls Nov. 4 "the day of the midterm elections", but TS022's page says Nov. 3. Both are recorded as their pages state them.
- These candidates were reviewed and not used: Snopes `anthem-blue-cross-shield-anesthesia` and `charles-rogers-medal-of-honor` (rated Outdated because a policy or page changed, not because old content was re-shared); `trump-springsteen-leave-usa` (the excerpt read gave no date for the original quote); `germany-gel-regrow-joint-cartilage` (a duplicate of TS003); `patrick-stewart-ian-mckellen-tribute` and `mechanical-dinosaur-death` (no event or post date found); BOOM's Kumbh photo and Arunachal items (mixed or unclear claims).
- The set is small (30) and Snopes-heavy (24/30). Report O5 numbers with that caveat.
