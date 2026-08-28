# data/ — datasets, provenance, licences, processed layout

`make download` fetches every raw file below into `data/raw/<dataset>/` **without a Kaggle account** and records size,
SHA-256 and download date in `data/raw/manifest.json` (mirrored in the table). `make data` turns them into the unified
parquet files in `data/processed/` and the split ID lists in `data/splits/`. `data/raw/`, `data/processed/` and the
split CSVs are git-ignored and regenerated; only this README and `data/splits/.gitkeep` are committed.

## Raw datasets (downloaded 2026-08-28)

| ID | Dataset | Source URL (as used by `scripts/download_data.py`) | Licence (verified on the source page) | Raw file(s) | Size (bytes) | SHA-256 | Rows |
|---|---|---|---|---|---|---|---|
| D1 | **WELFake** (Verma et al., 2021) | Zenodo record 4561253 → `https://zenodo.org/api/records/4561253/files/WELFake_Dataset.csv/content` (URL resolved through the Zenodo API) | **CC BY 4.0** (`metadata.license.id = cc-by-4.0` in the Zenodo API record; access right *open*) | `welfake/WELFake_Dataset.csv` | 245,086,152 | `665331424230fc452e9482c3547a6a199a2c29745ade8d236950d1d105223773` | 72,134 (35,028 real / 37,106 fake) |
| D2 | **ISOT Fake News** (Ahmed, Traore & Saad, 2017) | University of Victoria ISOT lab: `https://onlineacademiccommunity.uvic.ca/isot/wp-content/uploads/sites/7295/2023/03/News-_dataset.zip` (linked from https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/) | Free for research / academic use (ISOT lab page asks to cite Ahmed et al. 2017/2018; no formal licence text) | `isot/News-_dataset.zip` → `True.csv`, `Fake.csv` | 43,106,824 | `1846d4ca7abfedb53a4f7c233489d23a0e6c8bb79e7f56fb8345c4faceae5268` | 44,898 (21,417 true / 23,481 fake) |
| D3 | **LIAR** (Wang, 2017) | `https://www.cs.ucsb.edu/~william/data/liar_dataset.zip` (redirects to sites.cs.ucsb.edu) | Academic / research use (README in the zip: "for research purposes only"; PolitiFact content) | `liar/liar_dataset.zip` → `train.tsv`, `valid.tsv`, `test.tsv`, `README` | 1,013,571 | `611c1addad919743dde15822b87a60bfb760d8f85597f25289e34621800654c7` | 12,836 (train 10,269 / valid 1,284 / test 1,283) |
| D4 | **FEVER** v1.0 (Thorne et al., 2018) | `https://fever.ai/download/fever/train.jsonl`, `https://fever.ai/download/fever/shared_task_dev.jsonl` (from https://fever.ai/dataset/fever.html) | **CC BY-SA 3.0** (`"license": "https://creativecommons.org/licenses/by-sa/3.0/legalcode"` in the fever.ai page metadata; Wikipedia-derived) | `fever/train.jsonl` | 33,024,303 | `eba7e8f87076753f8494718b9a857827af7bf73e76c9e4b75420207d26e588b6` | 145,449 claims (SUPPORTS 80,035 / REFUTES 29,775 / NEI 35,639) |
| | | | | `fever/shared_task_dev.jsonl` | 4,349,935 | `e89865bfe1b4dd054e03dd57d7241a6fde24862905f31117cf0cd719f7c78df7` | 19,998 claims (6,666 each) |
| D5 | **FakeNewsNet — PolitiFact** (Shu et al., 2018) | `https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/dataset/politifact_fake.csv`, `…/politifact_real.csv` | Research use — the repo carries no licence file; README asks to cite Shu et al. (2018/2017); only titles + URLs are redistributed by the authors (no article text, no tweets) | `fnn_politifact/politifact_fake.csv` | 3,286,418 | `abe7fe7aad801b1e2ab5fc963cbd0881d9670190262ae3df85e91c3380bd004c` | 432 fake |
| | | | | `fnn_politifact/politifact_real.csv` | 8,278,658 | `2500f86a7addca0f59fe8cf089c0f802d3eca6980f53538992b3b0e9687dbae0` | 624 real |
| D6 | ClaimBuster | not downloaded at MSE1 (optional; Zenodo record 3609356) | — | — | — | — | — |
| D7 | Live Claims Set (ours) | hand-built in weeks 9–13 → `data/live_claims/` | created by the team | — | — | — | — |

Re-running `make download` is idempotent (skips files whose size and SHA-256 match the manifest); `--force` re-fetches.

**ISOT fallback order** in `scripts/download_data.py`: (a) the UVic direct zip above — **this is what worked on 2026-08-28**;
(b) a Hugging Face Hub mirror verified by row counts (21,417 / 23,481) — none configured because (a) works;
(c) if both fail the script prints instructions for the manual Kaggle download (`clmentbisaillon/fake-and-real-news-dataset`
→ copy `True.csv` + `Fake.csv` into `data/raw/isot/`) and `make data` continues without ISOT.

## Label conventions

* **WELFake quirk (important).** The Zenodo/Kaggle description says `0 = fake, 1 = real`, but the CSV is the other way round:
  label 0 rows (35,028) are the *real* articles (60 % start with a Reuters dateline, 0 % have `[VIDEO]` titles) and label 1 rows
  (37,106) are the *fake* ones (20 % end with "Featured image via …", 15 % `[VIDEO]` titles). The authors' own counts
  (35,028 real / 37,106 fake) confirm this. `src/preprocess/load.py` therefore maps **0 → real, 1 → fake**.
* **ISOT**: `True.csv` = real (all Reuters), `Fake.csv` = fake.
* **LIAR 6 → 5 / 3** (master doc §10.1): true, mostly-true → REAL / TRUE; half-true → PARTIALLY TRUE / MIXED;
  barely-true → MISLEADING / MIXED; false, pants-fire → FAKE / FALSE. UNVERIFIABLE never comes from a dataset.
* **FEVER → stance** (§10.2): SUPPORTS → SUPPORTS, REFUTES → REFUTES, NOT ENOUGH INFO → NEUTRAL.
* Binary datasets get `label5 = REAL/FAKE`, `label3 = TRUE/FALSE` so every parquet has the same columns.

## Processed layout (`make data`, seed 42)

Unified schema for every parquet: `id, title, text, label, label5, label3, source_dataset, date, split, meta`
(`meta` = JSON string with dataset-specific extras: ISOT subject, LIAR speaker/party/context/history counts,
FEVER evidence pages + sentence ids, FakeNewsNet news_url + tweet count).

| File | Content | Rows (2026-08-28 run) |
|---|---|---|
| `processed/welfake.parquet` | WELFake after artefact stripping, empty/short drop, exact + near dedup; `split` ∈ train/val/test (80/10/10) | 60,847 (34,296 real / 26,551 fake; train 48,677 / val 6,085 / test 6,085) |
| `processed/isot.parquet` | ISOT, same pipeline (Reuters datelines removed) | 37,608 (20,905 real / 16,703 fake; train 30,086 / val 3,761 / test 3,761) |
| `processed/liar.parquet` | LIAR, official splits kept (nothing dropped), `label5`/`label3` added | 12,836 |
| `processed/fever_subset.parquet` | balanced subset: 20,000 train + 3,000 dev (`split` = train/val) | 23,000 |
| `processed/fnn_politifact.parquet` | FakeNewsNet PolitiFact titles (passthrough) | 1,056 |
| `splits/{welfake,isot,liar}_{train,val,test}.csv` | `id,label` in sorted-id order (byte-identical across runs) | — |
| `processed/make_data_summary.json` | machine-readable per-dataset counts of the last run | — |

Per-dataset counts (rows in → dropped empty / short → exact dups → near dups → rows out) are in `docs/mse1_make_data.log`
and summarised in `agent-docs/README.md` / `notebooks/00_datasets.ipynb` §2.2.

Pipeline (master §11.1): load → *(ISOT only: artefact-only leakage classifier on the raw text)* → strip artefacts
(Reuters datelines / `(Reuters)`, "Featured image via…", "Read more:", "21st Century Wire says…", URLs, e-mails,
@handles, pic.twitter.com) → NFKC + quote/whitespace normalisation (case kept) → drop empty and < 20-token texts
(articles only) → exact dedup (SHA-1 of normalised text) → near dedup (MinHash, word 5-gram shingles, 128 perms,
Jaccard ≥ 0.9) → stratified 80/10/10 split (`random_state=42`) → cross-split overlap assertion (0) → parquet + CSVs.
