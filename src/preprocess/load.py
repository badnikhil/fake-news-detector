"""Load each raw dataset from ``data/raw/`` into the unified schema (master doc §11.1 step 1).

Unified columns (``src.config.SCHEMA``)::

    id              str   globally unique, deterministic ("welfake_12", "isot_true_7", "liar_2635", ...)
    title           str|None
    text            str   raw cased text (article body / statement / claim / headline)
    label           str   dataset-native label in project vocabulary: "fake"/"real" (WELFake, ISOT, FNN),
                          LIAR 6-way ("true", "mostly-true", ...), FEVER stance ("SUPPORTS"/"REFUTES"/"NEUTRAL")
    label5          str|None  project 5-class mapping (REAL/FAKE/PARTIALLY TRUE/MISLEADING); None for FEVER
    label3          str|None  3-way collapse (TRUE/MIXED/FALSE); None for FEVER
    source_dataset  str   "welfake" | "isot" | "liar" | "fever_subset" | "fnn_politifact"
    date            str|None  ISO date when the dataset has one (ISOT only)
    split           str|None  official split where one exists (LIAR train/val/test, FEVER train/val); filled later
    meta            str   JSON string with dataset-specific extras (ISOT subject, LIAR speaker/party, FEVER evidence, ...)

Only loading + schema mapping happens here; cleaning lives in ``clean.py`` / ``artefacts.py``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import (
    BINARY_TO_3,
    BINARY_TO_5,
    FEVER_TO_STANCE,
    LIAR_TO_3,
    LIAR_TO_5,
    RAW_DIR,
    SCHEMA,
)

LIAR_COLUMNS = [
    "id", "label", "statement", "subject", "speaker", "speaker_job", "state", "party",
    "barely_true_counts", "false_counts", "half_true_counts", "mostly_true_counts",
    "pants_on_fire_counts", "context",
]


def _finish(df: pd.DataFrame) -> pd.DataFrame:
    """Order columns per SCHEMA, coerce dtypes, make sure ids are unique."""
    for col in SCHEMA:
        if col not in df.columns:
            df[col] = None
    df = df[SCHEMA].copy()
    for col in ("id", "title", "text", "label", "label5", "label3", "source_dataset", "date", "split", "meta"):
        df[col] = df[col].astype("string")
    if df["id"].duplicated().any():
        raise ValueError(f"duplicate ids in {df['source_dataset'].iloc[0]}")
    return df.reset_index(drop=True)


def _parse_date(series: pd.Series) -> pd.Series:
    d = pd.to_datetime(series, errors="coerce", format="mixed")
    return d.dt.strftime("%Y-%m-%d").where(d.notna(), None)


# ------------------------------------------------------------------------------------------------
def load_welfake(raw: Path = RAW_DIR) -> pd.DataFrame:
    """WELFake_Dataset.csv: columns ``Unnamed: 0`` (serial), title, text, label.

    NOTE: the Zenodo/Kaggle description says "0 = fake, 1 = real" but the file itself is the other way round:
    label 0 rows (35,028) are the real articles (60% carry a Reuters dateline, none say [VIDEO]) and label 1
    rows (37,106) are the fake ones ("Featured image via ...", ALL-CAPS titles, [VIDEO]).  The paper's own
    counts (35,028 real / 37,106 fake) confirm this, so we map 0 -> real, 1 -> fake.  Verified 2026-08-28.
    """
    df = pd.read_csv(raw / "welfake" / "WELFake_Dataset.csv")
    serial = df.columns[0]
    out = pd.DataFrame(
        {
            "id": "welfake_" + df[serial].astype(int).astype(str),
            "title": df["title"],
            "text": df["text"],
            "label": df["label"].map({0: "real", 1: "fake"}),
            "source_dataset": "welfake",
            "meta": "{}",
        }
    )
    out["label5"] = out["label"].map(BINARY_TO_5)
    out["label3"] = out["label"].map(BINARY_TO_3)
    return _finish(out)


def load_isot(raw: Path = RAW_DIR) -> pd.DataFrame | None:
    """ISOT True.csv (Reuters, label real) + Fake.csv: columns title, text, subject, date."""
    d = raw / "isot"
    if not ((d / "True.csv").exists() and (d / "Fake.csv").exists()):
        return None
    parts = []
    for fname, label in (("True.csv", "real"), ("Fake.csv", "fake")):
        df = pd.read_csv(d / fname)
        parts.append(
            pd.DataFrame(
                {
                    "id": f"isot_{label}_" + pd.RangeIndex(len(df)).astype(str),
                    "title": df["title"],
                    "text": df["text"],
                    "label": label,
                    "source_dataset": "isot",
                    "date": _parse_date(df["date"]),
                    "meta": df["subject"].map(lambda s: json.dumps({"subject": s})),
                }
            )
        )
    out = pd.concat(parts, ignore_index=True)
    out["label5"] = out["label"].map(BINARY_TO_5)
    out["label3"] = out["label"].map(BINARY_TO_3)
    return _finish(out)


def load_liar(raw: Path = RAW_DIR) -> pd.DataFrame:
    """LIAR train/valid/test TSV (14 columns, no header). Official splits are kept (train/val/test)."""
    d = raw / "liar"
    parts = []
    for fname, split in (("train.tsv", "train"), ("valid.tsv", "val"), ("test.tsv", "test")):
        df = pd.read_csv(d / fname, sep="\t", header=None, names=LIAR_COLUMNS, quoting=3, dtype=str,
                         keep_default_na=False)
        meta_cols = [c for c in LIAR_COLUMNS if c not in ("id", "label", "statement")]
        parts.append(
            pd.DataFrame(
                {
                    "id": "liar_" + df["id"].str.replace(".json", "", regex=False),
                    "title": None,
                    "text": df["statement"],
                    "label": df["label"],
                    "source_dataset": "liar",
                    "split": split,
                    "meta": df[meta_cols].apply(lambda r: json.dumps(r.to_dict()), axis=1),
                }
            )
        )
    out = pd.concat(parts, ignore_index=True)
    bad = set(out["label"]) - set(LIAR_TO_5)
    if bad:
        raise ValueError(f"unexpected LIAR labels: {bad}")
    out["label5"] = out["label"].map(LIAR_TO_5)
    out["label3"] = out["label"].map(LIAR_TO_3)
    return _finish(out)


def _read_fever(path: Path) -> pd.DataFrame:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            ev = r.get("evidence") or []
            # evidence: list of annotation sets; each item = [annotation_id, evidence_id, page, sentence_id]
            pages = sorted({e[2] for s in ev for e in s if len(e) >= 4 and e[2]})
            rows.append(
                {
                    "id": f"fever_{r['id']}",
                    "text": r["claim"],
                    "label": r["label"],
                    "meta": json.dumps(
                        {"verifiable": r.get("verifiable"), "evidence_pages": pages,
                         "evidence": [[e[2], e[3]] for s in ev for e in s if len(e) >= 4 and e[2]]}
                    ),
                }
            )
    return pd.DataFrame(rows)


def load_fever(raw: Path = RAW_DIR) -> dict[str, pd.DataFrame]:
    """FEVER train.jsonl (145k) and shared_task_dev.jsonl (~20k). Returns the FULL sets keyed by split;
    balanced sampling to 20k/3k happens in ``run_all.sample_fever``."""
    d = raw / "fever"
    out = {}
    for fname, split in (("train.jsonl", "train"), ("shared_task_dev.jsonl", "val")):
        df = _read_fever(d / fname)
        df["title"] = None
        df["label"] = df["label"].map(FEVER_TO_STANCE)
        if df["label"].isna().any():
            raise ValueError("unexpected FEVER label")
        df["source_dataset"] = "fever_subset"
        df["split"] = split
        out[split] = _finish(df)
    return out


def load_fnn_politifact(raw: Path = RAW_DIR) -> pd.DataFrame:
    """FakeNewsNet politifact_{fake,real}.csv: id, news_url, title, tweet_ids (titles only; no crawling)."""
    d = raw / "fnn_politifact"
    parts = []
    for fname, label in (("politifact_fake.csv", "fake"), ("politifact_real.csv", "real")):
        df = pd.read_csv(d / fname, dtype=str, keep_default_na=False)
        parts.append(
            pd.DataFrame(
                {
                    # two politifact ids occur in BOTH fake and real CSVs -> label goes into the id
                    "id": f"fnn_{label}_" + df["id"],
                    "title": df["title"],
                    "text": df["title"],  # title is the only text we have (offline index uses it)
                    "label": label,
                    "source_dataset": "fnn_politifact",
                    "meta": df.apply(
                        lambda r: json.dumps(
                            {"news_url": r["news_url"], "n_tweets": len(r["tweet_ids"].split("\t")) if r["tweet_ids"] else 0}
                        ),
                        axis=1,
                    ),
                }
            )
        )
    out = pd.concat(parts, ignore_index=True)
    out["label5"] = out["label"].map(BINARY_TO_5)
    out["label3"] = out["label"].map(BINARY_TO_3)
    return _finish(out)
