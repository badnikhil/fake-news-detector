"""MSE1 data-pipeline checks (run `make data` first; tests skip when an artefact is missing)."""
from __future__ import annotations

import pandas as pd
import pytest

from src.config import (
    LIAR_TO_3,
    LIAR_TO_5,
    PROCESSED_DIR,
    SCHEMA,
    SPLIT_DATASETS,
    SPLIT_NAMES,
    SPLIT_RATIOS,
    SPLITS_DIR,
)

LIAR_OFFICIAL = {"train": 10_269, "val": 1_284, "test": 1_283}


def _parquet(name: str) -> pd.DataFrame:
    p = PROCESSED_DIR / f"{name}.parquet"
    if not p.exists():
        pytest.skip(f"{p} missing - run `make data`")
    return pd.read_parquet(p)


def _split(name: str, s: str) -> pd.DataFrame:
    p = SPLITS_DIR / f"{name}_{s}.csv"
    if not p.exists():
        pytest.skip(f"{p} missing - run `make data`")
    return pd.read_csv(p)


# ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["welfake", "isot", "liar", "fever_subset", "fnn_politifact"])
def test_schema_columns_identical(name):
    df = _parquet(name)
    assert list(df.columns) == SCHEMA
    assert df["id"].is_unique
    assert (df["source_dataset"] == name).all()
    assert (df["text"].str.len() > 0).all()


@pytest.mark.parametrize("name", ["welfake", "isot", "fnn_politifact"])
def test_binary_label_vocabulary(name):
    df = _parquet(name)
    assert set(df["label"].unique()) <= {"fake", "real"}
    assert set(df["label5"].unique()) <= {"FAKE", "REAL"}
    assert set(df["label3"].unique()) <= {"FALSE", "TRUE"}


def test_liar_label_mappings():
    df = _parquet("liar")
    assert set(df["label"].unique()) == set(LIAR_TO_5)
    assert (df["label5"] == df["label"].map(LIAR_TO_5)).all()
    assert (df["label3"] == df["label"].map(LIAR_TO_3)).all()


def test_liar_official_splits_unchanged():
    df = _parquet("liar")
    assert df["split"].value_counts().to_dict() == LIAR_OFFICIAL
    for s, n in LIAR_OFFICIAL.items():
        assert len(_split("liar", s)) == n


def test_fever_subset_balanced():
    df = _parquet("fever_subset")
    assert set(df["label"].unique()) == {"SUPPORTS", "REFUTES", "NEUTRAL"}
    for s, n in (("train", 20_000), ("val", 3_000)):
        part = df[df["split"] == s]
        assert len(part) == n
        counts = part["label"].value_counts()
        assert counts.max() - counts.min() <= 1


@pytest.mark.parametrize("name", ["welfake", "isot"])
def test_split_ratios(name):
    df = _parquet(name)
    n = len(df)
    for s in SPLIT_NAMES:
        frac = len(_split(name, s)) / n
        assert abs(frac - SPLIT_RATIOS[s]) <= 0.005, f"{name} {s}: {frac:.4f}"


@pytest.mark.parametrize("name", ["welfake", "isot"])
def test_split_stratified(name):
    df = _parquet(name)
    overall = df["label"].value_counts(normalize=True)
    for s in SPLIT_NAMES:
        part = df[df["split"] == s]["label"].value_counts(normalize=True)
        assert (part.reindex(overall.index).fillna(0) - overall).abs().max() <= 0.01


@pytest.mark.parametrize("name", SPLIT_DATASETS)
def test_split_csvs_disjoint_and_cover_parquet(name):
    df = _parquet(name)
    ids = {s: set(_split(name, s)["id"]) for s in SPLIT_NAMES}
    assert not (ids["train"] & ids["test"])
    assert not (ids["train"] & ids["val"])
    assert not (ids["val"] & ids["test"])
    assert ids["train"] | ids["val"] | ids["test"] == set(df["id"])
    for s in SPLIT_NAMES:
        assert (df.loc[df["split"] == s, "id"].sort_values().to_list()
                == sorted(ids[s]))


@pytest.mark.parametrize("name", ["welfake", "isot"])
def test_no_exact_duplicate_text_across_splits(name):
    from src.preprocess.dedupe import exact_hash

    df = _parquet(name)
    h = df["text"].map(exact_hash)
    assert not h.duplicated().any()
    train = set(h[df["split"] == "train"])
    test = set(h[df["split"] == "test"])
    assert not (train & test)


def test_isot_reuters_artefact_removed():
    df = _parquet("isot")
    assert int(df["text"].str.contains(r"\(Reuters\)", regex=True).sum()) == 0
    assert int(df["text"].str.contains(r"^\s*[A-Z ]+\(Reuters\)", regex=True).sum()) == 0


@pytest.mark.parametrize("name", ["welfake", "isot"])
def test_cased_text_preserved(name):
    df = _parquet(name)
    sample = df["text"].head(2000)
    assert (sample != sample.str.lower()).mean() > 0.9, "text looks lower-cased; parquet must keep case"


@pytest.mark.parametrize("name", ["welfake", "isot"])
def test_no_urls_or_handles_left(name):
    df = _parquet(name)
    assert int(df["text"].str.contains(r"https?://", regex=True).sum()) == 0
    assert int(df["text"].str.contains(r"pic\.twitter\.com", regex=True).sum()) == 0


def test_strip_artefacts_unit():
    from src.preprocess.artefacts import strip_artefacts

    assert strip_artefacts("WASHINGTON (Reuters) - The Senate voted.") == "The Senate voted."
    assert strip_artefacts("NEW YORK/LONDON (Reuters) - Stocks rose.") == "Stocks rose."
    assert "(Reuters)" not in strip_artefacts("Text (Reuters) middle")
    assert "@realDonaldTrump" not in strip_artefacts("Tweet by @realDonaldTrump today")
    assert "http" not in strip_artefacts("see https://example.com/x for more")
    assert "Featured image" not in strip_artefacts("Body.\nFeatured image via screengrab")


def test_stratified_split_deterministic():
    from src.preprocess.split import stratified_split

    df = pd.DataFrame({"label": ["a", "b"] * 500})
    s1 = stratified_split(df)
    s2 = stratified_split(df)
    assert s1.equals(s2)
    assert s1.value_counts().to_dict() == {"train": 800, "val": 100, "test": 100}
