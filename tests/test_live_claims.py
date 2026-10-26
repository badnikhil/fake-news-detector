"""Live Claims Set: schema, size, labels, dev/eval split (`src/eval/live_claims.py`)."""
from __future__ import annotations

import pandas as pd
import pytest

from src.config import VERDICTS
from src.eval.live_claims import PATH, load_live_claims, validate


@pytest.fixture(scope="module")
def df():
    return load_live_claims()


def test_size_and_labels(df):
    assert len(df) >= 60
    assert set(df.label) == set(VERDICTS)  # every class present


def test_dev_slice_30_disjoint_stratified(df):
    dev, ev = load_live_claims("dev"), load_live_claims("eval")
    assert len(dev) == 30 and len(dev) + len(ev) == len(df)
    assert not set(dev.id) & set(ev.id)
    assert set(dev.label) == set(VERDICTS)


def test_unique_urls_and_ids(df):
    assert df.factcheck_url.is_unique and df.id.is_unique


def test_validate_rejects_bad_label():
    bad = pd.read_csv(PATH, dtype=str, keep_default_na=False).replace({"": None})
    bad.loc[0, "label"] = "TRUE"
    with pytest.raises(ValueError, match="bad labels"):
        validate(bad)
