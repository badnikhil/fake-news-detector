"""Light MSE1 checks for the TF-IDF baselines (`src/models/baselines.py`).

Fast by design: a 2k-row WELFake sample trains in a few seconds; tests skip when `make data` has not been run.
The saved artefact test only runs when `data/models/tfidf_lr_welfake_v0.joblib` exists (produced by
`make train-baselines` or `notebooks/02_baselines.ipynb`).
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

from src.config import MODELS_DIR, PROCESSED_DIR, SPLITS_DIR
from src.models import baselines as B

pytestmark = pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")


def _need_data():
    if not (PROCESSED_DIR / "welfake.parquet").exists() or not (SPLITS_DIR / "welfake_val.csv").exists():
        pytest.skip("processed WELFake missing - run `make data`")


@pytest.fixture(scope="module")
def welfake_sample():
    _need_data()
    train = B.load_split("welfake", "train").sample(2_000, random_state=42).reset_index(drop=True)
    val = B.load_split("welfake", "val").sample(500, random_state=42).reset_index(drop=True)
    return train, val


# ---------------------------------------------------------------------------------------------------
def test_load_split_uses_saved_ids_and_labels():
    _need_data()
    val = B.load_split("welfake", "val")
    ids = pd.read_csv(SPLITS_DIR / "welfake_val.csv")
    assert len(val) == len(ids)
    assert list(val["id"]) == list(ids["id"])                 # CSV order preserved, nothing re-split
    assert set(val["label"]) <= {"real", "fake"}
    assert val["id"].is_unique


def test_make_input_prepends_title_and_keeps_case():
    df = pd.DataFrame({"title": ["Hello World", None], "text": ["Body TEXT.", "only body"]})
    out = B.make_input(df)
    assert out.iloc[0] == "Hello World\nBody TEXT."
    assert out.iloc[1] == "only body"


@pytest.mark.parametrize("model", B.MODEL_NAMES)
def test_pipeline_trains_and_predicts_on_2k_sample(welfake_sample, model):
    train, val = welfake_sample
    pipelines, results, cache = B.fit_eval_all("welfake", [model], train_df=train, eval_df=val, verbose=False)
    res = results[model]
    assert res.n_train == 2_000 and res.n_eval == 500
    assert 0.5 < res.accuracy <= 1.0 and 0.5 < res.f1_macro <= 1.0     # far above chance even on 2k rows
    cm = np.asarray(res.confusion)
    assert cm.shape == (2, 2) and cm.sum() == 500
    pipe = pipelines[model]
    preds = pipe.predict(["Some text that is clearly an article.", "Another article body about markets."])
    assert set(preds) <= {"real", "fake"}
    assert pipe.named_steps["tfidf"].lowercase is True
    assert pipe.named_steps["tfidf"].ngram_range == (1, 2)
    assert pipe.named_steps["tfidf"].sublinear_tf is True
    assert cache["X_train"].shape[0] == 2_000


def test_top_coefficients_and_leak_check(welfake_sample):
    train, val = welfake_sample
    pipelines, _, _ = B.fit_eval_all("welfake", ["lr"], train_df=train, eval_df=val, verbose=False)
    tab = B.top_coefficients(pipelines["lr"], k=10)
    assert list(tab.columns) == ["rank", "fake_token", "fake_coef", "real_token", "real_coef"]
    assert len(tab) == 10
    assert (tab["fake_coef"] > 0).all() and (tab["real_coef"] < 0).all()
    assert B.find_leak_tokens(["trump", "said on", "washington"]) == []
    assert B.find_leak_tokens(["reuters", "featured image", "pic twitter"]) == ["reuters", "featured image", "pic twitter"]


def test_joblib_roundtrip(tmp_path, welfake_sample):
    train, val = welfake_sample
    pipelines, _, _ = B.fit_eval_all("welfake", ["lr"], train_df=train, eval_df=val, verbose=False)
    p = B.save_pipeline(pipelines["lr"], tmp_path / "lr.joblib")
    reloaded = joblib.load(p)
    texts = [h for _, h in B.demo_headlines()]
    assert list(reloaded.predict(texts)) == list(pipelines["lr"].predict(texts))
    assert reloaded.predict_proba(texts).shape == (3, 2)


def test_compute_metrics_perfect_and_confusion_orientation():
    m = B.compute_metrics(["real", "fake", "fake"], ["real", "fake", "real"])
    assert m["confusion"] == [[1, 0], [1, 1]]           # rows = true (real, fake); cols = pred (real, fake)
    assert m["accuracy"] == pytest.approx(2 / 3)
    assert m["recall_fake"] == pytest.approx(0.5) and m["precision_fake"] == pytest.approx(1.0)


def test_saved_mse1_artefact_reloads_in_one_line():
    path = MODELS_DIR / "tfidf_lr_welfake_v0.joblib"
    if not path.exists():
        pytest.skip(f"{path} missing - run `make train-baselines`")
    clf = joblib.load(path)
    preds = clf.predict([h for _, h in B.demo_headlines()])
    assert len(preds) == 3 and set(preds) <= {"real", "fake"}
    assert clf.named_steps["tfidf"].max_features == B.TFIDF_DEFAULTS["max_features"]
