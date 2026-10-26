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


# ---------------------------------------------------------------------------------------------------
# MSE2: GridSearchCV grid + fold-cached TF-IDF
# ---------------------------------------------------------------------------------------------------
def test_param_grid_sizes_match_mse2_spec():
    from sklearn.model_selection import ParameterGrid
    assert {m: len(ParameterGrid(B.param_grid(m))) for m in B.MODEL_NAMES} == {"nb": 18, "lr": 36, "svm": 36}


def test_fold_cached_tfidf_gives_identical_cv_scores(welfake_sample):
    """The cache only saves time: cv_results_ must equal a plain-TfidfVectorizer GridSearchCV."""
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.pipeline import Pipeline
    train, _ = welfake_sample
    X, y = B.make_input(train.iloc[:600]), train["label"].iloc[:600].to_numpy()
    grid = [{"tfidf__ngram_range": [(1, 1)], "tfidf__max_features": [2_000], "clf__C": [0.1, 1.0]},
            {"tfidf__ngram_range": [(1, 2)], "tfidf__max_features": [5_000], "clf__C": [0.1, 1.0]}]
    scores = []
    for vec in (B.FoldCachedTfidf(**B.TFIDF_DEFAULTS), B.build_vectoriser()):
        B.FoldCachedTfidf._cache.clear()
        gs = GridSearchCV(Pipeline([("tfidf", vec), ("clf", B.build_classifier("svm"))]), grid, scoring="f1_macro",
                          cv=StratifiedKFold(3, shuffle=True, random_state=42), refit=False).fit(X, y)
        scores.append(gs.cv_results_["mean_test_score"])
    B.FoldCachedTfidf._cache.clear()
    np.testing.assert_allclose(scores[0], scores[1])


def test_best_params_parses_csv(tmp_path):
    csv = tmp_path / "g.csv"
    pd.DataFrame({"params": [str({"clf__C": 1.0, "tfidf__ngram_range": (1, 2)}), str({"clf__C": 10.0, "tfidf__ngram_range": (1, 1)})],
                  "mean_test_score": [0.9, 0.8], "best": [True, False]}).to_csv(csv, index=False)
    assert B.best_params("svm", csv) == ({"ngram_range": (1, 2)}, {"C": 1.0})
    assert B.best_params("svm", tmp_path / "missing.csv") is None


def test_welfake_minus_isot_removes_exact_and_near_copies():
    from src.eval.cross_dataset import welfake_minus_isot
    base = " ".join(f"word{i}" for i in range(300))
    isot = pd.DataFrame({"id": ["i1", "i2"], "text": [base, "completely different isot article " * 10],
                         "label": ["real", "fake"], "split": ["train", "test"]})
    welfake = pd.DataFrame({"id": ["w1", "w2", "w3"],
                            "text": [base.upper() + " !!", base + " word300", "an unrelated welfake story " * 10],
                            "label": ["real", "real", "fake"], "split": ["train", "val", "test"]})
    rows, stats = welfake_minus_isot(welfake, isot)
    assert stats["welfake_in_isot"] == 1 and stats["welfake_minus_isot_exact"] == 2   # w1 = case/punct variant of i1
    assert stats["near_dups_of_isot_removed"] == 1                                     # w2 = i1 + one word
    assert list(rows["id"]) == ["w3"] and list(rows.columns) == ["id", "label", "split"]


def test_fake_scores_orientation_for_svm_and_lr(welfake_sample):
    """Higher score must mean 'more fake' for both predict_proba (LR) and decision_function (LinearSVC)."""
    from src.eval.eval_models import fake_scores
    train, val = welfake_sample
    for m in ("lr", "svm"):
        pipes, _, _ = B.fit_eval_all("welfake", [m], train_df=train, eval_df=val, verbose=False)
        pred, score = fake_scores(pipes[m], B.make_input(val))
        assert score[pred == "fake"].min() >= score[pred == "real"].max()
