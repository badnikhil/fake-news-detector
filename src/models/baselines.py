"""TF-IDF + {Multinomial NB, Logistic Regression, LinearSVC} baselines (master doc §12.1–12.3, MSE1 §3.5).

Design rules
------------
* Text is **never re-split here**: the train/val/test membership comes from ``data/splits/{dataset}_{split}.csv``
  (``id,label``), joined onto the processed parquet by ``id``.  MSE1 evaluates on **val** only — the test split is
  reserved for MSE2 (§18).
* The parquet keeps cased text; lower-casing happens *inside* ``TfidfVectorizer`` (``lowercase=True``), so the data on
  disk stays usable for spaCy NER / claim extraction (§11.1 step 5).
* One TF-IDF matrix is fitted per dataset and shared by the three classifiers ("cache the TF-IDF matrix", §12.3).
  Each saved artefact is nevertheless a self-contained ``Pipeline(tfidf -> clf)`` so that
  ``joblib.load(path).predict([text])`` works in one line.
* The MSE1 configuration is a single point of the §12.2 grid (word 1–2-grams, ``sublinear_tf=True``,
  ``max_features=100k``, LR/SVM ``C=1``, NB ``alpha=1``); the full ``GridSearchCV`` lives in MSE2.

CLI
---
``python -m src.models.baselines --dataset welfake --model lr``  (one model, saves the joblib + a JSON log)
``python -m src.models.baselines --dataset welfake --model all``  (NB, LR, SVM on one TF-IDF matrix)
``make train-baselines``  = ``--dataset welfake --model all`` + ``--dataset isot --model all`` + table render.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.common.seed import set_seed
from src.config import MODELS_DIR, PROCESSED_DIR, RESULTS_DIR, ROOT, SEED, SPLITS_DIR

# ---------------------------------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------------------------------
LABELS = ["real", "fake"]            # fixed order for confusion matrices; "fake" is the positive class
POS_LABEL = "fake"
MODEL_NAMES = ["nb", "lr", "svm"]
TRAIN_LOG_DIR = RESULTS_DIR / "train_logs"
CLASSIFIER_TABLE = RESULTS_DIR / "classifier_table.md"

# MSE1 point of the §12.2 grid (ngram_range ∈ {(1,1),(1,2)}, max_features ∈ {50k,100k,200k}, sublinear_tf ∈ {T,F}).
TFIDF_DEFAULTS = dict(
    ngram_range=(1, 2),
    max_features=100_000,
    sublinear_tf=True,
    lowercase=True,          # lower-case inside the vectoriser only; the parquet stays cased
    min_df=2,                # drop hapax n-grams (mostly typos / names); shrinks the vocabulary ~10x before max_features
    max_df=0.95,
    strip_accents="unicode",
    dtype=np.float32,
)

# Tokens that would indicate that source / boiler-plate artefacts survived preprocessing (MSE1 §3.5 DoD:
# "top coefficients contain no leakage tokens").  Checked as whole tokens *and* as parts of bigrams.
LEAK_TOKENS = {
    "reuters", "featured", "image", "getty", "pic", "twitter", "com", "http", "https", "www",
    "video", "via", "wire", "21st", "century", "21wire", "read", "images", "ap", "afp", "bbc", "cnn",
    # residual publisher / format tokens found in the first MSE1 run (2026-08-28) and stripped since
    "breitbart", "follow", "photo", "screenshot", "screengrab", "york times", "nytimes", "politico",
}


@dataclass
class EvalResult:
    """Everything the notebook / table / tests need from one (dataset, model, split) evaluation."""

    dataset: str
    model: str
    split: str
    n_train: int
    n_eval: int
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_fake: float
    recall_fake: float
    f1_fake: float
    confusion: list[list[int]]          # rows = true (real, fake), cols = pred (real, fake)
    vectoriser_seconds: float
    fit_seconds: float
    predict_seconds: float
    n_features: int
    params: dict = field(default_factory=dict)
    artefact: str | None = None
    variant: str = "v0"                 # "v0" = MSE1 single config, "tuned" = GridSearchCV winner (MSE2)

    @property
    def train_seconds(self) -> float:
        return self.vectoriser_seconds + self.fit_seconds

    def to_dict(self) -> dict:
        d = asdict(self)
        d["train_seconds"] = self.train_seconds
        d["labels"] = LABELS
        return d


# ---------------------------------------------------------------------------------------------------
# data loading (join the saved split IDs — never re-split)
# ---------------------------------------------------------------------------------------------------
def make_input(df: pd.DataFrame) -> pd.Series:
    """Classifier input = title + body (head-truncation of title+text is what DistilBERT will see too, §11.1)."""
    title = df["title"].fillna("").astype(str) if "title" in df.columns else pd.Series("", index=df.index)
    text = df["text"].fillna("").astype(str)
    return (title.str.strip() + "\n" + text).str.strip()


def load_split(dataset: str, split: str, processed_dir: Path = PROCESSED_DIR, splits_dir: Path = SPLITS_DIR,
               df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rows of ``{dataset}.parquet`` whose ``id`` is listed in ``{dataset}_{split}.csv`` (order = CSV order).

    The CSV label must agree with the parquet label for every row (guards against stale splits).
    """
    ids = pd.read_csv(splits_dir / f"{dataset}_{split}.csv")
    if df is None:
        df = pd.read_parquet(processed_dir / f"{dataset}.parquet", columns=["id", "title", "text", "label", "split"])
    out = ids.merge(df, on="id", how="inner", suffixes=("_csv", ""), validate="one_to_one")
    if len(out) != len(ids):
        raise ValueError(f"{dataset}/{split}: {len(ids) - len(out)} split ids missing from the parquet")
    if not (out["label_csv"] == out["label"]).all():
        raise ValueError(f"{dataset}/{split}: split CSV labels disagree with the parquet")
    if not (out["split"] == split).all():
        raise ValueError(f"{dataset}/{split}: parquet `split` column disagrees with the CSV")
    return out.drop(columns=["label_csv", "split"]).reset_index(drop=True)


# ---------------------------------------------------------------------------------------------------
# model factories
# ---------------------------------------------------------------------------------------------------
def build_vectoriser(**overrides) -> TfidfVectorizer:
    return TfidfVectorizer(**{**TFIDF_DEFAULTS, **overrides})


def build_classifier(model: str, **overrides):
    if model == "nb":
        return MultinomialNB(**{"alpha": 1.0, **overrides})
    if model == "lr":
        # liblinear: deterministic, fast for binary L2 on a 100k-feature sparse matrix
        return LogisticRegression(**{"C": 1.0, "solver": "liblinear", "max_iter": 1000, "random_state": SEED, **overrides})
    if model == "svm":
        return LinearSVC(**{"C": 1.0, "random_state": SEED, "max_iter": 5000, **overrides})
    raise ValueError(f"unknown model {model!r}; choose from {MODEL_NAMES}")


def build_pipeline(model: str, tfidf: dict | None = None, clf: dict | None = None) -> Pipeline:
    """Unfitted ``Pipeline([('tfidf', TfidfVectorizer), ('clf', ...)])``."""
    return Pipeline([("tfidf", build_vectoriser(**(tfidf or {}))), ("clf", build_classifier(model, **(clf or {})))])


# ---------------------------------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------------------------------
def compute_metrics(y_true: Iterable[str], y_pred: Iterable[str]) -> dict:
    y_true, y_pred = np.asarray(list(y_true)), np.asarray(list(y_pred))
    p_m, r_m, f_m, _ = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    p_f, r_f, f_f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[POS_LABEL], average="binary",
                                                       pos_label=POS_LABEL, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(p_m), "recall_macro": float(r_m), "f1_macro": float(f_m),
        "precision_fake": float(p_f), "recall_fake": float(r_f), "f1_fake": float(f_f),
        "confusion": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }


# ---------------------------------------------------------------------------------------------------
# training / evaluation
# ---------------------------------------------------------------------------------------------------
def fit_eval_all(dataset: str, models: Iterable[str] = MODEL_NAMES, eval_split: str = "val",
                 tfidf: dict | None = None, clf: dict | None = None, train_df: pd.DataFrame | None = None,
                 eval_df: pd.DataFrame | None = None, verbose: bool = True,
                 ) -> tuple[dict[str, Pipeline], dict[str, EvalResult], dict]:
    """Fit ONE TF-IDF on ``train`` then every requested classifier; evaluate on ``eval_split`` (default val).

    Returns ``(pipelines, results, cache)`` where ``cache`` holds the fitted vectoriser and the transformed
    train/eval matrices (handy for learning curves in the notebook).  ``train_df``/``eval_df`` may be passed to
    reuse already-loaded frames (they must be the split frames from :func:`load_split` or a subsample of them).
    """
    set_seed(SEED)
    if eval_split == "test":
        print("WARNING: evaluating on TEST — reserved for MSE2 final numbers", file=sys.stderr)
    train_df = load_split(dataset, "train") if train_df is None else train_df
    eval_df = load_split(dataset, eval_split) if eval_df is None else eval_df
    X_train_txt, y_train = make_input(train_df), train_df["label"].to_numpy()
    X_eval_txt, y_eval = make_input(eval_df), eval_df["label"].to_numpy()

    vec = build_vectoriser(**(tfidf or {}))
    t0 = time.perf_counter()
    X_train = vec.fit_transform(X_train_txt)
    t_vec = time.perf_counter() - t0
    X_eval = vec.transform(X_eval_txt)
    if verbose:
        print(f"[{dataset}] tfidf fitted on {X_train.shape[0]:,} docs -> {X_train.shape[1]:,} features in {t_vec:.1f}s "
              f"(nnz {X_train.nnz:,})")

    pipelines, results = {}, {}
    for m in models:
        c = build_classifier(m, **(clf or {}))
        t0 = time.perf_counter()
        c.fit(X_train, y_train)
        t_fit = time.perf_counter() - t0
        t0 = time.perf_counter()
        y_pred = c.predict(X_eval)
        t_pred = time.perf_counter() - t0
        met = compute_metrics(y_eval, y_pred)
        res = EvalResult(dataset=dataset, model=m, split=eval_split, n_train=int(X_train.shape[0]),
                         n_eval=int(X_eval.shape[0]), vectoriser_seconds=t_vec, fit_seconds=t_fit,
                         predict_seconds=t_pred, n_features=int(X_train.shape[1]),
                         params={"tfidf": {k: (list(v) if isinstance(v, tuple) else v) for k, v in
                                           {**TFIDF_DEFAULTS, **(tfidf or {})}.items() if k != "dtype"},
                                 "clf": {k: v for k, v in c.get_params().items()
                                         if k in ("C", "alpha", "solver", "max_iter", "loss")}},
                         **met)
        pipelines[m] = Pipeline([("tfidf", vec), ("clf", c)])
        results[m] = res
        if verbose:
            print(f"[{dataset}] {m:>3} {eval_split}: acc {res.accuracy:.4f}  P {res.precision_macro:.4f}  "
                  f"R {res.recall_macro:.4f}  macro-F1 {res.f1_macro:.4f}  fit {t_fit:.1f}s  cm {res.confusion}")
    cache = {"vectoriser": vec, "X_train": X_train, "y_train": y_train, "X_eval": X_eval, "y_eval": y_eval,
             "train_ids": train_df["id"].to_numpy(), "eval_ids": eval_df["id"].to_numpy()}
    return pipelines, results, cache


def fit_eval(dataset: str, model: str = "lr", eval_split: str = "val", **kw) -> tuple[Pipeline, EvalResult]:
    """Train one TF-IDF + ``model`` pipeline on ``{dataset}`` train and evaluate on ``eval_split`` (val by default).

    Returns ``(fitted_pipeline, EvalResult)`` — accuracy / precision / recall / macro-F1 / confusion matrix.
    """
    pipelines, results, _ = fit_eval_all(dataset, [model], eval_split=eval_split, **kw)
    return pipelines[model], results[model]


# ---------------------------------------------------------------------------------------------------
# interpretation helpers
# ---------------------------------------------------------------------------------------------------
def top_coefficients(pipeline: Pipeline, k: int = 25) -> pd.DataFrame:
    """Top-k features pushing towards ``fake`` (positive) and towards ``real`` (negative) for a linear model."""
    vec, clf = pipeline.named_steps["tfidf"], pipeline.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        raise TypeError("top_coefficients needs a linear classifier with coef_ (lr / svm)")
    coef = np.asarray(clf.coef_).ravel()
    # sklearn orders classes_ alphabetically -> ["fake", "real"]; coef_ > 0 pushes towards classes_[1] == "real"
    if list(clf.classes_) == ["fake", "real"]:
        coef = -coef                       # flip so that positive == fake
    names = np.asarray(vec.get_feature_names_out())
    pos, neg = np.argsort(-coef)[:k], np.argsort(coef)[:k]
    return pd.DataFrame({"rank": range(1, k + 1),
                         "fake_token": names[pos], "fake_coef": coef[pos].round(3),
                         "real_token": names[neg], "real_coef": coef[neg].round(3)})


def find_leak_tokens(tokens: Iterable[str], leak: set[str] = LEAK_TOKENS) -> list[str]:
    """Return the tokens (uni- or bi-grams) that contain any leakage word; empty list == clean."""
    hits = []
    for t in tokens:
        parts = re.split(r"\s+", str(t).lower().strip())
        if any(p in leak for p in parts):
            hits.append(t)
    return hits


def demo_headlines() -> list[tuple[str, str]]:
    """Three hand-written headlines used by the notebook / tests to show the one-line reload + predict."""
    return [
        ("obviously fake", "BREAKING: Scientists confirm the Moon is hollow and NASA has been hiding it for decades!!!"),
        ("plausibly real", "The central bank held its benchmark interest rate steady on Thursday, citing slowing inflation."),
        ("satire-ish", "Local man declares himself emperor of his apartment complex after winning HOA vote by one ballot"),
    ]


# ---------------------------------------------------------------------------------------------------
# MSE2 hyperparameter tuning: GridSearchCV, 5-fold on WELFake train, scoring macro-F1 (MSE2 §4)
# ---------------------------------------------------------------------------------------------------
CV_FOLDS = 5
TFIDF_GRID = {"ngram_range": [(1, 1), (1, 2)], "max_features": [50_000, 100_000, 200_000], "sublinear_tf": [True, False]}
CLF_GRID = {"nb": {"alpha": [0.1, 0.5, 1.0]}, "lr": {"C": [0.1, 1.0, 10.0]}, "svm": {"C": [0.1, 1.0, 10.0]}}
# NB grid = 18 rows (MSE2 §4 DoD): sublinear_tf is fixed at the MSE1 value instead of searched.
NB_FIXED_TFIDF = {"sublinear_tf": [True]}


def gridsearch_csv(model: str) -> Path:
    return RESULTS_DIR / f"gridsearch_{model}.csv"


def param_grid(model: str) -> list[dict]:
    """GridSearchCV grid as a list of dicts, one per TF-IDF combination (so candidates sharing a TF-IDF config are
    consecutive — that is what lets :class:`FoldCachedTfidf` fit each (config, fold) only once)."""
    tf = {**TFIDF_GRID, **(NB_FIXED_TFIDF if model == "nb" else {})}
    clf = {f"clf__{k}": v for k, v in CLF_GRID[model].items()}
    return [{**{f"tfidf__{k}": [v] for k, v in combo.items()}, **clf} for combo in ParameterGrid(tf)]


class FoldCachedTfidf(TfidfVectorizer):
    """``TfidfVectorizer`` that memoises ``fit_transform`` / ``transform`` per (params, rows) — GridSearchCV only.

    GridSearchCV refits the vectoriser for every candidate × fold although only the classifier's C/alpha changed.
    Results are identical to a plain ``TfidfVectorizer`` (``tests/test_models.py`` checks it); only time is saved.
    ponytail: in-process cache, cleared whenever the TF-IDF params change -> needs n_jobs=1 and the grouped
    :func:`param_grid` order; a joblib disk cache would allow n_jobs>1 but costs ~6 GB of disk.
    """

    _cache: dict = {}

    def _pkey(self) -> str:
        return repr(sorted((k, v) for k, v in self.get_params().items() if k != "dtype"))

    def fit_transform(self, raw_documents, y=None, **kw):
        c = FoldCachedTfidf._cache
        if c.get("params") != self._pkey():
            c.clear()
            c["params"] = self._pkey()
        key = ("fit", hash(tuple(raw_documents.index)))
        if key in c:
            self.__dict__.update(c[key][0])
        else:
            Xt = super().fit_transform(raw_documents)
            c[key] = (dict(self.__dict__), Xt)
        self._fold_key = key[1]
        return c[key][1]

    def fit(self, raw_documents, y=None):
        self.fit_transform(raw_documents)
        return self

    def transform(self, raw_documents):
        c = FoldCachedTfidf._cache
        key = ("transform", getattr(self, "_fold_key", None), hash(tuple(raw_documents.index)))
        if c.get("params") != self._pkey() or key[1] is None:
            return super().transform(raw_documents)
        if key not in c:
            c[key] = super().transform(raw_documents)
        return c[key]


def grid_search(model: str, dataset: str = "welfake", train_df: pd.DataFrame | None = None, n_jobs: int = 1,
                verbose: int = 1, out: Path | None = None) -> pd.DataFrame:
    """Run GridSearchCV (5-fold stratified, seed 42, macro-F1) on ``{dataset}`` train; write the raw ``cv_results_``
    to ``docs/results/gridsearch_{model}.csv`` (+ a ``best`` flag column) and a JSON log; return the frame."""
    set_seed(SEED)
    FoldCachedTfidf._cache.clear()
    train_df = load_split(dataset, "train") if train_df is None else train_df
    X, y = make_input(train_df), train_df["label"].to_numpy()
    pipe = Pipeline([("tfidf", FoldCachedTfidf(**TFIDF_DEFAULTS)), ("clf", build_classifier(model))])
    gs = GridSearchCV(pipe, param_grid(model), scoring="f1_macro", n_jobs=n_jobs, refit=False, verbose=verbose,
                      cv=StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED), error_score="raise")
    t0 = time.perf_counter()
    gs.fit(X, y)
    wall = time.perf_counter() - t0
    FoldCachedTfidf._cache.clear()
    res = pd.DataFrame(gs.cv_results_)
    res.insert(0, "model", model)
    res["best"] = res["rank_test_score"] == 1
    out = out or gridsearch_csv(model)
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
    log = {"model": model, "dataset": dataset, "n_train": int(len(X)), "cv": f"StratifiedKFold({CV_FOLDS}, shuffle, seed {SEED})",
           "scoring": "f1_macro", "n_candidates": int(len(res)), "n_jobs": n_jobs, "wall_seconds": round(wall, 1),
           "best_params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in gs.best_params_.items()},
           "best_cv_f1_macro": float(gs.best_score_), "fixed_tfidf": {k: v for k, v in TFIDF_DEFAULTS.items() if k != "dtype"
                                                                      and k not in TFIDF_GRID},
           "csv": str(out.relative_to(ROOT)) if out.is_relative_to(ROOT) else str(out)}
    TRAIN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    (TRAIN_LOG_DIR / f"gridsearch_{model}.json").write_text(json.dumps(log, indent=2, default=str))
    if verbose:
        print(f"[gridsearch {model}] {len(res)} candidates x {CV_FOLDS} folds in {wall/60:.1f} min; best "
              f"{gs.best_score_:.4f} {gs.best_params_}")
    return res


def best_params(model: str, csv: Path | None = None) -> tuple[dict, dict] | None:
    """(tfidf_overrides, clf_overrides) of the ``best`` row in ``gridsearch_{model}.csv``; None if no CSV yet."""
    import ast
    csv = csv or gridsearch_csv(model)
    if not Path(csv).exists():
        return None
    df = pd.read_csv(csv)
    row = df[df["best"]].sort_values("mean_test_score", ascending=False).iloc[0]
    params = ast.literal_eval(row["params"])
    tf = {k.split("__", 1)[1]: v for k, v in params.items() if k.startswith("tfidf__")}
    clf = {k.split("__", 1)[1]: v for k, v in params.items() if k.startswith("clf__")}
    return tf, clf


def tuned_artefact_path(dataset: str, model: str) -> Path:
    """Canonical MSE2 names: ``tfidf_{nb,lr,svm}.joblib`` for WELFake (master §24), ``tfidf_{m}_isot.joblib`` for ISOT."""
    return MODELS_DIR / (f"tfidf_{model}.joblib" if dataset == "welfake" else f"tfidf_{model}_{dataset}.joblib")


def train_tuned(dataset: str, models: Iterable[str] = MODEL_NAMES, save: bool = True) -> dict[str, EvalResult]:
    """Fit each model with its WELFake GridSearchCV winner on ``{dataset}`` train (val metrics), save + log it.

    Falls back to the MSE1 defaults (and says so) when ``gridsearch_{model}.csv`` does not exist yet.
    """
    out = {}
    for m in models:
        bp = best_params(m)
        if bp is None:
            print(f"[{dataset}] {m}: no {gridsearch_csv(m).name} -> MSE1 defaults (run `make tune-baselines`)")
        tf, clf = bp or ({}, {})
        pipes, results, _ = fit_eval_all(dataset, [m], tfidf=tf, clf=clf)
        res = results[m]
        res.variant = "tuned" if bp else "v0-defaults"
        if save:
            saved = save_pipeline(pipes[m], tuned_artefact_path(dataset, m))
            res.artefact = str(saved.relative_to(ROOT)) if saved.is_relative_to(ROOT) else str(saved)
            print(f"saved {res.artefact} ({saved.stat().st_size / 2**20:.1f} MB)")
        print("log  ", save_log(res))
        out[m] = res
    return out


# ---------------------------------------------------------------------------------------------------
# artefacts
# ---------------------------------------------------------------------------------------------------
def default_artefact_path(dataset: str, model: str) -> Path:
    """Canonical MSE1 names: ``tfidf_lr_welfake_v0.joblib`` (master §24); same pattern for the others."""
    return MODELS_DIR / f"tfidf_{model}_{dataset}_v0.joblib"


def save_pipeline(pipeline: Pipeline, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path, compress=3)
    return path


def load_pipeline(path: Path = MODELS_DIR / "tfidf_lr_welfake_v0.joblib") -> Pipeline:
    return joblib.load(path)


def save_log(result: EvalResult, log_dir: Path = TRAIN_LOG_DIR) -> Path:
    """Write ``docs/results/train_logs/baselines_{dataset}_{model}_{split}.json``.

    If the new result carries no ``artefact`` but an earlier log (e.g. from the CLI) recorded one that still exists,
    the artefact name is kept so that a notebook re-run does not blank the column in ``classifier_table.md``.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    tag = "" if result.variant == "v0" else f"_{result.variant}"
    p = log_dir / f"baselines_{result.dataset}_{result.model}{tag}_{result.split}.json"
    if result.artefact is None and p.exists():
        try:
            prev = json.loads(p.read_text()).get("artefact")
            if prev and (ROOT / prev).exists():
                result.artefact = prev
        except (json.JSONDecodeError, OSError):
            pass
    p.write_text(json.dumps(result.to_dict(), indent=2))
    return p


# ---------------------------------------------------------------------------------------------------
# results table (docs/results/classifier_table.md) — rendered from the JSON logs so it never drifts
# ---------------------------------------------------------------------------------------------------
MODEL_LABEL = {"nb": "TF-IDF + Multinomial NB", "lr": "TF-IDF + Logistic Regression", "svm": "TF-IDF + LinearSVC"}

PENDING_ROWS = [
    # (model, dataset, split, notes) — rows no MSE2 eval log covers yet
    ("Bi-LSTM + GloVe 100d (optional)", "WELFake", "val / test", "MSE2 — only if time permits (week 9)"),
    ("DistilBERT (or TF-IDF + LR) LIAR 3-way head", "LIAR", "test", "MSE2 — `docs/results/liar_head.md`"),
]
# Rendered while DistilBERT is pending. Expected values only (own measured numbers as basis), never measured.
DISTILBERT_EXPECTED = [
    "",
    "### DistilBERT — expected (not measured), deferred to ESE",
    "",
    "| Model | Dataset | Expected test Macro-F1 † — not measured | Basis |",
    "|---|---|---|---|",
    "| DistilBERT (`distilbert-welfake-v1`) | WELFake | *≈ 0.98–0.99* † (O1 ≥ 0.95 *expected* met †; not checked) "
    "| 20k-subset val macro-F1 0.9863–0.9871 (`distilbert_sweep.md`); WELFake val–test gaps of the tuned classical models "
    "within ±0.003 (table above) |",
    "| DistilBERT (`distilbert-welfake-v1`) | ISOT (in-domain) | *expected high* † (no range) "
    "| ISOT ⊂ WELFake; ~80 % of ISOT test rows are in WELFake train |",
    "| DistilBERT (`distilbert-welfake-v1`) | WELFake∖ISOT ↔ ISOT (cross-dataset) | *expected to drop sharply, like the "
    "classical models* † (direction only) | measured SVM test macro-F1 0.6611 W∖I→ISOT, 0.5480 ISOT→W∖I (`cross_dataset.md`) |",
    "",
    "† expected, not measured; to be replaced by the ESE run. These cells are not results.",
]
DISTILBERT_PENDING = ("DistilBERT-base-uncased fine-tuned (`distilbert-welfake-v1`)", "WELFake / ISOT", "val + test",
                      "filled automatically by `make eval-models` once `data/models/distilbert-welfake-v1/` exists "
                      "(`make train-distilbert`); sweep: `docs/results/distilbert_sweep.md`")


def _fmt_cm(cm: list[list[int]]) -> str:
    return f"[[{cm[0][0]}, {cm[0][1]}], [{cm[1][0]}, {cm[1][1]}]]"


def _mse2_section(log_dir: Path) -> list[str]:
    """MSE2 rows from ``eval_{dataset}_{model}.json`` (written by ``make eval-models``): val + test, ROC-AUC."""
    recs = [json.loads(p.read_text()) for p in sorted(log_dir.glob("eval_*.json"))]
    if not recs:
        return ["", "## MSE2 — tuned models, val + test", "", "_No eval logs yet — run `make train-baselines && make eval-models`._"]
    order = {"welfake": 0, "isot": 1}
    morder = {"nb": 0, "lr": 1, "svm": 2, "distilbert": 3}
    recs.sort(key=lambda r: (order.get(r["dataset"], 9), morder.get(r["model"], 9)))
    label = {**MODEL_LABEL, "distilbert": "DistilBERT-base-uncased fine-tuned"}
    f = lambda v: "—" if v is None else f"{v:.4f}"  # noqa: E731
    lines = ["", "## MSE2 — tuned models (GridSearchCV winners / DistilBERT), val + **test**", "",
             "Generated by `make eval-models` (`src/eval/eval_models.py`) from `docs/results/train_logs/eval_*.json`. "
             "TF-IDF models use the WELFake GridSearchCV winner (`docs/results/gridsearch_{nb,lr,svm}.csv`, 5-fold CV on train, macro-F1), "
             "refitted on the dataset's full train split. ROC-AUC: NB/LR `predict_proba`, LinearSVC `decision_function`, DistilBERT softmax. "
             "Gap = val macro-F1 − test macro-F1 (over-fitting check).", "",
             "| Model | Dataset | Hyper-parameters | val Macro-F1 | test Acc | test P (macro) | test R (macro) | **test Macro-F1** | test ROC-AUC | P/R (fake) | Confusion [[TN, FP],[FN, TP]] | Gap |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in recs:
        t, v = r["test"], r.get("val") or {}
        hp = r.get("params", {})
        if r["model"] == "distilbert":
            hps = ", ".join(f"{k}={hp[k]}" for k in ("lr", "max_len", "epochs") if k in hp)
        else:
            tf, clf = hp.get("tfidf", {}), hp.get("clf", {})
            hps = (f"ngram {tuple(tf.get('ngram_range', []))}, max_features {tf.get('max_features', 0):,}, "
                   f"sublinear_tf={tf.get('sublinear_tf')}; " + ", ".join(f"{k}={x}" for k, x in clf.items()))
        gap = f"{v['f1_macro'] - t['f1_macro']:+.4f}" if v.get("f1_macro") is not None else "—"
        ds = "WELFake" if r["dataset"] == "welfake" else "ISOT (in-domain)"
        lines.append(f"| {label.get(r['model'], r['model'])} | {ds} | {hps} | {f(v.get('f1_macro'))} | {f(t['accuracy'])} | "
                     f"{f(t['precision_macro'])} | {f(t['recall_macro'])} | **{f(t['f1_macro'])}** | {f(t.get('roc_auc'))} | "
                     f"{t['precision_fake']:.3f} / {t['recall_fake']:.3f} | {_fmt_cm(t['confusion'])} | {gap} |")
    return lines


def render_classifier_table(log_dir: Path = TRAIN_LOG_DIR, out: Path = CLASSIFIER_TABLE) -> Path:
    """Write ``docs/results/classifier_table.md`` from every ``baselines_*.json`` log + the MSE2 pending rows."""
    logs = sorted(log_dir.glob("baselines_*.json"))
    rows = [r for r in (json.loads(p.read_text()) for p in logs) if r.get("variant", "v0") == "v0"]
    order = {"welfake": 0, "isot": 1, "liar": 2}
    morder = {"nb": 0, "lr": 1, "svm": 2}
    rows.sort(key=lambda r: (order.get(r["dataset"], 9), r["split"] != "val", morder.get(r["model"], 9)))
    lines = [
        "# Classifier results table (`docs/results/classifier_table.md`)",
        "",
        "Generated by `python -m src.models.baselines --render-table` (part of `make train-baselines`) from the JSON logs in "
        "`docs/results/train_logs/`. **MSE1 rows** are TF-IDF baselines evaluated on the **validation** split "
        "(the test split is untouched until MSE2). Macro-F1 is the selection metric (master §12.2 / §18). "
        "Confusion matrix rows = true (real, fake), columns = predicted (real, fake). Train time = TF-IDF fit + classifier fit "
        "on the laptop CPU (8 threads); the TF-IDF matrix is shared by the three classifiers of one dataset.",
        "",
        "## MSE1 — TF-IDF baselines (val split)",
        "",
        "| Model | Dataset | Split | n_train / n_eval | Acc | P (macro) | R (macro) | **Macro-F1** | P/R (fake) | Confusion [[TN, FP],[FN, TP]] | Train time (s) | Notes |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        tf = r["params"]["tfidf"]
        clf = r["params"]["clf"]
        hp = ", ".join(f"{k}={v}" for k, v in clf.items() if k in ("C", "alpha"))
        note = (f"word {tuple(tf['ngram_range'])}-grams, max_features={tf['max_features']:,}, sublinear_tf, "
                f"min_df={tf['min_df']}; {hp}; {r['n_features']:,} features; artefact `{Path(r['artefact']).name if r.get('artefact') else '—'}`")
        lines.append(
            f"| {MODEL_LABEL.get(r['model'], r['model'])} | {r['dataset'].upper() if r['dataset']=='isot' else r['dataset'].capitalize().replace('Welfake','WELFake')} "
            f"| {r['split']} | {r['n_train']:,} / {r['n_eval']:,} | {r['accuracy']:.4f} | {r['precision_macro']:.4f} | {r['recall_macro']:.4f} "
            f"| **{r['f1_macro']:.4f}** | {r['precision_fake']:.3f} / {r['recall_fake']:.3f} | {_fmt_cm(r['confusion'])} "
            f"| {r['train_seconds']:.1f} (tfidf {r['vectoriser_seconds']:.1f} + fit {r['fit_seconds']:.1f}) | {note} |")
    lines += _mse2_section(log_dir)
    lines += [
        "",
        "## Still pending",
        "",
        "| Model | Dataset | Split | Status / notes |",
        "|---|---|---|---|",
    ]
    pending = list(PENDING_ROWS)
    if not (log_dir / "eval_welfake_distilbert.json").exists():
        pending.insert(0, DISTILBERT_PENDING)
    for model, ds, split, note in pending:
        lines.append(f"| {model} | {ds} | {split} | **pending** — {note} |")
    if not (log_dir / "eval_welfake_distilbert.json").exists():
        lines += DISTILBERT_EXPECTED
    lines += [
        "",
        "Cross-dataset (WELFake∖ISOT ↔ ISOT, with / without artefact removal): `docs/results/cross_dataset.md`. "
        "ISOT rows above are **in-domain** (ISOT ⊂ WELFake, `docs/model_identification.md` §4.2) — never a cross-dataset number.",
        "Sanity bar (MSE1 §3.5 DoD): LR macro-F1 on WELFake val ≥ 0.90. Target O1: DistilBERT macro-F1 ≥ 0.95 on WELFake **test**.",
        "",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    return out


# ---------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="TF-IDF + NB/LR/SVM baselines (train on train, evaluate on val)")
    ap.add_argument("--dataset", default="welfake", choices=["welfake", "isot", "liar"])
    ap.add_argument("--model", default="lr", choices=MODEL_NAMES + ["all"])
    ap.add_argument("--split", default="val", choices=["val", "test"], help="evaluation split (test = MSE2 only)")
    ap.add_argument("--max-features", type=int, default=TFIDF_DEFAULTS["max_features"])
    ap.add_argument("--out", type=Path, default=None, help="joblib path (single model only); default = canonical name")
    ap.add_argument("--no-save", action="store_true", help="do not write joblib artefacts")
    ap.add_argument("--render-table", action="store_true", help="only (re)write docs/results/classifier_table.md from the logs")
    ap.add_argument("--grid-search", action="store_true",
                    help="MSE2: GridSearchCV (5-fold on train, macro-F1) -> docs/results/gridsearch_{model}.csv")
    ap.add_argument("--tuned", action="store_true",
                    help="MSE2: fit the GridSearchCV winner(s) on train -> data/models/tfidf_{model}.joblib (+ val log)")
    args = ap.parse_args(argv)
    models = MODEL_NAMES if args.model == "all" else [args.model]

    if args.render_table:
        print("wrote", render_classifier_table())
        return 0
    if args.grid_search:
        for m in models:
            grid_search(m, args.dataset, verbose=3)
        return 0
    if args.tuned:
        train_tuned(args.dataset, models, save=not args.no_save)
        print("wrote", render_classifier_table())
        return 0

    t0 = time.perf_counter()
    pipelines, results, _ = fit_eval_all(args.dataset, models, eval_split=args.split,
                                         tfidf={"max_features": args.max_features})
    for m in models:
        if not args.no_save:
            path = args.out if (args.out and len(models) == 1) else default_artefact_path(args.dataset, m)
            saved = save_pipeline(pipelines[m], path)
            try:
                results[m].artefact = str(saved.relative_to(ROOT))
            except ValueError:
                results[m].artefact = str(saved)
            print(f"saved {results[m].artefact} ({saved.stat().st_size / 2**20:.1f} MB)")
        print("log  ", save_log(results[m]))
    print("wrote", render_classifier_table())
    print(f"total {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
