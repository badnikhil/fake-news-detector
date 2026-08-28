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
    p = log_dir / f"baselines_{result.dataset}_{result.model}_{result.split}.json"
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
    # (model, dataset, split, notes)
    ("TF-IDF + NB / LR / SVM (GridSearchCV best)", "WELFake", "test", "MSE2 — `make train-baselines` full grid, `docs/results/gridsearch_{nb,lr,svm}.csv`"),
    ("Bi-LSTM + GloVe 100d (optional)", "WELFake", "val / test", "MSE2 — only if time permits (week 9)"),
    ("DistilBERT-base-uncased fine-tuned (best of 6-config sweep)", "WELFake", "val", "MSE2 — `docs/results/distilbert_sweep.md`"),
    ("DistilBERT-base-uncased fine-tuned", "WELFake", "test", "MSE2 — final number; ROC-AUC + `docs/figures/cm_distilbert_welfake.png`"),
    ("Best classical + DistilBERT", "ISOT", "test", "MSE2 — in-domain ISOT (leakage story)"),
    ("Cross-dataset: WELFake∖ISOT ↔ ISOT (redesigned)", "WELFake∖ISOT ↔ ISOT", "test", "MSE2 — `docs/results/cross_dataset.md`, with / without artefact removal. **Protocol (master §18):** ≈ 99.9 % of processed ISOT texts occur verbatim in WELFake (MSE1 finding, `docs/model_identification.md` §4.2), so WELFake → ISOT is in-domain; train on the hash-join-defined WELFake∖ISOT subset (≈ 23.3k rows, near-duplicates removed) as the second domain"),
    ("DistilBERT (or TF-IDF + LR) LIAR 3-way head", "LIAR", "test", "MSE2 — `docs/results/liar_head.md`"),
]


def _fmt_cm(cm: list[list[int]]) -> str:
    return f"[[{cm[0][0]}, {cm[0][1]}], [{cm[1][0]}, {cm[1][1]}]]"


def render_classifier_table(log_dir: Path = TRAIN_LOG_DIR, out: Path = CLASSIFIER_TABLE) -> Path:
    """Write ``docs/results/classifier_table.md`` from every ``baselines_*.json`` log + the MSE2 pending rows."""
    logs = sorted(log_dir.glob("baselines_*.json"))
    rows = [json.loads(p.read_text()) for p in logs]
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
    lines += [
        "",
        "## MSE2 — pending rows (to be filled by `make train-distilbert` / `make eval-models`)",
        "",
        "| Model | Dataset | Split | Acc | P | R | Macro-F1 | ROC-AUC | Train time | Status / notes |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for model, ds, split, note in PENDING_ROWS:
        lines.append(f"| {model} | {ds} | {split} | — | — | — | — | — | — | **pending (MSE2)** — {note} |")
    lines += [
        "",
        "Sanity bar (MSE1 §3.5 DoD): LR macro-F1 on WELFake val ≥ 0.90. Target O1 (≥ 0.95 macro-F1) applies to the final model on **test** at MSE2.",
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
    args = ap.parse_args(argv)

    if args.render_table:
        print("wrote", render_classifier_table())
        return 0

    models = MODEL_NAMES if args.model == "all" else [args.model]
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
