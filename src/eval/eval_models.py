"""``make eval-models`` — MSE2 result analysis for the content classifiers (MSE2 §3–5).

1. **Test (+ val) metrics** for the tuned TF-IDF NB / LR / SVM (``data/models/tfidf_{m}.joblib`` WELFake,
   ``tfidf_{m}_isot.joblib`` ISOT; written by ``make train-baselines``): accuracy, macro P/R/F1, fake-class P/R,
   confusion, ROC-AUC (NB/LR from ``predict_proba``, LinearSVC from ``decision_function`` margins) ->
   ``docs/results/train_logs/eval_{dataset}_{model}.json``.
2. **DistilBERT** (only if ``data/models/distilbert-welfake-v1/`` exists): uses the GPU agent's
   ``distilbert_welfake_v1_test.json`` + test prediction CSVs (runs ``src.models.distilbert.run_eval_test`` first if
   they are missing) -> ``eval_{dataset}_distilbert.json``.  Skipped with a message otherwise.
3. Figures: ``docs/figures/cm_{model}_{dataset}.png`` (counts + row %), ``roc_{dataset}.png`` (all models),
   ``sweep_{nb,lr,svm}_heatmap.png`` (from ``docs/results/gridsearch_*.csv``).
4. ``docs/results/classifier_table.md`` re-rendered (``baselines.render_classifier_table``).
5. ``docs/results/error_analysis.md``: misclassified WELFake test items of the final model (DistilBERT if present,
   else the classical model with the best val macro-F1), rule-categorised.
6. ``docs/results/cross_dataset.md`` (``src.eval.cross_dataset.run``; ``--no-cross`` skips it).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

from src.config import FIGURES_DIR, MODELS_DIR, RESULTS_DIR, ROOT  # noqa: E402
from src.models import baselines as B  # noqa: E402

LOG_DIR = B.TRAIN_LOG_DIR
DISTILBERT_DIR = MODELS_DIR / "distilbert-welfake-v1"
ERROR_MD = RESULTS_DIR / "error_analysis.md"
DATASETS = ["welfake", "isot"]
# fixed categorical order (dataviz reference palette slots 1-4): NB, LR, SVM, DistilBERT
COLOR = {"nb": "#2a78d6", "lr": "#eb6834", "svm": "#1baf7a", "distilbert": "#eda100"}
NAME = {**B.MODEL_LABEL, "distilbert": "DistilBERT (fine-tuned)"}


def eval_log(dataset: str, model: str):
    return LOG_DIR / f"eval_{dataset}_{model}.json"


# ---------------------------------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------------------------------
def fake_scores(pipe, texts) -> tuple[np.ndarray, np.ndarray]:
    """(predicted labels, score where higher = more fake): P(fake) for NB/LR, the signed margin for LinearSVC."""
    X = pipe[:-1].transform(texts)
    clf = pipe[-1]
    classes = list(clf.classes_)
    if hasattr(clf, "predict_proba"):
        s = clf.predict_proba(X)[:, classes.index("fake")]
    else:
        d = clf.decision_function(X)            # > 0 -> classes_[1]
        s = d if classes[1] == "fake" else -d
    return clf.predict(X), s


def metrics_with_auc(y_true, y_pred, score) -> dict:
    m = B.compute_metrics(y_true, y_pred)
    m["roc_auc"] = float(roc_auc_score(np.asarray(y_true) == "fake", score))
    m["n"] = int(len(y_true))
    return m


def eval_classical() -> dict:
    """{(dataset, model): {"test_df", "pred", "score", "pipe"}} for the figures / error analysis; writes eval logs."""
    out = {}
    for ds in DATASETS:
        frames = {s: B.load_split(ds, s) for s in ("val", "test")}
        for m in B.MODEL_NAMES:
            path = B.tuned_artefact_path(ds, m)
            if not path.exists():
                print(f"skip {ds}/{m}: {path.relative_to(ROOT)} missing (run `make train-baselines`)")
                continue
            pipe = joblib.load(path)
            rec = {"dataset": ds, "model": m, "artefact": str(path.relative_to(ROOT)),
                   "params": {"tfidf": {k: getattr(pipe[0], k) for k in ("ngram_range", "max_features", "sublinear_tf", "min_df")},
                              "clf": {k: v for k, v in pipe[-1].get_params().items() if k in ("C", "alpha")}},
                   "score": "predict_proba" if hasattr(pipe[-1], "predict_proba") else "decision_function"}
            for split, df in frames.items():
                t0 = time.perf_counter()
                pred, score = fake_scores(pipe, B.make_input(df))
                rec[split] = {**metrics_with_auc(df["label"], pred, score), "predict_seconds": round(time.perf_counter() - t0, 1)}
                if split == "test":
                    out[(ds, m)] = {"test_df": df, "pred": pred, "score": score, "pipe": pipe}
            rec["params"]["tfidf"]["ngram_range"] = list(rec["params"]["tfidf"]["ngram_range"])
            eval_log(ds, m).write_text(json.dumps(rec, indent=2))
            print(f"{ds:>7} {m:>3}  val F1 {rec['val']['f1_macro']:.4f}  test F1 {rec['test']['f1_macro']:.4f} "
                  f"AUC {rec['test']['roc_auc']:.4f}")
    return out


def eval_distilbert() -> dict:
    """Pick up the GPU agent's checkpoint + test predictions if they exist; {} otherwise."""
    if not DISTILBERT_DIR.exists():
        print(f"skip DistilBERT: {DISTILBERT_DIR.relative_to(ROOT)} not found (make train-distilbert)")
        return {}
    from src.models import distilbert as D
    if not D.TEST_LOG.exists():
        D.run_eval_test()
    test = json.loads(D.TEST_LOG.read_text())
    full = json.loads(D.FINAL_LOG.read_text()) if D.FINAL_LOG.exists() else {}
    best = full.get("per_epoch", [{}])[full["best_epoch"] - 1] if full.get("best_epoch") else {}
    out = {}
    for ds, r in test["results"].items():
        preds = pd.read_csv(ROOT / r["preds_csv"])
        df = B.load_split(ds, "test")
        preds = df.merge(preds[["id", "p_fake"]], on="id", validate="one_to_one")
        assert len(preds) == len(df), f"DistilBERT {ds} predictions do not cover the test split"
        pred = np.where(preds["p_fake"] >= test.get("threshold", 0.5), "fake", "real")
        rec = {"dataset": ds, "model": "distilbert", "artefact": str(DISTILBERT_DIR.relative_to(ROOT)),
               "params": {"max_len": test.get("max_len"), **full.get("config", {})}, "score": "softmax P(fake)",
               "test": metrics_with_auc(preds["label"], pred, preds["p_fake"].to_numpy())}
        if ds == "welfake" and best:
            rec["val"] = {"accuracy": best.get("val_accuracy"), "f1_macro": best.get("val_macro_f1"), "n": None}
        eval_log(ds, "distilbert").write_text(json.dumps(rec, indent=2))
        out[(ds, "distilbert")] = {"test_df": preds, "pred": pred, "score": preds["p_fake"].to_numpy(), "pipe": None,
                                   "max_len": test.get("max_len")}
        print(f"{ds:>7} distilbert  test F1 {rec['test']['f1_macro']:.4f} AUC {rec['test']['roc_auc']:.4f}")
    return out


# ---------------------------------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------------------------------
def plot_cm(cm, title: str, path) -> None:
    cm = np.asarray(cm)
    pct = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.2, 3.8), dpi=200)
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n{pct[i, j]:.1%}", ha="center", va="center",
                    color="white" if pct[i, j] > 0.6 else "#1a1a19", fontsize=10)
    ax.set_xticks([0, 1], ["real", "fake"])
    ax.set_yticks([0, 1], ["real", "fake"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_roc(runs: dict, ds: str, path) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 4.4), dpi=200)
    for (d, m), r in runs.items():
        if d != ds:
            continue
        y = r["test_df"]["label"].to_numpy() == "fake"
        fpr, tpr, _ = roc_curve(y, r["score"])
        ax.plot(fpr, tpr, lw=2, color=COLOR[m], label=f"{NAME[m]} (AUC {roc_auc_score(y, r['score']):.4f})")
    ax.plot([0, 1], [0, 1], lw=1, ls="--", color="#999999")
    ax.set_xlabel("false-positive rate (real flagged fake)")
    ax.set_ylabel("true-positive rate (fake caught)")
    ax.set_title(f"ROC — {'WELFake' if ds == 'welfake' else 'ISOT'} test", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_sweep_heatmap(model: str) -> None:
    csv = B.gridsearch_csv(model)
    if not csv.exists():
        print(f"skip sweep heat-map {model}: {csv.name} missing (make tune-baselines)")
        return
    df = pd.read_csv(csv)
    hp = "param_clf__alpha" if model == "nb" else "param_clf__C"
    df["row"] = "ngram " + df["param_tfidf__ngram_range"].astype(str) + ", sublinear " + df["param_tfidf__sublinear_tf"].astype(str)
    df["col"] = (df["param_tfidf__max_features"] // 1000).astype(str) + "k, " + hp.split("__")[1] + "=" + df[hp].astype(str)
    order_c = df.sort_values(["param_tfidf__max_features", hp])["col"].unique()
    tab = df.pivot(index="row", columns="col", values="mean_test_score")[order_c]
    fig, ax = plt.subplots(figsize=(1.0 * len(order_c) + 2.5, 0.6 * len(tab) + 1.8), dpi=200)
    im = ax.imshow(tab.to_numpy(), cmap="Blues", aspect="auto")
    best = df.loc[df["best"]].iloc[0]
    for i, r in enumerate(tab.index):
        for j, c in enumerate(order_c):
            v = tab.loc[r, c]
            is_best = r == best["row"] and c == best["col"]
            ax.text(j, i, f"{v:.4f}" + ("\n★ best" if is_best else ""), ha="center", va="center", fontsize=7,
                    color="white" if v > np.nanpercentile(tab.to_numpy(), 60) else "#1a1a19",
                    fontweight="bold" if is_best else "normal")
    ax.set_xticks(range(len(order_c)), order_c, rotation=35, ha="right", fontsize=7)
    ax.set_yticks(range(len(tab)), tab.index, fontsize=7)
    fig.colorbar(im, ax=ax, label="mean 5-fold CV macro-F1")
    ax.set_title(f"GridSearchCV — {NAME[model]} on WELFake train ({len(df)} configs × 5 folds)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"sweep_{model}_heatmap.png")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------
# error analysis
# ---------------------------------------------------------------------------------------------------
FIRST_PERSON = re.compile(r"\b(i|me|my|we|our|us)\b", re.I)
SATIRE = re.compile(r"\b(satire|satirical|parody|the onion|humou?r)\b", re.I)
OPINION = re.compile(r"\b(op-ed|opinion|commentary|editorial|column)\b", re.I)
REAL_CUES = re.compile(r"\b(reuters|associated press|breitbart|said on (monday|tuesday|wednesday|thursday|friday))\b", re.I)
WIRE = re.compile(r"\b(reuters|associated press|\(ap\))", re.I)
FAKE_CUES = re.compile(r"\b(via|watch|video|twitter|breaking|hillary)\b", re.I)
NEAR_DUP_COS = 0.8
SHORT_WORDS = 80
FIRST_PERSON_PER_100 = 2.0

CATEGORY_RULES = [
    ("near-duplicate with conflicting label (label noise)", f"TF-IDF cosine ≥ {NEAR_DUP_COS} to a train article with the other label"),
    ("truncation", "DistilBERT only: article > 2 × max_len words, so most of it was never seen"),
    ("short / low-content", f"< {SHORT_WORDS} words of title + text"),
    ("republished wire copy (site-level label)", "true fake, but the first 300 characters credit Reuters / AP: WELFake labels by "
     "publishing site, so wire copy re-posted on a fake-labelled site is 'fake' although the text is wire news"),
    ("satire / parody marker", "satire / parody / The Onion / humour mentioned"),
    ("opinion / first-person commentary", f"≥ {FIRST_PERSON_PER_100} first-person pronouns per 100 words, or op-ed / opinion / commentary / column"),
    ("source cue points the wrong way", "predicted real and contains Reuters / AP / Breitbart / 'said on <weekday>', or predicted fake and contains via / watch / video / twitter / breaking / hillary"),
    ("style overlap (wire-style fake / tabloid-style real)", "none of the above: the register itself is atypical for the true class"),
]


def categorise(text: str, true: str, pred: str, nn_cos: float, nn_label: str, max_len: int | None) -> tuple[str, str]:
    words = len(text.split())
    fp = len(FIRST_PERSON.findall(text)) * 100 / max(words, 1)
    cues = (REAL_CUES if pred == "real" else FAKE_CUES).findall(text)
    cue_words = sorted({(c[0] if isinstance(c, tuple) else c).lower() for c in cues})
    comment = f"{words} words; nearest train doc cos {nn_cos:.2f} ({nn_label}); first-person {fp:.1f}/100w"
    if cue_words:
        comment += f"; {pred}-leaning cues: {', '.join(cue_words)}"
    if nn_cos >= NEAR_DUP_COS and nn_label != true:
        return CATEGORY_RULES[0][0], comment
    if max_len and words > 2 * max_len:
        return CATEGORY_RULES[1][0], comment
    if words < SHORT_WORDS:
        return CATEGORY_RULES[2][0], comment
    if true == "fake" and WIRE.search(text[:300]):
        return CATEGORY_RULES[3][0], comment
    if SATIRE.search(text):
        return CATEGORY_RULES[4][0], comment
    if fp >= FIRST_PERSON_PER_100 or OPINION.search(text[:300]):
        return CATEGORY_RULES[5][0], comment
    if cue_words:
        return CATEGORY_RULES[6][0], comment
    return CATEGORY_RULES[7][0], comment


def error_analysis(runs: dict, n_show: int = 25) -> None:
    candidates = [m for (d, m) in runs if d == "welfake"]
    if not candidates:
        print("skip error analysis: no WELFake model evaluated")
        return
    if "distilbert" in candidates:
        model = "distilbert"
    else:
        model = max((m for m in candidates if m != "distilbert"),
                    key=lambda m: json.loads(eval_log("welfake", m).read_text())["val"]["f1_macro"])
    r = runs[("welfake", model)]
    df = r["test_df"].reset_index(drop=True)
    # TF-IDF neighbourhood (the best classical vectoriser) for the near-duplicate / label-noise rule
    tfidf_pipe = r["pipe"] or next(v["pipe"] for (d, m), v in runs.items() if d == "welfake" and v["pipe"] is not None)
    train = B.load_split("welfake", "train")
    Xtr = tfidf_pipe[0].transform(B.make_input(train))
    wrong = np.flatnonzero(np.asarray(r["pred"]) != df["label"].to_numpy())
    texts = B.make_input(df.iloc[wrong])
    sims = (tfidf_pipe[0].transform(texts) @ Xtr.T).toarray()
    nn = sims.argmax(1)
    thr = 0.5 if model in ("nb", "lr", "distilbert") else 0.0
    rows = []
    for k, i in enumerate(wrong):
        cat, comment = categorise(texts.iloc[k], df.at[i, "label"], r["pred"][i], float(sims[k, nn[k]]),
                                  train.at[nn[k], "label"], r.get("max_len"))
        rows.append({"id": df.at[i, "id"], "true": df.at[i, "label"], "pred": r["pred"][i], "score": float(r["score"][i]),
                     "confidence": abs(float(r["score"][i]) - thr), "category": cat, "comment": comment,
                     "nn_id": train.at[nn[k], "id"], "text": re.sub(r"\s+", " ", texts.iloc[k])[:220]})
    err = pd.DataFrame(rows)
    # most confident mistakes, both directions (missed fakes first — the dangerous cell)
    fn = err[err["true"] == "fake"].nlargest((n_show + 1) // 2, "confidence")
    fp = err[err["true"] == "real"].nlargest(n_show // 2, "confidence")
    shown = pd.concat([fn, fp])
    score_name = {"nb": "P(fake)", "lr": "P(fake)", "svm": "SVM margin (> 0 = fake)", "distilbert": "P(fake)"}[model]
    freq = pd.DataFrame({"all errors": err["category"].value_counts(), f"shown ({len(shown)})": shown["category"].value_counts()}
                        ).fillna(0).astype(int).reindex([c for c, _ in CATEGORY_RULES]).fillna(0).astype(int)
    L = ["# Error analysis — WELFake test (`docs/results/error_analysis.md`)", "",
         f"Generated by `make eval-models` (`src/eval/eval_models.py`). Model: **{NAME[model]}** "
         f"(`{json.loads(eval_log('welfake', model).read_text())['artefact']}`) — "
         + ("the final DistilBERT checkpoint." if model == "distilbert" else
            "the classical model with the best **val** macro-F1; `make eval-models` switches to DistilBERT automatically "
            "once `data/models/distilbert-welfake-v1/` exists.") +
         f" Test split: {len(df):,} articles, **{len(err):,} misclassified** "
         f"({(err['true'] == 'fake').sum():,} missed fakes = fake→real, {(err['true'] == 'real').sum():,} false alarms = real→fake).", "",
         "Categories are assigned by fixed, ordered rules (first match wins) so that the table is regenerable and not hand-picked; "
         "the comment column shows the evidence each rule looked at. The rules are heuristics — a category is a *hypothesis* about the "
         "error, to be read together with the text.", "",
         "| # | Category | Rule |", "|---|---|---|", *[f"| {i + 1} | {c} | {d} |" for i, (c, d) in enumerate(CATEGORY_RULES)], "",
         "## Category frequency", "", "| Category | all errors | shown below |", "|---|---|---|",
         *[f"| {c} | {a} | {b} |" for c, (a, b) in freq.iterrows()], f"| **total** | **{len(err)}** | **{len(shown)}** |", "",
         f"## {len(shown)} most confident mistakes ({len(fn)} missed fakes, then {len(fp)} false alarms)", "",
         f"Score = {score_name}. Text = title + body, first 220 characters.", "",
         "| id | true | pred | score | category | comment | text |", "|---|---|---|---|---|---|---|"]
    for _, e in shown.iterrows():
        t = e["text"].replace("|", "/")
        L.append(f"| `{e['id']}` | {e['true']} | {e['pred']} | {e['score']:.3f} | {e['category']} | {e['comment']}; "
                 f"nn `{e['nn_id']}` | {t} |")
    top = freq["all errors"].idxmax()
    L += ["", "## Reading", "",
          f"Most frequent category over all {len(err)} errors: **{top}** ({freq.at[top, 'all errors']} = "
          f"{freq.at[top, 'all errors'] / len(err):.0%}). Missed fakes (fake→real) are the user-facing danger; they are listed first. "
          "Near-duplicates with the opposite label are corpus label noise (WELFake merges four sources that sometimes carry the same "
          "story with different labels) — no classifier can be right on both copies. Wire-style fakes and tabloid-style reals show that "
          "the model reads register, not veracity — the reason the app's verdict comes from retrieved evidence and the classifier "
          "only adjusts confidence (master §14).", ""]
    ERROR_MD.write_text("\n".join(L))
    err.drop(columns=["text"]).to_csv(LOG_DIR / f"errors_welfake_{model}.csv", index=False)
    print("wrote", ERROR_MD, f"({len(err)} errors, model {model})")


# ---------------------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="MSE2 classifier evaluation (test split) + figures + analyses")
    ap.add_argument("--no-cross", action="store_true", help="skip the cross-dataset study (~5 min)")
    args = ap.parse_args(argv)
    t0 = time.perf_counter()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    runs = {**eval_classical(), **eval_distilbert()}
    for ds, m in runs:
        cm = json.loads(eval_log(ds, m).read_text())["test"]["confusion"]
        plot_cm(cm, f"{NAME[m]} — {'WELFake' if ds == 'welfake' else 'ISOT'} test", FIGURES_DIR / f"cm_{m}_{ds}.png")
    for ds in DATASETS:
        if any(d == ds for d, _ in runs):
            plot_roc(runs, ds, FIGURES_DIR / f"roc_{ds}.png")
    for m in B.MODEL_NAMES:
        plot_sweep_heatmap(m)
    print("wrote", B.render_classifier_table())
    error_analysis(runs)
    if not args.no_cross:
        from src.eval import cross_dataset
        cross_dataset.run()
    print(f"eval-models done in {(time.perf_counter() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
