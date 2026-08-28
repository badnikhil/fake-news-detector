#!/usr/bin/env python
"""Generate notebooks/02_baselines.ipynb (source only; execute it in place with
`make nb-run-baselines` so the outputs are saved for the examiners)."""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "02_baselines.ipynb"

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))  # noqa: E731
code = lambda s: cells.append(nbf.v4.new_code_cell(s))  # noqa: E731

md("""# 02 — TF-IDF baselines (NB / LR / SVM) on WELFake — Model Identification (MSE1)

**Fake News & Misinformation Detector** · MSE1 criterion *Model Identification* (master doc §12, MSE1 doc §3.5).

What this notebook shows

1. the processed WELFake parquet joined with the **saved split IDs** (`data/splits/welfake_{train,val}.csv`) — no re-splitting here; the **test split is not touched** (reserved for MSE2);
2. one TF-IDF matrix (word 1–2-grams, `sublinear_tf`, 100k features, lower-casing inside the vectoriser only) shared by **Multinomial NB, Logistic Regression and LinearSVC**; accuracy / precision / recall / **macro-F1** / confusion matrices on **val**;
3. the top-25 positive / negative LR coefficients and an explicit **leakage-token check** (Reuters, "featured image", pic.twitter, …);
4. a train-size sanity check (learning curve) and a quick ISOT-val run for comparison (+ a cross-dataset preview);
5. the saved artefact `data/models/tfidf_lr_welfake_v0.joblib` reloaded in one line and applied to three hand-written headlines;
6. the **model ladder** that these numbers justify (details: `docs/model_identification.md`).

All code lives in `src/models/baselines.py` (also the CLI behind `make train-baselines`). Runtime ≈ 3–5 min on the laptop CPU (budget ≤ 10 min).""")

code("""import json, sys, platform, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
from src import config as C
from src.models import baselines as B
import sklearn
warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_colwidth", 80); pd.set_option("display.width", 160); pd.set_option("display.precision", 4)
C.FIGURES_DIR.mkdir(parents=True, exist_ok=True); C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
print("python", platform.python_version(), "| sklearn", sklearn.__version__, "| pandas", pd.__version__)
print("TF-IDF config:", {k: v for k, v in B.TFIDF_DEFAULTS.items() if k != "dtype"})
T0 = time.time()""")

md("""## 1. Load the processed data and join the saved split IDs

`B.load_split(dataset, split)` reads `data/splits/{dataset}_{split}.csv` (`id,label`, written once by `make data`, seed 42) and joins it onto
`data/processed/{dataset}.parquet` by `id`; it raises if any id is missing or a label disagrees. The text stays **cased** on disk — lower-casing
happens inside `TfidfVectorizer`.""")

code("""train = B.load_split("welfake", "train"); val = B.load_split("welfake", "val")
test_ids = pd.read_csv(C.SPLITS_DIR / "welfake_test.csv")           # only counted, never loaded as text
assert not set(train["id"]) & set(val["id"]) and not set(train["id"]) & set(test_ids["id"])
summary = pd.DataFrame({"rows": {"train": len(train), "val": len(val), "test (untouched)": len(test_ids)},
                        "real": {"train": (train.label == "real").sum(), "val": (val.label == "real").sum(), "test (untouched)": (test_ids.label == "real").sum()},
                        "fake": {"train": (train.label == "fake").sum(), "val": (val.label == "fake").sum(), "test (untouched)": (test_ids.label == "fake").sum()}})
summary["fake_frac"] = (summary["fake"] / summary["rows"]).round(4)
X_train_txt = B.make_input(train)
print("classifier input = title + '\\\\n' + text; median length", int(X_train_txt.str.len().median()), "chars; cased sample:", repr(X_train_txt.iloc[0][:90]))
summary""")

md("""## 2. TF-IDF + NB / LR / SVM on WELFake train → **val**

Configuration = one point of the master §12.2 grid (the full `GridSearchCV` is MSE2 work): word 1–2-grams, `sublinear_tf=True`,
`max_features=100 000`, `min_df=2`, `max_df=0.95`; NB `alpha=1`, LR `C=1` (liblinear), LinearSVC `C=1`.
The TF-IDF matrix is fitted **once** on train and shared by the three classifiers (master §12.3: "cache the TF-IDF matrix").""")

code("""pipes, res, cache = B.fit_eval_all("welfake", B.MODEL_NAMES, train_df=train, eval_df=val)
rows = []
for m, r in res.items():
    rows.append({"model": B.MODEL_LABEL[m], "split": r.split, "accuracy": r.accuracy, "precision_macro": r.precision_macro,
                 "recall_macro": r.recall_macro, "macro_F1": r.f1_macro, "P_fake": r.precision_fake, "R_fake": r.recall_fake,
                 "train_s (tfidf+fit)": round(r.train_seconds, 1), "fit_s": round(r.fit_seconds, 2)})
metrics_welfake = pd.DataFrame(rows).set_index("model")
lr_f1 = res["lr"].f1_macro
print(f"\\nMSE1 sanity bar: LR macro-F1 on WELFake val = {lr_f1:.4f} -> {'PASS (>= 0.90)' if lr_f1 >= 0.90 else 'FAIL (< 0.90)'}")
print(f"TF-IDF: {cache['X_train'].shape[1]:,} features, train matrix {cache['X_train'].shape[0]:,} x {cache['X_train'].shape[1]:,}, nnz {cache['X_train'].nnz:,}")
metrics_welfake""")

md("### 2.1 Confusion matrices on val (rows = true, columns = predicted; `fake` is the positive class)")

code("""fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for ax, (m, r) in zip(axes, res.items()):
    ConfusionMatrixDisplay(np.asarray(r.confusion), display_labels=B.LABELS).plot(ax=ax, colorbar=False, values_format="d", cmap="Blues")
    ax.set_title(f"{B.MODEL_LABEL[m]}\\nval macro-F1 = {r.f1_macro:.4f}", fontsize=10)
fig.suptitle("WELFake validation (n = %d) — TF-IDF baselines" % len(val)); fig.tight_layout()
fig.savefig(C.FIGURES_DIR / "cm_baselines_welfake_val.png", dpi=150, bbox_inches="tight")

fig, ax = plt.subplots(figsize=(4.2, 3.8))
ConfusionMatrixDisplay(np.asarray(res["lr"].confusion), display_labels=B.LABELS).plot(ax=ax, colorbar=False, values_format="d", cmap="Blues")
ax.set_title(f"TF-IDF + LR — WELFake val\\nacc {res['lr'].accuracy:.4f} · macro-F1 {res['lr'].f1_macro:.4f}", fontsize=10)
fig.tight_layout(); fig.savefig(C.FIGURES_DIR / "cm_lr_welfake_val.png", dpi=150, bbox_inches="tight")
print("saved", C.FIGURES_DIR / "cm_lr_welfake_val.png", "and cm_baselines_welfake_val.png")
plt.show()""")

md("""### 2.2 Top-25 LR coefficients (→ fake vs. → real) and the leakage-token check

If a source / boiler-plate token such as `reuters`, `featured image`, `getty`, `pic twitter`, `video`, `via` … ranks among the top coefficients,
the classifier is learning *where the text came from* rather than *what it says* — that is a **preprocessing gap** and is reported below, not hidden.""")

code("""coef = B.top_coefficients(pipes["lr"], k=25)
leak_fake = B.find_leak_tokens(coef["fake_token"]); leak_real = B.find_leak_tokens(coef["real_token"])
print("LEAK CHECK (top-25 LR tokens vs. B.LEAK_TOKENS):")
print("  towards FAKE:", leak_fake if leak_fake else "none")
print("  towards REAL:", leak_real if leak_real else "none")
if leak_fake or leak_real:
    print("  -> PREPROCESSING GAP: these tokens survived artefact stripping (see the discussion cell below and agent-docs / eda_summary).")
else:
    print("  -> clean: no leakage tokens in the top-25 lists.")
# how often do the flagged tokens occur, per class, in the *training* text? (context for the discussion)
ctx = []
for t in (leak_fake + leak_real):
    pat = r"\\b" + r"\\s+".join(map(__import__("re").escape, t.split())) + r"\\b"
    has = X_train_txt.str.contains(pat, case=False, regex=True)
    ctx.append({"token": t, "docs_fake": int((has & (train.label == 'fake')).sum()), "docs_real": int((has & (train.label == 'real')).sum()),
                "frac_fake": round(float(has[train.label == 'fake'].mean()), 4), "frac_real": round(float(has[train.label == 'real'].mean()), 4)})
leak_context = pd.DataFrame(ctx)
coef""")

code("""print("Occurrence of the flagged tokens in the WELFake *train* text (per class):")
display(leak_context) if len(leak_context) else print("(nothing flagged)")
# a few example contexts for each flagged token, to see whether it is a residual artefact or ordinary prose
import re
for t in (leak_fake + leak_real)[:4]:
    pat = re.compile(r".{0,50}\\b" + r"\\s+".join(map(re.escape, t.split())) + r"\\b.{0,50}", re.I | re.S)
    hits = [pat.search(x).group(0).replace("\\n", " ") for x in X_train_txt[train.label == "fake"].head(4000) if pat.search(x)][:3]
    print(f"\\n[{t}] e.g.:"); [print("   …", h, "…") for h in hits]""")

md("""**Reading the coefficients.** Politically loaded, sensational and second-person tokens (e.g. *hillary*, *you*, *breaking*, *video*, *just*) drive
predictions towards *fake*; wire-service register (*said on*, *washington*, *reuters*/*minister*, *percent*, dates) drives them towards *real*. This is a
**style / register** signal, not a truth signal — it is exactly why the classifier only adjusts confidence in the fusion (master §14) and why the
cross-dataset drop (§18) is expected. Tokens flagged above are recorded in `agent-docs/README.md` and `docs/model_identification.md` as a preprocessing follow-up.""")

md("""## 3. Train-size sanity check (learning curve, LR)

Same fitted TF-IDF; LR refitted on random subsets of the train rows (seed 42). A curve that keeps rising means more data still helps; a flat curve
means the linear model has saturated (the transformer must then win on *representation*, not on data volume).""")

code("""rng = np.random.default_rng(C.SEED)
sizes = [1_000, 2_000, 5_000, 10_000, 20_000, cache["X_train"].shape[0]]
curve = []
for n in sizes:
    idx = np.sort(rng.choice(cache["X_train"].shape[0], size=n, replace=False)) if n < cache["X_train"].shape[0] else np.arange(n)
    clf = B.build_classifier("lr"); t0 = time.time(); clf.fit(cache["X_train"][idx], cache["y_train"][idx])
    m = B.compute_metrics(cache["y_eval"], clf.predict(cache["X_eval"]))
    curve.append({"n_train": n, "val_macro_F1": round(m["f1_macro"], 4), "val_acc": round(m["accuracy"], 4), "fit_s": round(time.time() - t0, 2)})
curve = pd.DataFrame(curve)
fig, ax = plt.subplots(figsize=(6, 3.6))
ax.plot(curve["n_train"], curve["val_macro_F1"], marker="o"); ax.set_xscale("log"); ax.set_xlabel("train rows (log)"); ax.set_ylabel("val macro-F1")
ax.axhline(0.90, ls="--", c="grey", lw=1); ax.text(sizes[0], 0.902, "MSE1 sanity bar 0.90", fontsize=8, color="grey")
ax.set_title("TF-IDF + LR learning curve — WELFake val"); ax.grid(alpha=.3); fig.tight_layout()
fig.savefig(C.FIGURES_DIR / "train_lr_welfake_learning_curve.png", dpi=150, bbox_inches="tight"); plt.show()
curve""")

md("""## 4. Save the LR pipeline → `data/models/tfidf_lr_welfake_v0.joblib` and reload in one line

The artefact is a self-contained sklearn `Pipeline(tfidf → clf)`: `joblib.load(path).predict([text])` is all a CPU-only teammate needs.""")

code("""path = B.save_pipeline(pipes["lr"], C.MODELS_DIR / "tfidf_lr_welfake_v0.joblib")
print("saved", path, f"({path.stat().st_size / 2**20:.1f} MB)")
B.save_log(res["lr"].__class__(**{**res["lr"].__dict__, "artefact": str(path.relative_to(ROOT))}))   # JSON log for classifier_table.md
import joblib
clf = joblib.load(C.MODELS_DIR / "tfidf_lr_welfake_v0.joblib")            # <- the one-line reload
heads = B.demo_headlines()
proba = clf.predict_proba([h for _, h in heads]); fake_col = list(clf.classes_).index("fake")
pd.DataFrame({"intent": [i for i, _ in heads], "headline": [h for _, h in heads],
              "pred": clf.predict([h for _, h in heads]), "P(fake)": proba[:, fake_col].round(3)})""")

md("""> The headline predictions are a *feasibility* demo only: a 2-sentence input has almost no TF-IDF mass, and the model was trained on full
> articles (title + body). The web app will use the classifier the same way — as one confidence signal — never as the verdict (hard rule in `CLAUDE.md`).""")

md("""## 5. Quick ISOT-val run for comparison (+ cross-dataset preview)

ISOT is the "leakage story" dataset (raw artefact-only accuracy ≈ 0.996, see `docs/mse1_make_data.log`). After artefact stripping the same three
baselines are fitted on ISOT train and scored on ISOT val. As a preview of MSE2's cross-dataset table, the **WELFake-trained** LR is also applied to
ISOT val without any adaptation (final cross-dataset numbers use the *test* splits at MSE2).""")

code("""del cache; import gc; gc.collect()                    # free the WELFake matrices before the second TF-IDF fit
isot_train = B.load_split("isot", "train"); isot_val = B.load_split("isot", "val")
pipes_isot, res_isot, cache_isot = B.fit_eval_all("isot", B.MODEL_NAMES, train_df=isot_train, eval_df=isot_val)
for m, r in res_isot.items():
    B.save_log(r)
metrics_isot = pd.DataFrame([{"model": B.MODEL_LABEL[m], "dataset": "ISOT", "split": r.split, "accuracy": r.accuracy, "precision_macro": r.precision_macro,
                              "recall_macro": r.recall_macro, "macro_F1": r.f1_macro, "train_s (tfidf+fit)": round(r.train_seconds, 1)} for m, r in res_isot.items()]).set_index("model")
# cross-dataset preview: WELFake LR -> ISOT val (no retraining)
xm = B.compute_metrics(isot_val["label"], pipes["lr"].predict(B.make_input(isot_val)))
xm2 = B.compute_metrics(val["label"], pipes_isot["lr"].predict(B.make_input(val)))
cross = pd.DataFrame([{"train → eval": "WELFake → ISOT val", "accuracy": xm["accuracy"], "macro_F1": xm["f1_macro"], "confusion": xm["confusion"]},
                      {"train → eval": "ISOT → WELFake val", "accuracy": xm2["accuracy"], "macro_F1": xm2["f1_macro"], "confusion": xm2["confusion"]}])
print("In-domain ISOT val:"); display(metrics_isot)
print("Cross-dataset preview (LR, C=1, no adaptation):"); cross""")

md("""### 5.1 Why WELFake → ISOT does *not* drop: ISOT is (almost entirely) contained in WELFake

WELFake was assembled from four sources — Kaggle, McIntire, **Reuters (= ISOT)** and BuzzFeed Political (Verma et al. 2021). A hash join on the
normalised processed text quantifies the overlap; this decides what "cross-dataset" can mean at MSE2.""")

code("""import hashlib, re
norm = lambda s: re.sub(r"\\W+", " ", s.lower()).strip()
h = lambda s: hashlib.sha1(norm(s).encode()).hexdigest()
w_all = pd.read_parquet(C.PROCESSED_DIR / "welfake.parquet", columns=["id", "text", "split"]); i_all = pd.read_parquet(C.PROCESSED_DIR / "isot.parquet", columns=["id", "text", "label", "split"])
w_all["h"] = w_all.text.map(h); i_all["h"] = i_all.text.map(h)
shared = set(w_all.h) & set(i_all.h); w_train_h = set(w_all[w_all.split == "train"].h)
overlap = pd.DataFrame([
    {"check": "ISOT rows whose text also occurs in WELFake", "n": int(i_all.h.isin(shared).sum()), "of": len(i_all)},
    {"check": "WELFake rows involved", "n": int(w_all.h.isin(shared).sum()), "of": len(w_all)},
    {"check": "ISOT val rows present in WELFake TRAIN", "n": int(i_all[i_all.split == "val"].h.isin(w_train_h).sum()), "of": int((i_all.split == "val").sum())},
    {"check": "ISOT test rows present in WELFake TRAIN", "n": int(i_all[i_all.split == "test"].h.isin(w_train_h).sum()), "of": int((i_all.split == "test").sum())},
    {"check": "WELFake rows NOT in ISOT (= the only truly separate domain)", "n": int((~w_all.h.isin(shared)).sum()), "of": len(w_all)}])
overlap["frac"] = (overlap.n / overlap.of).round(4)
del w_all, i_all
overlap""")

md("""**Interpretation.** In-domain ISOT stays very high even after removing the dateline/`(Reuters)` artefacts (the two classes still come from
disjoint outlets with distinct house styles; the residual bare "Reuters" in prose and the fake side's apostrophe-stripping quirk are documented in
`data/README.md`). The cross-dataset preview is asymmetric for a structural reason: **≈ 99.9 % of ISOT is inside WELFake**, so *WELFake → ISOT*
is in-domain (most ISOT-val rows were even in WELFake train) and must not be reported as cross-dataset; *ISOT → WELFake* (≈ 0.83 macro-F1) is the
honest out-of-domain number for a bag-of-n-grams model — it learns *register*, not *veracity*. MSE2's `docs/results/cross_dataset.md` should therefore
use the hash-join-defined **WELFake∖ISOT ↔ ISOT** pair (see `docs/model_identification.md` §4.2). Either way the classifier cannot be the final
verdict — it becomes one signal in the evidence-first fusion (master §14, rule R8 / ±0.10 confidence).""")

md("""## 6. Model ladder (what these numbers justify)

| Tier | Model | Why included | Status |
|---|---|---|---|
| Baseline | TF-IDF + Multinomial NB | seconds to train; classic floor | trained here (val) |
| Baseline | **TF-IDF + Logistic Regression** | strong linear baseline; interpretable coefficients (leakage audit); `predict_proba` for fusion | **trained here, saved as `tfidf_lr_welfake_v0.joblib`** |
| Baseline | TF-IDF + LinearSVC | usually the best classical model; CPU fallback for `MODE=lite` | trained here (val) |
| Middle (optional) | Bi-LSTM + GloVe 100d | sequence modelling for comparison; only if time permits (week 9) | MSE2 (optional) |
| **Final** | **DistilBERT-base-uncased fine-tuned** | best accuracy/size trade-off; 66 M params; fp16 · max_len 256 · batch 16 fits the 4 GB RTX 2050; 6-config sweep (lr × max_len) on a 20 k subset, then full retrain | MSE2 |
| Fine-grained | DistilBERT (or TF-IDF+LR) on LIAR 3-way | truth-shade probability for short claims | MSE2 |
| Stance | `cross-encoder/nli-deberta-v3-small` | NLI ≡ supports / refutes / neutral; zero training; CPU-usable | MSE2 |
| Embedding | `all-MiniLM-L6-v2` + FAISS | fast retrieval / offline index | MSE2 |

Rejected: BERT-base / RoBERTa-large (2–3× memory and time on a 4 GB GPU for a component that only adjusts confidence), GPT-style LLMs and zero-shot
prompting (cost, non-reproducibility, hallucinated "evidence" conflicts with NFR-3). Full rationale, DistilBERT configuration, hardware budget and the
CPU/Colab fallback: **`docs/model_identification.md`**; numbers: **`docs/results/classifier_table.md`**.""")

code("""B.render_classifier_table()
print("classifier table ->", B.CLASSIFIER_TABLE)
print(f"notebook runtime: {time.time() - T0:.0f} s")""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"name": "fakenews", "display_name": "Python (fakenews)", "language": "python"},
    "language_info": {"name": "python"},
}
nbf.write(nb, OUT)
print("wrote", OUT, f"({len(cells)} cells)")
