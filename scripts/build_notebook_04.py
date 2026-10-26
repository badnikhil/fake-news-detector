#!/usr/bin/env python
"""Generate notebooks/04_tuning.ipynb (source only; execute it in place with `make nb-run-tuning`).

The notebook never trains: it reads docs/results/gridsearch_{nb,lr,svm}.csv (`make tune-baselines`), the eval logs
docs/results/train_logs/eval_*.json (`make eval-models`) and, if present, docs/results/distilbert_sweep.md."""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "04_tuning.ipynb"

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))  # noqa: E731
code = lambda s: cells.append(nbf.v4.new_code_cell(s))  # noqa: E731

md("""# 04 — Hyperparameter tuning (MSE2 criterion 2)

**Fake News & Misinformation Detector** · MSE2 §4 (master doc §12.2).

1. the **GridSearchCV** grids for TF-IDF + NB / LR / LinearSVC (5-fold stratified CV on WELFake *train*, seed 42, scoring macro-F1) — every
   configuration tried, straight from the raw `cv_results_` exports `docs/results/gridsearch_{nb,lr,svm}.csv`;
2. the SVM (and LR / NB) heat-maps `docs/figures/sweep_*_heatmap.png`;
3. the marginal effect of each hyperparameter;
4. the **val-vs-test gap** of the winners (from `make eval-models`);
5. the DistilBERT sweep table (`docs/results/distilbert_sweep.md`, written by the GPU agent's `make train-distilbert`) if it exists.

This notebook only reads saved results (≈ 10 s). Regenerate the inputs with `make tune-baselines` (≈ 2 h CPU) → `make train-baselines` → `make eval-models`.""")

code("""import json, sys, time
from pathlib import Path
import pandas as pd
from IPython.display import Image, Markdown, display

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
from src import config as C
from src.models import baselines as B
from src.eval import eval_models as E
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30); pd.set_option("display.precision", 4)
T0 = time.time()""")

md("""## 1. The grids

`B.param_grid(model)` — one dict per TF-IDF combination so that `FoldCachedTfidf` fits each (TF-IDF config, fold) only once and re-uses it
for the three C / alpha values (identical scores to a plain `TfidfVectorizer`, `tests/test_models.py`). Fixed for every candidate:
`min_df=2`, `max_df=0.95`, lower-casing inside the vectoriser, `strip_accents="unicode"`. NB keeps `sublinear_tf=True` fixed (18 configs).""")

code("""from sklearn.model_selection import ParameterGrid
print({m: len(ParameterGrid(B.param_grid(m))) for m in B.MODEL_NAMES})
print("TF-IDF grid:", B.TFIDF_GRID); print("classifier grids:", B.CLF_GRID)
logs = {m: json.loads((B.TRAIN_LOG_DIR / f"gridsearch_{m}.json").read_text()) for m in B.MODEL_NAMES}
pd.DataFrame({m: {"candidates": l["n_candidates"], "n_train": l["n_train"], "cv": l["cv"], "wall_min": round(l["wall_seconds"] / 60, 1),
                  "best CV macro-F1": round(l["best_cv_f1_macro"], 4), "best params": l["best_params"]} for m, l in logs.items()}).T""")

md("""## 2. Every configuration tried (no cherry-picking)

Sorted by mean CV macro-F1; `best` = `rank_test_score == 1`. `mean_fit_time` is small for all but the first C/alpha of each TF-IDF
configuration because the fitted vectoriser is re-used (see §1).""")

code("""cols = lambda m: ["param_tfidf__ngram_range", "param_tfidf__max_features", "param_tfidf__sublinear_tf",
                  "param_clf__alpha" if m == "nb" else "param_clf__C", "mean_test_score", "std_test_score", "rank_test_score",
                  "mean_fit_time", "best"]
grids = {m: pd.read_csv(B.gridsearch_csv(m)) for m in B.MODEL_NAMES}
for m, g in grids.items():
    print(f"\\n=== {B.MODEL_LABEL[m]} — {len(g)} configurations ===")
    display(g[cols(m)].sort_values("mean_test_score", ascending=False).reset_index(drop=True))""")

md("## 3. Heat-maps (regenerated from the CSVs; `make eval-models` writes the same files)")

code("""for m in ["svm", "lr", "nb"]:
    E.plot_sweep_heatmap(m)
    display(Image(str(C.FIGURES_DIR / f"sweep_{m}_heatmap.png"), width=900))""")

md("""## 4. Marginal effect of each hyperparameter

Mean CV macro-F1 over all other settings, per value — a quick read of which knob matters.""")

code("""rows = []
for m, g in grids.items():
    for p in [c for c in g.columns if c.startswith("param_")]:
        for v, s in g.groupby(p)["mean_test_score"].mean().items():
            rows.append({"model": m, "hyperparameter": p.replace("param_", ""), "value": v, "mean CV macro-F1": s})
marg = pd.DataFrame(rows)
marg.pivot_table(index=["model", "hyperparameter", "value"], values="mean CV macro-F1")""")

md("""## 5. Winners: CV vs val vs test (over-fitting check)

CV = mean 5-fold score on train (selection); val / test = the winner refitted on the full train split (`make train-baselines`) and scored by
`make eval-models`. A small, consistent gap means the selection did not over-fit; the test split was used once, for this table.""")

code("""gap = []
for m in B.MODEL_NAMES + ["distilbert"]:
    p = B.TRAIN_LOG_DIR / f"eval_welfake_{m}.json"
    if not p.exists():
        continue
    r = json.loads(p.read_text())
    gap.append({"model": m, "CV macro-F1": logs[m]["best_cv_f1_macro"] if m in logs else None,
                "val macro-F1": r.get("val", {}).get("f1_macro"), "test macro-F1": r["test"]["f1_macro"],
                "test ROC-AUC": r["test"]["roc_auc"], "val − test": (r.get("val", {}).get("f1_macro") or float("nan")) - r["test"]["f1_macro"]})
pd.DataFrame(gap)""")

md("""## 6. DistilBERT sweep (lr × max_len, 20k subset) — written by `make train-distilbert`""")

code("""p = C.RESULTS_DIR / "distilbert_sweep.md"
display(Markdown(p.read_text()) if p.exists() else Markdown("_`docs/results/distilbert_sweep.md` not generated yet (`make train-distilbert`)._"))
print(f"notebook runtime: {time.time() - T0:.0f} s")""")

md("""**Reading** (marginal means from §4, `gridsearch_*.csv`, 26 Oct 2026). **Bigrams beat unigrams for every model** (SVM 0.9367 vs 0.9243,
LR 0.9178 vs 0.9121, NB 0.8595 vs 0.8496) and `sublinear_tf=True` helps LR/SVM (SVM 0.9344 vs 0.9267). **Vocabulary size matters least**
(≤ 0.002 between 50k and 200k); the unigram rows are identical at 100k and 200k because the unigram vocabulary after `min_df=2` is smaller than
100k. **Regularisation is the big knob for LR** (C 0.1 → 10: 0.887 → 0.934 mean) — the winner C = 10 sits at the edge of the grid, so a
larger C is worth one extra check at ESE; LinearSVC peaks at C = 1. NB prefers the smallest alpha (0.1) and stays the weakest model everywhere.
Winners: SVM (1,2)-grams / 200k / sublinear / C = 1 (CV 0.9459), LR (1,2) / 100k / sublinear / C = 10 (CV 0.9428), NB (1,2) / 200k / alpha 0.1
(CV 0.8684). Their val-vs-test gaps (§5) are within ±0.003 on WELFake, so CV selection did not over-fit.""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"name": "fakenews", "display_name": "Python (fakenews)", "language": "python"},
    "language_info": {"name": "python"},
}
nbf.write(nb, OUT)
print("wrote", OUT, f"({len(cells)} cells)")
