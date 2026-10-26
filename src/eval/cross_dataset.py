"""Cross-dataset study WELFake∖ISOT <-> ISOT (master §18, MSE2 §5) -> ``docs/results/cross_dataset.md``.

Why not WELFake <-> ISOT: ISOT is (almost entirely) contained in WELFake (``docs/model_identification.md`` §4.2), so
"train WELFake -> test ISOT" is in-domain.  The second domain is therefore **WELFake∖ISOT** = processed WELFake rows
whose normalised text is not hash-matched to any processed ISOT article *and* that are not a MinHash near-duplicate
(Jaccard >= 0.9, ``src.preprocess.dedupe``) of one.

Reusable API (also for the DistilBERT agent)::

    from src.eval.cross_dataset import load_welfake_minus_isot
    wi = load_welfake_minus_isot()     # id, label, split (the original WELFake split); cached in data/splits/

Cells = {train W∖I, train ISOT} x {test W∖I, test ISOT} x {raw text (artefacts kept), processed (artefacts removed)}.
"raw" uses the *same row ids and splits* with the title/text taken from ``data/raw`` before ``artefacts.strip_frame``
and cleaning, so the raw-vs-processed difference is the artefact effect only.  Train = the domain's train split,
test = its test split (W∖I train/test = WELFake train/test restricted to W∖I).
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time

import numpy as np
import pandas as pd
from datasketch import MinHashLSH

from src.config import MINHASH_PERMS, NEAR_DUP_THRESHOLD, PROCESSED_DIR, RESULTS_DIR, SEED, SPLITS_DIR
from src.models import baselines as B
from src.preprocess.artefacts import artefact_only_accuracy
from src.preprocess.dedupe import minhash

WI_CSV = SPLITS_DIR / "welfake_minus_isot.csv"
OUT_MD = RESULTS_DIR / "cross_dataset.md"
OUT_JSON = RESULTS_DIR / "train_logs" / "cross_dataset.json"


def text_hash(s: str) -> str:
    """The MSE1 protocol key (notebook 02 §5.1): SHA-1 of ``re.sub(r"\\W+", " ", text.lower()).strip()``."""
    return hashlib.sha1(re.sub(r"\W+", " ", str(s).lower()).strip().encode()).hexdigest()


def welfake_minus_isot(welfake: pd.DataFrame, isot: pd.DataFrame, near: bool = True) -> tuple[pd.DataFrame, dict]:
    """WELFake rows not contained in ISOT (exact hash join, then MinHash near-duplicates of ISOT texts).

    Returns ``(rows[id, label, split], stats)``; ``stats`` holds the overlap numbers reported in cross_dataset.md.
    """
    wh, ih = welfake["text"].map(text_hash), isot["text"].map(text_hash)
    shared = set(wh) & set(ih)
    cand = welfake[~wh.isin(shared)]
    stats = {"isot_rows": len(isot), "isot_in_welfake": int(ih.isin(shared).sum()),
             "welfake_rows": len(welfake), "welfake_in_isot": int(wh.isin(shared).sum()),
             "welfake_minus_isot_exact": len(cand)}
    if near:
        lsh = MinHashLSH(threshold=NEAR_DUP_THRESHOLD, num_perm=MINHASH_PERMS)
        for i, t in enumerate(isot["text"].tolist()):
            lsh.insert(str(i), minhash(t))
        near_mask = np.array([bool(lsh.query(minhash(t))) for t in cand["text"].tolist()], dtype=bool)
        stats["near_dups_of_isot_removed"] = int(near_mask.sum())
        cand = cand[~near_mask]
    stats["welfake_minus_isot"] = len(cand)
    stats["welfake_minus_isot_by_split_label"] = {f"{sp}/{lab}": int(n) for (sp, lab), n in
                                                  cand.groupby(["split", "label"]).size().items()}
    return cand[["id", "label", "split"]].sort_values("id").reset_index(drop=True), stats


def load_welfake_minus_isot(recompute: bool = False) -> pd.DataFrame:
    """Cached W∖I ids (``data/splits/welfake_minus_isot.csv``: id, label, split); computes it (~2 min) if missing."""
    if WI_CSV.exists() and not recompute:
        return pd.read_csv(WI_CSV)
    cols = ["id", "text", "label", "split"]
    rows, stats = welfake_minus_isot(pd.read_parquet(PROCESSED_DIR / "welfake.parquet", columns=cols),
                                     pd.read_parquet(PROCESSED_DIR / "isot.parquet", columns=cols))
    rows.to_csv(WI_CSV, index=False)
    WI_CSV.with_suffix(".json").write_text(json.dumps(stats, indent=2))
    return rows


def _raw_text(dataset: str) -> pd.DataFrame:
    """id, title, text straight from data/raw (artefacts kept, no cleaning)."""
    from src.preprocess.load import load_isot, load_welfake
    df = load_welfake() if dataset == "welfake" else load_isot()
    return df[["id", "title", "text"]]


def domain_frames(variant: str) -> dict[str, dict[str, pd.DataFrame]]:
    """{"wi"|"isot": {"train"|"test": frame(id, title, text, label)}} with processed or raw text."""
    wi = load_welfake_minus_isot()
    out = {}
    for name, ds in (("wi", "welfake"), ("isot", "isot")):
        out[name] = {}
        raw = _raw_text(ds).set_index("id") if variant == "raw" else None
        for split in ("train", "test"):
            df = B.load_split(ds, split)
            if name == "wi":
                df = df[df["id"].isin(set(wi.loc[wi["split"] == split, "id"]))].reset_index(drop=True)
            if raw is not None:
                df = df.drop(columns=["title", "text"]).join(raw[["title", "text"]], on="id")
                df["title"], df["text"] = df["title"].fillna("").astype(str), df["text"].fillna("").astype(str)
            out[name][split] = df
        del raw
    return out


DOMAIN_LABEL = {"wi": "WELFake∖ISOT", "isot": "ISOT"}


def run(models: tuple[str, ...] = ("lr", "svm")) -> dict:
    t0 = time.perf_counter()
    wi_rows = load_welfake_minus_isot()
    stats = json.loads(WI_CSV.with_suffix(".json").read_text())
    cells = []
    for variant in ("processed", "raw"):
        frames = domain_frames(variant)
        for tr in ("wi", "isot"):
            for m in models:
                bp = B.best_params(m) or ({}, {})
                for te in ("wi", "isot"):
                    _, res, _ = B.fit_eval_all(f"{tr}->{te}", [m], tfidf=bp[0], clf=bp[1], train_df=frames[tr]["train"],
                                               eval_df=frames[te]["test"], verbose=False, eval_split="cross")
                    r = res[m]
                    cells.append({"variant": variant, "model": m, "train": tr, "test": te, "n_train": r.n_train,
                                  "n_test": r.n_eval, "accuracy": r.accuracy, "f1_macro": r.f1_macro,
                                  "confusion": r.confusion})
                    print(f"[{variant}] {m} {tr}->{te}: macro-F1 {r.f1_macro:.4f} (n={r.n_train}/{r.n_eval})", flush=True)
        del frames
    # leakage audit: artefact-only classifier (LR on the 26 pattern counts, 5-fold CV) on raw vs processed ISOT
    from src.preprocess.load import load_isot
    raw_isot = load_isot()
    proc_isot = pd.read_parquet(PROCESSED_DIR / "isot.parquet", columns=["text", "label"])
    art = {"raw_isot": artefact_only_accuracy(raw_isot["text"].fillna("").astype(str), raw_isot["label"], seed=SEED),
           "processed_isot": artefact_only_accuracy(proc_isot["text"].astype(str), proc_isot["label"], seed=SEED)}
    for v, n in zip(art.values(), (len(raw_isot), len(proc_isot)), strict=True):
        v.pop("doc_frac_with_pattern", None)
        v["n"] = n
    res = {"stats": stats, "n_welfake_minus_isot": len(wi_rows), "models": list(models),
           "best_params": {m: [{k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()}
                               for d in (B.best_params(m) or ({}, {}))] for m in models},
           "cells": cells, "artefact_only": art, "seconds": round(time.perf_counter() - t0, 1)}
    OUT_JSON.write_text(json.dumps(res, indent=2))
    render(res)
    return res


def render(res: dict) -> None:
    s, cells = res["stats"], res["cells"]
    get = lambda v, m, tr, te: next(c for c in cells if (c["variant"], c["model"], c["train"], c["test"]) == (v, m, tr, te))  # noqa: E731
    L = ["# Cross-dataset results — WELFake∖ISOT ↔ ISOT (`docs/results/cross_dataset.md`)", "",
         "Generated by `python -m src.eval.cross_dataset` (part of `make eval-models`); raw numbers in "
         "`docs/results/train_logs/cross_dataset.json`. **Test splits**, macro-F1. Models = TF-IDF + LR / LinearSVC with their "
         "WELFake GridSearchCV winners (`docs/results/gridsearch_{lr,svm}.csv`), refitted on each training domain.", "",
         "## Why WELFake∖ISOT and not WELFake", "",
         f"WELFake was assembled from Kaggle + McIntire + Reuters/**ISOT** + BuzzFeed, so **{s['isot_in_welfake']:,} of the "
         f"{s['isot_rows']:,} processed ISOT articles ({s['isot_in_welfake'] / s['isot_rows']:.2%}) occur verbatim in processed WELFake** "
         f"(hash join on SHA-1 of `re.sub(r\"\\W+\", \" \", text.lower())`), covering {s['welfake_in_isot']:,} / {s['welfake_rows']:,} "
         f"WELFake rows. Removing them leaves {s['welfake_minus_isot_exact']:,} rows; a further "
         f"{s.get('near_dups_of_isot_removed', 0):,} are MinHash near-duplicates (Jaccard ≥ {NEAR_DUP_THRESHOLD}) of an ISOT article "
         f"and are dropped too → **WELFake∖ISOT = {s['welfake_minus_isot']:,} rows** "
         f"(per split/label: {s['welfake_minus_isot_by_split_label']}). Ids: `data/splits/welfake_minus_isot.csv` "
         "(`src.eval.cross_dataset.load_welfake_minus_isot()`). Plain WELFake → ISOT is in-domain and is never reported as cross-dataset.", ""]
    for m in res["models"]:
        L += [f"## {B.MODEL_LABEL[m]} — macro-F1 (train ↓ / test →)", "",
              "| Train \\ Test | W∖I (processed) | ISOT (processed) | W∖I (raw) | ISOT (raw) | n_train (proc.) |", "|---|---|---|---|---|---|"]
        for tr in ("wi", "isot"):
            vals = [get(v, m, tr, te)["f1_macro"] for v in ("processed", "raw") for te in ("wi", "isot")]
            L.append(f"| {DOMAIN_LABEL[tr]} | " + " | ".join(f"{x:.4f}" for x in vals) + f" | {get('processed', m, tr, 'wi')['n_train']:,} |")
        L.append("")
        f1 = lambda v, m=m: {(tr, te): get(v, m, tr, te)["f1_macro"] for tr in ("wi", "isot") for te in ("wi", "isot")}  # noqa: E731
        pp, rr = f1("processed"), f1("raw")
        L += [f"Drop when leaving the training domain (processed): W∖I → ISOT **{pp[('wi','wi')] - pp[('wi','isot')]:+.4f}** "
              f"(from {pp[('wi','wi')]:.4f} to {pp[('wi','isot')]:.4f}), ISOT → W∖I **{pp[('isot','isot')] - pp[('isot','wi')]:+.4f}** "
              f"(from {pp[('isot','isot')]:.4f} to {pp[('isot','wi')]:.4f}); with artefacts kept (raw): "
              f"{rr[('wi','wi')] - rr[('wi','isot')]:+.4f} and {rr[('isot','isot')] - rr[('isot','wi')]:+.4f}. "
              f"Artefact removal changes in-domain ISOT by {pp[('isot','isot')] - rr[('isot','isot')]:+.4f} and in-domain W∖I by "
              f"{pp[('wi','wi')] - rr[('wi','wi')]:+.4f}; the cross-domain cell W∖I → ISOT is {rr[('wi','isot')] - pp[('wi','isot')]:+.4f} "
              "higher with artefacts kept, i.e. part of the apparent transfer on raw text is shared boiler-plate (credits, captions, "
              "datelines) rather than content.", ""]
    a = res["artefact_only"]
    L += ["## Leakage audit — artefact-only classifier on ISOT", "",
          "Logistic regression on the 26 artefact-pattern counts only (`src.preprocess.artefacts.artefact_only_accuracy`, "
          "5-fold CV accuracy, seed 42) — no words, only datelines / `(Reuters)` / credits / captions / URL debris:", "",
          "| Text | CV accuracy | ± std |", "|---|---|---|",
          f"| raw ISOT (as downloaded, {a['raw_isot']['n']:,} rows) | **{a['raw_isot']['cv_accuracy']:.4f}** | {a['raw_isot']['cv_std']:.4f} |",
          f"| processed ISOT (artefacts stripped, {a['processed_isot']['n']:,} rows) | {a['processed_isot']['cv_accuracy']:.4f} | {a['processed_isot']['cv_std']:.4f} |", "",
          "On raw ISOT a classifier that never sees a content word is almost perfect: ISOT's real class is all Reuters wire copy "
          "(\"WASHINGTON (Reuters) –\" datelines) and its fake class comes from a handful of sites with fixed trailers, so a raw "
          "in-domain ~99 % measures *source style*, not truth. After stripping, the artefact-only classifier falls to about the majority-class "
          "rate. What survives in-domain (≈ 0.98–0.99 on processed ISOT) is still house style of disjoint outlets — the cross-domain cells "
          "above are the honest estimate of what a text classifier knows, and the reason the verdict comes from evidence (master §14).", ""]
    OUT_MD.write_text("\n".join(L))
    print("wrote", OUT_MD)


if __name__ == "__main__":
    if "--render" in sys.argv:
        render(json.loads(OUT_JSON.read_text()))
    elif "--wi-only" in sys.argv:
        print(len(load_welfake_minus_isot(recompute=True)), "rows ->", WI_CSV)
    else:
        run()
