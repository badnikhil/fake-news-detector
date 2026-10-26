"""LIAR 3-way head (TRUE / MIXED / FALSE) — DistilBERT on the statement text only (master §12, MSE2 §3).

Same recipe as :mod:`src.models.distilbert` (fp16, warmup 0.1, wd 0.01, seed 42, best epoch by val macro-F1) with
class-weighted cross-entropy (sklearn "balanced" weights from the train counts), max_len 64, batch 16, on the LIAR
**official** splits.  Speaker / party never enter the model (label proxies, EDA).  Sweep lr {2e-5, 3e-5} × epochs {3, 5};
the winner's best-epoch weights go to ``data/models/distilbert-liar3-v1/``.

For the Wang (2017) comparison (his numbers are 6-way) the winning config is also trained once 6-way; its predictions
are reported on 6-way gold and, collapsed 6→3 with ``LIAR_TO_3``, next to the 3-way head ("mapped 6-way").

``python -m src.models.liar_head`` (= ``make train-liar``); ``--render`` rewrites the md/figure from the logs.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.config import FIGURES_DIR, LIAR_LABELS, LIAR_TO_3, MODELS_DIR, PROCESSED_DIR, RESULTS_DIR, SEED
from src.models.baselines import load_split
from src.models.distilbert import LOG_DIR, TextDataset, fit, plot_curve, save_json

LABELS3 = ["TRUE", "MIXED", "FALSE"]
MAX_LEN = 64
GRID = [(lr, ep) for lr in (2e-5, 3e-5) for ep in (3, 5)]
FINAL_DIR = MODELS_DIR / "distilbert-liar3-v1"
TMP_DIR = MODELS_DIR / "_liar_tmp"
LOG = LOG_DIR / "distilbert_liar3_v1.json"
MD = RESULTS_DIR / "liar_head.md"
# Wang (2017) Table 2, 6-way test accuracy (docs/literature_table.md row 1)
WANG = {"majority": 0.208, "text-only CNN": 0.270, "hybrid CNN (text + all metadata)": 0.274}


def splits() -> dict[str, pd.DataFrame]:
    df = pd.read_parquet(PROCESSED_DIR / "liar.parquet", columns=["id", "title", "text", "label", "label3", "split"])
    out = {s: load_split("liar", s, df=df.drop(columns=["label3"])).merge(df[["id", "label3"]], on="id") for s in
           ("train", "val", "test")}
    assert not set(out["train"]["id"]) & set(out["test"]["id"]) and not set(out["train"]["id"]) & set(out["val"]["id"])
    return out


def ds(df: pd.DataFrame, labels: list[str], col: str) -> TextDataset:
    return TextDataset(df["text"].astype(str), df[col].map({lab: i for i, lab in enumerate(labels)}))  # statement only


def balanced_weights(y: pd.Series, labels: list[str]) -> list[float]:
    counts = y.value_counts()
    return [round(len(y) / (len(labels) * counts[lab]), 4) for lab in labels]


def predict(trainer, dataset) -> np.ndarray:
    return trainer.predict(dataset).predictions.argmax(-1)


def scores(y, yhat) -> dict:
    return {"accuracy": accuracy_score(y, yhat), "macro_f1": f1_score(y, yhat, average="macro")}


def run() -> None:
    sp = splits()
    w3 = balanced_weights(sp["train"]["label3"], LABELS3)
    tr, va = ds(sp["train"], LABELS3, "label3"), ds(sp["val"], LABELS3, "label3")
    runs, best = [], None
    for lr, ep in GRID:
        out = TMP_DIR / f"lr{lr:.0e}_ep{ep}"
        print(f"=== LIAR 3-way lr={lr} epochs={ep} ===", flush=True)
        trainer, rec = fit(tr, va, lr=lr, max_len=MAX_LEN, epochs=ep, num_labels=3, id2label=dict(enumerate(LABELS3)),
                           out_dir=out, class_weights=w3, batch=16, accum=1)
        rec.update(name=f"liar3_lr{lr:.0e}_ep{ep}")
        if best is None or rec["best_val_macro_f1"] > best[1]["best_val_macro_f1"]:
            if best is not None:
                shutil.rmtree(best[2], ignore_errors=True)
            trainer.state.save_to_json(str(out / "trainer_state.json"))
            (out / "training_args.json").write_text(trainer.args.to_json_string())
            best = (trainer, rec, out)
        else:
            shutil.rmtree(out, ignore_errors=True)
        runs.append({k: v for k, v in rec.items() if k != "log_history"})
        print(f"-> best epoch {rec['best_epoch']} val macro-F1 {rec['best_val_macro_f1']:.4f} ({rec['minutes']} min)")
        torch.cuda.empty_cache()

    trainer, rec, out = best
    te3 = ds(sp["test"], LABELS3, "label3")
    yhat3 = predict(trainer, te3)
    test3 = {**scores(te3.labels, yhat3), "confusion": confusion_matrix(te3.labels, yhat3, labels=[0, 1, 2]).tolist()}
    val3 = scores(va.labels, predict(trainer, va))
    del trainer
    shutil.rmtree(FINAL_DIR, ignore_errors=True)
    shutil.move(str(out), str(FINAL_DIR))
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    torch.cuda.empty_cache()

    # 6-way twin with the winning config (metrics only, weights not kept) for the Wang (2017) comparison
    lr, ep = rec["config"]["lr"], rec["config"]["epochs"]
    w6 = balanced_weights(sp["train"]["label"], LIAR_LABELS)
    print(f"=== LIAR 6-way twin lr={lr} epochs={ep} ===", flush=True)
    t6, rec6 = fit(ds(sp["train"], LIAR_LABELS, "label"), ds(sp["val"], LIAR_LABELS, "label"), lr=lr, max_len=MAX_LEN,
                   epochs=ep, num_labels=6, id2label=dict(enumerate(LIAR_LABELS)), out_dir=TMP_DIR, class_weights=w6,
                   batch=16, accum=1)
    te6 = ds(sp["test"], LIAR_LABELS, "label")
    yhat6 = predict(t6, te6)
    mapped = [LABELS3.index(LIAR_TO_3[LIAR_LABELS[i]]) for i in yhat6]
    test6 = {**scores(te6.labels, yhat6), "confusion": confusion_matrix(te6.labels, yhat6, labels=range(6)).tolist()}
    del t6
    shutil.rmtree(TMP_DIR, ignore_errors=True)

    majority6 = sp["train"]["label"].value_counts().idxmax()
    majority3 = sp["train"]["label3"].value_counts().idxmax()
    log = {**rec, "name": "distilbert_liar3_v1", "dataset": "liar (official splits)", "model_dir": str(FINAL_DIR),
           "labels": LABELS3, "sweep": runs, "val_3way": val3, "test_3way": test3,
           "six_way_twin": {k: v for k, v in rec6.items() if k != "log_history"}, "test_6way": test6,
           "test_6way_mapped_to_3": scores(te3.labels, mapped),
           "majority_test": {"6way": float((sp["test"]["label"] == majority6).mean()),
                             "3way": float((sp["test"]["label3"] == majority3).mean())},
           "wang2017_6way_test_acc": WANG}
    save_json(log, LOG)
    render()


def render() -> None:
    log = json.loads(LOG.read_text())
    rows = ["| lr | epochs | best epoch | val macro-F1 | val acc | minutes |", "|---|---|---|---|---|---|"]
    for r in log["sweep"]:
        e = r["per_epoch"][r["best_epoch"] - 1]
        star = " **(winner)**" if r["name"] == f"liar3_lr{log['config']['lr']:.0e}_ep{log['config']['epochs']}" else ""
        rows.append(f"| {r['config']['lr']:.0e}{star} | {r['config']['epochs']} | {r['best_epoch']} | "
                    f"{r['best_val_macro_f1']:.4f} | {e['val_accuracy']:.4f} | {r['minutes']} |")
    t3, t6, m = log["test_3way"], log["test_6way"], log["test_6way_mapped_to_3"]
    c = log["config"]
    md = ["# LIAR 3-way head — DistilBERT (generated by `python -m src.models.liar_head --render`)", "",
          f"`distilbert-base-uncased`, statement text only (no speaker/party), max_len {MAX_LEN}, batch 16, fp16, warmup 0.1, "
          f"wd 0.01, seed {SEED}, class-weighted CE (balanced, train counts: {dict(zip(LABELS3, c['class_weights'], strict=True))}), "
          "best epoch by val macro-F1, LIAR official splits (train 10,269 / val 1,284 / test 1,283; 5 train∩val and 4 "
          "train∩test duplicate statements kept, disclosed). Model: `data/models/distilbert-liar3-v1/`; log: "
          "`docs/results/train_logs/distilbert_liar3_v1.json`.", "", "## Sweep (val)", "", *rows, "",
          "## Test (official LIAR test, 1,283 statements)", "",
          "| model | label space | test accuracy | test macro-F1 |", "|---|---|---|---|",
          f"| majority class (train) | 3-way | {log['majority_test']['3way']:.4f} | — |",
          f"| **DistilBERT 3-way head (ours)** | 3-way | **{t3['accuracy']:.4f}** | **{t3['macro_f1']:.4f}** |",
          f"| DistilBERT 6-way twin, predictions collapsed 6→3 | 3-way | {m['accuracy']:.4f} | {m['macro_f1']:.4f} |",
          f"| majority class (train) | 6-way | {log['majority_test']['6way']:.4f} | — |",
          f"| DistilBERT 6-way twin (same config, ours) | 6-way | {t6['accuracy']:.4f} | {t6['macro_f1']:.4f} |",
          *[f"| Wang (2017) {k} | 6-way | {v:.3f} | n/r |" for k, v in log["wang2017_6way_test_acc"].items()], "",
          "Wang (2017) reports only 6-way accuracy (Table 2), so the 6-way twin is the like-for-like comparison; the 3-way "
          "head is what fusion consumes. 3-way confusion (rows = true TRUE/MIXED/FALSE, cols = predicted): "
          f"`{t3['confusion']}`.", ""]
    MD.write_text("\n".join(md) + "\n")
    plot_curve(log, FIGURES_DIR / "train_liar.png", "DistilBERT LIAR 3-way head")
    print("wrote", MD)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--render", action="store_true", help="only rewrite liar_head.md + train_liar.png from the log")
    a = ap.parse_args(argv)
    render() if a.render else run()


if __name__ == "__main__":
    sys.exit(main())
