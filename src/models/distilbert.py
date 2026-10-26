"""Fine-tuned DistilBERT content classifier on WELFake (master doc §12.2-12.3, MSE2 §3-4).

Recipe (``docs/model_identification.md`` §3): ``distilbert-base-uncased``, input = ``title [SEP] text`` (sentence pair,
head-truncated to ``max_len``), AdamW, linear schedule, warmup ratio 0.1, weight decay 0.01, fp16, 3 epochs with the
**best epoch by val macro-F1** kept, seed 42.  Batch 16 at max_len 256, batch 8 x grad-accum 2 at 512.

Splits are never re-made here: rows come from ``data/splits/*.csv`` via :func:`src.models.baselines.load_split`.
Tokenisation is lazy (in the collator, per batch) so nothing but the raw text frame sits in RAM.

CLI (``make train-distilbert`` runs all four steps in order; every step resumes from what is already on disk)
---
``python -m src.models.distilbert --sweep``     6 configs on a 20k stratified train subset -> train_logs/distilbert_sweep_*.json
``python -m src.models.distilbert --full``      sweep winner on the full train split -> data/models/distilbert-welfake-v1/
``python -m src.models.distilbert --eval-test`` final model on WELFake test + ISOT test -> train_logs/distilbert_welfake_v1_test.json
``python -m src.models.distilbert --render``    docs/results/distilbert_sweep.md + docs/figures/{train,sweep}_distilbert*.png
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.common.seed import set_seed
from src.config import FIGURES_DIR, MODELS_DIR, RESULTS_DIR, ROOT, SEED, SPLITS_DIR
from src.models.baselines import load_split

CHECKPOINT = "distilbert-base-uncased"
LABELS = ["real", "fake"]                      # id 0 = real, 1 = fake (WELFake convention; "fake" = positive class)
GRID_LR = [2e-5, 3e-5, 5e-5]
GRID_MAXLEN = [256, 512]
SUBSET_N = 20_000
EPOCHS = 3
LOG_DIR = RESULTS_DIR / "train_logs"
FINAL_DIR = MODELS_DIR / "distilbert-welfake-v1"
FINAL_LOG = LOG_DIR / "distilbert_welfake_v1_full.json"
TEST_LOG = LOG_DIR / "distilbert_welfake_v1_test.json"
SWEEP_MD = RESULTS_DIR / "distilbert_sweep.md"


def batch_plan(max_len: int) -> tuple[int, int]:
    """(per-device batch, grad-accum) -> effective batch 16 on the 4 GB card."""
    return (16, 1) if max_len <= 256 else (8, 2)


def hardware() -> dict:
    import transformers
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {"gpu": gpu, "gpu_mem_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2) if gpu else None,
            "cpu": platform.processor() or platform.machine(), "python": platform.python_version(),
            "torch": torch.__version__, "transformers": transformers.__version__}


# ---------------------------------------------------------------------------------------------------
# data: raw strings in a Dataset, tokenised per batch in the collator
# ---------------------------------------------------------------------------------------------------
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, titles=None):
        self.texts, self.labels = list(texts), [int(y) for y in labels]
        self.titles = None if titles is None else list(titles)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return {"title": None if self.titles is None else self.titles[i], "text": self.texts[i], "label": self.labels[i]}


class Collator:
    """Pairs (title, text) when titles exist, single sequence otherwise; dynamic padding, head truncation."""

    def __init__(self, tokenizer, max_len: int):
        self.tok, self.max_len = tokenizer, max_len

    def __call__(self, batch):
        texts = [b["text"] for b in batch]
        args = (texts,) if batch[0]["title"] is None else ([b["title"] for b in batch], texts)
        enc = self.tok(*args, truncation="longest_first", max_length=self.max_len, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor([b["label"] for b in batch])
        return enc


def welfake_dataset(df: pd.DataFrame) -> TextDataset:
    return TextDataset(df["text"].fillna("").astype(str), (df["label"] == "fake").astype(int),
                       titles=df["title"].fillna("").astype(str).str.strip())


def stratified_subset(df: pd.DataFrame, n: int = SUBSET_N, seed: int = SEED) -> pd.DataFrame:
    sub, _ = train_test_split(df, train_size=n, stratify=df["label"], random_state=seed)
    return sub.reset_index(drop=True)


# ---------------------------------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------------------------------
def make_metrics(num_labels: int):
    def compute(pred):
        logits, y = pred.predictions, pred.label_ids
        yhat = logits.argmax(-1)
        out = {"macro_f1": f1_score(y, yhat, average="macro"), "accuracy": accuracy_score(y, yhat)}
        if num_labels == 2:
            p = torch.softmax(torch.tensor(logits, dtype=torch.float32), -1)[:, 1].numpy()
            out["roc_auc"] = roc_auc_score(y, p)
        return out
    return compute


def fit(train_ds, val_ds, *, lr: float, max_len: int, epochs: int, num_labels: int = 2, id2label: dict | None = None,
        out_dir: Path | None = None, class_weights: list[float] | None = None, **extra) -> tuple[object, dict]:
    """One fine-tuning run; evaluates every epoch on val and (if ``out_dir``) keeps only the best epoch's weights."""
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    set_seed(SEED)
    batch, accum = batch_plan(max_len)
    batch, accum = extra.pop("batch", batch), extra.pop("accum", accum)
    tok = AutoTokenizer.from_pretrained(CHECKPOINT)
    id2label = id2label or dict(enumerate(LABELS))
    model = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT, num_labels=num_labels, id2label=id2label, label2id={v: k for k, v in id2label.items()})
    scratch = out_dir or Path(os.environ.get("TMPDIR", "/tmp")) / "distilbert_run"
    args = TrainingArguments(
        output_dir=str(scratch), learning_rate=lr, num_train_epochs=epochs, per_device_train_batch_size=batch,
        gradient_accumulation_steps=accum, per_device_eval_batch_size=64 if max_len <= 256 else 32,
        warmup_ratio=0.1, weight_decay=0.01, lr_scheduler_type="linear", fp16=torch.cuda.is_available(), seed=SEED,
        eval_strategy=extra.pop("eval_strategy", "epoch"), logging_strategy="steps", logging_steps=50, report_to=[],
        save_strategy="no",  # SaveBest below writes only the best epoch's weights: no checkpoint-*/optimizer copies (disk)
        dataloader_num_workers=2, remove_unused_columns=False, disable_tqdm=False, **extra)

    trainer_cls = Trainer
    if class_weights is not None:
        w = torch.tensor(class_weights, dtype=torch.float32)

        class _Weighted(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kw):
                labels = inputs.pop("labels")
                out = model(**inputs)
                loss = torch.nn.functional.cross_entropy(out.logits.float(), labels, weight=w.to(out.logits.device))
                return (loss, out) if return_outputs else loss
        trainer_cls = _Weighted

    class SaveBest(TrainerCallback):
        best = -1.0

        def on_evaluate(self, args, state, control, metrics=None, model=None, **kw):
            if out_dir and metrics["eval_macro_f1"] > self.best:
                self.best = metrics["eval_macro_f1"]
                model.save_pretrained(str(out_dir))

    trainer = trainer_cls(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                          data_collator=Collator(tok, max_len), compute_metrics=make_metrics(num_labels),
                          callbacks=[SaveBest()])
    trainer.tokenizer_ = tok
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    wall = time.time() - t0
    if out_dir:  # hand back the best epoch, not the last one
        trainer.model = AutoModelForSequenceClassification.from_pretrained(out_dir).to(trainer.args.device)
        tok.save_pretrained(str(out_dir))

    hist = trainer.state.log_history
    evals = [h for h in hist if "eval_macro_f1" in h]
    per_epoch = []
    for e in evals:
        ep = round(e["epoch"])
        losses = [h["loss"] for h in hist if "loss" in h and ep - 1 < h["epoch"] <= ep + 1e-9]
        per_epoch.append({"epoch": ep, "train_loss": float(np.mean(losses)) if losses else None,
                          "val_loss": e["eval_loss"], "val_macro_f1": e["eval_macro_f1"],
                          "val_accuracy": e["eval_accuracy"], "val_roc_auc": e.get("eval_roc_auc")})
    best = max(per_epoch, key=lambda r: r["val_macro_f1"], default={"epoch": None, "val_macro_f1": None})
    record = {
        "checkpoint": CHECKPOINT, "seed": SEED,
        "config": {"lr": lr, "max_len": max_len, "epochs": epochs, "per_device_batch": batch, "grad_accum": accum,
                   "effective_batch": batch * accum, "warmup_ratio": 0.1, "weight_decay": 0.01, "scheduler": "linear",
                   "fp16": args.fp16, "optimizer": "adamw_torch", "class_weights": class_weights,
                   "input": "title [SEP] text (pair), head truncation" if train_ds.titles is not None else "text",
                   **{k: v for k, v in extra.items()}},
        "n_train": len(train_ds), "n_val": len(val_ds), "hardware": hardware(),
        "wall_clock_s": round(wall, 1), "minutes": round(wall / 60, 1),
        "peak_gpu_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None,
        "per_epoch": per_epoch, "best_epoch": best["epoch"], "best_val_macro_f1": best["val_macro_f1"],
        "log_history": hist,
    }
    return trainer, record


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float))


def sweep_log(lr: float, max_len: int) -> Path:
    return LOG_DIR / f"distilbert_sweep_lr{lr:.0e}_len{max_len}.json"


def run_sweep() -> None:
    train = stratified_subset(load_split("welfake", "train"))
    val = load_split("welfake", "val")
    assert not set(train["id"]) & set(val["id"])
    tr, va = welfake_dataset(train), welfake_dataset(val)
    for max_len in GRID_MAXLEN:
        for lr in GRID_LR:
            path = sweep_log(lr, max_len)
            if path.exists():
                print(f"skip {path.name} (exists)")
                continue
            print(f"=== sweep lr={lr} max_len={max_len} ===", flush=True)
            trainer, rec = fit(tr, va, lr=lr, max_len=max_len, epochs=EPOCHS)
            rec.update(name=path.stem, dataset="welfake", train_subset=f"{SUBSET_N} stratified (seed {SEED})")
            save_json(rec, path)
            print(f"-> best epoch {rec['best_epoch']} val macro-F1 {rec['best_val_macro_f1']:.4f} in {rec['minutes']} min")
            del trainer
            torch.cuda.empty_cache()


def sweep_records() -> list[dict]:
    recs = [json.loads(sweep_log(lr, m).read_text()) for m in GRID_MAXLEN for lr in GRID_LR if sweep_log(lr, m).exists()]
    return recs


def winner() -> dict:
    recs = sweep_records()
    if len(recs) < len(GRID_LR) * len(GRID_MAXLEN):
        raise SystemExit(f"sweep incomplete ({len(recs)}/6 runs) - run --sweep first")
    return max(recs, key=lambda r: r["best_val_macro_f1"])


def run_full(max_len: int | None = None) -> None:
    w = winner()
    lr, ml = w["config"]["lr"], max_len or w["config"]["max_len"]
    train, val = load_split("welfake", "train"), load_split("welfake", "val")
    test_ids = set(pd.read_csv(SPLITS_DIR / "welfake_test.csv")["id"])
    assert not set(train["id"]) & set(val["id"]) and not (set(train["id"]) | set(val["id"])) & test_ids
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    trainer, rec = fit(welfake_dataset(train), welfake_dataset(val), lr=lr, max_len=ml, epochs=EPOCHS, out_dir=FINAL_DIR)
    trainer.state.save_to_json(str(FINAL_DIR / "trainer_state.json"))
    (FINAL_DIR / "training_args.json").write_text(trainer.args.to_json_string())
    rec.update(name="distilbert_welfake_v1_full", dataset="welfake", train_subset="full train split",
               sweep_winner=w["name"], winner_equals_sweep=(ml == w["config"]["max_len"]), model_dir=str(FINAL_DIR))
    save_json(rec, FINAL_LOG)
    print(f"full retrain: best epoch {rec['best_epoch']} val macro-F1 {rec['best_val_macro_f1']:.4f} ({rec['minutes']} min)")


# ---------------------------------------------------------------------------------------------------
# inference / test evaluation (consumed by src/eval/eval_models.py)
# ---------------------------------------------------------------------------------------------------
def load_model(model_dir: Path = FINAL_DIR, device: str | None = None):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    return tok, model.to(device or ("cuda" if torch.cuda.is_available() else "cpu"))


@torch.no_grad()
def predict_proba(df: pd.DataFrame, tok, model, max_len: int, batch: int = 64) -> np.ndarray:
    """P(fake) per row of a frame with ``title``/``text`` columns."""
    ds = welfake_dataset(df.assign(label="real"))
    coll, out = Collator(tok, max_len), []
    for i in range(0, len(ds), batch):
        enc = coll([ds[j] for j in range(i, min(i + batch, len(ds)))])
        enc.pop("labels")
        with torch.autocast(model.device.type, enabled=model.device.type == "cuda"):
            logits = model(**{k: v.to(model.device) for k, v in enc.items()}).logits
        out.append(torch.softmax(logits.float(), -1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def binary_metrics(y_true: np.ndarray, p_fake: np.ndarray) -> dict:
    yhat = (p_fake >= 0.5).astype(int)
    p_m, r_m, f_m, _ = precision_recall_fscore_support(y_true, yhat, average="macro", zero_division=0)
    p_f, r_f, f_f, _ = precision_recall_fscore_support(y_true, yhat, average="binary", pos_label=1, zero_division=0)
    return {"n": int(len(y_true)), "accuracy": accuracy_score(y_true, yhat), "precision_macro": p_m,
            "recall_macro": r_m, "f1_macro": f_m, "precision_fake": p_f, "recall_fake": r_f, "f1_fake": f_f,
            "roc_auc": roc_auc_score(y_true, p_fake), "labels": LABELS,
            "confusion": confusion_matrix(y_true, yhat, labels=[0, 1]).tolist()}


def run_eval_test() -> None:
    max_len = json.loads(FINAL_LOG.read_text())["config"]["max_len"]
    tok, model = load_model()
    res = {"model_dir": str(FINAL_DIR), "max_len": max_len, "threshold": 0.5, "hardware": hardware(), "results": {}}
    for ds in ["welfake", "isot"]:
        df = load_split(ds, "test")
        t0 = time.time()
        p = predict_proba(df, tok, model, max_len)
        y = (df["label"] == "fake").astype(int).to_numpy()
        res["results"][ds] = {**binary_metrics(y, p), "predict_seconds": round(time.time() - t0, 1),
                              "preds_csv": str((LOG_DIR / f"distilbert_welfake_v1_test_preds_{ds}.csv").relative_to(ROOT))}
        pd.DataFrame({"id": df["id"], "label": df["label"], "p_fake": p.round(6)}).to_csv(
            LOG_DIR / f"distilbert_welfake_v1_test_preds_{ds}.csv", index=False)
        print(ds, {k: round(v, 4) for k, v in res["results"][ds].items() if isinstance(v, float)})
    res["note"] = ("ISOT test is NOT a cross-dataset result: ISOT is contained in WELFake (~80% of ISOT test rows occur in "
                   "WELFake train), so this row is in-domain/contaminated; see docs/model_identification.md §4.2.")
    save_json(res, TEST_LOG)


# ---------------------------------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------------------------------
LR5E5_NOTE = ("at max_len 256 it fits the 20k subset fastest (train loss 0.0077 at epoch 3 vs 0.0153 / 0.0105 for 2e-5 / "
              "3e-5) and its val loss rises from epoch 2 (0.0572) to epoch 3 (0.0671), i.e. it starts to overfit, while val "
              "macro-F1 stays flat (0.9870 -> 0.9871); it ties lr 2e-5 (0.9869) within 0.0002 — no lr is clearly better at 256.")
NOT_RUN = "not run (stopped on user decision, 2026-10-26) — deferred to ESE"
# Shown while no test log exists. Derived only from our own measured numbers; never a measurement.
EXPECTED_NOTE = (
    "**Expected — not measured** (†expected, not measured; to be replaced by the ESE run). "
    "Full-train WELFake **test** macro-F1 of the retrained winner: *≈ 0.98–0.99*† — basis: the three completed "
    "20k-subset runs reach val macro-F1 0.9863–0.9871, and the WELFake val–test gaps of the three tuned classical "
    "models are within ±0.003 (`classifier_table.md`); training on all 48,639 rows is not expected to lower it. "
    "On that basis O1 (≥ 0.95) is *expected* to be met† — it is still **not checked**. "
    "ISOT test (in-domain, ISOT ⊂ WELFake, ~80 % of ISOT test rows are in WELFake train): *expected high*†, no numeric "
    "range. Cross-dataset WELFake∖ISOT ↔ ISOT: *expected to drop sharply, like the classical models*† (measured SVM test "
    "macro-F1 0.6611 W∖I→ISOT and 0.5480 ISOT→W∖I, `cross_dataset.md`) — direction only, no number.")


def render() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs = sweep_records()
    w = max(recs, key=lambda r: r["best_val_macro_f1"]) if recs else None
    full = json.loads(FINAL_LOG.read_text()) if FINAL_LOG.exists() else None
    rows = ["| run | lr | max_len | batch × accum | best epoch / 3 | warmup | seed | val macro-F1 (best ep.) | val acc | per-epoch val macro-F1 | minutes | peak GPU GiB | hardware |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]

    def row(r, label):
        c = r["config"]
        ep = r["per_epoch"][r["best_epoch"] - 1]
        curve = " → ".join(f"{e['val_macro_f1']:.4f}" for e in r["per_epoch"])
        return (f"| {label} | {c['lr']:.0e} | {c['max_len']} | {c['per_device_batch']} × {c['grad_accum']} | {r['best_epoch']} | "
                f"{c['warmup_ratio']} | {r['seed']} | **{r['best_val_macro_f1']:.4f}** | {ep['val_accuracy']:.4f} | {curve} | "
                f"{r['minutes']} | {r['peak_gpu_mem_gib']} | {r['hardware']['gpu'] or 'CPU'} |")
    for r in recs:
        rows.append(row(r, ("**best completed** " if r is w else "") + f"sweep 20k ({r['n_train']})"))
    done = {(r["config"]["lr"], r["config"]["max_len"]) for r in recs}
    for ml in GRID_MAXLEN:
        for lr in GRID_LR:
            if (lr, ml) not in done:
                rows.append(f"| sweep 20k | {lr:.0e} | {ml} | {'16 × 1' if ml <= 256 else '8 × 2'} | {NOT_RUN} |  |  |  |  |  |  |  |  |")
    if full:
        rows.append(row(full, f"**full train** ({full['n_train']})"))
    else:
        rows.append(f"| full-train retrain of the winner (48,639) | — | — | — | {NOT_RUN} |  |  |  |  |  |  |  |  |")
    md = ["# DistilBERT sweep — WELFake (generated by `python -m src.models.distilbert --render`)", "",
          f"`{CHECKPOINT}`, input = title [SEP] text (head truncation), AdamW + linear schedule, warmup ratio 0.1, weight decay 0.01, "
          f"fp16, {EPOCHS} epochs with the best epoch (val macro-F1) kept, seed {SEED}. Sweep = 20,000-row stratified subset of "
          "WELFake train, evaluated on the full WELFake val split (6,080). Selection metric: val macro-F1. "
          "Per-run logs: `docs/results/train_logs/distilbert_sweep_*.json`.", "", *rows, ""]
    if w:
        md += [f"**Best completed run:** lr = {w['config']['lr']:.0e}, max_len = {w['config']['max_len']} "
               f"(val macro-F1 {w['best_val_macro_f1']:.4f} at epoch {w['best_epoch']}). "
               + ("" if len(recs) == len(GRID_LR) * len(GRID_MAXLEN) else
                  f"Only {len(recs)} of {len(GRID_LR) * len(GRID_MAXLEN)} configs ran, so this is the best of the completed "
                  "runs, not a sweep winner; the sweep runs save no weights, so **no DistilBERT checkpoint exists** and no "
                  "DistilBERT test number exists (`data/models/distilbert-welfake-v1/` was not created; the served classifier "
                  "stays TF-IDF). MSE2 objective O1 (DistilBERT macro-F1 ≥ 0.95 on WELFake test) is therefore not checked.")]
    if full:
        same = "equals" if full.get("winner_equals_sweep") else "**differs from**"
        md += [f"The final model `data/models/distilbert-welfake-v1/` was retrained on the full train split with this config "
               f"(sweep winner `{full['sweep_winner']}`; the full-train config {same} the winner): best epoch "
               f"{full['best_epoch']}, val macro-F1 {full['best_val_macro_f1']:.4f}, {full['minutes']} min."]
    md += ["", f"**lr = 5e-5:** {LR5E5_NOTE}", ""]
    if not TEST_LOG.exists():
        md += [EXPECTED_NOTE, ""]
    if TEST_LOG.exists():
        t = json.loads(TEST_LOG.read_text())["results"]
        md += ["## Held-out test (final model, threshold 0.5) — `train_logs/distilbert_welfake_v1_test.json`", "",
               "| test split | n | accuracy | macro-F1 | P_fake | R_fake | ROC-AUC |", "|---|---|---|---|---|---|---|"]
        md += [f"| {d} | {m['n']} | {m['accuracy']:.4f} | {m['f1_macro']:.4f} | {m['precision_fake']:.4f} | "
               f"{m['recall_fake']:.4f} | {m['roc_auc']:.4f} |" for d, m in t.items()]
        md += ["", "ISOT test is in-domain, not cross-dataset (ISOT ⊂ WELFake; ~80 % of ISOT test rows are in WELFake train)."]
    SWEEP_MD.write_text("\n".join(md) + "\n")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if recs:
        fig, ax = plt.subplots(figsize=(6, 4))
        for ml, mk in zip(GRID_MAXLEN, "os", strict=True):
            pts = sorted((r["config"]["lr"], r["best_val_macro_f1"]) for r in recs if r["config"]["max_len"] == ml)
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=mk, label=f"max_len {ml}")
        ax.set_xscale("log")
        ax.set_xticks(GRID_LR)
        ax.set_xticklabels([f"{x:.0e}" for x in GRID_LR])
        ax.set_xlabel("learning rate")
        ax.set_ylabel("val macro-F1 (best of 3 epochs)")
        ax.set_title("DistilBERT sweep on WELFake (20k subset)")
        ax.grid(alpha=.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "sweep_distilbert_lr.png", dpi=150)
        plt.close(fig)
    if full:
        plot_curve(full, FIGURES_DIR / "train_distilbert.png", "DistilBERT on WELFake (full train)")
    elif w:
        plot_curve(w, FIGURES_DIR / "train_distilbert.png", "DistilBERT on WELFake — best sweep run (20k subset, no full retrain)")
    print("wrote", SWEEP_MD)


def plot_curve(rec: dict, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    steps = [(h["epoch"], h["loss"]) for h in rec["log_history"] if "loss" in h]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([s[0] for s in steps], [s[1] for s in steps], color="tab:blue", alpha=.5, lw=1, label="train loss (every 50 steps)")
    ax.plot([e["epoch"] for e in rec["per_epoch"]], [e["train_loss"] for e in rec["per_epoch"]], "o-", color="tab:blue",
            label="train loss (epoch mean)")
    ax.plot([e["epoch"] for e in rec["per_epoch"]], [e["val_loss"] for e in rec["per_epoch"]], "s--", color="tab:cyan",
            label="val loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax2 = ax.twinx()
    ax2.plot([e["epoch"] for e in rec["per_epoch"]], [e["val_macro_f1"] for e in rec["per_epoch"]], "D-", color="tab:red",
             label="val macro-F1")
    b = rec["per_epoch"][rec["best_epoch"] - 1]
    ax2.scatter([b["epoch"]], [b["val_macro_f1"]], s=220, facecolors="none", edgecolors="black", zorder=5,
                label=f"best epoch {b['epoch']} ({b['val_macro_f1']:.4f})")
    ax2.set_ylabel("val macro-F1", color="tab:red")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right", fontsize=8)
    c = rec["config"]
    ax.set_title(f"{title}\nlr {c['lr']:.0e}, max_len {c['max_len']}, eff. batch {c['effective_batch']}, fp16, seed {rec['seed']}")
    ax.set_xticks(range(1, c["epochs"] + 1))
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--max-len", type=int, help="override the winner's max_len for --full (documented trade-off)")
    ap.add_argument("--eval-test", action="store_true")
    ap.add_argument("--render", action="store_true")
    a = ap.parse_args(argv)
    if not any([a.sweep, a.full, a.eval_test, a.render]):
        ap.error("choose at least one of --sweep/--full/--eval-test/--render")
    if a.sweep:
        run_sweep()
    if a.full:
        run_full(a.max_len)
    if a.eval_test:
        run_eval_test()
    if a.render:
        render()


if __name__ == "__main__":
    sys.exit(main())
