"""Stance evaluation on the 3k balanced FEVER dev subset WITH gold sentence text (MSE2 §3, master §13.3 / §10.2).

Sentence text source: `data/raw/fever` stores only evidence pointers (page, sentence id), so the text comes from the
HF dataset **copenlu/fever_gold_evidence** (CC BY-SA 3.0, Atanasova et al., EMNLP 2020; revision a6b8d89), whose
`valid.jsonl` / `test.jsonl` are the FEVER shared-task dev claims with their gold evidence sentences resolved from the
official June-2017 FEVER Wikipedia dump (joined on `original_id` = FEVER claim id). For NOT ENOUGH INFO claims that file
ships sentences retrieved by the Papelo system (Malon, 2018) — i.e. a realistic "related but not sufficient" sentence,
which is the "NEI -> sentence from the same/another page" protocol of master §10.2.

Premise = the first gold evidence set's sentences joined (FEVER PTB tokens such as -LRB- undone); hypothesis = claim.
    python -m src.stance.fever_eval            # -> docs/results/stance_fever.md, docs/figures/cm_stance_fever.png
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import FIGURES_DIR, PROCESSED_DIR, RAW_DIR, RESULTS_DIR, SEED
from src.stance.nli import NLI_MODEL, STANCES, stance_probs

GOLD_DIR = RAW_DIR / "fever_gold"
GOLD_REV = "a6b8d891d393e97a4efac791afffb2d7de5e57c6"
GOLD_URL = f"https://huggingface.co/datasets/copenlu/fever_gold_evidence/resolve/{GOLD_REV}/{{name}}.jsonl"
GOLD_TO_STANCE = {"SUPPORTS": "SUPPORTS", "REFUTES": "REFUTES", "NOT ENOUGH INFO": "NEUTRAL"}
PTB = {"-LRB-": "(", "-RRB-": ")", "-LSB-": "[", "-RSB-": "]", "-LCB-": "{", "-RCB-": "}", "``": '"', "''": '"'}
OUT_MD = RESULTS_DIR / "stance_fever.md"
OUT_FIG = FIGURES_DIR / "cm_stance_fever.png"
PAIRS_PATH = PROCESSED_DIR / "fever_dev_pairs.parquet"
O4_TARGET = 0.85


def download_gold() -> None:
    import requests
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("valid", "test"):
        p = GOLD_DIR / f"{name}.jsonl"
        if not p.exists():
            r = requests.get(GOLD_URL.format(name=name), timeout=120)
            r.raise_for_status()
            p.write_bytes(r.content)


def detok(s: str) -> str:
    for k, v in PTB.items():
        s = s.replace(k, v)
    s = re.sub(r"\s+([,.;:!?%)\]}'])", r"\1", s)
    s = re.sub(r"([(\[{$])\s+", r"\1", s)
    return re.sub(r"\s+'s\b", "'s", s).strip()


def load_gold() -> pd.DataFrame:
    rows = []
    for name in ("valid", "test"):
        with open(GOLD_DIR / f"{name}.jsonl") as f:
            for line in f:
                r = json.loads(line)
                rows.append({"fever_id": int(r["original_id"]), "gold_label": GOLD_TO_STANCE[r["label"]],
                             "premise": detok(" ".join(f"{e[0].replace('_', ' ')}: {e[2]}" for e in r["evidence"])),
                             "premise_plain": detok(" ".join(e[2] for e in r["evidence"])),
                             "pages": "|".join(dict.fromkeys(e[0] for e in r["evidence"]))})
    g = pd.DataFrame(rows)
    g = g[g["premise"].str.len() > 0]
    return g.drop_duplicates("fever_id", keep="first")  # one premise per claim (first evidence set)


def build_pairs(n_per_class: int = 1000) -> tuple[pd.DataFrame, dict]:
    """Our `fever_subset` val split (3k balanced dev claims, `make data`) joined to the gold text. Claims without gold
    text are topped up with other shared-task-dev claims of the same class (seed 42) so the set stays 1k/1k/1k."""
    sub = pd.read_parquet(PROCESSED_DIR / "fever_subset.parquet", columns=["id", "text", "label", "split"])
    sub = sub[sub["split"] == "val"].copy()
    sub["fever_id"] = sub["id"].str.removeprefix("fever_").astype(int)
    gold = load_gold()
    m = sub.merge(gold, on="fever_id", how="left")
    assert (m.dropna(subset=["gold_label"])["gold_label"] == m.dropna(subset=["gold_label"])["label"]).all()
    have = m.dropna(subset=["premise"])
    info = {"subset_rows": len(sub), "with_gold_text": len(have), "topped_up": 0}
    dev = pd.read_json(RAW_DIR / "fever" / "shared_task_dev.jsonl", lines=True)[["id", "claim"]]
    parts = []
    for lab in STANCES:
        got = have[have["label"] == lab]
        need = n_per_class - len(got)
        if need > 0:
            pool = gold[(gold["gold_label"] == lab) & ~gold["fever_id"].isin(sub["fever_id"])]
            pool = pool.merge(dev, left_on="fever_id", right_on="id").rename(columns={"claim": "text"})
            extra = pool.sample(need, random_state=SEED).assign(label=lab, id=lambda d: "fever_" + d["fever_id"].astype(str))
            got = pd.concat([got, extra], ignore_index=True)
            info["topped_up"] += need
        parts.append(got.head(n_per_class))
    pairs = pd.concat(parts, ignore_index=True)[["id", "fever_id", "text", "label", "premise", "premise_plain", "pages"]]
    return pairs.rename(columns={"text": "claim"}).sort_values("id").reset_index(drop=True), info


def evaluate(pairs: pd.DataFrame, batch_size: int = 16, col: str = "premise") -> dict:
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    t0 = time.time()
    probs = stance_probs(list(zip(pairs[col], pairs["claim"], strict=True)), batch_size=batch_size)
    secs = time.time() - t0
    pred = [STANCES[i] for i in probs.argmax(1)]
    y = pairs["label"].tolist()
    return {"accuracy": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro"),
            "report": classification_report(y, pred, labels=STANCES, output_dict=True, zero_division=0),
            "cm": confusion_matrix(y, pred, labels=STANCES).tolist(), "pred": pred, "probs": probs,
            "seconds": secs, "per_pair_ms": 1000 * secs / len(pairs)}


def plot_cm(cm: list[list[int]], path: Path = OUT_FIG) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    a = np.array(cm)
    pct = a / a.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{a[i, j]}\n{pct[i, j]:.1%}", ha="center", va="center",
                    color="white" if pct[i, j] > 0.6 else "black", fontsize=9)
    ax.set_xticks(range(3), STANCES); ax.set_yticks(range(3), STANCES)  # noqa: E702
    ax.set_xlabel("predicted (NLI)"); ax.set_ylabel("gold (FEVER)")  # noqa: E702
    ax.set_title("nli-deberta-v3-small on FEVER dev (n=3,000)", fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def write_md(res: dict, info: dict, n: int, device: str, path: Path = OUT_MD) -> Path:
    rep = res["report"]
    rows = "\n".join(f"| {s} | {rep[s]['precision']:.4f} | {rep[s]['recall']:.4f} | {rep[s]['f1-score']:.4f} | "
                     f"{int(rep[s]['support'])} |" for s in STANCES)
    cm = res["cm"]
    cm_rows = "\n".join(f"| **{s}** | " + " | ".join(str(v) for v in cm[i]) + " |" for i, s in enumerate(STANCES))
    met = "meets" if res["accuracy"] >= O4_TARGET else "does NOT meet"
    acc_sr = (cm[0][0] + cm[1][1]) / (sum(cm[0]) + sum(cm[1]))  # gold SUPPORTS/REFUTES rows only, 3-way prediction
    nei_as_ref = cm[2][1] / sum(cm[2])
    reading = "The O4 target is met." if res["accuracy"] >= O4_TARGET else (
        f"""**The O4 target (≥ {O4_TARGET:.0%}) is not met: {res['accuracy']:.1%}.** The shortfall is almost entirely the
NOT ENOUGH INFO class: on the gold SUPPORTS / REFUTES pairs the model is right {acc_sr:.1%} of the time, but
{nei_as_ref:.1%} of NEI pairs are predicted REFUTES — the retrieved NEI sentences are about the same entity and often
state a *different* fact, which an MNLI-trained model reads as a contradiction. In the pipeline this error mode is
damped by fusion (R3 needs ≥ 2 reliable refuting sources and |S| ≥ τ), but it is the main stance risk. Planned fixes
(ESE, in order of cost): a REFUTES-probability threshold tuned on FEVER *train* pairs (never on this dev set), then the
optional 1-epoch fine-tune on the 20k `fever_subset` train claims.""")
    path.write_text(f"""# Stance detection on FEVER dev (MSE2)

Generated by `python -m src.stance.fever_eval` on {time.strftime('%Y-%m-%d')} — do not edit by hand.

**Model:** `{NLI_MODEL}` (pre-trained, **not fine-tuned by us**); premise = gold evidence, hypothesis = claim;
entailment → SUPPORTS, contradiction → REFUTES, neutral → NEUTRAL (labels read from the checkpoint's `id2label`).
**Device:** {device} · batch 16 · {res['seconds']:.0f} s total ({res['per_pair_ms']:.0f} ms / pair).

## Data

- {n:,} balanced pairs (1,000 per class) = the `fever_subset` **val** split built by `make data` (3k balanced claims
  sampled from FEVER `shared_task_dev.jsonl`, seed 42).
- `data/raw/fever` holds only evidence *pointers*, so the sentence **text** comes from the HF dataset
  `copenlu/fever_gold_evidence` (rev `{GOLD_REV[:7]}`, CC BY-SA 3.0; gold evidence resolved from the June-2017 FEVER
  Wikipedia dump), joined on the FEVER claim id; labels were asserted equal on every joined row.
- {info['with_gold_text']:,} of the {info['subset_rows']:,} subset claims have gold text there; **{info['topped_up']}
  were topped up** with other shared-task-dev claims of the same class (seed 42) to keep the 1k/1k/1k balance.
- SUPPORTS / REFUTES premise = the first gold evidence set, each sentence prefixed with its Wikipedia page title
  (`"Papua: It was formerly called Irian Jaya ..."` — standard FEVER practice, because gold sentences often start with
  a pronoun), sentences joined, PTB tokens such as `-LRB-` undone. The no-title variant is reported as an ablation.
  NOT ENOUGH INFO premise = the sentence(s) that file ships for NEI claims, retrieved by the Papelo system
  (Malon, 2018) — related Wikipedia text that does not settle the claim (master §10.2 "a sentence from the same /
  another page").
- Pairs saved to `data/processed/fever_dev_pairs.parquet` (git-ignored, regenerated by the script).

## Results

| Metric | Value |
|---|---|
| **Accuracy** | **{res['accuracy']:.4f}** |
| Macro-F1 | {res['macro_f1']:.4f} |
| O4 target (accuracy ≥ {O4_TARGET:.2f}) | {met} the target |
| Accuracy on the 2,000 gold SUPPORTS / REFUTES pairs only (3-way prediction) | {acc_sr:.4f} |
| Ablation: premise **without** page-title prefix — accuracy / macro-F1 | {res['plain']['accuracy']:.4f} / {res['plain']['macro_f1']:.4f} |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
{rows}

Confusion matrix (rows = gold, columns = predicted; figure `docs/figures/cm_stance_fever.png`):

| gold \\ pred | SUPPORTS | REFUTES | NEUTRAL |
|---|---|---|---|
{cm_rows}

## Reading the result

{reading}
The page-title prefix is worth {res['accuracy'] - res['plain']['accuracy']:+.1%} accuracy: without it many gold
sentences start with a pronoun ("It was formerly called …") and the model cannot link them to the claim.

## Caveat (state this in the viva and the paper)

The model card (checked 2026-10-26) lists **SNLI + MultiNLI only** — FEVER / FEVER-NLI is *not* in its training
data, so this is a zero-shot transfer number (the earlier milestone text assumed FEVER-NLI was in the mix; it is not).
It is still optimistic for our use case: FEVER claims are short single-fact rewrites of the gold Wikipedia sentence,
whereas news claims paired with retrieved news / fact-check passages are longer, noisier and often only topically
related. The end-to-end Live Claims evaluation (ESE) is the realistic number. NEI premises here are retrieved
sentences, not random ones, which is likely a harder NEUTRAL test than the "random sentence" variant.

## Published reference points (other work) — not our measurement

Copied from the model cards (read 2026-10-26); we did not reproduce them, and they are in-domain NLI test numbers,
not FEVER numbers, so they are **not comparable** to the 0.7267 FEVER-dev accuracy above.

| source | model | dataset / split | metric | published value |
|---|---|---|---|---|
| model card `cross-encoder/nli-deberta-v3-small` | the cross-encoder we use | SNLI test | accuracy | 91.65 |
| model card `cross-encoder/nli-deberta-v3-small` | the cross-encoder we use | MNLI mismatched | accuracy | 87.55 |
| model card `microsoft/deberta-v3-small` (He et al., DeBERTaV3) | DeBERTa-v3-small fine-tuned by Microsoft | MNLI-m / mm | accuracy | 88.3 / 87.7 |

Sources: <https://huggingface.co/cross-encoder/nli-deberta-v3-small>, <https://huggingface.co/microsoft/deberta-v3-small>.
The gap between ≈ 88–92 % in-domain NLI and our 72.7 % on FEVER dev is the zero-shot transfer cost described above,
concentrated in the NEUTRAL (NOT ENOUGH INFO) class.
""")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="FEVER stance evaluation")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N pairs (smoke test)")
    a = ap.parse_args(argv)
    import torch
    torch.set_num_threads(4)
    download_gold()
    pairs, info = build_pairs()
    pairs.to_parquet(PAIRS_PATH, index=False)
    if a.limit:
        pairs = pairs.groupby("label").head(a.limit // 3).reset_index(drop=True)
    print("pairs:", len(pairs), pairs["label"].value_counts().to_dict(), info)
    res = evaluate(pairs, a.batch_size)
    plain = evaluate(pairs, a.batch_size, col="premise_plain")
    res["plain"] = {k: plain[k] for k in ("accuracy", "macro_f1", "cm")}
    print(f"no-title ablation: accuracy {plain['accuracy']:.4f}  macro-F1 {plain['macro_f1']:.4f}")
    print(f"accuracy {res['accuracy']:.4f}  macro-F1 {res['macro_f1']:.4f}  {res['seconds']:.0f}s")
    if not a.limit:
        save_all(res, info, len(pairs))
    return 0


def save_all(res: dict, info: dict, n: int) -> None:
    """stance_fever.md + cm_stance_fever.png + train_logs/stance_fever.json (used by main() and notebook 05)."""
    device = f"CPU ({platform.processor() or platform.machine()}, 4 torch threads)"
    write_md(res, info, n, device)
    plot_cm(res["cm"])
    log = {k: v for k, v in res.items() if k not in ("pred", "probs")}
    log.update(info=info, model=NLI_MODEL, n=n, gold_source=f"copenlu/fever_gold_evidence@{GOLD_REV}",
               date=time.strftime("%Y-%m-%d"), device=device)
    (RESULTS_DIR / "train_logs").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "train_logs" / "stance_fever.json").write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
