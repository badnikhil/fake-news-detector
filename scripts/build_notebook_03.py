#!/usr/bin/env python
"""Generate notebooks/03_transformers.ipynb (source only; execute it in place with
`make nb-run-transformers` so the outputs are saved for the examiners).

The notebook never trains: it reads the logs written by `make train-distilbert` / `make train-liar`
(`docs/results/train_logs/distilbert_*.json`) and loads the saved checkpoints for a short inference demo."""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "03_transformers.ipynb"

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))  # noqa: E731
code = lambda s: cells.append(nbf.v4.new_code_cell(s))  # noqa: E731

md("""# 03 — Fine-tuned DistilBERT (WELFake sweep) and the LIAR 3-way head (MSE2)

**Fake News & Misinformation Detector** · MSE2 criteria *Model Training* and *Model Tuning* (master doc §12.2–12.3, MSE2 doc §3–4).

This notebook **does not train**: it loads what `make train-distilbert` / `make train-liar` saved on the RTX 2050.

**Status (2026-10-26):** the WELFake sweep was **stopped on user decision after 3 of 6 configs** (lr {2e-5, 3e-5, 5e-5} at
max_len 256). The three max_len 512 runs and the full-train retrain did not run and are deferred to ESE. Sweep runs keep no weights, so
there is **no DistilBERT WELFake checkpoint and no DistilBERT test number**. The served classifier stays TF-IDF and MSE2 objective O1
is not checked. The LIAR 3-way head ran completely.

Code: `src/models/distilbert.py`, `src/models/liar_head.py`.""")

code("""import json, sys
from pathlib import Path
import pandas as pd
from IPython.display import Image, Markdown, display

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
from src import config as C
from src.models import distilbert as D
from src.models import liar_head as L
pd.set_option("display.precision", 4); pd.set_option("display.width", 160)
recs = D.sweep_records()
print("hardware:", recs[0]["hardware"])
print("WELFake DistilBERT checkpoint present:", (D.FINAL_DIR / "config.json").exists())""")

md("""## 1. Sweep on a 20k stratified subset (selection metric: val macro-F1)

Fixed settings: AdamW, linear schedule, warmup ratio 0.1, weight decay 0.01, fp16, seed 42, input `title [SEP] text` head-truncated, batch 16 at
max_len 256. Validation = the full WELFake val split (6,080 rows).""")

code("""rows = []
for r in recs:
    c = r["config"]
    rows.append({"lr": f"{c['lr']:.0e}", "max_len": c["max_len"], "batch×accum": f"{c['per_device_batch']}×{c['grad_accum']}",
                 "best_epoch": r["best_epoch"], **{f"val_F1_ep{e['epoch']}": e["val_macro_f1"] for e in r["per_epoch"]},
                 **{f"val_loss_ep{e['epoch']}": e["val_loss"] for e in r["per_epoch"]},
                 "best_val_macro_F1": r["best_val_macro_f1"], "minutes": r["minutes"], "peak_GPU_GiB": r["peak_gpu_mem_gib"]})
pd.DataFrame(rows).sort_values("best_val_macro_F1", ascending=False).reset_index(drop=True)""")

code("""display(Markdown(D.SWEEP_MD.read_text().split("\\n\\n", 2)[2]))
display(Image(str(C.FIGURES_DIR / "sweep_distilbert_lr.png")))""")

md("""## 2. Training curve of the best completed sweep run

There was no full-train retrain, so `docs/figures/train_distilbert.png` shows the best **20k-subset** run: train loss logged every 50 steps
plus the per-epoch mean, val loss, and val macro-F1 per epoch with the best epoch marked.""")

code("""best = max(recs, key=lambda r: r["best_val_macro_f1"])
display(pd.DataFrame(best["per_epoch"]).set_index("epoch"))
display(Image(str(C.FIGURES_DIR / "train_distilbert.png")))""")

md("""## 5. LIAR 3-way head (TRUE / MIXED / FALSE)

Statement text only (speaker / party are label proxies), class-weighted cross-entropy, max_len 64, LIAR official splits.
Wang (2017) reports 6-way accuracy only, so the winning config is also trained 6-way for a like-for-like comparison.""")

code("""liar = json.loads(L.LOG.read_text())
display(pd.DataFrame([{"lr": f"{r['config']['lr']:.0e}", "epochs": r["config"]["epochs"], "best_epoch": r["best_epoch"],
                       "best_val_macro_F1": r["best_val_macro_f1"], "minutes": r["minutes"]} for r in liar["sweep"]]))
display(Markdown(L.MD.read_text().split("## Test")[1].join(["### Test", ""])))
display(Image(str(C.FIGURES_DIR / "train_liar.png")))""")

md("""### Reload the LIAR checkpoint with `AutoModelForSequenceClassification`

The head shades claims TRUE / MIXED / FALSE. It is a confidence signal for fusion, never the verdict.""")

code("""import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
tok = AutoTokenizer.from_pretrained(L.FINAL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(L.FINAL_DIR).eval()
claims = ["Says the unemployment rate fell to its lowest level in 50 years.",
          "Says vaccines contain microchips that track people.",
          "The federal budget deficit doubled under the current administration."]
with torch.no_grad():
    probs = torch.softmax(model(**tok(claims, padding=True, truncation=True, max_length=L.MAX_LEN, return_tensors="pt")).logits, -1)
pd.DataFrame(probs.numpy().round(3), columns=[model.config.id2label[i] for i in range(3)], index=claims)""")

md("""## Summary

* The completed sweep runs and the LIAR head can be reproduced with `make train-distilbert` / `make train-liar` (seed 42; the logs in
  `docs/results/train_logs/` record the full config, hardware and wall-clock). `--sweep` skips runs whose log already exists, so re-running
  continues with the max_len 512 configs (deferred to ESE).
* Tables: `docs/results/distilbert_sweep.md`, `docs/results/liar_head.md`; figures: `docs/figures/train_distilbert.png`,
  `docs/figures/sweep_distilbert_lr.png`, `docs/figures/train_liar.png`.""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"name": "fakenews", "display_name": "Python (fakenews)", "language": "python"},
    "language_info": {"name": "python"},
}
nbf.write(nb, OUT)
print("wrote", OUT, f"({len(cells)} cells)")
