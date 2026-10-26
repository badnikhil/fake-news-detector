#!/usr/bin/env python
"""Generate notebooks/05_stance.ipynb (source only; `make nb-run-stance` builds + executes it in place)."""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "05_stance.ipynb"

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))  # noqa: E731
code = lambda s: cells.append(nbf.v4.new_code_cell(s))  # noqa: E731

md("""# 05 — Stance detection on FEVER + offline evidence demo (MSE2)

**Fake News & Misinformation Detector** · MSE2 criteria *Model Training* / *Result Analysis* (master doc §13.3, §10.2).

1. Build the **3,000 balanced FEVER dev pairs** (1k SUPPORTS / 1k REFUTES / 1k NEI) from our `fever_subset` val split,
   with the gold sentence **text** from `copenlu/fever_gold_evidence` (our raw FEVER files only hold evidence pointers).
2. Run `cross-encoder/nli-deberta-v3-small` on CPU (premise = evidence, hypothesis = claim) — pre-trained, not fine-tuned.
3. Accuracy / macro-F1 / confusion matrix vs. the O4 target (≥ 85 %), plus the no-page-title ablation.
4. Error examples, then the offline FAISS index and one `/analyze`-style pipeline call in `MODE=offline`.

All code: `src/stance/{nli,fever_eval}.py`, `src/evidence/offline_index.py`, `src/api/pipeline.py`.""")

code("""import sys, time, json, warnings
from pathlib import Path
import pandas as pd
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
import torch; torch.set_num_threads(4)
from src.stance import fever_eval as F, nli
pd.set_option("display.max_colwidth", 110); pd.set_option("display.width", 180)
T0 = time.time()""")

md("## 1. Pairs (gold sentence text joined on the FEVER claim id)")
code("""F.download_gold()
pairs, info = F.build_pairs()
print(info); print(pairs["label"].value_counts().to_dict())
pairs.groupby("label").head(2)[["label", "claim", "premise"]]""")

md("## 2. NLI stance on CPU (label order read from the checkpoint's `id2label`)")
code("""print(nli.model().model.config.id2label)
res = F.evaluate(pairs)
plain = F.evaluate(pairs, col="premise_plain")
res["plain"] = {k: plain[k] for k in ("accuracy", "macro_f1", "cm")}
print(f"accuracy {res['accuracy']:.4f} | macro-F1 {res['macro_f1']:.4f} | {res['per_pair_ms']:.0f} ms/pair")
print(f"no-title ablation: accuracy {plain['accuracy']:.4f} | macro-F1 {plain['macro_f1']:.4f}")
print("O4 target (>= 0.85):", "met" if res["accuracy"] >= F.O4_TARGET else "NOT met")
pd.DataFrame(res["report"]).T.round(4)""")

md("## 3. Confusion matrix (rows = gold, columns = predicted)")
code("""from IPython.display import Image
F.save_all(res, info, len(pairs))   # writes docs/results/stance_fever.md, the figure and train_logs/stance_fever.json
Image(str(F.OUT_FIG))""")

md("""## 4. Where it fails

A few gold-vs-predicted disagreements per gold class. The dominant cell is **gold NEI → predicted REFUTES**: the
retrieved NEI sentence is about the same entity, and the model often reads "a different fact about X" as a
contradiction. Smaller cells: SUPPORTS claims that need world knowledge or multi-hop evidence (e.g. "studied in
Annapolis" vs. "graduated from the U.S. Naval Academy") predicted NEUTRAL.""")
code("""pairs["pred"] = res["pred"]
err = pairs[pairs.pred != pairs.label]
print(f"{len(err)} errors"); print(pd.crosstab(err.label, err.pred))
err.groupby("label").head(3)[["label", "pred", "claim", "premise"]]""")

md("## 5. Offline FAISS index (LIAR + FakeNewsNet PolitiFact, MiniLM, IndexFlatIP)")
code("""from src.evidence import offline_index as OI
print(json.loads((OI.INDEX_DIR / "build_info.json").read_text()))
t = time.time(); hits = OI.query("Says the unemployment rate doubled under Barack Obama", k=5)
print(f"query {1000 * (time.time() - t):.0f} ms")
pd.DataFrame(hits)[["cosine", "record_id", "dataset", "rating", "text"]]""")

md("""## 6. One end-to-end call in MODE=offline (claims → offline evidence → stance → temporal → fusion)

Same function the API's `POST /analyze` calls. Evidence decides; `p_fake` only adjusts confidence / breaks the R8 tie.""")
code("""from src.api import pipeline as P
for text in ["Says the unemployment rate doubled under Barack Obama.",
             "The bakery on Maple Street in Springfield ran out of croissants at 9 am last Tuesday."]:
    out = P.analyze(text=text, input_type="headline", mode="offline", k=5)
    print(f"\\n{text}\\n -> {out['verdict']} ({out['rule_fired']}, confidence {out['confidence']}) | {out['reason']}")
    print("    classifier:", out["classifier"])
    for e in out["claims"][0]["evidence"]:
        print(f"    [{e['stance']:<8} p={e['p']:.2f} cos={e['cosine']:.2f}] {e['record_id']} ({e['rating']}): {e['snippet'][:90]}")
print(f"\\nnotebook runtime: {time.time() - T0:.0f} s")""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"name": "fakenews", "display_name": "Python (fakenews)", "language": "python"},
    "language_info": {"name": "python"},
}
nbf.write(nb, OUT)
print("wrote", OUT, f"({len(cells)} cells)")
