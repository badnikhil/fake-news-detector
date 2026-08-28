#!/usr/bin/env python
"""Generate notebooks/00_datasets.ipynb (source only; execute it with `make nb-run` so outputs are saved)."""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "00_datasets.ipynb"

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))  # noqa: E731
code = lambda s: cells.append(nbf.v4.new_code_cell(s))  # noqa: E731

md("""# 00 — Datasets: raw files, unified schema, counts and label mappings

**Fake News & Misinformation Detector** (MSE1 · Dataset criterion). This notebook

1. loads every **raw** file in `data/raw/` (fetched by `make download`, no Kaggle needed) and prints row counts + label distributions,
2. loads the **processed** parquet files produced by `make data` (`src/preprocess/`), shows the unified schema, counts, label distributions, split sizes and the first rows,
3. prints the label-mapping tables (LIAR 6 → 5 / 3, FEVER → stance) and the download manifest (URL, SHA-256, date).

Expected sizes (master doc §10): WELFake 72,134 · ISOT 44,898 · LIAR 12,836 · FEVER subset 20k + 3k · FakeNewsNet-PolitiFact ~1,056.
Runtime ≈ 1 min on CPU.""")

code("""import json, sys, platform, time
from pathlib import Path
import pandas as pd

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
from src import config as C
pd.set_option("display.max_colwidth", 90); pd.set_option("display.width", 160)
print("python", platform.python_version(), "| pandas", pd.__version__)
print("raw:", C.RAW_DIR, "| processed:", C.PROCESSED_DIR, "| splits:", C.SPLITS_DIR)
t0 = time.time()""")

md("## 1. Raw files (as downloaded)")

code("""EXPECTED = {"welfake": 72_134, "isot": 44_898, "liar": 12_836, "fever_train_full": 145_449,
            "fever_dev_full": 19_998, "fnn_politifact": 1_056}
raw = {}
raw["welfake"] = pd.read_csv(C.RAW_DIR / "welfake" / "WELFake_Dataset.csv")
isot_true = pd.read_csv(C.RAW_DIR / "isot" / "True.csv"); isot_fake = pd.read_csv(C.RAW_DIR / "isot" / "Fake.csv")
raw["isot"] = pd.concat([isot_true.assign(label="real"), isot_fake.assign(label="fake")], ignore_index=True)
from src.preprocess.load import LIAR_COLUMNS
raw["liar"] = pd.concat([pd.read_csv(C.RAW_DIR / "liar" / f, sep="\\t", header=None, names=LIAR_COLUMNS, quoting=3,
                                     dtype=str, keep_default_na=False).assign(split=s)
                         for f, s in (("train.tsv", "train"), ("valid.tsv", "val"), ("test.tsv", "test"))], ignore_index=True)
raw["fever_train_full"] = pd.read_json(C.RAW_DIR / "fever" / "train.jsonl", lines=True)
raw["fever_dev_full"] = pd.read_json(C.RAW_DIR / "fever" / "shared_task_dev.jsonl", lines=True)
raw["fnn_politifact"] = pd.concat([pd.read_csv(C.RAW_DIR / "fnn_politifact" / f, dtype=str).assign(label=l)
                                   for f, l in (("politifact_fake.csv", "fake"), ("politifact_real.csv", "real"))], ignore_index=True)
rows = [{"dataset": k, "rows": len(v), "expected": EXPECTED[k], "match": len(v) == EXPECTED[k], "columns": ", ".join(map(str, v.columns[:6]))}
        for k, v in raw.items()]
pd.DataFrame(rows)""")

md("""### 1.1 Raw label distributions

> **WELFake label quirk.** The Zenodo/Kaggle description says `0 = fake, 1 = real`, but the file is the other way round:
> label **0** rows (35,028) are the *real* articles (60 % carry a Reuters dateline) and label **1** rows (37,106) are the *fake* ones
> (`Featured image via …`, `[VIDEO]`, ALL-CAPS titles). The paper's own counts (35,028 real / 37,106 fake) confirm it, so
> `src/preprocess/load.py` maps `0 → real`, `1 → fake`.""")

code("""w = raw["welfake"].copy(); w["text"] = w["text"].fillna(""); w["title"] = w["title"].fillna("")
print("WELFake raw label counts:", w["label"].value_counts().sort_index().to_dict())
print(pd.DataFrame({"label": [0, 1],
                    "frac_with_(Reuters)": [w[w.label == l]["text"].str.contains(r"\\(Reuters\\)").mean().round(3) for l in (0, 1)],
                    "frac_title_has_[VIDEO]": [w[w.label == l]["title"].str.contains(r"\\[VIDEO\\]").mean().round(3) for l in (0, 1)],
                    "frac_'Featured image'": [w[w.label == l]["text"].str.contains("Featured image").mean().round(3) for l in (0, 1)]}).to_string(index=False))
print("\\nISOT:", raw["isot"]["label"].value_counts().to_dict(), "| subjects:", raw["isot"]["subject"].value_counts().to_dict())
print("ISOT dates:", pd.to_datetime(raw["isot"]["date"], errors="coerce", format="mixed").agg(["min", "max"]).dt.date.to_dict())
print("\\nLIAR 6-way:", raw["liar"]["label"].value_counts().to_dict(), "| official splits:", raw["liar"]["split"].value_counts().to_dict())
print("\\nFEVER train (full):", raw["fever_train_full"]["label"].value_counts().to_dict())
print("FEVER shared_task_dev (full):", raw["fever_dev_full"]["label"].value_counts().to_dict())
print("\\nFakeNewsNet PolitiFact:", raw["fnn_politifact"]["label"].value_counts().to_dict())""")

md("### 1.2 Three raw rows per dataset")
code("""for k in ("welfake", "isot", "liar", "fever_train_full", "fnn_politifact"):
    print(f"--- {k} ---")
    cols = [c for c in raw[k].columns if c not in ("tweet_ids",)][:6]
    display(raw[k][cols].head(3))""")

md("""## 2. Processed data (`make data` → `data/processed/*.parquet`)

Unified schema for **every** dataset (`src/config.SCHEMA`):

| column | meaning |
|---|---|
| `id` | deterministic unique id (`welfake_12`, `isot_real_7`, `liar_2635`, `fever_75397`, `fnn_fake_politifact14984`) |
| `title` | headline (None for LIAR / FEVER) |
| `text` | cased, cleaned text (article body / statement / claim / headline) |
| `label` | dataset-native label in project vocabulary: `fake`/`real`; LIAR 6-way; FEVER stance `SUPPORTS`/`REFUTES`/`NEUTRAL` |
| `label5` | project 5-class mapping (`REAL` / `FAKE` / `PARTIALLY TRUE` / `MISLEADING`); None for FEVER |
| `label3` | 3-way collapse (`TRUE` / `MIXED` / `FALSE`); None for FEVER |
| `source_dataset` | `welfake` · `isot` · `liar` · `fever_subset` · `fnn_politifact` |
| `date` | ISO publication date where available (ISOT only) |
| `split` | `train` / `val` / `test` (stratified 80/10/10 seed 42; LIAR official; FEVER train/dev subset) |
| `meta` | JSON string with dataset-specific extras (ISOT subject, LIAR speaker/party/context, FEVER evidence pages, FNN url) |""")

code("""proc = {n: pd.read_parquet(C.PROCESSED_DIR / f"{n}.parquet") for n in C.DATASETS if (C.PROCESSED_DIR / f"{n}.parquet").exists()}
summary = pd.DataFrame([{"dataset": n, "rows": len(d), "columns_ok": list(d.columns) == C.SCHEMA,
                         "splits": d["split"].value_counts().to_dict(), "label": d["label"].value_counts().to_dict()}
                        for n, d in proc.items()])
summary""")

md("### 2.1 Schema table (identical across datasets)")
code("""schema_rows = []
for n, d in proc.items():
    for c in d.columns:
        schema_rows.append({"dataset": n, "column": c, "dtype": str(d[c].dtype), "non_null": int(d[c].notna().sum())})
sch = pd.DataFrame(schema_rows).pivot(index="column", columns="dataset", values="non_null").reindex(C.SCHEMA)
sch""")

md("### 2.2 Counts: raw → processed, per dataset (see `docs/mse1_make_data.log` for the drop/dedup breakdown)")
code("""summ = json.loads((C.PROCESSED_DIR / "make_data_summary.json").read_text())
cols = ["dataset", "rows_in", "dropped_empty", "dropped_short", "exact_dups_removed", "near_dups_removed", "rows_out", "seconds"]
pd.DataFrame([{c: s.get(c, 0) for c in cols} for s in summ])""")

md("### 2.3 Label distributions (label / label5 / label3) and split sizes")
code("""for n, d in proc.items():
    print(f"=== {n}: {len(d):,} rows ===")
    print("  label :", d["label"].value_counts().to_dict())
    print("  label5:", d["label5"].value_counts(dropna=False).to_dict())
    print("  label3:", d["label3"].value_counts(dropna=False).to_dict())
    print("  split :", d["split"].value_counts(dropna=False).to_dict())
    ct = pd.crosstab(d["split"], d["label"], normalize="index").round(3)
    display(ct)""")

md("### 2.4 Split ID files (`data/splits/{welfake,isot,liar}_{train,val,test}.csv`)")
code("""rows = []
for n in C.SPLIT_DATASETS:
    total = sum(len(pd.read_csv(C.SPLITS_DIR / f"{n}_{s}.csv")) for s in C.SPLIT_NAMES)
    for s in C.SPLIT_NAMES:
        k = len(pd.read_csv(C.SPLITS_DIR / f"{n}_{s}.csv"))
        rows.append({"dataset": n, "split": s, "rows": k, "fraction": round(k / total, 4)})
pd.DataFrame(rows).pivot(index="dataset", columns="split", values=["rows", "fraction"])""")

md("### 2.5 First rows of each processed table")
code("""for n, d in proc.items():
    print(f"--- {n} ---")
    display(d.drop(columns=["meta"]).head(3).assign(text=lambda x: x["text"].str.slice(0, 120) + "…"))""")

md("### 2.6 Artefact removal check: `(Reuters)` must be gone from processed ISOT / WELFake")
code("""for n in ("isot", "welfake"):
    if n in proc:
        print(n, "rows containing '(Reuters)':", int(proc[n]["text"].str.contains(r"\\(Reuters\\)").sum()),
              "| 'http':", int(proc[n]["text"].str.contains("http").sum()))
ex = raw["isot"].loc[raw["isot"]["label"] == "real", "text"].iloc[0][:160]
print("\\nraw ISOT   :", ex)
print("processed  :", proc["isot"].loc[proc["isot"]["id"] == "isot_real_0", "text"].iloc[0][:160])""")

md("## 3. Label mappings (master doc §10.1 / §10.2)")
code("""print("LIAR 6 -> project 5-class / 3-way")
display(pd.DataFrame({"liar_label": C.LIAR_LABELS, "label5": [C.LIAR_TO_5[l] for l in C.LIAR_LABELS],
                      "label3": [C.LIAR_TO_3[l] for l in C.LIAR_LABELS]}))
print("Binary datasets (WELFake / ISOT / FakeNewsNet):", C.BINARY_TO_5, C.BINARY_TO_3)
print("FEVER -> stance:", C.FEVER_TO_STANCE)
print("Verdict vocabulary:", C.VERDICTS, "(UNVERIFIABLE arises only from retrieval failure at run time)")""")

md("## 4. Download manifest (URL · SHA-256 · date) — mirrored in `data/README.md`")
code("""man = json.loads((C.RAW_DIR / "manifest.json").read_text())
pd.DataFrame([{"file": k, **v} for k, v in man.items()])[["file", "dataset", "size", "sha256", "downloaded", "url"]]""")

md("## 5. Summary: expected vs. actual")
code("""actual = {"welfake": len(proc["welfake"]), "isot": len(proc["isot"]) if "isot" in proc else 0, "liar": len(proc["liar"]),
          "fever_subset": len(proc["fever_subset"]), "fnn_politifact": len(proc["fnn_politifact"])}
exp = {"welfake": "72,134 raw (after dedup: see log)", "isot": "44,898 raw (after dedup: see log)", "liar": "12,836",
       "fever_subset": "23,000 (20k train + 3k dev)", "fnn_politifact": "~1,056"}
display(pd.DataFrame({"expected": exp, "raw_rows": {"welfake": len(raw["welfake"]), "isot": len(raw["isot"]), "liar": len(raw["liar"]),
                                                    "fever_subset": len(raw["fever_train_full"]) + len(raw["fever_dev_full"]),
                                                    "fnn_politifact": len(raw["fnn_politifact"])}, "processed_rows": actual}))
print(f"notebook runtime: {time.time() - t0:.1f}s")""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"name": "fakenews", "display_name": "Python (fakenews)", "language": "python"},
    "language_info": {"name": "python"},
}
OUT.parent.mkdir(exist_ok=True)
nbf.write(nb, OUT)
print("wrote", OUT)
