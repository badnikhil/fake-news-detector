"""Offline fact-check index (master §13.2 step 3, MSE2 §3): all-MiniLM-L6-v2 embeddings of the LIAR statements
(all three official splits, with their PolitiFact ruling) + FakeNewsNet PolitiFact titles (with the fake/real label),
L2-normalised, in a FAISS ``IndexFlatIP`` (inner product on unit vectors = cosine).

    make index                                      # python -m src.evidence.offline_index --build
    python -m src.evidence.offline_index --query "Says the unemployment rate doubled under Obama"

Outputs: ``data/index/factcheck.faiss`` + ``data/index/meta.parquet`` (row i of meta = vector i) + ``build_info.json``.
The stored statement is what is embedded; the ruling is metadata only (it must not move the vector).
"""
from __future__ import annotations

import argparse
import json
import time
from functools import lru_cache

import numpy as np
import pandas as pd

from src.config import INDEX_DIR, LIAR_TO_3, PROCESSED_DIR

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH = INDEX_DIR / "factcheck.faiss"
META_PATH = INDEX_DIR / "meta.parquet"
FNN_TO_3 = {"fake": "FALSE", "real": "TRUE"}


@lru_cache(maxsize=1)
def embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL, device="cpu")


def embed(texts: list[str], batch_size: int = 64) -> np.ndarray:
    return embedder().encode(list(texts), batch_size=batch_size, normalize_embeddings=True,
                             convert_to_numpy=True, show_progress_bar=False).astype("float32")


def build_records() -> pd.DataFrame:
    liar = pd.read_parquet(PROCESSED_DIR / "liar.parquet", columns=["id", "text", "label", "split", "meta"])
    liar = pd.DataFrame({
        "record_id": liar["id"], "text": liar["text"], "dataset": "liar", "split": liar["split"],
        "rating": liar["label"], "rating3": liar["label"].map(LIAR_TO_3),
        "speaker": [json.loads(m).get("speaker") for m in liar["meta"]],
        "url": "", "source": "politifact.com"})
    fnn = pd.read_parquet(PROCESSED_DIR / "fnn_politifact.parquet", columns=["id", "title", "text", "label", "meta"])
    fnn_text = fnn["title"].fillna("").where(fnn["title"].fillna("").str.len() > 0, fnn["text"])
    fnn = pd.DataFrame({
        "record_id": fnn["id"], "text": fnn_text, "dataset": "fnn_politifact", "split": "",
        "rating": fnn["label"], "rating3": fnn["label"].map(FNN_TO_3), "speaker": "",
        "url": [json.loads(m).get("news_url") for m in fnn["meta"]], "source": "politifact.com"})
    df = pd.concat([liar, fnn], ignore_index=True)
    df["text"] = df["text"].astype(str).str.strip()
    return df[df["text"].str.len() > 0].reset_index(drop=True)


def build() -> dict:
    import faiss
    t0 = time.time()
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    meta = build_records()
    index = faiss.IndexFlatIP(384)
    for i in range(0, len(meta), 2048):  # batched to keep peak RAM low
        index.add(embed(meta["text"].iloc[i:i + 2048].tolist()))
    faiss.write_index(index, str(INDEX_PATH))
    meta.to_parquet(META_PATH, index=False)
    info = {"vectors": int(index.ntotal), "dim": 384, "model": EMBED_MODEL, "index": "IndexFlatIP (normalised)",
            "by_dataset": meta["dataset"].value_counts().to_dict(),
            "faiss_mb": round(INDEX_PATH.stat().st_size / 2**20, 2), "seconds": round(time.time() - t0, 1)}
    (INDEX_DIR / "build_info.json").write_text(json.dumps(info, indent=2))
    load.cache_clear()
    return info


@lru_cache(maxsize=1)
def load():
    import faiss
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"{INDEX_PATH} missing - run `make index`")
    return faiss.read_index(str(INDEX_PATH)), pd.read_parquet(META_PATH)


def query(text: str, k: int = 5, min_cosine: float = 0.0) -> list[dict]:
    """Top-k records for `text`: dicts with record_id, text, rating, rating3, dataset, cosine (desc)."""
    index, meta = load()
    scores, ids = index.search(embed([text]), k)
    out = []
    for s, i in zip(scores[0], ids[0], strict=True):
        if i < 0 or s < min_cosine:
            continue
        rec = meta.iloc[int(i)].to_dict()
        rec["cosine"] = round(float(s), 4)
        out.append(rec)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--query", default=None)
    ap.add_argument("-k", type=int, default=5)
    a = ap.parse_args(argv)
    if a.build:
        print(json.dumps(build(), indent=2))
    if a.query:
        t0 = time.time()
        for r in query(a.query, a.k):
            print(f"{r['cosine']:.3f}  {r['record_id']:<22} {r['rating']:<11} {r['text'][:100]}")
        print(f"query {time.time() - t0:.2f}s (incl. index + model load)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
