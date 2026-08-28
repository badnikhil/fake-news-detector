"""Exact and near-duplicate removal (master doc §11.1 step 3).

* exact: SHA-1 of the *normalised* text (lower-cased, alphanumerics only, whitespace collapsed)
* near:  MinHash (``datasketch``) over word 5-gram shingles, 128 permutations, LSH threshold 0.9
         (estimated Jaccard); greedy — the first occurrence in frame order is kept, later near-copies dropped.

Both are deterministic given the row order, so ``make data`` twice yields the same rows.
"""
from __future__ import annotations

import hashlib
import re

import pandas as pd
from datasketch import MinHash, MinHashLSH

from src.config import MINHASH_PERMS, NEAR_DUP_THRESHOLD, SHINGLE_SIZE

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def norm_key(text: str) -> str:
    return _NON_ALNUM.sub(" ", str(text).lower()).strip()


def exact_hash(text: str) -> str:
    return hashlib.sha1(norm_key(text).encode("utf-8")).hexdigest()


def drop_exact_duplicates(df: pd.DataFrame, col: str = "text") -> tuple[pd.DataFrame, int]:
    h = df[col].map(exact_hash)
    keep = ~h.duplicated(keep="first")
    return df[keep].reset_index(drop=True), int((~keep).sum())


def _shingles(text: str, k: int = SHINGLE_SIZE) -> set[bytes]:
    toks = norm_key(text).split()
    if len(toks) <= k:
        return {" ".join(toks).encode("utf-8")}
    return {" ".join(toks[i : i + k]).encode("utf-8") for i in range(len(toks) - k + 1)}


def minhash(text: str, num_perm: int = MINHASH_PERMS) -> MinHash:
    m = MinHash(num_perm=num_perm, seed=1)
    m.update_batch(list(_shingles(text)))
    return m


def find_near_duplicates(
    texts: pd.Series, threshold: float = NEAR_DUP_THRESHOLD, num_perm: int = MINHASH_PERMS
) -> tuple[list[int], MinHashLSH, list[MinHash]]:
    """Return (positions_to_drop, lsh_index_of_kept_rows, minhashes) using a greedy first-wins scan."""
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    drop: list[int] = []
    hashes: list[MinHash] = []
    for pos, text in enumerate(texts.tolist()):
        m = minhash(text, num_perm)
        hashes.append(m)
        if lsh.query(m):
            drop.append(pos)
        else:
            lsh.insert(pos, m)
    return drop, lsh, hashes


def drop_near_duplicates(df: pd.DataFrame, col: str = "text") -> tuple[pd.DataFrame, int, list[MinHash]]:
    drop, _, hashes = find_near_duplicates(df[col])
    keep_mask = pd.Series(True, index=df.index)
    keep_mask.iloc[drop] = False
    dropped = set(drop)
    kept_hashes = [h for i, h in enumerate(hashes) if i not in dropped]
    return df[keep_mask].reset_index(drop=True), len(drop), kept_hashes


def cross_split_overlap(
    df: pd.DataFrame, hashes: list[MinHash], split_col: str = "split", text_col: str = "text"
) -> dict[str, int]:
    """Count (a) exact-hash collisions and (b) MinHash near-duplicate pairs between train and test/val.

    ``hashes`` must be aligned with ``df`` rows.  Should be 0 everywhere because dedup runs before splitting.
    """
    h = df[text_col].map(exact_hash)
    out = {}
    for other in ("val", "test"):
        a = set(h[df[split_col] == "train"])
        b = set(h[df[split_col] == other])
        out[f"exact_train_{other}"] = len(a & b)
    lsh = MinHashLSH(threshold=NEAR_DUP_THRESHOLD, num_perm=MINHASH_PERMS)
    train_pos = [i for i, s in enumerate(df[split_col].tolist()) if s == "train"]
    for i in train_pos:
        lsh.insert(i, hashes[i])
    for other in ("val", "test"):
        n = 0
        for i, s in enumerate(df[split_col].tolist()):
            if s == other and lsh.query(hashes[i]):
                n += 1
        out[f"near_train_{other}"] = n
    return out
