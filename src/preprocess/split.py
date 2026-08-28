"""Stratified 80/10/10 train/val/test splits with a fixed seed (master doc §11.1 step 8).

Splits are written as ``data/splits/<dataset>_<split>.csv`` with columns ``id,label`` in sorted-id
order so that the files are byte-identical across runs.  LIAR keeps its official splits.
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import SEED, SPLIT_NAMES, SPLIT_RATIOS, SPLITS_DIR


def stratified_split(df: pd.DataFrame, label_col: str = "label", seed: int = SEED) -> pd.Series:
    """Return a Series of 'train'/'val'/'test' aligned with ``df`` (80/10/10, stratified on ``label_col``)."""
    idx = df.index.to_numpy()
    y = df[label_col].to_numpy()
    test_frac = SPLIT_RATIOS["test"]
    val_frac = SPLIT_RATIOS["val"] / (1.0 - test_frac)
    trainval_idx, test_idx = train_test_split(idx, test_size=test_frac, stratify=y, random_state=seed)
    y_tv = df.loc[trainval_idx, label_col].to_numpy()
    train_idx, val_idx = train_test_split(trainval_idx, test_size=val_frac, stratify=y_tv, random_state=seed)
    split = pd.Series("train", index=df.index, dtype="string")
    split.loc[val_idx] = "val"
    split.loc[test_idx] = "test"
    return split


def write_split_csvs(df: pd.DataFrame, name: str, out_dir=SPLITS_DIR) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sizes = {}
    for s in SPLIT_NAMES:
        part = df.loc[df["split"] == s, ["id", "label"]].sort_values("id", kind="mergesort")
        part.to_csv(out_dir / f"{name}_{s}.csv", index=False, lineterminator="\n")
        sizes[s] = len(part)
    return sizes


def split_report(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    """Per-split size, fraction and class proportions (for the log / tests)."""
    total = len(df)
    rows = []
    overall = df[label_col].value_counts(normalize=True)
    for s in SPLIT_NAMES:
        part = df[df["split"] == s]
        dist = part[label_col].value_counts(normalize=True)
        rows.append(
            {
                "split": s,
                "rows": len(part),
                "fraction": round(len(part) / total, 4),
                **{f"p({k})": round(float(dist.get(k, 0.0)), 4) for k in overall.index},
                "max_class_dev": round(float((dist.reindex(overall.index).fillna(0) - overall).abs().max()), 4),
            }
        )
    return pd.DataFrame(rows)
