"""Artefact / leakage removal (master doc §11.1 step 4) and the artefact-only leakage classifier.

Patterns stripped from article text (WELFake and ISOT):

* Reuters datelines  ``"WASHINGTON (Reuters) - "``, ``"(Reuters) - "``, and any remaining ``"(Reuters)"``
* fake-site trailers ``"Featured image via …"``, ``"Read more: …"``, ``"21st Century Wire says …"``
* URLs, e-mail addresses, Twitter handles (``@realDonaldTrump``), ``pic.twitter.com/…`` links

``artefact_features`` counts these patterns per document so that a classifier trained **only** on them
quantifies how much of the in-domain accuracy is leakage (expected ≥ 0.9 on raw ISOT).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

# Order matters: datelines first, then the bare token.
DATELINE_RE = re.compile(
    r"^\s*(?:[A-Z][A-Za-z0-9.'’/&-]*(?:[ ,]+[A-Z][A-Za-z0-9.'’/&-]*){0,6}\s*)?\((?:Reuters)\)\s*[-–—:]*\s*",
)
REUTERS_RE = re.compile(r"\(\s*Reuters\s*\)")
FEATURED_IMAGE_RE = re.compile(r"Featured image (?:via|by|from)[^\n]*$", re.IGNORECASE | re.MULTILINE)
READ_MORE_RE = re.compile(r"Read more:?[^\n]*$", re.IGNORECASE | re.MULTILINE)
WIRE_RE = re.compile(r"21st Century Wire says[^\n]*", re.IGNORECASE)
URL_RE = re.compile(r"(?:https?:\s*//\s*|www\.)\S+", re.IGNORECASE)  # WELFake has "https:// twitter.com/..."
PIC_RE = re.compile(r"pic\.twitter\.com/\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
HANDLE_RE = re.compile(r"(?<![\w])@\w{1,30}")
MULTISPACE_RE = re.compile(r"[ \t]{2,}")

# (name, regex) pairs used both for stripping and for the leakage-feature counts
PATTERNS: list[tuple[str, re.Pattern]] = [
    ("dateline", DATELINE_RE),
    ("reuters", REUTERS_RE),
    ("featured_image", FEATURED_IMAGE_RE),
    ("read_more", READ_MORE_RE),
    ("wire", WIRE_RE),
    ("url", URL_RE),
    ("pic_twitter", PIC_RE),
    ("email", EMAIL_RE),
    ("handle", HANDLE_RE),
]


def strip_artefacts(text: str) -> str:
    """Remove every artefact pattern from one document."""
    if not text:
        return text
    for _, rx in PATTERNS:
        text = rx.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text)
    return text.strip()


def strip_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply ``strip_artefacts`` to ``text``; return (frame, {pattern: n_docs_matched})."""
    counts = {name: int(df["text"].str.contains(rx, regex=True).sum()) for name, rx in PATTERNS}
    out = df.copy()
    out["text"] = out["text"].map(strip_artefacts).astype("string")
    return out, counts


def artefact_features(texts: pd.Series) -> np.ndarray:
    """Per-document counts of each artefact pattern (shape: n_docs × n_patterns)."""
    cols = [texts.str.count(rx.pattern, flags=rx.flags).fillna(0).to_numpy() for _, rx in PATTERNS]
    return np.column_stack(cols).astype(float)


def artefact_only_accuracy(texts: pd.Series, labels: pd.Series, seed: int = 42) -> dict:
    """Train a logistic regression on the artefact counts only (5-fold CV accuracy)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    X = artefact_features(texts)
    y = labels.to_numpy()
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    acc = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    per_pattern = {name: float((X[:, i] > 0).mean()) for i, (name, _) in enumerate(PATTERNS)}
    return {"cv_accuracy": float(acc.mean()), "cv_std": float(acc.std()), "doc_frac_with_pattern": per_pattern}
