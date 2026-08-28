"""Text normalisation and empty/short-row filtering (master doc §11.1 steps 2 and 5).

The processed parquet keeps *cased* text: NFKC normalisation, quote/dash normalisation and whitespace
collapse only.  Lower-casing is left to the TF-IDF vectoriser / the DistilBERT-uncased tokenizer.
"""
from __future__ import annotations

import re
import unicodedata

import pandas as pd

from src.config import MIN_TOKENS

_QUOTES = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "–": "-", "—": "-", "−": "-", " ": " ",
}
_QUOTE_RE = re.compile("|".join(map(re.escape, _QUOTES)))
_WS_RE = re.compile(r"[ \t\r\f\v]+")
_NL_RE = re.compile(r"\s*\n\s*")


def normalise(text: str | None) -> str:
    """NFKC + quote normalisation + whitespace collapse; keeps case and paragraph breaks."""
    if text is None or (isinstance(text, float) and pd.isna(text)) or text is pd.NA:
        return ""
    t = unicodedata.normalize("NFKC", str(text))
    t = _QUOTE_RE.sub(lambda m: _QUOTES[m.group(0)], t)
    t = _WS_RE.sub(" ", t)
    t = _NL_RE.sub("\n", t)
    return t.strip()


def n_tokens(text: str) -> int:
    return len(text.split())


def clean_frame(df: pd.DataFrame, min_tokens: int | None = MIN_TOKENS) -> tuple[pd.DataFrame, dict]:
    """Normalise title/text and drop empty or too-short texts.

    Returns (kept_frame, stats) with stats = {rows_in, dropped_empty, dropped_short, rows_out}.
    ``min_tokens=None`` disables the short-text filter (used for statements/claims/headlines).
    """
    stats = {"rows_in": len(df)}
    df = df.copy()
    df["title"] = df["title"].map(normalise).astype("string")
    df["text"] = df["text"].map(normalise).astype("string")
    df.loc[df["title"] == "", "title"] = pd.NA
    empty = df["text"] == ""
    stats["dropped_empty"] = int(empty.sum())
    df = df[~empty]
    if min_tokens:
        short = df["text"].map(n_tokens) < min_tokens
        stats["dropped_short"] = int(short.sum())
        df = df[~short]
    else:
        stats["dropped_short"] = 0
    stats["rows_out"] = len(df)
    return df.reset_index(drop=True), stats
