"""LIME token weights for the content classifier (master §15): whatever `/classify` serves (DistilBERT if
data/models/distilbert-welfake-v1/config.json exists, else TF-IDF + LR). Positive weight = pushes towards "fake".
Writing-style signal only; it never decides the verdict. Cached per input hash (repeat requests are free)."""
from __future__ import annotations

import hashlib
import time

import numpy as np

from src.api.pipeline import classifier

NOTE = "Writing-style signals only - these do not decide the verdict."
_cache: dict[str, dict] = {}


def explain(text: str, num_features: int = 10) -> dict:
    key = hashlib.sha1(f"{num_features}|{text}".encode()).hexdigest()
    if key in _cache:
        return {**_cache[key], "cached": True}
    name, predict = classifier()
    if predict is None:
        return {"model": None, "p_fake": None, "tokens": [], "note": NOTE, "cached": False}
    t0 = time.time()
    transformer = name.startswith("distilbert")
    # ponytail: each LIME sample is one classifier call; DistilBERT on CPU only fits the 3 s budget with a shorter text
    # and fewer samples (TF-IDF LR does 500 in ~0.3 s). Raise both if a GPU serves the model.
    if transformer:
        text = " ".join(text.split()[:60])
    n = 80 if transformer else 500

    def proba(texts):
        p = np.asarray(predict(list(texts)), dtype=float)
        return np.c_[1 - p, p]

    from lime.lime_text import LimeTextExplainer
    exp = LimeTextExplainer(class_names=["real", "fake"], random_state=42).explain_instance(
        text, proba, num_features=num_features, num_samples=n)
    out = {"model": name, "p_fake": round(float(proba([text])[0, 1]), 4),
           "tokens": [[w, round(float(v), 4)] for w, v in exp.as_list()], "num_samples": n,
           "ms": int((time.time() - t0) * 1000), "note": NOTE}
    if len(_cache) > 512:
        _cache.clear()
    _cache[key] = out
    return {**out, "cached": False}
