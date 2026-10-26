"""NLI stance (master §13.3): cross-encoder/nli-deberta-v3-small, premise = evidence, hypothesis = claim.
entailment -> SUPPORTS, contradiction -> REFUTES, neutral -> NEUTRAL; p = softmax probability of the argmax.
Runs on CPU by default (STANCE_DEVICE env to override)."""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
NLI_TO_STANCE = {"entailment": "SUPPORTS", "contradiction": "REFUTES", "neutral": "NEUTRAL"}
STANCES = ["SUPPORTS", "REFUTES", "NEUTRAL"]


@lru_cache(maxsize=1)
def model():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(NLI_MODEL, device=os.environ.get("STANCE_DEVICE", "cpu"), max_length=512)


def _order() -> list[str]:
    """Stance name per output column, read from the checkpoint's own id2label (never hard-coded)."""
    id2label = model().model.config.id2label
    return [NLI_TO_STANCE[id2label[i].lower()] for i in range(len(id2label))]


def stance_probs(pairs: list[tuple[str, str]], batch_size: int = 16) -> np.ndarray:
    """(evidence, claim) pairs -> array [n, 3] of probabilities in STANCES order."""
    if not pairs:
        return np.zeros((0, 3), dtype="float32")
    p = model().predict([list(x) for x in pairs], batch_size=batch_size, apply_softmax=True,
                        show_progress_bar=False, convert_to_numpy=True)
    cols = _order()
    return np.stack([p[:, cols.index(s)] for s in STANCES], axis=1)


def stance(pairs: list[tuple[str, str]], batch_size: int = 16) -> list[tuple[str, float, dict]]:
    """-> [(stance, p_stance, {SUPPORTS: p, REFUTES: p, NEUTRAL: p})]."""
    out = []
    for row in stance_probs(pairs, batch_size):
        j = int(row.argmax())
        out.append((STANCES[j], float(row[j]), {s: round(float(v), 4) for s, v in zip(STANCES, row, strict=True)}))
    return out
