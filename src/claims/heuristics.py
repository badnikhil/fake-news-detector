"""Claim identifier v1 (master §13.1): spaCy en_core_web_sm sentence split + the heuristic check-worthiness score h,
normalised to [0, 1] (no ClaimBuster classifier yet, so score = h_norm), MiniLM near-duplicate removal (cos > 0.85),
top-k with character offsets. The headline (if given) is always claim #1 with score 1.0."""
from __future__ import annotations

import re
from functools import lru_cache

import numpy as np

NE_STRONG = {"PERSON", "ORG", "GPE", "EVENT", "NORP", "LOC"}
NE_NUMERIC = {"CARDINAL", "PERCENT", "MONEY", "DATE", "QUANTITY"}
REPORTING = re.compile(r"\b(said|says|announced|confirmed|claimed|claims|reported|according to|stated)\b", re.I)
OPINION = re.compile(r"\b(i think|i believe|we believe|in my opinion|should|must|ought to)\b", re.I)
HEDGE = re.compile(r"\b(may|might|could|perhaps|possibly)\b", re.I)
NUMBERS = re.compile(r"\d|\b(more|less|fewer|higher|lower|most|least|biggest|largest|smallest)\b|\w+er than\b", re.I)
H_MAX = 1.0 + 0.8 + 0.6 + 0.5 + 0.4  # max positive h -> h_norm in [0, 1]


@lru_cache(maxsize=1)
def nlp():
    import spacy
    return spacy.load("en_core_web_sm")


def heuristic_score(sent, position: int) -> float:
    """h of §13.1 for one spaCy Span; `position` = 0-based sentence index in the body."""
    labels = {e.label_ for e in sent.ents}
    t = sent.text
    h = (1.0 * bool(labels & NE_STRONG) + 0.8 * bool(labels & NE_NUMERIC) + 0.6 * bool(REPORTING.search(t))
         + 0.5 * (position < 3) + 0.4 * bool(NUMBERS.search(t))
         - 0.8 * bool(OPINION.search(t)) - 0.5 * bool(HEDGE.search(t)))
    return float(np.clip(h / H_MAX, 0.0, 1.0))


def _is_question_or_imperative(sent) -> bool:
    if sent.text.strip().endswith("?"):
        return True
    first = next((t for t in sent if not t.is_punct and not t.is_space), None)
    return first is not None and first.tag_ == "VB" and first.dep_ == "ROOT"  # bare-verb opener = imperative


def identify_claims(text: str, title: str | None = None, k: int = 5, dedupe: bool = True) -> list[dict]:
    """-> [{text, score, start, end, is_title}] sorted: title first, then by score desc."""
    claims = []
    if title and title.strip():
        claims.append({"text": title.strip(), "score": 1.0, "start": None, "end": None, "is_title": True})
    if text and text.strip() and text.strip() != (title or "").strip():
        doc = nlp()(text[:20_000])
        sents = list(doc.sents)
        if not claims and len(sents) == 1:  # a bare headline passed as text: keep it whatever its length
            s = sents[0]
            return [{"text": s.text.strip(), "score": 1.0, "start": s.start_char, "end": s.end_char, "is_title": False}]
        cands = []
        for i, s in enumerate(sents):
            n_tok = sum(not t.is_punct and not t.is_space for t in s)
            if n_tok < 6 or _is_question_or_imperative(s):
                continue
            cands.append({"text": s.text.strip(), "score": round(heuristic_score(s, i), 4),
                          "start": s.start_char, "end": s.end_char, "is_title": False})
        cands.sort(key=lambda c: -c["score"])
        claims += cands
    if dedupe and len(claims) > 1:
        claims = _dedupe(claims[: 4 * k])
    return claims[:k]


def _dedupe(claims: list[dict], thr: float = 0.85) -> list[dict]:
    from src.evidence.offline_index import embed
    vecs = embed([c["text"] for c in claims])
    keep: list[int] = []
    for i in range(len(claims)):
        if all(float(vecs[i] @ vecs[j]) <= thr for j in keep):
            keep.append(i)
    return [claims[i] for i in keep]
