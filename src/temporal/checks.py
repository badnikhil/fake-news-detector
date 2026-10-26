"""Temporal / context check v1 (master §13.4).

old_news          = recency cue in the input AND (t_post - median(dates of SUPPORTING evidence)) >= N months
out_of_context    = the main claim's GPE / year entities conflict with those of >= 2 supporting passages
mismatch_headline = NLI(premise = body, hypothesis = headline) is REFUTES with p >= mismatch_p
Dates: ISO strings or anything `dateparser` reads; nothing is invented — evidence without a date is simply not used.
"""
from __future__ import annotations

import re
from datetime import date, datetime
from statistics import median_low

from src.fusion.rules import config

YEAR = re.compile(r"\b(19|20)\d{2}\b")


def parse_date(s) -> date | None:
    if not s:
        return None
    if isinstance(s, datetime):
        return s.date()
    if isinstance(s, date):
        return s
    try:
        return date.fromisoformat(str(s)[:10])
    except ValueError:
        import dateparser
        d = dateparser.parse(str(s), settings={"PREFER_DAY_OF_MONTH": "first", "RETURN_AS_TIMEZONE_AWARE": False})
        return d.date() if d else None


def has_recency_cue(text: str, cues: list[str] | None = None) -> bool:
    cues = cues or config()["recency_cues"]
    return bool(re.search(r"\b(" + "|".join(map(re.escape, cues)) + r")\b", text or "", re.I))


def months_between(a: date, b: date) -> float:
    return (a - b).days / 30.4375


def _entities(text: str) -> tuple[set[str], set[str]]:
    from src.claims.heuristics import nlp
    doc = nlp()(text)
    gpe = {e.text.lower().removeprefix("the ") for e in doc.ents if e.label_ == "GPE"}
    years = {m.group(0) for e in doc.ents if e.label_ == "DATE" for m in YEAR.finditer(e.text)}
    return gpe, years


def _conflicts(claim: tuple[set, set], passage: tuple[set, set]) -> bool:
    """Conflict = for some entity type both sides name values and none of them overlap (fuzzy: substring)."""
    for c, p in zip(claim, passage, strict=True):
        if c and p and not any(a in b or b in a for a in c for b in p):
            return True
    return False


def temporal_check(input_text: str, main_evidence: list[dict], main_claim: str | None = None,
                   t_post=None, title: str | None = None, body: str | None = None,
                   n_months: float | None = None) -> dict:
    cfg = config()
    n_months = cfg["N_months"] if n_months is None else n_months
    t_post_d = parse_date(t_post) or date.today()
    supporting = [e for e in main_evidence if e.get("stance") == "SUPPORTS"]
    ev_dates = sorted(d for d in (parse_date(e.get("date")) for e in supporting) if d)
    t_event = median_low(ev_dates) if ev_dates else None  # dates cannot be averaged -> lower median
    cue = has_recency_cue(" ".join(x for x in (title, input_text) if x))
    gap = round(months_between(t_post_d, t_event), 1) if t_event else None
    old_news = bool(cue and gap is not None and gap >= n_months)

    out_of_context = False
    if main_claim and len(supporting) >= 2:
        ce = _entities(main_claim)
        if ce[0] or ce[1]:
            out_of_context = sum(_conflicts(ce, _entities(e.get("snippet", ""))) for e in supporting) >= 2

    mismatch, mismatch_p = False, None
    if title and body and body.strip() and body.strip() != title.strip():
        from src.stance.nli import stance
        st, p, _ = stance([(body[:2000], title)])[0]
        mismatch_p = round(p, 4) if st == "REFUTES" else None
        mismatch = st == "REFUTES" and p >= cfg["mismatch_p"]

    return {"old_news": old_news, "gap_months": gap, "recency_cue": cue, "out_of_context": out_of_context,
            "mismatch_headline": mismatch, "mismatch_p": mismatch_p, "input_date": t_post_d.isoformat(),
            "event_date": t_event.isoformat() if t_event else None,
            "evidence_dates": [d.isoformat() for d in ev_dates]}
