"""Temporal / context check v1 (master §13.4).

old_news          = recency cue in the input AND (t_post - median(dates of SUPPORTING evidence)) >= N months
                    OR (v2, `explicit_date_check`) the main claim names its own date/period (spaCy DATE -> latest
                    parsed date, year-only = 1 July) AND (claim_date - median(SUPPORTING dates)) >= N months, i.e.
                    the claim presents as recent something the evidence dates much earlier
evidence dates    = htmldate publication dates (+ v2 `evidence_text_dates`: DATE entities inside the passages, §13.4)
out_of_context    = the main claim's GPE / year entities conflict with those of >= 2 supporting passages
mismatch_headline = NLI(premise = body, hypothesis = headline) is REFUTES with p >= mismatch_p
Dates: ISO strings or anything `dateparser` reads; nothing is invented — evidence without a date is simply not used.
"""
from __future__ import annotations

import re
from datetime import date, datetime
from functools import lru_cache
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


@lru_cache(maxsize=20_000)
def _ent_dates(text: str, base: date | None = None) -> tuple[date, ...]:
    """Dates named in `text` (spaCy DATE entities). Year-only -> 1 July of that year; relative phrases ("today",
    "last week") are resolved against `base`; unparseable or bare-number entities are skipped."""
    from src.claims.heuristics import nlp
    out = []
    for e in nlp()(text[:5000]).ents:
        if e.label_ != "DATE":
            continue
        y = YEAR.fullmatch(e.text.strip().removeprefix("in ").strip())
        if y:
            out.append(date(int(y.group(0)), 7, 1))
            continue
        if not (YEAR.search(e.text) or re.search(r"[A-Za-z]{3}", e.text)):
            continue
        import dateparser
        settings = {"PREFER_DAY_OF_MONTH": "first", "RETURN_AS_TIMEZONE_AWARE": False}
        if base:
            settings["RELATIVE_BASE"] = datetime(base.year, base.month, base.day)
        d = dateparser.parse(e.text, settings=settings)
        if d and 1900 < d.year < 2100:
            out.append(d.date())
    return tuple(out)


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
                   n_months: float | None = None, cfg: dict | None = None) -> dict:
    cfg = cfg or config()
    n_months = cfg["N_months"] if n_months is None else n_months
    t_post_d = parse_date(t_post) or date.today()
    supporting = [e for e in main_evidence if e.get("stance") == "SUPPORTS"]
    ev_dates = [d for d in (parse_date(e.get("date")) for e in supporting) if d]
    if cfg.get("evidence_text_dates"):
        ev_dates += [d for e in supporting for d in _ent_dates(e.get("snippet") or "", t_post_d) if d <= t_post_d]
    ev_dates = sorted(ev_dates)
    t_event = median_low(ev_dates) if ev_dates else None  # dates cannot be averaged -> lower median
    cue = has_recency_cue(" ".join(x for x in (title, input_text) if x))
    gap = round(months_between(t_post_d, t_event), 1) if t_event else None
    old_news = bool(cue and gap is not None and gap >= n_months)
    claim_dates = _ent_dates(main_claim, t_post_d) if main_claim and cfg.get("explicit_date_check") else []
    claim_date = max(claim_dates) if claim_dates else None
    date_conflict = bool(claim_date and t_event and months_between(claim_date, t_event) >= n_months)
    old_news = old_news or date_conflict

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
            "evidence_dates": [d.isoformat() for d in ev_dates],
            "claim_date": claim_date.isoformat() if claim_date else None, "date_conflict": date_conflict}
