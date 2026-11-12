"""Online evidence retrieval v1 (master §13.2): ddgs (DuckDuckGo) text search, Wikipedia search + intros, Google Fact
Check Tools `claims:search` (only when GOOGLE_FACTCHECK_API_KEY is set), page fetch with trafilatura + htmldate,
top-3 sentences per page by MiniLM cosine. Every network answer is cached in SQLite (CACHE_PATH, default
data/cache/evidence.sqlite) keyed by sha1(source|query) so demos / evaluations are repeatable and rate-limit-safe.

Never called in MODE=offline (the pipeline only uses `offline_index` there). Every source failure -> [] (NFR-8).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from src.config import CACHE_DIR
from src.evidence.sources import source_info

CACHE_PATH = os.environ.get("CACHE_PATH", str(CACHE_DIR / "evidence.sqlite"))
TIMEOUT = float(os.environ.get("SEARCH_TIMEOUT_S", 6))
UA = {"User-Agent": "FakeNewsDetector/0.2 (KIET B.Tech student project; contact: repo issues)"}
RATING_FALSE = re.compile(r"false|pants on fire|fake|fabricated|incorrect|hoax|scam|no evidence", re.I)
RATING_TRUE = re.compile(r"^(true|correct|accurate|mostly true)\b", re.I)
CLAIMREVIEW = os.environ.get("CLAIMREVIEW", "1") != "0"  # parse ClaimReview on Tier-1 result pages (v2, 2026-11-12)
STATS = {"hit": 0, "miss": 0}  # cache hits / misses since import (latency eval separates cached and uncached requests)


def _db() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    con = sqlite3.connect(CACHE_PATH)
    con.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT, created REAL)")
    return con


def cached(source: str, query: str, fn):
    key = hashlib.sha1(f"{source}|{query}".encode()).hexdigest()
    with _db() as con:
        row = con.execute("SELECT value FROM cache WHERE key=?", (key,)).fetchone()
    if row:
        STATS["hit"] += 1
        return json.loads(row[0])
    STATS["miss"] += 1
    try:
        value = fn()
    except Exception:  # noqa: BLE001 - external source failures must never break the pipeline
        return []      # not cached: a transient failure should be retried next time
    with _db() as con:
        con.execute("INSERT OR REPLACE INTO cache VALUES (?,?,?)", (key, json.dumps(value), time.time()))
    return value


def rating3(text: str | None) -> str | None:
    if not text:
        return None
    if RATING_TRUE.search(text.strip()):
        return "TRUE"
    if RATING_FALSE.search(text):
        return "FALSE"
    return "MIXED"


def ddg_search(q: str, n: int = 8) -> list[dict]:
    def run():
        from ddgs import DDGS
        return [{"title": r.get("title"), "url": r.get("href"), "snippet": r.get("body")}
                for r in DDGS(timeout=int(TIMEOUT)).text(q, max_results=n)]
    return cached("ddgs", q, run)


def wikipedia_search(q: str, n: int = 2) -> list[dict]:
    def run():
        import requests
        api = "https://en.wikipedia.org/w/api.php"
        hits = requests.get(api, params={"action": "query", "list": "search", "srsearch": q, "srlimit": n,
                                         "format": "json"}, headers=UA, timeout=TIMEOUT).json()["query"]["search"]
        if not hits:
            return []
        titles = "|".join(h["title"] for h in hits)
        pages = requests.get(api, params={"action": "query", "prop": "extracts", "exintro": 1, "explaintext": 1,
                                          "titles": titles, "format": "json"}, headers=UA,
                             timeout=TIMEOUT).json()["query"]["pages"]
        return [{"title": p["title"], "url": "https://en.wikipedia.org/wiki/" + p["title"].replace(" ", "_"),
                 "text": p.get("extract", "")} for p in pages.values() if p.get("extract")]
    return cached("wikipedia", q, run)


def factcheck_search(q: str) -> list[dict]:
    key = os.environ.get("GOOGLE_FACTCHECK_API_KEY")
    if not key:
        return []

    def run():
        import requests
        r = requests.get("https://factchecktools.googleapis.com/v1alpha1/claims:search",
                         params={"query": q, "languageCode": "en", "pageSize": 5, "key": key}, timeout=TIMEOUT)
        out = []
        for c in r.json().get("claims", []):
            for rev in c.get("claimReview", [])[:1]:
                out.append({"claim": c.get("text"), "url": rev.get("url"), "rating": rev.get("textualRating"),
                            "date": (rev.get("reviewDate") or c.get("claimDate") or "")[:10] or None,
                            "publisher": (rev.get("publisher") or {}).get("site")})
        return out
    return cached("factcheck", q, run)  # the key is not part of the cache key


def fetch_page(url: str) -> dict:
    def run():
        import trafilatura
        from htmldate import find_date
        html = trafilatura.fetch_url(url)
        if not html:
            return {}
        return {"text": trafilatura.extract(html) or "", "date": find_date(html, original_date=True)}
    return cached("page", url, run) or {}


# ClaimReview rating -> rating3 with the LIAR / Live Claims mapping (data/live_claims/README.md): True / Mostly True ->
# TRUE; False / Pants on Fire / Fake -> FALSE; Half True, Mixture, Mostly False, Miscaptioned, Outdated, Misleading,
# Missing context -> MIXED; evidence-absence ratings (Unproven, Unverified, No Proof, ...) -> None (no stance).
CR_TRUE = {"true", "mostly true", "correct", "accurate", "correct attribution"}
CR_FALSE = {"false", "pants on fire", "pants on fire!", "fake", "fabricated", "scam", "hoax", "satire", "labeled satire"}
CR_NONE = re.compile(r"unproven|unverified|unfounded|no proof|no evidence|no reports|not official|what we know|"
                     r"research in progress|legend|explainer", re.I)


def claimreview_rating3(name: str | None) -> str | None:
    n = (name or "").strip().rstrip(".!").lower()
    if not n or CR_NONE.search(n):
        return None
    if n in CR_TRUE:
        return "TRUE"
    if n in CR_FALSE or n.split()[0].strip(".,:;!") in ("false", "fake"):  # Full Fact / BOOM free text: "False. ..."
        return "FALSE"
    return "MIXED"


def _text(v) -> str | None:  # JSON-LD fields are sometimes lists / objects
    while isinstance(v, list):
        v = v[0] if v else None
    return v.strip() if isinstance(v, str) and v.strip() else None


def parse_claimreview(html: str) -> dict:
    """schema.org ClaimReview JSON-LD + <link rel=canonical> of a fact-check page -> {canonical, claim_reviewed,
    rating, rating3, date}; {} parts when absent. Nothing is inferred from free text."""
    out = {}
    m = re.search(r'<link[^>]+rel=["\']canonical["\'][^>]*href=["\']([^"\']+)', html, re.I) or \
        re.search(r'<link[^>]+href=["\']([^"\']+)["\'][^>]*rel=["\']canonical', html, re.I)
    if m:
        out["canonical"] = m.group(1)
    stack = []
    for blob in re.findall(r'<script[^>]+application/ld\+json[^>]*>(.*?)</script>', html, re.S | re.I):
        try:
            stack.append(json.loads(blob.strip()))
        except ValueError:
            continue
    while stack:
        x = stack.pop()
        if isinstance(x, list):
            stack += x
        elif isinstance(x, dict):
            t = x.get("@type")
            if t == "ClaimReview" or (isinstance(t, list) and "ClaimReview" in t):
                rr = x.get("reviewRating") or {}
                rr = rr[0] if isinstance(rr, list) and rr else rr
                name = rr.get("alternateName") or rr.get("name") if isinstance(rr, dict) else None
                if not name and isinstance(rr, dict) and rr.get("ratingValue") not in (None, ""):
                    try:  # numeric scale only when no rating name is given
                        v, lo, hi = (float(rr.get(k)) for k in ("ratingValue", "worstRating", "bestRating"))
                        f = (v - lo) / (hi - lo)
                        name = "true" if f >= 0.8 else "false" if f <= 0.2 else "mixed"
                    except (TypeError, ValueError, ZeroDivisionError):
                        pass
                out.update(claim_reviewed=_text(x.get("claimReviewed")), rating=_text(name), rating3=claimreview_rating3(name),
                           date=(x.get("datePublished") or "")[:10] or None)
                return out
            stack += [v for v in x.values() if isinstance(v, (dict, list))]
    # politifact.com publishes no ClaimReview JSON-LD; its statement block + Truth-O-Meter image alt carry the same data
    q = re.search(r'pf-statement-quote[^>]*>\s*<p>(.*?)</p>', html, re.S)
    mtr = re.search(r'pf-statement-meter.*?<img[^>]+alt="([^"]+)"', html, re.S)
    if q and mtr:
        name = mtr.group(1).replace("-", " ").strip()
        out.update(claim_reviewed=re.sub(r"<[^>]+>|\s+", " ", q.group(1)).strip(), rating=name,
                   rating3=claimreview_rating3(name), date=None)
    return out


def fetch_claimreview(url: str) -> dict:
    """Fact-check pages only (Tier 1): ClaimReview + canonical URL, cached separately from `fetch_page`."""
    def run():
        import trafilatura
        html = trafilatura.fetch_url(url)
        return parse_claimreview(html) if html else {}
    return cached("claimreview", url, run) or {}


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text or "") if 25 <= len(s.strip()) <= 600]


def best_sentences(claim_vec: np.ndarray, text: str, n: int = 3) -> tuple[str, float]:
    from src.evidence.offline_index import embed
    sents = _sentences(text)[:200]
    if not sents:
        return "", 0.0
    sims = embed(sents) @ claim_vec
    top = sorted(np.argsort(-sims)[:n])
    return " ".join(sents[i] for i in top), float(sims.max())


def keyword_query(claim: str) -> str:
    from src.claims.heuristics import nlp
    doc = nlp()(claim)
    keep = [t.text for t in doc if t.ent_type_ or t.like_num or (t.pos_ in ("PROPN", "NOUN", "VERB") and not t.is_stop)]
    return " ".join(dict.fromkeys(keep)) or claim


def retrieve_online(claim: str, max_pages: int = 5) -> list[dict]:
    """-> evidence dicts {origin, source, url, title, snippet, date, tier, w_source, cosine, rating, rating3, is_factcheck}."""
    from src.evidence.offline_index import embed
    q_full = " ".join(claim.split()[:15])
    q_kw = keyword_query(claim)
    with ThreadPoolExecutor(4) as ex:
        f_ddg = ex.submit(ddg_search, q_full)
        f_ddg_fc = ex.submit(ddg_search, q_kw + " fact check")
        f_wiki = ex.submit(wikipedia_search, q_kw)
        f_gfc = ex.submit(factcheck_search, q_kw)
        web = {h["url"]: h for h in f_ddg.result() + f_ddg_fc.result() if h.get("url")}
        wiki, gfc = f_wiki.result(), f_gfc.result()
        pages = dict(zip(list(web)[:max_pages], ex.map(fetch_page, list(web)[:max_pages]), strict=True))
    cvec = embed([claim])[0]
    out = []
    fc_urls = [u for u in web if source_info(u)[1] == 1] if CLAIMREVIEW else []
    with ThreadPoolExecutor(4) as ex:
        reviews = dict(zip(fc_urls, ex.map(fetch_claimreview, fc_urls), strict=True))
    for url, h in web.items():
        page = pages.get(url) or {}
        text = page.get("text") or h.get("snippet") or ""
        passage, cos = best_sentences(cvec, text)
        cr = reviews.get(url) or {}
        cr = {**cr, "claim_reviewed": _text(cr.get("claim_reviewed"))}
        if cr["claim_reviewed"] and cr.get("rating3"):  # rated fact-check: same gate as LIAR / GFC items
            cos = float(embed([cr["claim_reviewed"]])[0] @ cvec)
            it = _item("ddgs", url, h.get("title"), cr["claim_reviewed"], cr.get("date") or page.get("date"), cos)
            it.update(rating=cr.get("rating"), rating3=cr["rating3"], is_factcheck=True, canonical=cr.get("canonical"))
            out.append(it)
        elif passage:
            date = None if "wikipedia.org" in url else page.get("date")  # wiki htmldate = page creation, not event
            out.append(_item("ddgs", url, h.get("title"), passage, date, cos) | {"canonical": cr.get("canonical")})
    for w in wiki:
        passage, cos = best_sentences(cvec, w["text"])
        if passage:
            out.append(_item("wikipedia", w["url"], w["title"], passage, None, cos))
    for g in gfc:
        if g.get("claim"):
            cos = float(embed([g["claim"]])[0] @ cvec)
            it = _item("factcheck", g.get("url") or g.get("publisher") or "", g.get("publisher"), g["claim"],
                       g.get("date"), cos)
            it.update(rating=g.get("rating"), rating3=rating3(g.get("rating")), is_factcheck=True)
            out.append(it)
    return out


def _item(origin, url, title, snippet, date, cos) -> dict:
    dom, tier, w = source_info(url)
    return {"origin": origin, "source": dom, "url": url, "title": title, "snippet": snippet, "date": date,
            "tier": tier, "w_source": w, "cosine": round(cos, 4), "rating": None, "rating3": None,
            "is_factcheck": False, "record_id": None}
