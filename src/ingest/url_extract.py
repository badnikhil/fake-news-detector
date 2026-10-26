"""URL -> {title, text, published, domain} via trafilatura + htmldate (master §9 ingest). Online mode only."""
from __future__ import annotations

from src.evidence.sources import domain_of


def extract(url: str) -> dict:
    import trafilatura
    from htmldate import find_date
    html = trafilatura.fetch_url(url)
    if not html:
        raise ValueError("extraction_failed")
    meta = trafilatura.extract_metadata(html)
    return {"title": getattr(meta, "title", None), "text": trafilatura.extract(html) or "",
            "published": find_date(html, original_date=True), "domain": domain_of(url)}
