"""Source-reliability lookup over `sources.yaml` (master §14.3): url/domain -> (domain, tier, w_s)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

import yaml

SOURCES_YAML = Path(__file__).with_name("sources.yaml")


@lru_cache(maxsize=1)
def _cfg() -> dict:
    return yaml.safe_load(SOURCES_YAML.read_text())


def domain_of(url_or_domain: str) -> str:
    s = (url_or_domain or "").strip().lower()
    host = urlparse(s if "//" in s else "//" + s).hostname or ""
    return host[4:] if host.startswith("www.") else host


def source_info(url_or_domain: str) -> tuple[str, int, float]:
    """Return (domain, tier, weight). Unknown domains -> default tier 4 / 0.3."""
    cfg = _cfg()
    dom = domain_of(url_or_domain)
    low = (url_or_domain or "").lower()
    if any(p in low for p in cfg.get("path_tier1", [])):
        return dom, 1, cfg["tiers"][1]["weight"]
    for tier in (0, 1, 2, 3):  # satire first so a hoax sub-domain never inherits a parent's weight
        for d in cfg["tiers"][tier]["domains"]:
            if d.startswith("*."):
                if dom.endswith(d[1:]):
                    return dom, tier, cfg["tiers"][tier]["weight"]
            elif dom == d or dom.endswith("." + d):
                return dom, tier, cfg["tiers"][tier]["weight"]
    return dom, cfg["default"]["tier"], cfg["default"]["weight"]


def is_satire(url_or_domain: str | None) -> bool:
    return bool(url_or_domain) and source_info(url_or_domain)[1] == 0
