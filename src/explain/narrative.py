"""Plain-language explanation (one Jinja2 template per fired rule, templates/rules.yaml) and the mandatory
uncertainty sentence (master §15). Input = the /analyze result dict; output = text only, never new facts."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from jinja2 import Environment

TEMPLATES = Path(__file__).parent / "templates" / "rules.yaml"


@lru_cache(maxsize=1)
def _templates() -> dict:
    env = Environment(autoescape=False, trim_blocks=True)  # plain text; the UI inserts it with textContent
    return {k: env.from_string(v) for k, v in yaml.safe_load(TEMPLATES.read_text()).items()}


def _sources(ev: list[dict], stance: str) -> str:
    names = list(dict.fromkeys(e.get("source_label") or e.get("source") or "unknown" for e in ev if e["stance"] == stance))
    return ", ".join(names[:3]) + (f" +{len(names) - 3} more" if len(names) > 3 else "")


def _q(text: str) -> str:
    """Quoted text without its own final full stop (the template sentence supplies punctuation)."""
    return text.strip().rstrip(".")


def _context(r: dict, url: str | None = None) -> dict:
    claims = r.get("claims") or []
    ev = claims[0]["evidence"] if claims else []
    per = (r.get("signals") or {}).get("per_claim") or []
    split = [{"text": _q(c["text"]), "n": s["E"], "S": s["S"]} for c, s in zip(claims[1:], per[1:], strict=False)]
    from src.evidence.sources import domain_of
    fc = next((e for e in ev if e.get("is_factcheck") and e.get("rating3") == "FALSE" and e["stance"] == "REFUTES"), {})
    return {"verdict": r["verdict"], "claim": _q(claims[0]["text"]) if claims else "", "n_ev": len(ev),
            "n_sup": sum(e["stance"] == "SUPPORTS" for e in ev), "n_ref": sum(e["stance"] == "REFUTES" for e in ev),
            "sup_sources": _sources(ev, "SUPPORTS"), "ref_sources": _sources(ev, "REFUTES"),
            "fc": {**fc, "snippet": _q(fc.get("snippet", ""))},
            "t": r.get("temporal") or {}, "p_fake": (r.get("classifier") or {}).get("p_fake"),
            "summary": claims[0].get("stance_summary", "") if claims else "",
            "refuted": min(split, key=lambda c: c["S"], default={"text": "", "n": 0}),
            "supported": max(split, key=lambda c: c["S"], default={"text": "", "n": 0}),
            "domain": domain_of(url) if url else None}


def explanation(result: dict, url: str | None = None) -> str:
    t = _templates().get(result["rule_fired"])
    try:
        return " ".join(t.render(**_context(result, url)).split()) if t else result["reason"]
    except Exception:  # noqa: BLE001 - an additive text field must never break /analyze; fall back to the rule reason
        return result["reason"]


def uncertainty(result: dict) -> str:
    verdict, sig, t = result["verdict"], result.get("signals") or {}, result.get("temporal") or {}
    if verdict == "UNVERIFIABLE":
        s = "We could not find reliable evidence either way. This does not mean the claim is false."
    elif sig.get("E_1") == 1:
        s = "Only one reliable source was found; this result could change as more coverage appears."
    elif t.get("old_news"):
        s = (f"The underlying event is verified, but the supporting reports date from {t['event_date']}, "
             f"{t['gap_months']} months before this input.")
    else:
        s = "Based on the retrieved evidence only; evidence dates are shown where known."
    if result.get("mode") == "offline":
        s += (" Offline mode: only the bundled PolitiFact fact-check index (LIAR + FakeNewsNet, ratings up to about"
              " 2018) was searched, so recent events cannot be checked.")
    return s
