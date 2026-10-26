"""Verdict fusion v1 (master §14): signals (§14.1) -> decision table R0-R9, first match wins (§14.2) -> confidence (§14.4).

Invariant (hard rule): evidence decides. ``p_fake`` is read in exactly two places — the R8 single-source tie-break and the
±clf_adj confidence adjustment for FAKE/REAL. No evidence -> UNVERIFIABLE (R1), never a classifier verdict.

Inputs (plain dicts, so the API, tests and the e2e evaluation share one code path):
    claims   = [{"text", "score" (c_i; main claim first), "evidence": [ev, ...]}, ...]
    ev       = {"stance": SUPPORTS|REFUTES|NEUTRAL, "p": float, "w_source": float, "tier": int,
                "cosine": float, "rating3": TRUE|MIXED|FALSE|None, "is_factcheck": bool, ...}
    temporal = {"old_news": bool, "out_of_context": bool, "mismatch_headline": bool}
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from statistics import mean

import yaml

FUSION_YAML = Path(__file__).with_name("fusion.yaml")
SIGN = {"SUPPORTS": 1, "REFUTES": -1, "NEUTRAL": 0}


@lru_cache(maxsize=1)
def config() -> dict:
    return yaml.safe_load(FUSION_YAML.read_text())


def signed(ev: dict) -> float:
    return SIGN[ev["stance"]] * ev["p"] * ev["w_source"]


def claim_signals(evidence: list[dict], cfg: dict) -> dict:
    s = sum(signed(e) for e in evidence)
    non_neutral = [e for e in evidence if e["stance"] != "NEUTRAL"]
    e_count = sum(e["w_source"] >= cfg["reliable_weight"] for e in non_neutral)
    stance = "SUPPORTS" if s >= cfg["tau"] else "REFUTES" if s <= -cfg["tau"] else "NEUTRAL"
    return {"S": round(s, 4), "E": e_count, "stance": stance, "n_evidence": len(evidence)}


def confidence(verdict: str, ev1: list[dict], sig1: dict, p_fake: float | None, cfg: dict) -> float:
    denom = sum(e["p"] * e["w_source"] for e in ev1)
    agreement = abs(sig1["S"]) / (denom + 1e-6)
    if verdict == "UNVERIFIABLE":
        return round(1 - max(agreement, 0.2), 4)
    coverage = min(sig1["E"], 4) / 4
    nn = [e["w_source"] for e in ev1 if e["stance"] != "NEUTRAL"]
    quality = mean(nn) if nn else 0.0
    base = 0.5 * agreement + 0.3 * coverage + 0.2 * quality
    adj = 0.0
    if p_fake is not None and verdict in ("FAKE", "REAL"):
        says_fake, says_real = p_fake >= cfg["clf_agree_fake"], p_fake <= cfg["clf_agree_real"]
        if (verdict == "FAKE" and says_fake) or (verdict == "REAL" and says_real):
            adj = cfg["clf_adj"]
        elif (verdict == "FAKE" and says_real) or (verdict == "REAL" and says_fake):
            adj = -cfg["clf_adj"]
    return round(min(max(base + adj, cfg["conf_min"]), cfg["conf_max"]), 4)


def fuse(claims: list[dict], temporal: dict | None = None, p_fake: float | None = None,
         input_is_satire: bool = False, is_english: bool = True, cfg: dict | None = None) -> dict:
    cfg = cfg or config()
    temporal = temporal or {}
    sigs = [claim_signals(c.get("evidence", []), cfg) for c in claims]
    ev1 = claims[0].get("evidence", []) if claims else []
    sig1 = sigs[0] if sigs else {"S": 0.0, "E": 0, "stance": "NEUTRAL", "n_evidence": 0}
    tau, k_min = cfg["tau"], cfg["min_evidence"]
    important = [(c, s) for c, s in zip(claims[1:], sigs[1:], strict=True) if c.get("score", 0) >= cfg["important_claim"]]
    all_ev = [e for c in claims for e in c.get("evidence", [])]
    main = sig1["stance"]
    flags = [f for f in ("old_news", "out_of_context", "mismatch_headline") if temporal.get(f)]

    def out(rule, verdict, reason, conf=None):
        return {"verdict": verdict, "rule_fired": rule, "reason": reason,
                "confidence": conf if conf is not None else confidence(verdict, ev1, sig1, p_fake, cfg),
                "signals": {"main_stance": main, "S_1": sig1["S"], "E_1": sig1["E"], "per_claim": sigs,
                            "temporal_flags": flags, "p_fake": p_fake}}

    if not is_english:
        return out("LANG", "UNVERIFIABLE", "English-only in this version.", conf=0.8)
    if input_is_satire:
        return out("R0", "FAKE", "The input originates from a satire / known-hoax site.", conf=cfg["satire_confidence"])
    if not all_ev or all(e["stance"] == "NEUTRAL" for e in all_ev) or sig1["E"] == 0:
        return out("R1", "UNVERIFIABLE", "No reliable evidence for or against the main claim was found. "
                                         "This does not mean the claim is false.")
    fc_false = [e for e in ev1 if e.get("is_factcheck") and e.get("tier") == 1 and e.get("rating3") == "FALSE"
                and e["stance"] == "REFUTES" and e.get("cosine", 0) >= cfg["factcheck_match_cosine"]]
    if fc_false:
        return out("R2", "FAKE", f"A fact-checker ({fc_false[0].get('source', 'Tier-1')}) rates a matching claim as false.")
    if main == "REFUTES" and sig1["E"] >= k_min:
        return out("R3", "FAKE", f"{sig1['E']} reliable sources contradict the main claim.")
    if main == "SUPPORTS" and flags:
        return out("R4", "MISLEADING", "The main claim is supported, but " + ", ".join(f.replace("_", " ") for f in flags)
                   + " was detected.")
    if main == "SUPPORTS" and any(s["S"] <= -tau and s["E"] >= k_min for _, s in important):
        return out("R5", "PARTIALLY TRUE", "The main claim is supported, but another important claim is refuted.")
    if main == "NEUTRAL" and any(s["S"] >= tau and s["E"] >= k_min for _, s in important) \
            and any(s["S"] <= -tau and s["E"] >= k_min for _, s in important):
        return out("R6", "PARTIALLY TRUE", "The important claims are split: some supported, some refuted.")
    if main == "SUPPORTS" and sig1["E"] >= k_min:
        return out("R7", "REAL", f"{sig1['E']} reliable sources support the main claim.")
    if main in ("SUPPORTS", "REFUTES") and sig1["E"] == 1:
        if p_fake is not None and main == "REFUTES" and p_fake >= cfg["tiebreak_fake"]:
            return out("R8", "FAKE", "One reliable source refutes the main claim; the style classifier breaks the tie.")
        if p_fake is not None and main == "SUPPORTS" and p_fake <= cfg["tiebreak_real"]:
            return out("R8", "REAL", "One reliable source supports the main claim; the style classifier breaks the tie.")
        return out("R8", "UNVERIFIABLE", "Only one source was found; this result could change as more coverage appears.")
    return out("R9", "UNVERIFIABLE", "The evidence is inconclusive.")
