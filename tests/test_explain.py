"""Explanation templates (src/explain/templates/rules.yaml) render for every rule without models or network."""
from src.explain.narrative import _templates, explanation, uncertainty


def _result(rule, verdict, stances=("SUPPORTS", "SUPPORTS"), temporal=None):
    ev = [{"stance": s, "source": "politifact.com", "source_label": "PolitiFact (LIAR)", "is_factcheck": True,
           "rating3": "FALSE", "rating": "false", "snippet": "Statement.", "cosine": 0.9} for s in stances]
    claims = [{"text": "Main claim.", "evidence": ev, "stance_summary": "x"}, {"text": "Second claim.", "evidence": []}]
    return {"verdict": verdict, "rule_fired": rule, "reason": "fallback", "claims": claims, "mode": "offline",
            "temporal": temporal or {}, "classifier": {"p_fake": 0.4},
            "signals": {"E_1": len(stances), "per_claim": [{"E": 2, "S": 1.0}, {"E": 2, "S": -1.0}]}}


def test_every_rule_has_a_template_that_renders():
    t = {"old_news": True, "event_date": "2018-08-17", "input_date": "2026-09-10", "gap_months": 96.8}
    for rule, verdict in (("LANG", "UNVERIFIABLE"), ("R0", "FAKE"), ("R1", "UNVERIFIABLE"), ("R2", "FAKE"),
                          ("R3", "FAKE"), ("R4", "MISLEADING"), ("R5", "PARTIALLY TRUE"), ("R6", "PARTIALLY TRUE"),
                          ("R7", "REAL"), ("R8", "UNVERIFIABLE"), ("R9", "UNVERIFIABLE")):
        stances = ("REFUTES", "REFUTES") if verdict == "FAKE" else ("SUPPORTS",) if rule == "R8" else ("SUPPORTS", "SUPPORTS")
        text = explanation(_result(rule, verdict, stances, t), url="https://www.theonion.com/x")
        assert rule in _templates() and text != "fallback" and "{{" not in text and '.".' not in text, (rule, text)
    assert "96.8 months" in explanation(_result("R4", "MISLEADING", temporal=t))


def test_uncertainty_always_present():
    assert "does not mean the claim is false" in uncertainty(_result("R1", "UNVERIFIABLE", ()))
    assert "Only one reliable source" in uncertainty(_result("R8", "FAKE", ("REFUTES",)))
    assert uncertainty(_result("R7", "REAL")).startswith("Based on the retrieved evidence")


def test_unknown_rule_falls_back_to_reason():
    assert explanation(_result("CLF", "FAKE")) == "fallback"
