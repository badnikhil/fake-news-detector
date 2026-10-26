"""Fusion rules R0-R9 + confidence (master §14) and the old-news check (§13.4). Pure Python, no models."""
from __future__ import annotations

import pytest

from src.fusion.rules import confidence, config, fuse
from src.temporal.checks import has_recency_cue, temporal_check


def ev(stance="SUPPORTS", p=0.9, w=0.8, tier=2, **kw):
    return {"stance": stance, "p": p, "w_source": w, "tier": tier, "cosine": 0.7, "rating3": None,
            "is_factcheck": False, "source": "x.com", **kw}


def claim(evidence, score=1.0):
    return {"text": "c", "score": score, "evidence": evidence}


def test_config_values_match_master_doc():
    c = config()
    assert (c["tau"], c["N_months"], c["k"], c["clf_adj"]) == (0.6, 6, 5, 0.10)


def test_lang_precheck():
    assert fuse([claim([ev(), ev()])], is_english=False)["verdict"] == "UNVERIFIABLE"


def test_r0_satire():
    r = fuse([claim([ev(), ev()])], input_is_satire=True)
    assert (r["rule_fired"], r["verdict"]) == ("R0", "FAKE")


@pytest.mark.parametrize("claims", [[], [claim([])], [claim([ev("NEUTRAL"), ev("NEUTRAL")])],
                                    [claim([ev(w=0.3), ev(w=0.3)])]])  # E_1 = 0: only unreliable sources
def test_r1_no_evidence_is_unverifiable(claims):
    r = fuse(claims, p_fake=0.99)
    assert (r["rule_fired"], r["verdict"]) == ("R1", "UNVERIFIABLE")


def test_r2_factchecker_false():
    fc = ev("REFUTES", p=0.95, w=1.0, tier=1, rating3="FALSE", is_factcheck=True, cosine=0.8)
    r = fuse([claim([fc])])
    assert (r["rule_fired"], r["verdict"]) == ("R2", "FAKE")
    fc["cosine"] = 0.7  # below factcheck_match_cosine -> no R2; single source -> R8
    assert fuse([claim([fc])])["rule_fired"] == "R8"


def test_r3_refuted():
    r = fuse([claim([ev("REFUTES"), ev("REFUTES")])])
    assert (r["rule_fired"], r["verdict"]) == ("R3", "FAKE")


def test_r4_misleading():
    r = fuse([claim([ev(), ev()])], temporal={"old_news": True})
    assert (r["rule_fired"], r["verdict"]) == ("R4", "MISLEADING")


def test_r5_partially_true():
    r = fuse([claim([ev(), ev()]), claim([ev("REFUTES"), ev("REFUTES")], score=0.7)])
    assert (r["rule_fired"], r["verdict"]) == ("R5", "PARTIALLY TRUE")
    # an unimportant claim (c_i < 0.6) does not trigger R5
    assert fuse([claim([ev(), ev()]), claim([ev("REFUTES"), ev("REFUTES")], score=0.3)])["rule_fired"] == "R7"


def test_r6_split_claims():
    main = claim([ev("SUPPORTS", p=0.5), ev("REFUTES", p=0.5)])  # S_1 = 0 -> NEUTRAL, E_1 = 2
    r = fuse([main, claim([ev(), ev()], 0.8), claim([ev("REFUTES"), ev("REFUTES")], 0.8)])
    assert (r["rule_fired"], r["verdict"]) == ("R6", "PARTIALLY TRUE")


def test_r7_real():
    r = fuse([claim([ev(), ev()])])
    assert (r["rule_fired"], r["verdict"]) == ("R7", "REAL")


def test_r8_single_source_tiebreak():
    one_ref, one_sup = [claim([ev("REFUTES", w=1.0)])], [claim([ev("SUPPORTS", w=1.0)])]
    assert fuse(one_ref, p_fake=0.9)["verdict"] == "FAKE"
    assert fuse(one_sup, p_fake=0.1)["verdict"] == "REAL"
    for c, p in ((one_ref, 0.5), (one_sup, 0.5), (one_ref, None), (one_ref, 0.1), (one_sup, 0.9)):
        r = fuse(c, p_fake=p)
        assert (r["rule_fired"], r["verdict"]) == ("R8", "UNVERIFIABLE")


def test_r9_inconclusive():
    r = fuse([claim([ev("SUPPORTS", p=0.5), ev("REFUTES", p=0.5)])])  # neutral main, nothing split
    assert (r["rule_fired"], r["verdict"]) == ("R9", "UNVERIFIABLE")


def test_invariant_classifier_never_flips_evidence_verdict():
    for p in (0.0, 0.5, 1.0):
        assert fuse([claim([ev(), ev()])], p_fake=p)["verdict"] == "REAL"
        assert fuse([claim([ev("REFUTES"), ev("REFUTES")])], p_fake=p)["verdict"] == "FAKE"


def test_confidence_formula_and_clf_adjustment():
    cfg = config()
    e = [ev(p=0.9, w=0.8), ev(p=0.9, w=0.8)]
    sig = {"S": 1.44, "E": 2}
    base = 0.5 * 1.0 + 0.3 * 0.5 + 0.2 * 0.8
    assert confidence("REAL", e, sig, None, cfg) == pytest.approx(base, abs=1e-3)
    assert confidence("REAL", e, sig, 0.1, cfg) == pytest.approx(base + 0.1, abs=1e-3)
    assert confidence("REAL", e, sig, 0.9, cfg) == pytest.approx(base - 0.1, abs=1e-3)
    assert confidence("UNVERIFIABLE", [], {"S": 0, "E": 0}, None, cfg) == pytest.approx(0.8)
    assert confidence("MISLEADING", e, sig, 0.9, cfg) == pytest.approx(base, abs=1e-3)  # no clf_adj


def test_old_news_flag():
    sup = [{"stance": "SUPPORTS", "date": "2018-08-17"}, {"stance": "SUPPORTS", "date": "2018-08-18"}]
    t = temporal_check("BREAKING: Kerala under water today", sup, t_post="2026-09-10")
    assert t["old_news"] and t["gap_months"] > 90 and t["event_date"] == "2018-08-17"
    assert not temporal_check("Kerala under water", sup, t_post="2026-09-10")["old_news"]  # no recency cue
    assert not temporal_check("BREAKING: Kerala", sup, t_post="2018-09-01")["old_news"]  # gap < 6 months
    assert has_recency_cue("Just in: x") and not has_recency_cue("Deliver the goods")  # 'live' is a whole word
