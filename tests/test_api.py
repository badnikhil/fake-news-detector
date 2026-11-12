"""Offline index query + FastAPI endpoints in MODE=offline, with every socket connect blocked (proves no network).
Skips when `make index` has not been run or the models are not cached yet (first run needs the network once)."""
from __future__ import annotations

import importlib
import socket

import pytest

from src.evidence.offline_index import INDEX_PATH

pytestmark = pytest.mark.skipif(not INDEX_PATH.exists(), reason="offline index missing - run `make index`")
LIAR_CLAIM = "Says the unemployment rate doubled under Barack Obama."
LOCAL_CLAIM = "The bakery on Maple Street in Springfield ran out of croissants at 9 am last Tuesday."


@pytest.fixture(scope="module")
def client():
    mp = pytest.MonkeyPatch()
    mp.setenv("MODE", "offline")
    mp.setenv("HF_HUB_OFFLINE", "1")
    mp.setenv("TRANSFORMERS_OFFLINE", "1")
    import huggingface_hub.constants
    import transformers.utils.hub
    mp.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)  # both read the env only at import time
    mp.setattr(transformers.utils.hub, "_is_offline_mode", True)

    def no_network(*a, **k):
        raise OSError("network access attempted in MODE=offline")
    mp.setattr(socket.socket, "connect", no_network)
    from fastapi.testclient import TestClient

    from src.api import main
    importlib.reload(main)
    try:
        c = TestClient(main.app)
        c.post("/analyze", json={"text": LIAR_CLAIM})  # warm-up: loads MiniLM + NLI from the local cache
    except OSError as e:
        mp.undo()
        pytest.skip(f"models not cached locally yet: {e}")
    yield c
    mp.undo()


def test_index_query_returns_topk_with_cosine_and_ids():
    from src.evidence.offline_index import load, query
    index, meta = load()
    assert index.ntotal == len(meta) > 13_000
    hits = query(LIAR_CLAIM, k=5)
    assert len(hits) == 5 and all(h["record_id"] and 0 < h["cosine"] <= 1.0001 for h in hits)
    assert [h["cosine"] for h in hits] == sorted((h["cosine"] for h in hits), reverse=True)
    exact = meta[meta.dataset == "liar"].iloc[0]
    assert query(exact["text"], k=1)[0]["record_id"] == exact["record_id"]  # a stored statement finds itself


def test_health(client):
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["mode"] == "offline" and r["models"]["offline_index"]


def test_classify_reports_model(client):
    r = client.post("/classify", json={"text": "BREAKING: You won't believe what Hillary just did"})
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["p_fake"] <= 1 and body["model"] in ("distilbert-welfake-v1", "tfidf_lr_welfake_v0")


def test_analyze_offline_end_to_end(client):
    r = client.post("/analyze", json={"input_type": "headline", "text": LIAR_CLAIM, "mode": "online", "k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "offline"  # a server started offline never goes online
    assert body["verdict"] in ("REAL", "FAKE", "PARTIALLY TRUE", "MISLEADING", "UNVERIFIABLE")
    assert body["rule_fired"].startswith("R") and 0 <= body["confidence"] <= 1
    cards = body["claims"][0]["evidence"]
    assert cards and all(c["origin"] == "offline_index" and c["record_id"] for c in cards)
    assert all({"source", "date", "cosine", "stance", "p"} <= set(c) for c in cards)


def test_analyze_hyperlocal_claim_is_unverifiable(client):
    body = client.post("/analyze", json={"text": LOCAL_CLAIM}).json()
    assert (body["verdict"], body["rule_fired"]) == ("UNVERIFIABLE", "R1")
    assert body["evidence_count"] == 0  # nothing fabricated


def test_analyze_article_and_validation(client):
    art = ("Barack Obama said on Tuesday that the unemployment rate had doubled since 2008. "
           "Officials in Washington confirmed the figures, according to the Labor Department. "
           "Some people think this should never happen again.")
    body = client.post("/analyze", json={"input_type": "article", "title": "Unemployment doubled", "text": art}).json()
    assert body["claims"][0]["text"] == "Unemployment doubled" and len(body["claims"]) >= 2
    assert client.post("/analyze", json={"text": "   "}).status_code == 422
    assert client.post("/analyze", json={"input_type": "url", "url": "https://x.org"}).status_code == 422


def test_analyze_edge_cases_never_500(client):  # ESE §3.6 edge-case table
    bodies = [{"text": "😀🔥🔥😂"}, {"text": "12345 !!! ???"}, {"text": "भारत ने 2023 में चंद्रयान-3 को चंद्रमा पर उतारा।"},
              {"text": "The government announced a new tax policy today. " * 1200},
              {"input_type": "article", "text": "Wow. Really? Go!"},
              {"text": "BREAKING: earthquake hits Delhi today", "published": "not-a-date"}]
    for b in bodies:
        r = client.post("/analyze", json=b)
        assert r.status_code == 200, (b, r.text[:200])
        assert "timings_ms" in r.json()
    assert len(client.post("/analyze", json=bodies[3]).json()["claims"][0]["text"]) <= 1000  # capped headline
    assert client.post("/analyze", json={"input_type": "video", "text": "x"}).status_code == 422


def test_whitespace_title_does_not_hide_text(client):
    body = client.post("/analyze", json={"text": "The Eiffel Tower is in Paris.", "title": "   "}).json()
    assert body["claims"][0]["text"] == "The Eiffel Tower is in Paris."


def test_ablation_flags(client):
    from src.api import pipeline
    r = pipeline.analyze(text=LIAR_CLAIM, mode="offline", classifier_only=True)
    assert r["rule_fired"] == "CLF" and r["verdict"] in ("REAL", "FAKE") and r["claims"] == []
    r = pipeline.analyze(text=LIAR_CLAIM, mode="offline", use_evidence=False)
    assert (r["rule_fired"], r["evidence_count"]) == ("R1", 0)
    r = pipeline.analyze(text="BREAKING: " + LIAR_CLAIM, mode="offline", use_temporal=False)
    assert r["temporal"] == {}


def test_e2e_metrics():
    from src.eval.eval_e2e import metrics
    m = metrics(["REAL", "FAKE", "UNVERIFIABLE", "FAKE"], ["REAL", "UNVERIFIABLE", "UNVERIFIABLE", "FAKE"])
    assert m["accuracy"] == 0.75 and m["per_class"]["UNVERIFIABLE"]["precision"] == 0.5
    assert m["per_class"]["FAKE"]["recall"] == 0.5 and sum(map(sum, m["cm"])) == 4


def test_is_english_short_headlines():
    from src.api.pipeline import is_english
    assert is_english("UK Aldi supermarkets don't sell halal meat.")  # Live Claims LC023, was LANG before the fix
    assert not is_english("El gobierno anunció hoy que los impuestos bajarán el próximo año.")
    assert not is_english("भारत ने 2023 में चंद्रयान-3 को चंद्रमा पर उतारा।")


# ---------------------------------------------------------------- ESE front-end / explain endpoints
def test_config_reports_sources_and_thresholds(client):
    from src.fusion.rules import config as fusion_config
    r = client.get("/config").json()
    assert r["mode"] == "offline" and r["sources"] and r["thresholds"]["tau"] == fusion_config()["tau"] \
        and r["thresholds"]["N_months"] == fusion_config()["N_months"]
    assert "1" in r["source_tiers"] and "AIza" not in str(r)


def test_health_reports_versions(client):
    r = client.get("/health").json()
    assert r["version"] and r["models"]["clf"] and r["models"]["nli"]


def test_static_frontend_served(client):
    for path, kind in (("/", "text/html"), ("/app.js", "javascript"), ("/styles.css", "text/css")):
        r = client.get(path)
        assert r.status_code == 200 and kind in r.headers["content-type"]
    assert "Download JSON" in client.get("/").text


def test_analyze_has_explanation_and_offline_uncertainty(client):
    body = client.post("/analyze", json={"text": LOCAL_CLAIM}).json()
    assert body["explanation"].startswith("Marked UNVERIFIABLE") and "does not mean the claim is false" in body["uncertainty"]
    assert "Offline mode" in body["uncertainty"]


def test_explain_lime_tokens_and_cache(client):
    first = client.post("/explain", json={"text": LIAR_CLAIM}).json()
    assert first["tokens"] and len(first["tokens"]) <= 10 and all(isinstance(w, float) for _, w in first["tokens"])
    assert "do not decide the verdict" in first["note"] and 0 <= first["p_fake"] <= 1
    again = client.post("/explain", json={"text": LIAR_CLAIM}).json()
    assert again["cached"] and again["tokens"] == first["tokens"]
    assert client.post("/explain", json={"text": ""}).status_code == 422


CR_HTML = """<html><head><link rel="canonical" href="https://www.snopes.com/fact-check/x/">
<script type="application/ld+json">{"@context": "https://schema.org", "@graph": [{"@type": "WebPage"},
{"@type": "ClaimReview", "claimReviewed": "The moon is made of cheese.", "datePublished": "2026-01-02T10:00:00",
 "reviewRating": {"@type": "Rating", "alternateName": "False", "ratingValue": 1}}]}</script></head></html>"""


def test_parse_claimreview_and_mapping():
    from src.evidence.online import claimreview_rating3, parse_claimreview
    cr = parse_claimreview(CR_HTML)
    assert cr == {"canonical": "https://www.snopes.com/fact-check/x/", "claim_reviewed": "The moon is made of cheese.",
                  "rating": "False", "rating3": "FALSE", "date": "2026-01-02"}
    assert [claimreview_rating3(x) for x in ("Mostly True", "Pants on Fire!", "Mixture", "Outdated", "Unproven",
                                             "No Proof", "False. The video is from 2019.", None)] == \
        ["TRUE", "FALSE", "MIXED", "MIXED", None, None, "FALSE", None]
    assert parse_claimreview("<html>no json-ld</html>") == {}


def test_leakage_guard_drops_own_factcheck(monkeypatch):
    from src.api import pipeline
    assert pipeline.norm_url("https://www.Snopes.com/fact-check/x/amp/?utm=1#a") == "snopes.com/fact-check/x"
    own = {"origin": "ddgs", "url": "https://snopes.com/fact-check/x", "canonical": None, "w_source": 1.0, "cosine": 0.9}
    canon = {**own, "url": "https://m.example.org/y", "canonical": "https://www.snopes.com/fact-check/x/"}
    other = {**own, "url": "https://www.politifact.com/factchecks/2026/x/"}
    monkeypatch.setattr(pipeline, "offline_evidence", lambda *a: [own, canon, other])
    excluded = []
    ev = pipeline.gather_evidence("c", "offline", 5, {"max_evidence_per_claim": 10}, {},
                                  {pipeline.norm_url("https://www.snopes.com/fact-check/x/")}, excluded)
    assert ev == [other] and len(excluded) == 2


def test_parse_politifact_statement_block():
    from src.evidence.online import parse_claimreview
    html = ('<div class="pf-statement-quote mb-4"> <p>New York leads the nation in population loss.</p></div>'
            '<div class="pf-statement-meter d-flex"> <img src="x/meter-half-true.jpg" alt="Half-True"></div>')
    cr = parse_claimreview(html)
    assert (cr["claim_reviewed"], cr["rating"], cr["rating3"]) == ("New York leads the nation in population loss.",
                                                                   "Half True", "MIXED")


def test_gallery_examples_offline(client):
    """The front-end example gallery (app/examples.json) must keep returning its labelled class in MODE=offline -
    if a fusion retune breaks this, update the examples (run them offline, use what they really return)."""
    import json
    from pathlib import Path
    for ex in json.loads((Path(__file__).resolve().parents[1] / "app" / "examples.json").read_text()):
        req = {k: ex[k] for k in ("input_type", "text", "title") if k in ex}
        body = client.post("/analyze", json={**req, "mode": "offline"}).json()
        assert (body["verdict"], body["rule_fired"]) == (ex["expect"], ex["rule"]), ex["text"][:60]
