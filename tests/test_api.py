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
