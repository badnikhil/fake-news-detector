"""FastAPI backend (master §16.2). Run: `make run` (= uvicorn src.api.main:app --port 8000); `MODE=offline make run`
for the no-network demo. In MODE=offline the HF hub is also put in offline mode, so models must already be cached
(`make index` caches MiniLM; the first stance run caches nli-deberta-v3-small)."""
from __future__ import annotations

import os

if os.environ.get("MODE") == "offline":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from pathlib import Path  # noqa: E402
from typing import Literal  # noqa: E402

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from src.api import pipeline  # noqa: E402
from src.explain.narrative import explanation, uncertainty  # noqa: E402
from src.fusion.rules import config as fusion_config  # noqa: E402

app = FastAPI(title="Fake News & Misinformation Detector", version="0.2-mse2")


class ClassifyIn(BaseModel):
    text: str = Field(min_length=1)


class AnalyzeIn(BaseModel):
    input_type: Literal["headline", "article", "url"] = "headline"
    text: str = ""
    title: str | None = None
    url: str | None = None
    published: str | None = None
    mode: Literal["online", "offline", "lite"] | None = None
    k: int | None = Field(default=None, ge=1, le=10)


def _mode(req_mode: str | None) -> str:
    env = os.environ.get("MODE", "online")
    return "offline" if env == "offline" else (req_mode or env)  # a server started offline stays offline


@app.get("/health")
def health():
    return {"status": "ok", "mode": os.environ.get("MODE", "online"), "version": app.version,
            "models": pipeline.model_status()}


@app.get("/config")
def get_config():
    from src.evidence.sources import _cfg as source_cfg
    c, mode = fusion_config(), os.environ.get("MODE", "online")
    online = ["DuckDuckGo (ddgs)", "Wikipedia REST API"] + \
        (["Google Fact Check Tools API"] if os.environ.get("GOOGLE_FACTCHECK_API_KEY") else [])  # key presence only
    return {"mode": mode,
            "sources": ["Offline fact-check index (LIAR + FakeNewsNet PolitiFact)"] + ([] if mode == "offline" else online),
            "source_tiers": {str(t): v for t, v in source_cfg()["tiers"].items()},
            "thresholds": {"tau": c["tau"], "N_months": c["N_months"], "k": c["k"]}}


@app.post("/classify")
def classify(req: ClassifyIn):
    out = pipeline.classify(req.text)
    if out["p_fake"] is None:
        raise HTTPException(503, "no classifier checkpoint found in data/models/")
    out["note"] = "Writing-style signal only; it does not decide the verdict."
    return out


@app.post("/analyze")
def analyze(req: AnalyzeIn):
    mode = _mode(req.mode)
    if req.input_type == "url":
        if not req.url:
            raise HTTPException(422, "url is required for input_type=url")
        if mode == "offline":
            raise HTTPException(422, "URL input needs online mode; paste the text instead")
    elif not (req.text.strip() or (req.title or "").strip()):
        raise HTTPException(422, "text is empty")
    try:
        out = pipeline.analyze(text=req.text, title=req.title, url=req.url, published=req.published,
                               input_type=req.input_type, mode=mode, k=req.k)
    except ValueError as e:
        if str(e) == "extraction_failed":
            raise HTTPException(502, {"error": "extraction_failed"}) from e
        raise
    out["explanation"] = explanation(out, req.url)  # additive: per-rule Jinja2 template (src/explain/templates/)
    out["uncertainty"] = uncertainty(out)  # same sentences as before + temporal / offline-mode notes
    return out


class ExplainIn(BaseModel):
    text: str = Field(min_length=1)


@app.post("/explain")
def explain(req: ExplainIn):
    """LIME token weights of the style classifier (positive = towards "fake"); cached per input. Never decides."""
    from src.explain.lime_explain import explain as lime_explain
    out = lime_explain(req.text)
    if out["model"] is None:
        raise HTTPException(503, "no classifier checkpoint found in data/models/")
    return out


APP_DIR = Path(__file__).resolve().parents[2] / "app"
if APP_DIR.is_dir():  # plain HTML/CSS/JS front-end; mounted last so every API route above wins
    app.mount("/", StaticFiles(directory=APP_DIR, html=True), name="app")
