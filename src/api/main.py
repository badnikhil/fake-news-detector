"""FastAPI backend (master §16.2). Run: `make run` (= uvicorn src.api.main:app --port 8000); `MODE=offline make run`
for the no-network demo. In MODE=offline the HF hub is also put in offline mode, so models must already be cached
(`make index` caches MiniLM; the first stance run caches nli-deberta-v3-small)."""
from __future__ import annotations

import os

if os.environ.get("MODE") == "offline":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from typing import Literal  # noqa: E402

from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from src.api import pipeline  # noqa: E402
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
    return {"status": "ok", "mode": os.environ.get("MODE", "online"), "models": pipeline.model_status()}


@app.get("/config")
def get_config():
    c = fusion_config()
    return {"mode": os.environ.get("MODE", "online"),
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
        return pipeline.analyze(text=req.text, title=req.title, url=req.url, published=req.published,
                                input_type=req.input_type, mode=mode, k=req.k)
    except ValueError as e:
        if str(e) == "extraction_failed":
            raise HTTPException(502, {"error": "extraction_failed"}) from e
        raise
