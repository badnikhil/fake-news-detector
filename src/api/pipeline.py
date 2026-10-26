"""End-to-end pipeline behind POST /analyze (master §9): claims -> evidence (offline index; + online sources unless
MODE=offline) -> NLI stance -> temporal check -> fusion R0-R9 -> JSON. Models load lazily, once, on CPU."""
from __future__ import annotations

import os
import time
import uuid
from functools import lru_cache
from pathlib import Path

from src.config import MODELS_DIR
from src.fusion.rules import config as fusion_config
from src.fusion.rules import fuse

DISTILBERT_DIR = MODELS_DIR / "distilbert-welfake-v1"
TFIDF_LR = MODELS_DIR / "tfidf_lr_welfake_v0.joblib"


# ---------------------------------------------------------------- content classifier (style signal only)
@lru_cache(maxsize=1)
def classifier():
    """-> (name, predict_fn text -> p_fake). DistilBERT checkpoint if present, else the TF-IDF + LR baseline."""
    if (DISTILBERT_DIR / "config.json").exists():
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(DISTILBERT_DIR)
        mdl = AutoModelForSequenceClassification.from_pretrained(DISTILBERT_DIR).eval()
        id2label = {int(k): str(v).lower() for k, v in mdl.config.id2label.items()}
        fake_idx = next((i for i, v in id2label.items() if "fake" in v), 1)  # project convention: 1 = fake

        def predict(text: str) -> float:
            with torch.no_grad():
                enc = tok(text, truncation=True, max_length=256, return_tensors="pt")
                return float(torch.softmax(mdl(**enc).logits, -1)[0, fake_idx])
        return "distilbert-welfake-v1", predict
    if TFIDF_LR.exists():
        import joblib
        pipe = joblib.load(TFIDF_LR)
        fake_idx = list(pipe.classes_).index("fake")
        return TFIDF_LR.stem, lambda text: float(pipe.predict_proba([text])[0, fake_idx])
    return None, None


def classify(text: str) -> dict:
    name, predict = classifier()
    if predict is None:
        return {"p_fake": None, "label": None, "model": None}
    p = round(predict(text), 4)
    return {"p_fake": p, "label": "fake" if p >= 0.5 else "real", "model": name}


# ---------------------------------------------------------------- evidence
def offline_evidence(claim: str, k: int, cfg: dict) -> list[dict]:
    from src.evidence.offline_index import query
    out = []
    for r in query(claim, k=k, min_cosine=cfg["offline_min_cosine"]):
        label = "PolitiFact (LIAR)" if r["dataset"] == "liar" else "PolitiFact (FakeNewsNet)"
        out.append({"origin": "offline_index", "source": "politifact.com", "source_label": label, "url": r.get("url") or None,
                    "title": None, "snippet": r["text"], "date": None, "tier": 1, "w_source": 1.0,
                    "cosine": r["cosine"], "rating": r["rating"], "rating3": r["rating3"], "is_factcheck": True,
                    "record_id": r["record_id"]})
    return out


def assign_stance(claim: str, evidence: list[dict], cfg: dict) -> None:
    """NLI stance per (evidence, claim); fact-check items whose statement the claim restates take the rating instead
    (p = factcheck_p, §13.3). In-place."""
    from src.stance.nli import stance
    if not evidence:
        return
    for ev, (st, p, probs) in zip(evidence, stance([(e["snippet"], claim) for e in evidence]), strict=True):
        ev["nli"] = probs
        if ev.get("is_factcheck") and ev.get("rating3"):
            restates = st == "SUPPORTS" or (st != "REFUTES" and ev["cosine"] >= cfg["factcheck_match_cosine"])
            if restates:  # the claim says what the fact-checked statement says -> its rating decides
                st = {"TRUE": "SUPPORTS", "FALSE": "REFUTES", "MIXED": "NEUTRAL"}[ev["rating3"]]
                p = cfg["factcheck_p"]
            elif st == "REFUTES" and ev["rating3"] != "TRUE":
                st, p = "NEUTRAL", probs["NEUTRAL"]  # negating a false/mixed statement proves nothing
        ev["stance"], ev["p"] = st, round(float(p), 4)


def gather_evidence(claim: str, mode: str, k: int, cfg: dict) -> list[dict]:
    ev = offline_evidence(claim, k, cfg)
    if mode not in ("offline", "lite"):  # MODE=offline must never touch the network
        from src.evidence.online import retrieve_online
        ev += retrieve_online(claim)
    ev = [e for e in ev if e["w_source"] > 0]  # satire items are dropped as evidence (§13.2 step 5)
    ev.sort(key=lambda e: -e["cosine"] * e["w_source"])
    return ev[: cfg["max_evidence_per_claim"]]


def is_english(text: str) -> bool:
    """langdetect on the first 500 chars (FR-13), backed by an English stop-word ratio because langdetect is unreliable
    on short headlines ("Barack Obama was born in Kenya." is detected as non-English)."""
    from spacy.lang.en.stop_words import STOP_WORDS
    words = text[:500].lower().split()
    if not words or sum(w.strip(".,!?;:\"'") in STOP_WORDS for w in words) / len(words) >= 0.2:
        return True
    try:
        from langdetect import DetectorFactory, detect
        DetectorFactory.seed = 0
        return detect(text[:500]) == "en"
    except Exception:  # noqa: BLE001 - undetectable (e.g. only digits) -> do not block the input
        return True


def analyze(text: str = "", title: str | None = None, url: str | None = None, published: str | None = None,
            input_type: str = "headline", mode: str | None = None, k: int | None = None) -> dict:
    t0 = time.time()
    cfg = fusion_config()
    mode = mode or os.environ.get("MODE", "online")
    k = k or cfg["k"]
    from src.claims.heuristics import identify_claims
    from src.evidence.sources import is_satire
    from src.temporal.checks import temporal_check

    if input_type == "url" and url and mode not in ("offline", "lite"):
        from src.ingest.url_extract import extract
        page = extract(url)
        title, text, published = title or page.get("title"), page.get("text") or text, published or page.get("published")
    full = "\n".join(x for x in (title, text) if x)
    english = is_english(full)
    satire = is_satire(url)
    claims = []
    if english and not satire:
        if input_type == "headline":
            h = (title or text).strip()
            claims = [{"text": h, "score": 1.0, "start": None, "end": None, "is_title": True}]
        else:
            claims = identify_claims(text, title=title, k=k)
        for c in claims:
            c["evidence"] = gather_evidence(c["text"], mode, k, cfg)
            assign_stance(c["text"], c["evidence"], cfg)
    clf = classify(full) if english else {"p_fake": None, "label": None, "model": None}
    main_ev = claims[0]["evidence"] if claims else []
    temporal = temporal_check(full, main_ev, main_claim=claims[0]["text"] if claims else None, t_post=published,
                              title=title if input_type != "headline" else None,
                              body=text if input_type != "headline" else None) if claims else {}
    fused = fuse(claims, temporal, p_fake=clf["p_fake"], input_is_satire=satire, is_english=english, cfg=cfg)
    for c in claims:
        sup = sum(e["stance"] == "SUPPORTS" for e in c["evidence"])
        ref = sum(e["stance"] == "REFUTES" for e in c["evidence"])
        c["stance_summary"] = f"{sup} supporting / {ref} refuting / {len(c['evidence']) - sup - ref} neutral"
    n_ev = sum(len(c["evidence"]) for c in claims)
    uncertainty = ("We could not find reliable evidence either way. This does not mean the claim is false."
                   if fused["verdict"] == "UNVERIFIABLE" else
                   "Only one reliable source was found; this result could change as more coverage appears."
                   if fused["signals"]["E_1"] == 1 else
                   "Based on the retrieved evidence only; evidence dates are shown where known.")
    return {"request_id": uuid.uuid4().hex[:12], "mode": mode, "verdict": fused["verdict"],
            "confidence": fused["confidence"], "rule_fired": fused["rule_fired"], "reason": fused["reason"],
            "claims": claims, "temporal": temporal, "classifier": clf, "signals": fused["signals"],
            "uncertainty": uncertainty, "evidence_count": n_ev, "latency_ms": int((time.time() - t0) * 1000)}


def model_status() -> dict:
    name = "distilbert-welfake-v1" if (DISTILBERT_DIR / "config.json").exists() else \
        (TFIDF_LR.stem if TFIDF_LR.exists() else None)
    from src.evidence.offline_index import INDEX_PATH
    return {"clf": name, "nli": "nli-deberta-v3-small", "embedder": "all-MiniLM-L6-v2",
            "offline_index": Path(INDEX_PATH).exists()}
