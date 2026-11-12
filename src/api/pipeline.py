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
MAX_HEADLINE_CHARS = 1000


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

        def predict(text):  # str -> float; list[str] -> list[float] (batched, used by LIME in src/explain)
            texts = [text] if isinstance(text, str) else list(text)
            out = []
            with torch.no_grad():
                for i in range(0, len(texts), 32):
                    enc = tok(texts[i:i + 32], truncation=True, max_length=256, padding=True, return_tensors="pt")
                    out += torch.softmax(mdl(**enc).logits, -1)[:, fake_idx].tolist()
            return out[0] if isinstance(text, str) else out
        return "distilbert-welfake-v1", predict
    if TFIDF_LR.exists():
        import joblib
        pipe = joblib.load(TFIDF_LR)
        fake_idx = list(pipe.classes_).index("fake")

        def predict_lr(text):  # str -> float; list[str] -> list[float]
            p = pipe.predict_proba([text] if isinstance(text, str) else list(text))[:, fake_idx].tolist()
            return p[0] if isinstance(text, str) else p
        return TFIDF_LR.stem, predict_lr
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


def norm_url(u: str | None) -> str:
    """Scheme-, www-, query-, fragment-, trailing-slash- and /amp-insensitive form of a URL (leakage guard)."""
    from urllib.parse import urlsplit
    if not u:
        return ""
    x = urlsplit(u.strip() if "://" in u else "https://" + u.strip())
    path = x.path.rstrip("/").removesuffix("/amp")
    return (x.netloc.lower().removeprefix("www.") + path).lower()


def gather_evidence(claim: str, mode: str, k: int, cfg: dict, timings: dict | None = None,
                    exclude_urls: set[str] | None = None, excluded: list | None = None) -> list[dict]:
    timings = {} if timings is None else timings
    t = time.perf_counter()
    ev = offline_evidence(claim, k, cfg)
    timings["evidence_offline"] = timings.get("evidence_offline", 0) + _ms(t)
    if mode not in ("offline", "lite"):  # MODE=offline must never touch the network
        from src.evidence.online import retrieve_online
        t = time.perf_counter()
        ev += retrieve_online(claim)
        timings["evidence_online"] = timings.get("evidence_online", 0) + _ms(t)
    ev = [e for e in ev if e["w_source"] > 0]  # satire items are dropped as evidence (§13.2 step 5)
    if exclude_urls:  # evaluation leakage guard: never use the item's own fact-check page (URL or canonical URL)
        drop = [e for e in ev if {norm_url(e.get("url")), norm_url(e.get("canonical"))} & exclude_urls]
        ev = [e for e in ev if e not in drop]
        if excluded is not None:
            excluded += [{"url": e.get("url"), "canonical": e.get("canonical")} for e in drop]
    ev.sort(key=lambda e: -e["cosine"] * e["w_source"])
    return ev[: cfg["max_evidence_per_claim"]]


def _ms(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)


def is_english(text: str) -> bool:
    """langdetect on the first 500 chars (FR-13), backed by an English stop-word ratio because langdetect is unreliable
    on short headlines ("Barack Obama was born in Kenya." is detected as non-English)."""
    from spacy.lang.en.stop_words import STOP_WORDS
    words = text[:500].lower().split()
    if not words or sum(w.strip(".,!?;:\"'") in STOP_WORDS for w in words) / len(words) >= 0.2:
        return True
    # ponytail: langdetect misreads short ASCII headlines ("UK Aldi supermarkets don't sell halal meat." -> not en);
    # short ASCII-only input is accepted as English, so a short ASCII Spanish/French line slips through (then R1/R9).
    if len(words) < 10 and text[:500].isascii():
        return True
    try:
        from langdetect import DetectorFactory, detect
        DetectorFactory.seed = 0
        return detect(text[:500]) == "en"
    except Exception:  # noqa: BLE001 - undetectable (e.g. only digits) -> do not block the input
        return True


def analyze(text: str = "", title: str | None = None, url: str | None = None, published: str | None = None,
            input_type: str = "headline", mode: str | None = None, k: int | None = None,
            use_evidence: bool = True, use_temporal: bool = True, classifier_only: bool = False,
            exclude_urls: list[str] | None = None) -> dict:
    """Ablation flags (docs/results/ablation.md): use_evidence=False skips retrieval + stance (-> R1), use_temporal=False
    skips the temporal/context check, classifier_only=True reports the style classifier's label as the verdict
    (rule "CLF"; a style estimate, never fact-checked - evaluation only, not exposed by the API)."""
    t0 = time.time()
    timings: dict[str, int] = {}
    excl = {norm_url(u) for u in exclude_urls or []} - {""}
    excluded: list[dict] = []  # evidence removed by the leakage guard (evaluation only)
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
    t = time.perf_counter()
    english = is_english(full)
    timings["language"] = _ms(t)
    satire = is_satire(url)
    claims = []
    if english and not satire and not classifier_only:
        t = time.perf_counter()
        if input_type == "headline":
            # a whitespace-only title must not hide the text; very long "headlines" are capped (spaCy / NLI limits)
            h = ((title or "").strip() or text.strip())[:MAX_HEADLINE_CHARS]
            claims = [{"text": h, "score": 1.0, "start": None, "end": None, "is_title": True}] if h else []
        else:
            claims = identify_claims(text, title=title, k=k)
        timings["claims"] = _ms(t)
        for c in claims:
            c["evidence"] = []
            if use_evidence:
                c["evidence"] = gather_evidence(c["text"], mode, k, cfg, timings, excl, excluded)
                t = time.perf_counter()
                assign_stance(c["text"], c["evidence"], cfg)
                timings["stance"] = timings.get("stance", 0) + _ms(t)
    t = time.perf_counter()
    clf = classify(full) if english else {"p_fake": None, "label": None, "model": None}
    timings["classifier"] = _ms(t)
    main_ev = claims[0]["evidence"] if claims else []
    t = time.perf_counter()
    temporal = temporal_check(full, main_ev, main_claim=claims[0]["text"] if claims else None, t_post=published,
                              title=title if input_type != "headline" else None,
                              body=text if input_type != "headline" else None) if claims and use_temporal else {}
    timings["temporal"] = _ms(t)
    t = time.perf_counter()
    if classifier_only and english and clf["label"]:
        fused = {"verdict": clf["label"].upper(), "rule_fired": "CLF", "confidence": round(max(clf["p_fake"], 1 - clf["p_fake"]), 4),
                 "reason": "Style-based estimate from the content classifier only - not fact-checked (ablation).",
                 "signals": {"main_stance": "NEUTRAL", "S_1": 0.0, "E_1": 0, "per_claim": [], "temporal_flags": [],
                             "p_fake": clf["p_fake"]}}
    else:
        fused = fuse(claims, temporal, p_fake=clf["p_fake"], input_is_satire=satire, is_english=english, cfg=cfg)
    timings["fusion"] = _ms(t)
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
            "uncertainty": uncertainty, "evidence_count": n_ev, "latency_ms": int((time.time() - t0) * 1000),
            "timings_ms": timings, **({"excluded_evidence": excluded} if exclude_urls else {})}


def model_status() -> dict:
    name = "distilbert-welfake-v1" if (DISTILBERT_DIR / "config.json").exists() else \
        (TFIDF_LR.stem if TFIDF_LR.exists() else None)
    from src.evidence.offline_index import INDEX_PATH
    return {"clf": name, "nli": "nli-deberta-v3-small", "embedder": "all-MiniLM-L6-v2",
            "offline_index": Path(INDEX_PATH).exists()}
