"""Fast MSE2 checks for the DistilBERT models (`src/models/distilbert.py`, `src/models/liar_head.py`).

No training here: checkpoints / logs are produced by `make train-distilbert` / `make train-liar` and the tests skip when
they are absent (fresh clone) or the tokenizer is not in the local HF cache.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from src.models import distilbert as D
from src.models import liar_head as L


@pytest.fixture(scope="module")
def tok():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(D.CHECKPOINT, local_files_only=True)
    except OSError:
        pytest.skip("distilbert-base-uncased tokenizer not cached")


def test_collator_pairs_title_and_text_and_truncates(tok):
    ds = D.TextDataset(["word " * 600, "short body"], [1, 0], titles=["A title", ""])
    enc = D.Collator(tok, 256)([ds[0], ds[1]])
    assert enc["input_ids"].shape == (2, 256) and enc["labels"].tolist() == [1, 0]
    first = enc["input_ids"][0].tolist()
    assert first.count(tok.sep_token_id) == 2                     # [CLS] title [SEP] text [SEP]
    single = D.Collator(tok, 64)([D.TextDataset(["a claim"], [2])[0]])
    assert single["input_ids"][0].tolist().count(tok.sep_token_id) == 1


def test_batch_plan_matches_spec():
    assert D.batch_plan(256) == (16, 1) and D.batch_plan(512) == (8, 2)


def test_stratified_subset_keeps_balance():
    df = pd.DataFrame({"id": range(1000), "label": ["real"] * 600 + ["fake"] * 400})
    sub = D.stratified_subset(df, 100)
    assert len(sub) == 100 and (sub["label"] == "fake").sum() == 40


@pytest.mark.parametrize("model_dir,labels", [(D.FINAL_DIR, ["real", "fake"]), (L.FINAL_DIR, L.LABELS3)])
def test_checkpoint_loads_and_predicts(model_dir, labels):
    if not (model_dir / "config.json").exists():
        pytest.skip(f"{model_dir.name} missing - run make train-distilbert / train-liar")
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    assert [model.config.id2label[i] for i in range(len(labels))] == labels
    for f in ("trainer_state.json", "training_args.json"):
        assert (model_dir / f).exists()
    if labels == ["real", "fake"]:
        tok, m = D.load_model(model_dir, device="cpu")
        p = D.predict_proba(pd.DataFrame({"title": ["Senate passes budget bill"], "text": ["The Senate voted 52-48."]}),
                            tok, m, max_len=64)
        assert p.shape == (1,) and 0.0 <= p[0] <= 1.0


def test_logs_have_required_fields():
    paths = sorted(D.LOG_DIR.glob("distilbert_*.json"))
    paths = [p for p in paths if not p.name.endswith("_test.json")]
    if not paths:
        pytest.skip("no DistilBERT train logs yet")
    for p in paths:
        rec = json.loads(p.read_text())
        assert rec["seed"] == 42 and rec["hardware"] and rec["wall_clock_s"] > 0, p.name
        assert rec["per_epoch"] and all("train_loss" in e and "val_macro_f1" in e for e in rec["per_epoch"]), p.name
    if D.FINAL_LOG.exists():
        full = json.loads(D.FINAL_LOG.read_text())
        assert full["config"]["lr"] == D.winner()["config"]["lr"]       # final config = sweep winner (lr)
