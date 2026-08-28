"""MSE1 EDA checks (light): figures exist, the canonical names are present, and docs/eda_summary.md has exactly one
line per figure. Run `make eda` first; the tests skip when no EDA figure exists yet (fresh clone before `make eda`)."""
from __future__ import annotations

import re

import pytest

from src.config import DOCS_DIR, FIGURES_DIR

# names fixed by docs/milestones/MSE1_ProjectDetails.md §3.4 (artefact table)
CANONICAL = [
    "eda_class_balance", "eda_length_hist", "eda_top_ngrams_before", "eda_top_ngrams_after", "eda_wordclouds",
    "eda_ner_types", "eda_isot_dates", "eda_duplicates", "eda_liar_labels", "eda_fever_balance", "eda_style_features",
]
# extra figures produced by notebooks/01_eda.ipynb (recorded in agent-docs + docs/eda_summary.md)
EXTRA = ["eda_maxlen_coverage", "eda_title_length", "eda_leakage_tokens", "eda_artefact_classifier",
         "eda_liar_speakers", "eda_vocab_overlap", "eda_sources"]
SUMMARY = DOCS_DIR / "eda_summary.md"


def _figures():
    figs = sorted(FIGURES_DIR.glob("eda_*.png"))
    if not figs:
        pytest.skip(f"no EDA figures in {FIGURES_DIR} - run `make eda`")
    return figs


def test_at_least_ten_figures():
    assert len(_figures()) >= 10


@pytest.mark.parametrize("name", CANONICAL + EXTRA)
def test_named_figure_exists_and_is_not_empty(name):
    _figures()
    p = FIGURES_DIR / f"{name}.png"
    assert p.exists(), f"missing {p}"
    assert p.stat().st_size > 10_000, f"{p} is suspiciously small ({p.stat().st_size} bytes)"


def test_every_figure_is_a_png_with_pixels():
    for p in _figures():
        with p.open("rb") as fh:
            assert fh.read(8) == b"\x89PNG\r\n\x1a\n", f"{p} is not a PNG"


def test_summary_has_exactly_one_line_per_figure():
    figs = _figures()
    assert SUMMARY.exists(), f"{SUMMARY} missing"
    lines = SUMMARY.read_text(encoding="utf-8").splitlines()
    for p in figs:
        hits = [ln for ln in lines if p.name in ln]
        assert len(hits) == 1, f"{p.name} mentioned {len(hits)} times in {SUMMARY.name} (expected exactly 1)"
        # the line must be a table row with insight + design decision columns
        assert hits[0].startswith("|") and hits[0].count("|") >= 4, f"{p.name}: not a '| figure | insight | decision |' row"
    listed = re.findall(r"`(eda_[a-z0-9_]+\.png)`", SUMMARY.read_text(encoding="utf-8"))
    assert set(listed) == {p.name for p in figs}, "eda_summary.md lists figures that are not on disk (or vice versa)"


def test_summary_states_leakage_and_duplicate_numbers():
    _figures()
    text = SUMMARY.read_text(encoding="utf-8")
    assert re.search(r"[Aa]rtefact-only classifier accuracy on raw ISOT: 0\.9\d+", text)
    assert "Cross-split overlap: 0 exact, 0 near" in text
    assert "max_len = 256" in text
