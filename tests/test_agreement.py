"""src/eval/agreement.py on a tiny fixture."""
import pandas as pd
import pytest

from src.eval.agreement import agreement


def test_agreement_small_fixture():
    gold = pd.DataFrame({"id": list("abcdef"),
                         "label": ["REAL", "FAKE", "FAKE", "MISLEADING", "PARTIALLY TRUE", "UNVERIFIABLE"]})
    second = pd.DataFrame({"id": list("abcdef"),
                           "label2": ["real", "FAKE", "MISLEADING", "MISLEADING", "", "UNREADABLE"]})
    r = agreement(gold, second)
    assert r["n"] == 4 and r["excluded"] == ["e", "f"]
    assert r["raw"] == 0.75 and r["disagree"].id.tolist() == ["c"]
    assert r["confusion"].values.sum() == 4 and r["confusion"].loc["FAKE", "MISLEADING"] == 1
    assert r["per_class"]["REAL"] == 1.0 and r["per_class"]["MISLEADING"] == pytest.approx(2 / 3)
    assert 0 < r["kappa"] < 1
    assert agreement(gold, gold.rename(columns={"label": "label2"}))["kappa"] == 1.0


def test_agreement_rejects_unknown_label():
    with pytest.raises(ValueError):
        agreement(pd.DataFrame({"id": ["a"], "label": ["REAL"]}), pd.DataFrame({"id": ["a"], "label2": ["TRUE"]}))
