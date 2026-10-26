"""Live Claims Set loader + schema check (master §10 D7, §18; `data/live_claims/README.md`).

`load_live_claims(split="dev")` returns the 30-item fusion-tuning slice, `split="eval"` the held-out rest.
CLI: `python -m src.eval.live_claims` validates the CSV and prints label / publisher / split counts.
"""
from __future__ import annotations

import pandas as pd

from src.config import LIVE_CLAIMS_DIR, VERDICTS

PATH = LIVE_CLAIMS_DIR / "live_claims.csv"
COLUMNS = ["id", "claim", "label", "original_rating", "publisher", "factcheck_url", "factcheck_date",
           "claim_date", "claimant", "context", "input_type", "modality", "split", "notes"]
REQUIRED = ["id", "claim", "label", "original_rating", "publisher", "factcheck_url", "factcheck_date", "input_type", "split"]


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Raise ValueError on any schema violation; return df unchanged."""
    errs = []
    if list(df.columns) != COLUMNS:
        errs.append(f"columns {list(df.columns)} != {COLUMNS}")
    else:
        empty = [c for c in REQUIRED if (df[c].fillna("").astype(str).str.strip() == "").any()]
        errs += [f"empty values in {empty}"] if empty else []
        errs += [f"bad labels {set(df.label) - set(VERDICTS)}"] if not set(df.label) <= set(VERDICTS) else []
        errs += [f"bad split {set(df.split) - {'dev', 'eval'}}"] if not set(df.split) <= {"dev", "eval"} else []
        errs += [f"bad input_type {set(df.input_type)}"] if not set(df.input_type) <= {"headline", "article", "url"} else []
        for c in ("id", "factcheck_url"):
            if df[c].duplicated().any():
                errs.append(f"duplicate {c}: {df.loc[df[c].duplicated(), c].tolist()}")
        if not df.factcheck_url.str.startswith("https://").all():
            errs.append("factcheck_url must be https")
        for c in ("factcheck_date", "claim_date"):
            bad = df[c].notna() & pd.to_datetime(df[c], format="%Y-%m-%d", errors="coerce").isna()
            errs += [f"bad {c}: {df.loc[bad, 'id'].tolist()}"] if bad.any() else []
    if errs:
        raise ValueError("live_claims.csv: " + "; ".join(errs))
    return df


def load_live_claims(split: str | None = None, path=PATH) -> pd.DataFrame:
    df = validate(pd.read_csv(path, dtype=str, keep_default_na=False).replace({"": None}))
    return df if split is None else df[df.split == split].reset_index(drop=True)


if __name__ == "__main__":
    d = load_live_claims()
    print(f"{len(d)} items OK\n", pd.crosstab(d.label, d.split, margins=True), "\n", d.publisher.value_counts())
