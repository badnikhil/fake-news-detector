"""``make data``: raw files in data/raw/  ->  data/processed/*.parquet + data/splits/*.csv  (master doc §11.1).

Per dataset the pipeline is: load -> (artefact-only leakage classifier on raw ISOT) -> strip artefacts ->
normalise + drop empty/short -> exact dedup -> near dedup (MinHash) -> stratified 80/10/10 split (seed 42)
-> parquet + split CSVs.  LIAR keeps its official splits; FEVER is sampled to a balanced 20k/3k subset;
FakeNewsNet-PolitiFact titles pass through.  Everything is logged (``docs/mse1_make_data.log`` via make).

Run:  python -m src.preprocess.run_all [--datasets welfake isot ...] [--skip-leakage-check]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime

import pandas as pd

from src.common.seed import set_seed
from src.config import (
    DATASETS,
    FEVER_DEV_N,
    FEVER_TRAIN_N,
    MIN_TOKENS,
    NEAR_DUP_THRESHOLD,
    PROCESSED_DIR,
    SEED,
    SPLITS_DIR,
)
from src.preprocess import load as L
from src.preprocess.artefacts import artefact_only_accuracy, strip_frame
from src.preprocess.clean import clean_frame
from src.preprocess.dedupe import cross_split_overlap, drop_exact_duplicates, drop_near_duplicates
from src.preprocess.split import split_report, stratified_split, write_split_csvs

log = logging.getLogger("make_data")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


def _dist(df: pd.DataFrame, col: str = "label") -> str:
    return ", ".join(f"{k}={v:,}" for k, v in df[col].value_counts().sort_index().items())


def _write_parquet(df: pd.DataFrame, name: str) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False, engine="pyarrow", compression="snappy")
    log.info("wrote %s (%s rows, %.1f MB)", path, f"{len(df):,}", path.stat().st_size / 1e6)


# ------------------------------------------------------------------------------------------------
def process_articles(name: str, df: pd.DataFrame, leakage_check: bool) -> dict:
    """WELFake / ISOT: artefact strip -> clean -> dedup -> split."""
    t0 = time.time()
    stats: dict = {"dataset": name, "rows_in": len(df)}
    log.info("[%s] rows in: %s | labels: %s", name, f"{len(df):,}", _dist(df))

    if leakage_check and name == "isot":
        t = time.time()
        raw_text = df["text"].fillna("").astype(str)
        res = artefact_only_accuracy(raw_text, df["label"], seed=SEED)
        stats["artefact_only_classifier"] = res
        log.info(
            "[%s] ARTEFACT-ONLY classifier (LR on pattern counts, 5-fold CV) accuracy = %.4f ± %.4f  (%.1fs)",
            name, res["cv_accuracy"], res["cv_std"], time.time() - t,
        )
        log.info("[%s]   fraction of docs containing each pattern: %s", name,
                 {k: round(v, 4) for k, v in res["doc_frac_with_pattern"].items()})

    df["text"] = df["text"].fillna("").astype(str)
    df, pat_counts = strip_frame(df)
    stats["artefact_docs_matched"] = pat_counts
    log.info("[%s] artefact patterns stripped (docs matched): %s", name, pat_counts)

    df, cstats = clean_frame(df, MIN_TOKENS)
    stats.update({k: cstats[k] for k in ("dropped_empty", "dropped_short")})
    log.info("[%s] dropped empty=%s short(<%d tok)=%s -> %s", name, f"{cstats['dropped_empty']:,}", MIN_TOKENS,
             f"{cstats['dropped_short']:,}", f"{cstats['rows_out']:,}")

    df, n_exact = drop_exact_duplicates(df)
    stats["exact_dups_removed"] = n_exact
    log.info("[%s] exact duplicates removed: %s -> %s", name, f"{n_exact:,}", f"{len(df):,}")

    t = time.time()
    df, n_near, hashes = drop_near_duplicates(df)
    stats["near_dups_removed"] = n_near
    log.info("[%s] near duplicates removed (MinHash Jaccard ≥ %.2f): %s -> %s  (%.1fs)", name,
             NEAR_DUP_THRESHOLD, f"{n_near:,}", f"{len(df):,}", time.time() - t)

    df["split"] = stratified_split(df, "label", SEED)
    rep = split_report(df)
    log.info("[%s] split report:\n%s", name, rep.to_string(index=False))
    stats["splits"] = rep.to_dict(orient="records")

    ov = cross_split_overlap(df, hashes)
    stats["cross_split_overlap"] = ov
    log.info("[%s] cross-split overlap (must be 0): %s", name, ov)
    if any(ov.values()):
        raise RuntimeError(f"{name}: duplicates span splits: {ov}")

    df = df.sort_values("id", kind="mergesort").reset_index(drop=True)
    _write_parquet(df, name)
    sizes = write_split_csvs(df, name)
    log.info("[%s] split CSVs written: %s", name, sizes)
    stats["rows_out"] = len(df)
    stats["label_dist_out"] = df["label"].value_counts().to_dict()
    stats["seconds"] = round(time.time() - t0, 1)
    log.info("[%s] rows out: %s | labels: %s | %.1fs", name, f"{len(df):,}", _dist(df), stats["seconds"])
    return stats


def process_liar(df: pd.DataFrame) -> dict:
    """LIAR: official splits kept unchanged; only normalisation (no row is dropped)."""
    t0 = time.time()
    stats = {"dataset": "liar", "rows_in": len(df)}
    log.info("[liar] rows in: %s | official splits: %s", f"{len(df):,}", df["split"].value_counts().to_dict())
    log.info("[liar] 6-way labels: %s", _dist(df))
    df, cstats = clean_frame(df, min_tokens=None)
    from src.preprocess.dedupe import exact_hash

    n_exact = int(df["text"].map(exact_hash).duplicated().sum())
    stats.update(dropped_empty=cstats["dropped_empty"], dropped_short=0, exact_dups_removed=0,
                 near_dups_removed=0, exact_dups_present_kept=n_exact)
    log.info("[liar] dropped empty=%d; exact duplicate statements present=%d (KEPT: official splits are frozen)",
             cstats["dropped_empty"], n_exact)
    log.info("[liar] label5: %s | label3: %s", _dist(df, "label5"), _dist(df, "label3"))
    df = df.sort_values("id", kind="mergesort").reset_index(drop=True)
    _write_parquet(df, "liar")
    sizes = write_split_csvs(df, "liar")
    log.info("[liar] split CSVs written: %s", sizes)
    stats.update(rows_out=len(df), splits=sizes, seconds=round(time.time() - t0, 1))
    return stats


def sample_fever(parts: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    """Balanced 20k train / 3k dev subset, fixed seed."""
    t0 = time.time()
    stats = {"dataset": "fever_subset"}
    out = []
    for split, n_total in (("train", FEVER_TRAIN_N), ("val", FEVER_DEV_N)):
        df = parts[split]
        stats[f"{split}_rows_in"] = len(df)
        log.info("[fever] %s full: %s | %s", split, f"{len(df):,}", _dist(df))
        df, cstats = clean_frame(df, min_tokens=None)
        df, n_exact = drop_exact_duplicates(df)
        stats[f"{split}_exact_dups_removed"] = n_exact
        labels = sorted(df["label"].unique())
        per = [n_total // len(labels) + (1 if i < n_total % len(labels) else 0) for i in range(len(labels))]
        picked = []
        for lab, k in zip(labels, per, strict=True):
            pool = df[df["label"] == lab].sort_values("id", kind="mergesort")
            picked.append(pool.sample(n=min(k, len(pool)), random_state=SEED))
        sub = pd.concat(picked).sort_values("id", kind="mergesort").reset_index(drop=True)
        sub["split"] = split
        log.info("[fever] %s subset: %s | %s (exact dups removed before sampling: %d)", split, f"{len(sub):,}",
                 _dist(sub), n_exact)
        stats[f"{split}_rows_out"] = len(sub)
        out.append(sub)
    df = pd.concat(out, ignore_index=True)
    stats.update(rows_in=stats["train_rows_in"] + stats["val_rows_in"], dropped_empty=0, dropped_short=0,
                 exact_dups_removed=stats["train_exact_dups_removed"] + stats["val_exact_dups_removed"],
                 near_dups_removed=0, rows_out=len(df), seconds=round(time.time() - t0, 1))
    return df, stats


def process_fnn(df: pd.DataFrame) -> dict:
    t0 = time.time()
    stats = {"dataset": "fnn_politifact", "rows_in": len(df)}
    log.info("[fnn_politifact] rows in: %s | %s", f"{len(df):,}", _dist(df))
    df, cstats = clean_frame(df, min_tokens=None)
    from src.preprocess.dedupe import exact_hash

    n_exact = int(df["text"].map(exact_hash).duplicated().sum())
    log.info("[fnn_politifact] dropped empty titles=%d; duplicate titles present=%d (kept: passthrough)",
             cstats["dropped_empty"], n_exact)
    df = df.sort_values("id", kind="mergesort").reset_index(drop=True)
    _write_parquet(df, "fnn_politifact")
    stats.update(dropped_empty=cstats["dropped_empty"], dropped_short=0, exact_dups_removed=0,
                 near_dups_removed=0, exact_dups_present_kept=n_exact, rows_out=len(df),
                 seconds=round(time.time() - t0, 1))
    return stats


# ------------------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="*", choices=DATASETS, default=DATASETS)
    ap.add_argument("--skip-leakage-check", action="store_true")
    args = ap.parse_args(argv)
    _setup_logging()
    set_seed(SEED)
    t_all = time.time()
    log.info("make data started %s | seed=%d | datasets=%s", datetime.now().isoformat(timespec="seconds"), SEED,
             args.datasets)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    summary: list[dict] = []

    if "welfake" in args.datasets:
        summary.append(process_articles("welfake", L.load_welfake(), leakage_check=not args.skip_leakage_check))
    if "isot" in args.datasets:
        df = L.load_isot()
        if df is None:
            log.warning("[isot] raw files missing (data/raw/isot/True.csv, Fake.csv) - SKIPPED; see data/README.md")
            summary.append({"dataset": "isot", "rows_in": 0, "rows_out": 0, "skipped": True})
        else:
            summary.append(process_articles("isot", df, leakage_check=not args.skip_leakage_check))
    if "liar" in args.datasets:
        summary.append(process_liar(L.load_liar()))
    if "fever_subset" in args.datasets:
        df, st = sample_fever(L.load_fever())
        df = df.sort_values(["split", "id"], kind="mergesort").reset_index(drop=True)
        _write_parquet(df, "fever_subset")
        summary.append(st)
    if "fnn_politifact" in args.datasets:
        summary.append(process_fnn(L.load_fnn_politifact()))

    log.info("=" * 100)
    log.info("SUMMARY (rows in -> dropped empty / short -> exact dups -> near dups -> rows out)")
    cols = ["dataset", "rows_in", "dropped_empty", "dropped_short", "exact_dups_removed", "near_dups_removed",
            "rows_out", "seconds"]
    tab = pd.DataFrame([{c: s.get(c, 0) for c in cols} for s in summary])
    log.info("\n%s", tab.to_string(index=False))
    for s in summary:
        if "artefact_only_classifier" in s:
            log.info("artefact-only classifier accuracy on raw %s: %.4f", s["dataset"],
                     s["artefact_only_classifier"]["cv_accuracy"])
        if "cross_split_overlap" in s:
            log.info("cross-split overlap %s: %s", s["dataset"], s["cross_split_overlap"])
    (PROCESSED_DIR / "make_data_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    log.info("make data finished in %.1fs (%.1f min)", time.time() - t_all, (time.time() - t_all) / 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
