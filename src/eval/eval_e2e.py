"""End-to-end evaluation on the Live Claims Set (ESE §3; master §21.3) — every number comes from real pipeline runs.

    python -m src.eval.eval_e2e run --split eval --mode online [--variant full|no_temporal|clf_only]
    python -m src.eval.eval_e2e tune         # fusion grid on the 30 dev items (re-fuses saved dev/online runs)
    python -m src.eval.eval_e2e temporal     # O5 old-news flag on data/temporal_set/temporal_set.csv (online)
    python -m src.eval.eval_e2e latency      # 50 cached-online + 50 offline replays -> train_logs/latency_*.jsonl
    python -m src.eval.eval_e2e cases        # ESE §3.4 example inputs (online) + edge cases through the FastAPI app
    python -m src.eval.eval_e2e report       # e2e.md, ablation.md, latency.md, case_studies.md, cm_e2e.png
    python -m src.eval.eval_e2e all          # = `make eval-e2e` (eval split only; `tune` is separate: `make tune-fusion`)

Runs append to docs/results/train_logs/e2e_{split}_{mode}_{variant}.jsonl and skip ids already there (resumable; delete
the file to re-run). Input = the claim as a headline, published = claim_date, else factcheck_date. Online evidence is
cached in data/cache/evidence.sqlite, so a re-run is repeatable. Dev items are only used by `tune`, never in the reports.
"""
from __future__ import annotations

import argparse
import itertools
import json
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_DIR, FIGURES_DIR, RESULTS_DIR, VERDICTS
from src.eval.live_claims import load_live_claims

LOGS = RESULTS_DIR / "train_logs"
UNCACHED_DIR = RESULTS_DIR / "train_logs" / "e2e_tfidf_superseded"  # first (uncached) online pass, TF-IDF era
TEMPORAL_CSV = DATA_DIR / "temporal_set" / "temporal_set.csv"
VARIANTS = {"full": {}, "no_temporal": {"use_temporal": False}, "clf_only": {"classifier_only": True}}
GRID = {"tau": [0.3, 0.45, 0.6, 0.8, 1.0], "min_evidence": [1, 2, 3], "N_months": [3, 6, 12]}


TAG = ""  # "" = v1 logs (no leakage guard); "v2" = 2026-11-12 improvement round (guard + ClaimReview + dated claims)
V1_MD = LOGS / "v1_md"  # frozen v1 result pages; v2 reports are written above them, never over them


def log_path(split, mode, variant="full", tag=None) -> Path:
    tag = TAG if tag is None else tag
    return LOGS / f"e2e_{split}_{mode}_{variant}{'_' + tag if tag else ''}.jsonl"


def tagged(name: str) -> Path:  # train_logs/<name> with the run tag before the extension
    stem, ext = name.rsplit(".", 1)
    return LOGS / (f"{stem}_{TAG}.{ext}" if TAG else name)


def out_md(name: str, lines: list[str]) -> None:
    if TAG:
        write_versioned(name, lines)
    else:
        (RESULTS_DIR / name).write_text("\n".join(lines) + "\n")


def write_versioned(name: str, v2_lines: list[str]) -> None:
    """docs/results/<name> = v2 section on top + the frozen v1 page (headings demoted) underneath."""
    v1 = (V1_MD / name).read_text() if (V1_MD / name).exists() else ""
    v1 = "\n".join("#" + ln if ln.startswith("#") else ln for ln in v1.splitlines())
    (RESULTS_DIR / name).write_text("\n".join(v2_lines) + "\n\n---\n\n# v1 results (frozen, 2026-11-12 first pass — kept for "
                                    "comparison, see the leakage note above)\n\n" + v1 + "\n")


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def md_table(df: pd.DataFrame) -> str:
    """DataFrame -> GitHub markdown table (pandas' own needs `tabulate`, which is not a project dependency)."""
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)  # noqa: E731
    rows = [" | ".join(map(str, df.columns)), " | ".join("---" for _ in df.columns)]
    rows += [" | ".join(fmt(v) for v in r) for r in df.itertuples(index=False)]
    return "\n".join(f"| {r} |" for r in rows)


def peak_rss_mb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


# ---------------------------------------------------------------- running the pipeline
def run(split: str, mode: str, variant: str = "full") -> Path:
    from src.api import pipeline
    from src.evidence import online
    df = load_live_claims(split)
    out = log_path(split, mode, variant)
    done = {r["id"] for r in read_jsonl(out)}
    with out.open("a") as f:
        for i, r in enumerate(df.itertuples(), 1):
            if r.id in done:
                continue
            miss0 = online.STATS["miss"]
            t = time.time()
            guard = {"exclude_urls": [r.factcheck_url]} if TAG else {}  # v2: never the item's own fact-check
            res = pipeline.analyze(text=r.claim, published=r.claim_date or r.factcheck_date, input_type="headline",
                                   mode=mode, **VARIANTS[variant], **guard)
            rec = {"id": r.id, "split": split, "label": r.label, "modality": r.modality, "publisher": r.publisher,
                   "verdict": res["verdict"], "rule_fired": res["rule_fired"], "confidence": res["confidence"],
                   "clf_model": res["classifier"]["model"], "wall_ms": int((time.time() - t) * 1000),
                   "cache_misses": online.STATS["miss"] - miss0, "result": res}
            f.write(json.dumps(rec, default=str) + "\n")
            f.flush()
            print(f"[{i}/{len(df)}] {r.id} gold={r.label:<14} pred={res['verdict']:<14} {res['rule_fired']:<4} "
                  f"ev={res['evidence_count']:<2} {rec['wall_ms']} ms misses={rec['cache_misses']}", flush=True)
    print(f"{out}: peak RSS {peak_rss_mb()} MB")
    return out


# ---------------------------------------------------------------- metrics
def metrics(gold: list[str], pred: list[str]) -> dict:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
    p, r, f, s = precision_recall_fscore_support(gold, pred, labels=VERDICTS, zero_division=0)
    return {"n": len(gold), "accuracy": round(accuracy_score(gold, pred), 4),
            "macro_f1": round(f1_score(gold, pred, labels=VERDICTS, average="macro", zero_division=0), 4),
            "per_class": {c: {"precision": round(p[i], 4), "recall": round(r[i], 4), "f1": round(f[i], 4),
                              "support": int(s[i]), "predicted": int(sum(x == c for x in pred))}
                          for i, c in enumerate(VERDICTS)},
            "cm": confusion_matrix(gold, pred, labels=VERDICTS).tolist()}


def refuse(rec: dict, cfg: dict) -> str:
    """Re-run temporal check + fusion on a saved result with other thresholds (retrieval/stance unchanged)."""
    from src.fusion.rules import fuse
    from src.temporal.checks import temporal_check
    res = rec["result"]
    if res["rule_fired"] in ("LANG", "R0"):
        return res["verdict"]
    claims = res["claims"]
    main_ev = claims[0]["evidence"] if claims else []
    temporal = temporal_check(claims[0]["text"], main_ev, main_claim=claims[0]["text"],
                              t_post=res["temporal"].get("input_date"), n_months=cfg["N_months"], cfg=cfg) if claims else {}
    return fuse(claims, temporal, p_fake=res["classifier"]["p_fake"], cfg=cfg)["verdict"]


def tune() -> dict:
    from src.fusion.rules import config
    recs = read_jsonl(log_path("dev", "online"))
    assert len(recs) == 30 and all(r["split"] == "dev" for r in recs), "run `run --split dev --mode online` first"
    base = dict(config())
    assert [refuse(r, base) for r in recs] == [r["verdict"] for r in recs], "re-fusion must reproduce the pipeline"
    gold = [r["label"] for r in recs]
    rows = []
    for tau, me, nm in itertools.product(*GRID.values()):
        cfg = {**base, "tau": tau, "min_evidence": me, "N_months": nm}
        m = metrics(gold, [refuse(r, cfg) for r in recs])
        dist = abs(tau - base["tau"]) + abs(me - base["min_evidence"]) + abs(nm - base["N_months"]) / 6
        rows.append({"tau": tau, "min_evidence": me, "N_months": nm, "macro_f1": m["macro_f1"],
                     "accuracy": m["accuracy"], "unv_precision": m["per_class"]["UNVERIFIABLE"]["precision"],
                     "dist_from_default": round(dist, 3)})
    grid = pd.DataFrame(rows).sort_values(["macro_f1", "accuracy", "dist_from_default"],
                                          ascending=[False, False, True]).reset_index(drop=True)
    best = grid.iloc[0].to_dict()
    default = grid[(grid.tau == base["tau"]) & (grid.min_evidence == base["min_evidence"])
                   & (grid.N_months == base["N_months"])].iloc[0].to_dict()
    LOGS.joinpath("fusion_grid.json").write_text(json.dumps({"rows": rows, "best": best, "default": default}, indent=1))
    n_best = int((grid.macro_f1 == best["macro_f1"]).sum())
    rules = pd.Series([r["rule_fired"] for r in recs]).value_counts().to_dict()
    at_me = grid[grid.min_evidence == base["min_evidence"]].macro_f1
    tau_rng = (at_me.min(), at_me.max())
    old_p = UNCACHED_DIR / "fusion_grid.json"
    old = json.loads(old_p.read_text()) if old_p.exists() else None
    md = [
        "# Fusion-threshold tuning (dev slice of the Live Claims Set)", "",
        "Generated by `python -m src.eval.eval_e2e tune` (`make tune-fusion`). Data: the **30 `dev` items** only "
        "(`split == dev`; the 65 `eval` items are never used here). Each dev claim was run once through `/analyze` "
        "logic in **online** mode (offline index + ddgs + Wikipedia; no Google Fact Check key; evidence cached in "
        "`data/cache/evidence.sqlite`), log `train_logs/e2e_dev_online_full.jsonl`. Retrieval and NLI stance are "
        "threshold-independent, so every grid point re-runs only the temporal check (`N_months`) and fusion R0–R9 "
        "(`tau`, `min_evidence`) on the saved evidence; at the defaults this reproduces the pipeline's verdicts "
        "exactly (asserted). Objective: 5-class macro-F1; ties broken by accuracy, then by closeness to the MSE2 "
        "defaults.", "",
        f"Grid: tau {GRID['tau']} × min_evidence {GRID['min_evidence']} × N_months {GRID['N_months']} = {len(grid)} "
        f"configs. Rules fired on dev at the defaults: {rules}. Content classifier (R8 tie-break, ±0.10 confidence): "
        f"{', '.join(sorted({str(r['clf_model']) for r in recs}))}.", "",
        f"**Default (MSE2)** tau {base['tau']}, min_evidence {base['min_evidence']}, N_months {base['N_months']}: "
        f"macro-F1 **{default['macro_f1']:.4f}**, accuracy {default['accuracy']:.4f}.  ",
        f"**Best** tau {best['tau']}, min_evidence {int(best['min_evidence'])}, N_months {int(best['N_months'])}: "
        f"macro-F1 **{best['macro_f1']:.4f}**, accuracy {best['accuracy']:.4f} ({n_best} configs share this macro-F1).",
        "", "Caveat: 30 items (3 UNVERIFIABLE, 6 MISLEADING); one item = 0.033 accuracy. Differences of one or two "
        "items are noise — read the table as a sanity check, not as a precise optimum.", "",
        f"**Decision (2026-11-12): `fusion.yaml` keeps the defaults (tau {base['tau']}, min_evidence "
        f"{base['min_evidence']}, N_months {base['N_months']}).** The best grid point "
        f"(tau {best['tau']}, min_evidence {int(best['min_evidence'])}) gets {round(best['accuracy'] * 30)} vs "
        f"{round(default['accuracy'] * 30)} of 30 dev items right (macro-F1 +{best['macro_f1'] - default['macro_f1']:.4f}). "
        "min_evidence = 1 would also make R8 unreachable (R3 / R7 would fire on a single source before the "
        "single-source rule), i.e. it silently removes the documented single-source safeguard (master §14.2); a gain "
        "of at most one item on 30 does not justify that design change. N_months has no effect on dev (no dev claim "
        "carries a recency cue, so `old_news` never fires). At the default min_evidence the five tau values span "
        f"macro-F1 {tau_rng[0]:.4f}–{tau_rng[1]:.4f}.", "",
        *([f"Earlier run of the same grid with the TF-IDF + LR classifier (before the DistilBERT checkpoint existed; "
           f"`train_logs/e2e_tfidf_superseded/fusion_grid.json`): default macro-F1 {old['default']['macro_f1']:.4f}, best "
           f"{old['best']['macro_f1']:.4f} at tau {old['best']['tau']}, min_evidence {int(old['best']['min_evidence'])} "
           "— same decision."] if old else []), "",
        md_table(grid), ""]
    (RESULTS_DIR / "fusion_thresholds.md").write_text("\n".join(md))
    print("\n".join(md[:12]))
    return {"best": best, "default": default}


GRID_V2 = {**GRID, "explicit_date_check": [False, True], "evidence_text_dates": [False, True]}


def control_old_news(rec: dict, cfg: dict) -> bool:
    """Temporal flag of a saved temporal-set run recomputed with other thresholds."""
    from src.temporal.checks import temporal_check
    res = rec["result"]
    if not res["claims"]:
        return False
    c = res["claims"][0]
    return bool(temporal_check(c["text"], c["evidence"], main_claim=c["text"], t_post=res["temporal"].get("input_date"),
                               n_months=cfg["N_months"], cfg=cfg)["old_news"])


def tune_v2() -> dict:
    """v2 grid on the 30 dev items (5-class macro-F1) + the 15 temporal-set CONTROLS (false old-news flags); the 15
    temporal positives and the 65 eval items are never read here. Selection (fixed before looking): highest dev
    macro-F1, then fewest control false positives, then accuracy, then closeness to the current values;
    min_evidence = 1 is shown but not selectable (it makes R8 unreachable, see v1 decision)."""
    from src.fusion.rules import FUSION_YAML, config
    recs = read_jsonl(log_path("dev", "online"))
    ctrl = [r for r in read_jsonl(tagged("temporal_set_online.jsonl")) if r["old_as_new"] == 0]
    assert len(recs) == 30 and len(ctrl) == 15 and all("result" in r for r in ctrl), "run dev + temporal (v2) first"
    base = dict(config())
    assert [refuse(r, base) for r in recs] == [r["verdict"] for r in recs], "re-fusion must reproduce the pipeline"
    gold = [r["label"] for r in recs]
    rows = []
    for vals in itertools.product(*GRID_V2.values()):
        cfg = {**base, **dict(zip(GRID_V2, vals, strict=True))}
        m = metrics(gold, [refuse(r, cfg) for r in recs])
        dist = sum(abs(float(cfg[k]) - float(base[k])) / (6 if k == "N_months" else 1) for k in GRID_V2)
        rows.append({**dict(zip(GRID_V2, vals, strict=True)), "macro_f1": m["macro_f1"], "accuracy": m["accuracy"],
                     "control_fp": sum(control_old_news(r, cfg) for r in ctrl), "dist": round(dist, 3)})
    grid = pd.DataFrame(rows).sort_values(["macro_f1", "control_fp", "accuracy", "dist"],
                                          ascending=[False, True, False, True]).reset_index(drop=True)
    best = grid[grid.min_evidence != 1].iloc[0].to_dict()
    cur = grid[np.logical_and.reduce([grid[k] == base[k] for k in GRID_V2])].iloc[0].to_dict()
    tagged("fusion_grid.json").write_text(json.dumps({"rows": rows, "best": best, "current": cur}, indent=1, default=str))
    changed = {k: best[k] for k in GRID_V2 if best[k] != base[k]}
    if changed:  # minimal in-place edit of fusion.yaml (comments kept)
        import re as _re
        y = FUSION_YAML.read_text()
        for k, v in changed.items():
            v = (str(bool(v)).lower() if isinstance(base[k], bool) else
                 int(v) if k in ("min_evidence", "N_months") else float(v))
            y = _re.sub(rf"^{k}:\s*\S+", f"{k}: {v}", y, count=1, flags=_re.M)
        FUSION_YAML.write_text(y)
        config.cache_clear()
    md = ["# Fusion-threshold tuning — v2 (2026-11-12 improvement round)", "",
          "Generated by `python -m src.eval.eval_e2e tune --tag v2`. Data: the **30 `dev` Live Claims** (v2 pipeline: "
          "leakage guard, ClaimReview ratings; log `train_logs/e2e_dev_online_full_v2.jsonl`) and the **15 controls** "
          "of the temporal set (log `train_logs/temporal_set_online_v2.jsonl`; the 15 old-as-new positives and the 65 "
          "eval items are never read). Saved evidence is re-fused per grid point (asserted identical to the pipeline at "
          "the current values).", "",
          f"Grid: {' × '.join(f'{k} {v}' for k, v in GRID_V2.items())} = {len(grid)} configs. Selection rule (fixed "
          "in advance): dev macro-F1 ↓, control false positives ↑, accuracy ↓, closeness to current values; "
          "min_evidence 1 not selectable (makes R8 unreachable).", "",
          f"**Before tuning (v2 code, v1 values):** {', '.join(f'{k} {base[k]}' for k in GRID_V2)}: dev macro-F1 "
          f"**{cur['macro_f1']:.4f}**, accuracy {cur['accuracy']:.4f}, control false positives {cur['control_fp']}/15.  ",
          f"**Selected:** {', '.join(f'{k} {best[k]}' for k in GRID_V2)}: dev macro-F1 **{best['macro_f1']:.4f}**, "
          f"accuracy {best['accuracy']:.4f}, control false positives {best['control_fp']}/15.  ",
          f"`fusion.yaml` changed: {changed or 'nothing'}.", "",
          "Caveat: 30 + 15 items; one item = 0.033 dev accuracy. Read as a sanity check, not a precise optimum.", "",
          "Top 40 configs:", "", md_table(grid.head(40)), ""]
    write_versioned("fusion_thresholds.md", md)
    print("\n".join(md[:12]))
    return {"best": best, "current": cur, "changed": changed}


# ---------------------------------------------------------------- temporal set (O5)
def temporal() -> None:
    from src.api import pipeline
    df = pd.read_csv(TEMPORAL_CSV, dtype=str, keep_default_na=False)
    out = tagged("temporal_set_online.jsonl")
    done = {r["id"] for r in read_jsonl(out)}
    with out.open("a") as f:
        for r in df.itertuples():
            if r.id in done:
                continue
            guard = {"exclude_urls": [r.factcheck_url]} if TAG else {}  # v2 leakage guard: not its own source page
            res = pipeline.analyze(text=r.claim, published=r.t_post or r.factcheck_date, input_type="headline",
                                   mode="online", **guard)
            t = res["temporal"]
            rec = {"id": r.id, "old_as_new": int(r.old_as_new), "old_news": bool(t.get("old_news")),
                   "recency_cue": t.get("recency_cue"), "gap_months": t.get("gap_months"),
                   "event_date": t.get("event_date"), "n_dated_supporting": len(t.get("evidence_dates") or []),
                   "verdict": res["verdict"], "rule_fired": res["rule_fired"], "evidence_count": res["evidence_count"],
                   "clf_model": res["classifier"]["model"], "date_conflict": t.get("date_conflict"),
                   "claim_date": t.get("claim_date"), **({"result": res} if TAG else {})}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(rec, flush=True)
    recs = pd.DataFrame(read_jsonl(out)).merge(df[["id", "claim", "t_post", "t_event", "publisher"]], on="id")
    y, p = recs.old_as_new.astype(int), recs.old_news.astype(int)
    tp, fp, fn = int(((y == 1) & (p == 1)).sum()), int(((y == 0) & (p == 1)).sum()), int(((y == 1) & (p == 0)).sum())
    acc = float((y == p).mean())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec_ = tp / (tp + fn) if tp + fn else float("nan")
    pos = recs[y == 1]
    miss = pos[~pos.old_news.astype(bool)]
    cue = miss.recency_cue.astype(bool)
    dated = miss.get("claim_date", pd.Series(None, index=miss.index)).notna() if TAG else pd.Series(False, index=miss.index)
    why = {"no dated SUPPORTS evidence": int((miss.n_dated_supporting == 0).sum()),
           "dated evidence, but neither a recency cue nor a claim date": int(((miss.n_dated_supporting > 0) & ~cue & ~dated).sum()),
           "dated evidence + cue/claim date, but gap < N": int(((miss.n_dated_supporting > 0) & (cue | dated)).sum())}
    md = ["# Temporal \"old news presented as new\" test set (O5)", "",
          f"Generated by `python -m src.eval.eval_e2e temporal`. Items: `data/temporal_set/temporal_set.csv` "
          f"({len(recs)} verified fact-checks: {int(y.sum())} old-as-new, {int((y == 0).sum())} controls; provenance in "
          "`data/temporal_set/README.md`). Each claim was run through the pipeline **online** as a headline with "
          "`published = t_post` (the date the claim was shared; for the rows whose page gives no share date, "
          "`factcheck_date`, an upper bound); the prediction is `temporal.old_news`. "
          "Log: `train_logs/temporal_set_online.jsonl`. Content classifier (affects only the verdict column via R8): "
          f"{', '.join(sorted(set(map(str, recs.clf_model))))}.", "",
          "| metric | value |", "|---|---|",
          f"| items | {len(recs)} |", f"| TP / FP / FN / TN | {tp} / {fp} / {fn} / {int(((y == 0) & (p == 0)).sum())} |",
          f"| old-news flag precision | {prec:.4f} |" if tp + fp else "| old-news flag precision | undefined (flag never fired) |",
          f"| old-news flag recall | {rec_:.4f} |",
          f"| % correct (flag == gold) | {acc:.1%} |",
          f"| **O5 (≥ 80 % correct)** | **{'met' if acc >= 0.8 else 'NOT met'}** |", "",
          f"Why the {len(miss)} missed positives were missed: " + ", ".join(f"{k}: {v}" for k, v in why.items()) + ".", "",
          "Verdicts on the old-as-new items: " + str(pos.verdict.value_counts().to_dict()) + ".", "",
          "## Per item", "",
          recs[["id", "old_as_new", "old_news", "recency_cue", "gap_months", "event_date", "t_post", "t_event",
                "verdict", "rule_fired", "claim"]].assign(claim=lambda d: d.claim.str.slice(0, 90)).pipe(md_table), ""]
    if TAG:
        from src.fusion.rules import config
        cfg_now = config()
        flip = {**cfg_now, "evidence_text_dates": not cfg_now["evidence_text_dates"]}
        full = read_jsonl(out)
        v1 = pd.DataFrame(read_jsonl(LOGS / "temporal_set_online.jsonl"))
        y1, p1 = v1.old_as_new.astype(int), v1.old_news.astype(int)
        md[0] += " — v2 (2026-11-12 improvement round)"
        md[2:2] = ["v2 = explicit claim dates vs supporting-evidence dates (`explicit_date_check`; `evidence_text_dates` "
                   "as tuned on dev + the 15 controls only — the 15 positives were not used for tuning) + ClaimReview "
                   "ratings + leakage guard (the item's own fact-check page is never evidence). "
                   f"**v1 → v2:** % correct {float((y1 == p1).mean()):.1%} → {acc:.1%}; TP {int(((y1 == 1) & (p1 == 1)).sum())}"
                   f" → {tp}, FP {int(((y1 == 0) & (p1 == 1)).sum())} → {fp}. Date-conflict path fired on "
                   f"{int(recs.date_conflict.fillna(False).astype(bool).sum())} items.", "",
                   "Post-hoc diagnostic (NOT used for any selection; it reads the positives): with "
                   f"`evidence_text_dates` flipped to {not cfg_now['evidence_text_dates']} the flag would fire on "
                   f"{sum(control_old_news(r, flip) for r in full if r['old_as_new'] == 1)}/15 positives and "
                   f"{sum(control_old_news(r, flip) for r in full if r['old_as_new'] == 0)}/15 controls.", ""]
    out_md("temporal.md", md)
    print("\n".join(md[:14]))


# ---------------------------------------------------------------- latency replays (cached online + offline)
def latency(n: int = 50) -> None:
    from src.api import pipeline
    from src.evidence import online
    ids = load_live_claims("eval").head(n)
    for mode in ("offline", "online"):
        out = tagged(f"latency_{mode}.jsonl")
        out.unlink(missing_ok=True)
        pipeline.analyze(text="Warm-up claim about the unemployment rate.", mode="offline")  # model load excluded
        with out.open("w") as f:
            for r in ids.itertuples():
                miss0 = online.STATS["miss"]
                t = time.perf_counter()
                res = pipeline.analyze(text=r.claim, published=r.claim_date or r.factcheck_date, mode=mode)
                f.write(json.dumps({"id": r.id, "mode": mode, "wall_ms": int((time.perf_counter() - t) * 1000),
                                    "cache_misses": online.STATS["miss"] - miss0, "timings_ms": res["timings_ms"]}) + "\n")
    tagged("latency_peak_rss.json").write_text(json.dumps({"peak_rss_mb": peak_rss_mb()}))
    print("latency replays done, peak RSS", peak_rss_mb(), "MB")


# ---------------------------------------------------------------- case studies + edge cases
EXAMPLES = [  # ESE doc §3.4 inputs, verbatim
    ("REAL", {"input_type": "headline", "text": "ISRO's Chandrayaan-3 landed near the lunar south pole on 23 Aug 2023."}),
    ("FAKE", {"input_type": "headline", "text": "WHO declared that 5G towers spread COVID-19."}),
    ("PARTIALLY TRUE", {"input_type": "article",
                        "text": "India's population overtook China's in 2023, reaching 2 billion."}),
    ("MISLEADING", {"input_type": "headline", "text": "BREAKING: Kerala under water today, thousands stranded",
                    "published": "2026-09-10"}),
    ("UNVERIFIABLE", {"input_type": "headline", "text": "A local shop in Ghaziabad sold 500 phones in one hour."}),
]
EDGE = [
    ("empty text", {"text": ""}),
    ("whitespace only", {"text": "   \n\t "}),
    ("whitespace title + real text", {"text": "The Eiffel Tower is in Paris.", "title": "   "}),
    ("emoji only", {"text": "😀🔥🔥😂"}),
    ("digits / punctuation only", {"text": "12345 !!! ???"}),
    ("non-English (Hindi)", {"text": "भारत ने 2023 में चंद्रयान-3 को चंद्रमा पर उतारा।"}),
    ("non-English (Spanish)", {"text": "El gobierno anunció hoy que los impuestos bajarán el próximo año."}),
    ("very long headline (60k chars)", {"text": "The government announced a new tax policy today. " * 1200}),
    ("very long article (60k chars)", {"input_type": "article",
                                       "text": "The government announced a new tax policy today in Delhi. " * 1000}),
    ("article with no check-worthy sentence", {"input_type": "article", "text": "Wow. Really? Go!"}),
    ("URL input in offline mode", {"input_type": "url", "url": "https://example.com/x", "mode": "offline"}),
    ("URL input without url", {"input_type": "url"}),
    ("invalid published date", {"text": "BREAKING: earthquake hits Delhi today", "published": "not-a-date"}),
    ("k out of range (k=50)", {"text": "Some claim about taxes.", "k": 50}),
    ("unknown input_type", {"input_type": "video", "text": "x"}),
    ("HTML / script injection", {"text": "<script>alert(1)</script> The moon is made of cheese."}),
]


def cases() -> None:
    from fastapi.testclient import TestClient

    from src.api import main
    c = TestClient(main.app, raise_server_exceptions=False)
    ex = []
    for cls, body in EXAMPLES:
        r = c.post("/analyze", json={**body, "mode": "online"})
        ex.append({"class": cls, "input": body, "status": r.status_code, "result": r.json()})
    edge = []
    for name, body in EDGE:
        body = {"mode": "offline", **body}
        r = c.post("/analyze", json=body)
        j = r.json()
        edge.append({"case": name, "status": r.status_code,
                     "verdict": j.get("verdict") if isinstance(j, dict) else None,
                     "rule_fired": j.get("rule_fired") if isinstance(j, dict) else None,
                     "detail": (str(j.get("detail"))[:120] if isinstance(j, dict) and "detail" in j else None)})
        print(edge[-1], flush=True)
    tagged("case_studies.json").write_text(json.dumps({"examples": ex, "edge": edge}, indent=1, default=str))


# ---------------------------------------------------------------- reports
def _metric_rows(name: str, m: dict) -> str:
    return (f"| {name} | {m['n']} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | "
            f"{m['per_class']['UNVERIFIABLE']['precision']:.4f} |")


def _per_class_table(m: dict) -> list[str]:
    rows = ["| class | precision | recall | F1 | support | predicted |", "|---|---|---|---|---|---|"]
    rows += [f"| {c} | {v['precision']:.4f} | {v['recall']:.4f} | {v['f1']:.4f} | {v['support']} | {v['predicted']} |"
             for c, v in m["per_class"].items()]
    return rows


def _cm_table(m: dict) -> list[str]:
    rows = ["| gold \\ pred | " + " | ".join(VERDICTS) + " |", "|---|" + "---|" * len(VERDICTS)]
    rows += [f"| {g} | " + " | ".join(map(str, m["cm"][i])) + " |" for i, g in enumerate(VERDICTS)]
    return rows


def _cm_figure(m: dict, title: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.array(m["cm"])
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.imshow(cm, cmap="Blues")
    short = ["REAL", "FAKE", "PART. TRUE", "MISLEAD.", "UNVERIF."]
    ax.set_xticks(range(5), short, rotation=30, ha="right")
    ax.set_yticks(range(5), short)
    for i, j in itertools.product(range(5), range(5)):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("gold (fact-checker rating)")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _trim(res: dict) -> dict:
    keep = ("verdict", "confidence", "rule_fired", "reason", "evidence_count", "uncertainty", "mode", "latency_ms")
    out = {k: res.get(k) for k in keep}
    if res.get("claims"):
        c = res["claims"][0]
        out["main_claim"] = {"text": c["text"], "stance_summary": c.get("stance_summary"), "evidence_top3": [
            {k: e.get(k) for k in ("source", "date", "stance", "p", "cosine", "rating", "snippet")}
            | {"snippet": (e.get("snippet") or "")[:160]} for e in c["evidence"][:3]]}
        out["n_claims"] = len(res["claims"])
    out["temporal"] = {k: res.get("temporal", {}).get(k) for k in ("old_news", "gap_months", "recency_cue",
                                                                   "out_of_context", "event_date", "input_date")}
    out["classifier"] = res.get("classifier")
    return out


def _own_url_hits(rec: dict, live: dict) -> list[dict]:
    from src.api.pipeline import norm_url
    own = norm_url(live[rec["id"]])
    return [e for c in rec["result"]["claims"] for e in c["evidence"]
            if own in (norm_url(e.get("url")), norm_url(e.get("canonical")))]


def _other_factchecks(rec: dict, live: dict, cos: float) -> list[dict]:
    """Retained Tier-1 / rated items that match the claim (cosine >= factcheck_match_cosine) and are not its own page."""
    from src.api.pipeline import norm_url
    own = norm_url(live[rec["id"]])
    ev = rec["result"]["claims"][0]["evidence"] if rec["result"]["claims"] else []
    return [e for e in ev if (e.get("tier") == 1 or e.get("is_factcheck")) and e.get("cosine", 0) >= cos
            and own not in (norm_url(e.get("url")), norm_url(e.get("canonical")))]


def v2_header(m: dict, onl: list[dict]) -> list[str]:
    """v1 vs v2 side by side + the leakage-guard disclosure (only for the v2 report)."""
    from src.fusion.rules import config
    cfg = config()
    live = dict(zip(*load_live_claims()[["id", "factcheck_url"]].T.values, strict=True))
    v1_on, v1_off = read_jsonl(log_path("eval", "online", tag="")), read_jsonl(log_path("eval", "offline", tag=""))
    m1 = {"online": metrics([r["label"] for r in v1_on], [r["verdict"] for r in v1_on]),
          "offline": metrics([r["label"] for r in v1_off], [r["verdict"] for r in v1_off])}
    leaked = [r for r in v1_on if _own_url_hits(r, live)]
    v1cfg = {**cfg, "explicit_date_check": False, "evidence_text_dates": False}
    stripped = []
    for r in v1_on:  # v1 evidence with the item's own fact-check page removed, re-fused with the v1 rules
        r2 = json.loads(json.dumps(r))
        hits = {id(e) for e in _own_url_hits(r2, live)}
        for c in r2["result"]["claims"]:
            c["evidence"] = [e for e in c["evidence"] if id(e) not in hits]
        stripped.append(refuse(r2, v1cfg))
    m1s = metrics([r["label"] for r in v1_on], stripped)
    removed = [r for r in onl if r["result"].get("excluded_evidence")]
    other = [r for r in onl if _other_factchecks(r, live, cfg["factcheck_match_cosine"])]
    only_own = [r for r in removed if r not in other]
    row = lambda name, x: (f"| {name} | {x['accuracy']:.4f} | {x['macro_f1']:.4f} | "  # noqa: E731
                           f"{x['per_class']['UNVERIFIABLE']['precision']:.4f} |")
    return [
        "## v2 (2026-11-12 improvement round) vs v1", "",
        "v2 = v1 + (1) **leakage guard**: evidence whose normalised URL or canonical URL equals the item's own "
        "fact-check URL is removed before ranking (`analyze(exclude_urls=...)`, evaluation only); (2) **ClaimReview**: "
        "Tier-1 result pages are parsed for schema.org ClaimReview and, when the claim restates `claimReviewed` "
        "(the existing NLI / cosine ≥ 0.75 gate), the rating becomes the item's stance (p 0.95) with the LIAR / Live "
        "Claims mapping; (3) **dated claims** in the temporal check (`explicit_date_check`, `evidence_text_dates` "
        "in `fusion.yaml`). Thresholds tuned on the 30 dev items + temporal-set controls only "
        "(`fusion_thresholds.md`). The eval items were run once per mode.", "",
        "| run | accuracy | macro-F1 | UNVERIFIABLE precision |", "|---|---|---|---|",
        row("v1 online (no leakage guard)", m1["online"]),
        row("v1 online, own fact-check evidence removed post hoc (re-fused, v1 rules)", m1s),
        row("**v2 online**", m["online"]), row("v1 offline", m1["offline"]), row("**v2 offline**", m["offline"]), "",
        "### Leakage disclosure", "",
        f"- **v1 was leaky:** in {len(leaked)} of 65 v1 online runs the kept evidence contained the item's *own* "
        "fact-check page (the page the gold label comes from; matched on normalised URL — v1 stored no canonical "
        "URLs, so this is a lower bound). Removing those items post hoc and re-fusing gives the second row above.",
        f"- **v2 guard:** the item's own page was retrieved and removed for {len(removed)} of 65 items "
        f"({sum(len(r['result']['excluded_evidence']) for r in removed)} evidence items).",
        f"- {len(other)} of 65 items kept ≥ 1 *other* Tier-1 / rated fact-check matching the claim (cosine ≥ "
        f"{cfg['factcheck_match_cosine']}) — legitimate evidence (another fact-checker, or another article of the same "
        f"publisher); {len(only_own)} items had their own page as the only matching fact-check (removed).",
        "- The guard is evaluation-only: in the deployed app, retrieving an existing fact-check is the intended "
        "behaviour.", ""]


def report() -> None:
    off, onl = read_jsonl(log_path("eval", "offline")), read_jsonl(log_path("eval", "online"))
    sfx = f"_{TAG}" if TAG else ""
    for name, recs in (("offline", off), ("online", onl)):
        assert len(recs) == 65 and all(r["split"] == "eval" for r in recs), f"eval/{name} run incomplete"
    clf_models = {r["clf_model"] for r in off + onl}
    m = {}
    for name, recs in (("offline", off), ("online", onl)):
        m[name] = metrics([r["label"] for r in recs], [r["verdict"] for r in recs])
        text = [r for r in recs if r["modality"] == "text"]
        m[name + "_text"] = metrics([r["label"] for r in text], [r["verdict"] for r in text])
    sfx = f"_{TAG}" if TAG else ""
    _cm_figure(m["online"], f"End-to-end, Live Claims eval (n=65), online {TAG}", FIGURES_DIR / f"cm_e2e{sfx}.png")
    _cm_figure(m["offline"], f"End-to-end, Live Claims eval (n=65), offline {TAG}", FIGURES_DIR / f"cm_e2e_offline{sfx}.png")

    def coverage(recs, origin=None):
        return np.mean([any(origin is None or e["origin"] != "offline_index" for c in r["result"]["claims"]
                            for e in c["evidence"]) for r in recs])
    cov_any, cov_web = coverage(onl), coverage(onl, origin="web")
    cov_off = coverage(off)
    rules = {n: pd.Series([r["rule_fired"] for r in recs]).value_counts().to_dict() for n, recs in
             (("offline", off), ("online", onl))}
    o6f, o6u = m["online"]["macro_f1"], m["online"]["per_class"]["UNVERIFIABLE"]["precision"]
    from src.fusion.rules import config
    cfg = config()
    md = ["# End-to-end evaluation — Live Claims Set (eval split)", "",
          "Generated by `make eval-e2e` (`python -m src.eval.eval_e2e`). Items: the **65 `eval` items** of "
          "`data/live_claims/live_claims.csv` (the 30 `dev` items were used only for fusion tuning and are excluded). "
          "Each claim goes through `src.api.pipeline.analyze` (the code behind `POST /analyze`) as a headline with "
          "`published = claim_date` (else `factcheck_date`). Gold = the fact-checker's rating mapped to 5 classes "
          "(`data/live_claims/README.md`). Fusion thresholds: "
          f"tau {cfg['tau']}, min_evidence {cfg['min_evidence']}, N_months {cfg['N_months']} (`fusion.yaml`). "
          f"Content classifier used: {', '.join(sorted(map(str, clf_models)))}. Online = offline FAISS index + ddgs "
          "(DuckDuckGo) + Wikipedia; Google Fact Check was **not** used (no API key). Predictions: "
          f"`train_logs/e2e_eval_{{offline,online}}_full{sfx}.jsonl`; confusion matrices `docs/figures/cm_e2e{sfx}.png`, "
          f"`cm_e2e_offline{sfx}.png`.", "",
          *(v2_header(m, onl) if TAG else []),
          "## Headline numbers", "",
          "| run | n | accuracy | 5-class macro-F1 | UNVERIFIABLE precision |", "|---|---|---|---|---|",
          _metric_rows("online, all items", m["online"]), _metric_rows("online, text-only (no media items)", m["online_text"]),
          _metric_rows("offline, all items", m["offline"]), _metric_rows("offline, text-only", m["offline_text"]), "",
          f"Media items = the 12 `modality == media` claims (about an image/video), {65 - m['online_text']['n']} of them in eval; a text-only "
          "pipeline can at best match the fact-check itself.", "",
          "## Objectives", "", "| objective | target | measured (online, eval) | status |", "|---|---|---|---|",
          f"| O6 5-class macro-F1 | ≥ 0.55 | {o6f:.4f} | {'met' if o6f >= 0.55 else 'NOT met'} |",
          f"| O6 UNVERIFIABLE precision | ≥ 0.70 | {o6u:.4f} | {'met' if o6u >= 0.70 else 'NOT met'} |",
          f"| O3 online evidence coverage (≥ 1 passage) | ≥ 85 % of items | {cov_any:.1%} any evidence; {cov_web:.1%} with "
          f"≥ 1 web/Wikipedia passage among the kept top-{cfg['max_evidence_per_claim']} | "
          f"{'met' if cov_any >= 0.85 else 'NOT met'} (any evidence) |", "",
          f"Offline mode: {cov_off:.1%} of items get ≥ 1 offline-index hit (cosine ≥ {cfg['offline_min_cosine']}).", "",
          "## Per-class (online, all 65)", "", *_per_class_table(m["online"]), "",
          "## Confusion matrix (online, all 65) — `docs/figures/cm_e2e.png`", "", *_cm_table(m["online"]), "",
          "## Per-class (offline, all 65)", "", *_per_class_table(m["offline"]), "",
          "## Confusion matrix (offline) — `docs/figures/cm_e2e_offline.png`", "", *_cm_table(m["offline"]), "",
          f"## Rules fired\n\nonline: {rules['online']}  \noffline: {rules['offline']}", ""]
    if TAG:
        md[0] += " — v2 (2026-11-12 improvement round)"
    out_md("e2e.md", md)

    # ---- ablation
    abl = [("(a) classifier only (style label, no evidence)", "online", "clf_only"),
           ("(b) + evidence & stance (no temporal check)", "online", "no_temporal"),
           ("(c) + temporal / context = full system (online)", "online", "full"),
           ("(d) full system, offline only", "offline", "full")]
    rows, cmp_rows = [], []
    for name, mode, variant in abl:
        recs = read_jsonl(log_path("eval", mode, variant))
        assert len(recs) == 65, (mode, variant)
        mm = metrics([r["label"] for r in recs], [r["verdict"] for r in recs])
        pc = mm["per_class"]
        rows.append(f"| {name} | {mm['accuracy']:.4f} | {mm['macro_f1']:.4f} | " + " | ".join(
            f"{pc[c]['f1']:.3f}" for c in VERDICTS) + f" | {pc['UNVERIFIABLE']['precision']:.3f} |")
        if TAG:
            v1 = read_jsonl(log_path("eval", mode, variant, tag=""))
            m1 = metrics([r["label"] for r in v1], [r["verdict"] for r in v1])
            cmp_rows.append(f"| {name} | {m1['accuracy']:.4f} | {m1['macro_f1']:.4f} | {mm['accuracy']:.4f} | "
                            f"{mm['macro_f1']:.4f} |")
    md = ["# Ablation — Live Claims Set (65 eval items)", "",
          "Generated by `make eval-e2e`. Same pipeline code with flags (`src.api.pipeline.analyze`): (a) "
          "`classifier_only=True` — the TF-IDF/DistilBERT style label is reported as REAL/FAKE (it can never say "
          "PARTIALLY TRUE / MISLEADING / UNVERIFIABLE); (b) `use_temporal=False`; (c) the full system; (d) "
          "`mode=offline`. (a)–(c) online, run back-to-back on the evidence cache (a source that failed is not cached "
          "and is retried, so a few items can differ in evidence between arms). Logs `train_logs/e2e_eval_*.jsonl`. "
          f"Content classifier in every arm: {', '.join(sorted(map(str, clf_models)))}.", "",
          "| variant | accuracy | macro-F1 | F1 REAL | F1 FAKE | F1 PART. TRUE | F1 MISLEADING | F1 UNVERIF. | UNV precision |",
          "|---|---|---|---|---|---|---|---|---|", *rows, ""]
    if TAG:
        md[0] = "# Ablation — Live Claims Set (65 eval items) — v2 (2026-11-12 improvement round)"
        md += ["### v1 vs v2 (v1 had no leakage guard — see `e2e.md`)", "",
               "| variant | v1 accuracy | v1 macro-F1 | v2 accuracy | v2 macro-F1 |", "|---|---|---|---|---|", *cmp_rows, ""]
    out_md("ablation.md", md)

    # ---- latency
    def stat(vals):
        v = np.array(vals, dtype=float)
        return f"{np.median(v):.0f} | {np.percentile(v, 90):.0f}" if len(v) else "– | –"
    stages = ["language", "claims", "evidence_offline", "evidence_online", "stance", "classifier", "temporal", "fusion"]
    groups = {
        "offline (warm, latency replay)": read_jsonl(tagged("latency_offline.jsonl")),
        "online, cached (latency replay, 0 cache misses)": [r for r in read_jsonl(tagged("latency_online.jsonl"))
                                                            if r["cache_misses"] == 0],
        "online, uncached (first e2e pass, ≥ 1 cache miss)": [
            {"wall_ms": r["wall_ms"], "timings_ms": r["result"]["timings_ms"]}
            for r in read_jsonl(UNCACHED_DIR / "e2e_dev_online_full.jsonl") + read_jsonl(UNCACHED_DIR / "e2e_eval_online_full.jsonl")
            if r["cache_misses"] > 0],
    }
    rss = json.loads(tagged("latency_peak_rss.json").read_text())["peak_rss_mb"] if tagged("latency_peak_rss.json").exists() else None
    md = ["# Latency (laptop CPU, ms)", "",
          "Generated by `make eval-e2e`. Per-stage timings come from the `timings_ms` field that `/analyze` now returns. "
          "Offline and cached-online rows: 50 eval claims replayed after a warm-up request (model loading excluded). "
          "Uncached-online rows: the first e2e pass over dev + eval items (2026-11-12, `train_logs/e2e_tfidf_superseded/`; "
          "that pass used the TF-IDF + LR classifier, ≈ 1 ms, before the DistilBERT checkpoint existed) that needed "
          "≥ 1 network fetch (search, "
          "Wikipedia or page fetch, 6 s timeout per source). Machine: 8 CPU threads, CPU-only inference, `nice -n 19`, "
          "other jobs (GPU training, Docker build) running concurrently — absolute numbers are pessimistic. "
          f"Classifier in the offline / cached rows: {', '.join(sorted(map(str, clf_models)))}.", "",
          "| group | n | end-to-end median | p90 | " + " | ".join(f"{s} median | p90" for s in stages) + " |",
          "|---|---|---|---|" + "---|---|" * len(stages)]
    for g, recs in groups.items():
        md.append(f"| {g} | {len(recs)} | {stat([r['wall_ms'] for r in recs])} | " + " | ".join(
            stat([r["timings_ms"].get(s, 0) for r in recs]) for s in stages) + " |")
    md += ["", f"Peak RSS of the latency-replay process (all models loaded: MiniLM, nli-deberta-v3-small, spaCy, "
           f"classifier): **{rss} MB**." if rss else ""]
    if TAG:
        md[0] += " — v2 (2026-11-12 improvement round; v2 also fetches ClaimReview from Tier-1 result pages)"
    out_md("latency.md", md)

    # ---- case studies
    cs = json.loads(tagged("case_studies.json").read_text())
    md = ["# Case studies and edge cases", "",
          "Generated by `make eval-e2e` (`python -m src.eval.eval_e2e cases` + `report`). Real outputs only, JSON trimmed "
          "(full JSON: `train_logs/case_studies.json`, `train_logs/e2e_eval_online_full.jsonl`).", "",
          "## One example per class (ESE doc §3.4 inputs, online)", ""]
    for e in cs["examples"]:
        md += [f"### Intended {e['class']}", "", f"Input: `{json.dumps(e['input'], ensure_ascii=False)}`  ",
               f"HTTP {e['status']} → **{e['result'].get('verdict')}** via `{e['result'].get('rule_fired')}`"
               + (" (matches the intended class)" if e["result"].get("verdict") == e["class"] else
                  " (does **not** match the intended class)"), "",
               "```json", json.dumps(_trim(e["result"]), indent=1, ensure_ascii=False), "```", ""]
    correct = {r["label"]: r for r in reversed(onl) if r["label"] == r["verdict"]}
    md += ["## Correct Live Claims eval items per class (online)", ""]
    for c in VERDICTS:
        r = correct.get(c)
        md += [f"- **{c}**: " + (f"{r['id']} \"{r['result']['claims'][0]['text'] if r['result']['claims'] else ''}\" → "
                                 f"{r['verdict']} via {r['rule_fired']}, confidence {r['confidence']}"
                                 if r else "no eval item of this class was predicted correctly.")]
    fails = [r for r in onl if r["label"] != r["verdict"] and r["label"] in ("FAKE", "REAL")][:2]
    md += ["", "## Failure cases (online, eval)", ""]
    for r in fails:
        md += [f"### {r['id']} — gold {r['label']}, predicted {r['verdict']} ({r['rule_fired']})", "",
               "```json", json.dumps(_trim(r["result"]), indent=1, ensure_ascii=False), "```", ""]
    md += ["## Edge cases (FastAPI TestClient, offline mode unless the case says otherwise)", "",
           "| case | HTTP | verdict | rule | detail |", "|---|---|---|---|---|"]
    md += [f"| {e['case']} | {e['status']} | {e['verdict'] or ''} | {e['rule_fired'] or ''} | {e['detail'] or ''} |"
           for e in cs["edge"]]
    n500 = sum(e["status"] >= 500 for e in cs["edge"])
    md += ["", f"HTTP 5xx responses: **{n500}** of {len(cs['edge'])}.", ""]
    if TAG:
        md[0] += " — v2 (2026-11-12 improvement round, leakage guard on for Live Claims items)"
    out_md("case_studies.md", md)
    print(open(RESULTS_DIR / "e2e.md").read()[:3000])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "tune", "temporal", "latency", "cases", "report", "all"])
    ap.add_argument("--split", default="eval", choices=["dev", "eval"])
    ap.add_argument("--mode", default="online", choices=["online", "offline"])
    ap.add_argument("--variant", default="full", choices=list(VARIANTS))
    ap.add_argument("--tag", default="v2", help='"" = v1 logs (no leakage guard), "v2" = current pipeline')
    a = ap.parse_args()
    global TAG
    TAG = a.tag
    if a.cmd == "run":
        run(a.split, a.mode, a.variant)
    elif a.cmd == "all":
        for mode, variant in (("offline", "full"), ("online", "full"), ("online", "no_temporal"),
                              ("online", "clf_only")):
            run("eval", mode, variant)
        latency()
        cases()
        report()
    else:
        {"tune": tune_v2 if TAG else tune, "temporal": temporal, "latency": latency, "cases": cases, "report": report}[a.cmd]()


if __name__ == "__main__":
    main()
