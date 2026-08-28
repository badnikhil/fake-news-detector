#!/usr/bin/env python
"""Download every raw dataset into ``data/raw/`` WITHOUT Kaggle credentials (``make download``).

Sources (all free, all scriptable; see ``data/README.md`` for licences):

* WELFake        - Zenodo record 4561253 (``WELFake_Dataset.csv``; file URL resolved via the Zenodo API)
* ISOT           - (a) University of Victoria ISOT lab direct zip
                   (b) a Hugging Face Hub mirror verified by row counts (21,417 true / 23,481 fake)
                   (c) otherwise: printed instructions for the manual Kaggle download
* LIAR           - UCSB ``liar_dataset.zip`` (train/valid/test TSV)
* FEVER          - fever.ai ``train.jsonl`` + ``shared_task_dev.jsonl``
* FakeNewsNet    - ``politifact_fake.csv`` + ``politifact_real.csv`` from the KaiDMML/FakeNewsNet GitHub repo

Idempotent: a file is skipped when it already exists with the expected size *and* (if known) the
SHA-256 recorded in ``data/raw/manifest.json``.  Every download records url, size, sha256 and
download date in that manifest, which ``data/README.md`` mirrors.

Usage::

    python scripts/download_data.py            # everything
    python scripts/download_data.py --only liar fever
    python scripts/download_data.py --force    # re-download even if present
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from datetime import date
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import RAW_DIR  # noqa: E402

MANIFEST = RAW_DIR / "manifest.json"
CHUNK = 1 << 20
TIMEOUT = 60
UA = {"User-Agent": "fakenews-detector-college-project/0.1 (+https://github.com; requests)"}

# Archives/files that make up each dataset.  ``size`` is the Content-Length observed on the
# first successful download (2026-08-28); used only for the cheap "already present" check.
ZENODO_RECORD = "4561253"
SOURCES: dict[str, list[dict]] = {
    "welfake": [
        {
            "name": "WELFake_Dataset.csv",
            "url": f"https://zenodo.org/api/records/{ZENODO_RECORD}",  # resolved through the API
            "resolver": "zenodo",
            "size": 245_086_152,
        }
    ],
    "isot": [
        {
            "name": "News-_dataset.zip",
            "url": "https://onlineacademiccommunity.uvic.ca/isot/wp-content/uploads/sites/7295/2023/03/News-_dataset.zip",
            "size": 43_106_824,
            "unzip": True,
        }
    ],
    "liar": [
        {
            "name": "liar_dataset.zip",
            "url": "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip",
            "size": 1_013_571,
            "unzip": True,
        }
    ],
    "fever": [
        {"name": "train.jsonl", "url": "https://fever.ai/download/fever/train.jsonl", "size": 33_024_303},
        {
            "name": "shared_task_dev.jsonl",
            "url": "https://fever.ai/download/fever/shared_task_dev.jsonl",
            "size": 4_349_935,
        },
    ],
    "fnn_politifact": [
        {
            "name": "politifact_fake.csv",
            "url": "https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/dataset/politifact_fake.csv",
            "size": 3_286_418,
        },
        {
            "name": "politifact_real.csv",
            "url": "https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/dataset/politifact_real.csv",
            "size": 8_278_658,
        },
    ],
}

# Hugging Face Hub mirrors of ISOT to try if the UVic link dies.  Each must yield exactly
# 21,417 true + 23,481 fake rows to be accepted (see ``_isot_from_hf``).
ISOT_HF_CANDIDATES: list[dict] = [
    # {"repo": "<user>/<dataset>", "text_col": "text", "title_col": "title", "label_col": "label"},
]
ISOT_EXPECTED = {"True.csv": 21_417, "Fake.csv": 23_481}


# ----------------------------------------------------------------------------------------------
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return {}


def save_manifest(m: dict) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(m, indent=2, sort_keys=True) + "\n")


def resolve_zenodo(record: str, filename: str) -> tuple[str, str, int]:
    """Return (download_url, licence_id, size) for ``filename`` in a Zenodo record."""
    r = requests.get(f"https://zenodo.org/api/records/{record}", timeout=TIMEOUT, headers=UA)
    r.raise_for_status()
    rec = r.json()
    lic = (rec.get("metadata", {}).get("license") or {}).get("id", "unknown")
    for f in rec.get("files", []):
        if f.get("key") == filename:
            return f["links"]["self"], lic, int(f.get("size", 0))
    raise RuntimeError(f"{filename} not found in Zenodo record {record}")


def stream_download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=TIMEOUT, headers=UA, allow_redirects=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0) or 0)
        done = 0
        with tmp.open("wb") as fh:
            for block in r.iter_content(CHUNK):
                fh.write(block)
                done += len(block)
                if total:
                    print(f"\r    {dest.name}: {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)
        print()
    tmp.replace(dest)


def present(dest: Path, entry: dict, manifest: dict) -> bool:
    if not dest.exists():
        return False
    size_ok = entry.get("size") is None or dest.stat().st_size == entry["size"]
    if not size_ok:
        return False
    rec = manifest.get(str(dest.relative_to(RAW_DIR)))
    if rec and rec.get("sha256"):
        return sha256(dest) == rec["sha256"]
    return True


def fetch(dataset: str, entry: dict, manifest: dict, force: bool) -> Path | None:
    ddir = RAW_DIR / dataset
    ddir.mkdir(parents=True, exist_ok=True)
    dest = ddir / entry["name"]
    key = str(dest.relative_to(RAW_DIR))
    url, licence = entry["url"], entry.get("licence")
    if entry.get("resolver") == "zenodo":
        url, licence, size = resolve_zenodo(ZENODO_RECORD, entry["name"])
        entry = {**entry, "size": size or entry.get("size")}
    if not force and present(dest, entry, manifest):
        print(f"  [skip] {key} already present ({dest.stat().st_size:,} bytes)")
    else:
        print(f"  [get ] {key} <- {url}")
        stream_download(url, dest)
        manifest[key] = {
            "dataset": dataset,
            "url": url,
            "size": dest.stat().st_size,
            "sha256": sha256(dest),
            "downloaded": date.today().isoformat(),
            **({"licence": licence} if licence else {}),
        }
        save_manifest(manifest)
    if entry.get("unzip"):
        with zipfile.ZipFile(dest) as zf:
            members = [m for m in zf.namelist() if not m.startswith("__MACOSX") and not m.endswith("/")]
            missing = [m for m in members if not (ddir / Path(m).name).exists()]
            if missing or force:
                for m in members:
                    target = ddir / Path(m).name  # flatten nested folders
                    with zf.open(m) as src, target.open("wb") as out:
                        shutil.copyfileobj(src, out)
                print(f"  [unz ] {len(members)} file(s) -> {ddir.relative_to(ROOT)}/")
    return dest


# ----------------------------------------------------------------------------------------------
def _count_csv_rows(path: Path) -> int:
    import pandas as pd

    return len(pd.read_csv(path))


def _isot_ok(ddir: Path) -> bool:
    try:
        return all((ddir / f).exists() and _count_csv_rows(ddir / f) == n for f, n in ISOT_EXPECTED.items())
    except Exception:  # noqa: BLE001
        return False


def _isot_from_hf(ddir: Path, manifest: dict) -> bool:
    """Fallback (b): rebuild True.csv/Fake.csv from a Hugging Face Hub mirror, verified by row counts."""
    if not ISOT_HF_CANDIDATES:
        print("  [isot] no Hugging Face mirror candidates configured")
        return False
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [isot] `datasets` not installed; cannot try Hugging Face mirrors")
        return False
    import pandas as pd

    for cand in ISOT_HF_CANDIDATES:
        try:
            print(f"  [isot] trying Hugging Face mirror {cand['repo']} ...")
            ds = load_dataset(cand["repo"])
            df = pd.concat([ds[s].to_pandas() for s in ds.keys()], ignore_index=True)
            lab = df[cand["label_col"]].astype(str).str.lower()
            true = df[lab.isin({"1", "true", "real"})]
            fake = df[lab.isin({"0", "false", "fake"})]
            if len(true) == ISOT_EXPECTED["True.csv"] and len(fake) == ISOT_EXPECTED["Fake.csv"]:
                for name, part in (("True.csv", true), ("Fake.csv", fake)):
                    out = part.rename(columns={cand["title_col"]: "title", cand["text_col"]: "text"})
                    out = out.reindex(columns=["title", "text", "subject", "date"])
                    out.to_csv(ddir / name, index=False)
                    manifest[f"isot/{name}"] = {
                        "dataset": "isot",
                        "url": f"hf://datasets/{cand['repo']}",
                        "size": (ddir / name).stat().st_size,
                        "sha256": sha256(ddir / name),
                        "downloaded": date.today().isoformat(),
                    }
                save_manifest(manifest)
                return True
            print(f"  [isot] row counts {len(true)}/{len(fake)} do not match 21,417/23,481 - rejected")
        except Exception as exc:  # noqa: BLE001
            print(f"  [isot] mirror failed: {exc}")
    return False


ISOT_MANUAL = """
  ISOT could not be downloaded automatically.  Manual step (needs a Kaggle login):
    1. open https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
    2. download and unzip; copy True.csv (21,417 rows) and Fake.csv (23,481 rows) into
         data/raw/isot/
    3. re-run `make download` (it will record the checksums) and then `make data`.
  The pipeline continues WITHOUT ISOT until then (isot.parquet / isot_* splits are skipped).
"""


def download_isot(manifest: dict, force: bool) -> bool:
    ddir = RAW_DIR / "isot"
    ddir.mkdir(parents=True, exist_ok=True)
    if not force and _isot_ok(ddir):
        print("  [skip] isot/True.csv + Fake.csv present with expected row counts")
        return True
    try:  # (a) UVic direct link
        fetch("isot", SOURCES["isot"][0], manifest, force)
        if _isot_ok(ddir):
            return True
        print("  [isot] UVic archive did not contain the expected True.csv/Fake.csv row counts")
    except Exception as exc:  # noqa: BLE001
        print(f"  [isot] UVic download failed: {exc}")
    if _isot_from_hf(ddir, manifest):  # (b)
        return True
    print(ISOT_MANUAL)  # (c)
    return False


# ----------------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", choices=sorted(SOURCES), help="subset of datasets")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args(argv)

    wanted = args.only or list(SOURCES)
    manifest = load_manifest()
    status: dict[str, bool] = {}
    for ds in wanted:
        print(f"\n== {ds} ==")
        if ds == "isot":
            status[ds] = download_isot(manifest, args.force)
            continue
        try:
            for entry in SOURCES[ds]:
                fetch(ds, entry, manifest, args.force)
            status[ds] = True
        except Exception as exc:  # noqa: BLE001
            print(f"  [FAIL] {ds}: {exc}")
            status[ds] = False

    print("\nSummary:")
    for ds, ok in status.items():
        print(f"  {ds:15s} {'ok' if ok else 'MISSING'}")
    print(f"Manifest: {MANIFEST.relative_to(ROOT)}")
    required = {"welfake", "liar", "fever", "fnn_politifact"} & set(wanted)
    return 0 if all(status[d] for d in required) else 1


if __name__ == "__main__":
    raise SystemExit(main())
