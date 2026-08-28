"""Project-wide constants: paths, seed, split ratios, canonical file names (master doc §11, §24).

Everything downstream imports from here so that a path or ratio is defined exactly once.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# --- data layout (master §24) ---
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
INDEX_DIR = DATA_DIR / "index"
CACHE_DIR = DATA_DIR / "cache"
MODELS_DIR = Path(os.environ.get("MODEL_DIR", DATA_DIR / "models"))
LIVE_CLAIMS_DIR = DATA_DIR / "live_claims"
DOCS_DIR = ROOT / "docs"
FIGURES_DIR = DOCS_DIR / "figures"
RESULTS_DIR = DOCS_DIR / "results"
MAKE_DATA_LOG = DOCS_DIR / "mse1_make_data.log"

# --- reproducibility ---
SEED = 42
SPLIT_RATIOS = {"train": 0.80, "val": 0.10, "test": 0.10}  # stratified, master §11.1 step 8

# --- preprocessing thresholds (master §11.1) ---
MIN_TOKENS = 20              # drop texts shorter than this (whitespace tokens)
NEAR_DUP_THRESHOLD = 0.90    # MinHash Jaccard estimate above which two texts are near-duplicates
MINHASH_PERMS = 128
SHINGLE_SIZE = 5             # word 5-grams for MinHash

# --- FEVER subset sizes (master §10 / §11.1) ---
FEVER_TRAIN_N = 20_000
FEVER_DEV_N = 3_000

# --- canonical artefact names (agreed across master + milestone docs) ---
DATASETS = ["welfake", "isot", "liar", "fever_subset", "fnn_politifact"]
SPLIT_DATASETS = ["welfake", "isot", "liar"]
SPLIT_NAMES = ["train", "val", "test"]

# Unified parquet schema (master §11.1 step 1 + MSE1 §3.2 DoD: label5/label3 for LIAR)
SCHEMA = ["id", "title", "text", "label", "label5", "label3", "source_dataset", "date", "split", "meta"]

# --- label mappings (master §10.1 / §10.2) ---
LIAR_LABELS = ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]
LIAR_TO_5 = {
    "true": "REAL",
    "mostly-true": "REAL",
    "half-true": "PARTIALLY TRUE",
    "barely-true": "MISLEADING",
    "false": "FAKE",
    "pants-fire": "FAKE",
}
LIAR_TO_3 = {
    "true": "TRUE",
    "mostly-true": "TRUE",
    "half-true": "MIXED",
    "barely-true": "MIXED",
    "false": "FALSE",
    "pants-fire": "FALSE",
}
BINARY_TO_5 = {"real": "REAL", "fake": "FAKE"}
BINARY_TO_3 = {"real": "TRUE", "fake": "FALSE"}
FEVER_TO_STANCE = {"SUPPORTS": "SUPPORTS", "REFUTES": "REFUTES", "NOT ENOUGH INFO": "NEUTRAL"}
VERDICTS = ["REAL", "FAKE", "PARTIALLY TRUE", "MISLEADING", "UNVERIFIABLE"]

# --- runtime (master §17) ---
MODE = os.environ.get("MODE", "online")  # online | offline | lite | classifier_only
