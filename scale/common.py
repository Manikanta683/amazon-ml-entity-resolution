#!/usr/bin/env python3
"""Shared constants, vectorized normalization, and helpers for the scaled pipeline."""
import os
import re
import time

import numpy as np
import pandas as pd

BASE = os.path.expanduser("~/workspace/amazon_ml")
RES = os.path.join(BASE, "student_resource", "student_resource")
DATASET = os.path.join(RES, "dataset")
SCALE = os.path.join(BASE, "scale")
NORM = os.path.join(SCALE, "norm")
ART = os.path.join(SCALE, "artifacts")
OUT_REAL = os.path.join(BASE, "output_real")
TMP = os.path.join(SCALE, "tmp")
for _d in (SCALE, NORM, ART, OUT_REAL, TMP):
    os.makedirs(_d, exist_ok=True)

TRAIN_FILES = {
    "s1": "train/train_source1.tsv",
    "s2": "train/train_source2.tsv",
    "s3": "train/train_source3.tsv",
    "gt": "train/train_ground_truth.tsv",
}
TEST_FILES = {
    "s1": "test/test_source1.tsv",
    "s2": "test/test_source2.tsv",
    "s3": "test/test_source3.tsv",
}

NORM_COLS = ["entity_id", "name_norm", "address_norm", "country_norm", "text"]
CHUNK_ROWS = 200_000
SEED = 42

# ---------------------------------------------------------------------------
# Normalization (fully vectorized pandas str ops)
# ---------------------------------------------------------------------------
JUNK_LITERALS = {
    "nan", "none", "null", "n/a", "na", "-", "--", "---", "##", "#", "###",
    "nil", "unknown", "unk", "tbd", ".", "..",
}

MS_PREFIX_RE = r"^m\s*/\s*s\.?\s*"

SUFFIX_RE = re.compile(
    r"\b(private\s+limited|public\s+limited|limited\s+liability(\s+company)?|"
    r"pvt\.?|private|ltd\.?|limited|inc\.?|incorporated|corp\.?|corporation|"
    r"llc|llp|pllc|plc|gmbh|sarl|sas|eurl|pty|srl|bv|ag|ab|oy|aps|sl|"
    r"\bsa\b|co\.?|company)\b"
)

ABBR_MAP = [
    ("road", "rd"), ("street", "st"), ("avenue", "ave"), ("boulevard", "blvd"),
    ("lane", "ln"), ("drive", "dr"), ("highway", "hwy"), ("plaza", "plz"),
    ("square", "sq"), ("building", "bldg"), ("floor", "fl"), ("apartment", "apt"),
    ("marg", "mg"), ("nagar", "ngr"), ("chowk", "chk"),
    ("technologies", "tech"), ("technology", "tech"),
    ("services", "svc"), ("industries", "ind"),
]
ABBR_RES = [(re.compile(r"\b%s\b" % re.escape(k)), v) for k, v in ABBR_MAP]

# keep a-z 0-9, Devanagari, Latin Extended (French accents), whitespace
PUNCT_RE = re.compile(r"[^a-z0-9\u0900-\u097f\u00c0-\u024f\s]")
WS_RE = re.compile(r"\s+")
NUM_RE = re.compile(r"\d+")


def _base_clean(s: pd.Series) -> pd.Series:
    s = s.fillna("")
    s = s.str.lower()
    s = s.mask(s.isin(JUNK_LITERALS), "")
    return s


def norm_name_col(s: pd.Series) -> pd.Series:
    s = _base_clean(s)
    s = s.str.replace(MS_PREFIX_RE, "", regex=True)
    for _ in range(2):
        s = s.str.replace(SUFFIX_RE, " ", regex=True)
    for pat, rep in ABBR_RES:
        s = s.str.replace(pat, rep, regex=True)
    s = s.str.replace(PUNCT_RE, " ", regex=True)
    s = s.str.replace(WS_RE, " ", regex=True).str.strip()
    return s


def norm_address_col(s: pd.Series) -> pd.Series:
    s = _base_clean(s)
    for pat, rep in ABBR_RES:
        s = s.str.replace(pat, rep, regex=True)
    s = s.str.replace(PUNCT_RE, " ", regex=True)
    s = s.str.replace(WS_RE, " ", regex=True).str.strip()
    return s


def norm_country_col(s: pd.Series) -> pd.Series:
    s = s.fillna("").str.strip().str.casefold()
    return s


def normalize_chunk(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "entity_id": df["entity_id"],
        "name_norm": norm_name_col(df["business_name"]),
        "address_norm": norm_address_col(df["business_address"]),
        "country_norm": norm_country_col(df["country"]),
    })
    out["text"] = (out["name_norm"] + " " + out["address_norm"]).str.strip()
    return out[NORM_COLS]


def read_chunks(path, usecols=None, chunksize=CHUNK_ROWS):
    return pd.read_csv(
        path, sep="\t", dtype=str, na_filter=False, keep_default_na=False,
        engine="c", chunksize=chunksize, usecols=usecols,
    )


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Metric: macro-averaged F_0.5 per S1 entity
# ---------------------------------------------------------------------------
def macro_f05_from_counts(tp, fp, fn):
    """tp/fp/fn: per-entity arrays. Returns macro F_0.5."""
    tp = np.asarray(tp, dtype=float)
    fp = np.asarray(fp, dtype=float)
    fn = np.asarray(fn, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
        rec = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0.0)
        # entities with no truth and no predictions -> 1.0
        empty_both = (tp + fp == 0) & (tp + fn == 0)
        denom = 0.25 * prec + rec
        f = np.where(denom > 0, 1.25 * prec * rec / np.maximum(denom, 1e-12), 0.0)
        f = np.where(empty_both, 1.0, f)
    return float(np.mean(f))


def pick_threshold_midplateau(thresholds, scores):
    """Midpoint of the contiguous max plateau around the argmax."""
    thresholds = np.asarray(thresholds)
    scores = np.asarray(scores)
    best = scores.max()
    i = int(np.argmax(scores))
    lo = hi = i
    while lo > 0 and scores[lo - 1] >= best - 1e-9:
        lo -= 1
    while hi < len(scores) - 1 and scores[hi + 1] >= best - 1e-9:
        hi += 1
    return float((thresholds[lo] + thresholds[hi]) / 2), float(best)
