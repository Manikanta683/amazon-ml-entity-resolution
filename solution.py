#!/usr/bin/env python3
"""Entity-resolution solution for the Amazon ML Challenge 2026 (smoke-test build).

Pipeline
--------
1. Normalize name/address text (lowercase, unicode NFKD, legal-suffix strip,
   address-abbreviation standardization, punctuation removal).
2. Blocking: TF-IDF char_wb (3-5 grams) over normalized name+address, cosine
   nearest-neighbours, top-K=80 per S1 record from the same-country S2+S3 pool
   (country labels matched dynamically; empty country or tiny pool falls back
   to the global S2+S3 pool). The candidate list per S1 is exactly what lands
   in candidate_pairs.tsv.
3. Pair features (rapidfuzz ratios, token/char Jaccard, TF-IDF cosine,
   country flag, length diffs, address-number recall).
4. sklearn LogisticRegression (class_weight='balanced') on train pairs:
   positives = ground-truth pairs found by blocking, negatives = other blocked
   candidates.
5. Per-pair probability threshold tuned on a held-out split of train S1
   entities to maximize macro-averaged F_0.5; best threshold applied to test.
   Per entity: all candidates above threshold (sorted by probability); empty
   allowed.

Outputs (both exactly the challenge format, tab-separated)
----------------------------------------------------------
    output/matching_results.tsv   source1_entity_id, matched_entity_ids
    output/candidate_pairs.tsv    source1_entity_id, candidate_entity_ids

Usage
-----
    python3 solution.py --data-dir synthetic --output-dir output [--baseline]
"""
import argparse
import csv
import os
import re
import time
import unicodedata
from collections import defaultdict

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

TOP_K = 80
MIN_POOL_SIZE = 10          # below this, same-country pool -> global pool
SEED = 42
BETA2 = 0.25                # F_0.5

# ---------------------------------------------------------------------------
# 1. Text normalization
# ---------------------------------------------------------------------------
LEGAL_SUFFIX_RE = re.compile(
    r"\b(private\s+limited|public\s+limited|limited\s+liability|pvt|private|"
    r"ltd|limited|inc|incorporated|corp|corporation|llc|llp|pllc|plc|co|"
    r"company|gmbh|sarl|sas|sa|eurl|pty|srl|spa|bv|ag|ab|oy|aps|sl|gbr)\s*$"
)

ABBR_MAP = {
    "road": "rd", "street": "st", "avenue": "ave", "boulevard": "blvd",
    "lane": "ln", "drive": "dr", "plaza": "plz", "square": "sq",
    "highway": "hwy", "marg": "mg", "nagar": "ngr", "chowk": "chk",
    "corporation": "corp", "incorporated": "inc", "company": "co",
    "limited": "ltd", "private": "pvt", "technologies": "tech",
    "technology": "tech", "services": "svc", "industries": "ind",
}

PUNCT_RE = re.compile(r"[^a-z0-9\s]")
WS_RE = re.compile(r"\s+")


def normalize(text):
    """Lowercase, strip accents, drop legal suffixes, standardize abbreviations."""
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = unicodedata.normalize("NFKD", t)
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = PUNCT_RE.sub(" ", t)
    t = WS_RE.sub(" ", t).strip()
    # strip legal suffixes repeatedly from the end
    while True:
        new = LEGAL_SUFFIX_RE.sub("", t).strip()
        if new == t:
            break
        t = new
    # standardize abbreviations token-wise
    t = " ".join(ABBR_MAP.get(tok, tok) for tok in t.split())
    return t


def norm_country(c):
    return "" if not isinstance(c, str) else c.strip().casefold()


NUM_RE = re.compile(r"\d+")


def numbers(text):
    return set(NUM_RE.findall(text or ""))


def token_jaccard(a, b):
    sa, sb = set(a.split()), set(b.split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def char_bigram_jaccard(a, b):
    def bigrams(s):
        s = s.replace(" ", "")
        return {s[i:i + 2] for i in range(max(0, len(s) - 1))}
    ba, bb = bigrams(a), bigrams(b)
    if not ba and not bb:
        return 1.0
    if not ba or not bb:
        return 0.0
    return len(ba & bb) / len(ba | bb)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_split(data_dir, split, with_truth):
    def rd(name):
        df = pd.read_csv(f"{data_dir}/{split}_{name}.tsv", sep="\t",
                         dtype=str, keep_default_na=False)
        for col in ("business_name", "business_address", "country"):
            if col not in df.columns:
                df[col] = ""
        return df

    s1 = rd("source1")
    s2 = rd("source2")
    s3 = rd("source3")
    truth = {}
    if with_truth:
        gt = pd.read_csv(f"{data_dir}/{split}_ground_truth.tsv", sep="\t",
                         dtype=str, keep_default_na=False)
        for _, row in gt.iterrows():
            raw = row["matched_entity_ids"].strip()
            truth[row["source1_entity_id"]] = (
                [x.strip() for x in raw.split(",") if x.strip()] if raw else [])
    return s1, s2, s3, truth


def add_norm(df):
    df = df.copy()
    df["n_name"] = df["business_name"].map(normalize)
    df["n_addr"] = df["business_address"].map(normalize)
    df["combined"] = df["n_name"] + " " + df["n_addr"]
    df["n_country"] = df["country"].map(norm_country)
    return df


# ---------------------------------------------------------------------------
# 2. Blocking
# ---------------------------------------------------------------------------
class Blocker:
    """TF-IDF char_wb blocking with per-country cosine-NN indexes."""

    def __init__(self, top_k=TOP_K, seed=SEED):
        self.top_k = top_k
        self.seed = seed

    def fit(self, pool_df):
        self.pool = pool_df.reset_index(drop=True)
        self.pool_ids = self.pool["entity_id"].to_numpy()
        self.pool_country = self.pool["n_country"].to_numpy()
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                   lowercase=False, min_df=1)
        self.X_pool = self.vec.fit_transform(self.pool["combined"])
        # per-country indexes + one global fallback index
        self.nn = {}
        self.pools = {}
        countries = sorted(set(self.pool_country))
        self.pools["__global__"] = np.arange(len(self.pool))
        for c in countries:
            idx = np.where(self.pool_country == c)[0]
            if len(idx) >= MIN_POOL_SIZE:
                self.pools[c] = idx
        for key, idx in self.pools.items():
            nn = NearestNeighbors(
                n_neighbors=min(self.top_k, len(idx)), metric="cosine")
            nn.fit(self.X_pool[idx])
            self.nn[key] = nn
        return self

    def query(self, s1_df):
        """Return dict s1_id -> list of candidate pool entity_ids (NN rank order)."""
        X_s1 = self.vec.transform(s1_df["combined"])
        s1_ids = s1_df["entity_id"].to_numpy()
        s1_country = s1_df["n_country"].to_numpy()
        out = {}
        # group S1 rows by resolved pool for vectorized queries
        groups = defaultdict(list)
        for i, c in enumerate(s1_country):
            key = c if c in self.pools else "__global__"
            groups[key].append(i)
        zero_rows = np.asarray(X_s1.sum(axis=1)).ravel() == 0
        for key, rows in groups.items():
            idx = self.pools[key]
            Xq = X_s1[rows]
            dist, loc = self.nn[key].kneighbors(Xq)
            for r, drow, lrow in zip(rows, dist, loc):
                if zero_rows[r]:
                    # degenerate (empty) query vector: arbitrary pool head
                    cands = [self.pool_ids[j] for j in idx[:self.top_k]]
                else:
                    cands = [self.pool_ids[idx[j]] for j in lrow]
                out[s1_ids[r]] = cands
        # preserve S1 input order
        return {sid: out[sid] for sid in s1_ids}


# ---------------------------------------------------------------------------
# 3. Pair features
# ---------------------------------------------------------------------------
def pair_features(s1_row, cand_rows, tfidf_cos):
    """cand_rows: DataFrame slice; tfidf_cos: cosine vs S1 vector (aligned)."""
    feats = []
    n1, a1 = s1_row["n_name"], s1_row["n_addr"]
    nums1 = numbers(a1)
    c1 = s1_row["n_country"]
    for j, (_, cr) in enumerate(cand_rows.iterrows()):
        n2, a2 = cr["n_name"], cr["n_addr"]
        nums2 = numbers(a2)
        f = [
            fuzz.ratio(n1, n2) / 100.0,
            fuzz.token_set_ratio(n1, n2) / 100.0,
            fuzz.token_sort_ratio(n1, n2) / 100.0,
            fuzz.partial_ratio(n1, n2) / 100.0,
            fuzz.ratio(a1, a2) / 100.0,
            fuzz.token_set_ratio(a1, a2) / 100.0,
            fuzz.token_sort_ratio(a1, a2) / 100.0,
            fuzz.partial_ratio(a1, a2) / 100.0,
            token_jaccard(n1, n2),
            token_jaccard(a1, a2),
            char_bigram_jaccard(n1, n2),
            float(tfidf_cos[j]),
            1.0 if (c1 and c1 == cr["n_country"]) else 0.0,
            abs(len(n1) - len(n2)) / max(len(n1), len(n2), 1),
            abs(len(a1) - len(a2)) / max(len(a1), len(a2), 1),
            (len(nums1 & nums2) / len(nums1)) if nums1 else 1.0,
        ]
        feats.append(f)
    return np.asarray(feats, dtype=np.float64)


def build_pair_matrix(s1_df, pool_df, candidates, X_s1=None, blocker=None):
    """Feature matrix + (s1_id, cand_id) keys for all blocked pairs."""
    if X_s1 is None:
        X_s1 = blocker.vec.transform(s1_df["combined"])
    pool_pos = {eid: i for i, eid in enumerate(pool_df["entity_id"])}
    s1_pos = {eid: i for i, eid in enumerate(s1_df["entity_id"])}
    X_list, keys = [], []
    X_pool = blocker.X_pool
    for sid, cands in candidates.items():
        i = s1_pos[sid]
        srow = s1_df.iloc[i]
        v = X_s1[i]
        cidx = np.array([pool_pos[c] for c in cands])
        cos = np.asarray((v @ X_pool[cidx].T).todense()).ravel()
        # clip tiny negatives from float error
        cos = np.clip(cos, 0.0, 1.0)
        F = pair_features(srow, pool_df.iloc[cidx], cos)
        X_list.append(F)
        keys.extend((sid, c) for c in cands)
    return np.vstack(X_list), keys


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------
def entity_f05(pred, true):
    p, t = set(pred), set(true)
    if not p and not t:
        return 1.0
    if not p or not t:
        return 0.0
    tp = len(p & t)
    prec, rec = tp / len(p), tp / len(t)
    if prec + rec == 0:
        return 0.0
    return (1 + BETA2) * prec * rec / (BETA2 * prec + rec)


def macro_f05(pred_map, truth_map, s1_ids):
    return float(np.mean([entity_f05(pred_map.get(s, []), truth_map.get(s, []))
                          for s in s1_ids]))


# ---------------------------------------------------------------------------
# 4/5. Train, tune threshold, predict
# ---------------------------------------------------------------------------
def tune_threshold(probs_by_s1, truth_map, val_ids):
    grid = np.arange(0.02, 1.0, 0.02)
    scores = []
    for t in grid:
        pred = {s: [c for c, p in probs_by_s1[s] if p >= t] for s in val_ids}
        scores.append(macro_f05(pred, truth_map, val_ids))
    scores = np.array(scores)
    best_f = scores.max()
    # midpoint of the max plateau: more robust than the first argmax
    plateau = grid[np.isclose(scores, best_f)]
    best_t = float((plateau.min() + plateau.max()) / 2)
    return best_t, float(best_f)


def probs_by_entity(keys, proba):
    out = defaultdict(list)
    for (sid, cid), p in zip(keys, proba):
        out[sid].append((cid, float(p)))
    for sid in out:
        out[sid].sort(key=lambda x: -x[1])
    return out


def train_model(X, y):
    clf = LogisticRegression(class_weight="balanced", max_iter=1000,
                             random_state=SEED)
    clf.fit(X, y)
    return clf


def make_labels(keys, truth_map):
    truth_sets = {s: set(v) for s, v in truth_map.items()}
    return np.array([1 if cid in truth_sets.get(sid, set()) else 0
                     for sid, cid in keys], dtype=int)


# ---------------------------------------------------------------------------
# Baseline (optional comparison): pure fuzzy rule on blocked candidates
# ---------------------------------------------------------------------------
def baseline_predict(s1_df, pool_df, candidates,
                     name_t=0.90, addr_t=0.60):
    pool_by_id = {eid: r for eid, r in pool_df.set_index("entity_id").iterrows()}
    pred = {}
    for _, srow in s1_df.iterrows():
        sid = srow["entity_id"]
        hits = []
        for cid in candidates[sid]:
            cr = pool_by_id[cid]
            if (fuzz.token_set_ratio(srow["n_name"], cr["n_name"]) / 100.0 >= name_t
                    and fuzz.token_set_ratio(srow["n_addr"], cr["n_addr"]) / 100.0 >= addr_t):
                hits.append(cid)
        pred[sid] = hits
    return pred


# ---------------------------------------------------------------------------
# Output + format validation
# ---------------------------------------------------------------------------
def write_pairs(path, s1_ids, id_map, header):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["source1_entity_id", header])
        for sid in s1_ids:
            w.writerow([sid, ",".join(id_map.get(sid, []))])


def validate_outputs(match_path, cand_path, s1_ids, pool_ids):
    """Return list of format violations (empty = all good)."""
    errs = []
    pool_ids = set(pool_ids)
    m = pd.read_csv(match_path, sep="\t", dtype=str, keep_default_na=False)
    c = pd.read_csv(cand_path, sep="\t", dtype=str, keep_default_na=False)
    if list(m.columns) != ["source1_entity_id", "matched_entity_ids"]:
        errs.append(f"matching_results columns: {list(m.columns)}")
    if list(c.columns) != ["source1_entity_id", "candidate_entity_ids"]:
        errs.append(f"candidate_pairs columns: {list(c.columns)}")
    if list(m["source1_entity_id"]) != list(s1_ids):
        errs.append("matching_results rows != one row per S1 in order")
    if list(c["source1_entity_id"]) != list(s1_ids):
        errs.append("candidate_pairs rows != one row per S1 in order")

    def parse(s):
        s = s.strip()
        return [x.strip() for x in s.split(",") if x.strip()] if s else []

    cand_map = {}
    for _, row in c.iterrows():
        ids = parse(row["candidate_entity_ids"])
        cand_map[row["source1_entity_id"]] = ids
        if len(ids) != len(set(ids)):
            errs.append(f"{row['source1_entity_id']}: duplicate candidate ids")
        for i in ids:
            if not (i.startswith("S2-") or i.startswith("S3-")):
                errs.append(f"{row['source1_entity_id']}: bad candidate id {i}")
            if i not in pool_ids:
                errs.append(f"{row['source1_entity_id']}: candidate {i} not in S2/S3")
    for _, row in m.iterrows():
        ids = parse(row["matched_entity_ids"])
        sid = row["source1_entity_id"]
        if len(ids) != len(set(ids)):
            errs.append(f"{sid}: duplicate matched ids")
        for i in ids:
            if not (i.startswith("S2-") or i.startswith("S3-")):
                errs.append(f"{sid}: bad matched id {i}")
            if i not in pool_ids:
                errs.append(f"{sid}: matched {i} not in S2/S3")
        if not set(ids) <= set(cand_map.get(sid, [])):
            errs.append(f"{sid}: matched not subset of candidates")
    return errs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--baseline", action="store_true",
                    help="also score a fuzzy-rule baseline on the val split")
    args = ap.parse_args()
    t0 = time.time()

    s1_tr, s2_tr, s3_tr, truth = load_split(args.data_dir, "train", True)
    s1_te, s2_te, s3_te, _ = load_split(args.data_dir, "test", False)
    s1_tr, s2_tr, s3_tr = add_norm(s1_tr), add_norm(s2_tr), add_norm(s3_tr)
    s1_te, s2_te, s3_te = add_norm(s1_te), add_norm(s2_te), add_norm(s3_te)
    pool_tr = pd.concat([s2_tr, s3_tr], ignore_index=True)
    pool_te = pd.concat([s2_te, s3_te], ignore_index=True)

    # --- split train S1 entities -> sub-train / validation (for threshold) ---
    tr_ids = s1_tr["entity_id"].tolist()
    sub_ids, val_ids = train_test_split(tr_ids, test_size=0.25,
                                        random_state=SEED)
    s1_sub = s1_tr[s1_tr["entity_id"].isin(sub_ids)].reset_index(drop=True)
    s1_val = s1_tr[s1_tr["entity_id"].isin(val_ids)].reset_index(drop=True)

    # --- blocking (fit on train pool, query sub/val S1) ---
    blocker = Blocker().fit(pool_tr)
    cand_sub = blocker.query(s1_sub)
    cand_val = blocker.query(s1_val)

    # --- features + labels for sub-train ---
    X_sub_all = build_pair_matrix(s1_sub, pool_tr, cand_sub, blocker=blocker)
    X_sub, keys_sub = X_sub_all
    y_sub = make_labels(keys_sub, truth)
    print(f"train pairs: {len(y_sub)} ({y_sub.sum()} positive)")

    # --- train, tune threshold on val ---
    clf = train_model(X_sub, y_sub)
    X_val, keys_val = build_pair_matrix(s1_val, pool_tr, cand_val,
                                        blocker=blocker)
    p_val = clf.predict_proba(X_val)[:, 1]
    pv = probs_by_entity(keys_val, p_val)
    thr, val_f05 = tune_threshold(pv, truth, val_ids)
    print(f"chosen threshold: {thr:.2f}  |  validation macro F_0.5: {val_f05:.4f}")

    if args.baseline:
        base_pred = baseline_predict(s1_val, pool_tr, cand_val)
        base_f = macro_f05(base_pred, truth, val_ids)
        print(f"fuzzy baseline validation macro F_0.5: {base_f:.4f}")

    # --- retrain on ALL train pairs, apply to test ---
    X_all, keys_all = build_pair_matrix(
        s1_tr, pool_tr, {**cand_sub, **cand_val}, blocker=blocker)
    y_all = make_labels(keys_all, truth)
    clf = train_model(X_all, y_all)

    blocker_te = Blocker().fit(pool_te)
    cand_te = blocker_te.query(s1_te)
    X_te, keys_te = build_pair_matrix(s1_te, pool_te, cand_te,
                                      blocker=blocker_te)
    p_te = clf.predict_proba(X_te)[:, 1]
    pt = probs_by_entity(keys_te, p_te)
    te_ids = s1_te["entity_id"].tolist()
    matches = {s: [c for c, p in pt.get(s, []) if p >= thr] for s in te_ids}

    out_m = os.path.join(args.output_dir, "matching_results.tsv")
    out_c = os.path.join(args.output_dir, "candidate_pairs.tsv")
    write_pairs(out_m, te_ids, matches, "matched_entity_ids")
    write_pairs(out_c, te_ids, cand_te, "candidate_entity_ids")

    errs = validate_outputs(out_m, out_c, te_ids,
                            pool_te["entity_id"].tolist())
    if errs:
        print("FORMAT VIOLATIONS:")
        for e in errs[:20]:
            print("  -", e)
    else:
        print("format checks: OK "
              f"({len(te_ids)} S1 rows, matched<=candidates, ids valid)")
    n_match = sum(len(v) for v in matches.values())
    print(f"test predictions: {n_match} matches across {len(te_ids)} S1 "
          f"({sum(1 for v in matches.values() if v)} non-empty)")
    print(f"runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
