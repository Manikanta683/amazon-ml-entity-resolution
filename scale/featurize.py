#!/usr/bin/env python3
"""Stage-2 pair features (rapidfuzz + Jaccard + TF-IDF cosine + misc)."""
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from rapidfuzz import fuzz

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E402

NUM_RE = re.compile(r"\d+")
N_FEATURES = 20


def _prep(text):
    toks = text.split()
    nospace = text.replace(" ", "")
    bigrams = {nospace[i:i + 2] for i in range(len(nospace) - 1)} if len(nospace) >= 2 else set()
    return {
        "t": text,
        "tok": set(toks),
        "bi": bigrams,
        "nums": set(NUM_RE.findall(text)),
        "len": len(text),
    }


def _jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _containment(a, b):
    """|a ∩ b| / min(|a|, |b|): is the smaller set contained in the larger?"""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _pair_features(p1, p2, cos, ctry_match):
    n1, a1 = p1["name"], p1["addr"]
    n2, a2 = p2["name"], p2["addr"]
    t1, t2 = n1["t"], n2["t"]
    s1, s2 = a1["t"], a2["t"]
    f = np.empty(N_FEATURES, dtype=np.float32)
    f[0] = fuzz.ratio(t1, t2)
    f[1] = fuzz.token_set_ratio(t1, t2)
    f[2] = fuzz.token_sort_ratio(t1, t2)
    f[3] = fuzz.partial_ratio(t1, t2)
    f[4] = fuzz.ratio(s1, s2)
    f[5] = fuzz.token_set_ratio(s1, s2)
    f[6] = fuzz.token_sort_ratio(s1, s2)
    f[7] = fuzz.partial_ratio(s1, s2)
    f[8] = _jaccard(n1["tok"], n2["tok"])
    f[9] = _jaccard(a1["tok"], a2["tok"])
    f[10] = _jaccard(n1["bi"], n2["bi"])
    f[11] = cos * 100.0
    f[12] = 1.0 if ctry_match else 0.0
    f[13] = abs(n1["len"] - n2["len"]) / max(1, max(n1["len"], n2["len"]))
    f[14] = abs(a1["len"] - a2["len"]) / max(1, max(a1["len"], a2["len"]))
    f[15] = (len(a1["nums"] & a2["nums"]) / len(a1["nums"])
              if a1["nums"] else 1.0)
    # v2: token containment (handles "mr innovative systems" vs "innovative systems")
    f[16] = _containment(n1["tok"], n2["tok"])
    f[17] = _containment(a1["tok"], a2["tok"])
    # v2: bigram containment
    f[18] = _containment(n1["bi"], n2["bi"])
    f[19] = _containment(a1["bi"], a2["bi"])
    return f


def featurize_chunk(s1_recs, cand_rows_list, cand_cos_list,
                    pool_names, pool_addrs, pool_ctry, n_jobs=2):
    """s1_recs: list of (name, addr, ctry). cand_rows_list: per-S1 pool row idx
    (already stage-1 filtered). Returns X (n_pairs, 16) float32 and an array
    mapping each pair -> s1 position."""
    # precompute per-S1 and per-unique-candidate string stats
    s1_pre = [{"name": _prep(n), "addr": _prep(a), "ctry": c}
              for n, a, c in s1_recs]
    uniq_rows = set()
    for rows in cand_rows_list:
        uniq_rows.update(rows)
    cand_pre = {r: {"name": _prep(pool_names[r]), "addr": _prep(pool_addrs[r]),
                    "ctry": pool_ctry[r]} for r in uniq_rows}

    pairs = []  # (s1_pos, row, cos)
    for pos, (rows, coss) in enumerate(zip(cand_rows_list, cand_cos_list)):
        for r, c in zip(rows, coss):
            pairs.append((pos, r, c))
    n_pairs = len(pairs)
    X = np.empty((n_pairs, N_FEATURES), dtype=np.float32)
    pair_s1 = np.empty(n_pairs, dtype=np.int32)

    def work(lo, hi):
        for k in range(lo, hi):
            pos, r, c = pairs[k]
            p1, p2 = s1_pre[pos], cand_pre[r]
            X[k] = _pair_features(p1, p2, c, p1["ctry"] != "" and p1["ctry"] == p2["ctry"])
            pair_s1[k] = pos

    if n_jobs > 1 and n_pairs > 2000:
        bounds = np.linspace(0, n_pairs, n_jobs + 1, dtype=int)
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            list(ex.map(lambda b: work(b[0], b[1]),
                        zip(bounds[:-1], bounds[1:])))
    else:
        work(0, n_pairs)
    return X, pair_s1
