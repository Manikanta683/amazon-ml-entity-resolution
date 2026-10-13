#!/usr/bin/env python3
"""v4 pair features: v3's 30 + 3 new = 33.

New vs v3:
  f30 exact_name_binary   1.0 if raw name_norm strings are identical else 0.0
                          (blocking-leg signal as a feature, NOT a rule)
  f31 addr_num_match      house/flat number-set Jaccard with leading-zero
                          normalization ("00309" == "309"); both empty -> 0.5
                          (uninformative), one empty -> 0.0
  f32 folded_name_jacc    token Jaccard on transliteration-folded names
                          (Devanagari tokens mapped via mined tmap); falls
                          back to raw token Jaccard when fold_map is None

All features computed only from the provided TSVs.
"""
import os
import re
import sys

import numpy as np
from rapidfuzz import fuzz

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E402
from v3_featurize import _prep, _jaccard, _containment, _seq_jaccard, v3_norm  # noqa: E402

N_FEATURES = 33
NUM_RE = re.compile(r"\d+")


def _norm_num(tok):
    s = tok.lstrip("0")
    return s if s else "0"


def _addr_nums(text):
    return {_norm_num(t) for t in NUM_RE.findall(text)}


def _fold_tokens(text, tmap, dev_suffix):
    """Tokenize with transliteration folding + Devanagari suffix stripping
    (mirrors v4_block_aug.fold_name, minus the join)."""
    if not tmap:
        return text.split()
    out = []
    for t in text.split():
        if t in dev_suffix:
            continue
        if any("\u0900" <= c <= "\u097f" for c in t):
            t = tmap.get(t, t)
        out.append(t)
    return out


def _pair_features_v4(p1, p2, cos, cos_name, cos_addr, ctry_match,
                      tmap, dev_suffix):
    n1, a1 = p1["name"], p1["addr"]
    n2, a2 = p2["name"], p2["addr"]
    t1, t2 = n1["t"], n2["t"]
    s1, s2 = a1["t"], a2["t"]
    f = np.empty(N_FEATURES, dtype=np.float32)
    # --- v3's 30 (unchanged semantics) ---
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
    f[16] = _containment(n1["tok"], n2["tok"])
    f[17] = _containment(a1["tok"], a2["tok"])
    f[18] = _containment(n1["bi"], n2["bi"])
    f[19] = _containment(a1["bi"], a2["bi"])
    f[20] = cos_name * 100.0
    f[21] = cos_addr * 100.0
    f[22] = fuzz.WRatio(t1, t2)
    f[23] = _jaccard(n1["tri"], n2["tri"])
    f[24] = _containment(n1["tri"], n2["tri"])
    f[25] = _seq_jaccard(n1["cons"], n2["cons"])
    f[26] = _seq_jaccard(a1["cons"], a2["cons"])
    f[27] = 1.0 if (n1["first"] and n1["first"] == n2["first"]) else 0.0
    f[28] = _jaccard(a1["nums"], a2["nums"])
    f[29] = _containment(n1["v3tok"], n2["v3tok"])
    # --- v4: 3 new ---
    f[30] = 1.0 if t1 == t2 and t1 else 0.0
    an1, an2 = _addr_nums(s1), _addr_nums(s2)
    if not an1 and not an2:
        f[31] = 0.5
    elif not an1 or not an2:
        f[31] = 0.0
    else:
        f[31] = len(an1 & an2) / len(an1 | an2)
    f[32] = _jaccard(set(_fold_tokens(t1, tmap, dev_suffix)),
                     set(_fold_tokens(t2, tmap, dev_suffix)))
    return f


def featurize_chunk_v4(s1_recs, cand_rows_list, cand_cos_list,
                       pool_names, pool_addrs, pool_ctry,
                       cos_name_list=None, cos_addr_list=None, n_jobs=1,
                       fold_map=None, dev_suffix=frozenset()):
    """33-feature chunk featurizer (mirrors featurize_chunk_v3 API + fold_map)."""
    from concurrent.futures import ThreadPoolExecutor
    s1_pre = [{"name": _prep(n), "addr": _prep(a), "ctry": c}
              for n, a, c in s1_recs]
    uniq_rows = set()
    for rows in cand_rows_list:
        uniq_rows.update(rows)
    cand_pre = {r: {"name": _prep(pool_names[r]), "addr": _prep(pool_addrs[r]),
                    "ctry": pool_ctry[r]} for r in uniq_rows}

    pairs = []
    for pos, (rows, coss) in enumerate(zip(cand_rows_list, cand_cos_list)):
        cn = cos_name_list[pos] if cos_name_list is not None else None
        ca = cos_addr_list[pos] if cos_addr_list is not None else None
        for t, r in enumerate(rows):
            c = coss[t]
            pairs.append((pos, r, c,
                          cn[t] if cn is not None else c,
                          ca[t] if ca is not None else c))
    n_pairs = len(pairs)
    X = np.empty((n_pairs, N_FEATURES), dtype=np.float32)
    pair_s1 = np.empty(n_pairs, dtype=np.int32)

    def work(lo, hi):
        for k in range(lo, hi):
            pos, r, c, cn, ca = pairs[k]
            p1, p2 = s1_pre[pos], cand_pre[r]
            X[k] = _pair_features_v4(
                p1, p2, c, cn, ca,
                p1["ctry"] != "" and p1["ctry"] == p2["ctry"],
                fold_map, dev_suffix)
            pair_s1[k] = pos

    if n_jobs > 1 and n_pairs > 2000:
        bounds = np.linspace(0, n_pairs, n_jobs + 1, dtype=int)
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            list(ex.map(lambda b: work(b[0], b[1]),
                        zip(bounds[:-1], bounds[1:])))
    else:
        work(0, n_pairs)
    return X, pair_s1
