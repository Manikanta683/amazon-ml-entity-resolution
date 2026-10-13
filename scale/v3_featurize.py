#!/usr/bin/env python3
"""v3 pair features: v2's 20 + 10 new (name/addr TF-IDF cosines, WRatio,
trigram sims, consonant-skeleton sims, first-token, digit Jaccard, v3-norm
containment). All features computed only from the provided TSVs.

Also exposes cheap_prefilter_stats() for the two-tier inference: a small
set-op-only feature bundle used to cut top-100 -> survivors before the
expensive rapidfuzz features are computed.
"""
import os
import re
import sys
import unicodedata
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from rapidfuzz import fuzz

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E402

NUM_RE = re.compile(r"\d+")
N_FEATURES = 30

# ---- v3 normalization helpers (India: transliteration/diacritics) ----
DOUBLE_RE = re.compile(r"([a-z])\1")          # aggarwal -> agarwal
VOWEL_RE = re.compile(r"[aeiou]")


def fold_ascii(s: str) -> str:
    """NFKD diacritic fold: ríbbon -> ribbon; drops non-Latin scripts."""
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")


def v3_norm(s: str) -> str:
    """Diacritic-folded + double-letter-collapsed (transliteration-robust)."""
    return DOUBLE_RE.sub(r"\1", fold_ascii(s))


def consonant_skeleton(s: str) -> str:
    """Vowel-stripped nospace string: mohan/mohun -> mhn."""
    return VOWEL_RE.sub("", s.replace(" ", ""))


def _trigrams(nospace):
    return ({nospace[i:i + 3] for i in range(len(nospace) - 2)}
            if len(nospace) >= 3 else set())


def _prep(text):
    toks = text.split()
    nospace = text.replace(" ", "")
    bigrams = ({nospace[i:i + 2] for i in range(len(nospace) - 1)}
               if len(nospace) >= 2 else set())
    v3t = v3_norm(text)
    return {
        "t": text,
        "tok": set(toks),
        "bi": bigrams,
        "tri": _trigrams(nospace),
        "cons": consonant_skeleton(text),
        "nums": set(NUM_RE.findall(text)),
        "len": len(text),
        "first": toks[0] if toks else "",
        "v3tok": set(v3t.split()),
    }


def _jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _containment(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _pair_features(p1, p2, cos, cos_name, cos_addr, ctry_match):
    n1, a1 = p1["name"], p1["addr"]
    n2, a2 = p2["name"], p2["addr"]
    t1, t2 = n1["t"], n2["t"]
    s1, s2 = a1["t"], a2["t"]
    f = np.empty(N_FEATURES, dtype=np.float32)
    # --- v2's 20 (unchanged semantics) ---
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
    # --- v3: 10 new ---
    f[20] = cos_name * 100.0          # name-only TF-IDF cosine
    f[21] = cos_addr * 100.0          # address-only TF-IDF cosine
    f[22] = fuzz.WRatio(t1, t2)       # weighted ratio on name
    f[23] = _jaccard(n1["tri"], n2["tri"])       # char-3gram Jaccard, name
    f[24] = _containment(n1["tri"], n2["tri"])   # char-3gram containment, name
    f[25] = _seq_jaccard(n1["cons"], n2["cons"])
    f[26] = _seq_jaccard(a1["cons"], a2["cons"])
    f[27] = 1.0 if (n1["first"] and n1["first"] == n2["first"]) else 0.0
    f[28] = _jaccard(a1["nums"], a2["nums"])
    f[29] = _containment(n1["v3tok"], n2["v3tok"])  # v3-norm token containment
    return f


def _seq_jaccard(c1, c2):
    """Jaccard over char multisets of consonant skeletons (order-free)."""
    if not c1 and not c2:
        return 1.0
    if not c1 or not c2:
        return 0.0
    s1, s2 = set(c1), set(c2)
    return len(s1 & s2) / len(s1 | s2)


# ---------------------------------------------------------------------------
# Cheap prefilter (set ops only, no rapidfuzz): used to cut top-100 -> survivors
# ---------------------------------------------------------------------------
N_PREFILTER = 6  # [cos, name_tok_cont, addr_tok_cont, name_bi_cont, addr_bi_cont, name_tri_cont]


def cheap_stats(p1, p2, cos):
    n1, a1 = p1["name"], p1["addr"]
    n2, a2 = p2["name"], p2["addr"]
    return (
        cos,
        _containment(n1["tok"], n2["tok"]),
        _containment(a1["tok"], a2["tok"]),
        _containment(n1["bi"], n2["bi"]),
        _containment(a1["bi"], a2["bi"]),
        _containment(n1["tri"], n2["tri"]),
    )


def featurize_chunk_v3(s1_recs, cand_rows_list, cand_cos_list,
                       pool_names, pool_addrs, pool_ctry,
                       cos_name_list=None, cos_addr_list=None, n_jobs=1,
                       prefilter_only=False):
    """30-feature chunk featurizer (mirrors featurize.featurize_chunk API).

    cos_name_list / cos_addr_list: per-S1 lists of name-only / addr-only
    TF-IDF cosines; if None, falls back to the text cosine (feature 11 value).
    prefilter_only=True -> returns only the 6 cheap stats (no rapidfuzz).
    """
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
    nf = N_PREFILTER if prefilter_only else N_FEATURES
    X = np.empty((n_pairs, nf), dtype=np.float32)
    pair_s1 = np.empty(n_pairs, dtype=np.int32)

    def work(lo, hi):
        for k in range(lo, hi):
            pos, r, c, cn, ca = pairs[k]
            p1, p2 = s1_pre[pos], cand_pre[r]
            if prefilter_only:
                X[k] = cheap_stats(p1, p2, c)
            else:
                X[k] = _pair_features(
                    p1, p2, c, cn, ca,
                    p1["ctry"] != "" and p1["ctry"] == p2["ctry"])
            pair_s1[k] = pos

    if n_jobs > 1 and n_pairs > 2000:
        bounds = np.linspace(0, n_pairs, n_jobs + 1, dtype=int)
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            list(ex.map(lambda b: work(b[0], b[1]),
                        zip(bounds[:-1], bounds[1:])))
    else:
        work(0, n_pairs)
    return X, pair_s1
