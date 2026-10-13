#!/usr/bin/env python3
"""Country-sharded TF-IDF blocking index + candidate generation.

One ShardIndex per country shard:
  pool docs  = S2+S3 rows of that shard (normalized text)
  index      = word-unigram TF-IDF (sublinear_tf, min_df=2, max_df=0.01)
  query(S1 chunk) -> per-S1 top-50 cosine NN  UNION  exact normalized-name
  matches (from pool rows whose name is in the S1 name set), capped at 60,
  with cosine scores (exact-name extras get cosine from the TF-IDF rows).
"""
import gc
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import read_chunks, log  # noqa: E402

MIN_SHARD_DOCS = 500      # pool labels smaller than this -> "other" shard
TOP_NN = 50
CAND_CAP = 60


def discover_shards(pool_paths):
    """Count country labels over pool files.

    Returns (shard_map, shard_counts): {label: shard} and {shard: doc count}.
    Labels with < MIN_SHARD_DOCS docs (and empty labels) map to 'other'.
    """
    counts = Counter()
    for path in pool_paths:
        for chunk in read_chunks(path, usecols=["country_norm"]):
            counts.update(chunk["country_norm"].tolist())
    shard_map = {
        label: (label if (label and c >= MIN_SHARD_DOCS) else "other")
        for label, c in counts.items()
    }
    shard_counts = Counter()
    for label, c in counts.items():
        shard_counts[shard_map[label]] += c
    log(f"discovered {len(counts)} country labels; "
        f"shards: {dict(shard_counts)}; "
        f"top labels: {counts.most_common(6)}")
    return shard_map, dict(shard_counts)


def collect_s1_names(s1_paths, shard, shard_map, limit=None):
    """Distinct non-empty normalized names of a shard's S1 rows (for name map)."""
    names = set()
    n = 0
    for path in s1_paths:
        for chunk in read_chunks(path, usecols=["country_norm", "name_norm"]):
            m = chunk["country_norm"].map(lambda c: shard_map.get(c, "other")) == shard
            sub = chunk.loc[m, "name_norm"]
            sub = sub[sub != ""]
            names.update(sub.tolist())
            n += int(m.sum())
            if limit and n >= limit:
                break
        if limit and n >= limit:
            break
    log(f"[{shard}] S1 rows: {n:,}, distinct names: {len(names):,}")
    return names, n


class ShardIndex:
    """TF-IDF index over one shard's S2+S3 pool."""

    def __init__(self, pool_paths, shard, shard_map, s1_names, n_docs=None):
        t0 = time.time()
        self.shard = shard
        ids_list = []
        namemap = defaultdict(list)  # name -> [row idx]
        # max_df must satisfy max_df * n_docs >= min_df; adapt for tiny pools
        max_df = 0.01 if not n_docs else max(0.01, 3.0 / n_docs)
        max_id_len = 0

        def doc_iter():
            for path in pool_paths:
                for chunk in read_chunks(path):
                    m = (chunk["country_norm"].map(
                        lambda c: shard_map.get(c, "other")) == shard)
                    sub = chunk[m]
                    if len(sub) == 0:
                        continue
                    base = len(ids_list)
                    for j, (eid, nm, tx) in enumerate(zip(
                            sub["entity_id"], sub["name_norm"], sub["text"])):
                        ids_list.append(eid)
                        if len(eid) > max_id_len:
                            pass
                        if nm and nm in s1_names:
                            namemap[nm].append(base + j)
                        yield tx

        vec = TfidfVectorizer(
            analyzer="word", token_pattern=r"(?u)\b\w\w+\b",
            min_df=2, max_df=max_df, sublinear_tf=True, dtype=np.float32,
        )
        log(f"[{shard}] fitting TF-IDF ...")
        P = vec.fit_transform(doc_iter()).tocsr()
        self.vec = vec
        self.P = P
        self.Pt = P.T.tocsr()
        self.ids = np.array(ids_list, dtype="S16")
        assert (np.char.str_len(self.ids).max() <= 16), "entity_id too long!"
        del ids_list
        self.namemap = dict(namemap)
        assert len(self.ids) == P.shape[0], "id/matrix row mismatch"
        log(f"[{shard}] pool={P.shape[0]:,} docs, vocab={len(vec.vocabulary_):,}, "
            f"name-keys={len(self.namemap):,} in {time.time() - t0:.0f}s")
        gc.collect()

    def query(self, s1_texts, s1_names, topn=TOP_NN, cap=CAND_CAP):
        """Per-S1 (cand_entity_ids, cand_cosines), cosine-desc, len <= cap."""
        Q = self.vec.transform(s1_texts)
        S = Q.dot(self.Pt).tocsr()  # rows L2-normalized -> values are cosines
        indptr, indices, data = S.indptr, S.indices, S.data
        P, ids, namemap = self.P, self.ids, self.namemap
        all_ids, all_cos = [], []
        for i in range(len(s1_texts)):
            s, e = indptr[i], indptr[i + 1]
            rows, cos = [], []
            if e > s:
                d = data[s:e]
                ix = indices[s:e]
                k = topn if e - s > topn else e - s
                part = np.argpartition(-d, k - 1)[:k]
                order = part[np.argsort(-d[part])]
                rows = ix[order].tolist()
                cos = [float(x) for x in d[order]]
            rset = set(rows)
            nm = s1_names[i]
            if nm and len(rows) < cap:
                qv = None
                for r in namemap.get(nm, ()):
                    if r in rset:
                        continue
                    rset.add(r)
                    if qv is None:
                        qv = Q.getrow(i)
                    c = float(qv.dot(P.getrow(r).T)[0, 0])
                    rows.append(r)
                    cos.append(c)
                    if len(rows) >= cap:
                        break
            if len(rows) > 1:
                order = np.argsort(np.array(cos), kind="stable")[::-1]
                rows = [rows[j] for j in order]
                cos = [cos[j] for j in order]
            all_ids.append([ids[r].decode() for r in rows])
            all_cos.append(cos)
        return all_ids, all_cos
