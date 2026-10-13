#!/usr/bin/env python3
"""v3 blocking: dual word-TF-IDF indices, top-100 per S1.

Stage 1 (v2 behavior): word-unigram TF-IDF on `text`, top-15 + exact-name
extras.
Stage 2 (new): word-unigram TF-IDF on v3-normalized text
(diacritic-folded + double-letter-collapsed: ríbbon->ribbon,
aggarwal->agarwal), top-50. Union, dedup, cap 100.

Rationale: the India gap is transliteration variants that share NO word
tokens with the query (word TF-IDF cosine = 0, invisible at any top-K).
v3-normalization maps the common variant classes to the same tokens, so the
cheap sparse word index catches them. A char-bigram TF-IDF was prototyped
but rejected: common bigrams hit millions of pool docs, making the query
product near-dense (OOM on 4.7M-doc shards).

ShardIndexV3 subclasses block.ShardIndex and reuses its word index, pool ids
and exact-name map. All data comes from the provided TSVs.
"""
import os
import sys
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E402
from block import ShardIndex  # noqa: E402
from common import read_chunks, log  # noqa: E402
from v3_featurize import v3_norm  # noqa: E402

V3_STAGE1_N = 15
V3_STAGE2_N = 50
V3_FILL_TO = 100


def _reverse_row_lookup(shard_index):
    """entity_id -> pool row (from the S16 ids array)."""
    return {eid.decode(): r for r, eid in enumerate(shard_index.ids)}


class ShardIndexV3(ShardIndex):
    def build_v3norm_index(self, pool_paths, shard, shard_map, n_docs=None):
        """Word TF-IDF over v3-normalized pool texts (same row order)."""
        t0 = time.time()
        max_df = 0.01 if not n_docs else max(0.01, 3.0 / n_docs)

        def doc_iter():
            for path in pool_paths:
                for chunk in read_chunks(path):
                    m = (chunk["country_norm"].map(
                        lambda c: shard_map.get(c, "other")) == shard)
                    sub = chunk[m]
                    if len(sub) == 0:
                        continue
                    # vectorized-ish: python-level map over the text column
                    yield from (v3_norm(t) for t in sub["text"].tolist())

        vec = TfidfVectorizer(
            analyzer="word", token_pattern=r"(?u)\b\w\w+\b",
            min_df=2, max_df=max_df, sublinear_tf=True, dtype=np.float32,
        )
        log(f"[{shard}] fitting v3norm TF-IDF ...")
        V = vec.fit_transform(doc_iter()).tocsr()
        assert V.shape[0] == self.P.shape[0], "v3norm/word row mismatch"
        self.vvec = vec
        self.V = V
        self.Vt = V.T.tocsr()
        log(f"[{shard}] v3norm index: docs={V.shape[0]:,}, "
            f"vocab={len(vec.vocabulary_):,} in {time.time() - t0:.0f}s")

    def query_rows_stage1(self, s1_texts, s1_names, stage1_n=V3_STAGE1_N):
        """Stage-1 only, returning pool rows (not entity ids)."""
        s1_ids, s1_cos = self.query(s1_texts, s1_names, topn=stage1_n,
                                    cap=V3_FILL_TO)
        rev = _reverse_row_lookup(self)
        rows = [[rev[e] for e in ids_i] for ids_i in s1_ids]
        return rows, s1_cos


def query_two_stage(idx: ShardIndexV3, s1_texts, s1_names,
                    stage1_n=V3_STAGE1_N, stage2_n=V3_STAGE2_N,
                    fill_to=V3_FILL_TO):
    """Two-stage query.

    Returns per-S1 (rows, cos): pool row indices and the TF-IDF cosine from
    whichever index produced the candidate (word cos for stage-1, v3norm cos
    for stage-2-only extras). Stage-1 (incl. exact-name extras) first, then
    stage-2 extras by v3norm-cosine desc. len <= fill_to.
    """
    s1_ids, s1_cos = idx.query(s1_texts, s1_names, topn=stage1_n, cap=fill_to)
    rev = _reverse_row_lookup(idx)
    # stage-2 queries use v3-normalized S1 texts
    v3_texts = [v3_norm(t) for t in s1_texts]
    Q = idx.vvec.transform(v3_texts)
    S = Q.dot(idx.Vt).tocsr()
    indptr, indices, data = S.indptr, S.indices, S.data

    all_rows, all_cos = [], []
    for i in range(len(s1_texts)):
        rows = [rev[e] for e in s1_ids[i]]
        cos = list(s1_cos[i])
        rset = set(rows)
        s, e = indptr[i], indptr[i + 1]
        if e > s and len(rows) < fill_to:
            d = data[s:e]
            ix = indices[s:e]
            k = min(stage2_n * 4, e - s)
            part = np.argpartition(-d, k - 1)[:k]
            order = part[np.argsort(-d[part])]
            added = 0
            for t in order:
                r = int(ix[t])
                if r in rset:
                    continue
                rset.add(r)
                rows.append(r)
                cos.append(float(d[t]))
                added += 1
                if added >= stage2_n or len(rows) >= fill_to:
                    break
        all_rows.append(rows)
        all_cos.append(cos)
    return all_rows, all_cos
