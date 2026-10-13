#!/usr/bin/env python3
"""Regenerate candidate_pairs.tsv from top_rows (the exact inference set).

Per the problem statement, candidate_pairs.tsv must be "the exact set of
records you feed into your matching model for inference" -- i.e. top_rows
(STAGE1_TOP=15 per S1), NOT cand60 (the wider phase-A pool). The previous
merge wrote cand60, overstating the candidate count ~3.5x and hurting the
final-ranking tiebreaker that rewards smaller candidate sets.

Reads the cached per-shard npz files; no re-blocking or re-inference needed.
"""
import os
import numpy as np
from common import NORM, TMP, ART, read_chunks, log
from block import discover_shards

OUT_REAL = os.path.expanduser("~/workspace/amazon_ml/output_real")


def main():
    # test-time shard map (train-time file lacks france)
    pool_paths = [os.path.join(NORM, "test_source2.tsv"),
                  os.path.join(NORM, "test_source3.tsv")]
    shard_map, _ = discover_shards(pool_paths)
    shards = sorted(set(shard_map.values()))
    log(f"shards: {shards}")

    stores = {}
    for shard in shards:
        p = os.path.join(TMP, f"test_{shard}_cand.npz")
        if not os.path.exists(p):
            continue
        d = np.load(p, allow_pickle=True)
        s1_ids = [str(x) for x in d["s1_ids"].tolist()]
        stores[shard] = {
            "pos": {sid: i for i, sid in enumerate(s1_ids)},
            "top_rows": d["top_rows"],
            "pool_ids": d["pool_ids"],
        }
        log(f"[{shard}] loaded: {len(s1_ids):,} S1")

    cp = os.path.join(OUT_REAL, "candidate_pairs.tsv")
    n_out, tot_c, mx = 0, 0, 0
    with open(cp, "w") as fc:
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        for chunk in read_chunks(os.path.join(NORM, "test_source1.tsv"),
                                 usecols=["entity_id", "country_norm"]):
            cl = []
            for sid, ctry in zip(chunk["entity_id"], chunk["country_norm"]):
                shard = shard_map.get(ctry, "other")
                st = stores.get(shard)
                if st is None or sid not in st["pos"]:
                    cl.append(f"{sid}\t\n")
                    n_out += 1
                    continue
                p = st["pos"][sid]
                rows = st["top_rows"][p]
                cids = [st["pool_ids"][r].decode() for r in rows if r >= 0]
                # de-dupe preserving order (shouldn't happen, but be safe)
                seen, uniq = set(), []
                for c in cids:
                    if c not in seen:
                        seen.add(c)
                        uniq.append(c)
                tot_c += len(uniq)
                mx = max(mx, len(uniq))
                cl.append(f"{sid}\t{','.join(uniq)}\n")
                n_out += 1
            fc.writelines(cl)
    log(f"wrote {cp}: {n_out:,} rows, mean_cands={tot_c/max(n_out,1):.2f}, max={mx}")


if __name__ == "__main__":
    main()
