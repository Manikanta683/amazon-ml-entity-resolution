#!/usr/bin/env python3
"""v2 test scoring: featurize test candidates with 20 features, score with v2 model.

Uses cached test candidates (no re-blocking). Outputs matching_results.tsv.
"""
import os
import sys
import pickle
import gc
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import NORM, TMP, ART, read_chunks, log
from featurize import featurize_chunk, N_FEATURES

V2_ART = ART + "_v2"
OUT = os.path.expanduser("~/workspace/amazon_ml/output_v2")
os.makedirs(OUT, exist_ok=True)

log(f"v2 test scoring, N_FEATURES={N_FEATURES}")


def main():
    clf = pickle.load(open(os.path.join(V2_ART, "model.pkl"), "rb"))
    thr = float(open(os.path.join(V2_ART, "threshold.txt")).read().strip())
    log(f"model loaded, thr={thr}")

    # test S1 IDs per shard (from candidate files)
    results = {}  # s1_id -> list of matched cids

    for shard in ("india", "us", "france"):
        log(f"[{shard}] loading candidates...")
        d = np.load(os.path.join(TMP, f"test_{shard}_cand.npz"), allow_pickle=True)
        s1_ids = [str(x) for x in d["s1_ids"].tolist()]
        top_rows = d["top_rows"]
        top_cos = d["top_cos"]
        pool_ids_arr = np.array([x.decode() if isinstance(x, bytes) else str(x)
                                 for x in d["pool_ids"].tolist()])
        ref_rows = sorted(set(np.unique(top_rows).tolist()))
        row_map = {r: i for i, r in enumerate(ref_rows)}
        ref_pids = set(pool_ids_arr[r] for r in ref_rows)
        log(f"[{shard}] {len(s1_ids):,} S1, {len(ref_rows):,} ref rows")

        # load pool fields for ref rows (from normalized test files)
        pf = {}
        for src in ("test_source2", "test_source3"):
            p = os.path.join(NORM, f"{src}.tsv")
            for chunk in read_chunks(p):
                for eid, nm, ad, cy in zip(chunk["entity_id"], chunk["name_norm"],
                                           chunk["address_norm"], chunk["country_norm"]):
                    if eid in ref_pids:
                        pf[eid] = (nm, ad, cy)
                        if len(pf) >= len(ref_pids):
                            break
                if len(pf) >= len(ref_pids):
                    break
            if len(pf) >= len(ref_pids):
                break
        log(f"[{shard}] pool fields: {len(pf):,}")

        # test S1 fields
        s1f = {}
        p1 = os.path.join(NORM, "test_source1.tsv")
        for chunk in read_chunks(p1, usecols=["entity_id", "name_norm",
                                              "address_norm", "country_norm"]):
            for eid, nm, ad, cy in zip(chunk["entity_id"], chunk["name_norm"],
                                       chunk["address_norm"], chunk["country_norm"]):
                s1f[eid] = (nm, ad, cy)
        # filter to this shard's s1_ids
        s1f = {k: s1f[k] for k in s1_ids if k in s1f}
        log(f"[{shard}] s1 fields: {len(s1f):,}")

        pool_n = [pf[pool_ids_arr[r]][0] for r in ref_rows]
        pool_a = [pf[pool_ids_arr[r]][1] for r in ref_rows]
        pool_c = [pf[pool_ids_arr[r]][2] for r in ref_rows]
        comp_pids = [pool_ids_arr[r] for r in ref_rows]
        del pf
        gc.collect()

        # batch score
        B = 20000
        n_s1 = len(s1_ids)
        for b0 in range(0, n_s1, B):
            b1 = min(n_s1, b0 + B)
            s1_recs_b = [s1f[sid] for sid in s1_ids[b0:b1]]
            rows_b = [[row_map[r] for r in top_rows[i].tolist()]
                      for i in range(b0, b1)]
            cos_b = [top_cos[i].tolist() for i in range(b0, b1)]
            Xb, _ = featurize_chunk(s1_recs_b, rows_b, cos_b,
                                   pool_n, pool_a, pool_c, n_jobs=1)
            prob = clf.predict_proba(Xb)[:, 1]
            # per-S1: collect matches above threshold
            k = 0
            for pos in range(b1 - b0):
                sid = s1_ids[b0 + pos]
                matches = []
                for j, r in enumerate(rows_b[pos]):
                    if prob[k] >= thr:
                        matches.append(comp_pids[r])
                    k += 1
                if matches:
                    results[sid] = matches
            del Xb, prob
            gc.collect()
            if b0 % 100000 == 0:
                log(f"[{shard}] scored {b1:,}/{n_s1:,}")
        del pool_n, pool_a, pool_c
        gc.collect()
        log(f"[{shard}] done, {len(results):,} with matches so far")

    # write matching_results.tsv (all test S1, empty string for singletons)
    log("writing output...")
    # need all test S1 IDs in order
    all_s1 = []
    p1 = os.path.join(NORM, "test_source1.tsv")
    for chunk in read_chunks(p1, usecols=["entity_id"]):
        all_s1.extend(chunk["entity_id"].tolist())
    log(f"total test S1: {len(all_s1):,}")
    with open(os.path.join(OUT, "matching_results.tsv"), "w") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in all_s1:
            m = results.get(sid, [])
            f.write(f"{sid}\t{','.join(m)}\n")
    log(f"wrote {os.path.join(OUT, 'matching_results.tsv')}")
    log(f"singletons: {len(all_s1) - len(results):,}")


if __name__ == "__main__":
    main()
