#!/usr/bin/env python3
"""Minimal memory-efficient v2 scorer. Processes 50k S1 chunks, writes incrementally."""
import os
import sys
import pickle
import gc
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import NORM, TMP, ART, read_chunks, log
from featurize import featurize_chunk

V2_ART = ART + "_v2"
CHUNK = 50000


def score_shard(shard, clf, thr, out_f):
    d = np.load(os.path.join(TMP, f"test_{shard}_cand.npz"), allow_pickle=True)
    s1_ids = [str(x) for x in d["s1_ids"].tolist()]
    top_rows = d["top_rows"]
    top_cos = d["top_cos"]
    pool_ids_arr = np.array([x.decode() if isinstance(x, bytes) else str(x)
                             for x in d["pool_ids"].tolist()])
    n = len(s1_ids)
    log(f"[{shard}] {n:,} S1")

    # S1 fields (all, they're small: 810k max)
    want = set(s1_ids)
    s1f = {}
    for chunk in read_chunks(os.path.join(NORM, "test_source1.tsv")):
        m = chunk["entity_id"].isin(want)
        sub = chunk[m]
        for eid, nm, ad, cy in zip(sub["entity_id"], sub["name_norm"],
                                   sub["address_norm"], sub["country_norm"]):
            s1f[eid] = (nm, ad, cy)
        if len(s1f) >= len(want):
            break
    log(f"[{shard}] s1f: {len(s1f):,}")

    for c0 in range(0, n, CHUNK):
        c1 = min(n, c0 + CHUNK)
        # referenced rows for this chunk
        ref = set()
        for j in range(c0, c1):
            ref.update(int(r) for r in top_rows[j] if r >= 0)
        ref = sorted(ref)
        rmap = {r: i for i, r in enumerate(ref)}
        rpids = [pool_ids_arr[r] for r in ref]
        need = set(rpids)
        # load pool fields for ref only
        pf = {}
        for src in ("test_source2", "test_source3"):
            for chunk in read_chunks(os.path.join(NORM, f"{src}.tsv")):
                for eid, nm, ad, cy in zip(chunk["entity_id"], chunk["name_norm"],
                                           chunk["address_norm"], chunk["country_norm"]):
                    if eid in need:
                        pf[eid] = (nm, ad, cy)
                        if len(pf) >= len(need):
                            break
                if len(pf) >= len(need):
                    break
            if len(pf) >= len(need):
                break
        pn = [pf[pid][0] for pid in rpids]
        pa = [pf[pid][1] for pid in rpids]
        pc = [pf[pid][2] for pid in rpids]
        del pf
        gc.collect()

        # featurize chunk in sub-batches
        for b0 in range(c0, c1, 10000):
            b1 = min(c1, b0 + 10000)
            recs, rl, cl = [], [], []
            for j in range(b0, b1):
                rows = [rmap[int(r)] for r in top_rows[j] if r >= 0]
                if not rows:
                    continue
                nm, ad, cy = s1f[s1_ids[j]]
                recs.append((nm, ad, cy))
                rl.append(rows)
                cl.append([float(top_cos[j, t]) for t in range(len(rows))])
            if not recs:
                # write empty lines for these S1
                for j in range(b0, b1):
                    out_f.write(f"{s1_ids[j]}\t\n")
                continue
            Xb, _ = featurize_chunk(recs, rl, cl, pn, pa, pc, n_jobs=1)
            prob = clf.predict_proba(Xb)[:, 1]
            k = 0
            # map recs index back to s1 position
            ri = 0
            for j in range(b0, b1):
                rows = [rmap[int(r)] for r in top_rows[j] if r >= 0]
                if not rows:
                    out_f.write(f"{s1_ids[j]}\t\n")
                    continue
                # find prob slice for this S1
                # recs[ri] corresponds to s1_ids[j]
                mids = [rpids[rows[t]] for t in range(len(rows))
                        if prob[k + t] >= thr]
                k += len(rows)
                ri += 1
                out_f.write(f"{s1_ids[j]}\t{','.join(mids)}\n")
            del Xb, prob
            gc.collect()
        del pn, pa, pc
        gc.collect()
        log(f"[{shard}] done {c1:,}/{n:,}")
        out_f.flush()


def main():
    clf = pickle.load(open(os.path.join(V2_ART, "model.pkl"), "rb"))
    thr = float(open(os.path.join(V2_ART, "threshold.txt")).read().strip())
    log(f"v2 model, thr={thr}")
    out_path = os.path.expanduser("~/workspace/amazon_ml/output_v2/matching_results.tsv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as out_f:
        out_f.write("source1_entity_id\tmatched_entity_ids\n")
        # need S1 in file order: get all test S1 IDs in order
        # We'll write per-shard but need global order. Instead, collect per shard
        # then merge. Simpler: process shards, store in dict, write at end.
        # But that's memory heavy. Alternative: write per shard to temp, merge.
        for shard in ("france", "india", "us"):
            tmp = out_path + f".{shard}"
            with open(tmp, "w") as tf:
                score_shard(shard, clf, thr, tf)
    log("done scoring shards, merging...")
    # merge in test_source1 order
    order = []
    for chunk in read_chunks(os.path.join(NORM, "test_source1.tsv"),
                             usecols=["entity_id"]):
        order.extend(chunk["entity_id"].tolist())
    md = {}
    for shard in ("france", "india", "us"):
        with open(out_path + f".{shard}") as f:
            for line in f:
                sid, _, mids = line.rstrip("\n").partition("\t")
                md[sid] = mids
    with open(out_path, "w") as out_f:
        out_f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in order:
            out_f.write(f"{sid}\t{md.get(sid, '')}\n")
    log(f"wrote {out_path}")


if __name__ == "__main__":
    main()
