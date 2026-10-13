#!/usr/bin/env python3
"""Sample false-negative tune pairs (true pairs the model scores low)
and print their raw names/addresses to guide feature design."""
import os
import pickle
import numpy as np
from common import NORM, TMP, ART, read_chunks, log


def main():
    model = pickle.load(open(os.path.join(ART, "model.pkl"), "rb"))
    thr = float(open(os.path.join(ART, "threshold.txt")).read().strip())

    # load pool fields for display
    fields = {}
    for src in ("train_source2", "train_source3"):
        for chunk in read_chunks(os.path.join(NORM, f"{src}.tsv"),
                                 usecols=["entity_id", "name_norm", "address_norm"]):
            for eid, nm, ad in zip(chunk["entity_id"], chunk["name_norm"],
                                    chunk["address_norm"]):
                fields[eid] = (nm, ad)
    s1fields = {}
    for chunk in read_chunks(os.path.join(NORM, "train_source1.tsv"),
                             usecols=["entity_id", "name_norm", "address_norm"]):
        for eid, nm, ad in zip(chunk["entity_id"], chunk["name_norm"],
                                chunk["address_norm"]):
            s1fields[eid] = (nm, ad)
    log(f"fields loaded: {len(fields):,} pool, {len(s1fields):,} S1")

    for shard in ("india", "us"):
        d = np.load(os.path.join(TMP, f"feat_tune_{shard}.npz"), allow_pickle=True)
        X, y, ent = d["X"], d["y"], d["ent"]
        s1_ids = d["s1_ids"]
        cand_ids = d["cand_ids"]
        prob = model.predict_proba(X)[:, 1]
        fn_mask = (y == 1) & (prob < thr)
        idx = np.where(fn_mask)[0]
        # pick the lowest-scoring FNs
        order = np.argsort(prob[idx])[:8]
        print(f"\n=== {shard} worst FNs ===")
        for k in order:
            i = idx[k]
            sid = str(s1_ids[ent[i]])
            cid = str(cand_ids[i])
            sn, sa = s1fields.get(sid, ("?", "?"))
            cn, ca = fields.get(cid, ("?", "?"))
            print(f"prob={prob[i]:.3f} {sid} -> {cid}")
            print(f"  S1: {sn} | {sa}")
            print(f"  GT: {cn} | {ca}")


if __name__ == "__main__":
    main()
