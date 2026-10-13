#!/usr/bin/env python3
"""Sample ground-truth pairs and show raw names/addresses to see noise patterns."""
import os
import random
import numpy as np
from common import NORM, TMP, ART, DATASET, read_chunks, log


def main():
    sp = np.load(os.path.join(ART, "samples.npz"), allow_pickle=True)
    tune_ids = set(str(x) for x in sp["tune_ids"].tolist())

    gt = {}
    for chunk in read_chunks(os.path.join(DATASET, "train", "train_ground_truth.tsv")):
        for sid, mids in zip(chunk["source1_entity_id"], chunk["matched_entity_ids"]):
            if sid in tune_ids:
                gt[sid] = [m for m in str(mids).split(",") if m and m != "nan"]

    fields = {}
    for src in ("train_source1", "train_source2", "train_source3"):
        for chunk in read_chunks(os.path.join(NORM, f"{src}.tsv"),
                                 usecols=["entity_id", "name_norm", "address_norm"]):
            for eid, nm, ad in zip(chunk["entity_id"], chunk["name_norm"],
                                    chunk["address_norm"]):
                fields[eid] = (nm, ad)
    log(f"fields: {len(fields):,}")

    # sample pairs, focus on india (weaker shard)
    pairs = [(s, c) for s, cs in gt.items() for c in cs]
    random.seed(0)
    random.shuffle(pairs)
    print("\n=== sample true pairs ===")
    for s, c in pairs[:12]:
        sn, sa = fields.get(s, ("?", "?"))
        cn, ca = fields.get(c, ("?", "?"))
        print(f"{s} -> {c}")
        print(f"  S1: {sn} | {sa}")
        print(f"  GT: {cn} | {ca}")


if __name__ == "__main__":
    main()
