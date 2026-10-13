"""Convert gpu_candidates.tsv -> v4 npz format for v4_test.py.

Usage:
  python3 tsv_to_npz.py --input gpu_candidates.tsv --output test_gpu_cand100.npz

Builds pool_ids from unique pool entity IDs in the TSV.
"""
import argparse
import numpy as np
import os
import sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--fill-to", type=int, default=100)
    args = ap.parse_args()

    print("reading TSV...", flush=True)
    s1_ids = []
    cand_lists = []  # list of [(pool_id, cos), ...]
    pool_id_set = set()
    with open(args.input) as f:
        header = f.readline()
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 2:
                continue
            s1_ids.append(p[0])
            cands = []
            for item in p[1].split(","):
                if ":" not in item:
                    continue
                pid, cos = item.rsplit(":", 1)
                try:
                    c = float(cos)
                except:
                    continue
                cands.append((pid, c))
                pool_id_set.add(pid)
            cand_lists.append(cands)
    n = len(s1_ids)
    print(f"S1: {n:,}, unique pool IDs: {len(pool_id_set):,}", flush=True)

    print("building pool index...", flush=True)
    pool_ids = sorted(pool_id_set)  # deterministic order
    pmap = {pid: i for i, pid in enumerate(pool_ids)}

    print("building arrays...", flush=True)
    F = args.fill_to
    top_rows = np.full((n, F), -1, dtype=np.int32)
    top_cos = np.zeros((n, F), dtype=np.float32)
    for i, cands in enumerate(cand_lists):
        # sort by cosine desc, take top F
        cands.sort(key=lambda x: -x[1])
        k = min(len(cands), F)
        for t in range(k):
            pid, c = cands[t]
            top_rows[i, t] = pmap[pid]
            top_cos[i, t] = c
        if (i + 1) % 200_000 == 0:
            print(f"  {i+1:,}/{n:,}", flush=True)

    print(f"writing {args.output}...", flush=True)
    np.savez(args.output,
             s1_ids=np.array(s1_ids),
             top_rows=top_rows,
             top_cos=top_cos,
             pool_ids=np.array(pool_ids))
    print("done", flush=True)

if __name__ == "__main__":
    main()
