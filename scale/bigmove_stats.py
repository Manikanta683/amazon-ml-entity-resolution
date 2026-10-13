#!/usr/bin/env python3
"""Big-move stats: duplicate distribution + exact-match rates on train GT."""
import csv
import random
from collections import Counter

random.seed(0)
D = "/home/hatch/workspace/amazon_ml/student_resource/student_resource/dataset/train/train_ground_truth.tsv"
NORM = "/home/hatch/workspace/amazon_ml/scale/norm/"

pairs = []
cnt = Counter()
n = 0
with open(D) as f:
    r = csv.DictReader(f, delimiter="\t")
    for row in r:
        n += 1
        mids = row["matched_entity_ids"].split(",") if row["matched_entity_ids"] else []
        cnt[len(mids)] += 1
        if random.random() < 0.03:
            for m in mids:
                pairs.append((row["source1_entity_id"], m))
print(f"S1 total={n:,}  sampled_pairs={len(pairs):,}", flush=True)
tot = sum(k * v for k, v in cnt.items())
print(f"mean_dupes_per_S1={tot / n:.2f}", flush=True)
for k in sorted(cnt)[:8]:
    print(f"  dupes={k}: {cnt[k]:,} ({cnt[k] / n * 100:.1f}%)", flush=True)
print(f"  dupes>=8: {sum(v for k, v in cnt.items() if k >= 8):,}", flush=True)

want = set()
for a, b in pairs:
    want.add(a)
    want.add(b)
print(f"unique entities to lookup: {len(want):,}", flush=True)
fld = {}
for src in ("train_source1", "train_source2", "train_source3"):
    with open(NORM + src + ".tsv") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            e = row["entity_id"]
            if e in want:
                fld[e] = (row["name_norm"], row["address_norm"])
                if len(fld) >= len(want):
                    break
        if len(fld) >= len(want):
            break
print(f"found: {len(fld):,}", flush=True)

nm = ad = both = same_addr_diff_name = tot2 = 0
for a, b in pairs:
    if a in fld and b in fld:
        tot2 += 1
        sn, sa = fld[a]
        tn, ta = fld[b]
        en = (sn == tn)
        ea = (sa == ta)
        nm += en
        ad += ea
        both += (en and ea)
        if ea and not en and sa:
            same_addr_diff_name += 1
print(f"pairs evaluated: {tot2:,}", flush=True)
print(f"name exact: {nm / tot2 * 100:.1f}%  addr exact: {ad / tot2 * 100:.1f}%  "
      f"both: {both / tot2 * 100:.1f}%", flush=True)
print(f"same-addr-diff-name (nonempty addr): {same_addr_diff_name / tot2 * 100:.1f}%", flush=True)
print("DONE", flush=True)
