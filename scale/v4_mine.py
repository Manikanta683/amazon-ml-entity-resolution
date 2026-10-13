#!/usr/bin/env python3
"""v4 transliteration mining: learn Devanagari->Latin token mappings from
train ground-truth pairs.

Finds GT pairs where one side's name is Devanagari-script and the other
side's is Latin-only, aligns tokens positionally (after stripping corporate
suffixes on both sides), and counts dev_token -> latin_token mappings.
Keeps mappings with count>=3 and share>=0.6.

Output: artifacts_v4/fold_dict.pkl = {"tmap": {...}, "dev_suffix": [...],
"stats": {...}}. All data from the provided TSVs only.

Memory: streaming; only small dicts in RAM.
"""
import os
import pickle
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E402
from common import NORM, DATASET, read_chunks, log  # noqa: E402

V4_ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts_v4")

DEV_SUFFIX = {"प्राइवेट", "लिमिटेड", "प्रा", "लि", "एलएलपी", "कंपनी", "कम्पनी"}


def has_dev(s):
    return any("\u0900" <= c <= "\u097f" for c in s)


def is_latin_side(name):
    return not has_dev(name)


def align(dev_name, lat_name):
    """Yield (dev_token, latin_token) for positionally-aligned pairs."""
    dt = [t for t in dev_name.split() if t not in DEV_SUFFIX]
    lt = lat_name.split()
    if not dt or len(dt) != len(lt):
        return
    for d, l in zip(dt, lt):
        if has_dev(d) and l.isascii() and l:
            yield d, l


def main():
    os.makedirs(V4_ART, exist_ok=True)
    s1_path = os.path.join(NORM, "train_source1.tsv")
    gt_path = os.path.join(DATASET, "train", "train_ground_truth.tsv")
    pool_paths = [os.path.join(NORM, "train_source2.tsv"),
                  os.path.join(NORM, "train_source3.tsv")]

    # Pass A: pool ids with Devanagari names (S1 is Latin-only; dev lives
    # in the S2/S3 pool)
    log("pass A: scanning train pool for Devanagari names ...")
    pool_dev = {}
    n_pool = 0
    for path in pool_paths:
        for chunk in read_chunks(path, usecols=["entity_id", "name_norm"]):
            for eid, nm in zip(chunk["entity_id"], chunk["name_norm"]):
                n_pool += 1
                if nm and has_dev(nm):
                    pool_dev[eid] = nm
    log(f"pool rows={n_pool:,} with Devanagari names={len(pool_dev):,}")
    if not pool_dev:
        log("no Devanagari pool names; writing empty fold dict")
        with open(os.path.join(V4_ART, "fold_dict.pkl"), "wb") as f:
            pickle.dump({"tmap": {}, "dev_suffix": sorted(DEV_SUFFIX),
                         "stats": {"n_pool": n_pool}}, f)
        return

    # Pass B: GT rows whose pool dups include dev names -> (dev_name, s1_id)
    log("pass B: scanning GT for dev-pool pairs ...")
    s1_want = set()
    devp_pairs = []
    n_gt = 0
    for chunk in read_chunks(gt_path):
        for sid, mids in zip(chunk["source1_entity_id"],
                             chunk["matched_entity_ids"]):
            n_gt += 1
            for m in str(mids).split(","):
                if m in pool_dev:
                    devp_pairs.append((pool_dev[m], sid))
                    s1_want.add(sid)
    # dedup pairs (same dev name <-> s1 can repeat across GT rows)
    seen = set()
    uniq = []
    for dn, sid in devp_pairs:
        if (dn, sid) not in seen:
            seen.add((dn, sid))
            uniq.append((dn, sid))
    devp_pairs = uniq
    log(f"GT rows={n_gt:,} dev-pool pairs={len(devp_pairs):,} "
        f"s1 ids wanted={len(s1_want):,}")

    # Pass C: S1 names for those ids (expected Latin-only)
    log("pass C: loading S1 names ...")
    s1_names = {}
    for chunk in read_chunks(s1_path, usecols=["entity_id", "name_norm"]):
        for eid, nm in zip(chunk["entity_id"], chunk["name_norm"]):
            if eid in s1_want:
                s1_names[eid] = nm
        if len(s1_names) >= len(s1_want):
            break
    log(f"S1 names loaded={len(s1_names):,}")

    counts = defaultdict(Counter)
    n_aligned = n_skipped_latin = 0
    for dev_nm, sid in devp_pairs:
        lat_nm = s1_names.get(sid, "")
        if not lat_nm or not is_latin_side(lat_nm):
            n_skipped_latin += 1
            continue
        for d, l in align(dev_nm, lat_nm):
            counts[d][l] += 1
            n_aligned += 1

    tmap = {}
    dbg = {}
    for d, c in counts.items():
        tot = sum(c.values())
        top, cnt = c.most_common(1)[0]
        dbg[d] = (top, cnt, tot)
        if cnt >= 3 and cnt / tot >= 0.6:
            tmap[d] = top
    # diagnostic: tokens that look mappable but were cut
    for d, (top, cnt, tot) in sorted(dbg.items(), key=lambda x: -x[1][1])[:40]:
        if d not in tmap and cnt >= 2:
            log(f"  CUT {d} -> {top} (cnt={cnt} tot={tot})")
    log(f"aligned token pairs={n_aligned:,} distinct dev tokens={len(counts):,} "
        f"kept mappings={len(tmap):,} skipped_nonlatin_s1={n_skipped_latin}")
    for d in list(tmap)[:20]:
        log(f"  {d} -> {tmap[d]}")

    out = {"tmap": tmap, "dev_suffix": sorted(DEV_SUFFIX),
           "stats": {"n_pool": n_pool, "n_pool_dev": len(pool_dev),
                     "n_gt": n_gt, "n_pairs": len(devp_pairs),
                     "n_aligned": n_aligned,
                     "n_distinct_dev": len(counts)}}
    with open(os.path.join(V4_ART, "fold_dict.pkl"), "wb") as f:
        pickle.dump(out, f)
    log("wrote artifacts_v4/fold_dict.pkl")


if __name__ == "__main__":
    main()
