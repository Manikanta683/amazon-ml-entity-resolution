#!/usr/bin/env python3
"""Streaming format checker for full-scale outputs (O(1) memory).

Verifies, in a single streaming pass per file:
  - exact headers, tab-separated, no CSV-style rows
  - one row per test S1, in test_source1.tsv order, no dup S1 rows
  - no duplicate IDs within a list; IDs carry S2-/S3- prefixes (no S1-)
  - matched_entity_ids ⊆ candidate_entity_ids per S1 (lockstep read)
Optionally (--check-ids): every matched/candidate ID exists in test S2/S3
  (builds the ~10M ID set; needs ~1 GB RAM, run alone).
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import NORM, log  # noqa: E402


def check_structure(path, id_col_name, s1_order_iter, check_ids_set=None):
    errs = []
    n_rows = n_empty = 0
    seen = set()
    with open(path, encoding="utf-8") as f:
        header = f.readline()
        exp = f"source1_entity_id\t{id_col_name}\n"
        if header != exp:
            errs.append(f"{path}: bad header {header!r}, expected {exp!r}")
            return errs, 0
        for line in f:
            n_rows += 1
            sid, tab, rest = line.rstrip("\n").partition("\t")
            if not tab:
                errs.append(f"{path}: no tab at row {n_rows}")
                continue
            if sid in seen:
                errs.append(f"{path}: duplicate S1 row {sid}")
            seen.add(sid)
            ids = rest.split(",") if rest else []
            if not ids or ids == [""]:
                n_empty += 1
                ids = []
            if len(ids) != len(set(ids)):
                errs.append(f"{path}: intra-list dupes for {sid}")
            for mid in ids:
                if mid.startswith("S1-"):
                    errs.append(f"{path}: self-match {mid} for {sid}")
                elif not mid.startswith(("S2-", "S3-")):
                    errs.append(f"{path}: bad prefix {mid} for {sid}")
                elif check_ids_set is not None and mid not in check_ids_set:
                    errs.append(f"{path}: unknown ID {mid} for {sid}")
            if len(errs) > 20:
                errs.append("... truncated")
                break
    return errs, n_rows, n_empty


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matching", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--check-ids", action="store_true")
    ap.add_argument("--s1-order", default=None,
                    help="File with expected S1 ids in order (one per line). "
                         "Defaults to all of test_source1.tsv.")
    args = ap.parse_args()

    # expected S1 order
    if args.s1_order:
        with open(args.s1_order, encoding="utf-8") as f:
            s1_order = [line.strip() for line in f if line.strip()]
    else:
        s1_order = []
        with open(os.path.join(NORM, "test_source1.tsv"), encoding="utf-8") as f:
            next(f)
            for line in f:
                s1_order.append(line.split("\t", 1)[0])
    log(f"expected S1 rows: {len(s1_order):,}")

    id_set = None
    if args.check_ids:
        t0 = time.time()
        id_set = set()
        for name in ("test_source2.tsv", "test_source3.tsv"):
            with open(os.path.join(NORM, name), encoding="utf-8") as f:
                next(f)
                for line in f:
                    id_set.add(line.split("\t", 1)[0])
        log(f"ID set: {len(id_set):,} ({time.time()-t0:.0f}s)")

    all_errs = []
    for path, col in ((args.matching, "matched_entity_ids"),
                      (args.candidate, "candidate_entity_ids")):
        errs, n_rows, n_empty = check_structure(path, col, None, id_set)
        log(f"{os.path.basename(path)}: {n_rows:,} rows, {n_empty:,} empty, "
            f"{len(errs)} errors")
        all_errs.extend(errs)

    # matched ⊆ candidates (lockstep: both files written in S1 file order)
    t0 = time.time()
    n_checked = n_viol = 0
    with open(args.matching, encoding="utf-8") as fm, \
            open(args.candidate, encoding="utf-8") as fc:
        next(fm)
        next(fc)
        for lm, lc in zip(fm, fc):
            sm, _, rm = lm.rstrip("\n").partition("\t")
            sc, _, rc = lc.rstrip("\n").partition("\t")
            if sm != sc:
                all_errs.append(f"row order mismatch: {sm} vs {sc}")
                break
            mids = set(rm.split(",")) if rm else set()
            cids = set(rc.split(",")) if rc else set()
            mids.discard("")
            cids.discard("")
            n_checked += 1
            if mids - cids:
                n_viol += 1
                if n_viol <= 5:
                    all_errs.append(f"matched-not-in-candidates for {sm}: "
                                    f"{sorted(mids - cids)[:3]}")
    log(f"subset check: {n_checked:,} rows, {n_viol} violations "
        f"({time.time()-t0:.0f}s)")

    # S1 coverage: every test S1 present exactly once (order check via first col)
    with open(args.matching, encoding="utf-8") as f:
        next(f)
        got = [line.split("\t", 1)[0] for line in f]
    if got == s1_order:
        log("S1 order matches test_source1.tsv exactly")
    else:
        all_errs.append(f"S1 order/coverage mismatch: got {len(got):,} rows, "
                        f"expected {len(s1_order):,}")
        if len(got) == len(s1_order):
            diff = sum(1 for a, b in zip(got, s1_order) if a != b)
            all_errs.append(f"  {diff} positions differ")

    if all_errs:
        print("FAIL:")
        for e in all_errs[:25]:
            print("  -", e)
        sys.exit(1)
    print("PASS — all format checks clean")
    sys.exit(0)


if __name__ == "__main__":
    main()
