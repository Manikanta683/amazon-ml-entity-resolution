#!/usr/bin/env python3
"""Normalize all raw TSVs once -> scale/norm/*.tsv (chunked, low memory)."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    DATASET, NORM, NORM_COLS, TRAIN_FILES, TEST_FILES,
    normalize_chunk, read_chunks, log,
)

JOBS = [
    ("train_source1", os.path.join(DATASET, TRAIN_FILES["s1"])),
    ("train_source2", os.path.join(DATASET, TRAIN_FILES["s2"])),
    ("train_source3", os.path.join(DATASET, TRAIN_FILES["s3"])),
    ("test_source1", os.path.join(DATASET, TEST_FILES["s1"])),
    ("test_source2", os.path.join(DATASET, TEST_FILES["s2"])),
    ("test_source3", os.path.join(DATASET, TEST_FILES["s3"])),
]


def main():
    t0 = time.time()
    for name, path in JOBS:
        out_path = os.path.join(NORM, name + ".tsv")
        if os.path.exists(out_path):
            log(f"skip {name} (exists)")
            continue
        log(f"normalizing {name} ...")
        n = 0
        first = True
        for chunk in read_chunks(path):
            norm = normalize_chunk(chunk)
            norm.to_csv(out_path, sep="\t", index=False, header=first,
                        mode="w" if first else "a")
            first = False
            n += len(norm)
            if n % 1_000_000 < 200_000:
                log(f"  {name}: {n:,} rows")
        log(f"done {name}: {n:,} rows -> {out_path}")
    log(f"preprocess finished in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
