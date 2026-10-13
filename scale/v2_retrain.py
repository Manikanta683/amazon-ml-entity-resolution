#!/usr/bin/env python3
"""v2: memory-efficient re-featurize + retrain.

Processes one shard at a time. For each shard, loads only the pool IDs
needed by that shard's candidates (not the full 10M pool).
"""
import os
import sys
import pickle
import gc
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import NORM, TMP, ART, DATASET, read_chunks, log
from featurize import featurize_chunk, N_FEATURES

log(f"v2 N_FEATURES={N_FEATURES}")


def load_gt(train_ids, tune_ids):
    gt = {}
    for chunk in read_chunks(os.path.join(DATASET, "train", "train_ground_truth.tsv")):
        for sid, mids in zip(chunk["source1_entity_id"], chunk["matched_entity_ids"]):
            if sid in train_ids or sid in tune_ids:
                gt[sid] = set(m for m in str(mids).split(",") if m and m != "nan")
    return gt


def process_shard(shard, gt, train_ids, tune_ids):
    """Returns (Xtr, ytr, Xte, yte, ent_te, gc_te) for one shard."""
    # Load candidate files
    cand_data = {}
    for tag in ("train", "tune"):
        d = np.load(os.path.join(TMP, f"{tag}_{shard}_cand.npz"), allow_pickle=True)
        s1_ids = [str(x) for x in d["s1_ids"].tolist()]
        pool_ids_arr = np.array([x.decode() if isinstance(x, bytes) else str(x)
                                 for x in d["pool_ids"].tolist()])
        cand_data[tag] = (s1_ids, d["top_rows"], d["top_cos"], pool_ids_arr)
    # Union of rows referenced across train+tune for this shard
    ref_rows = set()
    for tag in ("train", "tune"):
        _, top_rows, _, _ = cand_data[tag]
        ref_rows.update(np.unique(top_rows).tolist())
    ref_rows = sorted(ref_rows)
    log(f"[{shard}] referenced pool rows: {len(ref_rows):,}")
    # Map row -> pool_id for referenced rows only
    # (use the pool_ids_arr from train; tune should be same pool)
    pool_ids_arr = cand_data["train"][3]
    ref_pids = set(pool_ids_arr[r] for r in ref_rows)
    log(f"[{shard}] referenced pool IDs: {len(ref_pids):,}")

    # Stream pool TSV, keep only referenced
    pf = {}
    for src in ("train_source2", "train_source3"):
        for chunk in read_chunks(os.path.join(NORM, f"{src}.tsv")):
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
    log(f"[{shard}] loaded {len(pf):,} pool fields")
    del ref_pids
    gc.collect()

    # S1 fields for this shard's train/tune IDs
    wanted_s1 = train_ids | tune_ids
    s1f = {}
    for chunk in read_chunks(os.path.join(NORM, "train_source1.tsv")):
        m = chunk["entity_id"].isin(wanted_s1)
        sub = chunk[m]
        for eid, nm, ad, cy in zip(sub["entity_id"], sub["name_norm"],
                                   sub["address_norm"], sub["country_norm"]):
            s1f[eid] = (nm, ad, cy)
    log(f"[{shard}] s1 fields: {len(s1f):,}")

    out = {}
    for tag in ("train", "tune"):
        s1_ids, top_rows, top_cos, pool_ids_arr = cand_data[tag]
        # Build compact pool arrays for referenced rows only, remap indices
        row_map = {r: i for i, r in enumerate(ref_rows)}
        pool_n = [pf[pool_ids_arr[r]][0] for r in ref_rows]
        pool_a = [pf[pool_ids_arr[r]][1] for r in ref_rows]
        pool_c = [pf[pool_ids_arr[r]][2] for r in ref_rows]
        # for labels, need pool_id per compact row
        comp_pids = [pool_ids_arr[r] for r in ref_rows]

        # Batch featurization to avoid OOM (cand_pre too big for all S1 at once)
        X_parts, y_parts = [], []
        B = 10000
        n_s1 = len(s1_ids)
        for b0 in range(0, n_s1, B):
            b1 = min(n_s1, b0 + B)
            s1_recs_b = [s1f[sid] for sid in s1_ids[b0:b1]]
            rows_b = [[row_map[r] for r in top_rows[i].tolist()]
                      for i in range(b0, b1)]
            cos_b = [top_cos[i].tolist() for i in range(b0, b1)]
            log(f"[{shard}/{tag}] featurizing batch {b0:,}-{b1:,}...")
            Xb, p_s1_b = featurize_chunk(s1_recs_b, rows_b, cos_b,
                                        pool_n, pool_a, pool_c, n_jobs=1)
            # labels for this batch
            yb = np.zeros(len(Xb), dtype=np.int8)
            k = 0
            for pos in range(b1 - b0):
                sid = s1_ids[b0 + pos]
                g = gt.get(sid, set())
                for r in rows_b[pos]:
                    yb[k] = 1 if comp_pids[r] in g else 0
                    k += 1
            X_parts.append(Xb)
            y_parts.append(yb)
            del Xb, yb
            gc.collect()
        X = np.vstack(X_parts)
        y = np.concatenate(y_parts)
        del X_parts, y_parts
        gc.collect()
        # Reconstruct rows_list for entity grouping (tune only needs it)
        rows_list = [[row_map[r] for r in top_rows[i].tolist()]
                     for i in range(n_s1)]
        log(f"[{shard}/{tag}] X={X.shape}, pos={y.sum():,}")
        out[tag] = (X, y, s1_ids, rows_list, comp_pids)
        del pool_n, pool_a, pool_c
        gc.collect()

    del pf, s1f
    gc.collect()
    return out


def main():
    sp = np.load(os.path.join(ART, "samples.npz"), allow_pickle=True)
    train_ids = set(str(x) for x in sp["train_ids"].tolist())
    tune_ids = set(str(x) for x in sp["tune_ids"].tolist())
    log("loading GT...")
    gt = load_gt(train_ids, tune_ids)
    log(f"GT: {len(gt):,}")

    Xtr_all, ytr_all = [], []
    Xte_all, yte_all, ent_all, gc_all = [], [], [], []
    s1_offset = 0

    for shard in ("india", "us"):
        out = process_shard(shard, gt, train_ids, tune_ids)
        Xtr, ytr, _, _, _ = out["train"]
        Xte, yte, s1_ids, rows_list, pool_ids = out["tune"]
        Xtr_all.append(Xtr)
        ytr_all.append(ytr)
        Xte_all.append(Xte)
        yte_all.append(yte)
        # entity indices for tune
        n_s1 = len(s1_ids)
        ent = np.repeat(np.arange(s1_offset, s1_offset + n_s1),
                        [len(r) for r in rows_list])
        ent_all.append(ent)
        gt_counts = np.array([len(gt.get(sid, set())) for sid in s1_ids], dtype=np.int32)
        gc_all.append(gt_counts)
        s1_offset += n_s1
        del out
        gc.collect()

    Xtr = np.vstack(Xtr_all)
    ytr = np.concatenate(ytr_all)
    del Xtr_all, ytr_all
    gc.collect()
    log(f"train: {Xtr.shape}, pos={ytr.mean():.4f}")

    log("training HGB...")
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                        max_leaf_nodes=63, min_samples_leaf=50,
                                        l2_regularization=1.0, random_state=0)
    clf.fit(Xtr, ytr)
    log("trained")
    del Xtr, ytr
    gc.collect()

    Xte = np.vstack(Xte_all)
    yte = np.concatenate(yte_all)
    ent = np.concatenate(ent_all)
    log(f"tune: {Xte.shape}")
    prob = clf.predict_proba(Xte)[:, 1]
    del Xte
    gc.collect()

    best_t, best_f = 0.5, -1
    for t in np.arange(0.3, 0.8, 0.025):
        pred = prob >= t
        s = 0.0
        n = 0
        for e in np.unique(ent):
            m = ent == e
            yp, yt = pred[m], yte[m]
            tp = int(np.sum(yp & (yt == 1)))
            fp = int(np.sum(yp & (yt == 0)))
            fn = int(np.sum((~yp) & (yt == 1)))
            if tp + fp + fn == 0:
                s += 1.0
            else:
                prec = tp / max(tp + fp, 1)
                rec = tp / max(tp + fn, 1)
                s += 1.25 * prec * rec / max(0.25 * prec + rec, 1e-9)
            n += 1
        f05 = s / n
        if f05 > best_f:
            best_f, best_t = f05, float(t)
    log(f"v2 tune macro F_0.5={best_f:.4f} @ thr={best_t:.3f}")

    os.makedirs(ART + "_v2", exist_ok=True)
    pickle.dump(clf, open(os.path.join(ART + "_v2", "model.pkl"), "wb"))
    open(os.path.join(ART + "_v2", "threshold.txt"), "w").write(f"{best_t:.4f}\n")
    with open(os.path.join(ART + "_v2", "report.txt"), "w") as f:
        f.write(f"v2 tune macro F_0.5={best_f:.4f} @ thr={best_t:.3f}\n")
    log("saved to artifacts_v2/")


if __name__ == "__main__":
    main()
