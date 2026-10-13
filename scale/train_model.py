#!/usr/bin/env python3
"""Train the stage-2 matcher on a train-S1 sample.

1. Stratified sample: 150k train S1 (model fit) + 50k separate (threshold tune).
2. Block each sample against the TRAIN S2+S3 pool (per shard), stage-1 top-15.
3. Featurize pairs, label from ground truth.
4. Train LogisticRegression and HistGradientBoostingClassifier; pick the winner
   by tuned validation macro F_0.5 on the 50k sample.
5. Save artifacts: model, threshold, report.
"""
import gc
import os
import sys
import time
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    NORM, ART, TMP, DATASET, SEED, macro_f05_from_counts,
    pick_threshold_midplateau, read_chunks, log,
)
from block import discover_shards, ShardIndex  # noqa: E402
from featurize import featurize_chunk  # noqa: E402

N_TRAIN = 150_000
N_TUNE = 50_000
STAGE1_TOP = 15
QCHUNK = 1000
FCHUNK = 10_000


def sample_s1():
    sp = os.path.join(ART, "samples.npz")
    if os.path.exists(sp):
        d = np.load(sp, allow_pickle=True)
        tr = set(d["train_ids"].tolist())
        tu = set(d["tune_ids"].tolist())
        log(f"loaded existing samples: train={len(tr):,} tune={len(tu):,}")
        return tr, tu
    rng = np.random.default_rng(SEED)
    ids, ctry = [], []
    for chunk in read_chunks(os.path.join(NORM, "train_source1.tsv"),
                             usecols=["entity_id", "country_norm"]):
        ids.append(chunk["entity_id"].values)
        ctry.append(chunk["country_norm"].values)
    ids = np.concatenate(ids)
    ctry = np.concatenate(ctry)
    log(f"train S1 loaded: {len(ids):,}")
    order = rng.permutation(len(ids))
    train_idx, tune_idx = [], []
    for c in np.unique(ctry):
        idx = order[ctry[order] == c]
        n_tr = int(round(N_TRAIN * (idx.size / len(ids))))
        n_tu = int(round(N_TUNE * (idx.size / len(ids))))
        train_idx.append(idx[:n_tr])
        tune_idx.append(idx[n_tr:n_tr + n_tu])
    train_idx = np.concatenate(train_idx)
    tune_idx = np.concatenate(tune_idx)
    log(f"sampled train={len(train_idx):,} tune={len(tune_idx):,}")
    np.savez(sp, train_ids=ids[train_idx], tune_ids=ids[tune_idx])
    return set(ids[train_idx].tolist()), set(ids[tune_idx].tolist())


def load_gt_for_ids(id_set):
    gt_path = os.path.join(DATASET, "train", "train_ground_truth.tsv")
    gt = {}
    for chunk in pd.read_csv(gt_path, sep="\t", dtype=str, na_filter=False,
                             keep_default_na=False, engine="c",
                             chunksize=200_000,
                             usecols=["source1_entity_id", "matched_entity_ids"]):
        m = chunk["source1_entity_id"].isin(id_set)
        sub = chunk[m]
        for s1, mids in zip(sub["source1_entity_id"], sub["matched_entity_ids"]):
            gt[s1] = set(mids.split(",")) if mids else set()
    log(f"GT loaded for {len(gt):,} sampled S1")
    return gt


def shard_s1_rows(s1_ids_set, shard, shard_map):
    """List of (id, name, addr, ctry, text) for sampled S1s of a shard, file order."""
    recs = []
    for chunk in read_chunks(os.path.join(NORM, "train_source1.tsv")):
        m = chunk["entity_id"].isin(s1_ids_set)
        if not m.any():
            continue
        sub = chunk[m]
        keep = sub["country_norm"].map(lambda c: shard_map.get(c, "other")) == shard
        sub = sub[keep]
        recs.extend(zip(sub["entity_id"], sub["name_norm"], sub["address_norm"],
                        sub["country_norm"], sub["text"]))
    return recs


def block_phase(shard, shard_map, shard_counts, s1_rows, gt, tag):
    """Block sample S1s -> TMP/{tag}_{shard}_cand.npz; return recall stats."""
    out_path = os.path.join(TMP, f"{tag}_{shard}_cand.npz")
    if os.path.exists(out_path):
        log(f"[{shard}/{tag}] cached {out_path}")
        d = np.load(out_path, allow_pickle=True)
        return {"cached": True, "n": len(d["s1_ids"])}
    t0 = time.time()
    pool_paths = [os.path.join(NORM, "train_source2.tsv"),
                  os.path.join(NORM, "train_source3.tsv")]
    s1_names = {r[1] for r in s1_rows if r[1]}
    index = ShardIndex(pool_paths, shard, shard_map, s1_names,
                       n_docs=shard_counts.get(shard))
    id2row = {e.decode(): r for r, e in enumerate(index.ids)}

    n = len(s1_rows)
    top_rows = np.full((n, STAGE1_TOP), -1, dtype=np.int32)
    top_cos = np.zeros((n, STAGE1_TOP), dtype=np.float32)
    s1_ids = [r[0] for r in s1_rows]

    rec_cand = rec_top = 0.0
    n_gt_ent = 0
    cand_hist = []
    for a in range(0, n, QCHUNK):
        b = min(n, a + QCHUNK)
        texts = [r[4] for r in s1_rows[a:b]]
        names = [r[1] for r in s1_rows[a:b]]
        cand_ids, cand_cos = index.query(texts, names)
        for j in range(b - a):
            s1id = s1_ids[a + j]
            cids, ccos = cand_ids[j], cand_cos[j]
            cand_hist.append(len(cids))
            g = gt.get(s1id, set())
            if g:
                n_gt_ent += 1
                s = set(cids)
                rec_cand += len(g & s) / len(g)
                k = min(STAGE1_TOP, len(cids))
                rec_top += len(g & set(cids[:k])) / len(g)
            k = min(STAGE1_TOP, len(cids))
            for t in range(k):
                top_rows[a + j, t] = id2row[cids[t]]
                top_cos[a + j, t] = ccos[t]
        if (a // QCHUNK) % 10 == 0:
            log(f"[{shard}/{tag}] blocked {b:,}/{n:,}")
    np.savez_compressed(out_path, s1_ids=np.array(s1_ids),
                        top_rows=top_rows, top_cos=top_cos, pool_ids=index.ids)
    stats = {
        "n": n,
        "blocking_recall": rec_cand / max(1, n_gt_ent),
        "stage1_recall": rec_top / max(1, n_gt_ent),
        "n_gt_ent": n_gt_ent,
        "mean_cands": float(np.mean(cand_hist)) if cand_hist else 0.0,
        "secs": time.time() - t0,
    }
    log(f"[{shard}/{tag}] recall_block={stats['blocking_recall']:.4f} "
        f"recall_stage1={stats['stage1_recall']:.4f} "
        f"mean_cands={stats['mean_cands']:.1f} ({stats['secs']:.0f}s)")
    del index, id2row
    gc.collect()
    return stats


def load_pool_lists(shard_paths, shard=None, shard_map=None):
    """Load pool name/address/country as Python lists (dedup strings).

    When shard is given, keep only rows whose country maps to that shard,
    in file order -- matching ShardIndex's row layout exactly.
    """
    names, addrs, ctrys = [], [], []
    seen = {}
    for path in shard_paths:
        for chunk in read_chunks(path, usecols=["name_norm", "address_norm",
                                                "country_norm"]):
            if shard is not None:
                m = chunk["country_norm"].map(
                    lambda c: shard_map.get(c, "other")) == shard
                chunk = chunk[m]
                if len(chunk) == 0:
                    continue
            for nm, ad, ct in zip(chunk["name_norm"], chunk["address_norm"],
                                   chunk["country_norm"]):
                # local dedup: chains share exact names/addresses
                if nm in seen:
                    nm = seen[nm]
                else:
                    seen[nm] = nm
                if ad in seen:
                    ad = seen[ad]
                else:
                    seen[ad] = ad
                if ct in seen:
                    ct = seen[ct]
                else:
                    seen[ct] = ct
                names.append(nm)
                addrs.append(ad)
                ctrys.append(ct)
    del seen
    gc.collect()
    log(f"pool lists loaded: {len(names):,} rows")
    return names, addrs, ctrys


def featurize_phase(shard, tags, gt, shard_map):
    """Featurize top-15 pairs for npz files of given tags. Returns dict tag ->
    (X, y, ent_idx_local, gt_counts_local, s1_ids)."""
    npz_paths = {t: os.path.join(TMP, f"{t}_{shard}_cand.npz") for t in tags}
    pool_paths = [os.path.join(NORM, "train_source2.tsv"),
                  os.path.join(NORM, "train_source3.tsv")]
    pool_names, pool_addrs, pool_ctry = load_pool_lists(
        pool_paths, shard=shard, shard_map=shard_map)
    out = {}
    for t in tags:
        d = np.load(npz_paths[t], allow_pickle=True)
        s1_ids = [str(x) for x in d["s1_ids"].tolist()]
        top_rows = d["top_rows"]
        top_cos = d["top_cos"]
        pool_ids = d["pool_ids"]
        n = len(s1_ids)
        # S1 fields in the same order: re-read from norm TSV via id set
        want = set(s1_ids)
        fields = {}
        for chunk in read_chunks(os.path.join(NORM, "train_source1.tsv")):
            m = chunk["entity_id"].isin(want)
            sub = chunk[m]
            for eid, nm, ad, ct in zip(sub["entity_id"], sub["name_norm"],
                                       sub["address_norm"], sub["country_norm"]):
                fields[eid] = (nm, ad, ct)
            if len(fields) >= len(want):
                break
        Xs, ys, es = [], [], []
        gt_counts = np.zeros(n, dtype=np.int32)
        for a in range(0, n, FCHUNK):
            b = min(n, a + FCHUNK)
            s1_recs, rows_list, cos_list, pos_list = [], [], [], []
            for j in range(a, b):
                s1id = s1_ids[j]
                gt_counts[j] = len(gt.get(s1id, set()))
                rows = [int(r) for r in top_rows[j] if r >= 0]
                if not rows:
                    continue
                nm, ad, ct = fields[s1id]
                s1_recs.append((nm, ad, ct))
                rows_list.append(rows)
                cos_list.append([float(top_cos[j, t2])
                                 for t2 in range(len(rows))])
                pos_list.append(j)
            if not s1_recs:
                continue
            Xc, p_s1 = featurize_chunk(s1_recs, rows_list, cos_list,
                                       pool_names, pool_addrs, pool_ctry)
            # labels
            yc = np.zeros(len(Xc), dtype=np.int8)
            k = 0
            for pi, (rows, pos) in enumerate(zip(rows_list, pos_list)):
                s1id = s1_ids[pos]
                g = gt.get(s1id, set())
                for r in rows:
                    if pool_ids[r].decode() in g:
                        yc[k] = 1
                    k += 1
            Xs.append(Xc)
            ys.append(yc)
            es.append(np.array(pos_list, dtype=np.int32)[p_s1])
            log(f"[{shard}/{t}] featurized {b:,}/{n:,}")
        X = np.concatenate(Xs) if Xs else np.zeros((0, 16), np.float32)
        y = np.concatenate(ys) if ys else np.zeros(0, np.int8)
        ent = np.concatenate(es) if es else np.zeros(0, np.int32)
        out[t] = (X, y, ent, gt_counts, s1_ids)
        log(f"[{shard}/{t}] pairs={len(X):,} pos_rate={y.mean():.3f}")
        gc.collect()
    del pool_names, pool_addrs, pool_ctry
    gc.collect()
    return out


def tune_threshold(y_true, y_prob, ent_idx, gt_counts, n_ent):
    thresholds = np.linspace(0.02, 0.98, 120)
    y_true = np.asarray(y_true, dtype=np.int8)
    scores = []
    for t in thresholds:
        pred = y_prob >= t
        tp = np.bincount(ent_idx[pred & (y_true == 1)], minlength=n_ent)
        fp = np.bincount(ent_idx[pred & (y_true == 0)], minlength=n_ent)
        fn = gt_counts - tp
        scores.append(macro_f05_from_counts(tp, fp, fn))
    best_t, best_f = pick_threshold_midplateau(thresholds, np.array(scores))
    return best_t, best_f, thresholds, scores


def main():
    t_all = time.time()
    train_set, tune_set = sample_s1()
    gt = load_gt_for_ids(train_set | tune_set)

    pool_paths = [os.path.join(NORM, "train_source2.tsv"),
                  os.path.join(NORM, "train_source3.tsv")]
    shard_map, shard_counts = discover_shards(pool_paths)
    # shards present in the sampled S1s
    with open(os.path.join(ART, "shard_map.txt"), "w") as f:
        for k, v in sorted(shard_map.items()):
            f.write(f"{k}\t{v}\n")

    # figure out which shards the samples touch (via country of sampled ids)
    id2shard = {}
    for chunk in read_chunks(os.path.join(NORM, "train_source1.tsv"),
                             usecols=["entity_id", "country_norm"]):
        for eid, c in zip(chunk["entity_id"], chunk["country_norm"]):
            if eid in train_set or eid in tune_set:
                id2shard[eid] = shard_map.get(c, "other")
    shards = sorted(set(id2shard.values()))
    log(f"sample shards: {shards}")

    report = {}
    feat = {}
    for shard in shards:
        sstats = {}
        for tag, sset in (("train", train_set), ("tune", tune_set)):
            wanted = {e for e, s in id2shard.items()
                      if s == shard and e in sset}
            rows = shard_s1_rows(wanted, shard, shard_map)
            log(f"[{shard}/{tag}] S1 rows: {len(rows):,}")
            sstats[tag] = block_phase(shard, shard_map, shard_counts, rows, gt, tag)
        report[shard] = sstats
        fz = featurize_phase(shard, ["train", "tune"], gt, shard_map)
        for tag in ("train", "tune"):
            X, y, ent, gc_, ids_ = fz[tag]
            # rebase entity indices to global per-tag numbering later; store raw
            np.savez_compressed(os.path.join(TMP, f"feat_{tag}_{shard}.npz"),
                                X=X, y=y, ent=ent, gt_counts=gc_,
                                s1_ids=np.array(ids_))
        feat[shard] = True
        gc.collect()

    # assemble train / tune into disk-backed memmaps (X is GBs; RAM is 7 GB)
    def assemble(tag):
        total, n_ent = 0, 0
        for shard in shards:
            with np.load(os.path.join(TMP, f"feat_{tag}_{shard}.npz"),
                         allow_pickle=True) as d:
                total += len(d["X"])
                n_ent += len(d["gt_counts"])
        log(f"[{tag}] pairs={total:,} entities={n_ent:,}")
        Xmm = np.memmap(os.path.join(TMP, f"X_{tag}.dat"), dtype=np.float32,
                        mode="w+", shape=(total, 16))
        y = np.empty(total, dtype=np.int8)
        ent = np.empty(total, dtype=np.int32)
        gt_counts = np.empty(n_ent, dtype=np.int32)
        o = e = 0
        for shard in shards:
            with np.load(os.path.join(TMP, f"feat_{tag}_{shard}.npz"),
                         allow_pickle=True) as d:
                Xd, yd, ed, gd = d["X"], d["y"], d["ent"], d["gt_counts"]
                n, ne = len(Xd), len(gd)
                Xmm[o:o + n] = Xd
                y[o:o + n] = yd
                ent[o:o + n] = ed + e
                gt_counts[e:e + ne] = gd
                o += n
                e += ne
            del Xd, yd, ed, gd
            gc.collect()
        Xmm.flush()
        return Xmm, y, ent, gt_counts, n_ent

    Xtr, ytr, etr, gtr, ntr = assemble("train")
    Xtu, ytu, etu, gtu, ntu = assemble("tune")
    log(f"train pairs={len(Xtr):,} pos={ytr.mean():.3f} | "
        f"tune pairs={len(Xtu):,}")

    results = {}
    for name, clf in [
        ("logreg", LogisticRegression(class_weight="balanced", max_iter=500)),
        ("hgb", HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06)),
    ]:
        t0 = time.time()
        clf.fit(Xtr, ytr)
        prob = clf.predict_proba(Xtu)[:, 1]
        bt, bf, _, _ = tune_threshold(ytu, prob, etu, gtu, ntu)
        results[name] = (clf, bt, bf)
        log(f"{name}: tune_F05={bf:.4f} thr={bt:.3f} ({time.time()-t0:.0f}s)")

    winner = max(results, key=lambda k: results[k][2])
    clf, best_t, best_f = results[winner]
    log(f"WINNER: {winner} F05={best_f:.4f} thr={best_t:.3f}")

    with open(os.path.join(ART, "model.pkl"), "wb") as f:
        pickle.dump(clf, f)
    with open(os.path.join(ART, "threshold.txt"), "w") as f:
        f.write(f"{best_t}\n")
    with open(os.path.join(ART, "report.txt"), "w") as f:
        f.write(f"model={winner}\nval_F05={best_f:.4f}\nthreshold={best_t:.4f}\n")
        for shard, sstats in report.items():
            for tag, st in sstats.items():
                f.write(f"{shard}/{tag}: {st}\n")
    log(f"ALL DONE in {(time.time() - t_all) / 60:.1f} min")


if __name__ == "__main__":
    main()
