#!/usr/bin/env python3
"""v4 training: 33 features + HGB with hard-positive upweighting.

The binding constraint is HARD POSITIVES: the classifier collapses on
low-cosine true dups (recall 0.97 @ cos~1 -> 0.26 @ cos~0.35) because v3
trained on top-15 (easy) candidates but infers on a wider, harder
distribution. v4 trains on the INFERENCE distribution (two-stage top-100 +
transliteration/v3norm legs) and upweights low-cosine positives:
    w(y=1) = 1 + pos_k * (1 - cos_word)

Modes:
  --sample --proto-dir logs/rd_990
      quick validation reusing step8's proto cand100 npz (1400 train + 600
      tune S1s, india+us): augments with L2/L3 legs, trains v4, evaluates on
      the tune split with the correct metric, and trains an identical-sample
      v3-style baseline (top-15, 30 feats, uniform) for the delta.
      Writes logs/v4/VALIDATION_sample.txt.
  --full --n-train 60000 --n-tune 12000
      builds v4 train blocking per shard (HEAVY: dual index; refuses while
      v3_block_full.py is alive), featurizes, trains, tunes threshold,
      saves artifacts_v4/.

All features come from the provided TSVs only.
"""
import os
import sys
import gc
import pickle
import argparse
from collections import defaultdict

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E402
from common import NORM, TMP, ART, DATASET, read_chunks, log, \
    macro_f05_from_counts  # noqa: E402
from v3_train import load_gt, load_pool_fields, load_s1_fields, \
    fit_name_addr_vecs, cos_for_pairs, global_threshold_tune, \
    per_entity_counts, shard_data  # noqa: E402
from v3_featurize import featurize_chunk_v3  # noqa: E402 (baseline)
from v4_featurize import featurize_chunk_v4, N_FEATURES  # noqa: E402
from v4_block_aug import load_fold_dict, fold_name, build_name_maps, \
    build_train_block  # noqa: E402
from v3_featurize import v3_norm  # noqa: E402
from block import discover_shards  # noqa: E402

V4_ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts_v4")
V4_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "v4")
V4_TRAIN_FILL = 128


def featurize_npz(npz_path, s1_ids, gt, tmap, dev_suffix, n_feats=33,
                  batch=5000, leg_rows_extra=None):
    """Featurize (s1_ids x candidates) from a v4/proto cand npz.

    leg_rows_extra: optional {s1_pos: [extra pool rows]} appended per S1
    (used in --sample to add L2/L3 legs onto step8's proto npz).
    Returns (X, y, ent, gt_counts, mean_cands).
    """
    d = np.load(npz_path, allow_pickle=True)
    s1_all = [str(x) for x in d["s1_ids"].tolist()]
    top_rows, top_cos = d["top_rows"], d["top_cos"]
    pool_ids = np.array([x.decode() if isinstance(x, bytes) else str(x)
                         for x in d["pool_ids"].tolist()])
    pos = {sid: i for i, sid in enumerate(s1_all)}
    idx = [pos[sid] for sid in s1_ids]
    W = top_rows.shape[1]

    # empty candidates -> return empty (skip TF-IDF)
    if (top_rows[idx] < 0).all():
        log(f"featurize {os.path.basename(npz_path)}: no candidates, skipping")
        return (np.zeros((0, n_feats), dtype=np.float32), np.zeros(0, dtype=np.int8),
                np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), None)

    # per-S1 row lists (with optional leg extras appended)
    s1_rows, s1_cos = [], []
    for j, gi in enumerate(idx):
        rows = [int(r) for r in top_rows[gi] if int(r) >= 0]
        cos = [float(top_cos[gi, t]) for t in range(len(rows))]
        if leg_rows_extra and j in leg_rows_extra:
            seen = set(rows)
            for r in leg_rows_extra[j]:
                if r not in seen:
                    seen.add(r)
                    rows.append(r)
                    cos.append(0.0)
        s1_rows.append(rows)
        s1_cos.append(cos)

    ref_rows = sorted(set(r for rows in s1_rows for r in rows))
    rmap = {r: i for i, r in enumerate(ref_rows)}
    rpids = [pool_ids[r] for r in ref_rows]
    pf = load_pool_fields(pool_ids, ref_rows)
    s1f = load_s1_fields(set(s1_ids))
    pn = [pf[pid][0] for pid in rpids]
    pa = [pf[pid][1] for pid in rpids]
    pc = [pf[pid][2] for pid in rpids]
    del pf
    gc.collect()

    log(f"featurize {os.path.basename(npz_path)}: fitting name/addr TF-IDF "
        f"on {len(pn):,} pool rows")
    vn, va = fit_name_addr_vecs(pn, pa)
    X_parts, y_parts, ent_parts = [], [], []
    for b0 in range(0, len(s1_ids), batch):
        b1 = min(len(s1_ids), b0 + batch)
        recs = [s1f[s1_ids[j]] for j in range(b0, b1)]
        rl = [[rmap[r] for r in s1_rows[j]] for j in range(b0, b1)]
        cl = [s1_cos[j] for j in range(b0, b1)]
        pairs = [(j - b0, r) for j in range(b0, b1) for r in rl[j - b0]]
        cn = cos_for_pairs(vn, [r[0] for r in recs], pn, pairs)
        ca = cos_for_pairs(va, [r[1] for r in recs], pa, pairs)
        cn_l, ca_l, k = [], [], 0
        for rows in rl:
            cn_l.append(cn[k:k + len(rows)])
            ca_l.append(ca[k:k + len(rows)])
            k += len(rows)
        if n_feats == 33:
            Xb, _ = featurize_chunk_v4(recs, rl, cl, pn, pa, pc, cn_l, ca_l,
                                       n_jobs=1, fold_map=tmap,
                                       dev_suffix=dev_suffix)
        else:
            # v3-style baseline: truncate to first 15 rows, 30 feats
            rl15 = [rows[:15] for rows in rl]
            cl15 = [c[:15] for c in cl]
            cn15 = [c[:15] for c in cn_l]
            ca15 = [c[:15] for c in ca_l]
            Xb, _ = featurize_chunk_v3(recs, rl15, cl15, pn, pa, pc,
                                       cn15, ca15, n_jobs=1)
        yb = np.zeros(len(Xb), dtype=np.int8)
        k = 0
        for j in range(b0, b1):
            g = gt.get(s1_ids[j], set())
            for r in s1_rows[j]:
                yb[k] = 1 if rpids[rmap[r]] in g else 0
                k += 1
        eb = np.repeat(np.arange(b0, b1, dtype=np.int32),
                       [len(r) for r in s1_rows[b0:b1]])
        X_parts.append(Xb)
        y_parts.append(yb)
        ent_parts.append(eb)
        del Xb, yb, eb
        gc.collect()
    d.close()
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    ent = np.concatenate(ent_parts)
    gt_counts = np.array([len(gt.get(s, set())) for s in s1_ids], dtype=np.int32)
    mean_c = np.mean([len(r) for r in s1_rows])
    log(f"X={X.shape} pos={y.sum():,} ({y.mean():.4f}) mean_cands={mean_c:.1f}")
    del X_parts, y_parts, ent_parts, vn, va
    gc.collect()
    return X, y, ent, gt_counts, mean_c


def train_hgb(Xtr, ytr, pos_k):
    sw = None
    if pos_k and pos_k > 0:
        cos_w = np.clip(Xtr[:, 11] / 100.0, 0, 1)  # f11 = word cos * 100
        sw = np.ones(len(ytr), dtype=np.float32)
        pm = ytr == 1
        sw[pm] = 1.0 + pos_k * (1.0 - cos_w[pm])
        log(f"pos weighting k={pos_k}: mean pos w={sw[pm].mean():.2f} "
            f"max={sw[pm].max():.2f}")
    hgb = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=63,
        min_samples_leaf=50, l2_regularization=1.0, random_state=0)
    log(f"training HGB on {Xtr.shape} ...")
    hgb.fit(Xtr, ytr, sample_weight=sw)
    return hgb


def leg_coverage(npz_path, s1_ids, gt, leg_rows_extra):
    """Of GT dups in stage-2 rows (pos>=15), what fraction do legs catch?"""
    d = np.load(npz_path, allow_pickle=True)
    s1_all = [str(x) for x in d["s1_ids"].tolist()]
    top_rows = d["top_rows"]
    pool_ids = np.array([x.decode() if isinstance(x, bytes) else str(x)
                         for x in d["pool_ids"].tolist()])
    pos = {sid: i for i, sid in enumerate(s1_all)}
    tot, caught = 0, 0
    for j, sid in enumerate(s1_ids):
        gi = pos[sid]
        g = gt.get(sid, set())
        if not g:
            continue
        s2 = {int(r) for r in top_rows[gi, 15:] if int(r) >= 0}
        legs = set(leg_rows_extra.get(j, ()))
        for r in s2:
            if pool_ids[r] in g:
                tot += 1
                if r in legs:
                    caught += 1
    d.close()
    return tot, caught


def build_sample_legs(proto_dir, shard, s1_ids, tmap):
    """L2/L3 leg rows for sample S1s on the TRAIN pool (light streaming)."""
    d = np.load(os.path.join(proto_dir, f"proto_{shard}_train_cand100.npz"),
                allow_pickle=True)
    pool_ids = np.array([x.decode() if isinstance(x, bytes) else str(x)
                         for x in d["pool_ids"].tolist()])
    d.close()
    pool_paths = [os.path.join(NORM, "train_source2.tsv"),
                  os.path.join(NORM, "train_source3.tsv")]
    shard_map, _ = discover_shards(pool_paths)
    # S1 folded/v3n names
    want = set(s1_ids)
    s1f = load_s1_fields(want)
    dev_suffix = set()
    try:
        with open(os.path.join(V4_ART, "fold_dict.pkl"), "rb") as f:
            dev_suffix = set(pickle.load(f)["dev_suffix"])
    except FileNotFoundError:
        pass
    s1_leg = {sid: (fold_name(s1f[sid][0], tmap, dev_suffix),
                    v3_norm(s1f[sid][0])) for sid in s1_ids}
    want_folded = {fn for fn, _ in s1_leg.values() if fn}
    want_v3n = {vn for _, vn in s1_leg.values() if vn}
    pids, fold_map, v3n_map = build_name_maps(
        pool_paths, shard, shard_map, tmap, dev_suffix, want_folded, want_v3n)
    assert len(pids) == len(pool_ids), "train pool row order mismatch"
    # map pool row -> same row space (identical order) -> legs per S1 pos
    legs = {}
    for j, sid in enumerate(s1_ids):
        fn, vn = s1_leg[sid]
        extra = []
        if fn:
            extra.extend(fold_map.get(fn, ()))
        if vn:
            extra.extend(v3n_map.get(vn, ()))
        if extra:
            legs[j] = extra
    log(f"[{shard}] sample legs: {len(legs)}/{len(s1_ids)} S1s have extras, "
        f"mean extras={np.mean([len(v) for v in legs.values()]):.1f}"
        if legs else f"[{shard}] sample legs: none")
    return legs


def run_sample(proto_dir, pos_k):
    tmap, dev_suffix = load_fold_dict()
    Xtr_l, ytr_l, Xte_l, yte_l, ent_l, gc_l = [], [], [], [], [], []
    ent_off = 0
    cov_tot, cov_caught = 0, 0
    for shard in ("india", "us"):
        d = np.load(os.path.join(proto_dir, f"proto_{shard}_train_cand100.npz"),
                    allow_pickle=True)
        tr_ids = [str(x) for x in d["s1_ids"].tolist()]
        d.close()
        d = np.load(os.path.join(proto_dir, f"proto_{shard}_tune_cand100.npz"),
                    allow_pickle=True)
        te_ids = [str(x) for x in d["s1_ids"].tolist()]
        d.close()
        log(f"[{shard}] sample S1: train={len(tr_ids)} tune={len(te_ids)}")
        gt = load_gt(set(tr_ids) | set(te_ids))
        legs_tr = build_sample_legs(proto_dir, shard, tr_ids, tmap)
        legs_te = build_sample_legs(proto_dir, shard, te_ids, tmap)
        t2, c2 = leg_coverage(
            os.path.join(proto_dir, f"proto_{shard}_tune_cand100.npz"),
            te_ids, gt, legs_te)
        cov_tot += t2
        cov_caught += c2
        log(f"[{shard}] leg coverage of stage-2 GT dups: {c2}/{t2} "
            f"= {c2 / max(t2, 1):.3f}")
        Xtr, ytr, _, _, _ = featurize_npz(
            os.path.join(proto_dir, f"proto_{shard}_train_cand100.npz"),
            tr_ids, gt, tmap, dev_suffix, n_feats=33,
            leg_rows_extra=legs_tr)
        Xte, yte, ent, gtc, mc = featurize_npz(
            os.path.join(proto_dir, f"proto_{shard}_tune_cand100.npz"),
            te_ids, gt, tmap, dev_suffix, n_feats=33,
            leg_rows_extra=legs_te)
        Xtr_l.append(Xtr); ytr_l.append(ytr)
        Xte_l.append(Xte); yte_l.append(yte)
        ent_l.append(ent + ent_off); gc_l.append(gtc)
        ent_off += len(te_ids)
        del Xtr, ytr, Xte, yte, ent, gtc, gt
        gc.collect()

    Xtr = np.vstack(Xtr_l); ytr = np.concatenate(ytr_l)
    Xte = np.vstack(Xte_l); yte = np.concatenate(yte_l)
    ent = np.concatenate(ent_l); gcounts = np.concatenate(gc_l)
    n_ent = ent_off
    del Xtr_l, ytr_l, Xte_l, yte_l, ent_l, gc_l
    gc.collect()
    log(f"TRAIN {Xtr.shape} pos={ytr.mean():.4f} | TUNE {Xte.shape} "
        f"n_ent={n_ent}")

    hgb = train_hgb(Xtr, ytr, pos_k)
    del Xtr, ytr
    gc.collect()
    prob = hgb.predict_proba(Xte)[:, 1]
    thr, f_v4 = global_threshold_tune(prob, yte, ent, gcounts, n_ent)
    log(f"V4-sample: tune F05={f_v4:.4f} @ thr={thr:.3f}")
    # cosine-binned recall for the report
    cos_te = Xte[:, 11] / 100.0
    edges = [0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    bins = []
    for b in range(len(edges) - 1):
        m = (cos_te >= edges[b]) & (cos_te < edges[b + 1]) & (yte == 1)
        r = (prob[m] >= thr).mean() if m.sum() else float("nan")
        bins.append(f"{edges[b]:.1f}-{edges[b+1]:.1f}:{r:.2f} (n={m.sum()})")
    del prob
    gc.collect()

    # identical-sample v3-style baseline: top-15, 30 feats, uniform HGB
    log("=== identical-sample v3-style baseline (top-15, 30 feats) ===")
    Xb_tr_l, yb_tr_l, Xb_te_l, yb_te_l, entb_l, gcb_l = [], [], [], [], [], []
    off = 0
    for shard in ("india", "us"):
        d = np.load(os.path.join(proto_dir, f"proto_{shard}_train_cand100.npz"),
                    allow_pickle=True)
        tr_ids = [str(x) for x in d["s1_ids"].tolist()]
        d.close()
        d = np.load(os.path.join(proto_dir, f"proto_{shard}_tune_cand100.npz"),
                    allow_pickle=True)
        te_ids = [str(x) for x in d["s1_ids"].tolist()]
        d.close()
        gt = load_gt(set(tr_ids) | set(te_ids))
        Xtr, ytr, _, _, _ = featurize_npz(
            os.path.join(proto_dir, f"proto_{shard}_train_cand100.npz"),
            tr_ids, gt, tmap, dev_suffix, n_feats=30)
        Xte, yte, ent, gtc, _ = featurize_npz(
            os.path.join(proto_dir, f"proto_{shard}_tune_cand100.npz"),
            te_ids, gt, tmap, dev_suffix, n_feats=30)
        Xb_tr_l.append(Xtr); yb_tr_l.append(ytr)
        Xb_te_l.append(Xte); yb_te_l.append(yte)
        entb_l.append(ent + off); gcb_l.append(gtc)
        off += len(te_ids)
        del Xtr, ytr, Xte, yte, ent, gtc, gt
        gc.collect()
    Xb_tr = np.vstack(Xb_tr_l); yb_tr = np.concatenate(yb_tr_l)
    Xb_te = np.vstack(Xb_te_l); yb_te = np.concatenate(yb_te_l)
    entb = np.concatenate(entb_l); gcb = np.concatenate(gcb_l)
    hb = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=63,
        min_samples_leaf=50, l2_regularization=1.0, random_state=0)
    hb.fit(Xb_tr, yb_tr)
    pb = hb.predict_proba(Xb_te)[:, 1]
    thr_b, f_base = global_threshold_tune(pb, yb_te, entb, gcb, off)
    log(f"BASELINE: tune F05={f_base:.4f} @ thr={thr_b:.3f}")
    log(f"DELTA (v4 - baseline) = {f_v4 - f_base:+.4f}")

    os.makedirs(V4_LOG, exist_ok=True)
    with open(os.path.join(V4_LOG, "VALIDATION_sample.txt"), "w") as f:
        f.write("v4 sample validation (step8 proto: 1400 train + 600 tune S1, "
                "india+us, correct metric incl. blocking-miss FNs)\n")
        f.write(f"baseline (v3-style top-15/30f/uniform): F05={f_base:.4f} "
                f"@ thr={thr_b:.3f}\n")
        f.write(f"v4 (top-100+legs/33f/pos_k={pos_k}):      F05={f_v4:.4f} "
                f"@ thr={thr:.3f}\n")
        f.write(f"DELTA = {f_v4 - f_base:+.4f}\n")
        f.write(f"leg coverage of stage-2 GT dups: {cov_caught}/{cov_tot} = "
                f"{cov_caught / max(cov_tot, 1):.3f}\n")
        f.write("v4 recall by word-cos bin @ tuned thr:\n")
        for b in bins:
            f.write(f"  {b}\n")
        if f_v4 - f_base < 0.01:
            f.write("VERDICT: FAIL (< +0.01) - do NOT ship this v4; see "
                    "fallback notes.\n")
        else:
            f.write("VERDICT: PASS (>= +0.01) - proceed to full run.\n")
    log("wrote logs/v4/VALIDATION_sample.txt")
    return f_v4 - f_base


def run_full(n_train, n_tune, pos_k):
    import subprocess
    r = subprocess.run(["pgrep", "-f", "v3_block_full.py"], capture_output=True)
    if r.returncode == 0:
        log("REFUSING: v3_block_full.py is alive (OOM risk).")
        sys.exit(2)
    tmap, dev_suffix = load_fold_dict()
    sp = np.load(os.path.join(ART, "samples.npz"), allow_pickle=True)
    train_ids_all = [str(x) for x in sp["train_ids"].tolist()]
    tune_ids_all = [str(x) for x in sp["tune_ids"].tolist()]
    pool_paths = [os.path.join(NORM, "train_source2.tsv"),
                  os.path.join(NORM, "train_source3.tsv")]
    shard_map, shard_counts = discover_shards(pool_paths)
    shards = sorted(set(shard_map.values()) - {"other"})
    log(f"train shards: {shards}")

    # shard membership from v2 train cand npz
    shard_s1 = {}
    for shard in shards:
        p = os.path.join(TMP, f"train_{shard}_cand.npz")
        if os.path.exists(p):
            d = np.load(p, allow_pickle=True)
            shard_s1[shard] = [str(x) for x in d["s1_ids"].tolist()]
            d.close()
    tot = sum(len(v) for v in shard_s1.values())
    Xtr_l, ytr_l, Xte_l, yte_l, ent_l, gc_l = [], [], [], [], [], []
    ent_off = 0
    for shard in shards:
        sids = shard_s1.get(shard, [])
        if not sids:
            log(f"[{shard}] no shard membership file; skipping")
            continue
        s_tr = [i for i in train_ids_all if i in set(sids)]
        s_te = [i for i in tune_ids_all if i in set(sids)]
        ntr = max(500, int(n_train * len(sids) / max(tot, 1)))
        nte = max(200, int(n_tune * len(sids) / max(tot, 1)))
        rng = np.random.RandomState(7)
        tr_ids = [s_tr[i] for i in rng.choice(len(s_tr), min(ntr, len(s_tr)),
                                              replace=False)]
        te_ids = [s_te[i] for i in rng.choice(len(s_te), min(nte, len(s_te)),
                                              replace=False)]
        log(f"[{shard}] train S1={len(tr_ids)} tune S1={len(te_ids)}")
        want = set(tr_ids) | set(te_ids)
        # texts: use the norm 'text' col (name+addr); reload quickly
        s1t = {}
        for chunk in read_chunks(os.path.join(NORM, "train_source1.tsv"),
                                 usecols=["entity_id", "name_norm", "text",
                                          "country_norm"]):
            m = chunk["entity_id"].isin(want)
            sub = chunk[m]
            for eid, nm, tx in zip(sub["entity_id"], sub["name_norm"],
                                   sub["text"]):
                s1t[eid] = (tx, nm)
            if len(s1t) >= len(want):
                break
        out_tr = os.path.join(TMP, f"v4_train_{shard}_cand.npz")
        out_te = os.path.join(TMP, f"v4_tune_{shard}_cand.npz")
        build_train_block(shard, tr_ids, [s1t[i][0] for i in tr_ids],
                          [s1t[i][1] for i in tr_ids], out_tr)
        build_train_block(shard, te_ids, [s1t[i][0] for i in te_ids],
                          [s1t[i][1] for i in te_ids], out_te)
        del s1t
        gc.collect()
        gt = load_gt(want)
        Xtr, ytr, _, _, _ = featurize_npz(out_tr, tr_ids, gt, tmap, dev_suffix,
                                          n_feats=33)
        Xte, yte, ent, gtc, _ = featurize_npz(out_te, te_ids, gt, tmap,
                                              dev_suffix, n_feats=33)
        Xtr_l.append(Xtr); ytr_l.append(ytr)
        Xte_l.append(Xte); yte_l.append(yte)
        ent_l.append(ent + ent_off); gc_l.append(gtc)
        ent_off += len(te_ids)
        del Xtr, ytr, Xte, yte, ent, gtc, gt
        gc.collect()

    Xtr = np.vstack(Xtr_l); ytr = np.concatenate(ytr_l)
    Xte = np.vstack(Xte_l); yte = np.concatenate(yte_l)
    ent = np.concatenate(ent_l); gcounts = np.concatenate(gc_l)
    n_ent = ent_off
    del Xtr_l, ytr_l, Xte_l, yte_l, ent_l, gc_l
    gc.collect()
    log(f"FULL train {Xtr.shape} pos={ytr.mean():.4f} | tune {Xte.shape}")

    hgb = train_hgb(Xtr, ytr, pos_k)
    del Xtr, ytr
    gc.collect()
    prob = hgb.predict_proba(Xte)[:, 1]
    thr, f = global_threshold_tune(prob, yte, ent, gcounts, n_ent)
    log(f"FULL v4: tune F05={f:.4f} @ thr={thr:.3f}")
    del prob, Xte, yte, ent, gcounts
    gc.collect()

    os.makedirs(V4_ART, exist_ok=True)
    pickle.dump({"hgb": hgb}, open(os.path.join(V4_ART, "model.pkl"), "wb"))
    with open(os.path.join(V4_ART, "threshold.txt"), "w") as fh:
        fh.write(f"{thr:.4f}\n")
    with open(os.path.join(V4_ART, "report.txt"), "w") as fh:
        fh.write(f"v4 full tune macro F_0.5={f:.4f} @ thr={thr:.3f} "
                 f"(33 feats, pos_k={pos_k}, correct metric)\n")
    log(f"saved artifacts_v4/ (F05={f:.4f} @ thr={thr:.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--proto-dir", default=None)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--n-train", type=int, default=60000)
    ap.add_argument("--n-tune", type=int, default=12000)
    ap.add_argument("--pos-k", type=float, default=5.0,
                    help="hard-positive upweight strength (step7 validated)")
    args = ap.parse_args()
    if args.sample:
        assert args.proto_dir, "--proto-dir required with --sample"
        run_sample(args.proto_dir, args.pos_k)
    elif args.full:
        run_full(args.n_train, args.n_tune, args.pos_k)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
