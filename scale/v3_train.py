#!/usr/bin/env python3
"""v3 training: 30 features + HGB/LR ensemble + per-entity adaptive threshold.

Modes:
  --sample N   quick validation: N S1 per shard from train, 2N/3... (see code)
               trains BOTH a v2-style baseline (20 feats, HGB, global thr)
               and v3 (30 feats, HGB+LR ensemble, adaptive thr) on the same
               sample; reports tune macro F_0.5 for both.
  --full       train v3 on the full 150k/50k split, save artifacts_v3/.

All features come from the provided TSVs only (no external data).
"""
import os
import sys
import gc
import pickle
import argparse

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E402
from common import NORM, TMP, ART, DATASET, read_chunks, log, \
    macro_f05_from_counts, pick_threshold_midplateau  # noqa: E402
from featurize import featurize_chunk as featurize_v2  # noqa: E402
from v3_featurize import featurize_chunk_v3, N_FEATURES, N_PREFILTER  # noqa: E402

V3_ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts_v3")


def load_gt(want):
    gt = {}
    for chunk in read_chunks(os.path.join(DATASET, "train", "train_ground_truth.tsv")):
        for sid, mids in zip(chunk["source1_entity_id"], chunk["matched_entity_ids"]):
            if sid in want:
                gt[sid] = set(m for m in str(mids).split(",") if m and m != "nan")
    return gt


def shard_data(shard, tag):
    d = np.load(os.path.join(TMP, f"{tag}_{shard}_cand.npz"), allow_pickle=True)
    s1_ids = [str(x) for x in d["s1_ids"].tolist()]
    pool_ids = np.array([x.decode() if isinstance(x, bytes) else str(x)
                         for x in d["pool_ids"].tolist()])
    return s1_ids, d["top_rows"], d["top_cos"], pool_ids


def load_pool_fields(pool_ids_arr, ref_rows):
    """Stream norm pool TSVs, keep only referenced pool ids."""
    ref_pids = set(pool_ids_arr[r] for r in ref_rows)
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
    return pf


def load_s1_fields(want):
    s1f = {}
    for chunk in read_chunks(os.path.join(NORM, "train_source1.tsv")):
        m = chunk["entity_id"].isin(want)
        sub = chunk[m]
        for eid, nm, ad, cy in zip(sub["entity_id"], sub["name_norm"],
                                   sub["address_norm"], sub["country_norm"]):
            s1f[eid] = (nm, ad, cy)
        if len(s1f) >= len(want):
            break
    return s1f


def fit_name_addr_vecs(names, addrs):
    kw = dict(analyzer="word", token_pattern=r"(?u)\b\w\w+\b",
              min_df=2, max_df=0.5, sublinear_tf=True, dtype=np.float32)
    vn = TfidfVectorizer(**kw).fit(names)
    va = TfidfVectorizer(**kw).fit(addrs)
    return vn, va


def cos_for_pairs(vec, s1_strs, pool_strs, pairs):
    """pairs: list of (s1_pos, pool_pos) -> cosine array (rows L2-normalized)."""
    A = vec.transform(s1_strs)
    B = vec.transform(pool_strs).tocsr()
    out = np.empty(len(pairs), dtype=np.float32)
    by_pos = {}
    for k, (p, r) in enumerate(pairs):
        by_pos.setdefault(p, []).append((k, r))
    for p, krs in by_pos.items():
        ks = [k for k, _ in krs]
        rs = [r for _, r in krs]
        sims = np.asarray(A.getrow(p).dot(B[rs].T).todense()).ravel()
        for k, s in zip(ks, sims):
            out[k] = s
    return out


def featurize_shard_tag(shard, tag, s1_ids, gt, v3=True, batch=10000):
    """Featurize (s1_ids x top_rows) -> (X, y, ent, gt_counts, cheap)."""
    s1_all, top_rows, top_cos, pool_ids = shard_data(shard, tag)
    pos = {sid: i for i, sid in enumerate(s1_all)}
    idx = [pos[sid] for sid in s1_ids]
    ref_rows = sorted(set(int(r) for i in idx for r in top_rows[i] if r >= 0))
    rmap = {r: i for i, r in enumerate(ref_rows)}
    rpids = [pool_ids[r] for r in ref_rows]
    pf = load_pool_fields(pool_ids, ref_rows)
    s1f = load_s1_fields(set(s1_ids))
    pn = [pf[pid][0] for pid in rpids]
    pa = [pf[pid][1] for pid in rpids]
    pc = [pf[pid][2] for pid in rpids]
    del pf
    gc.collect()

    vn, va = None, None
    if v3:
        log(f"[{shard}/{tag}] fitting name/addr TF-IDF on {len(pn):,} pool rows")
        vn, va = fit_name_addr_vecs(pn, pa)

    X_parts, y_parts, c_parts = [], [], []
    ent_parts = []
    for b0 in range(0, len(s1_ids), batch):
        b1 = min(len(s1_ids), b0 + batch)
        recs, rl, cl = [], [], []
        pairs = []  # (s1_pos_in_batch, pool_pos)
        for j in range(b0, b1):
            gi = idx[j]
            rows = [rmap[int(r)] for r in top_rows[gi] if int(r) >= 0]
            nm, ad, cy = s1f[s1_ids[j]]
            recs.append((nm, ad, cy))
            rl.append(rows)
            cl.append([float(top_cos[gi, t]) for t in range(len(rows))])
            for r in rows:
                pairs.append((j - b0, r))
        if v3:
            cn = cos_for_pairs(vn, [r[0] for r in recs], pn, pairs)
            ca = cos_for_pairs(va, [r[1] for r in recs], pa, pairs)
            # split per-S1
            cn_l, ca_l, k = [], [], 0
            for rows in rl:
                cn_l.append(cn[k:k + len(rows)])
                ca_l.append(ca[k:k + len(rows)])
                k += len(rows)
            Xb, pb = featurize_chunk_v3(recs, rl, cl, pn, pa, pc,
                                        cn_l, ca_l, n_jobs=1)
            Cb, _ = featurize_chunk_v3(recs, rl, cl, pn, pa, pc,
                                       n_jobs=1, prefilter_only=True)
        else:
            Xb, pb = featurize_v2(recs, rl, cl, pn, pa, pc, n_jobs=1)
            Cb = None
        yb = np.zeros(len(Xb), dtype=np.int8)
        k = 0
        for pos_j, sid in enumerate(s1_ids[b0:b1]):
            g = gt.get(sid, set())
            for r in rl[pos_j]:
                yb[k] = 1 if rpids[r] in g else 0
                k += 1
        eb = np.repeat(np.arange(b0, b1, dtype=np.int32),
                       [len(r) for r in rl])
        X_parts.append(Xb)
        y_parts.append(yb)
        ent_parts.append(eb)
        if Cb is not None:
            c_parts.append(Cb)
        del Xb, yb, eb
        gc.collect()
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    ent = np.concatenate(ent_parts) if ent_parts else np.zeros(0, dtype=np.int32)
    cheap = np.vstack(c_parts) if c_parts else None
    gt_counts = np.array([len(gt.get(s, set())) for s in s1_ids], dtype=np.int32)
    log(f"[{shard}/{tag}] X={X.shape} pos={y.sum():,} ({y.mean():.4f})")
    return X, y, ent, gt_counts, cheap


def per_entity_counts(ent, pred, y, gt_counts, n_ent):
    tp = np.bincount(ent[(pred == 1) & (y == 1)], minlength=n_ent)
    fp = np.bincount(ent[(pred == 1) & (y == 0)], minlength=n_ent)
    fn = gt_counts - tp  # includes dups missed by blocking
    return tp, fp, fn


def global_threshold_tune(prob, y, ent, gt_counts, n_ent):
    thresholds = np.arange(0.05, 0.95, 0.025)
    scores = []
    for t in thresholds:
        pred = (prob >= t).astype(np.int8)
        tp, fp, fn = per_entity_counts(ent, pred, y, gt_counts, n_ent)
        scores.append(macro_f05_from_counts(tp, fp, fn))
    return pick_threshold_midplateau(thresholds, scores)


def adaptive_threshold_tune(prob, y, ent, gt_counts, n_ent,
                            betas=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
                            los=(0.10, 0.15, 0.20, 0.25, 0.30, 0.40)):
    """Per-entity rule: predict {p >= max(lo, beta * p_max)}.

    Vectorized over the (beta, lo) grid using per-entity p_max.
    """
    order = np.argsort(ent, kind="stable")
    ent_s = ent[order]
    prob_s = prob[order]
    bounds = np.searchsorted(ent_s, np.arange(n_ent + 1))
    p_max = np.zeros(n_ent)
    for e in range(n_ent):
        s, ee = bounds[e], bounds[e + 1]
        if ee > s:
            p_max[e] = prob_s[s:ee].max()
    pmax_pair = p_max[ent]

    best = (-1, None, None)
    for beta in betas:
        for lo in los:
            thr_pair = np.maximum(lo, beta * pmax_pair)
            pred = (prob >= thr_pair).astype(np.int8)
            tp, fp, fn = per_entity_counts(ent, pred, y, gt_counts, n_ent)
            f = macro_f05_from_counts(tp, fp, fn)
            if f > best[0]:
                best = (f, beta, lo)
    return best  # (f05, beta, lo)


def tune_prefilter(cheap, y, ent):
    """Grid-search a cheap keep-rule: keep if cos>=c0 or name_tok_cont>=t1
    or name_bi_cont>=t2. Objective: positive recall >= 0.999, min survivors."""
    cos, ntc, atc, nbc, abc, ntrc = (cheap[:, i] for i in range(N_PREFILTER))
    pos = y == 1
    n_pos = pos.sum()
    best = None
    for c0 in (0.0, 0.01, 0.02, 0.05):
        for t1 in (0.3, 0.5, 0.7):
            for t2 in (0.4, 0.6, 0.8):
                keep = (cos >= c0) | (ntc >= t1) | (nbc >= t2)
                rec = keep[pos].sum() / max(n_pos, 1)
                surv = keep.mean()
                if rec >= 0.999 and (best is None or surv < best[1]):
                    best = ((c0, t1, t2), surv, rec)
    return best


def train_models(Xtr, ytr):
    log(f"training HGB on {Xtr.shape}...")
    hgb = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=63,
        min_samples_leaf=50, l2_regularization=1.0, random_state=0)
    hgb.fit(Xtr, ytr)
    log("training LR (scaled)...")
    lr = make_pipeline(StandardScaler(),
                       LogisticRegression(max_iter=2000, C=0.5,
                                          class_weight="balanced",
                                          random_state=0))
    lr.fit(Xtr, ytr)
    return hgb, lr


def per_entity_expected_f05(prob, y, ent, gt_counts, n_ent):
    """Per-entity adaptive rule: for each entity, sort candidates by prob
    desc, pick the top-k (k=0..K) maximizing the *expected* F_0.5 where
    expectations use the probs as truth probabilities:
      E[TP|top-k] = sum(probs), E[FP] = k - E[TP], E[FN] = total_p - E[TP].
    Returns (macro F_0.5 on true labels, mean k chosen).
    """
    order = np.argsort(ent, kind="stable")
    ent_s = ent[order]
    prob_s = prob[order]
    y_s = y[order]
    bounds = np.searchsorted(ent_s, np.arange(n_ent + 1))
    pred = np.zeros_like(prob, dtype=bool)
    ks = np.zeros(n_ent, dtype=np.int32)
    for e in range(n_ent):
        s, ee = bounds[e], bounds[e + 1]
        if ee <= s:
            continue
        p = prob_s[s:ee]
        ix = np.argsort(-p, kind="stable")
        ps = p[ix]
        cum_tp = np.cumsum(ps)
        kk = np.arange(1, len(ps) + 1)
        total_p = ps.sum()
        exp_prec = cum_tp / kk
        exp_rec = cum_tp / max(total_p, 1e-9)
        denom = 0.25 * exp_prec + exp_rec
        with np.errstate(divide="ignore", invalid="ignore"):
            f = np.where(denom > 0,
                         1.25 * exp_prec * exp_rec / np.maximum(denom, 1e-12),
                         0.0)
        f0 = float(np.prod(1.0 - ps))  # expected F for predicting nothing
        k_best = int(np.argmax(np.concatenate([[f0], f])))
        ks[e] = k_best
        if k_best > 0:
            # map back to positions in the original (unsorted) arrays
            orig_idx = order[s:ee][ix[:k_best]]
            pred[orig_idx] = True
    pred = pred.astype(np.int8)
    tp, fp, fn = per_entity_counts(ent, pred, y, gt_counts, n_ent)
    return macro_f05_from_counts(tp, fp, fn), float(ks.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0,
                    help="N S1 per shard for quick validation (0=full)")
    ap.add_argument("--baseline", action="store_true",
                    help="also train v2-style baseline (20 feats, HGB, global thr)")
    args = ap.parse_args()

    sp = np.load(os.path.join(ART, "samples.npz"), allow_pickle=True)
    train_ids_all = [str(x) for x in sp["train_ids"].tolist()]
    tune_ids_all = [str(x) for x in sp["tune_ids"].tolist()]
    log(f"train={len(train_ids_all):,} tune={len(tune_ids_all):,}")

    Xtr_l, ytr_l, Xte_l, yte_l, ent_l, gcounts_l, cheap_l = [], [], [], [], [], [], []
    ent_off = 0
    for shard in ("india", "us"):
        s1_tr, _, _, _ = shard_data(shard, "train")
        s1_te, _, _, _ = shard_data(shard, "tune")
        s_tr = set(s1_tr)
        s_te = set(s1_te)
        tr_ids = [i for i in train_ids_all if i in s_tr]
        te_ids = [i for i in tune_ids_all if i in s_te]
        if args.sample:
            rng = np.random.RandomState(0)
            tr_ids = [tr_ids[i] for i in
                      rng.choice(len(tr_ids), min(args.sample, len(tr_ids)),
                                 replace=False)]
            te_ids = [te_ids[i] for i in
                      rng.choice(len(te_ids), min(args.sample // 2, len(te_ids)),
                                 replace=False)]
        log(f"[{shard}] train S1={len(tr_ids):,} tune S1={len(te_ids):,}")
        want = set(tr_ids) | set(te_ids)
        gt = load_gt(want)
        Xtr, ytr, _, _, _ = featurize_shard_tag(shard, "train", tr_ids, gt, v3=True)
        Xte, yte, ent, gtc, cheap = featurize_shard_tag(shard, "tune", te_ids, gt, v3=True)
        Xtr_l.append(Xtr); ytr_l.append(ytr)
        Xte_l.append(Xte); yte_l.append(yte)
        ent_l.append(ent + ent_off); gcounts_l.append(gtc); cheap_l.append(cheap)
        ent_off += len(te_ids)
        del Xtr, ytr, Xte, yte, ent, gtc, cheap
        gc.collect()

    Xtr = np.vstack(Xtr_l); ytr = np.concatenate(ytr_l)
    Xte = np.vstack(Xte_l); yte = np.concatenate(yte_l)
    ent = np.concatenate(ent_l); cheap = np.vstack(cheap_l); gcounts = np.concatenate(gcounts_l)
    n_ent = ent_off
    del Xtr_l, ytr_l, Xte_l, yte_l, ent_l, cheap_l
    gc.collect()
    log(f"train {Xtr.shape} pos={ytr.mean():.4f} | tune {Xte.shape} n_ent={n_ent:,}")

    hgb, lr = train_models(Xtr, ytr)
    del Xtr, ytr
    gc.collect()
    ph = hgb.predict_proba(Xte)[:, 1]
    pl = lr.predict_proba(Xte)[:, 1]

    # ensemble weight sweep (HGB weight), each with adaptive threshold tune
    results = {}
    for w in (0.5, 0.6, 0.7, 0.8, 1.0):
        pe = w * ph + (1 - w) * pl
        f_ad, beta, lo = adaptive_threshold_tune(pe, yte, ent, gcounts, n_ent)
        t_glob, f_glob = global_threshold_tune(pe, yte, ent, gcounts, n_ent)
        results[w] = (f_ad, beta, lo, t_glob, f_glob)
        log(f"w_hgb={w:.1f}: adapt F05={f_ad:.4f} (beta={beta}, lo={lo}) | "
            f"global F05={f_glob:.4f} @ {t_glob:.3f}")
    best_w = max(results, key=lambda w: results[w][0])
    f_ad, beta, lo, t_glob, f_glob = results[best_w]
    log(f"BEST v3: w_hgb={best_w} adapt F05={f_ad:.4f} (beta={beta}, lo={lo}) "
        f"vs global {f_glob:.4f}")

    # per-entity expected-F_0.5 rule (no tuning knobs)
    pe_best = best_w * ph + (1 - best_w) * pl
    f_exp, mean_k = per_entity_expected_f05(pe_best, yte, ent, gcounts, n_ent)
    log(f"per-entity expected-F05 rule: F05={f_exp:.4f} (mean k={mean_k:.2f})")

    # cache tune probs for fast offline threshold experiments
    if args.sample:
        np.savez_compressed(
            os.path.join(TMP, "v3_tune_sample_probs.npz"),
            prob_hgb=ph, prob_lr=pl, y=yte, ent=ent,
            gt_counts=gcounts, n_ent=np.int64(n_ent))
        log("cached tune probs -> tmp/v3_tune_sample_probs.npz")

    pf = tune_prefilter(cheap, yte, ent)
    log(f"prefilter: {pf} (rule, mean_survivors, pos_recall)")
    del cheap
    gc.collect()

    if args.baseline:
        log("=== v2-style baseline on same sample (20 feats, HGB, global thr) ===")
        Xb_tr_l, yb_tr_l, Xb_te_l, yb_te_l, entb_l, gcb_l = [], [], [], [], [], []
        off = 0
        for shard in ("india", "us"):
            s1_tr, _, _, _ = shard_data(shard, "train")
            s1_te, _, _, _ = shard_data(shard, "tune")
            s_tr = set(s1_tr); s_te = set(s1_te)
            tr_ids = [i for i in train_ids_all if i in s_tr]
            te_ids = [i for i in tune_ids_all if i in s_te]
            if args.sample:
                rng = np.random.RandomState(0)
                tr_ids = [tr_ids[i] for i in rng.choice(
                    len(tr_ids), min(args.sample, len(tr_ids)), replace=False)]
                te_ids = [te_ids[i] for i in rng.choice(
                    len(te_ids), min(args.sample // 2, len(te_ids)), replace=False)]
            want = set(tr_ids) | set(te_ids)
            gt = load_gt(want)
            Xtr, ytr, _, _, _ = featurize_shard_tag(shard, "train", tr_ids, gt, v3=False)
            Xte, yte, ent, gtc_b, _ = featurize_shard_tag(shard, "tune", te_ids, gt, v3=False)
            Xb_tr_l.append(Xtr); yb_tr_l.append(ytr)
            Xb_te_l.append(Xte); yb_te_l.append(yte); entb_l.append(ent + off)
            gcb_l.append(gtc_b)
            off += len(te_ids)
            del Xtr, ytr, Xte, yte, ent, gtc_b
            gc.collect()
        Xb_tr = np.vstack(Xb_tr_l); yb_tr = np.concatenate(yb_tr_l)
        Xb_te = np.vstack(Xb_te_l); yb_te = np.concatenate(yb_te_l)
        entb = np.concatenate(entb_l); gcb = np.concatenate(gcb_l)
        hb = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, max_leaf_nodes=63,
            min_samples_leaf=50, l2_regularization=1.0, random_state=0)
        hb.fit(Xb_tr, yb_tr)
        pb = hb.predict_proba(Xb_te)[:, 1]
        t_b, f_b = global_threshold_tune(pb, yb_te, entb, gcb, off)
        log(f"BASELINE v2-style: tune F05={f_b:.4f} @ thr={t_b:.3f}")

    if not args.sample:
        os.makedirs(V3_ART, exist_ok=True)
        pickle.dump({"hgb": hgb, "lr": lr, "w_hgb": best_w},
                    open(os.path.join(V3_ART, "model.pkl"), "wb"))
        # global threshold wins over per-entity rules (validated) -> plain float
        with open(os.path.join(V3_ART, "threshold.txt"), "w") as f:
            f.write(f"{t_glob:.4f}\n")
        with open(os.path.join(V3_ART, "report.txt"), "w") as f:
            f.write(f"v3 tune macro F_0.5={f_glob:.4f} @ thr={t_glob:.3f} "
                    f"w_hgb={best_w} (correct metric, incl. blocking-miss FNs)\n"
                    f"adaptive(beta,lo)={f_ad:.4f} expected-F-rule={f_exp:.4f}\n"
                    f"prefilter={pf}\n")
        log(f"saved artifacts_v3/ (F05={f_glob:.4f} @ thr={t_glob:.3f})")


if __name__ == "__main__":
    main()
