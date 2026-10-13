"""GPU workflow for RTX 4050 6GB — Entity Resolution via embeddings + FAISS.

Uses Sentence-BERT for semantic blocking (catches transliterations/fuzzy
that TF-IDF misses), then trains HGB classifier on pair features.

Requirements:
  pip install sentence-transformers faiss-cpu torch numpy pandas scikit-learn rapidfuzz

Dataset: student_resource/ with dataset/ inside (same as v4).

Time: ~8 hours on RTX 4050.
Expected: higher recall than TF-IDF, esp. for transliterated names.
"""
import os, sys, gc, pickle
import numpy as np
import pandas as pd

# --- Config ---
MODEL_NAME = "all-MiniLM-L6-v2"  # 384-dim, fast, fits 6GB
BATCH_SIZE = 256  # safe for 6GB VRAM
TOP_K = 50  # candidates per S1 from FAISS
DATA = "./student_resource/student_resource/dataset"
NORM = "./norm"  # reuse v4 norm files if present, else set to DATA

def get_paths():
    # Use norm TSVs if they exist (from v4 workflow), else raw
    import glob
    norm_s1 = os.path.join(NORM, "test_source1.tsv")
    if os.path.exists(norm_s1):
        return (os.path.join(NORM, "test_source1.tsv"),
                os.path.join(NORM, "test_source2.tsv"),
                os.path.join(NORM, "test_source3.tsv"),
                os.path.join(NORM, "train_source1.tsv"),
                os.path.join(NORM, "train_source2.tsv"),
                os.path.join(NORM, "train_source3.tsv"))
    # fallback to raw dataset
    d = os.path.join(DATA, "dataset")
    return (os.path.join(d, "test_source1.tsv"), os.path.join(d, "test_source2.tsv"),
            os.path.join(d, "test_source3.tsv"), os.path.join(d, "train_source1.tsv"),
            os.path.join(d, "train_source2.tsv"), os.path.join(d, "train_source3.tsv"))

def encode_texts(texts, model, batch_size=BATCH_SIZE):
    """Encode in batches on GPU. Returns float32 numpy."""
    from sentence_transformers import SentenceTransformer
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        e = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        embs.append(e.astype(np.float32))
        if (i // batch_size) % 20 == 0:
            print(f"  encoded {i}/{len(texts)}", flush=True)
    return np.vstack(embs)

def build_faiss(embs):
    """CPU FAISS index (10M vectors too big for 6GB VRAM)."""
    import faiss
    d = embs.shape[1]
    # IVF for speed on 10M
    nlist = 4096
    quant = faiss.IndexFlatIP(d)
    idx = faiss.IndexIVFFlat(quant, d, nlist)
    print("training IVF...", flush=True)
    idx.train(embs)
    print("adding...", flush=True)
    idx.add(embs)
    idx.nprobe = 32
    return idx

def main():
    import torch
    print(f"MPS: {torch.backends.mps.is_available()}", flush=True)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME, device="mps")
    print(f"model {MODEL_NAME} on CUDA", flush=True)

    s1_path, s2_path, s3_path, tr1, tr2, tr3 = get_paths()
    print("loading test pool...", flush=True)
    # Load S2+S3 (pool), encode name+address
    pool_frames = []
    for p in [s2_path, s3_path]:
        for chunk in pd.read_csv(p, sep="\t", usecols=["entity_id", "name_norm", "address_norm"],
                                 chunksize=200_000, dtype=str):
            chunk = chunk.fillna("")
            pool_frames.append(chunk)
    pool = pd.concat(pool_frames, ignore_index=True)
    print(f"pool: {len(pool)}", flush=True)
    pool_texts = (pool["name_norm"] + " " + pool["address_norm"]).tolist()
    del pool_frames; gc.collect()

    print("encoding pool on GPU...", flush=True)
    pool_embs = encode_texts(pool_texts, model)
    # normalize for cosine via IP
    pool_embs /= (np.linalg.norm(pool_embs, axis=1, keepdims=True) + 1e-9)
    pool_ids = pool["entity_id"].tolist()
    del pool, pool_texts; gc.collect()
    # save to disk to free RAM
    np.save("pool_embs.npy", pool_embs)

    print("building FAISS index...", flush=True)
    idx = build_faiss(pool_embs)
    del pool_embs; gc.collect()

    print("loading test S1...", flush=True)
    s1 = pd.read_csv(s1_path, sep="\t", usecols=["entity_id", "name_norm", "address_norm"],
                     dtype=str).fillna("")
    s1_texts = (s1["name_norm"] + " " + s1["address_norm"]).tolist()
    s1_ids = s1["entity_id"].tolist()
    print(f"S1: {len(s1_ids)}", flush=True)

    print("encoding S1 on GPU...", flush=True)
    s1_embs = encode_texts(s1_texts, model)
    s1_embs /= (np.linalg.norm(s1_embs, axis=1, keepdims=True) + 1e-9)

    print("FAISS search...", flush=True)
    D, I = idx.search(s1_embs, TOP_K)  # D=scores, I=pool indices
    print(f"search done: {I.shape}", flush=True)

    # Save candidates: for each S1, list of (pool_entity_id, cosine)
    print("saving candidates...", flush=True)
    with open("gpu_candidates.tsv", "w") as f:
        f.write("s1_id\tcandidates\n")
        for i, sid in enumerate(s1_ids):
            cands = []
            for k in range(TOP_K):
                pi = int(I[i, k])
                if pi < 0: continue
                cands.append(f"{pool_ids[pi]}:{float(D[i, k]):.4f}")
            f.write(f"{sid}\t{','.join(cands)}\n")
            if (i+1) % 200_000 == 0:
                print(f"  wrote {i+1}/{len(s1_ids)}", flush=True)
    print("DONE: gpu_candidates.tsv", flush=True)
    print("Next: train HGB on these candidates (see v4_train.py featurize), or use as blocking for v4.")

if __name__ == "__main__":
    main()
