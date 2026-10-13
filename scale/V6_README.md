# v6 GPU Workflow — RTX 4050 6GB

## What
Semantic blocking via Sentence-BERT + FAISS. Catches transliterations and
fuzzy matches that TF-IDF misses. Higher recall than v4's blocking.

## Setup
```
pip install sentence-transformers faiss-cpu torch numpy pandas scikit-learn rapidfuzz
```
Need: student_resource dataset (same as v4).

## Run
```
python3 v6_gpu_block.py
```
- Encodes 10M pool + 1.7M S1 on GPU (~4 hours).
- FAISS search top-50 per S1 (~1 hour).
- Output: `gpu_candidates.tsv` (s1_id -> pool_id:cosine,...).

## Next
Use `gpu_candidates.tsv` as the blocking for v4_train/v4_test
(replaces the TF-IDF npz). Or train a new classifier on the pairs.

## Time
~5-6 hours for blocking. +3h train +2.5h test = ~11h total.
Tight for midnight deadline — start ASAP.
