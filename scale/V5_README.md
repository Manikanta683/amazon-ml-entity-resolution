# v5 Clustering — Apply on v4 output

## What this is
Transitive closure on v4's pair predictions. Finds connected components
and expands each S1's predictions to include component members.
This is the recall booster for the 0.99 push.

## Run (after v4 finishes)
```bash
python3 v5_cluster.py \
  --input output_v4/matching_results.tsv \
  --output output_v4/matching_results_v5.tsv \
  --max-expand 50
```

## What it does
1. Loads v4's S1 -> [dupes] predictions
2. Builds graph from all predicted pairs
3. Finds connected components (BFS)
4. For each S1: adds component members not already predicted
5. Skips giant components (>50 members) to avoid FP blowup

## Output
`matching_results_v5.tsv` — upload this instead of v4's file.

## Tuning
- `--max-expand 50`: higher = more recall, more FP risk. Try 30/50/100.
