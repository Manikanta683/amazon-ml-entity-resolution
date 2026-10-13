"""v5: Entity clustering layer (0.99 attempt).
Takes pair predictions (v4 output) + applies transitive closure with
precision controls. The 0.99 = high-precision pairs + closure for recall.

Usage:
  python3 v5_cluster.py --input matching_results.tsv --output matching_results_v5.tsv
  [--score-file candidate_pairs_scored.tsv] [--threshold 0.8]
"""
import argparse
import os
import sys
from collections import defaultdict, deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import log


def load_predictions(path):
    """Load S1 -> [dupes] from matching_results.tsv."""
    s1_dupes = {}
    order = []
    with open(path) as f:
        header = f.readline().rstrip("\n")
        for line in f:
            p = line.rstrip("\n").split("\t")
            s1 = p[0]
            order.append(s1)
            dupes = [d for d in p[1].split(",") if d] if len(p) > 1 and p[1] else []
            s1_dupes[s1] = dupes
    return header, s1_dupes, order


def build_graph(s1_dupes, min_component=2):
    """Build undirected graph from predictions."""
    graph = defaultdict(set)
    for s1, dupes in s1_dupes.items():
        for d in dupes:
            graph[s1].add(d)
            graph[d].add(s1)
    return graph


def find_components(graph):
    """BFS connected components. Returns dict node -> root, and root -> members."""
    comp_of = {}
    comp_members = defaultdict(list)
    for node in graph:
        if node in comp_of:
            continue
        # BFS
        dq = deque([node])
        comp_of[node] = node
        members = [node]
        while dq:
            u = dq.popleft()
            for v in graph[u]:
                if v not in comp_of:
                    comp_of[v] = node
                    members.append(v)
                    dq.append(v)
        for m in members:
            comp_of[m] = node
        comp_members[node] = members
    return comp_of, comp_members


def apply_closure(s1_dupes, order, comp_of, comp_members, max_expand=50):
    """Apply transitive closure with safety cap.
    
    For each S1: closure = direct dupes + component members (excluding self).
    Caps expansion to max_expand to prevent giant-component FP blowup.
    Returns (new_s1_dupes, n_expanded).
    """
    out = {}
    n_expanded = 0
    for s1 in order:
        direct = s1_dupes[s1]
        dset = set(direct)
        root = comp_of.get(s1)
        if root is None:
            out[s1] = direct
            continue
        members = comp_members[root]
        # Only expand if component is reasonable size (precision control)
        if len(members) > max_expand + 1:
            # Giant component: likely FP-driven, don't expand
            out[s1] = direct
            continue
        extra = [m for m in members if m != s1 and m not in dset]
        if extra:
            n_expanded += 1
        out[s1] = direct + extra
    return out, n_expanded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="matching_results.tsv (v4 output)")
    ap.add_argument("--output", required=True, help="output matching_results_v5.tsv")
    ap.add_argument("--max-expand", type=int, default=50,
                    help="max component size for closure expansion (precision guard)")
    args = ap.parse_args()

    log(f"v5: loading {args.input}")
    header, s1_dupes, order = load_predictions(args.input)
    log(f"v5: {len(order)} S1 entities")

    log("v5: building graph")
    graph = build_graph(s1_dupes)
    log(f"v5: graph nodes={len(graph)}")

    log("v5: finding components")
    comp_of, comp_members = find_components(graph)
    sizes = [len(v) for v in comp_members.values()]
    log(f"v5: components={len(sizes)}, max_size={max(sizes) if sizes else 0}")

    log(f"v5: applying closure (max_expand={args.max_expand})")
    closed, n_exp = apply_closure(s1_dupes, order, comp_of, comp_members,
                                  max_expand=args.max_expand)
    log(f"v5: expanded {n_exp} / {len(order)} S1s")

    log(f"v5: writing {args.output}")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as out:
        out.write(header + "\n")
        for s1 in order:
            out.write(f"{s1}\t{','.join(closed[s1])}\n")
    log("v5: done")


if __name__ == "__main__":
    main()
