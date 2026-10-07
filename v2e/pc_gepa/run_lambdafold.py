"""Blocker B: blocklib PPRLIndexLambdaFold with Lambda=1 on a clean-clean dataset.

Produces MUTUALLY EXCLUSIVE blocks: with Lambda=1 every record gets exactly one
LSH signature, hence sits in exactly one block (asserted, per P1 validation
protocol: assert blocks-per-record == 1, never validate on pair counts).

We deliberately AVOID blocklib.generate_blocks() (the multiparty entry point):
its signature filter DELETES out-of-range blocks, silently dropping records.
We use generate_candidate_blocks() per party and intersect signatures ourselves,
keeping the full partition of each table.

Sweep K (bits kept per fold) to move PC: larger K -> finer blocks -> lower PC.

Outputs per K under runs/lambdafold/<dataset>/K<K>/:
  assignments.csv   record side,id,block signature (the full partition)
  metrics.json      PC, RR, |C|, block stats, missed gold count
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, "src"))
import loaders  # noqa: E402

from blocklib import generate_candidate_blocks  # noqa: E402


def concat_text(row, cols):
    return " ".join(str(row[c]) for c in cols if c in row and str(row[c]).strip())


def block_one_table(records, K, lam, bf_len, nhash, seed):
    """records: list of (rec_id, text). Returns dict rec_id -> signature str."""
    data = [(rid, txt) for rid, txt in records]
    config = {
        "type": "lambda-fold",
        "version": 1,
        "config": {
            "blocking-features": [1],
            "Lambda": lam,
            "bf-len": bf_len,
            "num-hash-funcs": nhash,
            "K": K,
            "random_state": seed,
            "input-clks": False,
        },
    }
    res = generate_candidate_blocks(data, config, header=["id", "text"])
    assign = {}
    dup = 0
    for sig, members in res.blocks.items():
        for m in members:
            # blocklib may store positions or ids depending on version; normalize
            rid = data[m][0] if isinstance(m, int) and not isinstance(m, bool) else m
            if rid in assign:
                dup += 1
            assign[rid] = str(sig)
    n = len(data)
    assert dup == 0, f"P1 VIOLATION: {dup} records in >1 block"
    assert len(assign) == n, (
        f"P1 VIOLATION: {n - len(assign)} records in 0 blocks "
        f"(signature filter dropped them?)")
    return assign


def evaluate(assign_a, assign_b, gold):
    blocks_a = defaultdict(list)
    blocks_b = defaultdict(list)
    for r, s in assign_a.items():
        blocks_a[s].append(r)
    for r, s in assign_b.items():
        blocks_b[s].append(r)
    common = set(blocks_a) & set(blocks_b)
    n_cand = sum(len(blocks_a[s]) * len(blocks_b[s]) for s in common)
    hit = sum(1 for a, b in gold
              if assign_a.get(a) is not None
              and assign_a.get(a) == assign_b.get(b))
    n_total = len(assign_a) * len(assign_b)
    sizes = [len(blocks_a.get(s, ())) + len(blocks_b.get(s, ())) for s in
             set(blocks_a) | set(blocks_b)]
    return {
        "PC": hit / len(gold),
        "RR": 1.0 - n_cand / n_total,
        "num_candidates": n_cand,
        "gold_total": len(gold),
        "gold_hit": hit,
        "gold_missed": len(gold) - hit,
        "blocks_A": len(blocks_a),
        "blocks_B": len(blocks_b),
        "blocks_common": len(common),
        "max_block_size": max(sizes),
        "mean_block_size": sum(sizes) / len(sizes),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="amazon-google")
    ap.add_argument("--K", type=int, nargs="+",
                    default=[10, 15, 20, 25, 30, 40])
    ap.add_argument("--Lambda", type=int, default=1)
    ap.add_argument("--bf-len", type=int, default=2000)
    ap.add_argument("--num-hash-funcs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), os.pardir,
        "runs", "lambdafold"))
    args = ap.parse_args()

    loader = getattr(loaders, "load_" + args.dataset.replace("-", "_"))
    ta, tb, gold, _ = loader()
    id_a, id_b = loaders.ID_COLUMNS[args.dataset]

    text_cols_a = [c for c in ta.columns if c != id_a]
    text_cols_b = [c for c in tb.columns if c != id_b]
    # drop long free-text (description) and price: keys the published lambda-fold
    # setups block on are name-like fields
    keep = ("title", "name", "manufacturer")
    ka = [c for c in text_cols_a if c in keep] or text_cols_a
    kb = [c for c in text_cols_b if c in keep] or text_cols_b

    recs_a = [(r[id_a], concat_text(r, ka)) for _, r in ta.iterrows()]
    recs_b = [(r[id_b], concat_text(r, kb)) for _, r in tb.iterrows()]
    print(f"{args.dataset}: |A|={len(recs_a)} |B|={len(recs_b)} "
          f"gold={len(gold)}  featA={ka} featB={kb}", flush=True)

    rows = []
    for K in args.K:
        t0 = time.time()
        aa = block_one_table(recs_a, K, args.Lambda, args.bf_len,
                             args.num_hash_funcs, args.seed)
        ab = block_one_table(recs_b, K, args.Lambda, args.bf_len,
                             args.num_hash_funcs, args.seed)
        m = evaluate(aa, ab, gold)
        m["K"] = K
        m["runtime_s"] = round(time.time() - t0, 2)
        m["params"] = {"Lambda": args.Lambda, "bf_len": args.bf_len,
                       "num_hash_funcs": args.num_hash_funcs,
                       "seed": args.seed, "features_A": ka, "features_B": kb}
        od = os.path.join(args.outdir, args.dataset, f"K{K}")
        os.makedirs(od, exist_ok=True)
        with open(os.path.join(od, "assignments.csv"), "w",
                  encoding="utf-8", newline="") as f:
            f.write("side,record_id,block\n")
            for r, s in aa.items():
                f.write(f"A,{r},{s}\n")
            for r, s in ab.items():
                f.write(f"B,{r},{s}\n")
        with open(os.path.join(od, "metrics.json"), "w") as f:
            json.dump(m, f, indent=2)
        rows.append(m)
        print(f"K={K:3d}  PC={m['PC']:.4f}  RR={m['RR']:.4f}  "
              f"|C|={m['num_candidates']:>10,}  missed={m['gold_missed']:4d}  "
              f"blocks(A/B/common)={m['blocks_A']}/{m['blocks_B']}/"
              f"{m['blocks_common']}  max|b|={m['max_block_size']}  "
              f"{m['runtime_s']}s", flush=True)

    with open(os.path.join(args.outdir, args.dataset, "sweep.json"), "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
