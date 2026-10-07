"""Multi-dataset episode bank for the generalization campaign (run5).

For each clean-clean dataset: run blocklib lambda-fold (Lambda=1, mutually
exclusive, P1 asserted per record), auto-pick K whose PC is closest to
--target-pc, extract directional block-pair episodes (>=2 missed gold
pairs: 1 seed + hidden targets), and store generic per-field records.

Bank layout: runs/bank/lambdafold/<dataset>.json
  {dataset, blocker, K, PC, method_metadata, dataset_metadata,
   records: {rid: {field: value, ...}},        # only records used
   episodes: [{episode_id, x_sig, y_sig, a_ids, b_ids, seed_pair,
               targets, split}]}

Split: train datasets -> episodes 75/25 train/val (seeded);
       test datasets  -> all episodes 'test'.
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, "src"))
import loaders  # noqa: E402
from blocklib import generate_candidate_blocks  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

TRAIN_DATASETS = ["amazon-google", "walmart-amazon", "dblp-acm",
                  "fodors-zagats"]
TEST_DATASETS = ["abt-buy", "dblp-scholar"]

DATASET_META = {
    "amazon-google": "software products: Amazon vs Google Shopping",
    "walmart-amazon": "retail products: Walmart vs Amazon",
    "dblp-acm": "bibliographic records: DBLP vs ACM (papers)",
    "fodors-zagats": "restaurants: Fodors vs Zagat guides",
    "abt-buy": "consumer electronics products: Abt vs Buy.com",
    "dblp-scholar": "bibliographic records: DBLP vs Google Scholar",
}


def rec_fields(row, id_col):
    return {k: str(v)[:400] for k, v in row.items()
            if k != id_col and str(v).strip()}


NAME_FIELDS = ("title", "name", "manufacturer", "brand", "authors",
               "venue", "addr", "city", "type", "phone", "modelno")


def concat_text(fields):
    """Blocking text: name-like fields first, long free text excluded —
    matches how the published lambda-fold setups key on name fields."""
    named = [fields[k] for k in NAME_FIELDS if k in fields]
    if not named:
        named = list(fields.values())
    return " ".join(named)[:150]


def block_table(records, K, seed):
    cfg = {"type": "lambda-fold", "version": 1,
           "config": {"blocking-features": [1], "Lambda": 1,
                      "bf-len": 2000, "num-hash-funcs": 5, "K": K,
                      "random_state": seed, "input-clks": False}}
    data = list(records)
    res = generate_candidate_blocks(data, cfg, header=["id", "text"])
    assign, dup = {}, 0
    for sig, members in res.blocks.items():
        for m in members:
            rid = data[m][0]
            if rid in assign:
                dup += 1
            assign[rid] = str(sig)
    assert dup == 0 and len(assign) == len(data), "P1 violation"
    return assign


def pc_of(assign_a, assign_b, gold):
    hit = sum(1 for a, b in gold if assign_a.get(a) == assign_b.get(b))
    return hit / len(gold)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=TRAIN_DATASETS + TEST_DATASETS)
    ap.add_argument("--target-pc", type=float, default=0.6)
    ap.add_argument("--K-grid", type=int, nargs="+",
                    default=[2, 3, 4, 6, 8, 10, 12, 15, 20, 25, 30])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    outdir = os.path.join(HERE, os.pardir, "runs", "bank", "lambdafold")
    os.makedirs(outdir, exist_ok=True)
    summary = []
    for ds in args.datasets:
        loader = getattr(loaders, "load_" + ds.replace("-", "_"))
        ta, tb, gold, _ = loader()
        id_a, id_b = loaders.ID_COLUMNS[ds]
        # ids are namespaced by side: some benchmarks (walmart-amazon)
        # reuse the same id space in both tables, which silently
        # collapses records when they share one dict
        fields_a = {"A:" + str(r[id_a]): rec_fields(r, id_a)
                    for _, r in ta.iterrows()}
        fields_b = {"B:" + str(r[id_b]): rec_fields(r, id_b)
                    for _, r in tb.iterrows()}
        recs_a = [(i, concat_text(f)) for i, f in fields_a.items()]
        recs_b = [(i, concat_text(f)) for i, f in fields_b.items()]
        gold = {("A:" + str(a), "B:" + str(b)) for a, b in gold}

        best = None
        for K in args.K_grid:
            aa = block_table(recs_a, K, args.seed)
            ab = block_table(recs_b, K, args.seed)
            pc = pc_of(aa, ab, gold)
            if best is None or abs(pc - args.target_pc) < best[0]:
                best = (abs(pc - args.target_pc), K, pc, aa, ab)
        _, K, pc, aa, ab = best

        aside = defaultdict(list)
        bside = defaultdict(list)
        for r, s in aa.items():
            aside[s].append(r)
        for r, s in ab.items():
            bside[s].append(r)
        missed = [(a, b) for a, b in gold
                  if a in aa and b in ab and aa[a] != ab[b]]
        by_bp = defaultdict(list)
        for a, b in missed:
            by_bp[(aa[a], ab[b])].append((a, b))
        multi = {bp: sorted(v) for bp, v in by_bp.items() if len(v) >= 2}

        rng = random.Random(f"{args.seed}:{ds}")
        episodes = []
        used_ids = set()
        for (s1, s2), pairs in sorted(multi.items()):
            rng.shuffle(pairs)
            seed_pair, targets = pairs[0], pairs[1:]
            a_ids = sorted(aside[s1])
            b_ids = sorted(bside[s2])
            episodes.append({
                "episode_id": f"{ds}_K{K}_{s1}_{s2}",
                "dataset": ds, "x_sig": s1, "y_sig": s2,
                "a_ids": a_ids, "b_ids": b_ids,
                "seed_pair": list(seed_pair),
                "targets": [list(t) for t in targets]})
            used_ids.update(a_ids)
            used_ids.update(b_ids)
        rng.shuffle(episodes)
        is_test = ds in TEST_DATASETS
        n_tr = int(len(episodes) * 0.75)
        for i, ep in enumerate(episodes):
            ep["split"] = ("test" if is_test
                           else "train" if i < n_tr else "val")

        records = {i: f for i, f in fields_a.items() if i in used_ids}
        for i, f in fields_b.items():
            if i in used_ids:
                records[i] = f
        bank = {
            "dataset": ds, "blocker": "blocklib lambda-fold Lambda=1",
            "K": K, "PC": pc, "n_gold": len(gold),
            "n_missed": len(missed),
            "dataset_metadata": DATASET_META[ds],
            "method_metadata": (
                "Blocker: Lambda-fold LSH with Lambda=1 (blocklib). Record "
                "text is tokenized into character bigrams, hashed into a "
                f"2000-bit Bloom filter; K={K} sampled bit positions form "
                "the block signature. Identical signature = same block; "
                "every record is in exactly one block. Matches get split "
                "when any sampled bit differs (spelling variants, extra "
                "tokens, word order, abbreviations)."),
            "records": records, "episodes": episodes}
        path = os.path.join(outdir, f"{ds}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(bank, f)
        n_t = sum(len(e["targets"]) for e in episodes)
        splits = defaultdict(int)
        for e in episodes:
            splits[e["split"]] += 1
        row = (f"{ds:15s} K={K:2d} PC={pc:.3f} missed={len(missed):5d} "
               f"episodes={len(episodes):4d} targets={n_t:4d} "
               f"splits={dict(splits)} recs={len(records)}")
        print(row, flush=True)
        summary.append(row)
    with open(os.path.join(outdir, "SUMMARY.txt"), "w") as f:
        f.write("\n".join(summary) + "\n")


if __name__ == "__main__":
    main()
