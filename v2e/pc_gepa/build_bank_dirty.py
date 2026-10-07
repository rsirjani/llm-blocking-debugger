"""Dirty-ER bank cells: one table, duplicates inside it.

Differences from the clean-clean builder:
* one partition over a single table, so a block-pair is UNDIRECTED --
  the canonical form is (min(sig), max(sig)) and each pair is counted
  once;
* gold pairs are unordered, canonicalized as sorted tuples;
* an episode's two record lists are simply the members of the two
  blocks (there is no A side / B side), exposed through the same
  a_ids / b_ids fields so run7 can mix these cells with clean-clean
  ones without knowing the difference.

Blockers: lambdafold (blocklib) and simcse (kNN graph + Louvain).
Cluster columns (musicbrainz CID/CTID/SourceID) are stripped from the
record text: they are ground truth, not data.
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
from build_bank import block_table, concat_text  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

DIRTY_META = {
    "cora": "bibliographic records with duplicates (Cora citations)",
    "musicbrainz-20k": "music recordings with duplicates across five "
                       "catalogue sources",
    "musicbrainz-200k": "music recordings with duplicates across five "
                        "catalogue sources, ten times the scale",
}
# columns that encode ground truth or provenance, never shown as data
GOLD_COLS = {"CID", "CTID", "SourceID", "cluster_id", "recid"}


def rec_fields(row, id_col):
    return {k: str(v)[:400] for k, v in row.items()
            if k != id_col and k not in GOLD_COLS and str(v).strip()}


def canon(a, b):
    return (a, b) if a <= b else (b, a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["cora", "musicbrainz-20k"])
    ap.add_argument("--blocker", choices=["lambdafold", "simcse"],
                    default="lambdafold")
    ap.add_argument("--target-pc", type=float, default=0.6)
    ap.add_argument("--K-grid", type=int, nargs="+",
                    default=[2, 3, 4, 6, 8, 10, 12, 15, 20, 25, 30])
    ap.add_argument("--test-datasets", nargs="+",
                    default=["musicbrainz-200k"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    outdir = os.path.join(HERE, os.pardir, "runs", "bank", args.blocker)
    os.makedirs(outdir, exist_ok=True)
    summary = []
    for ds in args.datasets:
        loader = getattr(loaders, "load_" + ds.replace("-", "_"))
        ta, _, gold, _ = loader()
        id_col = loaders.ID_COLUMNS[ds][0]
        fields = {str(r[id_col]): rec_fields(r, id_col)
                  for _, r in ta.iterrows()}
        gold = {canon(str(a), str(b)) for a, b in gold}
        recs = [(i, concat_text(f)) for i, f in fields.items()]
        print(f"{ds}: {len(recs)} records, {len(gold)} duplicate pairs",
              flush=True)

        best = None
        for K in args.K_grid:
            assign = block_table(recs, K, args.seed)
            pc = sum(1 for a, b in gold
                     if assign[a] == assign[b]) / len(gold)
            print(f"  K={K:3d} PC={pc:.3f} "
                  f"blocks={len(set(assign.values()))}", flush=True)
            if best is None or abs(pc - args.target_pc) < best[0]:
                best = (abs(pc - args.target_pc), K, pc, assign)
        _, K, pc, assign = best
        print(f"  chosen K={K} PC={pc:.3f}", flush=True)

        members = defaultdict(list)
        for i, s in assign.items():
            members[s].append(i)
        missed = [(a, b) for a, b in gold if assign[a] != assign[b]]
        by_bp = defaultdict(list)
        for a, b in missed:
            by_bp[canon(assign[a], assign[b])].append((a, b))
        multi = {bp: sorted(v) for bp, v in by_bp.items() if len(v) >= 2}

        rng = random.Random(f"{args.seed}:{ds}:{args.blocker}")
        episodes, used = [], set()
        for (s1, s2), pairs in sorted(multi.items()):
            rng.shuffle(pairs)
            seed_pair, targets = pairs[0], pairs[1:]
            x_ids, y_ids = sorted(members[s1]), sorted(members[s2])
            # orient every pair as (record in block s1, record in s2)
            def orient(p):
                a, b = p
                return [a, b] if assign[a] == s1 else [b, a]
            episodes.append({
                "episode_id": f"{ds}_{args.blocker}K{K}_{s1}_{s2}",
                "dataset": ds, "x_sig": s1, "y_sig": s2,
                "a_ids": x_ids, "b_ids": y_ids,
                "seed_pair": orient(seed_pair),
                "targets": [orient(t) for t in targets]})
            used.update(x_ids)
            used.update(y_ids)
        rng.shuffle(episodes)
        is_test = ds in args.test_datasets
        n_tr = int(len(episodes) * 0.75)
        for i, ep in enumerate(episodes):
            ep["split"] = ("test" if is_test
                           else "train" if i < n_tr else "val")

        bank = {"dataset": ds, "blocker": args.blocker, "K": K, "PC": pc,
                "n_gold": len(gold), "n_missed": len(missed),
                "er_mode": "dirty",
                "dataset_metadata": DIRTY_META[ds],
                "method_metadata": (
                    "Blocker: Lambda-fold LSH with Lambda=1 (blocklib) "
                    "over a single table containing duplicates; "
                    f"K={K} sampled Bloom-filter bits form each record's "
                    "block signature, and every record is in exactly one "
                    "block. Duplicate records get separated when any "
                    "sampled bit differs."),
                "records": {i: fields[i] for i in used},
                "episodes": episodes}
        json.dump(bank, open(os.path.join(outdir, f"{ds}.json"), "w"))
        n_t = sum(len(e["targets"]) for e in episodes)
        sp = defaultdict(int)
        for e in episodes:
            sp[e["split"]] += 1
        row = (f"{ds:17s} {args.blocker:10s} K={K:2d} PC={pc:.3f} "
               f"missed={len(missed):6d} episodes={len(episodes):5d} "
               f"targets={n_t:6d} splits={dict(sp)} recs={len(used)}")
        print(row, flush=True)
        summary.append(row)
    path = os.path.join(outdir, "SUMMARY_dirty.txt")
    open(path, "w").write("\n".join(summary) + "\n")


if __name__ == "__main__":
    main()
