"""Bank v2: one builder, four blockers, clean-clean and dirty alike.

Emits per (blocker, dataset) cell:
  {schema: 2, blocker, dataset, er_mode, param, PC, n_gold,
   gold: [[a, b], ...],                # every gold pair, for live PC
   assignment: {rid: block_sig},       # the full partition
   records: {rid: {field: value}},     # every record in the cell
   block_pairs: [{x_sig, y_sig, a_ids, b_ids, failed: [[a, b], ...]}]}

Instances are derived later (bankio.derive), so the seed choice is a
sampling decision at run time rather than something baked in here.
Case-2 block pairs (a single separated pair) are kept and tagged.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, "src"))
import loaders  # noqa: E402
import build_bank as BB  # noqa: E402
import build_bank_dirty as BD  # noqa: E402
import build_bank_r as BR  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = ["amazon-google", "walmart-amazon", "dblp-acm", "fodors-zagats",
         "abt-buy", "dblp-scholar"]
DIRTY = ["cora", "musicbrainz-20k", "musicbrainz-200k"]

GRIDS = {"lambdafold": [2, 3, 4, 6, 8, 10, 12, 15, 20, 25, 30],
         "reclin2": [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48],
         # coarse first: 50 blocks was already far past PC 0.6 on every
         # dataset, so the sweep has to start much coarser
         "klsh": [3, 5, 8, 12, 20, 30, 50, 100, 200, 400],
         # resolution below 1 gives fewer, larger communities; the old
         # grid bottomed out at 1.0 and never reached PC 0.6 on
         # the larger corpora
         "simcse": [(3, 0.1), (3, 0.3), (3, 0.6), (3, 1.0),
                    (5, 1.0), (5, 5.0), (10, 5.0), (5, 20.0),
                    (10, 20.0), (10, 60.0), (10, 150.0)]}


def load_cell(ds):
    """-> (fields, gold, er_mode). ids namespaced by side when two."""
    loader = getattr(loaders, "load_" + ds.replace("-", "_"))
    ta, tb, gold, _ = loader()
    if tb is None:
        idc = loaders.ID_COLUMNS[ds][0]
        fields = {str(r[idc]): BD.rec_fields(r, idc)
                  for _, r in ta.iterrows()}
        gold = {BD.canon(str(a), str(b)) for a, b in gold}
        return fields, gold, "dirty"
    ia, ib = loaders.ID_COLUMNS[ds]
    fa = {"A:" + str(r[ia]): BB.rec_fields(r, ia) for _, r in ta.iterrows()}
    fb = {"B:" + str(r[ib]): BB.rec_fields(r, ib) for _, r in tb.iterrows()}
    gold = {("A:" + str(a), "B:" + str(b)) for a, b in gold}
    return {**fa, **fb}, gold, "clean"


def assign_with(blocker, fields, param, seed=0):
    items = list(fields.items())
    if blocker == "lambdafold":
        return BB.block_table([(i, BB.concat_text(f)) for i, f in items],
                              param, seed)
    if blocker == "reclin2":
        rows = [(i, BR.norm_name(f)[:param] or "_", BR.rec_text(f))
                for i, f in items]
        return BR.run_r("A", rows)
    if blocker == "simcse":
        import build_bank_simcse as BS
        k, res = param
        cache = assign_with._emb_cache
        key = id(fields)
        if key not in cache:
            texts = [" ".join(f.values())[:200] for _, f in items]
            cache.clear()
            cache[key] = BS.encode(texts)
        emb = cache[key]
        lab = BS.communities(BS.knn_graph(emb, k), seed, res)
        return {items[n][0]: c for n, c in lab.items()}
    if blocker == "klsh":
        rows = [(i, "", BR.rec_text(f)) for i, f in items]
        nb = min(param, max(2, len(rows) // 2))
        return BR.run_r("C", rows, nb)
    raise ValueError(blocker)


assign_with._emb_cache = {}


def build(blocker, ds, target_pc, seed, outdir, grid):
    fields, gold, er_mode = load_cell(ds)
    print(f"{blocker}/{ds}: {len(fields)} records, {len(gold)} gold "
          f"pairs ({er_mode})", flush=True)
    # A setting is only useful if it lands near the target operating
    # point AND leaves enough Case-3 block pairs to debug. Ranking on
    # |PC - target| alone chose settings with three usable block pairs
    # in an entire cell.
    cands = []
    for param in grid:
        assign = assign_with(blocker, fields, param, seed)
        assert len(assign) == len(fields), "P1 violation"
        pc = sum(1 for a, b in gold
                 if assign[a] == assign[b]) / len(gold)
        by_bp = defaultdict(int)
        for a, b in gold:
            sa, sb = assign[a], assign[b]
            if sa != sb:
                by_bp[(sa, sb) if sa <= sb else (sb, sa)] += 1
        n_case3 = sum(1 for v in by_bp.values() if v > 1)
        print(f"  param={param} PC={pc:.3f} "
              f"blocks={len(set(assign.values()))} case3={n_case3}",
              flush=True)
        cands.append((param, pc, n_case3, assign))
    in_band = [c for c in cands
               if abs(c[1] - target_pc) <= 0.15 and c[2] >= 10]
    if in_band:
        param, pc, n_case3, assign = max(in_band, key=lambda c: c[2])
    else:
        usable = [c for c in cands if c[2] >= 10]
        pool = usable or cands
        param, pc, n_case3, assign = min(
            pool, key=lambda c: abs(c[1] - target_pc))
    print(f"  chosen param={param} PC={pc:.3f} case3={n_case3}",
          flush=True)

    members = defaultdict(list)
    for i, s in assign.items():
        members[s].append(i)
    for s in members:
        members[s].sort()

    # R = R_A u R_B: clean-clean and dirty are treated identically.
    # A block is a subset of R, a block pair is UNORDERED {x, y}, and a
    # failure is any gold pair split across the two, in either
    # direction. Each instance therefore shows both blocks whole.
    failed_by_bp = defaultdict(list)
    for a, b in gold:
        sa, sb = assign[a], assign[b]
        if sa == sb:
            continue
        x, y = (sa, sb) if sa <= sb else (sb, sa)
        pair = (a, b) if assign[a] == x else (b, a)
        failed_by_bp[(x, y)].append(pair)

    bps = [{"x_sig": x, "y_sig": y, "a_ids": members[x],
            "b_ids": members[y],
            "failed": [list(p) for p in sorted(f)]}
           for (x, y), f in sorted(failed_by_bp.items())]
    n_case3 = sum(1 for b in bps if len(b["failed"]) > 1)
    cell = {"schema": 2, "blocker": blocker, "dataset": ds,
            "er_mode": er_mode, "param": (list(param)
                                          if isinstance(param, tuple)
                                          else param), "PC": pc,
            "n_gold": len(gold),
            "n_failed": sum(len(b["failed"]) for b in bps),
            "n_block_pairs": len(bps), "n_case3": n_case3,
            "dataset_metadata": (BB.DATASET_META.get(ds)
                                 or BD.DIRTY_META.get(ds, ds)),
            "method_metadata": f"{blocker} at parameter {param}"}
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{ds}.json")
    json.dump({**cell, "gold": [list(g) for g in sorted(gold)],
               "assignment": assign, "records": fields,
               "block_pairs": bps}, open(path, "w"))
    row = (f"{blocker:11s} {ds:17s} {er_mode:5s} "
           f"param={str(param):<12s} "
           f"PC={pc:.3f} failed={cell['n_failed']:6d} "
           f"bp={len(bps):5d} case3={n_case3:5d} recs={len(fields)}")
    print(row, flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blockers", nargs="+",
                    default=["lambdafold", "reclin2", "klsh"])
    ap.add_argument("--datasets", nargs="+", default=CLEAN + DIRTY)
    ap.add_argument("--target-pc", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outroot", default=os.path.join(
        HERE, os.pardir, "runs", "bank2"))
    args = ap.parse_args()
    rows = []
    for blocker in args.blockers:
        for ds in args.datasets:
            try:
                rows.append(build(blocker, ds, args.target_pc, args.seed,
                                  os.path.join(args.outroot, blocker),
                                  GRIDS[blocker]))
            except Exception as e:  # noqa: BLE001
                msg = f"{blocker:11s} {ds:17s} FAILED: {type(e).__name__}: {e}"
                print(msg, flush=True)
                rows.append(msg)
    os.makedirs(args.outroot, exist_ok=True)
    open(os.path.join(args.outroot, "SUMMARY.txt"), "w").write(
        "\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
