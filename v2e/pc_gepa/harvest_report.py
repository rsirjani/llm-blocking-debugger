#!/usr/bin/env python3
"""Read the two harvest eval files and report seed vs best.

Reported per cell and never pooled into a single mean. cora contributes
roughly 0.049 of lambda-penalty per step against ~0.001 in the two-table
cells (base |C| is 10-22k there against 150-330k), so any average over
cells is a cora-weighted number wearing a disguise.

v0 is separated out from v1+ because v0 is the panel the search selected
against. Quoting it as a result would be reporting the objective that
was maximised.
"""
import json
import os
import statistics as S
import sys

RUN = sys.argv[1] if len(sys.argv) > 1 else "../runs/gepa/v2e_qwen38"
ARMS = ("seed", "best")


def load(arm):
    path = os.path.join(RUN, f"eval_val_{arm}.json")
    if not os.path.exists(path):
        sys.exit(f"missing {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def agg(rows, key):
    v = [r[key] for r in rows]
    return (S.fmean(v), S.pstdev(v) if len(v) > 1 else 0.0) if v else (0.0, 0.0)


def main():
    data = {a: load(a) for a in ARMS}
    rows = {a: data[a]["rows"] if "rows" in data[a] else data[a]["instances"]
            for a in ARMS}
    variants = sorted({r["variant"] for r in rows["best"]})
    cells = sorted({f"{r['cell'][0]}/{r['cell'][1]}" for r in rows["best"]})

    def sel(arm, cell=None, v=None, held_out=False):
        out = rows[arm]
        if cell:
            out = [r for r in out if f"{r['cell'][0]}/{r['cell'][1]}" == cell]
        if v is not None:
            out = [r for r in out if r["variant"] == v]
        if held_out:
            out = [r for r in out if r["variant"] > 0]
        return out

    print("=" * 78)
    print("CONTROL -- v0 is the panel GEPA selected against.")
    print("The best arm's v0 mean must reproduce the val mean the search")
    print("reported. A mismatch means this harness is wrong.")
    print("=" * 78)
    for arm in ARMS:
        m, sd = agg(sel(arm, v=0), "score")
        print(f"  {arm:<5} v0 score  mean {m:+.4f}  sd {sd:.4f}  "
              f"n={len(sel(arm, v=0))}")

    print(f"\nHELD-OUT (variants {variants[1:]}): block pairs no candidate")
    print("was ever selected on. Same partition, same records, same gold --")
    print("this is region transfer within a cell, not dataset transfer.\n")
    print(f"{'cell':<26}{'seed score':>12}{'best score':>12}{'delta':>9}"
          f"{'seed dPC':>10}{'best dPC':>10}{'d|C| best':>11}")
    for c in cells:
        s_sc, _ = agg(sel("seed", cell=c, held_out=True), "score")
        b_sc, _ = agg(sel("best", cell=c, held_out=True), "score")
        s_pc, _ = agg(sel("seed", cell=c, held_out=True), "d_pc")
        b_pc, _ = agg(sel("best", cell=c, held_out=True), "d_pc")
        b_dc, _ = agg(sel("best", cell=c, held_out=True), "d_c_rel")
        print(f"{c:<26}{s_sc:>12.4f}{b_sc:>12.4f}{b_sc - s_sc:>+9.4f}"
              f"{s_pc:>+10.4f}{b_pc:>+10.4f}{100 * b_dc:>+10.2f}%")

    print("\nper-cell win/loss on held-out variants (paired by instance id)")
    by_id = {}
    for arm in ARMS:
        for r in sel(arm, held_out=True):
            by_id.setdefault(r["instance_id"], {})[arm] = r
    paired = [v for v in by_id.values() if len(v) == 2]
    wins = sum(1 for v in paired if v["best"]["score"] > v["seed"]["score"])
    print(f"  best beats seed on {wins}/{len(paired)} paired instances")

    print("\nparse failures and skipped steps (held-out)")
    for arm in ARMS:
        r = sel(arm, held_out=True)
        pf = sum(x["parse_failures"] for x in r)
        sk = sum(x["skipped"] for x in r)
        st = sum(x["steps_run"] for x in r)
        print(f"  {arm:<5} steps_run {st:>6}  skipped {sk:>5}  "
              f"parse_failures {pf:>4} ({pf / max(1, st + pf):.1%})")

    print("\nCAVEAT to carry into any write-up: an instance is a whole-cell")
    print("replay, so all instances of a cell score the same partition.")
    print(f"Independent evaluated objects = {len(cells)} cells, not the")
    print("instance count. Four of those are the same 1,295 cora records")
    print("under different blockers.")


if __name__ == "__main__":
    main()
