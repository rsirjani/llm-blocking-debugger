"""Blockers A (reclin2 standard blocking) and C (klsh) -> run7 bank cells.

A: derived key, exact equality. Key = first L characters of the
   normalized name-like text; blocks are the key's equivalence classes,
   so P1 holds by construction. L is swept to land PC near --target-pc
   (this is the per-dataset human choice the method inherently needs;
   it is made mechanically here and recorded in the bank).
C: klsh k-means LSH over shingled record text; blocks are the clusters.
   num.blocks is swept for the same purpose.

Both run through block_r.R so the certified R implementations do the
work; this module only prepares input, sweeps the knob and builds
episodes in the shared bank schema.
"""
import argparse
import csv
import json
import os
import random
import re
import subprocess
import sys
import tempfile
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, "src"))
import loaders  # noqa: E402
from build_bank import (DATASET_META, NAME_FIELDS, TEST_DATASETS,
                        rec_fields)  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RSCRIPT = os.path.join(HERE, "block_r.R")

METHOD_META = {
    "reclin2": ("Blocker: standard blocking (reclin2). A blocking key is "
                "derived from each record's name-like text -- the first "
                "{param} characters after lowercasing and stripping "
                "non-alphanumerics -- and records whose keys are exactly "
                "equal form a block, so every record is in exactly one "
                "block. Matches get split when the derived keys differ "
                "at all."),
    "klsh": ("Blocker: KLSH (klsh package). Record text is shingled into "
             "character 2-grams, hashed by random projections and "
             "clustered by k-means into {param} blocks; each record "
             "belongs to exactly one cluster. Matches get split when "
             "they fall into different clusters."),
}


def norm_name(fields):
    named = [fields[k] for k in NAME_FIELDS if k in fields]
    if not named:
        named = list(fields.values())
    return re.sub(r"[^a-z0-9]", "", " ".join(named).lower())


def rec_text(fields):
    return re.sub(r"\s+", " ", " ".join(fields.values()))[:200]


def run_r(method, rows, param=None):
    with tempfile.TemporaryDirectory() as td:
        fin = os.path.join(td, "in.csv")
        fout = os.path.join(td, "out.csv")
        with open(fin, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, quoting=csv.QUOTE_ALL)
            w.writerow(["id", "key", "text"])
            w.writerows(rows)
        cmd = ["Rscript", RSCRIPT, method, fin, fout]
        if param is not None:
            cmd.append(str(param))
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=3600)
        if p.returncode != 0:
            raise RuntimeError(f"R failed: {p.stderr[-800:]}")
        assign = {}
        with open(fout, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                assign[row["id"]] = row["block"]
    return assign


def build(ds, method, grid, target_pc, seed, outdir):
    loader = getattr(loaders, "load_" + ds.replace("-", "_"))
    ta, tb, gold, _ = loader()
    id_a, id_b = loaders.ID_COLUMNS[ds]
    fa = {"A:" + str(r[id_a]): rec_fields(r, id_a) for _, r in ta.iterrows()}
    fb = {"B:" + str(r[id_b]): rec_fields(r, id_b) for _, r in tb.iterrows()}
    gold = {("A:" + str(a), "B:" + str(b)) for a, b in gold}
    allf = {**fa, **fb}

    best = None
    for param in grid:
        if method == "reclin2":
            rows = [(i, norm_name(f)[:param] or "_", rec_text(f))
                    for i, f in allf.items()]
            assign = run_r("A", rows)
        else:
            rows = [(i, "", rec_text(f)) for i, f in allf.items()]
            # k-means cannot ask for more clusters than records
            nb = min(param, max(2, len(rows) // 2))
            assign = run_r("C", rows, nb)
        assert len(assign) == len(allf), "P1 violation: missing records"
        hit = sum(1 for a, b in gold if assign.get(a) == assign.get(b))
        pc = hit / len(gold)
        nb = len(set(assign.values()))
        print(f"  {method} param={param} PC={pc:.3f} blocks={nb}",
              flush=True)
        if best is None or abs(pc - target_pc) < best[0]:
            best = (abs(pc - target_pc), param, pc, assign)
        if best[0] < 0.02:
            break
    _, param, pc, assign = best
    print(f"  chosen param={param} PC={pc:.3f}", flush=True)

    aside, bside = defaultdict(list), defaultdict(list)
    for i, c in assign.items():
        (aside if i.startswith("A:") else bside)[c].append(i)
    missed = [(a, b) for a, b in gold if assign[a] != assign[b]]
    by_bp = defaultdict(list)
    for a, b in missed:
        by_bp[(assign[a], assign[b])].append((a, b))
    multi = {bp: sorted(v) for bp, v in by_bp.items() if len(v) >= 2}

    rng = random.Random(f"{seed}:{ds}:{method}")
    episodes, used = [], set()
    for (s1, s2), pairs in sorted(multi.items()):
        rng.shuffle(pairs)
        seed_pair, targets = pairs[0], pairs[1:]
        a_ids, b_ids = sorted(aside[s1]), sorted(bside[s2])
        if not a_ids or not b_ids:
            continue
        episodes.append({"episode_id": f"{ds}_{method}{param}_{s1}_{s2}",
                         "dataset": ds, "x_sig": s1, "y_sig": s2,
                         "a_ids": a_ids, "b_ids": b_ids,
                         "seed_pair": list(seed_pair),
                         "targets": [list(t) for t in targets]})
        used.update(a_ids)
        used.update(b_ids)
    rng.shuffle(episodes)
    is_test = ds in TEST_DATASETS
    n_tr = int(len(episodes) * 0.75)
    for i, ep in enumerate(episodes):
        ep["split"] = ("test" if is_test
                       else "train" if i < n_tr else "val")
    bank = {"dataset": ds, "blocker": method, "K": param, "PC": pc,
            "n_gold": len(gold), "n_missed": len(missed),
            "dataset_metadata": DATASET_META[ds],
            "method_metadata": METHOD_META[method].format(param=param),
            "records": {i: allf[i] for i in used},
            "episodes": episodes}
    os.makedirs(outdir, exist_ok=True)
    json.dump(bank, open(os.path.join(outdir, f"{ds}.json"), "w"))
    n_t = sum(len(e["targets"]) for e in episodes)
    sp = defaultdict(int)
    for e in episodes:
        sp[e["split"]] += 1
    row = (f"{ds:15s} {method:8s} param={param} PC={pc:.3f} "
           f"missed={len(missed):5d} episodes={len(episodes):4d} "
           f"targets={n_t:5d} splits={dict(sp)} recs={len(used)}")
    print(row, flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["reclin2", "klsh"],
                    required=True)
    ap.add_argument("--datasets", nargs="+",
                    default=["amazon-google", "walmart-amazon",
                             "dblp-acm", "fodors-zagats", "abt-buy",
                             "dblp-scholar"])
    ap.add_argument("--target-pc", type=float, default=0.6)
    ap.add_argument("--grid", type=int, nargs="+", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    grid = args.grid or ([3, 4, 5, 6, 8, 10, 12]
                         if args.method == "reclin2"
                         else [50, 100, 200, 400, 800, 1600])
    outdir = os.path.join(HERE, os.pardir, "runs", "bank", args.method)
    rows = []
    for ds in args.datasets:
        print(f"{ds}: {args.method}", flush=True)
        rows.append(build(ds, args.method, grid, args.target_pc,
                          args.seed, outdir))
    open(os.path.join(outdir, "SUMMARY.txt"), "w").write(
        "\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
