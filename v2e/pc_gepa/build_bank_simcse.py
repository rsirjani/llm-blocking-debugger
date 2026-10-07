"""Blocker D: SimCSE k-NN graph -> Louvain communities (Mugeni & Amagasa).

Blocks are graph communities, so every record is in exactly one block
(P1 holds by construction; asserted anyway). Schema-agnostic: all field
values are concatenated into one string, as the paper specifies.

k of the k-NN graph is the granularity knob; swept per dataset to land
pair completeness closest to --target-pc.

Writes runs/bank/simcse/<dataset>.json in the same schema as
build_bank.py so run7 can mix cells across blockers.
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict

import networkx as nx
import numpy as np
import torch
from sklearn.preprocessing import normalize

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, "src"))
import loaders  # noqa: E402
from build_bank import (DATASET_META, TEST_DATASETS, rec_fields)  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "princeton-nlp/sup-simcse-bert-base-uncased"


def encode(texts, batch=128, device=None):
    from transformers import AutoModel, AutoTokenizer
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(device).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i + batch], padding=True, truncation=True,
                      max_length=64, return_tensors="pt").to(device)
            h = model(**enc).pooler_output
            out.append(h.cpu().numpy())
            if i % (batch * 40) == 0:
                print(f"  encoded {i}/{len(texts)}", flush=True)
    return normalize(np.vstack(out))


def knn_graph(emb, k):
    """Cosine k-NN graph over all records (both tables)."""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(emb)),
                          metric="cosine").fit(emb)
    dist, idx = nn.kneighbors(emb)
    g = nx.Graph()
    g.add_nodes_from(range(len(emb)))
    for i, (ds, js) in enumerate(zip(dist, idx)):
        for d, j in zip(ds, js):
            if i != j:
                g.add_edge(i, int(j), weight=float(1.0 - d))
    return g


def communities(g, seed=0, resolution=1.0):
    parts = nx.community.louvain_communities(g, weight="weight",
                                             resolution=resolution,
                                             seed=seed)
    lab = {}
    for c, nodes in enumerate(parts):
        for n in nodes:
            lab[n] = f"c{c}"
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["amazon-google", "walmart-amazon",
                             "dblp-acm", "fodors-zagats", "abt-buy",
                             "dblp-scholar"])
    ap.add_argument("--target-pc", type=float, default=0.6)
    ap.add_argument("--k-grid", type=int, nargs="+",
                    default=[3, 5, 10])
    ap.add_argument("--res-grid", type=float, nargs="+",
                    default=[1.0, 5.0, 20.0, 60.0, 150.0])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    outdir = os.path.join(HERE, os.pardir, "runs", "bank", "simcse")
    os.makedirs(outdir, exist_ok=True)
    summary = []
    for ds in args.datasets:
        loader = getattr(loaders, "load_" + ds.replace("-", "_"))
        ta, tb, gold, _ = loader()
        id_a, id_b = loaders.ID_COLUMNS[ds]
        # ids namespaced by side; see build_bank.py note
        fa = {"A:" + str(r[id_a]): rec_fields(r, id_a)
              for _, r in ta.iterrows()}
        fb = {"B:" + str(r[id_b]): rec_fields(r, id_b)
              for _, r in tb.iterrows()}
        gold = {("A:" + str(a), "B:" + str(b)) for a, b in gold}
        ids = list(fa) + list(fb)
        side = {i: ("A" if i in fa else "B") for i in ids}
        texts = [" ".join((fa.get(i) or fb.get(i) or {}).values())[:200]
                 for i in ids]
        print(f"{ds}: encoding {len(texts)} records", flush=True)
        emb = encode(texts)

        best = None
        for k in args.k_grid:
            g = knn_graph(emb, k)
            for res in args.res_grid:
                lab = communities(g, args.seed, res)
                assign = {ids[n]: c for n, c in lab.items()}
                assert len(assign) == len(ids), "P1 violation"
                hit = sum(1 for a, b in gold
                          if assign.get(a) == assign.get(b))
                pc = hit / len(gold)
                nb = len(set(assign.values()))
                print(f"  k={k:3d} res={res:6.1f} PC={pc:.3f} "
                      f"blocks={nb}", flush=True)
                if best is None or abs(pc - args.target_pc) < best[0]:
                    best = (abs(pc - args.target_pc), k, res, pc, assign)
                if best[0] < 0.02:
                    break
            if best[0] < 0.02:
                break
        _, k, res, pc, assign = best
        print(f"  chosen k={k} resolution={res} PC={pc:.3f}", flush=True)

        aside, bside = defaultdict(list), defaultdict(list)
        for i, c in assign.items():
            (aside if side[i] == "A" else bside)[c].append(i)
        missed = [(a, b) for a, b in gold if assign[a] != assign[b]]
        by_bp = defaultdict(list)
        for a, b in missed:
            by_bp[(assign[a], assign[b])].append((a, b))
        multi = {bp: sorted(v) for bp, v in by_bp.items() if len(v) >= 2}

        rng = random.Random(f"{args.seed}:{ds}:simcse")
        episodes, used = [], set()
        for (s1, s2), pairs in sorted(multi.items()):
            rng.shuffle(pairs)
            seed_pair, targets = pairs[0], pairs[1:]
            a_ids, b_ids = sorted(aside[s1]), sorted(bside[s2])
            episodes.append({"episode_id": f"{ds}_simcseK{k}r{res:g}_{s1}_{s2}",
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
        records = {i: (fa.get(i) or fb.get(i)) for i in used}
        bank = {"dataset": ds,
                "blocker": "SimCSE kNN graph + Louvain communities",
                "K": k, "resolution": res, "PC": pc, "n_gold": len(gold),
                "n_missed": len(missed),
                "dataset_metadata": DATASET_META[ds],
                "method_metadata": (
                    "Blocker: records are embedded with a frozen "
                    "sup-SimCSE BERT encoder over their concatenated "
                    f"field values; a cosine k-NN graph (k={k}) is built "
                    "over all records of both tables and partitioned by "
                    f"Louvain community detection (resolution {res:g}). Each community is a "
                    "block, so every record is in exactly one block. "
                    "Matches get split when the two records fall into "
                    "different communities."),
                "records": records, "episodes": episodes}
        json.dump(bank, open(os.path.join(outdir, f"{ds}.json"), "w"))
        n_t = sum(len(e["targets"]) for e in episodes)
        sp = defaultdict(int)
        for e in episodes:
            sp[e["split"]] += 1
        row = (f"{ds:15s} k={k:3d} res={res:g} PC={pc:.3f} missed={len(missed):5d} "
               f"episodes={len(episodes):4d} targets={n_t:5d} "
               f"splits={dict(sp)} recs={len(records)}")
        print(row, flush=True)
        summary.append(row)
    open(os.path.join(outdir, "SUMMARY.txt"), "w").write(
        "\n".join(summary) + "\n")


if __name__ == "__main__":
    main()
