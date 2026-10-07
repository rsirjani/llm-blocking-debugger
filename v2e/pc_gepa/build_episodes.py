"""Build debug episodes from a lambda-fold run.

Episode = directional block-pair (s1, s2) with >= 2 missed gold pairs
(a in A-side of s1, b in B-side of s2). One missed pair becomes the SEED
(shown to the LLM); the rest are TARGETS (scored, never shown).

Gold-blind candidate geometry: top-M cross pairs (A-side s1 x B-side s2)
by char-3gram TF-IDF cosine, fitted on the union of both tables' texts.
Ceiling@M = fraction of targets whose pair appears in the top-M list —
the max recall any prompt can reach in 'topm' mode.

Split: episodes shuffled with seed 0 -> train/val/test = 40/29/30.

Writes episodes.json + a ceiling report. Gold appears ONLY in the
seed fields and the (never-rendered) target fields.
"""
import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, "src"))
import loaders  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def rec_text(row, cols):
    return " ".join(str(row[c]) for c in cols
                    if c in row and str(row[c]).strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=15)
    ap.add_argument("--topm", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ta, tb, gold, _ = loaders.load_amazon_google()
    ta = ta.set_index("id", drop=False)
    tb = tb.set_index("id", drop=False)
    cols_a = ["title", "manufacturer"]
    cols_b = ["name", "manufacturer"]
    text_a = {r["id"]: rec_text(r, cols_a) for _, r in ta.iterrows()}
    text_b = {r["id"]: rec_text(r, cols_b) for _, r in tb.iterrows()}

    run = os.path.join(HERE, os.pardir, "runs", "lambdafold",
                       "amazon-google", f"K{args.K}")
    A, B = {}, {}
    with open(os.path.join(run, "assignments.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            (A if row["side"] == "A" else B)[row["record_id"]] = row["block"]

    aside = defaultdict(list)   # sig -> A-side record ids
    bside = defaultdict(list)
    for r, s in A.items():
        aside[s].append(r)
    for r, s in B.items():
        bside[s].append(r)

    missed = [(a, b) for a, b in gold if A[a] != B[b]]
    by_bp = defaultdict(list)
    for a, b in missed:
        by_bp[(A[a], B[b])].append((a, b))
    multi = {bp: sorted(v) for bp, v in by_bp.items() if len(v) >= 2}

    vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 3))
    all_ids = list(text_a) + list(text_b)
    all_txt = [text_a.get(i) or text_b.get(i) for i in all_ids]
    vec.fit(all_txt)

    rng = random.Random(args.seed)
    episodes = []
    ceil_hits = ceil_total = 0
    for (s1, s2), pairs in sorted(multi.items()):
        rng.shuffle(pairs)
        seed_pair, targets = pairs[0], pairs[1:]
        ids1 = sorted(aside[s1])
        ids2 = sorted(bside[s2])
        m1 = vec.transform([text_a[i] for i in ids1])
        m2 = vec.transform([text_b[i] for i in ids2])
        sim = cosine_similarity(m1, m2)
        flat = [(float(sim[i, j]), ids1[i], ids2[j])
                for i in range(len(ids1)) for j in range(len(ids2))]
        flat.sort(key=lambda t: (-t[0], t[1], t[2]))
        top = flat[:args.topm]
        top_set = {(a, b) for _, a, b in top}
        t_in = sum(1 for t in targets if t in top_set)
        ceil_hits += t_in
        ceil_total += len(targets)
        episodes.append({
            "episode_id": f"K{args.K}_{s1}_{s2}",
            "block1_sig": s1, "block2_sig": s2,
            "block1_a_ids": ids1, "block2_b_ids": ids2,
            "n_block1_total": sum(1 for x in A.values() if x == s1)
                              + sum(1 for x in B.values() if x == s1),
            "n_block2_total": sum(1 for x in A.values() if x == s2)
                              + sum(1 for x in B.values() if x == s2),
            "seed_pair": list(seed_pair),
            "targets": [list(t) for t in targets],
            "top_pairs": [{"a": a, "b": b, "sim": round(s, 4)}
                          for s, a, b in top],
            "ceiling_topm": t_in / len(targets) if targets else None,
        })

    rng.shuffle(episodes)
    n = len(episodes)
    n_tr, n_va = int(n * 0.40), int(n * 0.30)
    for i, ep in enumerate(episodes):
        ep["split"] = ("train" if i < n_tr else
                       "val" if i < n_tr + n_va else "test")

    out = {
        "dataset": "amazon-google", "blocker": "blocklib lambda-fold Lambda=1",
        "K": args.K, "topm": args.topm, "split_seed": args.seed,
        "record_texts_a": text_a, "record_texts_b": text_b,
        "episodes": episodes,
        "method_metadata": (
            "Blocker: Lambda-fold LSH (Karapiperis & Verykios 2015) with "
            "Lambda=1, as implemented in blocklib. Each record's text "
            "(title/name + manufacturer) is tokenized into character "
            f"bigrams, hashed into a 2000-bit Bloom filter, and K={args.K} "
            "randomly sampled bit positions form the record's block "
            "signature. Records with identical signatures share a block; "
            "every record is in exactly one block. Failure mode: two "
            "matching records disagree on any sampled bit (spelling "
            "variants, extra tokens, different word order, abbreviations) "
            "-> different blocks."),
    }
    path = os.path.join(run, "episodes.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f)
    by_split = defaultdict(int)
    tgt = defaultdict(int)
    for ep in episodes:
        by_split[ep["split"]] += 1
        tgt[ep["split"]] += len(ep["targets"])
    print(f"episodes={n} splits={dict(by_split)} targets/split={dict(tgt)}")
    print(f"gold-blind ceiling@{args.topm}: {ceil_hits}/{ceil_total} "
          f"= {ceil_hits/ceil_total:.3f}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
