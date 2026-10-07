"""How do missed gold pairs cluster into block-pairs? Drives episode design."""
import csv
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, "src"))
import loaders  # noqa: E402

ta, tb, gold, _ = loaders.load_amazon_google()

for K in (10, 12, 15, 20):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                        "runs", "lambdafold", "amazon-google", f"K{K}",
                        "assignments.csv")
    A, B = {}, {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            (A if row["side"] == "A" else B)[row["record_id"]] = row["block"]
    bsz = Counter()
    for s in list(A.values()) + list(B.values()):
        bsz[s] += 1
    missed = [(a, b) for a, b in gold if A[a] != B[b]]
    by_bp = defaultdict(list)
    for a, b in missed:
        by_bp[(A[a], B[b])].append((a, b))
    cnt = Counter(len(v) for v in by_bp.values())
    multi = {bp: v for bp, v in by_bp.items() if len(v) >= 2}
    pairs_in_multi = sum(len(v) for v in multi.values())
    # episode sizes: records in the two blocks involved
    szs = sorted(bsz[bp[0]] + bsz[bp[1]] for bp in multi)
    med = szs[len(szs) // 2] if szs else 0
    small = sum(1 for s in szs if s <= 80)
    print(f"K={K}: missed={len(missed)} block-pairs={len(by_bp)} "
          f"multi(>=2 missed)={len(multi)} pairs_in_multi={pairs_in_multi} "
          f"dist={dict(sorted(cnt.items()))}")
    print(f"      episode total-records: median={med} "
          f"max={szs[-1] if szs else 0} <=80recs={small}/{len(szs)}")
