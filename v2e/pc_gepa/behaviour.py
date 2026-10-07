"""Behavioural tables for the paper, recomputed on the HARVEST.

Two changes from the versions currently in Section VII:

1. Arms are the seed and candidate 7, not candidate 6.
2. Data is the harvest's held-out variants (v1, v2), not pooled training
   rollouts. Pooling was the confound: candidates were rolled out on
   different minibatches, so different cells, datasets and fields, and a
   difference between two arms was partly a difference in what they were
   shown. Here both arms see identical instances.
"""
import json, sys, re, collections

R = sys.argv[1]
STATS = re.compile(r"Group X holds (\d+) records.*?Group Y holds (\d+) records", re.S)


def load_events():
    """tag -> (|X|, |Y|, operation), from the rendered prompt and reply."""
    out = {}
    for l in open(R + "/llm_events.jsonl"):
        try:
            e = json.loads(l)
        except Exception:
            continue
        t = e.get("tag") or ""
        if not t.startswith("eval:"):
            continue
        m = STATS.search(e["messages"][0]["content"])
        if not m:
            continue
        try:
            r = json.loads(e.get("raw_response") or "")
            op = r.get("operation")
        except Exception:
            continue
        out.setdefault(t, (int(m.group(1)), int(m.group(2)), op, e["ts"]))
    return out


def arm_rows(f, heldout_only=True):
    d = json.load(open(R + "/" + f))
    return [r for r in d["rows"] if (r["variant"] != 0 or not heldout_only)]


def damage(rows):
    """Table X: what an edit costs when it goes wrong."""
    per = collections.defaultdict(lambda: dict(n=0, bad=0, dpc=0.0, dc=0.0, gold=0.0))
    for r in rows:
        ds = r["cell"][1]
        # len(gold) recovers absolute pair counts from the pc fractions
        ngold = r["base_hits"] / r["pc_before"] if r["pc_before"] else 0
        s = per[ds]
        for a in r["exemplars"]:
            if a["op"] == "none":
                continue
            s["n"] += 1
            if a["d_pc"] < 0:
                s["bad"] += 1
            s["dpc"] += a["d_pc"]
            s["dc"] += a["d_c"]
            s["gold"] += a["d_pc"] * ngold
    return per


def direction(rows, ev):
    """Table IX: was the cheaper direction chosen?

    Moving k records from a block of size a into one of size b changes
    |C| by k(b-a+k), so the cheap direction is always into the SMALLER
    block: move_to_x is cheaper exactly when X is the smaller one.
    """
    n = cheap = xsmall = 0
    cond = collections.Counter()
    for r in rows:
        for a in r["exemplars"]:
            if a["op"] not in ("move_to_x", "move_to_y"):
                continue
            tag = f"eval:{r['instance_id']}:{a['step']}"
            if tag not in ev:
                continue
            x, y, _, _ = ev[tag]
            if x == y:
                continue
            n += 1
            xs = x < y
            xsmall += xs
            if (a["op"] == "move_to_x") == xs:
                cheap += 1
            cond[("X smaller" if xs else "Y smaller", a["op"])] += 1
    return n, cheap, xsmall, cond


ev = load_events()
print(f"  events with parseable STATS: {len(ev)}\n")
arms = [("seed", "eval_val_seed.json"), ("cand7", "eval_val_best.json")]

print("=" * 74)
print("TABLE X  off-target damage, held-out variants only")
print("=" * 74)
print(f"  {'arm':<7}{'dataset':<16}{'edits':>7}{'damaging':>10}{'gold/edit':>11}{'d|C|/edit':>11}")
for name, f in arms:
    for ds, s in sorted(damage(arm_rows(f)).items()):
        if not s["n"]:
            continue
        print(f"  {name:<7}{ds:<16}{s['n']:>7}{100*s['bad']/s['n']:>9.1f}%"
              f"{s['gold']/s['n']:>11.2f}{s['dc']/s['n']:>11.1f}")
    print()

print("=" * 74)
print("TABLE IX  direction selection against the comparison-cost arithmetic")
print("=" * 74)
for name, f in arms:
    n, cheap, xs, cond = direction(arm_rows(f), ev)
    if not n:
        continue
    best_const = max(xs, n - xs) / n
    print(f"  {name}: directional moves {n}")
    print(f"    chose the cheaper direction : {100*cheap/n:.0f}%")
    print(f"    best constant policy gives  : {100*best_const:.0f}%")
    for k in ("X smaller", "Y smaller"):
        tot = sum(v for (c, _), v in cond.items() if c == k)
        if tot:
            mx = cond[(k, "move_to_x")]
            print(f"    when {k:<10}: move_to_x {100*mx/tot:5.1f}%   "
                  f"move_to_y {100*(tot-mx)/tot:5.1f}%   (n={tot})")
    print()
