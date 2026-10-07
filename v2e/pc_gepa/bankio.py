"""Bank schema v2 + instance derivation.

A bank cell records the raw material, not pre-baked episodes:

  {cell: {blocker, dataset, er_mode, K, PC, n_gold, ...},
   records: {rid: {field: value}},
   block_pairs: [{x_sig, y_sig, a_ids, b_ids, failed: [[a, b], ...]}]}

`failed` is every gold pair the blocker separated into this block pair
(Case 2 has one, Case 3 has more). Instances are derived from this:

  derive(bank, policy) -> [{a_ids, b_ids, seed_pair, targets, case}]

Two seed policies:
  "per_pair"  -- one randomly chosen seed in every block pair; targets
                 are that pair's remaining failures. This is the
                 cell-level isolated test.
  "fraction"  -- a random fraction p of ALL failures in the cell is
                 known. Block pairs holding at least one known failure
                 are visited; the instance seed is a random known one
                 and the targets are the failures still unknown. Models
                 partial knowledge, and produces zero-target instances
                 naturally (correct answer: propose nothing).

Both take a draw index so a measurement can be repeated under different
random seed sets.
"""
import random


def write_bank(path, cell, records, by_block_pair, members_x, members_y):
    import json
    bps = []
    for (s1, s2), failed in sorted(by_block_pair.items()):
        bps.append({"x_sig": s1, "y_sig": s2,
                    "a_ids": members_x[s1], "b_ids": members_y[s2],
                    "failed": [list(f) for f in sorted(failed)]})
    json.dump({"schema": 2, **cell, "records": records,
               "block_pairs": bps}, open(path, "w"))
    return bps


def derive(bank, policy="per_pair", p=None, draw=0, min_case=2):
    """Return instances. min_case=2 includes orphan (Case 2) block
    pairs, whose instances have no targets; min_case=3 excludes them."""
    key = f"{bank['blocker']}:{bank['dataset']}:{policy}:{p}:{draw}"
    rng = random.Random(key)
    out = []
    if policy == "per_pair":
        for bp in bank["block_pairs"]:
            failed = [tuple(f) for f in bp["failed"]]
            # Case 2 == exactly one separated pair, Case 3 == two or more
            if len(failed) < (1 if min_case == 2 else 2):
                continue
            seed = failed[rng.randrange(len(failed))]
            targets = [f for f in failed if f != seed]
            out.append(_inst(bank, bp, seed, targets))
        return out
    if policy != "fraction" or p is None:
        raise ValueError("policy must be per_pair, or fraction with p")
    everything = [(bp_i, tuple(f))
                  for bp_i, bp in enumerate(bank["block_pairs"])
                  for f in bp["failed"]]
    k = max(1, int(round(p * len(everything))))
    known = set(rng.sample(everything, min(k, len(everything))))
    by_bp = {}
    for bp_i, f in known:
        by_bp.setdefault(bp_i, []).append(f)
    for bp_i, seeds in sorted(by_bp.items()):
        bp = bank["block_pairs"][bp_i]
        seed = seeds[rng.randrange(len(seeds))]
        known_here = {f for f in seeds}
        targets = [tuple(f) for f in bp["failed"]
                   if tuple(f) not in known_here]
        out.append(_inst(bank, bp, seed, targets))
    return out


def _inst(bank, bp, seed, targets):
    return {"episode_id": (f"{bank['dataset']}_{bank['blocker']}_"
                           f"{bp['x_sig']}_{bp['y_sig']}"),
            "dataset": bank["dataset"], "blocker": bank["blocker"],
            "a_ids": bp["a_ids"], "b_ids": bp["b_ids"],
            "seed_pair": list(seed),
            "targets": [list(t) for t in targets],
            # every separated pair in this block pair, so a caller can
            # redraw which one is revealed without rebuilding the bank
            "failed": [list(f) for f in bp["failed"]],
            "case": 2 if len(bp["failed"]) == 1 else 3}
