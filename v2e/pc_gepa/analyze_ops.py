#!/usr/bin/env python3
"""Operation mix, positional drift and predicate anatomy, per candidate.

Covers backlog items E6 (what each operation is chosen for), E7 (whether
the decision changes as an instance progresses) and E8a (predicate
anatomy for candidate 7). All three read only llm_events.jsonl, so they
cost no judge calls and can run while the harvest holds the GPU.

Attribution to a candidate is structural. A candidate is a template
whose {STATS}, {SEED}, {SAMPLE_X} and {SAMPLE_Y} placeholders are
substituted mid-text, so it is not a literal prefix of the rendered
prompt: the fixed segments are matched in order with the substitutions
treated as wildcards. A prefix test alone would not do -- candidates 0,
1 and 3 share their first 100 characters, as do 6 and 7.

Usage: analyze_ops.py [RUN_DIR] > report.txt
"""
import collections
import json
import re
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v2   # noqa: E402  -- reuse the driver's reply parser verbatim

RUN = sys.argv[1] if len(sys.argv) > 1 else "../runs/gepa/v2e_qwen38"
# Optional second argument: restrict to one split's v0 variant. Without
# it the tables pool every call a candidate ever made, and candidates
# were rolled out on different training minibatches -- different cells,
# datasets and fields -- so a difference in operator mix between two
# candidates is partly a difference in what they were shown. The v0 val
# panel is the one matched sample: the SAME 40 frozen instances, the
# same cells, block pairs and seeds, for every candidate. Comparisons
# across candidates are only like-for-like inside it.
ONLY_SPLIT = sys.argv[2] if len(sys.argv) > 2 else None
OPS = ("move_to_x", "move_to_y", "new_block", "none")
NBUCKET = 5          # positional buckets within an instance, for E7


PLACEHOLDER = re.compile(r"\{(?:STATS|SEED|SAMPLE_X|SAMPLE_Y)\}")


def load_candidates():
    """Compile each candidate template into a matcher for rendered prompts.

    The candidate is a template: {STATS}, {SEED}, {SAMPLE_X} and
    {SAMPLE_Y} are substituted before the call, and they sit in the
    middle of the text, so the candidate is not a literal prefix of the
    prompt. Match the fixed segments in order with the substitutions
    treated as wildcards.

    Candidates 0, 1 and 3 share their first 100 characters and 6 shares
    with 7, so the leading segment alone cannot attribute a call; the
    cheap prefix test only narrows, and the full pattern decides.
    """
    with open(os.path.join(RUN, "candidates.json"), encoding="utf-8") as f:
        cands = json.load(f)
    text = [list(c.values())[0] if isinstance(c, dict) else str(c)
            for c in cands]

    compiled = []
    for i, t in enumerate(text):
        segs = PLACEHOLDER.split(t)
        pat = ".*?".join(re.escape(s) for s in segs)
        compiled.append((i, segs[0], re.compile(pat, re.DOTALL)))
    # Longest fixed prefix first: where one candidate's prefix extends
    # another's, the more specific one must win.
    return sorted(compiled, key=lambda p: -len(p[1]))


def attribute(content, cands):
    for i, prefix, rx in cands:
        if content.startswith(prefix) and rx.match(content):
            return i
    return None


def main():
    cands = load_candidates()
    ncand = len(cands)

    ops = collections.defaultdict(collections.Counter)        # cand -> op
    by_pos = collections.defaultdict(collections.Counter)     # (cand,b) -> op
    fields = collections.defaultdict(collections.Counter)
    operators = collections.defaultdict(collections.Counter)
    nactions = collections.defaultdict(collections.Counter)
    nclause = collections.defaultdict(collections.Counter)
    steps_seen = collections.defaultdict(set)   # (cand,instance) -> steps
    unattributed = 0
    filtered = 0
    malformed = 0
    nomsgs = 0
    unparsed = collections.Counter()
    total = 0

    # Two passes would need 1.3 GB read twice, so bucket positions against
    # a nominal instance length and correct afterwards is not possible --
    # instead record the max step per instance as we go and bucket at the
    # end from a retained per-event list. Keep only what is needed.
    events = []

    path = os.path.join(RUN, "llm_events.jsonl")
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                e = json.loads(line)
            except Exception:
                continue
            tag = e.get("tag") or ""
            if not tag.startswith("v2:"):
                continue
            total += 1
            msgs = e.get("messages") or []
            if not msgs:
                nomsgs += 1
                continue
            ci = attribute(msgs[0].get("content", ""), cands)
            if ci is None:
                unattributed += 1
                continue

            # Two tag shapes, because the id namespaces differ:
            #   val/test  v2:<split>:<blocker>:<dataset>:v<k>:<inst>:<step>
            #   train     v2:train:<blocker>:<dataset>:<inst>:<step>
            # Train specs carry no variant. Parsing every tag as the val
            # shape silently discards all 73,606 train calls.
            parts = tag.split(":")
            try:
                split, blocker, dataset = parts[1], parts[2], parts[3]
                if len(parts) == 7:
                    variant, inst, step = parts[4], parts[5], int(parts[6])
                elif len(parts) == 6:
                    variant, inst, step = "train", parts[4], int(parts[5])
                else:
                    malformed += 1
                    continue
            except (IndexError, ValueError):
                malformed += 1
                continue

            # Use the driver's own parser rather than json.loads. 18,517
            # replies wrap their JSON in prose; the driver recovers those
            # by scanning for the first balanced object carrying an
            # "operation" or "actions" key, and it executed them. Parsing
            # them naively drops them, and the drop rate is strongly
            # candidate-dependent (cand6 2737 against cand8 80), so a
            # naive parse does not merely lose data, it biases the mix.
            if ONLY_SPLIT and (split != ONLY_SPLIT or variant != "v0"):
                filtered += 1
                continue

            actions, _, err = v2.parse_edit(e.get("raw_response") or "")
            if err or not actions:
                unparsed[ci] += 1
                continue

            key = (ci, blocker, dataset, variant, inst)
            steps_seen[key].add(step)
            # MAX_ACTIONS allows up to five actions in one reply. Count
            # how often that is used -- this is the quantity the paper
            # reports, and it is NOT the number of predicate clauses.
            nactions[ci][min(len(actions), v2.MAX_ACTIONS)] += 1
            events.append((ci, key, step, actions[0].get("operation")))

            for a in actions:
                op = a.get("operation")
                if op not in OPS:
                    continue
                ops[ci][op] += 1
                pred = a.get("predicate")
                if isinstance(pred, list):
                    # A predicate is a conjunction of field conditions.
                    # Its length is a separate axis from action count.
                    nclause[ci][min(len(pred), 5)] += 1
                    for rule in pred:
                        if isinstance(rule, dict):
                            fields[ci][rule.get("field")] += 1
                            operators[ci][rule.get("operator")] += 1

    # E7: bucket each call by its relative position in its own instance.
    span = {k: max(v) for k, v in steps_seen.items() if v}
    for ci, key, step, op in events:
        hi = span.get(key, 0)
        b = 0 if hi == 0 else min(NBUCKET - 1, step * NBUCKET // (hi + 1))
        by_pos[(ci, b)][op] += 1

    out = print
    if ONLY_SPLIT:
        out(f"RESTRICTED to split={ONLY_SPLIT} variant=v0 -- the matched")
        out("panel every candidate was scored on. Cross-candidate")
        out("comparisons below are like-for-like.")
        out("")
    out(f"events tagged v2: {total}")
    if ONLY_SPLIT:
        out(f"  outside the matched panel  : {filtered}")
    out(f"  attributed to a candidate : {len(events)}")
    out(f"  response unparsable       : {sum(unparsed.values())}")
    out(f"  no candidate matched      : {unattributed}")
    out(f"  malformed tag / no msgs   : {malformed + nomsgs}")
    out("")
    out("  'no candidate matched' is expected, not noise: eight proposals")
    out("  were rolled out on the minibatch and then lost the paired gate,")
    out("  so their text was never written to candidates.json. Those calls")
    out("  are real judge decisions under an instruction we cannot name.")
    out("")

    out("=" * 72)
    out("E6  operation mix per candidate")
    out("=" * 72)
    out(f"{'cand':<6}{'n':>8}  " + "".join(f"{o:>12}" for o in OPS)
        + f"{'unparsed':>10}")
    for ci in range(ncand):
        c = ops[ci]
        n = sum(c.values())
        if not n:
            continue
        out(f"cand{ci:<2}{n:>8}  "
            + "".join(f"{100 * c[o] / n:>11.1f}%" for o in OPS)
            + f"{unparsed[ci]:>10}")

    out("")
    out("=" * 72)
    out(f"E7  operation mix by position within the instance ({NBUCKET} buckets)")
    out("=" * 72)
    for ci in range(ncand):
        if not sum(ops[ci].values()):
            continue
        out(f"cand{ci}")
        out(f"  {'bucket':<8}{'n':>7}  " + "".join(f"{o:>12}" for o in OPS))
        for b in range(NBUCKET):
            c = by_pos[(ci, b)]
            n = sum(c.values())
            if not n:
                continue
            lo, hi = 100 * b // NBUCKET, 100 * (b + 1) // NBUCKET
            out(f"  {lo:>3}-{hi:<4}{n:>7}  "
                + "".join(f"{100 * c[o] / n:>11.1f}%" for o in OPS))
        out("")

    out("=" * 72)
    out("E8a  predicate anatomy")
    out("=" * 72)
    out("multi-action = one reply carrying >1 {operation, predicate} action")
    out("              (MAX_ACTIONS=%d). multi-clause = one predicate whose"
        % v2.MAX_ACTIONS)
    out("              condition is a conjunction of >1 field tests.")
    out("              These are different axes; do not conflate them.")
    out("")
    for ci in range(ncand):
        na = nactions[ci]
        n = sum(na.values())
        if not n:
            continue
        multi_a = sum(v for k, v in na.items() if k > 1)
        nc = nclause[ci]
        nct = sum(nc.values())
        multi_c = sum(v for k, v in nc.items() if k > 1)
        out(f"cand{ci}  replies={n}  actions/reply>1={multi_a} "
            f"({100 * multi_a / n:.2f}%)   "
            f"clauses/predicate>1={multi_c} "
            f"({100 * multi_c / max(1, nct):.2f}%)")
        out("   operators: " + ", ".join(
            f"{k}={100 * v / max(1, sum(operators[ci].values())):.1f}%"
            for k, v in operators[ci].most_common(6)))
        out("   fields:    " + ", ".join(
            f"{k}={100 * v / max(1, sum(fields[ci].values())):.1f}%"
            for k, v in fields[ci].most_common(6)))
        out("")


if __name__ == "__main__":
    main()
