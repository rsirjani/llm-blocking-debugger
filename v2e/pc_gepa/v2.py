"""V2: the LLM repairs the partition, and only the prompt is optimized.

Differences from V1 (run6/run7):

* THE OUTPUT IS AN EDIT, NOT A PAIR LIST. For each seed the judge
  returns one of: move records matching a predicate from one block to
  the other, put matching records into a new block, or do nothing.
  There is no blanket merge -- every change must be stated as a
  predicate, so it is a rule over the whole block rather than a
  gesture at the sample. The harness applies it to a LIVE partition
  and asserts P1 after every step, so the deliverable stays a
  partition and a bad edit can lower pair completeness rather than
  merely costing candidate pairs.

* AN INSTANCE IS A WHOLE-CELL REPLAY. Sample a budget of Case-3 block
  pairs from one (blocker, dataset) cell, one seed each, process them
  in order against the accumulating partition, then score the cell as
  a whole. Later steps see the blocks earlier steps produced; if an
  earlier edit already reunited a later seed's records, that step is
  skipped as already solved.

* RETRIEVAL IS FIXED AND UNIFORM, AND NOTHING IS EVER TRUNCATED.
  Records shown per block are a uniform random sample, size strictly
  proportional to the block, and every record is rendered with all of
  its fields at full length. Block pairs whose full sample would not
  fit the context are excluded up front, and a cell is dropped
  entirely if more than 10% of its Case-3 pairs are such pairs --
  keeping only a cell's small block pairs would measure an easier
  problem than the cell poses. Nothing about which records are shown is optimized,
  so the measurement is of the judge's ability rather than of a
  retriever that hands it the answers.

* WHAT IS OPTIMIZED IS THE WHOLE MESSAGE TEMPLATE. The single evolved
  component is a template with {SEED}, {SAMPLE_X}, {SAMPLE_Y} and
  optional {STATS} slots. The harness fills the slots with the fixed
  uniform sample and writes no prose of its own.

* SCORE PRICES BOTH AXES.
      score = (net same-entity pairs reunited across the WHOLE cell
               / separated pairs in the block pairs visited)
              - LAMBDA * (d|C| / |C|)
  with LAMBDA = 5. The numerator is global, so an edit that drags a
  record out of a block it belonged in is debited for every pair it
  leaves behind; the denominator is local, so the signal is not
  diluted by the cell's untouched majority. F(PC, RR) was rejected as the
  objective: RR sits at 0.88-0.99 here, so it barely moves and the
  harmonic mean is effectively PC alone.

* ALL RANDOMNESS IS A FUNCTION OF THE INSTANCE ID. Re-running the same
  instance reproduces the same seeds, samples and order, so GEPA's
  accept/reject is a paired comparison. Training instances are drawn
  fresh every iteration (the space is effectively infinite), while
  validation and test are frozen panels; test additionally has K
  frozen variants so the spread over which-failure-you-knew is
  measurable, and every arm is evaluated on identical variants.

SECURITY: no LLM-written code is executed in V2. The judge returns
structured JSON edits which the harness interprets.
"""
import argparse
import difflib
import json
import math
import os
import random
import re
import subprocess
import time
from collections import Counter, defaultdict

import harness
import mask

HERE = os.path.dirname(os.path.abspath(__file__))
# Overridable so E5 can evaluate blockers at operating points other
# than the one the campaign was built on, without disturbing the bank
# every other result in the paper is computed from.
BANK_ROOT = os.path.join(HERE, os.pardir, "runs", "bank2")
# Judge prompts run ~3k tokens on average and up to ~13k. Context is
# not a throttle to be tuned down: 32k slots x 8 parallel workers once
# reserved 45 GB of KV cache on a 32 GB card and spilled to system
# memory, costing ~30x, so slots-per-server is the knob instead.
# Sized so that nothing anywhere is truncated: every admitted block
# pair shows its full proportional sample, and every record is rendered
# with all of its fields at full length. 18432 is the smallest context
# for which no block pair of the 14 admitted cells has to be excluded.
MAX_CTX = 18432
# Chars of sample the prompt can carry: the context, at a conservative
# 3.2 chars/token, minus room for the template and the reply. The sample
# is never trimmed to this — block pairs that would exceed it are
# excluded from the campaign instead (see fits_context).
BUDGET_CHARS = int(MAX_CTX * 3.2) - 1500
# Training signal is local: the fraction of the separated pairs in the
# block pairs this instance actually visited that end up reunited,
# minus the cost of the comparisons the edits created. Scoring the
# whole cell's pair completeness instead diluted every instance to
# ~0.003 -- eight block pairs are 1.5-10% of a cell -- so the score
# measured which pairs the dice drew rather than how well they were
# repaired. Reported test numbers stay absolute (dPC, d|C|, counts).
LAMBDA = 5.0
# No fixed per-block ceiling: the sample is proportional to block size
# and bounded only by what fits the context. A constant cap would show
# the same 60 records of a 100-record block and of a 5,000-record one,
# so the evidence would be a wildly different fraction of the thing the
# rule has to generalise over.
SAMPLE_FRACTION = 0.35   # of each block, before the context bound
# Retrieval is uniform by design: nothing about which records are shown
# is optimized, so a result measures the judge rather than a retriever
# that hands it the answer. SAMPLING="seed_similar" puts that choice on
# trial by showing the k records most similar to the disclosed pair
# instead of k at random. Same k, same blocks, same context cost --
# only the selection differs, so the comparison is clean.
SAMPLING = "uniform"
# Number of separated gold pairs disclosed per block pair, and how the
# set is chosen. A Case-3 block pair has at least two by definition, so
# two can be shown without changing which block pairs qualify. Every
# disclosed pair is excluded from credit, so showing more shrinks the
# denominator rather than handing out free hits.
N_SEEDS = 1
SEED_PICK = "random"        # random | similar | different


def _pair_grams(cell, pair):
    recs = cell["records"]
    return (_grams(render_record(pair[0], recs[pair[0]], 1))
            | _grams(render_record(pair[1], recs[pair[1]], 1)))


def _pick_seeds(cell, failed, n, how, rng):
    """Choose n disclosed pairs from the separated gold pairs available.

    "similar" takes the two most alike, so the judge sees a consistent
    pattern; "different" takes the two least alike, so it sees the
    failure's spread. Both are drawn from the same pool, so the only
    thing that varies between the arms is which examples are shown.
    """
    if n <= 1 or len(failed) < 2:
        return [failed[rng.randrange(len(failed))]]
    if how == "random":
        return rng.sample(failed, min(n, len(failed)))
    # Cap the candidate pool: this is O(m^2) and some block pairs carry
    # hundreds of separated pairs.
    pool = failed if len(failed) <= 24 else rng.sample(failed, 24)
    g = {p: _pair_grams(cell, p) for p in pool}
    best = None
    for i, a in enumerate(pool):
        for b in pool[i + 1:]:
            u = len(g[a] | g[b])
            j = (len(g[a] & g[b]) / u) if u else 0.0
            key = -j if how == "similar" else j
            if best is None or key < best[0]:
                best = (key, [a, b])
    return best[1] if best else rng.sample(failed, 2)


def _grams(txt, n=3):
    t = " " + " ".join(txt.lower().split()) + " "
    return {t[i:i + n] for i in range(max(0, len(t) - n + 1))}


def _similar_to(cell, ids, anchor_ids, k):
    """The k records in ids whose rendered text most overlaps the anchor.

    Character 3-gram Jaccard: no dependencies, and it keys on the same
    surface fragments the predicates are written over, so it ranks by the
    evidence the judge would actually use.
    """
    recs = cell["records"]
    anchor = set()
    for a in anchor_ids:
        anchor |= _grams(render_record(a, recs[a], 1))
    if not anchor:
        return None
    scored = []
    for r in ids:
        g = _grams(render_record(r, recs[r], 1))
        scored.append(((len(g & anchor) / len(g | anchor)) if g else 0.0, r))
    # -score then id: deterministic, independent of input order
    scored.sort(key=lambda p: (-p[0], str(p[1])))
    return [r for _, r in scored[:k]]
# Records are rendered whole. Capping fields (6) and field length (120
# chars) cut 55% and 23% of records respectively, and a field the judge
# never sees is a field it cannot write a predicate on. Showing
# everything costs 1.14x the prompt.

# Every operation that changes anything carries a predicate, so it
# is a rule over the whole block rather than a gesture at the
# sample. There is no blanket merge: to unite two groups the
# model must write a condition that characterises the records it
# wants united, and the feedback reports how many of the block it
# actually matched.
OPS = ("move_to_x", "move_to_y", "new_block", "none")
# gte/lte/between coerce both sides to a number, so they can express a
# threshold or a band. Substring operators cannot: dispersion in a field
# like year or price is naturally a range, and without one the only way
# to characterise a subgroup is a token it happens to share. This is the
# suspected reason new_block sat at 1 use in 2,441 actions.
PRED_OPS = ("equals", "contains", "not_contains", "regex",
            "is_null", "is_not_null", "gte", "lte", "between")
# Data is dirty: "1993.", "(1994).", "$12.99", "pages 148-156,". Pull
# the first number out rather than demanding a clean field.
_NUMRE = re.compile(r"-?\d+(?:\.\d+)?")
# One reply may carry several rules, applied in order against the live
# grouping, so a later rule sees what the earlier ones did. Capped so a
# single reply cannot reorganise everything in one blind step.
MAX_ACTIONS = 5
# E6-A. The judge picks which operation to apply, so the operations it
# chose are not a random assignment and their observed payoffs cannot be
# compared: new_block looks weak largely because it is reserved for the
# cases where moving would not have worked. To get a treatment effect we
# hold the block pair, the predicate and the board fixed and apply the
# SAME predicate under every operation on a throwaway copy of the
# partition, recording what each would have bought. Off by default: it
# costs four extra scorings per step, which is real time on cora.
COUNTERFACTUAL = False
# E6-B. The counterfactual of E6-A is off-policy: it re-applies a
# predicate that was written with a different operation in mind. To ask
# what an operation is worth when the judge writes FOR it, the operation
# has to be fixed by the experiment rather than chosen by the model. The
# prompt states the operation and the harness enforces it, so a reply
# that names a different one is still executed as the assigned
# treatment; how often that happens is reported as a compliance rate
# rather than silently repaired.
FORCE_OP = None
# Per-endpoint share of the server's slots (OLLAMA_NUM_PARALLEL=24,
# reached over three tunnels, so 8 each = 24 workers). Raised from 3
# because the cards sat at ~33% utilisation with 9 workers: per-call
# latency was fine (p50 37s) and the run was simply not issuing enough
# calls at once. One model copy leaves ~53 GB of VRAM free, so slots are
# no longer rationed by memory the way they were with three copies.
SLOTS_PER_SERVER = 8

SEED_TEMPLATE = """Two groups of records, group X and group Y, are shown \
below. One pair of records, one from each group, is known to describe \
the same real-world entity, yet the two records were placed in \
different groups.

{STATS}

Known same-entity pair:
{SEED}

Group X sample:
{SAMPLE_X}

Group Y sample:
{SAMPLE_Y}

Decide what change to the grouping would place records that describe \
the same entity together. State the change as a condition on record \
fields, not as a list of the records you were shown: the condition is \
applied to every record of the group, including the many that are not \
shown here. Answer with a single JSON object:
{"operation": "move_to_x" | "move_to_y" | "new_block" | "none", \
"predicate": [{"field": "<field name>", "operator": "equals" | \
"contains" | "not_contains" | "regex" | "is_null" | "is_not_null" | \
"gte" | "lte" | "between", "value": "<text>"}], "reason": \
"<one sentence>"}
"gte" and "lte" compare the first number found in the field against \
the value; "between" takes "value": [low, high] and is true when the \
number falls in that range, endpoints included.
You may give several rules at once, up to five, as \
{"actions": [{...}, {...}], "reason": "<one sentence>"}; they are \
applied in the order written, and each one acts on the grouping the \
previous rules left behind.
"move_to_x" moves every record of group Y that satisfies the condition \
into group X; "move_to_y" is the mirror. "new_block" takes the records \
of either group that satisfy the condition into a group of their own. \
"none" leaves the grouping unchanged and needs no condition. Conditions \
within one rule are combined with and."""


# ---------------------------------------------------------- partition

def load_cell(blocker, dataset):
    path = os.path.join(BANK_ROOT, blocker, f"{dataset}.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def candidate_count(assignment):
    """|C| = within-block pairs of the partition (R = R_A u R_B, so
    every within-block pair is a comparison the matcher would make)."""
    sizes = Counter(assignment.values())
    return sum(n * (n - 1) // 2 for n in sizes.values())


def pair_completeness(assignment, gold):
    hit = sum(1 for a, b in gold if assignment[a] == assignment[b])
    return hit / len(gold), hit


def sample_cost(cell, ids):
    """Chars the full proportional sample of one block would render to."""
    recs = cell["records"]
    k = max(1, int(round(SAMPLE_FRACTION * len(ids))))
    total = sum(len(render_record(r, recs[r], 1)) + 1 for r in ids)
    return total * k / len(ids) if ids else 0.0


def fits_context(cell, x_ids, y_ids, budget=BUDGET_CHARS):
    return sample_cost(cell, x_ids) + sample_cost(cell, y_ids) <= budget


def usable_block_pairs(cell, budget=BUDGET_CHARS):
    """Case-3 block pairs whose full sample fits the context.

    Excluding them here, once, is what lets the sample stay exactly
    proportional everywhere else. The pairs this drops are the giant
    ones, so the exclusion is reported per cell rather than buried."""
    members = block_members(cell["assignment"])
    keep, dropped = [], []
    for bp in cell["block_pairs"]:
        if len(bp["failed"]) < 2:
            continue
        x = members.get(bp["x_sig"], [])
        y = members.get(bp["y_sig"], [])
        if not x or not y:
            continue
        (keep if fits_context(cell, x, y, budget) else dropped).append(bp)
    return keep, dropped


def block_members(assignment):
    m = defaultdict(list)
    for rid, sig in assignment.items():
        m[sig].append(rid)
    return m


# ----------------------------------------------------------- instance

def make_instance(cell, instance_id, n_pairs, order):
    """A whole-cell replay: which block pairs, which seed in each, and
    in what order. Every draw comes from the instance id, so re-running
    the same instance reproduces it exactly."""
    rng = random.Random(f"instance:{instance_id}")
    case3 = cell["usable_block_pairs"]
    if not case3:
        return None
    chosen = (case3 if len(case3) <= n_pairs
              else rng.sample(case3, n_pairs))
    steps = []
    for bp in chosen:
        failed = [tuple(f) for f in bp["failed"]]
        seeds = _pick_seeds(cell, failed, N_SEEDS, SEED_PICK, rng)
        seed = seeds[0]
        steps.append({"x_sig": bp["x_sig"], "y_sig": bp["y_sig"],
                      "seed": list(seed),
                      "seeds": [list(p) for p in seeds],
                      "n_failed": len(failed),
                      "failed": [list(f) for f in failed]})
    if order == "random":
        rng.shuffle(steps)
    elif order == "small_first":
        sizes = Counter(cell["assignment"].values())
        steps.sort(key=lambda s: sizes[s["x_sig"]] + sizes[s["y_sig"]])
    elif order == "large_first":
        sizes = Counter(cell["assignment"].values())
        steps.sort(key=lambda s: -(sizes[s["x_sig"]] + sizes[s["y_sig"]]))
    return {"instance_id": instance_id, "cell": [cell["blocker"],
                                                 cell["dataset"]],
            "steps": steps, "order": order}


def sample_block(members, k, rng):
    return members if len(members) <= k else rng.sample(members, k)


def render_record(rid, fields, n):
    body = " | ".join(f"{k}: {v}" for k, v in fields.items())
    return f"{n}. {body}"


def fill_template(template, cell, x_ids, y_ids, seed, rng, budget):
    """Fixed, uniform retrieval: a random sample of each block, sized
    proportionally to the blocks but capped so the message fits."""
    recs = cell["records"]
    pairs = seed if seed and isinstance(seed[0], (list, tuple)) else [seed]
    kx = max(1, int(round(SAMPLE_FRACTION * len(x_ids))))
    ky = max(1, int(round(SAMPLE_FRACTION * len(y_ids))))
    if SAMPLING == "seed_similar" and seed:
        anchor = [r for p in pairs for r in p]
        sx = _similar_to(cell, x_ids, anchor, kx) or sample_block(x_ids, kx, rng)
        sy = _similar_to(cell, y_ids, anchor, ky) or sample_block(y_ids, ky, rng)
    else:
        sx = sample_block(x_ids, kx, rng)
        sy = sample_block(y_ids, ky, rng)
    # The sample is never trimmed to fit. Trimming would silently break
    # the one thing the retrieval promises — that what the model sees is
    # a fixed fraction of each block — and it would bite hardest exactly
    # on the largest blocks, biasing the campaign toward easy ones. Block
    # pairs whose full sample cannot fit are excluded up front by
    # fits_context(), so this is an assertion, not a fallback.
    def cost(ids):
        return sum(len(render_record(r, recs[r], 1)) + 1 for r in ids)
    if cost(sx) + cost(sy) > budget:
        raise RuntimeError(
            f"sample of {len(sx)}+{len(sy)} records needs "
            f"{cost(sx) + cost(sy)} chars > budget {budget}; this block "
            "pair should have been excluded by fits_context()")
    stats = (f"Group X holds {len(x_ids)} records, of which "
             f"{len(sx)} are shown. Group Y holds {len(y_ids)} "
             f"records, of which {len(sy)} are shown.")
    seed_txt = "\n".join(
        "  X: " + render_record(p[0], recs[p[0]], 0)[3:]
        + "\n  Y: " + render_record(p[1], recs[p[1]], 0)[3:]
        for p in pairs)
    sx_txt = "\n".join(render_record(r, recs[r], i)
                       for i, r in enumerate(sx, 1))
    sy_txt = "\n".join(render_record(r, recs[r], i)
                       for i, r in enumerate(sy, 1))
    msg = (template.replace("{STATS}", stats)
                   .replace("{SEED}", seed_txt)
                   .replace("{SAMPLE_X}", sx_txt)
                   .replace("{SAMPLE_Y}", sy_txt))
    return msg, sx, sy


# --------------------------------------------------------------- edit

_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def parse_edit(reply):
    """-> (actions, reason, error). Accepts a list of rules under
    "actions", or a single rule written inline."""
    if not reply or not reply.strip():
        return None, "", "the model returned nothing"
    obj = None
    for m in re.finditer(r"\{", reply):
        chunk = reply[m.start():]
        for end in range(len(chunk), 0, -1):
            if chunk[end - 1] != "}":
                continue
            try:
                cand = json.loads(chunk[:end])
            except Exception:  # noqa: BLE001
                continue
            if isinstance(cand, dict) and ("actions" in cand
                                           or "operation" in cand):
                obj = cand
                break
        if obj is not None:
            break
    if obj is None:
        return None, "", ('no JSON object with an "actions" list or an '
                          '"operation" key was found (the reply began: '
                          f"{reply[:120]!r})")
    raw = obj.get("actions")
    if raw is None:
        raw = [obj]
    if not isinstance(raw, list) or not raw:
        return None, "", '"actions" must be a non-empty list'
    if len(raw) > MAX_ACTIONS:
        return None, "", (f"at most {MAX_ACTIONS} rules per reply, got "
                          f"{len(raw)}")
    actions = []
    for a in raw:
        if not isinstance(a, dict):
            return None, "", "each rule must be an object"
        op = a.get("operation")
        if op not in OPS:
            return None, "", (f'"operation" must be one of {OPS}, got '
                              f"{op!r}")
        pred = a.get("predicate") or []
        if op != "none":
            if not isinstance(pred, list) or not pred:
                return None, "", (f'"{op}" needs a non-empty "predicate" '
                                  "list; every change must be stated as "
                                  "a condition that can be applied to "
                                  "records that were not shown")
            for c in pred:
                if not isinstance(c, dict) or "field" not in c:
                    return None, "", 'each condition needs a "field"'
                if c.get("operator") not in PRED_OPS:
                    return None, "", (f'"operator" must be one of '
                                      f'{PRED_OPS}, got '
                                      f'{c.get("operator")!r}')
        actions.append({"operation": op, "predicate": pred})
    return actions, str(obj.get("reason", ""))[:300], None


def match_predicate(pred, fields):
    for c in pred:
        val = str(fields.get(c["field"], ""))
        v, cv = val.lower(), str(c.get("value", "")).lower()
        op = c["operator"]
        if op == "equals":
            ok = v == cv
        elif op == "contains":
            ok = cv in v
        elif op == "not_contains":
            ok = cv not in v
        elif op == "regex":
            try:
                ok = re.search(str(c.get("value", "")), val, re.I) is not None
            except re.error:
                return False
        elif op == "is_null":
            ok = not v.strip()
        elif op == "is_not_null":
            ok = bool(v.strip())
        elif op in ("gte", "lte", "between"):
            m = _NUMRE.search(val)
            if not m:
                return False
            try:
                x = float(m.group())
                if op == "between":
                    lo, hi = (c.get("value") or [None, None])[:2]
                    ok = float(lo) <= x <= float(hi)
                else:
                    ok = x >= float(cv) if op == "gte" else x <= float(cv)
            except (TypeError, ValueError):
                return False
        else:
            return False
        if not ok:
            return False
    return True


def apply_action(state, cell, x_sig, y_sig, edit, step_no,
                 action_no):
    """Mutate the live partition. Returns (n_moved, note)."""
    recs = cell["records"]
    members = block_members(state)
    xs, ys = members.get(x_sig, []), members.get(y_sig, [])
    op = edit["operation"]
    if op == "none" or not xs or not ys:
        return 0, "no change"
    pred = edit["predicate"]
    if op == "move_to_x":
        sel = [r for r in ys if match_predicate(pred, recs[r])]
        for r in sel:
            state[r] = x_sig
        return len(sel), (f"the condition matched {len(sel)} of the "
                          f"{len(ys)} records in group Y, which moved "
                          "into group X")
    if op == "move_to_y":
        sel = [r for r in xs if match_predicate(pred, recs[r])]
        for r in sel:
            state[r] = y_sig
        return len(sel), (f"the condition matched {len(sel)} of the "
                          f"{len(xs)} records in group X, which moved "
                          "into group Y")
    if op == "new_block":
        sel = ([r for r in xs if match_predicate(pred, recs[r])]
               + [r for r in ys if match_predicate(pred, recs[r])])
        if not sel:
            return 0, ("the condition matched none of the records in "
                       "either group")
        new_sig = f"new{step_no}.{action_no}_{x_sig}_{y_sig}"
        for r in sel:
            state[r] = new_sig
        return len(sel), (f"the condition matched {len(sel)} records "
                          "across the two groups, which moved into a "
                          "group of their own")
    return 0, "no change"


def replay(client, template, cell, inst, tag="", collect=8):
    """Run one whole-cell replay against a live copy of the partition."""
    gold = {(a, b) for a, b in map(tuple, cell["gold"])}
    state = dict(cell["assignment"])
    base_pc, base_hits = pair_completeness(state, gold)
    base_c = candidate_count(state)
    rng_master = random.Random(f"replay:{inst['instance_id']}")

    # ops counts only the operations the judge chose. Steps that were
    # skipped or unreadable are bookkeeping and are reported in their own
    # sentences; mixing them in made "already_solved" read to the
    # reflector like an operation the judge had picked.
    trace, ops, book = [], Counter(), Counter()
    n_parse_fail = n_skipped = n_actions = 0
    # Per-step record of the running state, so any k-step PREFIX of this
    # instance can be scored offline from a single run. That turns the
    # seed-count question (does the k-th call still pay?) into a
    # by-product of every replay instead of a sweep over k, which would
    # otherwise cost a separate pass per value of k.
    #
    # Nothing here recomputes anything: hits and c are already computed
    # once per step below, and a skipped step leaves the state untouched,
    # so it carries the previous values forward rather than re-measuring.
    curve = []
    last_hits, last_c = base_hits, base_c
    # The score's numerator counts disclosed pairs; |T| does not, so a
    # judge that merely reunites what it was shown scores well where |T|
    # is small. Track the withheld pairs separately so the curve carries
    # an uncontaminated recall alongside the scored one.
    withheld = []
    for _s in inst["steps"]:
        _given = {tuple(p) for p in _s.get("seeds", [_s["seed"]])}
        withheld += [tuple(f) for f in _s["failed"] if tuple(f) not in _given]
    last_lh = sum(1 for a, b in withheld if state[a] == state[b])
    # Target counts per step are a property of the instance, not of the
    # state, so the prefix denominator can be accumulated up front.
    # Must exclude exactly what the score excludes -- every DISCLOSED
    # pair, not just the first. When these two disagreed the curve assert
    # fired and every instance aborted, which is the assert working, but
    # it is the kind of coupling that has to be kept in one place.
    step_targets = [sum(1 for f in s["failed"]
                        if tuple(f) not in {tuple(p) for p in
                                            s.get("seeds", [s["seed"]])})
                    for s in inst["steps"]]
    cum_targets = 0

    def note_step(k, ops_here):
        curve.append({"k": k, "hits": last_hits, "c": last_c,
                      "targets": cum_targets, "ops": ops_here,
                      # withheld pairs reunited so far: excludes every
                      # disclosed pair, so it is free of the credit the
                      # score gives for reuniting what was handed over
                      "lhits": last_lh, "ltot": len(withheld)})
    # Aggregates over every step, so feedback does not depend on an
    # arbitrary window of the replay, plus the handful of rules worth
    # quoting verbatim.
    reach_hist = Counter()
    rule_stats = {"helped": 0, "hurt": 0, "inert": 0}
    exemplars = []
    for k, step in enumerate(inst["steps"]):
        cum_targets += step_targets[k]
        sa, sb = step["seed"]
        xs_sig, ys_sig = state[sa], state[sb]
        if xs_sig == ys_sig:
            n_skipped += 1
            book["already_solved"] += 1
            note_step(k, [])
            continue
        members = block_members(state)
        # Earlier rules can empty a block entirely. The pair then still
        # "exists" as two signatures but has no records on one side, so
        # the prompt would show an empty group and the call is wasted.
        # Over a full sweep this matters: it is precisely the
        # reshuffling-invalidates-later-pairs effect under study.
        if not members[xs_sig] or not members[ys_sig]:
            n_skipped += 1
            book["block_dissolved"] += 1
            note_step(k, [])
            continue
        # Earlier rules move records, so a pair that fitted against the
        # original partition can outgrow the context by the time it is
        # reached. Skip it rather than show a trimmed sample.
        if not fits_context(cell, members[xs_sig], members[ys_sig]):
            n_skipped += 1
            book["skipped_oversize"] += 1
            note_step(k, [])
            continue
        rng = random.Random(f"{inst['instance_id']}:{k}")
        # Pass every disclosed pair; fill_template renders one line per
        # pair and seed-guided sampling anchors on all of them.
        shown = [tuple(p) for p in step.get("seeds", [[sa, sb]])]
        msg, sx, sy = fill_template(template, cell, members[xs_sig],
                                    members[ys_sig], shown, rng,
                                    BUDGET_CHARS)
        pc_before, _ = pair_completeness(state, gold)
        c_before = candidate_count(state)
        # A call that exhausts its retries costs this block pair, not the
        # run. Instances are independent and a replay already tolerates
        # an unreadable reply by skipping the step, so a call that never
        # came back is the same kind of loss. Previously the exception
        # propagated out of the panel's thread pool and destroyed every
        # completed instance with it -- five hours of work for one slow
        # request. Counted in book so a run degraded by timeouts is
        # visible rather than quietly thinner.
        try:
            raw = client.chat([{"role": "user", "content": msg}],
                              tag=f"{tag}:{k}")
        except Exception as exc:                       # noqa: BLE001
            n_skipped += 1
            book["call_failed"] += 1
            if len(trace) < collect:
                trace.append({"step": k, "error": f"call failed: {exc}"})
            note_step(k, [])
            continue
        actions, reason, err = parse_edit(raw)
        if err:
            n_parse_fail += 1
            book["unparseable"] += 1
            if len(trace) < collect:
                trace.append({"step": k, "error": err,
                              "raw": raw[:200]})
            note_step(k, [])
            continue
        # rules are applied in order against the live grouping, so a
        # later rule acts on what the earlier ones produced
        if FORCE_OP and actions:
            for a in actions:
                a["_model_op"] = a.get("operation")
                # An abstention carries no predicate, so there is
                # nothing to execute under the assigned operation; it is
                # left alone and counted as a refusal.
                if a.get("predicate"):
                    a["operation"] = FORCE_OP

        if COUNTERFACTUAL and actions:
            # Predicate and board held fixed; only the operation
            # varies. This MUST run before the chosen action is
            # applied: branching from the post-edit board would ask
            # what each operation buys on a partition the real edit
            # has already changed, and the predicate would find the
            # records it selected already moved.
            pc_cf, _ = pair_completeness(state, gold)
            c_cf = candidate_count(state)
            cf = {}
            for alt in OPS:
                probe = dict(state)
                n_alt, _ = apply_action(
                    probe, cell, xs_sig, ys_sig,
                    {"operation": alt,
                     "predicate": actions[0].get("predicate")},
                    k, 99)
                cf[alt] = {
                    "n_moved": n_alt,
                    "d_pc": pair_completeness(probe, gold)[0] - pc_cf,
                    "d_c": candidate_count(probe) - c_cf}
            step_cf = cf
        else:
            step_cf = None

        applied = []
        for j, act in enumerate(actions):
            pc_a = pair_completeness(state, gold)[0]
            c_a = candidate_count(state)
            n_moved, note = apply_action(state, cell, xs_sig, ys_sig,
                                         act, k, j)
            assert len(state) == len(cell["assignment"]), "P1 violated"
            ops[act["operation"]] += 1
            d_pc_a = pair_completeness(state, gold)[0] - pc_a
            d_c_a = candidate_count(state) - c_a
            bucket = ("0" if n_moved == 0 else "1" if n_moved == 1
                      else "2-5" if n_moved <= 5
                      else "6-20" if n_moved <= 20 else ">20")
            reach_hist[bucket] += 1
            rule_stats["helped" if d_pc_a > 0 else
                       "hurt" if d_pc_a < 0 else "inert"] += 1
            ex = {"step": k, "op": act["operation"],
                  "model_op": act.get("_model_op"),
                  "predicate": act["predicate"],
                  "n_moved": n_moved, "note": note,
                  "d_pc": d_pc_a, "d_c": d_c_a,
                  "seed": [sa, sb]}
            # The probe was taken from the board as it stood before any
            # of this reply's rules ran, so it belongs to the first rule.
            if j == 0 and step_cf is not None:
                ex["counterfactual"] = step_cf
            exemplars.append(ex)
            applied.append({"op": act["operation"],
                            "predicate": act["predicate"],
                            "n_moved": n_moved, "note": note,
                            "d_pc": d_pc_a, "d_c": d_c_a})
        n_actions += len(actions)
        pc_after, hits_after = pair_completeness(state, gold)
        c_after = candidate_count(state)
        last_hits, last_c = hits_after, c_after
        last_lh = sum(1 for a, b in withheld if state[a] == state[b])
        note_step(k, [a.get("operation") for a in actions])
        if len(trace) < collect:
            trace.append({"step": k, "reason": reason,
                          "actions": applied,
                          "d_pc": pc_after - pc_before,
                          "d_c": c_after - c_before,
                          "seed": [sa, sb],
                          "n_failed_here": step["n_failed"]})
    pc, hits = pair_completeness(state, gold)
    c = candidate_count(state)
    d_pc = pc - base_pc
    d_c_rel = (c - base_c) / base_c if base_c else 0.0
    # local recall: of the separated pairs sitting in the block pairs
    # this instance visited, how many are now together? The given seed
    # of each step is excluded, so reuniting only what was handed over
    # earns nothing.
    local_hits = local_targets = 0
    for step in inst["steps"]:
        # Exclude every pair that was SHOWN, not just the first. Crediting
        # a rule for reuniting a pair we handed it would inflate the score
        # by exactly the number of extra seeds disclosed.
        given = {tuple(p) for p in step.get("seeds", [step["seed"]])}
        for f in step["failed"]:
            t = tuple(f)
            if t in given:
                continue
            local_targets += 1
            local_hits += state[t[0]] == state[t[1]]
    local_recall = local_hits / local_targets if local_targets else 0.0
    # The numerator is the NET change in gold pairs grouped together
    # across the whole collection, so a rule is credited for pairs it
    # reunites in block pairs that were never visited and charged for
    # correctly grouped pairs it pulls apart. The denominator is only
    # what this instance had the opportunity to fix, which keeps cells
    # comparable and stops 8 block pairs out of 500 being scored as
    # ~0.003 of noise.
    net_ratio = ((hits - base_hits) / local_targets
                 if local_targets else 0.0)
    # The curve is only worth anything if its tail is the very state the
    # score was computed from. Checked rather than assumed: a silent
    # disagreement would mean every prefix score derived from it is
    # measuring something that is not the objective.
    if curve:
        assert curve[-1]["hits"] == hits, "curve tail != final hits"
        assert curve[-1]["c"] == c, "curve tail != final candidate count"
        assert curve[-1]["targets"] == local_targets, \
            "curve targets != score denominator"

    return {"score": net_ratio - LAMBDA * d_c_rel,
            "net_ratio": net_ratio,
            "net_pairs": hits - base_hits,
            "local_recall": local_recall, "local_hits": local_hits,
            "local_targets": local_targets,
            "d_pc": d_pc, "d_c_rel": d_c_rel,
            "pc_before": base_pc, "pc_after": pc,
            "c_before": base_c, "c_after": c,
            "pairs_gained": hits - base_hits,
            "steps_run": len(inst["steps"]) - n_skipped - n_parse_fail,
            "skipped": n_skipped, "parse_failures": n_parse_fail,
            "ops": dict(ops), "trace": trace,
            "n_actions": n_actions,
            "reach_hist": dict(reach_hist), "rule_stats": rule_stats,
            "exemplars": exemplars,
            "curve": curve, "base_hits": base_hits, "base_c": base_c,
            "lambda": LAMBDA,
            # "skipped" is the sum of two very different things. Keeping
            # the breakdown answers whether the drop-if-solved policy
            # actually bites (already_solved) or whether steps are being
            # lost to the context budget (skipped_oversize) -- the first
            # decides whether adaptive re-sampling is worth building, the
            # second is a sampling-fraction question.
            "book": dict(book),
            "blocks_before": len(set(cell["assignment"].values())),
            "blocks_after": len(set(state.values()))}


def feedback(cell, r, lex, mapping):
    recs = cell["records"]

    def m(rid):
        return lex.mask_text(
            " ".join(str(v) for v in recs[rid].values())[:70], mapping)

    collateral = r["net_pairs"] - r["local_hits"]
    head = (f"Over {r['steps_run']} edits, {r['net_pairs']:+d} more "
            f"same-entity pairs are grouped together across the whole "
            f"collection than before. {r['local_hits']} of those come "
            f"from the {r['local_targets']} separated pairs sitting in "
            f"the group pairs you were shown; the remaining "
            f"{collateral:+d} is the net effect everywhere else, where "
            f"your conditions also applied but you saw nothing -- a "
            f"negative number there means conditions pulled apart "
            f"records that were already correctly grouped. The number "
            f"of within-group comparisons changed by "
            f"{100 * r['d_c_rel']:+.2f}%. Score {r['score']:+.4f} = "
            f"net pairs over the {r['local_targets']} you could have "
            f"fixed, minus five times the relative growth in "
            f"comparisons. Operations chosen, out of the four "
            f"available: "
            + ", ".join(f"{o} {r['ops'].get(o, 0)}" for o in OPS) + ".")
    if r["parse_failures"]:
        head += (f" {r['parse_failures']} replies could not be read as "
                 "the required JSON object.")
    if r["skipped"]:
        head += (f" {r['skipped']} steps were skipped because an "
                 "earlier edit had already grouped that pair together.")
    lines = [head]
    rs = r["rule_stats"]
    total_rules = sum(rs.values())
    if total_rules:
        hist = r["reach_hist"]
        order = ["0", "1", "2-5", "6-20", ">20"]
        shape = ", ".join(f"{k}: {hist.get(k, 0)}" for k in order
                          if hist.get(k))
        lines.append(
            f"Across all {r['steps_run']} steps you wrote "
            f"{total_rules} conditions. How many records each one "
            f"matched: {shape}. Of those conditions, {rs['helped']} "
            f"grouped same-entity records together, {rs['hurt']} pulled "
            f"some apart, and {rs['inert']} changed nothing.")
    ex = [e for e in r["exemplars"]]
    best = sorted([e for e in ex if e["d_pc"] > 0],
                  key=lambda e: -e["d_pc"])[:3]
    worst = sorted([e for e in ex if e["d_pc"] < 0],
                   key=lambda e: e["d_pc"])[:3]
    inert_big = [e for e in ex if e["d_pc"] == 0 and e["n_moved"] == 0][:2]
    costly = sorted([e for e in ex if e["d_pc"] <= 0 and e["d_c"] > 0],
                    key=lambda e: -e["d_c"])[:2]

    def quote(e, verdict):
        return (f"  [{verdict}] {e['op']} where "
                f"{json.dumps(e['predicate'])[:170]} -> {e['note']}; "
                f"same-entity pairs {e['d_pc']:+.4f}, comparisons "
                f"{e['d_c']:+d}. The known pair at that step was "
                f"'{m(e['seed'][0])}' with '{m(e['seed'][1])}'.")

    if best:
        lines.append("Conditions that helped most:")
        lines += [quote(e, "helped") for e in best]
    if worst:
        lines.append("Conditions that did the most damage:")
        lines += [quote(e, "hurt") for e in worst]
    if costly:
        lines.append("Conditions that bought nothing but cost "
                     "comparisons:")
        lines += [quote(e, "wasteful") for e in costly]
    if inert_big:
        lines.append("Conditions that matched no records at all:")
        lines += [quote(e, "matched nothing") for e in inert_big]
    if r["parse_failures"]:
        bad = next((t for t in r["trace"] if "error" in t), None)
        if bad:
            lines.append(f"{r['parse_failures']} replies could not be "
                         f"read, for example: {bad['error'][:160]}")
    return "\n".join(lines)


REFLECT_TEMPLATE = """The text below is the message given to a small
language model -- an open-weights model of roughly 27 billion
parameters, run locally at temperature 0 with no reasoning step before
it answers. It is not a frontier model and you should not write for one.
It follows short, concrete, literal instructions well; it degrades on
long conditional chains, on rules that require holding several
constraints in mind at once, and on anything that asks it to infer what
you meant. Whatever you write has to work on the first read, with no
deliberation and no second pass. It is shown two groups of records that a grouping
procedure produced, together with one pair of records, one from each
group, that is known to describe the same real-world entity but was
split across the two. The model must answer with a change to the
grouping, stated as a condition on record fields: move every record of
one group satisfying the condition into the other, put the records of
either group satisfying it into a group of their own, or do nothing.
There is no way to name individual records; the condition is applied to
every record of the group, including the great majority that are not
shown to it, so a condition inferred from a handful of examples decides
the fate of records the model never saw.

Many such decisions are applied in sequence to the same collection, each
one to the grouping the previous ones produced, and the collection is
then scored as a whole:

    score = (net change in same-entity pairs grouped together across
              the whole collection, counted over the number of
              separated pairs in the group pairs you were shown)
            - 5 x (relative growth in the number of within-group
                   comparisons across the whole collection)

so reuniting a tenth of them is worth about as much as a two percent
growth in comparisons costs. A condition that
matches most of a group reunites a great deal but pays for every
comparison it creates; a narrow one is cheap but repairs little.

How the grouping mechanically responds, which the model is not
otherwise in a position to know:

- Which group is called X and which is called Y carries no meaning.
  The two are ordered by an internal identifier, not by size, content
  or quality, so any standing preference for one label is arbitrary.
- Moving k records out of a group of size a into a group of size b
  changes the number of comparisons by exactly k x (b - a + k). That
  is negative whenever k is smaller than a - b, so moving a modest
  number of records out of a much larger group into a smaller one
  can reunite records and lower the total at the same time.
- Taking ka records out of a group of size a and kb records out of a
  group of size b into one new group changes the count by
  ka x kb - ka x (a - ka) - kb x (b - kb). Drawing on only one of the
  two groups always lowers it, by k x (a - k). Drawing on both lowers
  it too, unless the two sets taken are large relative to what they
  leave behind; in the limit, taking all of both is arithmetically the
  same as uniting them.
- Records that leave a group are separated from everything left
  behind, including any they were already correctly grouped with.
  This is what the "net effect everywhere else" number reports.
- The records shown are a fixed uniform fraction of each group, so
  the number shown is proportional to the group's true size, which
  {STATS} reports.

These are facts about the machinery, not advice: which operation suits
a given pair of groups, and when, is yours to work out from the
outcomes below.

You are not told what these records are, where they come from, or what
procedure grouped them, and the record text in the feedback is
content-masked (<Tn> placeholders; the same placeholder means the same
original token).

Current message:
```
<curr_param>
```

Outcomes from recent collections:
<side_info>

The message must keep the slots {SEED}, {SAMPLE_X}, {SAMPLE_Y} and
{STATS}, which are filled with the known pair, a uniform random sample
of each group, and the true size of each group together with how many
of it are shown. You choose how they are introduced and what is said
around them. The reply
is read by a program that looks for a JSON object with an "operation"
key whose value is one of move_to_x, move_to_y, new_block or none, and
a "predicate" list of {"field", "operator", "value"} conditions. The
same program also accepts several rules at once, as an "actions" list
of up to five such objects sharing one "reason"; they are applied in
the order written, and each acts on the grouping the previous ones left
behind. Either shape is valid -- how the message asks for them is
yours to decide. There is no operation that merges two groups outright:
uniting records requires a condition that characterises them.

Return the complete message inside a single ``` block."""

# {STATS} is required, not optional: it carries the true size of each
# group, and the reflector is told how the comparison count responds to
# moving k records out of a group of size a into one of size b. A
# template that dropped {STATS} would leave the judge holding a rule
# whose inputs it cannot see.
REQUIRED_SLOTS = ("{SEED}", "{SAMPLE_X}", "{SAMPLE_Y}", "{STATS}")


# Opus from 2026-09-12; "sonnet" through candidate 3, briefly "fable"
# (which proposed only rejected candidates). Fable is unavailable to
# the CLI the driver actually resolves -- /usr/bin/claude is 2.1.123
# and rejects the model; the 2.1.220 install that accepts it sits on
# an fnm shim path under /run/user that does not survive a reboot.
# Record which model proposed which candidate when writing this up:
# the reflector_model field in llm_events.jsonl now carries it.
REFLECTOR_MODEL = "opus"
# Used after REFLECTOR_MODEL has failed three times -- almost always a
# subscription usage limit, which resets in hours. Blocking that long
# costs a whole iteration, so take a slightly weaker proposer instead.
FALLBACK_REFLECTOR = "sonnet"
# The model the last reflection actually used, so a candidate
# proposed after a fallback is not mis-attributed to Fable.
_LAST_REFLECTOR = REFLECTOR_MODEL


def reflector_alive(timeout=120):
    """Is the proposer answering? One trivial call, a few tokens."""
    try:
        p = subprocess.run(["claude", "-p", "--model", REFLECTOR_MODEL],
                           input="Reply with exactly: OK",
                           capture_output=True, text=True, timeout=timeout)
        return p.returncode == 0 and bool(p.stdout.strip())
    except Exception:  # noqa: BLE001 -- any failure means "not available"
        return False


def wait_for_reflector():
    """Block until the proposer answers.

    GEPA rolls the parent over the whole minibatch *before* calling
    reflection, so a dead proposer is otherwise discovered only after
    1,536 judge calls have already been spent -- and that iteration
    yields nothing: no proposal, no accept/reject, no information about
    what works. Eight such iterations burned 192 instances, 30% of all
    budget consumed, and taught us nothing. One trivial call up front
    costs nothing by comparison, so an expensive rollout only starts
    when the proposer can actually answer it.
    """
    waited = 0
    while not reflector_alive():
        wait = 60 if waited < 600 else 300 if waited < 3600 else 900
        print(f"[preflight] proposer unavailable; waited {waited}s, "
              f"sleeping {wait}s before starting a rollout", flush=True)
        time.sleep(wait)
        waited += wait
    if waited:
        print(f"[preflight] proposer back after {waited}s", flush=True)


def claude_reflection(prompt):
    """Reflection via the Claude CLI. Retries forever.

    Model: Fable 5 from 2026-09-10; candidates 0-3 came from Sonnet, so
    record which model produced which candidate when writing this up.
    Reflection is the one step where model capability converts directly
    into result quality -- the judge is fixed, retrieval is fixed, and
    this single call per iteration is the only source of new candidates.
    It synthesises ~174 quoted rule exemplars drawn from ~1,492
    block-pair decisions, which is the reasoning-heavy synthesis Fable is
    built for, at roughly one call per iteration.

    Reflection is the only source of new candidates: if it fails, the
    iteration produces nothing regardless, so there is nothing to be
    gained by giving up and moving on. The previous policy retried three
    times over 30 seconds and then raised -- against a subscription
    usage limit, which resets in hours, that guaranteed failure. Eight
    consecutive iterations each burned three doomed attempts and GEPA
    reported them as "did not propose a new candidate", so a silent
    infrastructure failure was indistinguishable from the search having
    converged. Two days were lost that way.

    Waiting is cheap: the judge is not involved, and one reflection is
    worth more than the hours it may cost to obtain. Every attempt logs
    its return code and stderr so the reason is never lost again.
    """
    attempt = 0
    while True:
        attempt += 1
        why = None
        # Fall back to Opus once Fable has failed three times (~3 min).
        # A subscription usage limit resets in hours, and waiting that
        # out costs an entire iteration of wall-clock we do not have
        # before the harvest deadline. Opus is a capable enough proposer
        # that a slightly weaker candidate beats no candidate at all.
        # Which model produced which candidate is recorded in the event
        # tag, so the write-up can still attribute them.
        model = REFLECTOR_MODEL if attempt <= 3 else FALLBACK_REFLECTOR
        global _LAST_REFLECTOR
        _LAST_REFLECTOR = model
        try:
            p = subprocess.run(["claude", "-p", "--model", model],
                               input=prompt, capture_output=True,
                               text=True, timeout=900)
            if p.returncode == 0 and p.stdout.strip():
                if attempt > 1:
                    print(f"[reflection] recovered on attempt {attempt} "
                          f"using {model}", flush=True)
                return p.stdout
            why = (f"rc={p.returncode} stdout={len(p.stdout)}B "
                   f"stderr={p.stderr.strip()[:400]!r}")
        except subprocess.TimeoutExpired:
            why = "timeout after 900s"
        except Exception as e:  # noqa: BLE001 -- surface anything else
            why = f"{type(e).__name__}: {e}"
        # 1m, 2m, 5m, 10m, 20m, then 30m forever. A usage limit needs
        # hours; a transient blip clears on the first retry.
        ladder = [60, 120, 300, 600, 1200]
        wait = ladder[attempt - 1] if attempt <= len(ladder) else 1800
        print(f"[reflection] attempt {attempt} failed ({why}); "
              f"retrying in {wait}s. prompt={len(prompt)} chars",
              flush=True)
        time.sleep(wait)


class Adapter:
    propose_new_texts = None

    def __init__(self, cells, client, lex, n_pairs, order):
        self.cells = cells
        self.client = client
        self.lex = lex
        self.n_pairs = n_pairs
        self.order = order

    def _instance(self, spec):
        cell = self.cells[tuple(spec["cell"])]
        return cell, make_instance(cell, spec["instance_id"],
                                   self.n_pairs, self.order)

    def evaluate(self, batch, candidate, capture_traces=False):
        from concurrent.futures import ThreadPoolExecutor
        # capture_traces marks the rollout whose only purpose is to feed
        # reflection. Don't spend it unless the proposer is up.
        if capture_traces:
            wait_for_reflector()
        from gepa.core.adapter import EvaluationBatch
        tpl = candidate["message"]
        bad = [s for s in REQUIRED_SLOTS if s not in tpl]

        def one(spec):
            # steps inside an instance must stay sequential (each edit
            # changes what the next step sees), but instances are
            # independent, so they run concurrently
            cell, inst = self._instance(spec)
            if bad:
                r = {"score": -1.0, "d_pc": 0.0, "d_c_rel": 0.0,
                     "pc_before": 0.0, "pc_after": 0.0, "c_before": 0,
                     "c_after": 0, "pairs_gained": 0, "steps_run": 0,
                     "skipped": 0, "parse_failures": 0,
                     "ops": {}, "trace": [],
                     "slot_error": f"missing slots {bad}",
                     "blocks_before": 0, "blocks_after": 0}
            else:
                r = replay(self.client, tpl, cell, inst,
                           tag=f"v2:{spec['instance_id']}")
                r["slot_error"] = None
            return spec, cell, r

        if len(batch) > 1:
            # An instance replays its block pairs sequentially, so this is
            # also the number of concurrent LLM calls. Must not exceed the
            # servers' total slots (OLLAMA_NUM_PARALLEL=3 each): queueing
            # beyond them buys nothing, and the cards are already ~85%
            # utilised. Calls go round-robin over OLLAMA_URL.
            n_servers = max(1, len(harness.OLLAMA_URLS))
            n_workers = min(SLOTS_PER_SERVER * n_servers, len(batch))
            # How GEPA chunks work decides whether the servers stay fed.
            # A batch drains only when its slowest instance finishes, so a
            # long tail leaves the GPUs idle; this records the shape so
            # the cost is visible rather than inferred.
            _t0 = time.time()
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                results = list(ex.map(one, batch))
            _el = time.time() - _t0
            print(f"[batch] n={len(batch)} workers={n_workers} "
                  f"{_el:.0f}s = {_el / 60:.1f} min "
                  f"({_el / max(1, len(batch)):.0f}s per instance)",
                  flush=True)
        else:
            results = [one(batch[0])] if batch else []
        outputs, scores = [], []
        trajs = [] if capture_traces else None
        for spec, cell, r in results:
            outputs.append({"d_pc": r["d_pc"], "d_c_rel": r["d_c_rel"],
                            "ops": r["ops"]})
            scores.append(r["score"])
            if trajs is not None:
                trajs.append({"spec": spec, "cell": cell, "r": r})
        return EvaluationBatch(outputs=outputs, scores=scores,
                               trajectories=trajs)

    def make_reflective_dataset(self, candidate, eval_batch,
                                components_to_update):
        recs = []
        for t in eval_batch.trajectories:
            r = t["r"]
            mapping = {}
            if r.get("slot_error"):
                fb = ("the message was rejected before it was ever "
                      f"sent: {r['slot_error']}. The slots "
                      f"{REQUIRED_SLOTS} must all appear.")
            else:
                fb = feedback(t["cell"], r, self.lex, mapping)
            recs.append({
                "Inputs": (f"a collection of "
                           f"{len(t['cell']['usable_block_pairs'])} group pairs,"
                           f" {r['blocks_before']} groups in total"),
                "Generated Outputs": json.dumps(r["ops"]),
                "Feedback": fb})
        return {"message": recs}


def train_specs(cells, keys, n, offset=0):
    """Fresh instance ids: the space of (which pairs, which seeds, what
    order) is effectively unbounded, so training never repeats one."""
    out = []
    for i in range(n):
        key = keys[i % len(keys)]
        out.append({"cell": list(key),
                    "instance_id": f"train:{key[0]}:{key[1]}:{offset + i}"})
    return out


def frozen_specs(keys, per_cell, prefix, variant=0):
    return [{"cell": list(k),
             "instance_id": f"{prefix}:{k[0]}:{k[1]}:v{variant}:{j}"}
            for k in keys for j in range(per_cell)]


def evaluate_panel(client, cells, specs, template, n_pairs, order,
                   path, variants=1):
    from concurrent.futures import ThreadPoolExecutor
    rows = []
    t0 = time.time()
    jobs = []
    for v in range(variants):
        for spec in specs:
            s = dict(spec)
            if variants > 1:
                s["instance_id"] = spec["instance_id"].replace(
                    ":v0:", f":v{v}:")
            jobs.append((v, s))

    def one_eval(job):
        v, s = job
        cell = cells[tuple(s["cell"])]
        inst = make_instance(cell, s["instance_id"], n_pairs, order)
        r = replay(client, template, cell, inst,
                   tag=f"eval:{s['instance_id']}")
        print(f"v{v} {s['cell'][1]:16s} dPC={r['d_pc']:+.4f} "
              f"d|C|={100 * r['d_c_rel']:+.2f}% "
              f"pairs={r['pairs_gained']:+d} ops={r['ops']}", flush=True)
        # curve and exemplars are carried through so the seed-count
        # question (E1) and the payoff-per-operation question (E6) are
        # answered from the panel's own output. Both were already being
        # computed inside replay() and discarded here, which meant each
        # would otherwise have cost a separate pass over the GPU.
        return {"variant": v, "cell": s["cell"],
                "instance_id": s["instance_id"],
                **{k: r[k] for k in
                   ("score", "d_pc", "d_c_rel", "pc_before",
                    "pc_after", "pairs_gained", "steps_run",
                    "skipped", "parse_failures", "ops",
                    "curve", "exemplars", "base_hits", "base_c",
                    "rule_stats", "reach_hist", "book",
                    "local_recall", "local_hits", "local_targets")}}

    # Instances are independent (steps inside one are not), so the panel
    # parallelises the same way training does. Left sequential this took
    # ~26 h per arm, which made a like-for-like seed-vs-best comparison
    # impractical.
    n_workers = min(SLOTS_PER_SERVER * max(1, len(harness.OLLAMA_URLS)),
                    len(jobs))
    print(f"[panel] {len(jobs)} instances, {n_workers} workers", flush=True)
    def guarded(job):
        """One instance's failure must not discard the other 39."""
        try:
            return one_eval(job)
        except Exception as exc:                       # noqa: BLE001
            print(f"[panel] instance {job[1]['instance_id']} failed: "
                  f"{exc}", flush=True)
            return None

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        rows = list(ex.map(guarded, jobs))
    n_lost = sum(r is None for r in rows)
    rows = [r for r in rows if r is not None]
    if n_lost:
        print(f"[panel] {n_lost} of {n_lost + len(rows)} instances were "
              f"lost to errors and are excluded from the means below",
              flush=True)
    per_cell = defaultdict(list)
    for row in rows:
        per_cell[f"{row['cell'][0]}/{row['cell'][1]}"].append(row)
    summary = {}
    for k, rs in sorted(per_cell.items()):
        d = [x["d_pc"] for x in rs]
        summary[k] = {
            "d_pc_mean": sum(d) / len(d),
            "d_pc_spread": max(d) - min(d),
            "d_c_rel_mean": sum(x["d_c_rel"] for x in rs) / len(rs),
            "pairs_gained_mean": sum(x["pairs_gained"] for x in rs) / len(rs),
            "score_mean": sum(x["score"] for x in rs) / len(rs),
            "n": len(rs)}
    out = {"variants": variants, "n_pairs_per_instance": n_pairs,
           "order": order, "per_cell": summary,
           "macro_d_pc": sum(v["d_pc_mean"] for v in summary.values())
                          / max(1, len(summary)),
           "macro_score": sum(v["score_mean"] for v in summary.values())
                           / max(1, len(summary)),
           # Degradation has to be legible in the artifact, not only in
           # the log: a panel thinned by timeouts must not be mistaken
           # for a clean one when the numbers are read back later.
           "instances_lost": n_lost,
           "calls_failed": sum((r.get("book") or {}).get("call_failed", 0)
                               for r in rows),
           "rows": rows, "runtime_s": round(time.time() - t0, 1)}
    json.dump(out, open(path, "w"), indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"},
                     indent=2)[:1500])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blockers", nargs="+", default=["lambdafold"])
    ap.add_argument("--train-datasets", nargs="+",
                    default=["amazon-google", "walmart-amazon",
                             "dblp-acm", "cora"])
    ap.add_argument("--test-datasets", nargs="+",
                    default=["abt-buy", "dblp-scholar"])
    ap.add_argument("--pairs-per-instance", type=int, default=15)
    ap.add_argument("--n-seeds", type=int, default=1,
                    help="disclosed gold pairs per block pair")
    ap.add_argument("--seed-pick", default="random",
                    choices=["random", "similar", "different"])
    ap.add_argument("--sampling", default="uniform",
                    choices=["uniform", "seed_similar"],
                    help="how the shown records are chosen")
    ap.add_argument("--bank-root", default=None,
                    help="cell bank to read (E5 operating points)")
    ap.add_argument("--force-op", default=None, choices=list(OPS),
                    help="override the operation the model names (E6-B)")
    ap.add_argument("--counterfactual", action="store_true",
                    help="record what each operation would have bought "
                         "with the predicate held fixed (E6-A)")
    ap.add_argument("--order", default="random",
                    choices=["random", "small_first", "large_first"])
    ap.add_argument("--val-per-cell", type=int, default=2)
    ap.add_argument("--test-per-cell", type=int, default=2)
    ap.add_argument("--variants", type=int, default=3)
    ap.add_argument("--min-pc", type=float, default=0.45)
    ap.add_argument("--max-pc", type=float, default=0.80)
    ap.add_argument("--max-oversize-frac", type=float, default=0.10,
                    help="drop a cell if more than this fraction of its "
                         "Case-3 block pairs cannot be shown in full")
    ap.add_argument("--min-case3", type=int, default=10)
    ap.add_argument("--model", default="qwen3:8b")
    # Thinking is compute-bound: measured 7.38s/call whether one
    # call runs or sixteen, versus 2.1s without. It buys reasoning
    # depth at roughly 3.5x the cost per decision.
    ap.add_argument("--think", action="store_true")
    # budget counts INSTANCES; each is pairs-per-instance judge
    # calls, so 250 x 15 = 3,750 calls
    ap.add_argument("--budget", type=int, default=250)
    ap.add_argument("--minibatch", type=int, default=1)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--eval-split", default=None, choices=[None, "test",
                                                           "val"])
    ap.add_argument("--template-file", default=None)
    ap.add_argument("--tag", default="best")
    args = ap.parse_args()
    global COUNTERFACTUAL
    COUNTERFACTUAL = args.counterfactual
    global FORCE_OP
    FORCE_OP = args.force_op
    global SAMPLING
    SAMPLING = args.sampling
    global N_SEEDS, SEED_PICK
    N_SEEDS, SEED_PICK = args.n_seeds, args.seed_pick
    if args.bank_root:
        global BANK_ROOT
        BANK_ROOT = args.bank_root
    os.makedirs(args.run_dir, exist_ok=True)

    # A cell only enters the matrix if its blocker actually sits at a
    # comparable operating point and leaves enough to debug. Some
    # blocker/dataset combinations cannot: k-means LSH over shingled
    # product titles separates half the true matches even with three
    # blocks, so those cells would contribute a different (much more
    # broken) problem rather than another sample of the same one.
    cells, dropped = {}, []
    for blk in args.blockers:
        for ds in args.train_datasets + args.test_datasets:
            p = os.path.join(BANK_ROOT, blk, f"{ds}.json")
            if not os.path.exists(p):
                dropped.append((blk, ds, "not built"))
                continue
            c = load_cell(blk, ds)
            if not (args.min_pc <= c["PC"] <= args.max_pc):
                dropped.append((blk, ds,
                                f"PC {c['PC']:.3f} outside "
                                f"[{args.min_pc}, {args.max_pc}]"))
                continue
            keep, oversize = usable_block_pairs(c)
            c["usable_block_pairs"] = keep
            c["n_oversize"] = len(oversize)
            # A cell whose large block pairs do not fit would be measured
            # on its small ones only -- a different, easier problem than
            # the cell actually poses. Admit it whole or not at all.
            n_c3 = len(keep) + len(oversize)
            frac = len(oversize) / n_c3 if n_c3 else 1.0
            if frac > args.max_oversize_frac:
                dropped.append((blk, ds,
                                f"{len(oversize)} of {n_c3} Case-3 block "
                                f"pairs ({100 * frac:.0f}%) are too large "
                                "for the context; keeping only the small "
                                "ones would bias the cell"))
                continue
            if len(keep) < args.min_case3:
                dropped.append((blk, ds,
                                f"only {len(keep)} usable Case-3 block "
                                f"pairs ({c['n_case3']} Case-3, "
                                f"{len(oversize)} too large for the "
                                "context)"))
                continue
            if oversize:
                print(f"[oversize] {blk}/{ds}: {len(oversize)} of "
                      f"{c['n_case3']} Case-3 block pairs excluded — "
                      "their full sample exceeds the context",
                      flush=True)
            cells[(blk, ds)] = c
    for blk, ds, why in dropped:
        print(f"[excluded] {blk}/{ds}: {why}", flush=True)
    train_keys = [k for k in cells if k[1] in set(args.train_datasets)]
    test_keys = [k for k in cells if k[1] in set(args.test_datasets)]
    if not train_keys:
        raise SystemExit("no training cells found")

    log = harness.EventLog(os.path.join(args.run_dir,
                                        "llm_events.jsonl"))
    client = harness.OllamaClient(args.model, log,
                                  think=args.think,
                                  num_ctx=MAX_CTX)
    lex = mask.ContentLexicon({k: cells[k] for k in train_keys})

    template = SEED_TEMPLATE
    if args.template_file:
        template = open(args.template_file, encoding="utf-8").read()

    if args.eval_split:
        keys = test_keys if args.eval_split == "test" else train_keys
        per = (args.test_per_cell if args.eval_split == "test"
               else args.val_per_cell)
        specs = frozen_specs(keys, per, args.eval_split)
        evaluate_panel(client, cells, specs, template,
                       args.pairs_per_instance, args.order,
                       os.path.join(args.run_dir,
                                    f"eval_{args.eval_split}_"
                                    f"{args.tag}.json"),
                       variants=args.variants)
        return

    import gepa
    n_train = max(400, args.budget * 2)
    train = train_specs(cells, train_keys, n_train)
    val = frozen_specs(train_keys, args.val_per_cell, "val")
    print(f"cells train={train_keys} test={test_keys}\n"
          f"train specs={len(train)} val={len(val)} "
          f"pairs/instance={args.pairs_per_instance}", flush=True)

    # A token counts as leaked only if it actually occurs in the record
    # text of the cells in play. The previous test -- rare in English
    # (wordfreq zipf < 3.0) and absent from the seed template -- fired on
    # ordinary technical vocabulary the reflector had introduced itself
    # ("predicates", "categorical", "keyed", "substring"), tripping on 4
    # of 8 reflections and forcing it to delete legitimate words.
    data_vocab = set()
    for c in cells.values():
        for r in c["records"].values():
            for v in r.values():
                data_vocab.update(re.findall(r"[a-z0-9]+",
                                             str(v).lower()))
    whitelist = set(lex.audit_prompt(SEED_TEMPLATE + " "
                                     + REFLECT_TEMPLATE))

    # What has already been proposed, so the reflector does not spend a
    # call re-deriving it. Three of eight reflections in the previous run
    # produced the same text, one pair byte-identical, because nothing
    # in the prompt recorded that an edit had been tried before.
    history = []          # (diff_vs_parent, verdict)

    def _record_attempt(diff, verdict):
        for i, (d, _) in enumerate(history):
            if d == diff:
                history[i] = (d, verdict)
                return
        history.append((diff, verdict))
        del history[:-8]

    def _history_block():
        if not history:
            return ""
        parts = ["\n\nEdits already proposed in earlier rounds, most "
                 "recent last. Propose something materially different "
                 "from these rather than restating them:"]
        for i, (d, verdict) in enumerate(history, 1):
            parts.append(f"\n--- earlier edit {i} ({verdict}) ---\n"
                         f"{d[:2500]}")
        return "\n".join(parts)

    def reflection_lm(p):
        if not isinstance(p, str):
            p = "\n\n".join(m.get("content", "") for m in p)
        p_aug = p + _history_block()
        # A failing reflection used to vanish: claude_reflection raises,
        # GEPA swallows it, and the iteration logs only "did not propose
        # a new candidate". Four such iterations looked like the search
        # converging when the proposer was simply down.
        try:
            out = claude_reflection(p_aug)
        except Exception as exc:  # noqa: BLE001 -- must be recorded
            log.append({"ts": time.time(), "tag": "reflection-failed",
                        "error": f"{type(exc).__name__}: {exc}"})
            print(f"[reflection FAILED] {type(exc).__name__}: {exc}",
                  flush=True)
            raise
        leaked = [t for t in lex.audit_prompt(out)
                  if t not in whitelist and t.lower() in data_vocab]
        gate = None
        if leaked:
            gate = leaked[:20]
            out = claude_reflection(
                p_aug + "\n\nYour draft contained tokens taken from the "
                f"data itself: {leaked[:20]}. Those will not exist in the "
                "data this is deployed on. Rewrite without them, same "
                "output format.")
        m = re.search(r"```(?:\w+)?\n(.*?)```", out, re.S)
        body = (m.group(1) if m else out).strip()
        diff = "\n".join(difflib.unified_diff(
            SEED_TEMPLATE.splitlines(), body.splitlines(),
            lineterm="", n=0))
        _record_attempt(diff or "(no change from the seed)", "proposed")
        log.append({"ts": time.time(), "tag": "reflection-sonnet",
                    "reflector_model": _LAST_REFLECTOR,
                    "messages": p_aug, "raw_response": out,
                    "n_history_shown": len(history) - 1,
                    "leakage_gate_triggered": gate})
        return out

    json.dump({"arm": "v2-partition-repair", "blockers": args.blockers,
               "cells_used": [list(k) for k in sorted(cells)],
               "cells_excluded": [[b, d, w] for b, d, w in dropped],
               "oversize_excluded": {f"{k[0]}/{k[1]}":
                                     cells[k]["n_oversize"]
                                     for k in cells},
               "budget_chars": BUDGET_CHARS,
               "max_oversize_frac": args.max_oversize_frac,
               "usability": {"min_pc": args.min_pc,
                             "max_pc": args.max_pc,
                             "min_case3": args.min_case3},
               "train_datasets": args.train_datasets,
               "test_datasets": args.test_datasets,
               "pairs_per_instance": args.pairs_per_instance,
               "think": args.think,
               "order": args.order, "lambda": LAMBDA,
               "sample_fraction": SAMPLE_FRACTION, "model": args.model,
               "budget": args.budget, "minibatch": args.minibatch,
               "score": "net gold pairs reunited across the whole cell / separated pairs in the visited block pairs - 5.0 * relative growth of |C|",
               "seed_template": template,
               "reflect_template": REFLECT_TEMPLATE,
               "reflector_model": REFLECTOR_MODEL,
               "use_merge": False},
              open(os.path.join(args.run_dir, "config.json"), "w"),
              indent=2)

    result = gepa.optimize(
        seed_candidate={"message": template},
        trainset=train, valset=val,
        adapter=Adapter(cells, client, lex, args.pairs_per_instance,
                        args.order),
        reflection_lm=reflection_lm,
        reflection_minibatch_size=args.minibatch,
        reflection_prompt_template=REFLECT_TEMPLATE,
        max_metric_calls=args.budget,
        # Merge is off. It is the designed mechanism for the Pareto
        # frontier headroom (+0.157 here), but a rejected merge makes
        # GEPA skip reflective mutation for that whole iteration -- so a
        # failed merge costs a search iteration and returns no proposal,
        # no accept/reject, and nothing about what works. It took 2 of
        # our first 8 search iterations that way, both merging the same
        # pair (1 and 2 via ancestor 0), both rejected, and it evaluates
        # on a 5-instance subsample so it runs at 5 of 24 workers.
        # With four candidates -- three of them ancestors of each other
        # -- there is not enough diversity for a merge to find anything.
        # Revisit once the pool is larger and genuinely varied.
        use_merge=False,
        # Without this, every resume re-evaluates the seed on the whole
        # val panel -- 40 instances x 64 pairs = 2,560 calls, 4.3 hours,
        # to recompute a number already sitting in gepa_state.bin. This
        # run gets interrupted often enough that it pays for itself.
        cache_evaluation=True,
        # The score is unbounded: net recall over local targets minus a
        # cost term, observed up to 2.12. Treating 1.0 as perfect made
        # GEPA skip 5.6% of instances as already solved.
        skip_perfect_score=False,
        run_dir=args.run_dir, seed=0, display_progress_bar=False,
        raise_on_exception=False)
    open(os.path.join(args.run_dir, "best_message.txt"), "w",
         encoding="utf-8").write(result.best_candidate["message"])
    print("FINISHED", result.total_metric_calls, "instances,",
          result.num_candidates, "candidates")


if __name__ == "__main__":
    main()
