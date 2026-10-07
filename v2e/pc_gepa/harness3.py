"""v3 harness: RULE-INDUCTION debugging (general-prompt.txt design).

One call per episode. The model sees the seed pair + proportional random
samples of both blocks and returns a JSON diagnosis with a remedy:
merge_blocks | move_records | split_new_block, where moves are field-level
predicates. The harness EXECUTES the predicates against the FULL blocks
(records the model never saw), materializes newly co-blocked cross-table
pairs, and scores PC recovery. No coverage ceiling: unshown targets are
recoverable through the rule.

Score = recovered_targets/|targets| − ADD_PENALTY × added_nongold_pairs.
ADD_PENALTY prices the addition budget (plan: additions capped ~1% |C|):
a blanket merge_blocks of two big blocks adds thousands of pairs and goes
deeply negative unless the blocks genuinely overlap.

GEPA candidate:
  "instruction"   — the ENTIRE task prompt (monolithic, per user rule:
                    no split without a sequential dependency). Placeholders
                    {SEED_BLOCK}, {SAMPLES} are filled by the harness.
  "sampling_spec" — harness-side config (executed BEFORE the call — the
                    one legitimate split): {"n_sample_cap": int 10..80,
                    "proportional": bool, "show_fields": [...],
                    "max_desc": int 0..300}
"""
import json
import random
import re

import harness

ADD_PENALTY = 0.0005
PROP_SEED = 0

FIELDS = ("title", "manufacturer", "price", "description")

DEFAULT_SPEC3 = {
    "n_sample_cap": 60,
    "proportional": True,
    "show_fields": ["title", "manufacturer", "price", "description"],
    "max_desc": 60,
}
SEED_SPEC3_TEXT = json.dumps(DEFAULT_SPEC3, indent=2)

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "failure_mode": {"type": "string"},
        "confidence": {"type": "string"},
        "rationale": {"type": "string"},
        "evidence_tokens": {"type": "array", "items": {"type": "string"}},
        "remedy": {"type": "string",
                   "enum": ["merge_blocks", "move_records",
                            "split_new_block"]},
        "move_specification": {
            "type": "object",
            "properties": {
                "move_to_block_X": {"$ref": "#/$defs/move"},
                "move_to_block_Y": {"$ref": "#/$defs/move"},
                "move_to_new_block": {"$ref": "#/$defs/move"},
            },
        },
        "generalization_confidence": {"type": "string"},
        "generalization_note": {"type": "string"},
    },
    "required": ["failure_mode", "remedy", "move_specification",
                 "rationale"],
    "$defs": {
        "move": {
            "type": "object",
            "properties": {
                "predicate": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "field": {"type": "string",
                                      "enum": list(FIELDS)},
                            "operator": {"type": "string",
                                         "enum": ["equals", "contains",
                                                  "not_contains", "regex",
                                                  "in_range", "is_null",
                                                  "is_not_null"]},
                            "value": {"type": ["string", "number",
                                               "array", "null"],
                                      "items": {"type": "number"}},
                        },
                        "required": ["field", "operator"],
                    },
                },
                "matched_sample_ids": {"type": "array",
                                       "items": {"type": "string"}},
            },
            "required": ["predicate"],
        },
    },
}


def build_record_fields(dataset_dir_data):
    """One-time: re-load tables to get per-field records (texts in
    episodes.json are concatenated). Returns {rec_id: {field: value}}."""
    import os
    import sys
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.abspath(__file__)), os.pardir, "src"))
    import loaders
    ta, tb, _, _ = loaders.load_amazon_google()
    recs = {}
    for _, r in ta.iterrows():
        recs[str(r["id"])] = {
            "side": "A", "title": str(r.get("title", "")),
            "manufacturer": str(r.get("manufacturer", "")),
            "price": str(r.get("price", "")),
            "description": str(r.get("description", ""))}
    for _, r in tb.iterrows():
        recs[str(r["id"])] = {
            "side": "B", "title": str(r.get("name", "")),
            "manufacturer": str(r.get("manufacturer", "")),
            "price": str(r.get("price", "")),
            "description": str(r.get("description", ""))}
    return recs


def parse_spec3(text):
    try:
        raw = json.loads(text)
    except Exception as e:  # noqa: BLE001
        return None, f"sampling_spec not valid JSON: {e}"
    unknown = set(raw) - set(DEFAULT_SPEC3)
    if unknown:
        return None, (f"unknown keys {sorted(unknown)}; allowed "
                      f"{sorted(DEFAULT_SPEC3)}")
    spec = {**DEFAULT_SPEC3, **raw}
    if not (isinstance(spec["n_sample_cap"], int)
            and 10 <= spec["n_sample_cap"] <= 80):
        return None, "n_sample_cap must be int in 10..80"
    if not (isinstance(spec["max_desc"], int)
            and 0 <= spec["max_desc"] <= 300):
        return None, "max_desc must be int in 0..300"
    if (not isinstance(spec["show_fields"], list)
            or not set(spec["show_fields"]) <= set(FIELDS)
            or "title" not in spec["show_fields"]):
        return None, f"show_fields must be a sublist of {FIELDS} incl title"
    return spec, None


def fmt_record(rid, rec, spec):
    parts = [f"[{rec['side']}{rid}] {rec['title']}"]
    if "manufacturer" in spec["show_fields"] and rec["manufacturer"]:
        parts.append(f"mfr: {rec['manufacturer']}")
    if "price" in spec["show_fields"] and rec["price"]:
        parts.append(f"price: {rec['price']}")
    if "description" in spec["show_fields"] and spec["max_desc"] > 0 \
            and rec["description"]:
        parts.append(f"desc: {rec['description'][:spec['max_desc']]}")
    return " | ".join(parts)


def block_members(ep, assignments):
    """Full membership (both table sides) of the episode's two blocks."""
    x, y = ep["block1_sig"], ep["block2_sig"]
    mx = [r for r, s in assignments.items() if s == x]
    my = [r for r, s in assignments.items() if s == y]
    return sorted(mx), sorted(my)


def sample_block(members, cap, proportional, other_n, rng):
    if not proportional or len(members) <= cap:
        n = min(cap, len(members))
    else:
        tot = len(members) + other_n
        n = max(10, min(cap, round(cap * 2 * len(members) / max(tot, 1))))
        n = min(n, len(members))
    return sorted(rng.sample(members, n)) if n < len(members) else members


def render_v3(ep, recs, assignments, instruction, spec):
    rng = random.Random(f"{PROP_SEED}:{ep['episode_id']}")
    mx, my = block_members(ep, assignments)
    sx = sample_block(mx, spec["n_sample_cap"], spec["proportional"],
                      len(my), rng)
    sy = sample_block(my, spec["n_sample_cap"], spec["proportional"],
                      len(mx), rng)
    sa, sb = ep["seed_pair"]
    seed_block = (
        f"Record 1 (Block X): {fmt_record(sa, recs[sa], spec)}\n"
        f"Record 2 (Block Y): {fmt_record(sb, recs[sb], spec)}")
    samples = [f"Block X — sample of {len(sx)} / {len(mx)} total:"]
    samples += [fmt_record(r, recs[r], spec) for r in sx]
    samples += ["", f"Block Y — sample of {len(sy)} / {len(my)} total:"]
    samples += [fmt_record(r, recs[r], spec) for r in sy]
    msg = (instruction
           .replace("{SEED_BLOCK}", seed_block)
           .replace("{SAMPLES}", "\n".join(samples)))
    return msg, sx, sy, mx, my


_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def eval_predicate(pred, rec):
    for cond in pred:
        field = cond.get("field", "")
        if field not in FIELDS:
            return False
        val = rec[field]
        op = cond.get("operator")
        cv = cond.get("value")
        cvs = str(cv).lower() if cv is not None else ""
        v = val.lower()
        if op == "equals":
            ok = v == cvs
        elif op == "contains":
            ok = cvs in v
        elif op == "not_contains":
            ok = cvs not in v
        elif op == "regex":
            try:
                ok = re.search(str(cv), val, re.I) is not None
            except re.error:
                return False
        elif op == "in_range":
            m = _NUM.search(val)
            try:
                lo, hi = float(cv[0]), float(cv[1])
                ok = m is not None and lo <= float(m.group()) <= hi
            except (TypeError, ValueError, IndexError):
                return False
        elif op == "is_null":
            ok = v.strip() == ""
        elif op == "is_not_null":
            ok = v.strip() != ""
        else:
            return False
        if not ok:
            return False
    return True


def execute_remedy(reply, recs, mx, my, gold_all):
    """Apply remedy → set of NEW co-blocked cross-table pairs (a_id, b_id).
    merge_blocks: full X∪Y. move_records: predicate-matched records join
    the other block. split_new_block: matched records from both form a new
    block. Returns (new_pairs, n_moved, exec_notes)."""
    xa = [r for r in mx if recs[r]["side"] == "A"]
    xb = [r for r in mx if recs[r]["side"] == "B"]
    ya = [r for r in my if recs[r]["side"] == "A"]
    yb = [r for r in my if recs[r]["side"] == "B"]
    remedy = reply.get("remedy")
    spec = reply.get("move_specification") or {}
    new_pairs = set()
    notes = []
    n_moved = 0

    def cross(a_ids, b_ids):
        return {(a, b) for a in a_ids for b in b_ids}

    if remedy == "merge_blocks":
        new_pairs = cross(xa, yb) | cross(ya, xb)
        notes.append(f"merged: +{len(new_pairs)} cross pairs")
    else:
        def matched(pool, key):
            mv = spec.get(key) or {}
            pred = mv.get("predicate") or []
            if not pred:
                return []
            return [r for r in pool if eval_predicate(pred, recs[r])]

        if remedy == "move_records":
            to_x = matched(my, "move_to_block_X")
            to_y = matched(mx, "move_to_block_Y")
            n_moved = len(to_x) + len(to_y)
            new_pairs |= cross(xa, [r for r in to_x
                                    if recs[r]["side"] == "B"])
            new_pairs |= cross([r for r in to_x
                                if recs[r]["side"] == "A"], xb)
            new_pairs |= cross(ya, [r for r in to_y
                                    if recs[r]["side"] == "B"])
            new_pairs |= cross([r for r in to_y
                                if recs[r]["side"] == "A"], yb)
            notes.append(f"moved {len(to_x)}->X, {len(to_y)}->Y: "
                         f"+{len(new_pairs)} cross pairs")
        elif remedy == "split_new_block":
            nx = matched(mx, "move_to_new_block")
            ny = matched(my, "move_to_new_block")
            n_moved = len(nx) + len(ny)
            na = [r for r in nx + ny if recs[r]["side"] == "A"]
            nb = [r for r in nx + ny if recs[r]["side"] == "B"]
            old = cross([r for r in nx if recs[r]["side"] == "A"],
                        [r for r in nx if recs[r]["side"] == "B"]) | \
                  cross([r for r in ny if recs[r]["side"] == "A"],
                        [r for r in ny if recs[r]["side"] == "B"])
            new_pairs = cross(na, nb) - old
            notes.append(f"new block {len(nx)}+{len(ny)} records: "
                         f"+{len(new_pairs)} cross pairs")
        else:
            notes.append(f"unknown remedy {remedy!r}: no-op")
    return new_pairs, n_moved, notes


def score_v3(ep, new_pairs, gold_all):
    targets = {tuple(t) for t in ep["targets"]}
    seed = tuple(ep["seed_pair"])
    hits = new_pairs & targets
    added_gold = new_pairs & gold_all
    added_nongold = len(new_pairs - added_gold - {seed})
    recall = len(hits) / len(targets) if targets else 0.0
    score = max(0.0, recall - ADD_PENALTY * added_nongold)
    return score, hits, targets, added_nongold


def run_episode_v3(client, instruction, spec_text, ep, recs, assignments,
                   gold_all, tag=""):
    spec, err = parse_spec3(spec_text)
    if err:
        return {"score": 0.0, "spec_error": err, "parse_error": None,
                "raw": "", "reply": None, "hits": set(),
                "targets": {tuple(t) for t in ep["targets"]},
                "added_nongold": 0, "notes": [], "user_message": ""}
    if "{SEED_BLOCK}" not in instruction or "{SAMPLES}" not in instruction:
        return {"score": 0.0,
                "spec_error": ("instruction must contain the literal "
                               "placeholders {SEED_BLOCK} and {SAMPLES}"),
                "parse_error": None, "raw": "", "reply": None,
                "hits": set(),
                "targets": {tuple(t) for t in ep["targets"]},
                "added_nongold": 0, "notes": [], "user_message": ""}
    msg, sx, sy, mx, my = render_v3(ep, recs, assignments, instruction,
                                    spec)
    raw = client.chat([{"role": "user", "content": msg}],
                      fmt=RESPONSE_SCHEMA, tag=tag)
    try:
        reply = json.loads(raw)
        perr = None
    except Exception as e:  # noqa: BLE001
        reply, perr = None, f"{type(e).__name__}: {e}"
    if reply is None:
        return {"score": 0.0, "spec_error": None, "parse_error": perr,
                "raw": raw, "reply": None, "hits": set(),
                "targets": {tuple(t) for t in ep["targets"]},
                "added_nongold": 0, "notes": [], "user_message": msg}
    new_pairs, n_moved, notes = execute_remedy(reply, recs, mx, my,
                                               gold_all)
    score, hits, targets, added_ng = score_v3(ep, new_pairs, gold_all)
    return {"score": score, "spec_error": None, "parse_error": None,
            "raw": raw, "reply": reply, "hits": hits, "targets": targets,
            "added_nongold": added_ng, "n_moved": n_moved, "notes": notes,
            "user_message": msg}


def feedback_v3(ep, r):
    if r["spec_error"]:
        return "REJECTED, scored 0: " + r["spec_error"]
    if r["parse_error"]:
        return f"reply not valid JSON: {r['parse_error']}"
    reply = r["reply"]
    parts = [
        f"remedy={reply.get('remedy')} failure_mode="
        f"{reply.get('failure_mode')} | {'; '.join(r['notes'])}",
        f"recovered {len(r['hits'])}/{len(r['targets'])} true split "
        f"pairs; added {r['added_nongold']} non-matching pairs "
        f"(each costs {ADD_PENALTY} — bulk merges only pay off when "
        "blocks truly overlap)",
    ]
    missed = r["targets"] - r["hits"]
    if missed:
        parts.append(f"{len(missed)} true pairs NOT captured by the "
                     "rule — the predicate was too narrow or targeted "
                     "the wrong field")
    return " | ".join(parts)
