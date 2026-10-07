"""run6 harness: BLOCK-CONTENTS scaffold with an evolvable FILTER.

Per user spec:
- The judge sees the two blocks as RECORD LISTS (all of block X's
  table-A records, all of block Y's table-B records), plus the seed.
- The evolvable sampler is a FILTER over each block: it returns
  (a_ids, b_ids) — which records of each side to show, in what order.
  Seed filter = identity (everything, id order).
- The judge answers with cross-list index pairs:
  {"matches": [[a_number, b_number], ...]}.
- Render truncates each list at a proportional share of CHAR_BUDGET;
  truncation is reported in feedback.

SECURITY NOTE: identical sandbox posture to harness4/harness5 —
LLM-evolved filter code runs under exec() with whitelisted builtins,
no import mechanism, no IO objects in scope, SIGALRM hard timeout,
output validated. See harness4 module docstring; do not widen.
"""
import json
import random
import signal
import traceback

import harness
import harness5

CHAR_BUDGET = 20000
MAX_PROPS = 25
FP_PENALTY = 0.02
CODE_TIMEOUT_S = 20

PAIRS_SCHEMA = {
    "type": "object",
    "properties": {
        "matches": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "integer"},
                      "minItems": 2, "maxItems": 2},
        },
    },
    "required": ["matches"],
}

SEED_FILTER = '''def sample(recs_x, recs_y, seed_pair, sim, rng):
    """FILTER the two blocks: decide which records the judge sees.

    recs_x: list of dicts — ALL records of block X (table A side).
            Every dict has "id" plus dataset-specific string fields.
    recs_y: same for block Y (table B side).
    seed_pair: (a_id, b_id) — confirmed true match the blocker split.
    sim(a_id, b_id) -> float — char-3gram TF-IDF cosine similarity.
    rng: seeded random.Random.

    Return (a_ids, b_ids): the record ids of each block to show, in
    display order. The harness renders each list until a shared
    character budget runs out — records beyond the cutoff are not
    shown, and record pairs where either side is unshown can never be
    recovered.
    """
    return [r["id"] for r in recs_x], [r["id"] for r in recs_y]
'''

_SANDBOX_GLOBALS = harness5._SANDBOX_GLOBALS


class _Timeout(Exception):
    pass


def _alarm(_sig, _frm):
    raise _Timeout(f"filter exceeded {CODE_TIMEOUT_S}s")


def run_filter(code, bank, ep):
    rx, ry, sim = harness5.episode_context(bank, ep)
    rng = random.Random(f"filter:{ep['episode_id']}")
    env = dict(_SANDBOX_GLOBALS)
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(CODE_TIMEOUT_S)
    try:
        # sandboxed exec of LLM-evolved filter — see module docstring
        exec(compile(code, "<filter_code>", "exec"), env)  # noqa: S102
        fn = env.get("sample")
        if not callable(fn):
            return [], [], "filter_code must define sample(...)"
        out = fn(rx, ry, tuple(ep["seed_pair"]), sim, rng)
    except _Timeout as e:
        return [], [], str(e)
    except Exception:  # noqa: BLE001
        return [], [], "filter raised:\n" + traceback.format_exc(limit=3)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    if not isinstance(out, (list, tuple)) or len(out) != 2:
        return [], [], "sample() must return (a_ids, b_ids)"
    va, vb = set(ep["a_ids"]), set(ep["b_ids"])

    def clean(lst, valid):
        if not isinstance(lst, (list, tuple)):
            return [], 1
        seen, kept, bad = set(), [], 0
        for i in lst:
            if i in valid and i not in seen:
                seen.add(i)
                kept.append(i)
            else:
                bad += 1
        return kept, bad

    a_ids, bad_a = clean(out[0], va)
    b_ids, bad_b = clean(out[1], vb)
    err = None
    if bad_a or bad_b:
        err = (f"{bad_a + bad_b} invalid/duplicate ids dropped "
               "(ids must come from the given blocks)")
    return a_ids, b_ids, err


def render_v6(bank, ep, a_ids, b_ids):
    recs = bank["records"]
    sa, sb = ep["seed_pair"]
    head = [
        "## Context",
        f"Dataset: {bank['dataset_metadata']}.",
        bank["method_metadata"],
        f"Block X: {len(ep['a_ids'])} table-A records. "
        f"Block Y: {len(ep['b_ids'])} table-B records.",
        "",
        "## SEED: confirmed true match wrongly split by the blocker",
        f"A: {harness5.fmt_fields(recs[sa])}",
        f"B: {harness5.fmt_fields(recs[sb])}",
        "",
    ]
    used = sum(len(x) + 1 for x in head) + 250
    share_a = (len(a_ids) / (len(a_ids) + len(b_ids))
               if a_ids or b_ids else 0.5)
    lists, shown = {}, {}
    for side, ids, share in (("A", a_ids, share_a),
                             ("B", b_ids, 1 - share_a)):
        cap = (CHAR_BUDGET - used) * share
        lines, spent, kept = [], 0, []
        for n, rid in enumerate(ids, 1):
            entry = f"{n}. {harness5.fmt_fields(recs[rid])}"
            if spent + len(entry) + 1 > cap:
                break
            lines.append(entry)
            spent += len(entry) + 1
            kept.append(rid)
        lists[side] = lines
        shown[side] = kept
    truncated = (len(shown["A"]) < len(a_ids)
                 or len(shown["B"]) < len(b_ids))
    body = (head
            + [f"## List A — records from block X "
               f"({len(shown['A'])} of {len(ep['a_ids'])} in block)"]
            + lists["A"] + [""]
            + [f"## List B — records from block Y "
               f"({len(shown['B'])} of {len(ep['b_ids'])} in block)"]
            + lists["B"]
            + ["", "Find pairs (one record from list A, one from list "
               "B) that refer to the same real-world entity. Return "
               'JSON {"matches": [[a_number, b_number], ...]}'])
    return "\n".join(body), shown["A"], shown["B"], truncated


def run_episode_v6(client, instruction, filter_code, bank, ep, tag=""):
    a_ids, b_ids, ferr = run_filter(filter_code, bank, ep)
    targets = {tuple(t) for t in ep["targets"]}
    if not a_ids or not b_ids:
        return {"score": 0.0, "filter_error": ferr or "empty filter",
                "parse_error": None, "raw": "", "shown_a": [],
                "shown_b": [], "truncated": False, "props": set(),
                "hits": set(), "targets": targets, "coverage": 0,
                "user_message": ""}
    user, ids_a, ids_b, trunc = render_v6(bank, ep, a_ids, b_ids)
    raw = client.chat([{"role": "system", "content": instruction},
                       {"role": "user", "content": user}],
                      fmt=PAIRS_SCHEMA, tag=tag)
    try:
        matches = json.loads(raw).get("matches", [])
        perr = None
    except Exception as e:  # noqa: BLE001
        matches, perr = [], f"{type(e).__name__}: {e}"
    seed = tuple(ep["seed_pair"])
    props = set()
    for m in matches[:MAX_PROPS]:
        if (isinstance(m, list) and len(m) == 2
                and isinstance(m[0], int) and isinstance(m[1], int)
                and 1 <= m[0] <= len(ids_a) and 1 <= m[1] <= len(ids_b)):
            p = (ids_a[m[0] - 1], ids_b[m[1] - 1])
            if p != seed:
                props.add(p)
    hits = props & targets
    sa_, sb_ = set(ids_a), set(ids_b)
    cov = sum(1 for a, b in targets if a in sa_ and b in sb_)
    recall = len(hits) / len(targets) if targets else 0.0
    score = max(0.0, recall - FP_PENALTY * len(props - hits))
    return {"score": score, "filter_error": ferr, "parse_error": perr,
            "raw": raw, "shown_a": ids_a, "shown_b": ids_b,
            "truncated": trunc, "props": props, "hits": hits,
            "targets": targets, "coverage": cov, "user_message": user}


def feedback_v6(bank, ep, r, component, lex, mapping):
    """BLIND feedback for the reflection LLM: no dataset name, no
    blocker identity, content tokens masked via lex/mapping."""
    recs = bank["records"]
    tgt = r["targets"]

    def mtext(rid):
        return lex.mask_text(
            " ".join(str(v) for v in recs[rid].values())[:80], mapping)

    base = (f"score={r['score']:.2f}: {len(r['hits'])}/{len(tgt)} true "
            f"split pairs recovered, {len(r['props'] - r['hits'])} "
            f"false positives. Filter kept "
            f"{len(r['shown_a'])}/{len(ep['a_ids'])} block-X records "
            f"and {len(r['shown_b'])}/{len(ep['b_ids'])} block-Y "
            "records in the prompt"
            + (" (render TRUNCATED at character budget)"
               if r["truncated"] else "")
            + f"; {r['coverage']}/{len(tgt)} true pairs had both "
            "records shown (a pair with an unshown record is "
            "unrecoverable).")
    if r["filter_error"]:
        base += f" FILTER ERROR: {r['filter_error']}"
    if r["parse_error"]:
        base += f" JUDGE PARSE ERROR: {r['parse_error']}"
    if component == "filter_code":
        sa_, sb_ = set(r["shown_a"]), set(r["shown_b"])
        missing = [t for t in sorted(tgt)
                   if not (t[0] in sa_ and t[1] in sb_)]
        if missing:
            base += " TRUE PAIRS LOST BY FILTER/BUDGET: " + "; ".join(
                f"'{mtext(a)}' == '{mtext(b)}'" for a, b in missing[:3])
        base += f" Block sizes {len(ep['a_ids'])}x{len(ep['b_ids'])}."
    else:
        fp = sorted(r["props"] - r["hits"])[:3]
        shown_pairs = {(a, b) for a in r["shown_a"]
                       for b in r["shown_b"]}
        fn = sorted((tgt & shown_pairs) - r["hits"])[:3]
        if fp:
            base += " FALSE: " + "; ".join(
                f"'{mtext(a)}' != '{mtext(b)}'" for a, b in fp)
        if fn:
            base += " MISSED-though-shown: " + "; ".join(
                f"'{mtext(a)}' == '{mtext(b)}'" for a, b in fn)
    return base
