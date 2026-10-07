"""run5 harness: multi-dataset, multi-blocker, dumb-start sampler.

Differences from harness4:
- Generic records: dict of arbitrary string fields per dataset.
- Judge prompt shows dataset + blocker metadata; NO similarity scores.
- sim() available to evolved sampler code as a primitive (lazy
  matrix-backed — episodes can be 430 x 15,000 records).
- Seed sampler: ALL cross pairs in id order (no ranking). Physics caps
  only: MAX_SHOW pairs and CHAR_BUDGET on the rendered list; truncation
  is reported in feedback so the optimizer feels it.
Same sandbox + security posture as harness4 (see its module docstring):
LLM-evolved sampler code runs under exec() with whitelisted builtins,
no imports/IO, SIGALRM timeout, output validation.
"""
import builtins as _bi
import collections
import itertools
import json
import math
import random
import re
import signal
import traceback

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import harness

MAX_SHOW = 250
CHAR_BUDGET = 20000
MAX_PROPS = 25
FP_PENALTY = 0.02
CODE_TIMEOUT_S = 20

SEED_SAMPLER = '''def sample(recs_x, recs_y, seed_pair, sim, rng):
    """Return candidate (a_id, b_id) pairs for the judge.

    recs_x: list of dicts — records of block X (table A side). Every
            dict has "id" plus dataset-specific text fields (products:
            title/manufacturer/price/...; bibliographic: title/authors/
            venue/year; restaurants: name/addr/city/...).
    recs_y: same for block Y (table B side).
    seed_pair: (a_id, b_id) — confirmed true match the blocker split.
    sim(a_id, b_id) -> float — char-3gram TF-IDF cosine similarity.
    rng: seeded random.Random — use for any randomness.
    The harness shows at most 250 pairs to the judge (fewer if the
    prompt character budget is hit) — pairs beyond that are dropped.
    """
    pairs = [(a["id"], b["id"]) for a in recs_x for b in recs_y]
    pairs.sort()
    return pairs
'''

_ALLOWED_BUILTINS = {
    n: getattr(_bi, n) for n in (
        "len", "range", "min", "max", "sorted", "sum", "set", "list",
        "dict", "tuple", "enumerate", "zip", "abs", "round", "float",
        "int", "str", "bool", "any", "all", "map", "filter", "reversed",
        "isinstance", "ValueError", "KeyError", "IndexError",
        "Exception", "StopIteration", "frozenset", "next", "iter",
        "repr", "hash", "divmod", "pow")
}
_SANDBOX_GLOBALS = {
    "__builtins__": _ALLOWED_BUILTINS,
    "re": re, "math": math, "itertools": itertools,
    "Counter": collections.Counter,
    "defaultdict": collections.defaultdict,
}

_SIM_CACHE = collections.OrderedDict()


def rec_text(fields):
    return " ".join(str(v) for v in fields.values())


def episode_context(bank, ep):
    """(recs_x, recs_y, sim_fn) with lazy matrix-backed sim."""
    key = ep["episode_id"]
    if key not in _SIM_CACHE:
        recs = bank["records"]
        ids1, ids2 = ep["a_ids"], ep["b_ids"]
        vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 3))
        tx = [rec_text(recs[i])[:300] for i in ids1]
        ty = [rec_text(recs[i])[:300] for i in ids2]
        vec.fit(tx + ty)
        m = cosine_similarity(vec.transform(tx), vec.transform(ty))
        ia = {r: i for i, r in enumerate(ids1)}
        ib = {r: i for i, r in enumerate(ids2)}
        _SIM_CACHE[key] = (ia, ib, m)
        while len(_SIM_CACHE) > 20:
            _SIM_CACHE.popitem(last=False)
    ia, ib, m = _SIM_CACHE[key]

    def sim(a, b):
        i, j = ia.get(a), ib.get(b)
        return float(m[i, j]) if i is not None and j is not None else 0.0

    recs = bank["records"]
    rx = [{"id": i, **recs[i]} for i in ep["a_ids"]]
    ry = [{"id": i, **recs[i]} for i in ep["b_ids"]]
    return rx, ry, sim


class _Timeout(Exception):
    pass


def _alarm(_sig, _frm):
    raise _Timeout(f"sampler exceeded {CODE_TIMEOUT_S}s")


def run_sampler(code, bank, ep):
    rx, ry, sim = episode_context(bank, ep)
    rng = random.Random(f"sampler:{ep['episode_id']}")
    env = dict(_SANDBOX_GLOBALS)
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(CODE_TIMEOUT_S)
    try:
        # sandboxed exec of LLM-evolved code — see module docstring
        exec(compile(code, "<sampler_code>", "exec"), env)  # noqa: S102
        fn = env.get("sample")
        if not callable(fn):
            return [], "sampler_code must define sample(...)"
        out = fn(rx, ry, tuple(ep["seed_pair"]), sim, rng)
    except _Timeout as e:
        return [], str(e)
    except Exception:  # noqa: BLE001
        return [], "sampler raised:\n" + traceback.format_exc(limit=3)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    if not isinstance(out, (list, tuple)):
        return [], f"sample() returned {type(out).__name__}, need list"
    va, vb = set(ep["a_ids"]), set(ep["b_ids"])
    pairs, seen, bad = [], set(), 0
    for p in out:
        if (isinstance(p, (list, tuple)) and len(p) == 2
                and p[0] in va and p[1] in vb):
            t = (p[0], p[1])
            if t not in seen:
                seen.add(t)
                pairs.append(t)
        else:
            bad += 1
    err = (f"{bad} invalid items dropped (must be (a_id from X, b_id "
           f"from Y))") if bad else None
    return pairs[:MAX_SHOW], err


def fmt_fields(fields, per_field=110):
    return " | ".join(f"{k}: {str(v)[:per_field]}"
                      for k, v in fields.items())


def render_v5(bank, ep, pairs):
    recs = bank["records"]
    sa, sb = ep["seed_pair"]
    lines = [
        "## Context",
        f"Dataset: {bank['dataset_metadata']}.",
        bank["method_metadata"],
        f"Block X: {len(ep['a_ids'])} table-A records. "
        f"Block Y: {len(ep['b_ids'])} table-B records.",
        "",
        "## SEED: confirmed true match wrongly split by the blocker",
        f"A: {fmt_fields(recs[sa])}",
        f"B: {fmt_fields(recs[sb])}",
        "",
        "## Candidate pairs (A-record from X vs B-record from Y)",
    ]
    used = sum(len(x) + 1 for x in lines)
    shown = []
    truncated = False
    for a, b in pairs:
        entry = [f"{len(shown) + 1}.",
                 f"   A: {fmt_fields(recs[a])}",
                 f"   B: {fmt_fields(recs[b])}"]
        used += sum(len(x) + 1 for x in entry)
        if used > CHAR_BUDGET:
            truncated = True
            break
        lines += entry
        shown.append((a, b))
    lines += ["", 'Return JSON: {"match_indices": [...]}']
    return "\n".join(lines), shown, truncated


def run_episode_v5(client, instruction, sampler_code, bank, ep, tag=""):
    pairs, samp_err = run_sampler(sampler_code, bank, ep)
    targets = {tuple(t) for t in ep["targets"]}
    if not pairs:
        return {"score": 0.0, "sampler_error": samp_err or "empty sample",
                "parse_error": None, "raw": "", "shown": [],
                "n_sampled": 0, "truncated": False, "props": set(),
                "hits": set(), "targets": targets, "coverage": 0,
                "user_message": ""}
    user, shown, truncated = render_v5(bank, ep, pairs)
    raw = client.chat([{"role": "system", "content": instruction},
                       {"role": "user", "content": user}],
                      fmt=harness.MATCH_SCHEMA, tag=tag)
    try:
        idx = [i for i in json.loads(raw).get("match_indices", [])
               if isinstance(i, int)]
        perr = None
    except Exception as e:  # noqa: BLE001
        idx, perr = [], f"{type(e).__name__}: {e}"
    seed = tuple(ep["seed_pair"])
    props = {shown[i - 1] for i in idx[:MAX_PROPS]
             if 1 <= i <= len(shown)} - {seed}
    hits = props & targets
    recall = len(hits) / len(targets) if targets else 0.0
    score = max(0.0, recall - FP_PENALTY * len(props - hits))
    return {"score": score, "sampler_error": samp_err,
            "parse_error": perr, "raw": raw, "shown": shown,
            "n_sampled": len(pairs), "truncated": truncated,
            "props": props, "hits": hits, "targets": targets,
            "coverage": sum(1 for t in targets if t in set(shown)),
            "user_message": user}


def feedback_v5(bank, ep, r, component):
    recs = bank["records"]
    tgt = r["targets"]
    base = (f"[{bank['dataset']}] score={r['score']:.2f}: "
            f"{len(r['hits'])}/{len(tgt)} true split pairs recovered, "
            f"{len(r['props'] - r['hits'])} false positives. Sampler "
            f"returned {r['n_sampled']} pairs, judge saw "
            f"{len(r['shown'])}"
            + (" (TRUNCATED at prompt budget)" if r["truncated"] else "")
            + f", covering {r['coverage']}/{len(tgt)} true pairs "
            "(unshown = unrecoverable).")
    if r["sampler_error"]:
        base += f" SAMPLER ERROR: {r['sampler_error']}"
    if r["parse_error"]:
        base += f" JUDGE PARSE ERROR: {r['parse_error']}"

    def name(i):
        f = recs[i]
        return str(next(iter(f.values())))[:60] if f else i

    if component == "sampler_code":
        not_shown = [t for t in sorted(tgt)
                     if t not in set(r["shown"])]
        if not_shown:
            base += " TRUE PAIRS NOT SHOWN: " + "; ".join(
                f"'{name(a)}' == '{name(b)}'" for a, b in not_shown[:3])
        base += (f" Block sizes {len(ep['a_ids'])}x{len(ep['b_ids'])} "
                 f"= {len(ep['a_ids']) * len(ep['b_ids'])} possible "
                 "pairs.")
    else:
        fp = sorted(r["props"] - r["hits"])[:3]
        fn = sorted((tgt & set(r["shown"])) - r["hits"])[:3]
        if fp:
            base += " FALSE: " + "; ".join(
                f"'{name(a)}' != '{name(b)}'" for a, b in fp)
        if fn:
            base += " MISSED-though-shown: " + "; ".join(
                f"'{name(a)}' == '{name(b)}'" for a, b in fn)
    return base
