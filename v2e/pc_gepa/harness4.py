"""v4 harness: PIPELINE evolution — the sampler is evolved Python code.

Candidate = {"instruction": <system prompt>, "sampler_code": <python src>}.
The sampler is a real pipeline stage (never LLM-invoked): it selects which
candidate pairs the cheap judge model sees. The reflection LLM (Claude
Sonnet via CLI) rewrites BOTH components from execution feedback; score =
target recall − 0.02 × false positives (identical accounting to run2).

SECURITY NOTE (deliberate, user-requested design): this module executes
LLM-GENERATED python (the evolved sampler) via exec(). Mitigations:
whitelisted builtins only, no import mechanism, no file/network objects in
scope, SIGALRM hard timeout, output schema-validated, runs on the local
box under the user's own account. This is the FunSearch/AlphaEvolve
pattern; do not widen the sandbox without revisiting these mitigations.
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

MAX_SHOW = 60
MAX_PROPS = 25
FP_PENALTY = 0.02
CODE_TIMEOUT_S = 10

SEED_SAMPLER = '''def sample(recs_x, recs_y, seed_pair, sim, rng):
    """Return up to 60 (a_id, b_id) candidate pairs for the judge.

    recs_x: list of dicts {"id","title","manufacturer","price",
            "description"} — records of block X from table A (Amazon).
    recs_y: same for block Y from table B (Google).
    seed_pair: (a_id, b_id) — confirmed true match the blocker split.
    sim(a_id, b_id) -> float — char-3gram TF-IDF cosine similarity.
    rng: seeded random.Random — use it for any randomness.
    """
    pairs = [(a["id"], b["id"]) for a in recs_x for b in recs_y]
    pairs.sort(key=lambda p: -sim(p[0], p[1]))
    return pairs[:40]
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

_SIM_CACHE = {}


def episode_context(ep, recs):
    """(recs_x, recs_y, sim_fn) for the sampler, cached per episode."""
    key = ep["episode_id"]
    if key not in _SIM_CACHE:
        ids1 = ep["block1_a_ids"]
        ids2 = ep["block2_b_ids"]
        tx = [" ".join((recs[i]["title"], recs[i]["manufacturer"]))
              for i in ids1]
        ty = [" ".join((recs[i]["title"], recs[i]["manufacturer"]))
              for i in ids2]
        vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 3))
        vec.fit(tx + ty)
        m = cosine_similarity(vec.transform(tx), vec.transform(ty))
        lut = {(a, b): float(m[i, j]) for i, a in enumerate(ids1)
               for j, b in enumerate(ids2)}
        _SIM_CACHE[key] = lut
        if len(_SIM_CACHE) > 60:
            _SIM_CACHE.pop(next(iter(_SIM_CACHE)))
    lut = _SIM_CACHE[key]

    def sim(a, b):
        return lut.get((a, b), 0.0)

    def slim(i):
        r = recs[i]
        return {"id": i, "title": r["title"],
                "manufacturer": r["manufacturer"], "price": r["price"],
                "description": r["description"][:300]}

    rx = [slim(i) for i in ep["block1_a_ids"]]
    ry = [slim(i) for i in ep["block2_b_ids"]]
    return rx, ry, sim


class _Timeout(Exception):
    pass


def _alarm(_sig, _frm):
    raise _Timeout(f"sampler exceeded {CODE_TIMEOUT_S}s")


def run_sampler(code, ep, recs):
    """Execute evolved sampler in the sandbox. (pairs, error_or_None)."""
    rx, ry, sim = episode_context(ep, recs)
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
    except Exception:  # noqa: BLE001 — traceback is the feedback
        return [], "sampler raised:\n" + traceback.format_exc(limit=3)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    if not isinstance(out, (list, tuple)):
        return [], f"sample() returned {type(out).__name__}, need list"
    valid_a = set(ep["block1_a_ids"])
    valid_b = set(ep["block2_b_ids"])
    pairs, seen, bad = [], set(), 0
    for p in out:
        if (isinstance(p, (list, tuple)) and len(p) == 2
                and p[0] in valid_a and p[1] in valid_b):
            t = (p[0], p[1])
            if t not in seen:
                seen.add(t)
                pairs.append(t)
        else:
            bad += 1
    err = None
    if bad:
        err = (f"{bad} returned items were not valid (a_id from X, "
               "b_id from Y) pairs and were dropped")
    return pairs[:MAX_SHOW], err


def render_v4(ep, recs, pairs, sim):
    sa, sb = ep["seed_pair"]

    def txt(i):
        r = recs[i]
        s = r["title"]
        if r["manufacturer"]:
            s += " | mfr: " + r["manufacturer"]
        if r["price"]:
            s += " | price: " + r["price"]
        return s[:200]

    lines = [
        "## Task context",
        "Two blocks produced by a Lambda-fold LSH blocker (K=15, char "
        "bigrams, Bloom filter). True matches can be split by spelling "
        "variants, word order, or extra tokens.",
        f"Block X: {len(ep['block1_a_ids'])} Amazon records. "
        f"Block Y: {len(ep['block2_b_ids'])} Google records.",
        "",
        "## SEED: confirmed true match wrongly split by the blocker",
        f"A: {txt(sa)}",
        f"B: {txt(sb)}",
        "",
        "## Candidate pairs",
    ]
    for i, (a, b) in enumerate(pairs, 1):
        lines.append(f"{i}. [sim {sim(a, b):.2f}]")
        lines.append(f"   A: {txt(a)}")
        lines.append(f"   B: {txt(b)}")
    lines += ["", 'Return JSON: {"match_indices": [...]}']
    return "\n".join(lines)


def run_episode_v4(client, instruction, sampler_code, ep, recs, tag=""):
    pairs, samp_err = run_sampler(sampler_code, ep, recs)
    targets = {tuple(t) for t in ep["targets"]}
    if not pairs:
        return {"score": 0.0, "sampler_error": samp_err or "empty sample",
                "parse_error": None, "raw": "", "pairs": [],
                "props": set(), "hits": set(), "targets": targets,
                "coverage": 0, "user_message": ""}
    _, _, sim = episode_context(ep, recs)
    user = render_v4(ep, recs, pairs, sim)
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
    props = {pairs[i - 1] for i in idx[:MAX_PROPS]
             if 1 <= i <= len(pairs)} - {seed}
    hits = props & targets
    recall = len(hits) / len(targets) if targets else 0.0
    score = max(0.0, recall - FP_PENALTY * len(props - hits))
    return {"score": score, "sampler_error": samp_err,
            "parse_error": perr, "raw": raw, "pairs": pairs,
            "props": props, "hits": hits, "targets": targets,
            "coverage": sum(1 for t in targets if t in set(pairs)),
            "user_message": user}


def feedback_v4(ep, recs, r, component):
    tgt = r["targets"]
    base = (f"score={r['score']:.2f}: {len(r['hits'])} of {len(tgt)} "
            f"true split pairs recovered, "
            f"{len(r['props'] - r['hits'])} false positives. "
            f"Sampler showed {len(r['pairs'])} pairs covering "
            f"{r['coverage']}/{len(tgt)} true pairs (uncovered pairs are "
            "unrecoverable no matter the judge prompt).")
    if r["sampler_error"]:
        base += f" SAMPLER ERROR: {r['sampler_error']}"
    if r["parse_error"]:
        base += f" JUDGE PARSE ERROR: {r['parse_error']}"
    if component == "sampler_code":
        not_shown = [t for t in sorted(tgt) if t not in set(r["pairs"])]
        if not_shown:
            base += " TRUE PAIRS NOT SHOWN: " + "; ".join(
                f"'{recs[a]['title'][:60]}' == '{recs[b]['title'][:60]}'"
                for a, b in not_shown[:3])
        base += (f" Block sizes {len(ep['block1_a_ids'])}x"
                 f"{len(ep['block2_b_ids'])}; "
                 f"{len(r['pairs'])} pairs cost ~"
                 f"{len(r['pairs']) * 55} prompt tokens.")
    else:
        fp = sorted(r["props"] - r["hits"])[:3]
        fn = sorted((tgt & set(r["pairs"])) - r["hits"])[:3]
        if fp:
            base += " FALSE: " + "; ".join(
                f"'{recs[a]['title'][:60]}' != '{recs[b]['title'][:60]}'"
                for a, b in fp)
        if fn:
            base += " MISSED-though-shown: " + "; ".join(
                f"'{recs[a]['title'][:60]}' == '{recs[b]['title'][:60]}'"
                for a, b in fn)
    return base
