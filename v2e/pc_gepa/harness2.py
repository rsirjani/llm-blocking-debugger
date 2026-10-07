"""v2 harness: the ENTIRE prompt scaffold is driven by an evolvable JSON spec.

GEPA candidate = {"instruction": <system prompt text>,
                  "sampling_spec": <JSON string in the DSL below>}

The harness interprets sampling_spec deterministically; the reflection LLM
mutates it as text, exactly like the instruction. Score = target recall
(per-episode PC recovery) minus a small false-positive penalty.

sampling_spec DSL (all keys optional, defaults shown):
{
  "default_sampler": {"name": "global_topm", "m": 40},
      // one of:
      //   {"name": "full"}                       all cross pairs
      //   {"name": "global_topm", "m": int}      top-m pairs by TF-IDF
      //   {"name": "per_record_topk", "k": int}  for every A-record of
      //                                          block X its k nearest
      //                                          B-records of block Y
      //   {"name": "hybrid", "m": int, "k": int} union of the above two
  "rules": [ {"if_max_side_le": 40, "sampler": {"name": "full"}} ],
      // first matching rule overrides default_sampler;
      // max side = max(|A-side of X|, |B-side of Y|)
  "max_text": 180,          // chars of each record text shown
  "show_sim": true,         // show TF-IDF similarity per pair
  "show_method_metadata": true,
  "show_block_stats": true,
  "show_seed": true,
  "char_budget": 24000      // hard cap on rendered candidate list; pair
                            // list is cut off there (coverage loss is
                            // reported in feedback)
}

Similarities are char-3gram TF-IDF cosine on the episode's two record
sets (gold-blind). Pair list is always sorted by similarity descending
and numbered 1..N; the model answers with match_indices.
"""
import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import harness

MAX_PAIRS = 120          # absolute cap regardless of spec
MAX_PROPS = 25           # proposal cap (budget asymmetry: additions liberal)
FP_PENALTY = 0.02        # score = recall - FP_PENALTY * n_false_positives

DEFAULT_SPEC = {
    "default_sampler": {"name": "global_topm", "m": 40},
    "rules": [],
    "max_text": 180,
    "show_sim": True,
    "show_method_metadata": True,
    "show_block_stats": True,
    "show_seed": True,
    "char_budget": 24000,
}

SEED_SPEC_TEXT = json.dumps(DEFAULT_SPEC, indent=2)

_VEC_CACHE = {}


def _sim_matrix(ep, data):
    """Cosine sims between A-side records of block X and B-side of Y."""
    key = ep["episode_id"]
    if key in _VEC_CACHE:
        return _VEC_CACHE[key]
    ta, tb = data["record_texts_a"], data["record_texts_b"]
    ids1, ids2 = ep["block1_a_ids"], ep["block2_b_ids"]
    vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 3))
    vec.fit([ta[i] for i in ids1] + [tb[i] for i in ids2])
    sim = cosine_similarity(vec.transform([ta[i] for i in ids1]),
                            vec.transform([tb[i] for i in ids2]))
    _VEC_CACHE[key] = (ids1, ids2, sim)
    if len(_VEC_CACHE) > 40:
        _VEC_CACHE.pop(next(iter(_VEC_CACHE)))
    return _VEC_CACHE[key]


def parse_spec(spec_text):
    """Returns (spec_dict, error_or_None). Unknown keys rejected loudly —
    they are the reflection LLM inventing DSL that does nothing."""
    try:
        raw = json.loads(spec_text)
    except Exception as e:  # noqa: BLE001
        return None, f"sampling_spec is not valid JSON: {e}"
    if not isinstance(raw, dict):
        return None, "sampling_spec must be a JSON object"
    unknown = set(raw) - set(DEFAULT_SPEC)
    if unknown:
        return None, (f"unknown keys {sorted(unknown)}; allowed: "
                      f"{sorted(DEFAULT_SPEC)}")
    spec = {**DEFAULT_SPEC, **raw}
    samplers = ("full", "global_topm", "per_record_topk", "hybrid")

    def _check_sampler(s):
        if not isinstance(s, dict) or s.get("name") not in samplers:
            return f"sampler must be an object with name in {samplers}"
        if s["name"] in ("global_topm", "hybrid") and not (
                isinstance(s.get("m"), int) and 1 <= s["m"] <= MAX_PAIRS):
            return f"global_topm/hybrid needs int m in 1..{MAX_PAIRS}"
        if s["name"] in ("per_record_topk", "hybrid") and not (
                isinstance(s.get("k"), int) and 1 <= s["k"] <= 5):
            return "per_record_topk/hybrid needs int k in 1..5"
        return None

    err = _check_sampler(spec["default_sampler"])
    if err:
        return None, "default_sampler: " + err
    if not isinstance(spec["rules"], list) or len(spec["rules"]) > 4:
        return None, "rules must be a list of at most 4 rule objects"
    for r in spec["rules"]:
        if (not isinstance(r, dict)
                or not isinstance(r.get("if_max_side_le"), int)
                or "sampler" not in r):
            return None, ('each rule needs {"if_max_side_le": int, '
                          '"sampler": {...}}')
        err = _check_sampler(r["sampler"])
        if err:
            return None, "rule sampler: " + err
    if not (isinstance(spec["max_text"], int)
            and 40 <= spec["max_text"] <= 400):
        return None, "max_text must be int in 40..400"
    if not (isinstance(spec["char_budget"], int)
            and 4000 <= spec["char_budget"] <= 60000):
        return None, "char_budget must be int in 4000..60000"
    return spec, None


def sample_pairs(ep, data, spec):
    """Apply the spec's sampler. Returns (pairs, sampler_used) where
    pairs = [(a_id, b_id, sim)] sorted by sim desc, capped at MAX_PAIRS."""
    ids1, ids2, sim = _sim_matrix(ep, data)
    max_side = max(len(ids1), len(ids2))
    sampler = spec["default_sampler"]
    for r in spec["rules"]:
        if max_side <= r["if_max_side_le"]:
            sampler = r["sampler"]
            break
    name = sampler["name"]
    chosen = set()
    if name in ("full",):
        chosen = {(i, j) for i in range(len(ids1))
                  for j in range(len(ids2))}
    if name in ("global_topm", "hybrid"):
        m = sampler.get("m", 40)
        flat = [(float(sim[i, j]), i, j) for i in range(len(ids1))
                for j in range(len(ids2))]
        flat.sort(key=lambda t: (-t[0], t[1], t[2]))
        chosen |= {(i, j) for _, i, j in flat[:m]}
    if name in ("per_record_topk", "hybrid"):
        k = sampler.get("k", 2)
        for i in range(len(ids1)):
            order = np.argsort(-sim[i])[:k]
            chosen |= {(i, int(j)) for j in order}
    pairs = sorted(((float(sim[i, j]), ids1[i], ids2[j])
                    for i, j in chosen),
                   key=lambda t: (-t[0], t[1], t[2]))[:MAX_PAIRS]
    return [(a, b, s) for s, a, b in pairs], name


def render_v2(ep, data, spec):
    """Returns (user_message, shown_pairs, render_info)."""
    ta, tb = data["record_texts_a"], data["record_texts_b"]
    mt = spec["max_text"]
    pairs, sampler_used = sample_pairs(ep, data, spec)
    lines = []
    if spec["show_method_metadata"]:
        lines += ["## Blocking method", data["method_metadata"], ""]
    if spec["show_block_stats"]:
        lines += [
            "## The two blocks being debugged",
            f"Block X (signature {ep['block1_sig']}): "
            f"{ep['n_block1_total']} records total, "
            f"{len(ep['block1_a_ids'])} from table A.",
            f"Block Y (signature {ep['block2_sig']}): "
            f"{ep['n_block2_total']} records total, "
            f"{len(ep['block2_b_ids'])} from table B.", ""]
    if spec["show_seed"]:
        sa, sb = ep["seed_pair"]
        lines += [
            "## SEED: confirmed true match wrongly split by the blocker",
            f"A-record (in block X): {ta[sa][:mt]}",
            f"B-record (in block Y): {tb[sb][:mt]}", ""]
    lines.append("## Candidate pairs (A-record from block X vs B-record "
                 "from block Y)")
    head_chars = sum(len(x) + 1 for x in lines)
    shown = []
    truncated = False
    for a, b, s in pairs:
        entry = [f"{len(shown) + 1}." + (f" [sim {s:.2f}]"
                                         if spec["show_sim"] else ""),
                 f"   A: {ta[a][:mt]}", f"   B: {tb[b][:mt]}"]
        head_chars += sum(len(x) + 1 for x in entry)
        if head_chars > spec["char_budget"]:
            truncated = True
            break
        lines += entry
        shown.append((a, b))
    lines += ["", 'Return JSON: {"match_indices": [...]}']
    info = {"sampler_used": sampler_used, "n_sampled": len(pairs),
            "n_shown": len(shown), "truncated": truncated,
            "chars": head_chars}
    return "\n".join(lines), shown, info


def score_v2(ep, match_indices, shown):
    seed = tuple(ep["seed_pair"])
    props = set()
    for i in match_indices[:MAX_PROPS]:
        if isinstance(i, int) and 1 <= i <= len(shown):
            p = shown[i - 1]
            if p != seed:
                props.add(p)
    targets = {tuple(t) for t in ep["targets"]}
    hits = props & targets
    recall = len(hits) / len(targets) if targets else 0.0
    score = max(0.0, recall - FP_PENALTY * len(props - hits))
    return score, props, hits, targets


def coverage(ep, shown):
    targets = {tuple(t) for t in ep["targets"]}
    return (sum(1 for t in targets if t in set(shown)), len(targets))


def feedback_v2(ep, data, props, hits, targets, shown, info,
                spec_error=None, parse_error=None):
    ta, tb = data["record_texts_a"], data["record_texts_b"]
    if spec_error:
        return ("sampling_spec REJECTED, episode scored 0: " + spec_error
                + " | Follow the DSL exactly; no other keys exist.")
    cov_hit, cov_tot = coverage(ep, shown)
    parts = [
        f"sampler={info['sampler_used']} shown={info['n_shown']} pairs "
        f"(~{info['chars']} chars"
        + (", TRUNCATED at char_budget" if info["truncated"] else "") + ")",
        f"coverage: {cov_hit}/{cov_tot} true pairs present in shown list "
        "(pairs not shown are unrecoverable no matter the instruction)",
    ]
    if parse_error:
        parts.append(f"reply unusable: {parse_error}")
        return " | ".join(parts)
    fp = props - targets
    fn_shown = [p for p in (targets - props) if p in set(shown)]
    parts.append(f"result: {len(hits)} correct, {len(fp)} false, "
                 f"{len(fn_shown)} shown-but-missed")
    if fp:
        parts.append("FALSE: " + "; ".join(
            f"{ta[a][:70]} != {tb[b][:70]}" for a, b in sorted(fp)[:4]))
    if fn_shown:
        parts.append("MISSED though shown: " + "; ".join(
            f"{ta[a][:70]} == {tb[b][:70]}"
            for a, b in sorted(fn_shown)[:4]))
    return " | ".join(parts)


def run_episode_v2(client, instruction, spec_text, ep, data, tag=""):
    spec, spec_err = parse_spec(spec_text)
    if spec_err:
        return {"raw": "", "indices": [], "parse_error": None,
                "spec_error": spec_err, "score": 0.0, "props": set(),
                "hits": set(), "targets": {tuple(t) for t in ep["targets"]},
                "shown": [], "info": {}, "user_message": ""}
    user, shown, info = render_v2(ep, data, spec)
    messages = [{"role": "system", "content": instruction},
                {"role": "user", "content": user}]
    raw = client.chat(messages, fmt=harness.MATCH_SCHEMA, tag=tag)
    try:
        idx = json.loads(raw).get("match_indices", [])
        idx = [i for i in idx if isinstance(i, int)]
        err = None
    except Exception as e:  # noqa: BLE001
        idx, err = [], f"{type(e).__name__}: {e}"
    score, props, hits, targets = score_v2(ep, idx, shown)
    return {"raw": raw, "indices": idx, "parse_error": err,
            "spec_error": None, "score": score, "props": props,
            "hits": hits, "targets": targets, "shown": shown,
            "info": info, "user_message": user}
