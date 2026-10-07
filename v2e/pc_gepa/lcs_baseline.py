"""Model-free baseline: the entity-fingerprint rule, written by a script.

For each step the replay would send to the judge, this emits the rule the
optimized instruction's first step describes, with no model call:
  - field: the field, present in both disclosed records, on which the two
    records share the longest common substring (case-insensitive);
  - predicate: that field `contains` the shared substring (>= 4 characters),
    else a decline;
  - direction: move matching records out of the larger block into the
    smaller one (step 5 of the optimized instruction).
Everything else (instances, step order, application, scoring, output) is
v2.py's own; only the client and the template filling are wrapped.

Usage (same arguments as an eval run of v2.py), e.g.
  python lcs_baseline.py --blockers ... --pairs-per-instance 700 \
     --eval-split val --variants 1 --tag exh_lcs --run-dir ...
"""
import difflib, json, threading
import harness, v2

_ctx = threading.local()
_orig_fill = v2.fill_template

def _fill(template, cell, x_ids, y_ids, seed, rng, budget):
    _ctx.cell, _ctx.x, _ctx.y, _ctx.seed = cell, list(x_ids), list(y_ids), seed
    return _orig_fill(template, cell, x_ids, y_ids, seed, rng, budget)

def _rule():
    cell, seed = _ctx.cell, _ctx.seed
    pair = seed[0] if seed and isinstance(seed[0], (list, tuple)) else seed
    a, b = cell["records"][str(pair[0])] if str(pair[0]) in cell["records"] else cell["records"][pair[0]], \
           cell["records"][str(pair[1])] if str(pair[1]) in cell["records"] else cell["records"][pair[1]]
    best = ("", "")
    for f in set(a) & set(b):
        va, vb = str(a.get(f) or ""), str(b.get(f) or "")
        if not va or not vb:
            continue
        sm = difflib.SequenceMatcher(None, va.lower(), vb.lower(), autojunk=False)
        m = sm.find_longest_match(0, len(va), 0, len(vb))
        s = va[m.a:m.a + m.size].strip()
        if len(s) > len(best[1]):
            best = (f, s)
    if len(best[1]) < 4:
        return {"operation": "none", "predicate": [], "reason": "no shared text"}
    op = "move_to_x" if len(_ctx.y) > len(_ctx.x) else "move_to_y"
    return {"operation": op,
            "predicate": [{"field": best[0], "operator": "contains", "value": best[1]}],
            "reason": "longest shared text of the disclosed pair"}

class FingerprintClient:
    def __init__(self, *a, **k):
        pass
    def chat(self, messages, fmt=None, tag=""):
        return json.dumps(_rule())

v2.fill_template = _fill
harness.OllamaClient = FingerprintClient

if __name__ == "__main__":
    v2.main()
