"""Rule-vocabulary statistics behind Section 5 of the paper.

Reads the shipped eval_*.json files and prints, per dataset, the counters the
paper's "Shape of the emitted rules" paragraph and Table 2's sweep rows are
built from: operation mix, conditions per predicate, operator mix, field mix,
value length, single-token values, and inert predicates (n_moved == 0).

Definitions used by the paper:
  - a predicate is counted once per non-"none" action with a non-empty
    predicate list; conditions are its list entries;
  - "inert" means the predicate selected no record (n_moved == 0);
  - "single-token value" means the condition value has one whitespace token;
  - "full-group sweep" (per-blocker geometry, Section 5) means the predicate
    matched every record of the group it was applied to, N == M, where M is
    the group size and N the number matched, both read from each exemplar's
    harness note.

Usage: python3 rule_stats.py [runs_dir]   (default: ../runs/gepa/v2e_qwen38)
"""
import json, os, re, sys
from collections import Counter, defaultdict

R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "runs", "gepa", "v2e_qwen38")
NOTE_NM = re.compile(r"(\d+)\s*(?:of|/)\s*(\d+)")

def stats(fname):
    d = json.load(open(os.path.join(R, fname)))
    per = defaultdict(lambda: defaultdict(Counter))
    agg = defaultdict(lambda: dict(vlen=[], single=0, nval=0, inert=0, npred=0, steps=0, pf=0, full=0, sized=0))
    for r in d["rows"]:
        ds = r["cell"][1]; a = agg[ds]
        a["steps"] += r.get("steps_run", 0); a["pf"] += r.get("parse_failures", 0)
        for e in r["exemplars"]:
            per[ds]["ops"][e["op"]] += 1
            if e["op"] != "none" and e.get("predicate"):
                a["npred"] += 1; per[ds]["nc"][len(e["predicate"])] += 1
                if e.get("n_moved", None) == 0: a["inert"] += 1
                m = NOTE_NM.search(str(e.get("note", "")))
                if m:
                    n, M = int(m.group(1)), int(m.group(2)); a["sized"] += 1; a["full"] += (n == M)
                for c in e["predicate"]:
                    per[ds]["opr"][c.get("operator")] += 1; per[ds]["fld"][c.get("field")] += 1
                    v = c.get("value", ""); v = " ".join(map(str, v)) if isinstance(v, list) else str(v)
                    a["vlen"].append(len(v)); a["nval"] += 1; a["single"] += (len(v.split()) == 1)
    return per, agg

if __name__ == "__main__":
    for f in ("eval_val_best.json", "eval_val_seed.json", "eval_val_exh_best.json", "eval_val_exh_seed.json"):
        if not os.path.exists(os.path.join(R, f)):
            print(f, "missing"); continue
        per, agg = stats(f)
        print("==", f)
        for ds in sorted(agg):
            a = agg[ds]; vl = sorted(a["vlen"]); med = vl[len(vl)//2] if vl else None
            print(f"  {ds:16s} preds={a['npred']:5d} inert={a['inert']:4d} 1-cond={per[ds]['nc'][1]:5d} "
                  f"single-token={a['single']:5d}/{a['nval']:5d} value-median={med} "
                  f"full-group={a['full']}/{a['sized']} ops={dict(per[ds]['ops'])} opr={dict(per[ds]['opr'])}")
