"""Cost of the trivial repair: merge the two blocks at every boundary the
optimized sweep visits, in visit order, and report the relative growth of the
candidate set per cell (Table 3, column mrg.). Run from v2e/pc_gepa on the server."""
import json, os, sys, random
sys.path.insert(0, os.getcwd()); import v2
R=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs", "gepa", "v2e_qwen38") + os.sep
best={tuple(r["cell"]):r for r in json.load(open(R+"eval_val_exh_best.json"))["rows"]}
for (blk,ds),row in sorted(best.items()):
    cell=v2.load_cell(blk,ds)
    cell["usable_block_pairs"]=v2.usable_block_pairs(cell)[0]
    inst=v2.make_instance(cell, row["instance_id"], 700, "random")
    a=dict(cell["assignment"]); base=v2.candidate_count(a)
    gold={tuple(p) for p in cell["gold"]}
    # merge the two blocks at every visited boundary, in visit order, as the sweep would
    for st in inst["steps"]:
        sa,sb=st["seed"]; ba,bb=a[sa],a[sb]
        if ba==bb: continue
        for r,b in list(a.items()):
            if b==bb: a[r]=ba
    c=v2.candidate_count(a)
    print(f"{blk}/{ds} merge-both dC_rel {100*(c-base)/base:+.1f}%")
