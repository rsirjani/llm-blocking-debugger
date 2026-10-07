"""run6 results package -> results_run6/.

Checkpoints = every candidate that set a NEW BEST val score, ranked by
relative gain. For each: prompt + filter, diff vs parent, and a real
re-rendered retrieval sample (what the judge actually saw).
"""
import difflib
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import harness6
from gepa_run5 import load_banks

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, os.pardir, "runs", "gepa",
                   "run6_blind_blockcontents")
OUT = os.path.join(HERE, "results_run6")
os.makedirs(OUT, exist_ok=True)

lin = json.load(open(os.path.join(RUN, "lineage.json")))
cands = json.load(open(os.path.join(RUN, "candidates.json")))
aggs, disc, parents = lin["aggs"], lin["discovery"], lin["parents"]

ckpts, best = [], 0.0
for i, a in enumerate(aggs):
    if a > best:
        rel = (a - best) / best if best > 0 else None
        p = parents[i][0]
        comp = ("seed" if p is None else
                ",".join(k for k in cands[i] if cands[i][k] != cands[p][k]))
        ckpts.append({"idx": i, "val": a, "rel_gain": rel,
                      "rollout": disc[i], "parent": p, "mutated": comp})
        best = a
json.dump(ckpts, open(os.path.join(OUT, "checkpoints.json"), "w"),
          indent=2, default=float)

# ---- curve ----
running, b = [], 0.0
for a in aggs:
    b = max(b, a)
    running.append(b)
plt.figure(figsize=(9, 5))
plt.scatter(disc, aggs, s=18, alpha=0.55, label="candidate val score")
plt.step(disc, running, where="post", color="crimson", lw=2,
         label="best so far")
for c in ckpts:
    lbl = (f"#{c['idx']}"
           + (f" +{c['rel_gain']*100:.0f}%" if c["rel_gain"] else ""))
    plt.annotate(lbl, (c["rollout"], c["val"]),
                 textcoords="offset points", xytext=(4, 6), fontsize=7)
plt.xlabel("task-LLM rollouts consumed at candidate discovery")
plt.ylabel("mixed-val episode score\n(hidden-target recall − 0.02·FP "
           "= per-episode PC recovery)")
plt.title("run6 (BLIND reflection, block-contents scaffold):\n"
          "pipeline evolution on mixed 4-dataset train bank")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "pc_vs_rollouts.png"), dpi=150)

# ---- evolution doc ----
banks = load_banks(["lambdafold"])
bank = banks[("lambdafold", "amazon-google")]
ep = next(e for e in bank["episodes"]
          if e["split"] == "train" and 60 <= len(e["a_ids"]) <= 120
          and len(e["b_ids"]) >= 100)
tgt_n = len(ep["targets"])


def sample_of(cand):
    a_ids, b_ids, err = harness6.run_filter(cand["filter_code"], bank, ep)
    if not a_ids:
        return f"(filter returned nothing; {err})", 0, 0, 0, False
    user, sa, sb, trunc = harness6.render_v6(bank, ep, a_ids, b_ids)
    cov = sum(1 for a, b in ep["targets"] if a in set(sa) and b in set(sb))
    lines = user.split("## List A")[1].split("\n")[:12]
    return "## List A" + "\n".join(lines), len(sa), len(sb), cov, trunc


md = ["# run6 evolution — checkpoints by relative val gain", "",
      f"Reference episode for retrieval samples: `{ep['episode_id']}` "
      f"(blocks {len(ep['a_ids'])}x{len(ep['b_ids'])} records, "
      f"{tgt_n} hidden targets). Filters are deterministic, so these "
      "re-renders are exactly what the judge saw.", "",
      "| # | val | rel. gain | rollout | mutated |", "|---|---|---|---|---|"]
for c in ckpts:
    rg = f"+{c['rel_gain']*100:.0f}%" if c["rel_gain"] else "— (seed)"
    md.append(f"| {c['idx']} | {c['val']:.3f} | {rg} | {c['rollout']} "
              f"| {c['mutated']} |")
md.append("")
for c in ckpts:
    i = c["idx"]
    samp, na, nb, cov, trunc = sample_of(cands[i])
    rg = f"+{c['rel_gain']*100:.0f}% relative" if c["rel_gain"] else "seed"
    md += [f"---\n\n## Checkpoint #{i} — val {c['val']:.3f} ({rg}), "
           f"rollout {c['rollout']}, mutated: {c['mutated']}", "",
           f"**Filter output on the reference episode:** showed {na} "
           f"A-records and {nb} B-records, covering {cov}/{tgt_n} "
           "hidden targets"
           + (" (render truncated at budget)" if trunc else "") + ".", "",
           "```", samp, "```", "",
           "### instruction", "```", cands[i]["instruction"].strip(),
           "```", "", "### filter_code", "```python",
           cands[i]["filter_code"].strip(), "```", ""]
    p = c["parent"]
    if p is not None:
        for k in ("instruction", "filter_code"):
            if cands[i][k] != cands[p][k]:
                d = list(difflib.unified_diff(
                    cands[p][k].splitlines(), cands[i][k].splitlines(),
                    lineterm="", n=1))[:80]
                md += [f"### {k}: diff vs parent #{p}", "```diff",
                       "\n".join(d), "```", ""]
open(os.path.join(OUT, "EVOLUTION.md"), "w", encoding="utf-8").write(
    "\n".join(md))
print(f"wrote {len(ckpts)} checkpoints, graph + EVOLUTION.md")
for c in ckpts:
    print(" #%d val=%.3f rel=%s rollout=%d %s" %
          (c["idx"], c["val"],
           f"{c['rel_gain']*100:.0f}%" if c["rel_gain"] else "seed",
           c["rollout"], c["mutated"]))
