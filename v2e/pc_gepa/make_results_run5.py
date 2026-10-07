"""Generate the organized run5 results package -> results_run5/.

- pc_vs_rollouts.png : val episode-score (PC-recovery proxy) vs rollouts
- EVOLUTION.md       : prompt+sampler at seed / each big jump / best,
                       with a real re-rendered retrieval sample each
- REPORT.md          : experiment + distribution description + final
                       numbers + cross-version comparison
Re-renders use the exact deterministic sampler on a fixed train episode,
reproducing byte-identically what the judge saw during the run.
"""
import difflib
import json
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import harness5
from gepa_run5 import load_banks

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, os.pardir, "runs", "gepa", "run5_general_sonnet")
OUT = os.path.join(HERE, "results_run5")
os.makedirs(OUT, exist_ok=True)

lineage = json.load(open(os.path.join(RUN, "lineage.json")))
cands = json.load(open(os.path.join(RUN, "candidates.json")))
aggs = lineage["aggs"]
disc = lineage["discovery"]
parents = lineage["parents"]
BEST = max(range(len(aggs)), key=lambda i: aggs[i])
SHOW = [0] + [j[0] for j in lineage["jumps"]] + [BEST]
SHOW = sorted(dict.fromkeys(SHOW))

# ---------- graph ----------
running = []
b = 0.0
for a in aggs:
    b = max(b, a)
    running.append(b)
plt.figure(figsize=(9, 5))
plt.scatter(disc, aggs, s=18, alpha=0.6, label="candidate val score")
plt.step(disc, running, where="post", color="crimson", lw=2,
         label="best so far")
for i in SHOW:
    plt.annotate(f"#{i}", (disc[i], aggs[i]), textcoords="offset points",
                 xytext=(4, 6), fontsize=8)
plt.xlabel("task-LLM rollouts consumed at candidate discovery")
plt.ylabel("mixed-val episode score\n(hidden-target recall − 0.02·FP "
           "≈ per-episode PC recovery)")
plt.title("run5: pipeline evolution on mixed 4-dataset train bank\n"
          "(seed = all-pairs dump, score 0.0005)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "pc_vs_rollouts.png"), dpi=150)
print("graph written")

# ---------- evolution ----------
banks = load_banks(["lambdafold"])
bank = banks[("lambdafold", "amazon-google")]
ep = next(e for e in bank["episodes"]
          if e["split"] == "train"
          and 60 <= len(e["a_ids"]) <= 120 and len(e["b_ids"]) >= 100)


def retrieval_sample(cand):
    pairs, err = harness5.run_sampler(cand["sampler_code"], bank, ep)
    if not pairs:
        return f"(sampler produced nothing; error: {err})", 0, 0, False
    user, shown, trunc = harness5.render_v5(bank, ep, pairs)
    tgt = {tuple(t) for t in ep["targets"]}
    cov = sum(1 for t in tgt if t in set(shown))
    head = user.split("## Candidate pairs")[1]
    lines = head.split("\n")[1:34]
    return "\n".join(lines), len(shown), cov, trunc


def comp_diff(i):
    p = parents[i][0]
    if p is None:
        return "seed"
    changed = [k for k in cands[i] if cands[i][k] != cands[p][k]]
    return f"parent #{p}, mutated: {', '.join(changed)}"


tgt_n = len(ep["targets"])
md = [
    "# run5 evolution: seed -> jumps -> best",
    "",
    f"Reference episode for all retrieval samples: `{ep['episode_id']}`"
    f" (blocks {len(ep['a_ids'])}x{len(ep['b_ids'])} records, "
    f"{tgt_n} hidden targets). Samplers are deterministic, so these "
    "re-renders reproduce exactly what the judge saw during the run.",
    "",
]
for i in SHOW:
    md += [f"---\n\n## Candidate #{i}  "
           f"(val {aggs[i]:.3f}, discovered at rollout {disc[i]}, "
           f"{comp_diff(i)})", ""]
    samp, n_shown, cov, trunc = retrieval_sample(cands[i])
    md += [f"**What its sampler actually retrieved on the reference "
           f"episode:** {n_shown} pairs shown to the judge, covering "
           f"{cov}/{tgt_n} hidden targets"
           + (", list TRUNCATED at prompt budget" if trunc else "")
           + ".", "", "Prompt excerpt as fed to the judge (first pairs):",
           "```", samp, "```", ""]
    md += ["### instruction (full)", "```",
           cands[i]["instruction"].strip(), "```", ""]
    md += ["### sampler_code (full)", "```python",
           cands[i]["sampler_code"].strip(), "```", ""]
    p = parents[i][0]
    if p is not None:
        for k in ("instruction", "sampler_code"):
            if cands[i][k] != cands[p][k]:
                diff = list(difflib.unified_diff(
                    cands[p][k].splitlines(), cands[i][k].splitlines(),
                    lineterm="", n=1))[:60]
                md += [f"### {k} diff vs parent #{p} (truncated)",
                       "```diff", "\n".join(diff), "```", ""]
with open(os.path.join(OUT, "EVOLUTION.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(md))
print("EVOLUTION.md written,", len(SHOW), "candidates")

# ---------- distribution stats for REPORT ----------
rows = []
for (blk, ds), b in sorted(banks.items()):
    eps = b["episodes"]
    dims = [(len(e["a_ids"]), len(e["b_ids"])) for e in eps]
    cross = sorted(a * c for a, c in dims)
    tgts = [len(e["targets"]) for e in eps]
    splits = {}
    for e in eps:
        splits[e["split"]] = splits.get(e["split"], 0) + 1
    rows.append({
        "dataset": ds, "domain": b["dataset_metadata"], "K": b["K"],
        "PC": round(b["PC"], 3), "gold": b["n_gold"],
        "missed": b["n_missed"], "episodes": len(eps), "splits": splits,
        "targets": sum(tgts),
        "med_targets_per_ep": statistics.median(tgts),
        "med_cross_pairs": int(statistics.median(cross)),
        "max_cross_pairs": max(cross),
    })
LEGEND = {
    "dataset": "benchmark name (two-table clean-clean ER dataset)",
    "domain": "what the records are and which two sources are matched",
    "K": ("lambda-fold blocker granularity: number of Bloom-filter bit "
          "positions sampled to form each record's block signature; "
          "auto-selected per dataset so PC lands closest to 0.6 "
          "(higher K = finer blocks = lower PC)"),
    "PC": ("pair completeness of the base blocking = fraction of gold "
           "matched pairs whose two records share a block, BEFORE any "
           "LLM debugging"),
    "gold": "total ground-truth matched pairs in the dataset",
    "missed": ("gold pairs the blocker split across two blocks "
               "(= (1-PC) * gold, the debuggable error mass)"),
    "episodes": ("debug episodes built from this dataset: directional "
                 "block-pairs containing >= 2 missed gold pairs "
                 "(1 seed + >= 1 hidden target each)"),
    "splits": ("episode counts per split; train datasets get train/val "
               "(75/25, seed 0), held-out datasets are test-only"),
    "targets": ("hidden target pairs across all episodes = missed gold "
                "pairs to recover, EXCLUDING the one exposed seed per "
                "episode; the denominator of recall"),
    "med_targets_per_ep": "median hidden targets per episode",
    "med_cross_pairs": ("median episode search-space size: |A-side "
                        "records of block X| x |B-side records of block "
                        "Y| = possible record pairs the sampler must "
                        "choose from in a typical episode"),
    "max_cross_pairs": ("same for the largest episode — worst-case "
                        "search space the sampler must survive "
                        "(20s time limit)"),
}
with open(os.path.join(OUT, "distribution_stats.json"), "w") as f:
    json.dump({"legend": LEGEND, "datasets": rows}, f, indent=2)
for r in rows:
    print(r["dataset"], r["PC"], r["episodes"], r["med_cross_pairs"],
          r["max_cross_pairs"])
