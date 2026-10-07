"""Publication figures for the run6 results section -> results_run6/fig*.png"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

plt.rcParams.update({
    "figure.dpi": 160, "savefig.dpi": 160, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "legend.frameon": False, "axes.titlesize": 10,
    "figure.constrained_layout.use": True,
})
C = {"seed": "#9aa5b1", "mid": "#5b8def", "best": "#e8663d",
     "filter": "#e8663d", "prompt": "#3f7d58", "base": "#c9ced6"}

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, os.pardir, "runs", "gepa",
                   "run6_blind_blockcontents")
BANK = os.path.join(HERE, os.pardir, "runs", "bank", "lambdafold")
OUT = os.path.join(HERE, "results_run6")
os.makedirs(OUT, exist_ok=True)

lin = json.load(open(os.path.join(RUN, "lineage.json")))
cands = json.load(open(os.path.join(RUN, "candidates.json")))
aggs, disc, parents = lin["aggs"], lin["discovery"], lin["parents"]
banks = {d: json.load(open(os.path.join(BANK, d + ".json")))
         for d in ("amazon-google", "walmart-amazon", "dblp-acm",
                   "fodors-zagats", "abt-buy", "dblp-scholar")}
ev = {k: json.load(open(os.path.join(RUN, f"eval_test_{k}.json")))
      for k in ("best", "mid16") if os.path.exists(
          os.path.join(RUN, f"eval_test_{k}.json"))}
seed_path = os.path.join(RUN, "eval_test_seed.json")
SEED_EVAL = (json.load(open(seed_path)) if os.path.exists(seed_path)
             else None)

# ---- fig 1: optimization curve ------------------------------------
ckpts, best = [], 0.0
for i, a in enumerate(aggs):
    if a > best:
        p = parents[i][0]
        comp = ("seed" if p is None else
                ",".join(k for k in cands[i] if cands[i][k] != cands[p][k]))
        ckpts.append((i, a, disc[i], comp,
                      None if best == 0 else (a - best) / best))
        best = a
fig, ax = plt.subplots(figsize=(6.4, 3.4))
run, b = [], 0.0
for a in aggs:
    b = max(b, a)
    run.append(b)
ax.scatter(disc, aggs, s=14, color="#9aa5b1", alpha=0.7,
           label="candidate", zorder=2)
ax.step(disc, run, where="post", color="#1f2933", lw=1.6,
        label="best so far", zorder=3)
for i, a, x, comp, rel in ckpts:
    col = C["filter"] if "filter" in comp else (
        C["prompt"] if "instruction" in comp else "#1f2933")
    ax.scatter([x], [a], s=34, color=col, zorder=4)
    if rel and rel > 0.08:
        ax.annotate(f"+{rel*100:.0f}%", (x, a), xytext=(3, 7),
                    textcoords="offset points", fontsize=7.5, color=col)
ax.scatter([], [], s=34, color=C["filter"], label="new best: retrieval code")
ax.scatter([], [], s=34, color=C["prompt"], label="new best: judge prompt")
ax.set_xlabel("task-model rollouts")
ax.set_ylabel("validation score\n(recall − 0.02·FP)")
ax.set_title("Pipeline optimization on the mixed four-dataset training bank")
ax.legend(loc="lower right", fontsize=8)
for ext in ("png","pdf"):
    fig.savefig(os.path.join(OUT, f"fig1_optimization.{ext}"))

# ---- fig 2: held-out PC before/after ------------------------------
if ev.get("best"):
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    rows = []
    for ds in ("abt-buy", "dblp-scholar"):
        b = banks[ds]
        rec = int(ev["best"]["per_dataset"][ds]["recovered"].split("/")[0])
        rows.append((ds, b["PC"], rec / b["n_gold"]))
    xs = range(len(rows))
    ax.bar(xs, [r[1] for r in rows], 0.55, color=C["base"],
           label="after blocking (base PC)")
    ax.bar(xs, [r[2] for r in rows], 0.55, bottom=[r[1] for r in rows],
           color=C["best"], label="recovered by the pipeline")
    for i, (ds, base, gain) in enumerate(rows):
        ax.text(i, base + gain + 0.012, f"+{gain*100:.1f} pts",
                ha="center", fontsize=8.5, color=C["best"])
        ax.text(i, base / 2, f"{base:.3f}", ha="center", fontsize=8.5,
                color="#3e4c59")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylim(0, 0.85)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylabel("pair completeness")
    ax.set_title("Held-out datasets: pair completeness recovered")
    ax.legend(fontsize=8, loc="upper left")
    for ext in ("png","pdf"):
        fig.savefig(os.path.join(OUT, f"fig2_heldout_pc.{ext}"))

# ---- fig 3: checkpoint progression, val vs held-out ---------------
pts = []
if SEED_EVAL:
    pts.append(("seed\n(#0)", aggs[0], SEED_EVAL, C["seed"]))
if ev.get("mid16"):
    pts.append(("midpoint\n(#16)", aggs[16], ev["mid16"], C["mid"]))
if ev.get("best"):
    pts.append(("optimized\n(#41)", aggs[41], ev["best"], C["best"]))
if pts:
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.9))
    labels = [p[0] for p in pts]
    xs = range(len(pts))
    axes[0].bar(xs, [p[1] for p in pts], 0.5,
                color=[p[3] for p in pts])
    axes[0].set_title("validation score")
    axes[1].bar(xs, [int(p[2]["targets_recovered"].split("/")[0]) /
                     int(p[2]["targets_recovered"].split("/")[1])
                     for p in pts], 0.5, color=[p[3] for p in pts])
    axes[1].set_title("held-out recall")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[2].bar(xs, [int(p[2]["coverage"].split("/")[0]) /
                     int(p[2]["coverage"].split("/")[1])
                     for p in pts], 0.5, color=[p[3] for p in pts])
    axes[2].set_title("held-out coverage\n(both records shown)")
    axes[2].yaxis.set_major_formatter(PercentFormatter(1.0))
    for ax in axes:
        ax.set_xticks(list(xs))
        ax.set_xticklabels(labels, fontsize=8)
    fig.suptitle("Three checkpoints: in-distribution selection tracks "
                 "out-of-distribution gain", fontsize=10)
    for ext in ("png","pdf"):
        fig.savefig(os.path.join(OUT, f"fig3_checkpoints.{ext}"))

# ---- fig 4: error population -------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 3.0))
names = ["amazon-google", "walmart-amazon", "dblp-acm", "fodors-zagats",
         "abt-buy", "dblp-scholar"]
seeds, tgts, orph = [], [], []
for d in names:
    b = banks[d]
    s = len(b["episodes"])
    t = sum(len(e["targets"]) for e in b["episodes"])
    seeds.append(s / b["n_missed"])
    tgts.append(t / b["n_missed"])
    orph.append((b["n_missed"] - s - t) / b["n_missed"])
xs = range(len(names))
ax.bar(xs, seeds, 0.6, color="#3f7d58", label="revealed as seeds")
ax.bar(xs, tgts, 0.6, bottom=seeds, color=C["best"],
       label="hidden targets (recoverable)")
ax.bar(xs, orph, 0.6, bottom=[s + t for s, t in zip(seeds, tgts)],
       color=C["base"], label="orphans (no episode exists)")
ax.set_xticks(list(xs))
ax.set_xticklabels(names, rotation=18, ha="right", fontsize=8)
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
ax.set_ylabel("share of blocker errors")
ax.set_title("What the blocker got wrong, and how much of it is addressable")
ax.legend(fontsize=8, ncol=3, loc="lower center",
          bbox_to_anchor=(0.5, -0.42))
for ext in ("png","pdf"):
    fig.savefig(os.path.join(OUT, f"fig4_error_population.{ext}"))

print("figures written to", OUT)
for f in sorted(os.listdir(OUT)):
    if f.endswith(".png"):
        print(" ", f)
