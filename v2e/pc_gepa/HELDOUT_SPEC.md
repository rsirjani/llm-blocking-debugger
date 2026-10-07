# Held-out evaluation sets for the v2e campaign — frozen 2026-09-11

Written *before* evaluation, so the sets cannot be chosen after seeing a
result. Nothing here is generated in advance: instance ids are the RNG
seed, so these sets are reproduced exactly whenever they are run, as
long as the flags in §4 are unchanged.

## 1. Why no data needs reserving

Instance ids come from two disjoint namespaces:

```python
frozen_specs: f"{prefix}:{blocker}:{dataset}:v{variant}:{j}"   # val:reclin2:cora:v0:3
train_specs:  f"train:{blocker}:{dataset}:{offset + i}"        # train:reclin2:cora:412
```

The search touches `train:…` (rollouts and minibatches) and
`val:…:v0:…` (the selection panel). It never constructs any other
variant. So `v1+` is untouched by construction, not by convention.

## 2. The three sets

| set | ids | cells | what it tests |
|---|---|---|---|
| **V0 — control** | `val:*:v0:0..3` | 10 train cells | Must reproduce the val mean the search reported **to within ±0.053** (§7). Not a result: it is the objective that was maximised. A mismatch *larger than that band* means the harvest harness is wrong. |
| **V1+ — in-regime held-out** | `val:*:v{1,2}:0..3` | 10 train cells | Block pairs no candidate was ever selected on. **Region transfer within a cell.** |
| **T — cross-dataset** | `test:*:v0:0..7` | abt-buy, dblp-scholar | Datasets never trained on. Dataset transfer. |

## 3. What V1+ does *not* test

An instance is a **whole-cell replay**: `state = dict(cell["assignment"])`,
scored globally (`pair_completeness` and `candidate_count` over the whole
partition). Variants differ only in which Case-3 block pairs are visited,
which failure is the seed in each, and step order. Every variant of a cell
therefore scores the *same* partition, records and gold.

So V1+ is not dataset transfer and not cell transfer. Both would need
cells outside train, and **every high-degree (cora) cell is in train**.
Testing the high-degree regime out-of-sample would require either a
restart with a cora blocker held out, or a dirty-ER dataset that is not
in the current matrix.

T does not cover it either: abt-buy and dblp-scholar are both clean
1:1 (each record has at most one true match), so the failure mode
characterised on 2026-09-11 — an edit dislodging records that are already
co-blocked with 13–16 of their own gold partners — cannot occur there.

**Independent evaluated objects = 10 cells, not the instance count.**
Four of those are the same 1,295 cora records under different blockers.

## 4. Flags that must not change

The cell set and the block-pair pool are derived from these. A single
differing flag silently evaluates a different matrix under the same name.

```
--blockers lambdafold simcse reclin2 klsh
--train-datasets amazon-google walmart-amazon dblp-acm cora
--test-datasets abt-buy dblp-scholar
--model qwen3.8:27b-q4_K_M
--pairs-per-instance 64 --val-per-cell 4 --test-per-cell 8
--order random
--min-pc 0.45 --max-pc 0.80 --max-oversize-frac 0.10 --min-case3 10
PYTHONHASHSEED=0
```

## 5. Variant diversity is not uniform

Each instance draws 64 block pairs from the cell's usable Case-3 pool.
Expected overlap between two variants of a cell is `k²/N`.

| cell | usable BPs | draw | coverage | expected v0∩v1 |
|---|---|---|---|---|
| reclin2:cora | 651 | 64 | 10% | 10% |
| lambdafold:cora | 547 | 64 | 12% | 12% |
| klsh:dblp-acm | 234 | 64 | 27% | 27% |
| klsh:cora | 220 | 64 | 29% | 29% |
| lambdafold:dblp-acm | 159 | 64 | 40% | 40% |
| simcse:dblp-acm | 134 | 64 | 48% | 48% |
| reclin2:amazon-google | 117 | 64 | 55% | 55% |
| simcse:cora | 111 | 64 | 58% | 58% |
| simcse:amazon-google | 90 | 64 | 71% | 71% |
| reclin2:walmart-amazon | 37 | 37 | 100% | 100% |

`reclin2:walmart-amazon` has fewer usable block pairs than the draw, so
`make_instance` takes the pool whole: every variant visits the same 37
block pairs, differing only in seed choice and order. **It carries no
held-out signal.** `simcse:cora` and `simcse:amazon-google` are thin.
The load-bearing held-out cells are `reclin2:cora` and
`lambdafold:cora`.

## 6. Reporting rule

Per cell, never pooled. cora contributes ~0.049 of λ-penalty per step
against ~0.001 in the two-table cells (base |C| is 10–22k there against
150–330k), so any mean over cells is a cora-weighted number in disguise.

## 7. Every score is one decoding draw — amended 2026-09-21

Not a defect and not a surprise: a prompt induces a distribution over
outputs, and we take one sample from it rather than a mean over
repetitions, because repeating every call would multiply the cost of
the campaign by the repetition count. What follows is the *size* of
that effect, measured rather than assumed, so the resolution of the
experiment is fixed before the harvest produces numbers to judge.

Candidates 0 and 1 are **byte-identical** (same sha256 over
`candidates.json`; candidate 1 was accepted at iteration 1 on a
proposal that changed nothing). They therefore constitute an accidental
replicate: the same prompt, scored twice on the same 40 frozen val
instances. They do not agree.

```
32 of 40 instances scored differently
paired difference   mean +0.0239   sd 0.1722   se 0.0272
95% CI on a mean difference at n=40:  +/- 0.0534
```

The mechanism is in the serving layer, not the harness. Replaying every
`(tag, prompt)` asked more than once:

```
same question asked twice          3381
differing executed edit (op+pred)  1015   (30.0%)
of those, the operation flipped     172   (move_to_x/move_to_y/none)
```

`temperature=0` does not buy determinism from Ollama under continuous
batching across 24 concurrent slots: batch composition varies, so the
reduction order in the forward pass varies, so logits vary. A single
flipped operation then cascades, because an instance is a sequential
replay and every later block pair is judged against a partition the
earlier edits moved.

Note the aggregate is far steadier than the per-call figure suggests:
a score already averages 40 instances and ~2,500 calls, so the ±0.053
band is what survives that averaging, not the 30% per-call divergence.

### Resolution of the experiment

Read differences against the ±0.053 band.

- **Below the band — do not claim.** `cand5→cand6` (+0.0528) and
  `cand6→cand7` (+0.0104) are not resolvable, and `cand0→cand1`
  (+0.0239) is a null by construction, the prompts being the same text.
  Resolving +0.0104 at 95% would take ~1,054 instances; the full
  harvest supplies 120 per arm.
- **Above the band — the comparison the harvest actually runs.** Seed
  (+0.1739) against best (+0.5920) is +0.418, roughly 8× the band at
  n=40 and wider still at n=120. That contrast is not in doubt.
- **Consequence for arm selection.** cand6 and cand7 are statistically
  indistinguishable, so adding cand6 as a third arm would cost ~18 h and
  still not separate them. We harvest cand7 because it holds the highest
  point estimate, and we say in the paper that the choice between the
  two is arbitrary.
- **Reproducibility claim.** Runs reproduce in distribution, not
  bit-exact. `PYTHONHASHSEED=0` pins the harness; it cannot pin the
  server.
