# LLM-Based Blocker Debugging via Reflective Prompt Evolution
### Status report — 10 September 2026
### Run `v2e_qwen38` (arm: v2 partition-repair)

---

## 1. The problem and the claim being tested

Blocking reduces the O(n²) comparison space of entity resolution to a candidate
set. Any blocker at a usable operating point separates some true matches: those
gold pairs land in different blocks and no downstream matcher can ever recover
them. **Pair completeness (PC)** — the fraction of gold pairs that are
co-blocked — is therefore a hard ceiling on end-to-end recall.

**The claim under test:** an LLM, given a blocker's output and a *single* known
mistake, can write general repair rules that recover other separated pairs it
was never shown, at acceptable cost in comparisons.

**No matcher is ever run.** All metrics live on the candidate-set plane:
PC, |C|, RR, and runtime. This is deliberate — it isolates the blocking
contribution from matcher quality.

### The object being edited is a partition, not a candidate set

This distinction matters and is easy to blur. The LLM does **not** add or remove
candidate pairs. It **moves records between blocks** of a mutually exclusive
partition — every record in exactly one block, asserted per record after every
single edit (P1).

|C| is therefore a *derived* quantity, fully determined by the block sizes:

```
|C| = Σ_blocks  n_i (n_i − 1) / 2
```

Nothing ever writes to a pair list. Consequences that follow from this and would
not hold for a pair-emitting system:

- **The deliverable stays a deployable blocking**, not a set of suggestions. Its
  output can be handed to any matcher.
- **A bad edit can *lower* PC.** Moving a record out of a block separates it from
  everything left behind, including partners it was correctly grouped with. A
  pair-adding system can only ever waste comparisons; this one can destroy
  recall, which is why the score's numerator is global (§6).
- **PC and |C| move together, not independently.** A single move changes both by
  a fixed arithmetic amount — moving *k* records from a group of size *a* into
  one of size *b* changes |C| by exactly *k(b − a + k)*. This is why a
  well-aimed move can raise PC *and* lower |C| simultaneously (§8), which is not
  possible when adding pairs to a candidate set.
- **|C| can go down at all.** In a pair-emitting arm, the candidate set is
  monotonically non-decreasing; here 16 of 42 instances *shrink* it.

The earlier arm (V1, `run6`/`run7`) did work on a candidate set — the LLM picked
pairs from a TF-IDF shortlist, and |C| could only grow. V2 replacing that with
partition repair is the central design change, and the reason the two arms'
numbers are not directly comparable (§11).

**What is optimized:** only the prompt. Retrieval is fixed and uniform so that
the measurement reflects the judge's ability rather than a retriever that hands
it the answers.

---

## 2. Data: the cell matrix

A **cell** is one (blocker × dataset) pair — one blocker run to a specific
operating point on one benchmark. 14 cells are in play, drawn from 4 blockers
and 6 datasets.

| blocker | dataset | base PC | usable block pairs | records | gold pairs | split |
|---|---|---:|---:|---:|---:|---|
| simcse | dblp-acm | 0.6830 | 134 | 4,910 | 2,224 | train |
| klsh | cora | 0.6053 | 220 | 1,295 | 17,184 | train |
| klsh | dblp-acm | 0.5751 | 234 | 4,910 | 2,224 | train |
| reclin2 | cora | 0.5548 | 651 | 1,295 | 17,184 | train |
| simcse | cora | 0.4866 | 111 | 1,295 | 17,184 | train |
| reclin2 | walmart-amazon | 0.4861 | 37 | 24,628 | 1,154 | train |
| lambdafold | dblp-acm | 0.4838 | 159 | 4,910 | 2,224 | train |
| simcse | amazon-google | 0.4715 | 90 | 4,589 | 1,300 | train |
| reclin2 | amazon-google | 0.4669 | 117 | 4,589 | 1,300 | train |
| lambdafold | cora | 0.4649 | 547 | 1,295 | 17,184 | train |
| reclin2 | dblp-scholar | 0.6781 | 102 | 66,879 | 5,347 | **TEST** |
| reclin2 | abt-buy | 0.6235 | 56 | 2,173 | 1,097 | **TEST** |
| simcse | abt-buy | 0.5333 | 73 | 2,173 | 1,097 | **TEST** |
| lambdafold | abt-buy | 0.4622 | 77 | 2,173 | 1,097 | **TEST** |

**10 train cells, 4 held-out test cells, 2,604 usable block pairs.**
Mean baseline PC on the 10 train cells: **0.5278**.

### Three levels of held-out-ness — important when reading any number below

| level | what it is | used for | held out? |
|---|---|---|---|
| **training instances** | fresh draws from 10,000 specs on the 10 train cells; never reused | minibatch rollouts, reflection feedback | no |
| **validation panel** | 40 frozen instances, 4 per train cell | **candidate selection** (accept/reject, Pareto frontier) | from training, yes — but *selected on* |
| **test cells** | 4 cells on abt-buy and dblp-scholar, entirely unseen datasets | final evaluation | **fully — and untouched** |

**Every PC and score number in §8 is on the validation panel.** The val panel is
drawn from the *same 10 train cells* as the training instances — different
instances, same cells, same datasets. It is therefore held out from *fitting* but
**not** from *selection*: candidates are chosen by their val score, so val
numbers are optimistically biased by the usual winner's-curse mechanism. That is
exactly why the panel was enlarged from 10 to 40 instances (§7), and it is why
the held-out test cells are the number that actually settles the claim.

### Blockers

| | method |
|---|---|
| `lambdafold` | blocklib λ-fold LSH, Λ=1 (Λ=1 forces mutual exclusivity) |
| `simcse` | SimCSE embeddings + Louvain community detection |
| `reclin2` | R `reclin2` standard blocking |
| `klsh` | k-means LSH over shingled text |

All four produce **mutually exclusive** blocks — every record in exactly one
block, asserted **per record**, never validated via pair counts (a pair-count
check can pass while records are silently dropped; `blocklib`'s
`generate_blocks()` does exactly that, which is why we call
`generate_candidate_blocks()` per table instead).

### Held-out design

`abt-buy` and `dblp-scholar` are **entirely unseen** — no cell from either
dataset appears in training. This tests transfer across *datasets*, not just
across instances. Three of four blockers appear on the test side, so blocker
transfer is also measured. `klsh` is train-only (its test cells failed
admission).

### Admission rules, and why

Cells enter the matrix only if:

1. **PC ∈ [0.45, 0.80]** — a comparable operating point. Outside this band the
   cell poses a different problem. `klsh` cannot reach the band on product
   catalogues (PC 0.362 / 0.377 / 0.442), so those cells are excluded and
   reported as a finding rather than tuned until they fit.
2. **≥10 usable Case-3 block pairs** — enough to debug.
3. **≤10% of Case-3 pairs oversized** — see §4. A cell whose large block pairs
   don't fit would be measured only on its small ones, which is an easier
   problem than the cell actually poses.

### Which cells were removed, and why

**24 cells were attempted** (4 blockers × 6 datasets). **14 admitted, 10
excluded.** Exclusion is by preregistered rule, applied before any LLM call, and
recorded in `config.json`:

| cell | base PC | Case-3 | fit | oversize | median block-pair size | excluded because |
|---|---:|---:|---:|---:|---:|---|
| klsh/amazon-google | 0.3623 | 10 | 0 | 10 | 1,817 | PC below band |
| klsh/walmart-amazon | 0.3769 | 10 | 0 | 10 | 9,960 | PC below band |
| klsh/abt-buy | 0.4421 | 10 | 2 | 8 | 756 | PC below band |
| simcse/walmart-amazon | 0.2071 | 67 | 0 | 67 | 2,076 | PC far below band |
| reclin2/dblp-acm | 0.6322 | **3** | 3 | 0 | 4 | too few Case-3 pairs (<10) |
| klsh/dblp-scholar | 0.4795 | 28 | 0 | 28 | **17,326** | 100% oversize |
| lambdafold/walmart-amazon | 0.5416 | 86 | 1 | 85 | 1,393 | 99% oversize |
| lambdafold/dblp-scholar | 0.4567 | 440 | 204 | 236 | 1,377 | 54% oversize |
| simcse/dblp-scholar | 0.7374 | 259 | 139 | 120 | 1,044 | 46% oversize |
| lambdafold/amazon-google | 0.5069 | 88 | 59 | 29 | 323 | 33% oversize |

**Three distinct reasons, none of them cherry-picking:**

**(a) Operating point — 4 cells.** PC outside [0.45, 0.80]. A blocker at PC 0.21
or 0.36 is posing a different problem: most gold pairs are separated, so the task
stops being "debug a mostly-working blocking" and becomes "rebuild it". Note this
rule is symmetric — it would exclude an over-permissive cell too, though none
occurred. **`klsh` fails this on every product catalogue** (amazon-google 0.362,
walmart-amazon 0.377, abt-buy 0.442). That is a reportable finding about k-means
LSH over shingled product titles, not a nuisance to be tuned away.

**(b) Nothing to debug — 1 cell.** `reclin2/dblp-acm` reaches PC 0.632 but has
only **3** Case-3 block pairs. Its errors are almost all orphans (Case 2), which
offer nothing to generalize from. Statistically empty.

**(c) Too large to show in full — 5 cells.** These are the interesting ones. Our
invariant is that the sample is *never truncated*: 35% of each block, all fields,
full length. Where the full sample would exceed the context, the block pair is
excluded — and if **>10% of a cell's Case-3 pairs** are in that state, the whole
cell goes.

The alternative would have been to keep those cells and quietly measure them on
their small block pairs only. That is strictly worse: it would report a number
for `lambdafold/walmart-amazon` computed from 1 of its 86 debuggable pairs, which
is an easier problem than the cell actually poses, and it would bias the entire
campaign toward small blocks — exactly where repair is easiest.

The extreme case is `klsh/dblp-scholar` at a **median block-pair size of 17,326
records**. That is not a partition worth debugging; it is one bucket holding
almost everything.

**What this costs us, stated plainly:**

- `walmart-amazon` survives on only one blocker (`reclin2`), and `dblp-scholar`
  on only one (`reclin2`) — so those two datasets are thinly represented.
- `lambdafold` loses 3 of its 6 datasets, keeping dblp-acm, cora and abt-buy.
- `klsh` is **train-only**: all three of its would-be test cells were excluded
  (two on PC, one on size), so no blocker-transfer claim can be made for it.
- The admitted set is biased toward **coarser partitions with moderate block
  sizes**, because that is what fits in full. Any claim about very large blocks
  is out of scope for this run, and would need a longer-context judge.

A cheaper context would readmit some of these: at 32,768 tokens, 93.2% of all
Case-3 pairs fit versus 81.8% at 12,288. We chose 18,432 as the smallest context
that yields **zero** exclusions among admitted cells, rather than maximising cell
count.

### P1 verified on every cell

Every cell was checked directly: each record assigned to exactly one block, no
record unassigned, no assignment referring to an unknown record, every gold-pair
endpoint present, and more than one block. |C| was also re-derived independently
(`Σ nᵢ(nᵢ−1)/2`) and matched the harness on every cell checked.

| cell | records | assigned | blocks | largest | singletons | P1 |
|---|---:|---:|---:|---:|---:|:--:|
| klsh/cora | 1,295 | 1,295 | 50 | 66 | 0 | OK |
| klsh/dblp-acm | 4,910 | 4,910 | 50 | 258 | 0 | OK |
| lambdafold/abt-buy | 2,173 | 2,173 | 74 | 404 | 13 | OK |
| lambdafold/cora | 1,295 | 1,295 | 170 | 111 | 54 | OK |
| lambdafold/dblp-acm | 4,910 | 4,910 | 308 | 450 | 91 | OK |
| reclin2/abt-buy | 2,173 | 2,173 | 451 | 197 | 214 | OK |
| reclin2/amazon-google | 4,589 | 4,589 | 374 | 248 | 136 | OK |
| reclin2/cora | 1,295 | 1,295 | 292 | 61 | 154 | OK |
| reclin2/dblp-scholar | 66,879 | 66,879 | 61,074 | 24 | 57,359 | OK |
| reclin2/walmart-amazon | 24,628 | 24,628 | 8,971 | 237 | 5,930 | OK |
| simcse/abt-buy | 2,173 | 2,173 | 152 | 30 | 0 | OK |
| simcse/amazon-google | 4,589 | 4,589 | 477 | 28 | 2 | OK |
| simcse/cora | 1,295 | 1,295 | 89 | 27 | 0 | OK |
| simcse/dblp-acm | 4,910 | 4,910 | 81 | 166 | 0 | OK |

**All 14 are valid partitions.** Worth noting the operating points differ
sharply in *shape*, not just in PC:

- `simcse` and `klsh` produce **few, large blocks** with almost no singletons
  (50–477 blocks, 0–2 singletons) — coarse partitions.
- `reclin2/dblp-scholar` is the opposite extreme: **61,074 blocks over 66,879
  records, 86% of them singletons**, largest block 24. It reaches PC 0.6781 by
  being extremely fine-grained.

This matters for the repair task. A singleton contributes no within-block
comparisons and cannot host a Case-3 block pair, so on a heavily fragmented cell
the reachable repairs concentrate in a small minority of the blocks. It also
changes the cost arithmetic: moving *k* records into a block of size *b* costs
*k(b − a + k)*, which is cheap when *b* is tiny. The campaign spans both regimes
deliberately, so a prompt that only works on coarse partitions would show up as
a per-cell split on the Pareto frontier.

---

## 3. Case taxonomy and what an instance is

For each gold pair, look at where the blocker put the two records:

| case | meaning | usable? |
|---|---|---|
| **Case 1** | both records co-blocked | already correct, nothing to do |
| **Case 2** | separated, but this block pair has exactly 1 failure | orphan — nothing to generalize from |
| **Case 3** | separated, and this block pair has **≥2** failures | **debuggable** |

Case 3 is the target: if two blocks have ≥2 gold pairs split across them, then
revealing one as a **seed** leaves the others as hidden **targets** that a
general rule could recover.

Block pairs are **symmetric/unordered** — `(X,Y)` and `(Y,X)` are one pair,
constructed by ordering on signature. Clean-clean and dirty ER are treated
identically as R = R_A ∪ R_B, so no side is privileged.

### An instance = a whole-cell replay

One instance is:

1. Sample **64 Case-3 block pairs** from one cell (deterministically, from the
   instance id).
2. For each, reveal **one** of its failed pairs as the seed; the rest are
   hidden targets.
3. Process the 64 in sequence **against a live, mutating partition**. Each step
   sees what earlier steps produced. If an earlier edit already reunited a later
   seed's records, that step is skipped as already-solved.
4. **Assert P1 after every single edit** — every record in exactly one block.
   The deliverable stays a partition throughout, so a bad edit can *lower* PC,
   not merely cost comparisons.
5. Score the cell as a whole at the end.

### How many block pairs actually get processed

Measured over 168 instances:

```
block pairs drawn per instance:        64  (or all of them, if the cell has fewer)
steps actually judged (LLM called):    mean 58.5   median 62   range 36..64
steps skipped as already-solved:       mean  0.3   median  0   max 4
replies unparseable:                   mean  2.7
```

**The already-solved skip is rare — 0.5% of drawn pairs.** Only 33 of 168
instances (20%) skipped even one step, and the distribution is
`0 skips: 135 | 1-2: 28 | 3-5: 5`, never more than 4. So the live-partition
carry-over between steps almost never removes work.

That is worth stating because it bounds a possible objection. If earlier edits
routinely resolved later seeds, the 64 steps would not be 64 independent repair
problems and the per-instance score would be inflated by double-counting. They
don't: **98% of drawn block pairs are still unresolved when their turn comes.**
The live partition matters for *cost accounting* (each edit changes the |C|
baseline the next step is measured against) far more than it does for removing
work.

The gap between 64 drawn and 58.5 judged is accounted for by:
`steps_run = drawn − skipped − unparseable`, plus one train cell
(`reclin2/walmart-amazon`) that has only **37** usable Case-3 pairs in total, so
its instances draw 37 rather than 64. The 2.7 unparseable replies per instance
(~4%) are the larger of the two losses and are pure waste — a reply the harness
cannot read costs a call and produces no edit.

Total judged steps across the campaign so far: roughly **58.5 × 650 ≈ 38,000
block-pair repair decisions**.

All randomness is a function of the instance id, so re-running an instance
reproduces the same block pairs, seeds, samples and order. **GEPA's accept/reject
is therefore a paired comparison**, not a comparison across different draws.

---

## 4. What the judge sees

### The prompt (seed version, verbatim)

```
Two groups of records, group X and group Y, are shown below. One pair of records,
one from each group, is known to describe the same real-world entity, yet the two
records were placed in different groups.

{STATS}

Known same-entity pair:
{SEED}

Group X sample:
{SAMPLE_X}

Group Y sample:
{SAMPLE_Y}

Decide what change to the grouping would place records that describe the same
entity together. State the change as a condition on record fields, not as a list
of the records you were shown: the condition is applied to every record of the
group, including the many that are not shown here. Answer with a single JSON object:
{"operation": "move_to_x" | "move_to_y" | "new_block" | "none", "predicate":
[{"field": "<field name>", "operator": "equals" | "contains" | "not_contains" |
"regex" | "is_null" | "is_not_null" | "gte" | "lte" | "between", "value": "<text>"}],
"reason": "<one sentence>"}
"gte" and "lte" compare the first number found in the field against the value;
"between" takes "value": [low, high] and is true when the number falls in that
range, endpoints included.
You may give several rules at once, up to five, as {"actions": [{...}, {...}],
"reason": "<one sentence>"}; they are applied in the order written, and each one
acts on the grouping the previous rules left behind.
"move_to_x" moves every record of group Y that satisfies the condition into group X;
"move_to_y" is the mirror. "new_block" takes the records of either group that satisfy
the condition into a group of their own. "none" leaves the grouping unchanged and
needs no condition. Conditions within one rule are combined with and.
```

There is **no system prompt**. That string is the entire input.

The four slots are filled by the harness:
- `{STATS}` → `Group X holds 3 records, of which 1 are shown. Group Y holds 2 records, of which 1 are shown.`
- `{SEED}` → the one known split pair, rendered whole
- `{SAMPLE_X}` / `{SAMPLE_Y}` → a uniform random 35% sample of each block

All four are **required slots** — a candidate that drops one is rejected before
being sent. `{STATS}` is required because the reflector is told how comparison
count responds to group sizes; a template without it would leave the judge
holding a rule whose inputs it cannot see.

### Critically: no leakage

The judge is **not** told the dataset, the blocker, that other split pairs
exist, or how many. Gold enters only as the single seed pair. Everything else
is post-hoc evaluation.

### Retrieval, and the no-truncation invariant

The sample is **35% of each block, proportional, with no cap**. Records render
with **all fields at full length**.

This is enforced rather than best-effort. Earlier versions shrank the sample to
fit the context, which silently cut 18% of block pairs — on the worst, a
2,940-record block showed 79 records instead of the intended 1,029 (3% instead
of 35%). That biased the campaign toward small, easy blocks precisely where the
hard ones were. Now:

- block pairs whose full sample won't fit are **excluded up front**, not trimmed
- `fill_template` **raises** rather than shrinking — it's an assertion, not a fallback
- a cell is dropped entirely if >10% of its Case-3 pairs are oversized
- `MAX_CTX = 18432` is the *smallest* context at which **zero** block pairs are
  excluded across all 14 cells

Field capping was also removed: the old limits (6 fields, 120 chars) truncated
55% and 23% of records respectively. A field the judge never sees is a field it
cannot write a predicate on. Full rendering costs 1.14× the prompt.

**Current state: 0 block pairs excluded for size, 0 truncation anywhere.**

---

## 5. The action space

The judge returns an **edit**, not a pair list. Every change must be a
**predicate** — a rule over the whole block, not a gesture at the sample.

| operation | effect |
|---|---|
| `move_to_x` | every record of Y satisfying the condition moves into X |
| `move_to_y` | mirror |
| `new_block` | matching records from **either** group form a new block |
| `none` | no change |

**There is no blanket merge.** To unite two groups the model must write a
condition characterising the records it wants united. This was a deliberate
design decision: a merge operation lets the model reunite the seed without
learning anything generalizable.

Predicate operators: `equals`, `contains`, `not_contains`, `regex`, `is_null`,
`is_not_null`, `gte`, `lte`, `between`. Conditions within a rule are ANDed. Up
to 5 rules per reply, applied in order against the live grouping.

**Scope:** a predicate is applied only to the two blocks in front of the judge —
never the whole partition. Generalization is therefore from the **35% shown** to
the **100% of those two blocks**, plus second-order effects on non-target pairs.

### Observed predicate usage (3,105 rules / 3,909 conditions sampled)

```
contains      74.4%        1 condition   80.1%
equals        24.1%        2 conditions  14.9%
regex          0.7%        3 conditions   4.0%
not_contains   0.6%        4+ conditions  1.0%
is_not_null    0.1%
```

Fields: `title` 60.6%, `name` 11.6%, `year` 6.4%, `venue` 6.3%,
`authors`/`author` 7.3%, `modelno` 3.1%, `manufacturer` 2.0%.

Median predicate value length is **28 characters** — it quotes substantial title
fragments, not keywords. Only 1.3% of `equals` conditions and 0.3% of `contains`
conditions match nothing anywhere, so the model is copying values verbatim from
the rendered sample (including trailing punctuation — 93.2% of field values end
in `.` `,` `;` or `:`).

---

## 6. Score function

```
score = (net same-entity pairs reunited across the WHOLE cell
         / separated pairs in the block pairs actually visited)
        − 5.0 × (relative growth in within-block comparisons)
```

```python
base_pc, base_hits = pair_completeness(state, gold)   # whole cell
d_c_rel   = (c - base_c) / base_c
net_ratio = (hits - base_hits) / local_targets        # global gain, local denominator
score     = net_ratio - LAMBDA * d_c_rel              # LAMBDA = 5.0
```

### What was rejected, and why

**F(PC, RR), the harmonic mean.** The obvious blocking metric, and wrong here:
RR sits at 0.88–0.99 across our cells, so it barely moves and the harmonic mean
degenerates to PC alone. It would have priced the cost axis at effectively zero.

**Whole-cell ΔPC as the numerator's denominator.** Tried and abandoned. An
instance touches 64 block pairs out of hundreds-to-thousands in a cell, so
whole-cell ΔPC came out at ±0.003 per instance — the score measured *which pairs
the dice drew*, not how well they were repaired. Signal was drowned by draw
variance.

**Local recall alone** (fraction of the visited block pairs' targets recovered).
Rejected because it is blind to damage: a rule that reunites its target while
separating ten correctly-grouped pairs elsewhere would score perfectly.

**Raw PC with no cost term.** Degenerate — the optimal policy is to merge
everything into one block, PC = 1.0, and the blocking becomes useless.

### The final form, term by term

**The numerator is global.** `hits` is recomputed over the *entire* partition
after every edit. An edit that drags a record out of a block it belonged in is
debited for every pair it leaves behind. This is what makes damage impossible to
hide, and it is why 7 of 144 instances score net negative rather than the metric
flattering them.

**The denominator is local** — the separated pairs sitting in the 64 block pairs
this instance actually visited (mean ≈ 1,110). This restores signal without
allowing the numerator to ignore collateral: gains and losses are counted
everywhere, but normalised by what this instance could plausibly have fixed.

**λ = 5 prices the second axis.** Reuniting a tenth of the local targets is
worth about what a 2% growth in comparisons costs.

### What the score actually incentivises

Worked from real numbers — one step inside an instance, block pair with
|X| = 61, |Y| = 24, instance-level local targets 1,110, cell |C| = 22,670:

| action | net pairs | Δ&#124;C&#124; | gain | cost | **score** |
|---|---:|---:|---:|---:|---:|
| blanket merge: all 61 into the 24 | 20 | +1,464 | 0.0180 | −0.3229 | **−0.3049** |
| broad rule: 30 records, 10 right | 10 | −210 | 0.0090 | +0.0463 | **+0.0553** |
| narrow rule: 5 records, 4 right | 4 | −160 | 0.0036 | +0.0353 | **+0.0389** |
| narrow rule: 2 records, 2 right | 2 | −70 | 0.0018 | +0.0154 | **+0.0172** |
| wrong rule: 12 records, 0 right | 0 | −300 | 0.0000 | +0.0662 | **+0.0662** |
| `none` | 0 | 0 | 0 | 0 | **0.0000** |

Five behaviours follow directly:

**1. Blanket merging is strongly punished.** Recovering *all 20* separated pairs
in the block pair still scores **−0.30**, because 20 pairs against an
instance-wide denominator of 1,110 is worth 0.018 while 1,464 added comparisons
cost 0.32. Break-even is **1 recovered pair per 4.1 comparisons added**. This is
the single most important property: the degenerate solution is not merely
discouraged, it is worse than doing nothing.

**2. Direction is priced.** Moving *k* records out of a group of size *a* into
one of size *b* changes |C| by exactly *k(b − a + k)*, which is **negative
whenever k < a − b**. So moving a modest number out of a large group into a
small one earns a *bonus*. Measured: 48.1% of block pairs have |X| > |Y|, and
the median size ratio is 2.08×, so direction is a live choice on nearly every
step.

**3. Precision beats coverage.** A narrow correct rule and a broad half-right
rule can score similarly, but the narrow one carries far less variance and no
damage risk. Note the fourth row: a *wrong* rule that moves 12 records out of a
big group into a small one scores **+0.066** purely on the comparison saving.
That is a real quirk of the objective, discussed below.

**4. `none` is a genuine option, not a cop-out.** It scores exactly 0, which
beats any negative-expectation guess. The judge uses it **529 times**, and the
evolved prompt explicitly recommends it when no safe condition exists.

**5. Collateral recovery is rewarded.** Because the numerator counts the whole
cell, pairs reunited beyond the step's target list count fully. Measured, this
is **15% of everything recovered** — and it is the behaviour the whole design
exists to elicit.

### Known gaps in the objective

Stated plainly, because they are real:

- **A wasteful-but-harmless move can score positive.** Row 5 above: 12 records
  moved, none of them right, +0.066 from the comparison saving alone. The score
  rewards shrinking |C| even when no recall is gained. This is defensible (RR is
  a real objective) but it means the metric is not purely a recall measure.
- **Inert conditions are free.** A condition matching 0 records costs nothing in
  score — only a wasted LLM call. This is why **38% of conditions are inert**:
  nothing in the objective pushes against them. The reach histogram in the
  *feedback* is what taught the prompt to avoid them, not the score.
- **No term for calls, latency or cost.** An instance that writes 64 careful
  rules scores the same as one that writes 5 and declines 59, if the outcomes
  match. Runtime is tracked but not optimised.
- **λ = 5 is a judgement, not a derivation.** It was chosen so the two axes are
  comparable in magnitude on these cells. A different λ would shift the
  precision/coverage balance, and we have not swept it.

## 7. The optimization loop (GEPA)

[GEPA](https://github.com/gepa-ai/gepa) — reflective prompt evolution. The
candidate is a dict of named text components; here exactly one component,
`message` (the whole template).

### Per iteration

1. **Select a parent** from the Pareto frontier over per-val-instance scores —
   not simply the best candidate. Different prompts win different cells, so this
   preserves diversity.
2. **Sample a minibatch of 24 training instances** (uniform random from 10,000
   fresh specs; training instances are never reused).
3. **Roll out the parent** on those 24 → scores + gold-aware feedback.
4. **Reflect**: package (inputs, outputs, feedback) → one Claude Sonnet call →
   rewritten template.
5. **Retest the child on the same 24** — paired comparison.
6. **Accept iff the sum improved**; if accepted, run the full 40-instance
   validation panel.
7. Periodically attempt a **merge** of two candidates via a common ancestor.

### Minibatch sizing — why 24

Originally 8. Measured failure: **2 of 3 accepted candidates were *worse* than
their parent** on the full 40-instance val panel. The accept gate was passing
noise.

Per-instance score sd ≈ 0.70; the improvement we're hunting (seed → best) is
≈ 0.286 per instance. Detection power:

| minibatch | sd of mean | detection | iterations in budget |
|---|---:|---:|---:|
| 8 | 0.247 | 1.16σ | 156 |
| 16 | 0.175 | 1.64σ | 104 |
| **24** | **0.143** | **2.00σ** | **78** |
| 32 | 0.124 | 2.31σ | 62 |

24 was chosen for a ~2σ gate. **Wall-clock is unaffected** — budget counts
instances, so minibatch size trades iteration count for accept reliability, not
time. Since the change, **4 of 4 accepts have beaten their parent.**

### Validation panel — 40 instances

4 per train cell × 10 cells = 40 instances = **2,560 block-pair repairs per
val pass**. Deliberately equalized per cell so no cell dominates selection.
Raised from 10 because selection noise on a thin panel is the known failure mode
here (an earlier V1 run lost its top-line result to winner's curse on a
29-episode panel).

### The reflector is blinded

The reflection LM is told **nothing** about dataset or blocker identity. Record
text in feedback is **content-masked** (`<Tn>` placeholders; the same
placeholder means the same original token), and a **leakage gate** re-prompts if
the draft contains tokens drawn from the data.

### What the reflector IS told — environment mechanics

These are facts true *by construction*, supplied so the optimizer isn't
guessing at the machinery. They are explicitly framed as mechanics, not advice:

- **Which group is X and which is Y carries no meaning** — they're ordered by an
  internal identifier, so any standing preference for one label is arbitrary.
- Moving *k* records out of a group of size *a* into one of size *b* changes the
  comparison count by exactly **k(b − a + k)** — negative whenever *k < a − b*,
  so a move can reunite records *and* lower the total.
- Taking *k_a* and *k_b* records from the two groups into one new group changes
  it by **k_a·k_b − k_a(a−k_a) − k_b(b−k_b)**; drawing on one group only always
  lowers it, by *k(a−k)*.
- Records leaving a group are separated from everything left behind, including
  pairs already correctly grouped.
- The sample is a fixed uniform fraction, so what is shown is proportional to
  true group size.

*Which* operation suits *which* situation is left entirely to the outcomes. The
line "these are facts about the machinery, not advice" is in the prompt.

### Feedback design

Size-bounded so it doesn't blow the reflector's context regardless of minibatch
size: a reach histogram over **all** steps (how many records each condition
matched: 0 / 1 / 2-5 / 6-20 / >20), helped/hurt/inert counts, then at most 3
best, 3 worst, 2 wasteful and 2 zero-match exemplars quoted with masked text.
Roughly 5.7k chars per instance regardless of instance size.

---

## 8. Results

### Current state

```
run:        runs/gepa/v2e_qwen38
iteration:  112        candidates: 4        evals: 650 of 5,000 budget
judge:      qwen3.8:27b-q4_K_M, thinking off, temperature 0
```

| candidate | parent | val mean | val median | harmful instances | found at |
|---|---|---:|---:|---:|---:|
| 0 (seed) | — | +0.1739 | +0.3843 | 11/40 | 0 calls |
| 1 | 0 | +0.1979 | +0.3969 | 11/40 | 88 |
| 2 | 0 | +0.2663 | +0.1755 | 10/40 | 176 |
| **3** | 1 | **+0.3128** | **+0.4090** | **7/40** | 365 |

**Best candidate is 1.80× the seed.** Per-instance Pareto frontier sits at
**+0.4694**, so **+0.157 of headroom** is already demonstrated — different
candidates win different instances (wins: 4 / 8 / 9 / 21), which is what merge
exists to exploit.

Note the improvement is mostly **reliability, not ceiling**: harmful instances
fell 11/40 → 7/40.

### Pair completeness — measured on the 40-instance VALIDATION panel

Not on training minibatches, and not on the held-out test cells. See the
held-out-ness table in §2: this panel sits on the same 10 train cells and is the
panel candidates are selected against.

**Important qualification.** The aggregate below is taken from GEPA's
`best_outputs_valset`, which records **the best candidate on each instance**. It
is therefore an *oracle-selected mixture* over candidates 0–3, not the
performance of any single deployable prompt. It is an upper bound on what one
prompt achieves.

```
baseline PC (mean over the 10 train cells):  0.5278
after repair (best-per-instance mixture):    0.5686
                                     +4.08 points, +7.7% relative

per-instance ΔPC:  mean +0.0408   median +0.0373
                   range −0.0063 .. +0.1205
                   positive in 39/42 instances
best single instance:  0.5278 -> 0.6483  (+12.05 points)
```

Per candidate, on the disjoint subset of instances where that candidate is the
best — so each row is that candidate *at its best*, and rows are not comparable
with one another:

| candidate | instances won | ΔPC on those | Δ&#124;C&#124; on those |
|---|---:|---:|---:|
| 0 (seed) | 4 | +0.0159 | −0.0438 |
| 1 | 8 | +0.0389 | +0.0145 |
| 2 | 9 | +0.0490 | +0.0483 |
| 3 (best) | 21 | +0.0427 | +0.0091 |

Even on its own favourable subset the seed manages **+1.59 PC points** against
candidate 3's **+4.27** — but this is not the clean comparison.

**What is missing, and how to get it.** The seed's ΔPC across *all 40*
validation instances has not been measured. `score = net_ratio − 5·d_c_rel` is
not invertible to ΔPC, so it cannot be recovered from stored state; it requires
re-running candidate 0 and candidate 3 over the same 40 instances — 2 × 2,560 =
**5,120 judge calls**, roughly a day at current throughput. Until that is run,
the defensible claim is the **val score**, which we do have in full on all 40
instances for every candidate:

```
cand 0 (seed)  +0.1739
cand 1         +0.1979
cand 2         +0.2663
cand 3         +0.3128     <- 1.80x the seed, all 40 instances, like-for-like
```

That 1.80× is the like-for-like optimisation result. The PC numbers above are
best read as *evidence that the score corresponds to real recall movement*,
not as a headline.

### Comparison cost (|C|, derived from the partition)

```
Δ|C| relative:  mean +1.35%
                |C| SHRANK in 16 of 42 instances
```

The recall gain is not being bought with a blow-up in comparisons. In the best
instances |C| actually falls while PC rises — possible only because the edits
restructure a partition rather than append to a pair list.

### Does it generalize beyond what it was shown?

Yes — measured across 144 instances with recorded feedback:

```
net pairs reunited (whole cell):  mean +451   median +76   max +2,748
  LOCAL   (blocks shown):         34.5% of targets recovered
  COLLATERAL (not the step's targets): mean +69, positive in 115/144 (80%)

totals: net +64,997 = local +55,080 + collateral +9,917
  -> 15% of everything recovered came from pairs the step wasn't aiming at
```

**Worked example** (cora cell, 63 edits, single instance):

```
+1,484 more same-entity pairs across the whole collection
   864 from the 1,324 separated pairs in the blocks shown  (65% local recall)
  +620 net effect elsewhere
comparisons changed by −6.56%      <- |C| SHRANK
```

The mechanism, one rule:
```
move_to_x  title contains "efficient noise-tolerant learning from statistical queries"
           matched 26 of the 61 records in group Y
           same-entity pairs +0.0182,  comparisons −520
```
From a single seed pair it wrote a title-substring rule that caught **26**
records — most of them not in the 35% sample — and because it moved them from a
larger group into a smaller one, `k(b−a+k)` came out negative. **Recall up,
comparisons down, from one rule.**

### Damage

```
8,372 conditions across 144 instances
  HELPED  4,866  (58.1%)
  HURT      297  ( 3.5%)
  INERT   3,209  (38.3%)

instances net negative overall: 7/144 (5%)
gains vs losses across instances: +65,214 vs −217  -> 301x
```

Damage has one clear signature — **broad category fields**:
```
venue equals "Very Large Data Bases"   matched 119 of 284 records   ΔPC −0.0247
venue equals "SIGMOD Conference"       matched  59 of 134 records   ΔPC −0.0220
```
Moving 119 records out of a 284-record block drags along everything published at
VLDB, and each is cut off from its correct partners left behind. **Damage comes
from moving records *out*, not from putting wrong ones together.** Blast radius
is bounded because a predicate only touches the two blocks in front of it —
worst instance in the entire campaign is −87 pairs.

### What the optimizer learned

The best candidate is **dataset-agnostic** — it names no dataset, blocker or
entity. Everything it added is predicate-writing craft:

> - The value you pick **must literally appear in the text** of the record(s) you
>   intend to match — a value invented from memory routinely matches 0 records.
> - Prefer a **distinctive, near-complete phrase** over a short generic token.
> - If the field that ties the seed pair together is **broad** (venue,
>   manufacturer, year), do not use it alone — **narrow it with a second field**,
>   or skip the edit.
> - A condition that would move most of a large group into a much smaller one
>   **almost always costs more in comparisons than it repairs**.
> - If you cannot find a value both specific and safe, **answer "none" rather
>   than guess**.

Plus a use-case for the operation that was previously unused:
> `"new_block"` ... **use this instead of move_to_x/move_to_y when you want to
> isolate a suspect subset without committing it to either existing group.**

Every one of these is traceable to a feedback channel: the "literally appear"
rule to the reach histogram's zero-bucket; the venue/manufacturer warning to the
damage exemplars above; the cost rule to the `k(b−a+k)` arithmetic.

Earlier candidates also absorbed the mechanics directly, e.g.
`X/Y labels carry no meaning — never let them factor into reasoning.`

### Operation mix

```
move_to_x 1,862 | none 529 | move_to_y 100 | new_block 22
```
`move_to_y` rose from 4% to ~5% of moves after the reflector was told labels are
arbitrary. `new_block` rose from **1 use in 2,441 actions** (previous run) to 22
— the numeric operators (`gte`/`lte`/`between`) and explicit "operations chosen,
out of the four available" reporting broke that exploration deadlock.

---

## 8b. The current best prompt, in full

Candidate 3, val **+0.3128**, 3,133 chars (seed was 1,617). Framing, slots and
format boilerplate are byte-identical to the seed; everything else is evolved.

```
Two groups of records, group X and group Y, are shown below. One pair of records, one from each group, is known to describe the same real-world entity, yet the two records were placed in different groups.

{STATS}

Known same-entity pair:
{SEED}

Group X sample:
{SAMPLE_X}

Group Y sample:
{SAMPLE_Y}

Decide what change to the grouping would place records that describe the same entity together. State the change as a condition on record fields, not as a list of the records you were shown: the condition is applied to every record of the group, including the many that are not shown here.

Before answering, check your condition against the seed pair and the samples above:
- The value you pick must literally appear in the text of the record(s) you intend to match (check the sample, not just the seed pair) — a value invented from memory of similar records elsewhere routinely matches 0 records.
- Prefer a distinctive, near-complete phrase (e.g. most of a title, a model number, an exact name) over a short generic token. Short or generic values (a common venue, manufacturer, single word) tend to match a large, unpredictable slice of the group instead of just the entity you found, which pulls in unrelated records and inflates comparisons far more than the reunited pair is worth.
- If the field that ties the seed pair together is broad (venue, manufacturer, year) and shared by many unrelated records, do not use it alone — narrow it by combining with a second, more specific field, or skip the edit ("none") rather than move a large chunk on a weak signal.
- A condition that would move most of a large group into a much smaller one, or unite two large chunks, almost always costs more in comparisons than it repairs — check the "how the grouping responds" arithmetic before writing it.
- If you cannot find a value in the sample that is both specific to the shared entity and unlikely to sweep in unrelated records, answer "none" for that step rather than guess.

Answer with a single JSON object:
{"operation": "move_to_x" | "move_to_y" | "new_block" | "none", "predicate": [{"field": "<field name>", "operator": "equals" | "contains" | "not_contains" | "regex" | "is_null" | "is_not_null" | "gte" | "lte" | "between", "value": "<text>"}], "reason": "<one sentence>"}
"gte" and "lte" compare the first number found in the field against the value; "between" takes "value": [low, high] and is true when the number falls in that range, endpoints included.
You may give several rules at once, up to five, as {"actions": [{...}, {...}], "reason": "<one sentence>"}; they are applied in the order written, and each one acts on the grouping the previous rules left behind.
"move_to_x" moves every record of group Y that satisfies the condition into group X; "move_to_y" is the mirror. "new_block" takes the records of either group that satisfy the condition into a group of their own — use this instead of move_to_x/move_to_y when you want to isolate a suspect subset without committing it to either existing group. "none" leaves the grouping unchanged and needs no condition. Conditions within one rule are combined with and.
```

The two evolved additions are (i) the `Before answering` checklist and (ii) the
appended `new_block` use-case on the last paragraph. Both were inferred from
outcome feedback — nothing in the reflector's input told it *when* to split, or
that broad category fields are dangerous.

---

## 8c. Operational incidents — why the run is slow and why it stalled

Three separate failures cost roughly **five days** of wall-clock. All are
diagnosed; none affected correctness of results, because GEPA persists state and
resumes.

### (1) Three-servers-per-GPU was an anti-pattern, not parallelism

Ollama's multi-GPU mode is a *layer split*: one model spread over three cards
makes them take turns rather than compute in parallel. The intended fix was one
server pinned per GPU. **This does not work** — Ollama 0.33.2 ignores
`CUDA_VISIBLE_DEVICES`, by index *and* by UUID, and manages devices itself.

The result went undetected for four days: each of three servers had layer-split
its **own full copy** across all three cards. We were paying **57 GB of weights
for one model's worth of work**, 12 of 16 CPU cores burned on coordination, load
average 23, and p50 latency of **101–174 s**.

`/proc/PID/environ` reported `CUDA_VISIBLE_DEVICES=0` the whole time — it is a
load-time snapshot and shows what was *set*, not what is used. Ground truth is
`nvidia-smi -i N --query-compute-apps`, which showed every runner holding ~6 GB
on every card.

Fixed by running **one server, one model copy, 24 slots**: 18.9 GB total,
load average 9, p50 latency **37 s**.

### (2) The GPUs were starved, not saturated

Even after consolidation, throughput was poor. The diagnostic that mattered:
**GPU utilisation was 33% while per-call latency was healthy.** That combination
means too few calls in flight, not too little hardware.

Time-weighted concurrency computed from the event log (each event carries `ts`
and `latency_s`, so start = ts − latency) confirmed **1.57 concurrent calls
against a 9-worker cap**. Raising `OLLAMA_NUM_PARALLEL` to 24 and matching the
client's worker cap took it to **18–22 concurrent**, a **4× throughput gain**.

Per-call latency rose from 37 s to ~226 s under full load — expected, and the
correct trade: aggregate throughput is what matters, and we are now close to
compute-bound.

### (3) The reflector failed silently for two days — the real cause of the stall

**This is why the best candidate has not changed since 8 September.**

`claude_reflection` retries the Sonnet CLI three times and then raises. GEPA
catches the exception and logs only:

```
Iteration 14: Reflective mutation did not propose a new candidate
Iteration 15: Reflective mutation did not propose a new candidate
Iteration 16: Reflective mutation did not propose a new candidate
```

That message is indistinguishable from "the optimizer had nothing to suggest".
**Our logging recorded the reflection call only *after* it returned**, so a
failing call left no trace at all. Eight consecutive iterations looked like
convergence when the proposer was simply down.

**Probable cause: a usage limit on the Claude subscription.** The driver log
carries the traceback (`RuntimeError: claude CLI reflection failed 3x`), so the
CLI genuinely failed, three attempts per iteration, from 09-08 18:39 until the
network outage. Ruled out by direct test: prompt size is not the issue — the CLI
accepts a 138,043-char (~34.5k token) prompt in 3 seconds. Auth is not the issue
— the same credentials work. Seven earlier calls succeeded and iteration 8
proposed a candidate, so nothing structural changed. Rate limiting is the
remaining explanation consistent with a transient two-day window. It cannot be
confirmed retrospectively because the original code captured the subprocess's
`stderr` and `returncode` and then discarded both.

Note the reflector shares the subscription quota with any interactive analysis
work on the same account, so the two compete.

**The compounding design flaw was the retry policy.** Backoff was `5s, 10s, 15s`
— **30 seconds of patience against a limit that resets in hours**. Every
iteration burned three attempts that could not possibly succeed, and then moved
on as though the optimizer had declined to propose. The retry policy was sized
for a network blip and met a quota exhaustion.

**Reflection now retries indefinitely** — 1m, 2m, 5m, 10m, 20m, then 30m
forever — logging each attempt's return code and stderr. Reflection is the sole
source of new candidates, so an iteration that loses it produces nothing
regardless; there is no value in proceeding past a step that is load-bearing.
The cost of waiting is idle GPU time, not lost work.

Because a long reflection wait is externally indistinguishable from a hang, the
liveness timer was changed to match: it repairs tunnels whenever they are
unreachable, but **relaunches the driver only when the process is actually
gone**. A live driver with stale judge calls is assumed to be waiting on
reflection and left alone — killing it there would destroy the step being waited
for and restart the loop that lost two days.

Compounding it: a **5-hour network outage** (ssh tunnels dropped) then produced
~100 iterations of `Exception during optimization: ollama call failed 8x`. With
`raise_on_exception=False`, GEPA absorbed each one and continued, so the run
span rather than dying loudly. Iteration counter went 6 → 113 with **zero work
done**.

The tunnels dropped because a supervisor process had been started manually and
never restored after an earlier intervention — a gap in *our* operational
scaffolding, not in the experiment.

### Fixes now in place

| fix | effect |
|---|---|
| reflection failures logged (`reflection-failed` event + stderr) | a down proposer is now visible immediately |
| `cache_evaluation=True` | resumes no longer re-run the seed on the full val panel (was 2,560 calls / 4.3 h per restart) |
| systemd user timer, 15-min interval | checks *has a judge call landed in 30 min*; reopens tunnels and relaunches the driver if not |
| launch scripts and logs moved out of `/tmp` | a machine reboot no longer destroys the ability to restart |

The liveness check deliberately watches the **event log**, not whether a
supervisor process exists — two supervisor processes have now died silently and
taken the run with them, and `systemd` restarts the checker itself.

### Current throughput and what remains

```
~16 s/call at 18-22 of 24 workers busy
650 of 5,000 instances (13%)
```

The nominal budget will not complete. The plan is to harvest at **600–800
instances**, comparable to what the previous arm reached, not to finish 5,000.
Value in GEPA is heavily front-loaded — in the prior run the winning candidate
appeared at 104 metric calls and nothing beat it in the following 168.

---

## 9. Infrastructure

- **Judge**: `qwen3.8:27b-q4_K_M` on Ollama, 3× RTX A5000 (dblab-gpu), thinking
  off (thinking is compute-bound at 7.4 s/call regardless of concurrency).
- **Reflector**: Claude Sonnet via CLI, run locally.
- **Serving**: **one** Ollama server with a single model copy across all three
  GPUs, 24 parallel slots. Three GPU-pinned servers do **not** work — Ollama
  0.33.2 ignores `CUDA_VISIBLE_DEVICES` (by index and by UUID) and layer-splits
  every model over all cards, so three "pinned" servers meant three copies
  contending: 57 GB of weights for one model's work, 12 of 16 cores burned,
  4× latency.
- **Throughput**: ~8–13 s/call at 18–22 of 24 workers busy. Diagnosed by
  computing time-weighted concurrency from the event log — GPU utilization sat
  at 33% with healthy per-call latency, which is starvation, not saturation.
  Fixing it was a **4× improvement**.
- **Reproducibility**: every prompt and raw response logged to
  `llm_events.jsonl`; `PYTHONHASHSEED=0` throughout; GEPA state persisted so the
  run resumes from `gepa_state.bin` after any interruption.

---

## 10. Honest limitations

1. **The held-out test evaluation has not been run.** Every number above is on
   the **validation panel** — 40 instances drawn from the same 10 train cells,
   and the same panel used to *select* candidates, so it carries selection bias.
   No number in this report is from an unseen dataset. The 4 test cells are untouched, deliberately — with only 4
   candidates it is too early to spend the split.
2. **Budget will not complete.** 650 of 5,000 instances after substantial
   wall-clock. The plan is to harvest at ~600–800 instances (comparable to what
   the previous run reached), not to finish the nominal budget.
3. **The search has stalled since 8 September.** Best candidate unchanged at
   +0.3128 through ~100 iterations. Root cause found today: reflection failures
   were being silently swallowed (the LM call raised, GEPA logged only "did not
   propose a new candidate"), compounded by a 5-hour network outage. Both now
   fixed — failures are logged, and a systemd timer checks judge-call liveness
   every 15 minutes.
4. **Multi-rule replies are not happening.** The prompt offers up to 5 rules and
   the best candidate explicitly recommends decomposition, but the judge emits
   one action per reply **99.9%** of the time. Unexploited headroom.
5. **`|C|` still grows in ~62% of instances.** The reduction result is real but
   not yet the norm.
6. **Comparison to the previous run is not clean.** `v2e`'s seed template
   differs from `v2d`'s (numeric operators added), so their seed baselines are
   not directly comparable. Only the held-out test number is comparable across
   runs.
7. **38% of conditions are inert** — they match nothing and burn a call. This is
   the largest single efficiency loss in the judge's behaviour.

---

## 11. Where this sits against prior arms — and why the numbers are NOT comparable

| arm | design | headline | measured over |
|---|---|---|---|
| v1 `prelim` | LLM as gatekeeper over a deterministic shortlist | −1.0 PC pts | full cell |
| v2 `profile` | semantic meta-blocking, LLM writes block profiles | positive | full cell |
| v3 `errordiag` | 10% gold error exposure → 8-way failure classification | +5.46 pts | full cell |
| V1 GEPA run5 | LLM picks pairs from a TF-IDF shortlist | +14.0 / +14.3 pts | **all episodes of the split** |
| **v2e (this)** | LLM repairs the partition via predicates | **+4.08 pts** | **64 block pairs per instance** |

**These headline numbers must not be put on the same axis.** Three independent
reasons:

**1. "Instance" means different things.** In V1 an *episode* was **one block
pair**, and the reported +14.0 / +14.3 came from processing **every episode in
the split** — the whole recoverable surface of those cells. In V2 an *instance*
is **64 block pairs sampled from one cell**, and cells hold 37 to 651 usable
Case-3 pairs (2,604 across the matrix). So v2e's +4.08 is what 64 repairs buy;
V1's +14 is what *all* repairs buy. A cell with 651 usable pairs is being
touched at roughly 10% of its surface per instance.

**2. The denominators differ.** V1's ΔPC was over the split's gold pairs; v2e's
ΔPC is over the cell's full gold set while only a sampled subset of block pairs
is visited. The two are not the same quantity.

**3. The task is different.** V1 framed the LLM as a **matcher** choosing pairs
from a TF-IDF shortlist; |C| could only grow, and it had a hard coverage ceiling
(the shortlist). V2 frames it as a **debugger** emitting general predicates over
a partition; |C| can fall, and there is no shortlist ceiling — but each rule must
generalise to records never shown.

**The honest cross-arm statement** is therefore about *mechanism*, not
magnitude: V1 recovered pairs it was shown a shortlist for; v2e recovers pairs
nobody pointed at — **15% of everything it recovers comes from outside the
targeted set**, and it does so while |C| stays roughly flat and falls outright
in 38% of instances. Neither of those properties was available to V1 by
construction.

**To make a comparable claim** we would have to run v2e to exhaustion on a cell
— process *all* its Case-3 block pairs rather than a 64-pair sample — and report
whole-cell ΔPC. That is a straightforward experiment (a cell like
`reclin2/walmart-amazon` has only 37 usable pairs, so one instance already
covers it fully) and it is not yet done.

## 12. Immediate next steps

1. Let the search run to ~600–800 instances now that the proposer is fixed and
   supervised, and see whether anything clears +0.3128.
2. **Run the held-out test evaluation** — 8 instances × 4 test cells = 32
   instances. This is the number the paper needs.
3. Investigate why multi-rule replies never happen (~5 rules available,
   1 used).
4. Attack the 38% inert-condition rate.
5. **Measure the seed and the best candidate on the same 40 validation
   instances** (5,120 calls) so the ΔPC improvement is like-for-like rather than
   an oracle-selected mixture.
6. **Run one cell to exhaustion** — all its Case-3 block pairs, not a 64-pair
   sample — to produce a whole-cell ΔPC that is comparable with the V1 and v3
   arms. `reclin2/walmart-amazon` (37 usable pairs) is covered fully by a single
   instance and is the cheapest candidate for this.
