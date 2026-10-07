# run5 — Generalized blocker-debug pipeline optimization: results report

Companion files: `pc_vs_rollouts.png` (optimization curve),
`EVOLUTION.md` (prompt/algorithm trajectory with real retrieval
samples), `distribution_stats.json` (raw numbers below),
`../RUN5_PLAN.md` (preregistered protocol).

## 1. Experiment

**Question.** Can one LLM-debugging pipeline — a judge prompt plus an
executable sampling stage, both evolved by GEPA with Claude Sonnet as
reflection LLM and qwen3:8b as the cheap pipeline/judge model — be
optimized across several datasets at once, and transfer to datasets it
never saw?

**Blocker.** blocklib `lambda-fold` LSH, Λ=1 (each record hashed to
exactly one signature → mutually exclusive blocks, per-record P1
asserted). Per dataset, K (signature bits) auto-selected from
{2,…,30} to bring PC closest to 0.6 — deliberately imperfect blocking.
Single blocker so far: blocker-generalization is untested (SimCSE,
reclin2, klsh cells planned).

**Episodes.** A blocker error is a gold pair split across two blocks.
Episode = directional block-pair (X, Y) holding ≥2 such missed pairs;
one is exposed to the LLM as the SEED, the rest are hidden TARGETS to
recover. Score per episode: recall of targets among proposed pairs −
0.02 × false positives, ≤25 proposals. Recall here = per-episode PC
recovery.

**Optimization.** GEPA, candidate = {instruction, sampler_code
(sandboxed Python)}. Seed candidate deliberately dumb: generic 9-line
prompt + sampler returning ALL cross-block pairs in id order (no
TF-IDF, no ranking). Minibatches (4 episodes) drawn epoch-shuffled from
the MIXED train bank, so specializing to one dataset is punished by the
next batch. 5,026 rollouts, 66 candidates, 175 Sonnet mutations,
round-robin between the two components. Gold appears only in seeds (by
design) and in reflection-side feedback on train episodes — never in
the judge prompt, never in val/test selection inputs beyond scoring.

## 2. Train / test split and the two distributions

Split is at DATASET level (generalization test), then episode level
inside train (75/25 train/val, seed 0):

| role | dataset | domain | K | PC | episodes (tr/va/te) | targets | med cross-pairs/ep | max cross-pairs/ep |
|---|---|---|---|---|---|---|---|---|
| train | amazon-google | software products | 8 | 0.597 | 52/18/– | 375 | 14,450 | 185,300 |
| train | walmart-amazon | retail products | 8 | 0.612 | 54/19/– | 255 | 130,732 | 1,445,220 |
| train | dblp-acm | bibliographic | 6 | 0.632 | 49/17/– | 672 | 14,097 | 313,851 |
| train | fodors-zagats | restaurants | 15 | 0.705 | 3/1/– | 6 | 1,681 | 6,900 |
| **test** | abt-buy | consumer electronics | 6 | 0.552 | –/–/58 | 376 | 2,786 | 53,728 |
| **test** | dblp-scholar | bibliographic (Scholar) | 6 | 0.592 | –/–/175 | 1,901 | 121,875 | 6,452,150 |

**Train distribution:** 158 train + 55 val episodes, 1,308 hidden
targets; dominated by two product catalogs (schema: title/manufacturer/
price/description) plus one clean bibliographic pair (title/authors/
venue/year) and a token restaurant presence. Median episode ≈ 14k–130k
possible cross pairs; worst ≈ 1.4M.

**Test distribution vs train:** deliberately shifted on three axes.
(1) *Domain*: abt-buy is consumer electronics — near the product
training domain but a different catalog style (long name field, no
manufacturer column on the Buy side); dblp-scholar is bibliographic
like dblp-acm but the Scholar side is noisy OCR-ish text with internal
duplicates (gold not 1:1). (2) *Scale*: dblp-scholar episodes are the
largest anywhere in the study — up to 430×15,047 records = 6.45M
cross pairs per episode, 4.5× the worst training episode; test median
cross-pairs on dblp-scholar (~122k) matches walmart-amazon, but the
tail is far heavier. (3) *Error mass*: test has 2,277 hidden targets —
1.7× the entire train+val target mass — and a lower blocking PC on
abt-buy (0.552) than any train set, i.e. a more broken blocker than
seen in training. Overlap that remains: same blocker family, same
seed-per-episode construction, and dblp-acm gives the model *a*
bibliographic schema in training (declared: this is dataset-transfer,
not domain-zero-shot for bibliography).

## 3. Optimization curve

`pc_vs_rollouts.png`: x = rollouts consumed at candidate discovery,
y = mixed-val episode score (PC-recovery proxy), red line = best so
far. Milestones (annotated):

| candidate | rollouts | val score | what changed (see EVOLUTION.md) |
|---|---|---|---|
| #0 seed | 0 | 0.0005 | dumb all-pairs dump — truncates at budget, coverage ~0 |
| #1 | 87 | 0.227 | first sampler rewrite: sim()-ranked top pairs |
| #2 | 222 | 0.281 | instruction rewrite: match criteria + surface-noise rules |
| #6 | 474 | 0.309 | sampler: per-record coverage + budget awareness |
| #13 | 947 | 0.425 | sampler: token-index prefilter for huge blocks + cost model |
| #54 (best) | 3,948 | 0.440 | refinements on the #13 line |

Note x-axis is rollouts, not GEPA "iterations": a mid-run tunnel outage
spun ~1,300 no-op iterations that consumed no rollouts; rollouts are
the honest effort axis.

## 4. Held-out test result (frozen, single pass)

| pipeline | recovered | recall | FP | coverage |
|---|---|---|---|---|
| seed | 10/2,277 | 0.4% | 594 | 16/2,277 |
| **optimized (#54)** | **918/2,277** | **40.3%** | 942 | 1,263/2,277 |

Per dataset, in PC points on the full dataset gold:

| dataset | recovered | precision | base PC | PC after | ΔPC |
|---|---|---|---|---|---|
| abt-buy | 154/376 | 0.60 | 0.552 | 0.692* | **+14.0 pts** |
| dblp-scholar | 764/1,901 | 0.48 | 0.592 | 0.735* | **+14.3 pts** |

\* counting recovered targets only; the 58/175 exposed seeds per
dataset are given, not credited. FP cost (~1 per true pair) is far
inside the 1%-of-|C| addition budget.

## 5. Comparison with earlier versions

Earlier runs optimized on a single dataset (amazon-google, K=15) and
were tested on held-out *episodes of the same dataset*; run5 is tested
on held-out *datasets*. Numbers are therefore not one table — the test
universes differ — but the arc is:

| run | candidate space | reflection | test universe | test result |
|---|---|---|---|---|
| run1 | instruction only | qwen3:8b | AG episodes | 49/95 targets (+3.8 PC pts), best single-dataset |
| run2 | instruction + spec DSL | qwen3:8b | AG episodes | 45/95; spec search null |
| run3b | rule-induction prompt | qwen3:8b | AG episodes | 6/95; predicates fragile (seed variant: 32/95 but 61k added pairs) |
| run4 | instruction + sampler code | Sonnet | AG episodes | 45/95; val 3.3× but winner's curse |
| **run5** | instruction + sampler code | Sonnet | **unseen datasets** | **918/2,277 = 40.3%, +14 PC pts per dataset** |

Two like-for-like observations: (a) on the shared recall-style metric,
run5's cross-dataset 40.3% sits just below run1's within-dataset 51.6%
— transferring to unseen data cost ~11 recall points, not collapse;
(b) every 8B-reflection attempt to search beyond prompt text failed
(runs 2–3b), while Sonnet-reflection succeeded twice (runs 4–5) — the
reflection model, not the judge, was the binding constraint on
pipeline-space search.

## 6. Reproducibility

Bank: `runs/bank/lambdafold/` (build_bank.py, PYTHONHASHSEED=0, seed 0).
Run artifacts: `runs/gepa/run5_general_sonnet/` — config.json (seed
candidate), candidates.json (all 66), lineage.json, gepa_state.bin,
llm_events.jsonl (every rollout + every Sonnet reflection, replayable),
best_instruction.txt, best_sampler.py, eval_test_{seed,best}.json.
Samplers are deterministic per episode; retrieval samples in
EVOLUTION.md are exact re-renders.
