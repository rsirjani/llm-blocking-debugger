# run5 — Generalization campaign plan (preregistered 2026-08-19)

User-specified design, verbatim intent: optimize ONE debugging
prompt/pipeline over MANY datasets and MANY blockers, hold out datasets
for generalization testing, and start the sampler from the dumbest
possible baseline (all records of both blocks + seed in the prompt),
NOT from TF-IDF.

## Protocol

1. **Run every blocking method on every dataset.** Each (dataset,
   blocker) cell yields a set of mutually exclusive blocks (P1: every
   record in exactly one block, asserted per record), at an operating
   point auto-tuned so PC ≈ 0.6 (imperfect on purpose). Each cell's
   blocks + missed-gold-derived episodes = one "block set" with its
   dataset/blocker origin recorded.
2. **Episode bank.** Episode = directional block-pair with ≥2 missed
   gold pairs; 1 exposed seed, rest hidden targets. Bank = union of
   episodes across all TRAIN (dataset × blocker) cells.
3. **Optimization.** GEPA with candidate = {instruction, sampler_code}.
   Rollouts sample episodes randomly across the mixed bank (epoch-
   shuffled), so the evolving prompt/sampler cannot specialize to one
   dataset or blocker. Judge: qwen3:8b local. Reflection: Claude Sonnet
   via CLI. Large budget (user target ~1000 iterations; executed in
   resumable chunks sized to the GPU).
4. **Seed candidate (deliberately dumb).** instruction = the generic
   run1 seed prompt. sampler_code = return ALL cross-block pairs in id
   order (no TF-IDF, no ranking). Hard physics caps only (max pairs +
   prompt char budget), with truncation reported in feedback so the
   optimizer feels the cost. sim() (char-3gram TF-IDF) remains available
   to EVOLVED code as a primitive — the optimizer may rediscover it, but
   does not start from it. Similarity scores are not shown to the judge.
5. **Held-out test.** The left-out datasets are blocked with ALL the
   same blocking methods; their episodes form the test bank. The frozen
   optimized candidate is evaluated on the test bank (random-sampled
   during any interim checks; full pass for the final number). Metric:
   micro target-recall (= PC recovery), FP count, per-dataset and
   per-blocker breakdown. Comparison arm: the dumb seed candidate.

## Dataset split (clean-clean, complete gold only)

- TRAIN: amazon-google, walmart-amazon, dblp-acm, fodors-zagats
- TEST (held out): abt-buy (near-domain product transfer),
  dblp-scholar (bibliographic transfer)
- Excluded: beer, itunes-amazon (incomplete gold per manifest),
  wdc-block (size), cora/musicbrainz/ncvr (single-table dirty — episode
  design is two-table; future extension).

## Blocker slate (staged)

- B — blocklib lambda-fold Λ=1 (Python): ACTIVE NOW, all datasets.
- D — SimCSE kNN + Louvain (Mugeni & Amagasa): next; needs torch+HF on
  tin-desktop (HF cache already at D:\hf, weights to D:).
- A — reclin2 (R), C — klsh (R): require R install on tin-desktop;
  queued after D. Bank format is blocker-agnostic; test bank gets
  regenerated over all landed blockers before the final test pass.

## Scoring

score(episode) = recall(hidden targets among proposed pairs)
                 − 0.02 × false positives, proposals capped at 25.
Selection on mixed val (25% of train-cell episodes). Test bank never
touched during optimization.

## Status log

- 2026-08-19: plan written; blocklib bank building; harness5/gepa_run5
  in progress. run4 artifacts (single-dataset pipeline evolution) are
  the direct ancestor — see RUN4_BEFORE_AFTER.md.
- 2026-08-19: bank built (6 datasets, PC 0.55–0.71, train 158 / val 55 /
  test 233 episodes). run5 launched: budget 5000, qwen3:8b judge via
  tunnel, Sonnet CLI reflection. One tunnel outage mid-run (no budget
  lost; watchdog added).
- 2026-08-20: run5 FINISHED — 5,026 rollouts, 66 candidates, mixed val
  0.0005 → 0.440. FROZEN TEST (held-out abt-buy + dblp-scholar, 233
  episodes, 2,277 hidden targets):
    seed pipeline:      10/2,277 recovered (0.4%), coverage 16, FP 594
    optimized pipeline: 918/2,277 (40.3%), coverage 1,263, FP 942
    abt-buy:      154/376, FP 104 → +14.0 PC pts (base PC 0.552)
    dblp-scholar: 764/1,901, FP 838 → +14.3 PC pts (base PC 0.592)
  Evolved sampler: schema-agnostic token inverted index prefilter →
  TF-IDF rerank → prompt-budget cost model → per-record round-robin
  coverage. Artifacts: runs/gepa/run5_general_sonnet/{best_instruction
  .txt,best_sampler.py,candidates.json,llm_events.jsonl}.
  LIMITS: single blocker so far (blocker-generalization untested — needs
  SimCSE/reclin2/klsh bank cells); seed-driven triage assumed; val
  winner picked from 66 candidates on 55 episodes (selection noise).
