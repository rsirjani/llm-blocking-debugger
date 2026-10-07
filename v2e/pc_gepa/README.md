# pc_gepa — GEPA prompt optimization for LLM blocker-debugging

Two designs have been run. **V2 (`v2.py`) is current**; V1 (`run6`/`run7`,
`gepa_run.py`, `harness.py`) is kept because its results stand and its
lessons motivated V2.

- **V1** ran on tin-desktop (Windows, RTX 5090) over Tailscale, project at
  `D:\blocking-gepa\` (C: is full — nothing may be written there).
- **V2** runs on [dblab-gpu] (3× A5000), driven from the linux box: the
  driver, GEPA and Sonnet reflection are local, the judge is reached over
  three ssh tunnels. See "Ops notes".

---

## V2 — the LLM repairs the partition (current)

The judge no longer picks pairs. It is shown two groups of records and one
known same-entity pair split across them, and returns an **edit to the
partition stated as a predicate**: move records matching a condition from
one group to the other, put matching records into a group of their own, or
do nothing. There is no blanket merge — every change must be a rule over
the whole group, so it applies to the many records the judge never saw.

**An instance is a whole-cell replay.** 64 Case-3 block pairs from one
(blocker, dataset) cell are processed in sequence against a live partition;
each step sees what earlier steps produced, and P1 (every record in exactly
one group) is asserted after every edit. The cell is scored as a whole.

**Score prices both axes.**

    score = (net same-entity pairs reunited across the WHOLE cell
             / separated pairs in the block pairs actually visited)
            - 5.0 * (relative growth in within-group comparisons)

The numerator is global, so an edit that drags a record out of a group it
belonged in is debited for every pair it leaves behind. The denominator is
local, so the signal is not diluted by the cell's untouched majority
(whole-cell ΔPC gave ±0.003 noise per instance).

**Nothing is ever truncated.** The sample is a fixed 0.35 fraction of each
group with no cap, and records render with all fields at full length.
Block pairs whose full sample would not fit the context are excluded up
front, and a cell is dropped entirely if >10% of its Case-3 pairs are such
(`--max-oversize-frac`) — keeping only a cell's small block pairs would
measure an easier problem than the cell poses. `MAX_CTX = 18432` is the
smallest context at which no block pair of the 14 admitted cells is
excluded; at 12288 four were, and capping fields at 6 / values at 120
chars cut 55% / 23% of records respectively.

### The matrix (14 cells, 4 blockers × 6 datasets)

| | amazon-google | walmart-amazon | dblp-acm | cora | abt-buy | dblp-scholar |
|---|---|---|---|---|---|---|
| lambdafold | oversize | oversize | train | train | TEST | oversize |
| simcse | train | PC .207 | train | train | TEST | oversize |
| reclin2 | train | train | 3 case-3 | train | TEST | TEST |
| klsh | PC .362 | PC .377 | train | train | PC .442 | oversize |

10 train / 4 test cells, 2,604 usable block pairs. Held-out datasets are
abt-buy and dblp-scholar. Cells are admitted only at a comparable
operating point (PC in [0.45, 0.80]); klsh cannot reach it on product
catalogues, and its dblp-scholar blocks have a median block-pair size of
17,458 records.

### What is optimized, and what the reflector knows

The single evolved component is the **whole message template**, with
required slots `{SEED}`, `{SAMPLE_X}`, `{SAMPLE_Y}`, `{STATS}`. The
harness writes no prose of its own. `{STATS}` is required because the
reflector is told how the comparison count responds to group sizes, and a
template that dropped it would leave the judge holding a rule whose inputs
it cannot see.

Reflection is Sonnet via `claude -p`, and is **blind**: no dataset or
blocker identity, record text content-masked (`<Tn>` placeholders), and a
leakage gate that re-prompts if the draft contains tokens taken from the
data. It is given facts about the machinery — not advice:

- which group is X and which is Y carries no meaning (ordered by internal
  identifier), so a standing preference for one label is arbitrary
- moving k records out of a group of size a into one of size b changes the
  comparison count by exactly `k(b − a + k)` — negative whenever
  `k < a − b`, so a move can reunite records *and* lower the total
- taking `ka` and `kb` records from the two groups into one new group
  changes it by `ka·kb − ka(a−ka) − kb(b−kb)`; drawing on one group only
  always lowers it, by `k(a−k)`
- records leaving a group are separated from everything left behind,
  including pairs already correctly grouped
- the sample is a fixed uniform fraction, so what is shown is proportional
  to true group size

Which operation suits which situation, and when, is left to the outcomes.

### Current run

`runs/gepa/v2d_qwen38` — qwen3.8:27b-q4_K_M judge, thinking off, 64 block
pairs per instance, budget 5000 instances (= 320k judge calls, ~345 h at
~3.7 s/call across 3 GPUs), minibatch 8, val 40 instances (4 per train
cell), test 8 per cell. Budget counts *instances*, so val panel size costs
no wall-clock — it moves budget from exploration to selection. Selection
noise on a thin val panel is the known failure mode here (run4's winner's
curse at 29 episodes).

GEPA persists state in the run dir, so the run is harvestable at any point;
stopping early yields a less-evolved candidate rather than nothing.

---

## V1 — the LLM picks pairs (superseded, results stand)

### Scenario

1. Mutually-exclusive blocker (each record in exactly one block, asserted
   per-record — never validated via pair counts) at a deliberately
   imperfect operating point (PC < 1).
2. Missed gold pairs grouped by block-pair → **episodes**; one missed pair
   exposed as SEED, the rest hidden TARGETS.
3. The LLM judges which of the top-m cross-block candidate pairs (char-3gram
   TF-IDF cosine) are true matches.
4. Score = F1 vs targets. Gold-derived feedback enters ONLY GEPA's
   reflection on the train split; test frozen.

### Operating point (runs 1–4)

amazon-google (1363×3226, 1300 gold); blocklib 0.1.7 `lambda-fold` Λ=1
(⇒ mutually exclusive), bf-len 2000, 5 hash funcs, seed 0. K sweep
(PYTHONHASHSEED=0): K=6→PC .64 | 10→.57 | 15→**.506** | 20→.39 | 30→.19.
K=15: 642 missed pairs, 99 episodes, splits 39/29/31, targets 121/140/95.
Gold-blind ceiling@40 = 0.803. Judge qwen3:8b.

Use `generate_candidate_blocks` per table, NOT `generate_blocks` — its
signature filter silently drops records.

### Results, frozen test (31 episodes, 95 targets, base PC 50.6%)

| arm | reflection | test targets | FP / added | PC pts |
|---|---|---|---|---|
| pair-pick seed | — | 24/95 | ~250 | +1.85 |
| run1 instruction-only | qwen3:8b | **49/95** | ~280 | **+3.77** |
| run2 instr+spec DSL | qwen3:8b | 45/95 | 303 | +3.46 |
| run3b rule-induction seed | — | 32/95 | 61k pairs (over budget) | +2.46 |
| run3b rule-induction best | qwen3:8b | 6/95 | 4.4k | +0.46 |
| run4 pipeline seed | — | 40/95 | 268 | +3.08 |
| run4 pipeline best | **Sonnet CLI** | 45/95 | 254 | +3.46 |

- run2: spec mutations all rejected on val — 8B reflection can't search the
  scaffold space.
- run3b: 8B predicates are either bulk-merge (61k added pairs) or too
  narrow (6/95). Val winner did not transfer (0.106 val vs 0.067 test).
- run4: candidate = {instruction, sampler_code}, 2030 rollouts, val
  0.115 → 0.383 (3.3×), but top-line stayed run1's 49/95 — **winner's
  curse** at 46 candidates on a 29-episode val panel. This is why V2's val
  panel is 40 instances (2,560 block-pair repairs).
- run5 generalization: mixed-dataset bank transferred to held-out sets
  (abt-buy +14.0 PC pts, dblp-scholar +14.3). See `RUN5_PLAN.md`.

### Lessons that V2 encodes

- LLM vetoes over a good deterministic shortlist destroy value — rank, do
  not exclude.
- Prompts must ask about block/family overlap, not product identity.
- A cap on proposals games the metric: in deployment the number of true
  positives is unknown, so no cap anywhere.

---

## Files

**V2:** `v2.py` (harness, scoring, feedback, adapter, CLI),
`build_bank2.py` (bank schema v2: gold, full assignment, records,
per-block-pair failures), `bankio.py` (instance derivation),
`mask.py` (`ContentLexicon`, leakage audit), `harness.py` (Ollama client
with round-robin over endpoints, event log).

**V1:** `run_lambdafold.py`, `analyze_missed.py`, `build_episodes.py`,
`gepa_run.py`, `smoke_and_eval.py`, `run7.py`.

## Ops notes

- **V2 serving:** one Ollama per GPU (`CUDA_VISIBLE_DEVICES=i`, ports
  11570–11572) — Ollama's multi-GPU mode is a layer split, so one model
  spread over three cards makes them take turns instead of computing in
  parallel. `OLLAMA_NUM_PARALLEL=3` at `num_ctx 18432` fits an A5000
  (~19 GB); 4 slots needs ~23.8 of 24.5 GB and spills. **Warm the servers
  sequentially** — three simultaneous loads make GPU discovery time out and
  a server silently falls back to partial CPU offload (6× latency). Verify
  with `/api/ps` and check `size_vram == size`; nvidia-smi memory alone
  does not reveal a partial offload.
- `OLLAMA_URL` takes a comma-separated list; calls go round-robin.
- `tunnel_dblab.sh` holds the three tunnels. It is deliberately gentle
  (5-min checks, two consecutive failures before acting, one attempt per
  cycle) — a tighter loop with no key loaded got this host fail2ban'd.
- **V1 / tin-desktop:** Ollama serve headless via ssh
  (`OLLAMA_MODELS=D:\ollama-models`); the Windows tray app fails under ssh.
  Avoid `2>nul` in cmd.exe (creates a literal file `nul` on D:); copy with
  `tar cf - | ssh tar xf -`.
- blocklib returns block members as INTEGER POSITIONS into the input list;
  keys are bit-string signatures.
- Always `PYTHONHASHSEED=0`.
