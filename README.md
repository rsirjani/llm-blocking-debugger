# LLM-Based Debugging and Explainability of Blocking Methods in Entity Resolution

Artifacts for the EDBT 2027 short paper of the same title (Ramtin Sirjani, Farhan Patel, Mostafa Milani;
The University of Western Ontario).

The method shows a large language model the two blocks that a known true pair was
split across and asks for an executable repair: a predicate over record fields paired
with a partition edit. Each repair is applied to the blocker's own partition and
scored by the exact gain in co-blocked true pairs and the exact cost in comparisons.
Explanation quality is measured by recovery of *withheld* pairs, those the model was
never shown.

## Layout

| path | what it is |
|---|---|
| `blocking_methods/` | the standard-key (`A_standard_reclin2`) and LSH (`B_lsh_blocklib`) blockers with their pilot outputs. The klsh and SimCSE/Louvain blockers and lambdafold are built by `v2e/pc_gepa/block_r.R`, `build_bank_r.py`, `build_bank_simcse.py` and `run_lambdafold.py`. |
| `setup_datasets.sh` | fetches the public benchmarks (Amazon-Google, Walmart-Amazon, DBLP-ACM, Cora, Abt-Buy, DBLP-Scholar, MB-20K). Datasets are not committed. |
| `v2e/pc_gepa/` | the search and evaluation code. `v2.py` is the driver: builds the judge message, replays a cell, scores it, and runs reflective prompt evolution. `pc_gepa/README.md` holds the full design notes, including the scoring formula and the 14-cell matrix. |
| `v2e/runs/gepa/v2e_qwen38/prompts/best.txt` | the optimized instruction the paper evaluates. (`v2e/pc_gepa/best_prompt.txt` is an earlier V1-design file and is not the evaluated instruction.) |
| `v2e/runs/gepa/v2e_qwen38/prompts/seed.txt` | the strategy-free initial prompt |
| `v2e/runs/gepa/v2e_qwen38/prompts/abl_no{1..7}.txt`, `force_*.txt`, `noseed.txt` | the manipulated instructions (one method step deleted; direction forced; disclosed pair withheld) |
| `v2e/q/jobs/` | one shell script per queued experiment (sweeps, replicate, manipulated instructions, cross-judge runs, operating-point rebuilds), with the exact arguments used. The validation-panel re-score, the held-out and MB-20K runs and the six-instance spot-check were launched by hand with `v2.py`; their result files are shipped, their launch lines are not. |
| `v2e/runs/gepa/v2e_qwen38/*.out`, `*.log` | the logs those jobs wrote |
| `v2e/src/loaders.py` | dataset loaders |

## Result files behind the paper

All under `v2e/runs/gepa/v2e_qwen38/` unless noted.

| file | paper |
|---|---|
| `eval_val_exh_best.json`, `eval_val_exh_seed.json` | exhaustive sweeps over the ten training cells, optimized and initial instruction (Table 3, Figure 2, the score decomposition) |
| `eval_val_exh_best_v1.json`, `eval_val_exh_seed_v1.json` | the replicate of both sweeps over the same boundary set in a different order |
| `eval_val_best.json`, `eval_val_seed.json` | the 120-instance validation panel, both instructions (win rate) |
| `eval_test_best.json` | the 80 held-out instances, optimized instruction |
| `../../../analysis/v2e_qwen38/run_log.txt` | the search's own log; the per-candidate validation means quoted in Section 5 are read from here (the trajectory figure of earlier drafts was removed) |
| `../../../analysis/v2e_qwen38/candidates.json` | the text of every accepted candidate instruction, in order |
| `serving_exh_v1_{best,seed}_{before,after}.json` | server state snapshots (model digest and keep-alive expiry) around the replicate sweeps. No such snapshot exists for the first sweeps; the paper identifies their server from the resident server's state. |
| `llm_events_spotcheck.jsonl` | the re-run of six validation instances against the resident server |

Behavioural tallies (Table 2) and the rule-vocabulary and per-blocker statistics of
Section 5 are computed from the per-step exemplars in the `eval_*` files by
`v2e/pc_gepa/rule_stats.py`, whose docstring states every definition used (what counts
as a predicate, an inert rule, a single-token value, a full-group sweep). The Section 5
shape figures (one-condition share, operator and field mix, full-group share, median
group size) are from the 120-instance validation panel, `eval_val_best.json` and
`eval_val_seed.json`, regrouped by blocker: run `rule_stats.py --per-cell` and pool the
cells of each blocker. The same counters over the exhaustive files give lower full-group
shares (for example 57.0% rather than 72.4% on reclin2/Cora), as the sweeps visit every
boundary while the panel draws 64 per instance. Table 2's sweep rows use the exhaustive
files. The validation-panel rows were cross-checked against `llm_events.jsonl`, which is
not shipped.

## Models and serving

The judge is `qwen3.8:27b` served by Ollama on three NVIDIA A5000 GPUs, reached over
ssh tunnels from the driver. Job scripts under `v2e/q/jobs/` name the model for every
run, including the cross-judge checks (`gemma4:26b`, `llama3.1:8b`, `qwen3.6:27b`; a
`mistral-small3.2:24b` run was attempted and failed on serving memory and time limits). The
cross-judge files carry score and parse counts but no withheld-pair accounting. The reflection step of the search called the Claude
CLI (Sonnet through the third accepted candidate, Opus afterwards, and briefly a third
model whose proposals were all rejected); `v2.py` records the sequence. The full
reflection prompt, including the objective and the comparison-count arithmetic given
to the reflection model, is `REFLECT_TEMPLATE` in `v2.py`. There is no job script for
the search itself; it was driven interactively by `v2.py` over eighteen days. The paper's threats
section notes that results reproduce bit-exactly only under a fixed serving
configuration.

## Not included

- The raw per-call LLM event logs (`llm_events.jsonl`, 0.9 GB and 1.4 GB) exceed
  GitHub's file limit. Available on request.
- Datasets: `setup_datasets.sh` fetches the Magellan/DeepMatcher benchmarks. Cora and
  MusicBrainz 20K are not fetched by it; both come from the Leipzig database group's
  entity-resolution benchmark page (https://dbs.uni-leipzig.de/research/projects/benchmark-datasets-for-entity-resolution).
  Place them where `v2e/src/loaders.py` expects.
- The block-pair banks under `v2e/runs/bank*` are regenerated by
  `v2e/pc_gepa/build_bank*.py`.
- Paths in the scripts point at the cluster the experiments ran on; adjust before use.
- The reclin2 and klsh blockers need R (`Rscript`) with the reclin2 package; the
  operating-point rebuilds of reclin2 failed for that reason, so only simcse was rebuilt
  (`eval_val_e5_pc{45,75}_*`).
- An initial-prompt run on the held-out cells was launched twice
  (`test_seed_run.log`, `harvest_test_seed.log`) and did not complete.
