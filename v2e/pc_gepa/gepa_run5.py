"""GEPA run5: generalization campaign (see RUN5_PLAN.md).

Mixed episode bank across train (dataset x blocker) cells; GEPA's
epoch-shuffled sampler draws minibatches randomly across cells. Judge
qwen3:8b (tunnel), reflection Claude Sonnet (CLI). Also the frozen
test-bank evaluator (--eval-split test).
"""
import argparse
import glob
import json
import os
import subprocess
import time
from collections import defaultdict

import harness
import harness5

HERE = os.path.dirname(os.path.abspath(__file__))
BANK_DIR = os.path.join(HERE, os.pardir, "runs", "bank")

CODE_TEMPLATE = """You are evolving the SAMPLER STAGE of an entity-blocking
debug pipeline that must work across MANY datasets (products,
bibliographic records, restaurants) and blocking methods. It is a pure
Python function executed automatically. It decides which candidate pairs
a small judge LLM sees; the harness then shows at most 250 pairs and
truncates the rendered list at a 20,000-character budget — pairs beyond
that are lost, and pairs never returned can never be recovered.

Current sampler source:
```
<curr_param>
```

Execution feedback from recent episodes (mixed datasets):
<side_info>

Contract (MUST hold):
- define sample(recs_x, recs_y, seed_pair, sim, rng)
- recs_x: list of dicts with key "id" plus dataset-specific string
  fields (schemas VARY BY DATASET — inspect keys, do not hardcode one
  schema); recs_y likewise
- seed_pair: (a_id, b_id) confirmed split match; sim(a_id,b_id)->float
  char-3gram TF-IDF cosine; rng: seeded random.Random
- return list of (a_id, b_id) tuples; ids from recs_x/recs_y. Available
  modules: re, math, itertools, Counter, defaultdict. No imports, no
  IO, 20s time limit (blocks can be 400 x 15,000 records — avoid full
  cross-products on huge blocks).

Write the improved COMPLETE function source inside a single ``` block."""

INSTR_TEMPLATE = """I provided an assistant with the instructions below to
judge candidate record pairs for an entity-blocking debugger that runs
across many datasets (products, bibliographic records, restaurants):
```
<curr_param>
```
Examples of its behavior with feedback:
<side_info>

Write improved instructions that fix the failure patterns shown while
staying DATASET-GENERIC (no product-only or paper-only rules unless
framed as examples of a general principle). The assistant must return
JSON {"match_indices": [...]}. Provide the new instructions inside a
single ``` block."""


def claude_reflection(prompt):
    if not isinstance(prompt, str):
        prompt = "\n\n".join(m.get("content", "") for m in prompt)
    for attempt in range(3):
        try:
            p = subprocess.run(
                ["claude", "-p", "--model", "sonnet"],
                input=prompt, capture_output=True, text=True, timeout=600)
            if p.returncode == 0 and p.stdout.strip():
                return p.stdout
        except subprocess.TimeoutExpired:
            pass
        time.sleep(5 * (attempt + 1))
    raise RuntimeError("claude CLI reflection failed 3x")


def load_banks(blockers):
    banks = {}
    for blk in blockers:
        for path in sorted(glob.glob(os.path.join(BANK_DIR, blk,
                                                  "*.json"))):
            with open(path, encoding="utf-8") as f:
                bank = json.load(f)
            banks[(blk, bank["dataset"])] = bank
    return banks


def split_items(banks, split):
    items = []
    for (blk, ds), bank in sorted(banks.items()):
        for ep in bank["episodes"]:
            if ep["split"] == split:
                items.append({"bank_key": [blk, ds], "ep": ep})
    return items


class GeneralAdapter:
    propose_new_texts = None

    def __init__(self, banks, client):
        self.banks = banks
        self.client = client

    def _bank(self, item):
        return self.banks[tuple(item["bank_key"])]

    def evaluate(self, batch, candidate, capture_traces=False):
        from gepa.core.adapter import EvaluationBatch
        outputs, scores, trajs = [], [], ([] if capture_traces else None)
        for item in batch:
            bank, ep = self._bank(item), item["ep"]
            r = harness5.run_episode_v5(
                self.client, candidate["instruction"],
                candidate["sampler_code"], bank, ep,
                tag=f"gepa5:{ep['episode_id']}")
            outputs.append({"raw": r["raw"], "coverage": r["coverage"],
                            "n_shown": len(r["shown"])})
            scores.append(r["score"])
            if trajs is not None:
                trajs.append({"item": item, "r": r})
        return EvaluationBatch(outputs=outputs, scores=scores,
                               trajectories=trajs)

    def make_reflective_dataset(self, candidate, eval_batch,
                                components_to_update):
        comp = components_to_update[0]
        out = []
        for t in eval_batch.trajectories:
            bank, ep = self._bank(t["item"]), t["item"]["ep"]
            r = t["r"]
            fb = harness5.feedback_v5(bank, ep, r, comp)
            inputs = (r["user_message"][:5000] if comp == "instruction"
                      else f"[{bank['dataset']}] episode "
                           f"{ep['episode_id']}, blocks "
                           f"{len(ep['a_ids'])}x{len(ep['b_ids'])}")
            out.append({"Inputs": inputs,
                        "Generated Outputs": r["raw"][:800],
                        "Feedback": fb})
        return {comp: out}


def evaluate_split(client, banks, split, instruction, sampler_code,
                   outpath):
    per = defaultdict(lambda: [0, 0, 0, 0])  # hits, targets, fp, cov
    t0 = time.time()
    for item in split_items(banks, split):
        bank, ep = banks[tuple(item["bank_key"])], item["ep"]
        r = harness5.run_episode_v5(client, instruction, sampler_code,
                                    bank, ep, tag=f"eval5:{split}")
        key = f"{bank['dataset']}"
        per[key][0] += len(r["hits"])
        per[key][1] += len(r["targets"])
        per[key][2] += len(r["props"] - r["hits"])
        per[key][3] += r["coverage"]
        print(f"{ep['episode_id']}: {len(r['hits'])}/{len(r['targets'])}"
              f" fp={len(r['props'] - r['hits'])} cov={r['coverage']}",
              flush=True)
    th = sum(v[0] for v in per.values())
    tt = sum(v[1] for v in per.values())
    out = {"split": split,
           "micro_recall": th / tt if tt else 0.0,
           "targets_recovered": f"{th}/{tt}",
           "false_positives": sum(v[2] for v in per.values()),
           "coverage": f"{sum(v[3] for v in per.values())}/{tt}",
           "per_dataset": {k: {"recovered": f"{v[0]}/{v[1]}",
                               "fp": v[2], "coverage": v[3]}
                           for k, v in sorted(per.items())},
           "runtime_s": round(time.time() - t0, 1)}
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blockers", nargs="+", default=["lambdafold"])
    ap.add_argument("--model", default="qwen3:8b")
    ap.add_argument("--budget", type=int, default=5000)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--eval-split", default=None)
    ap.add_argument("--instruction-file", default=None)
    ap.add_argument("--sampler-file", default=None)
    args = ap.parse_args()
    os.makedirs(args.run_dir, exist_ok=True)

    banks = load_banks(args.blockers)
    log = harness.EventLog(os.path.join(args.run_dir, "llm_events.jsonl"))
    client = harness.OllamaClient(args.model, log, think=False,
                                  num_ctx=16384)

    instr = harness.SEED_INSTRUCTION
    code = harness5.SEED_SAMPLER
    if args.instruction_file:
        with open(args.instruction_file, encoding="utf-8") as f:
            instr = f.read()
    if args.sampler_file:
        with open(args.sampler_file, encoding="utf-8") as f:
            code = f.read()

    if args.eval_split:
        tag = ("best" if args.instruction_file else "seed")
        evaluate_split(client, banks, args.eval_split, instr, code,
                       os.path.join(args.run_dir,
                                    f"eval_{args.eval_split}_{tag}.json"))
        return

    import gepa
    train = split_items(banks, "train")
    val = split_items(banks, "val")
    print(f"train={len(train)} val={len(val)} across "
          f"{sorted(set(tuple(i['bank_key']) for i in train))}",
          flush=True)

    def reflection_lm(prompt):
        out = claude_reflection(prompt)
        log.append({"ts": time.time(), "tag": "reflection-sonnet",
                    "raw_response": out, "messages": prompt})
        return out

    with open(os.path.join(args.run_dir, "config.json"), "w") as f:
        json.dump({"arm": "run5-generalization",
                   "blockers": args.blockers, "task_model": args.model,
                   "reflection": "claude sonnet CLI",
                   "budget": args.budget, "n_train": len(train),
                   "n_val": len(val),
                   "seed_candidate": {"instruction": instr,
                                      "sampler_code": code}},
                  f, indent=2)
    adapter = GeneralAdapter(banks, client)
    result = gepa.optimize(
        seed_candidate={"instruction": instr, "sampler_code": code},
        trainset=train, valset=val, adapter=adapter,
        reflection_lm=reflection_lm,
        reflection_minibatch_size=4,
        module_selector="round_robin",
        reflection_prompt_template={"instruction": INSTR_TEMPLATE,
                                    "sampler_code": CODE_TEMPLATE},
        max_metric_calls=args.budget,
        run_dir=args.run_dir, seed=0, display_progress_bar=False,
        raise_on_exception=False)
    best = result.best_candidate
    for name, key in (("best_instruction.txt", "instruction"),
                      ("best_sampler.py", "sampler_code")):
        with open(os.path.join(args.run_dir, name), "w",
                  encoding="utf-8") as f:
            f.write(best[key])
    print("FINISHED", result.total_metric_calls, "calls,",
          result.num_candidates, "candidates")


if __name__ == "__main__":
    main()
