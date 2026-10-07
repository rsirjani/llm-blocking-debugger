"""GEPA run4: pipeline evolution, run FROM the linux box.

Task LLM: qwen3:8b on tin-desktop's 5090 through the ssh tunnel
(OLLAMA_URL=http://127.0.0.1:11434). Reflection LLM: Claude Sonnet via
the local `claude` CLI (subprocess, argv-list, prompt on stdin — no
shell). Candidate = {instruction, sampler_code}; sampler_code is real
Python executed sandboxed by harness4.
"""
import argparse
import json
import os
import subprocess
import time

import gepa
from gepa.core.adapter import EvaluationBatch, GEPAAdapter

import harness
import harness3
import harness4

HERE = os.path.dirname(os.path.abspath(__file__))

CODE_TEMPLATE = """You are evolving the SAMPLER STAGE of an entity-blocking
debug pipeline. It is a pure Python function executed automatically (no
LLM calls inside). It decides which candidate record pairs a small judge
LLM gets to see; pairs it does not surface can never be recovered.

Current sampler source:
```
<curr_param>
```

Execution feedback from recent episodes (coverage = true pairs surfaced;
false positives happen later, at the judge — your job is coverage and
list quality under the token cost noted):
<side_info>

Contract (MUST hold):
- define sample(recs_x, recs_y, seed_pair, sim, rng)
- recs_x: list of dicts with keys id,title,manufacturer,price,description
  (block X, Amazon side); recs_y likewise (block Y, Google side)
- seed_pair: (a_id, b_id) confirmed split match; sim(a_id,b_id)->float
  char-3gram TF-IDF cosine; rng: seeded random.Random
- return list of (a_id, b_id) tuples, at most 60; ids must come from
  recs_x/recs_y. Available modules: re, math, itertools, Counter,
  defaultdict. No imports, no IO, 10s time limit.

Write the improved COMPLETE function source inside a single ``` block."""

INSTR_TEMPLATE = """I provided an assistant with the instructions below to
judge candidate record pairs for an entity-blocking debugger:
```
<curr_param>
```
Examples of its behavior with feedback:
<side_info>

Write improved instructions that fix the failure patterns shown in the
feedback while keeping what works. The assistant must return JSON
{"match_indices": [...]}. Provide the new instructions inside a single
``` block."""


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


class PipelineAdapter(GEPAAdapter):
    def __init__(self, recs, client, log):
        self.recs = recs
        self.client = client
        self.log = log

    def evaluate(self, batch, candidate, capture_traces=False):
        outputs, scores, trajs = [], [], ([] if capture_traces else None)
        for ep in batch:
            r = harness4.run_episode_v4(
                self.client, candidate["instruction"],
                candidate["sampler_code"], ep, self.recs,
                tag=f"gepa4:{ep['episode_id']}")
            outputs.append({"raw": r["raw"],
                            "n_pairs": len(r["pairs"]),
                            "coverage": r["coverage"]})
            scores.append(r["score"])
            if trajs is not None:
                trajs.append({"ep": ep, "r": r})
        return EvaluationBatch(outputs=outputs, scores=scores,
                               trajectories=trajs)

    def make_reflective_dataset(self, candidate, eval_batch,
                                components_to_update):
        comp = components_to_update[0]
        recs_out = []
        for t in eval_batch.trajectories:
            r = t["r"]
            fb = harness4.feedback_v4(t["ep"], self.recs, r, comp)
            inputs = (r["user_message"][:5000] if comp == "instruction"
                      else f"episode {t['ep']['episode_id']}, blocks "
                           f"{len(t['ep']['block1_a_ids'])}x"
                           f"{len(t['ep']['block2_b_ids'])}")
            recs_out.append({"Inputs": inputs,
                             "Generated Outputs": r["raw"][:800],
                             "Feedback": fb})
        return {comp: recs_out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default=os.path.join(
        HERE, os.pardir, "runs", "lambdafold", "amazon-google", "K15",
        "episodes.json"))
    ap.add_argument("--model", default="qwen3:8b")
    ap.add_argument("--budget", type=int, default=2000)
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.run_dir, exist_ok=True)

    data = harness.load_episodes(args.episodes)
    recs = harness3.build_record_fields(None)
    train = [e for e in data["episodes"] if e["split"] == "train"]
    val = [e for e in data["episodes"] if e["split"] == "val"]

    log = harness.EventLog(os.path.join(args.run_dir, "llm_events.jsonl"))
    client = harness.OllamaClient(args.model, log, think=False)

    def reflection_lm(prompt):
        out = claude_reflection(prompt)
        log.append({"ts": time.time(), "tag": "reflection-sonnet",
                    "model": "claude-sonnet-cli",
                    "messages": prompt if isinstance(prompt, str)
                    else prompt, "raw_response": out})
        return out

    seed_candidate = {"instruction": harness.SEED_INSTRUCTION,
                      "sampler_code": harness4.SEED_SAMPLER}
    with open(os.path.join(args.run_dir, "config.json"), "w") as f:
        json.dump({"arm": "pipeline-evolution", "task_model": args.model,
                   "reflection": "claude sonnet via CLI",
                   "budget": args.budget,
                   "score": "recall - 0.02*FP, cap 25",
                   "seed_candidate": seed_candidate}, f, indent=2)

    adapter = PipelineAdapter(recs, client, log)
    result = gepa.optimize(
        seed_candidate=seed_candidate,
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
    print("BEST SAMPLER:\n" + best["sampler_code"])


if __name__ == "__main__":
    main()
