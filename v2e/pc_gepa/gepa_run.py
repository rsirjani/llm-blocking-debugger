"""GEPA optimization of the block-debug instruction.

Task LLM: qwen3:8b via local Ollama, think=False, temperature 0,
structured-output JSON. Reflection LLM: qwen3:8b, think=True.
Gold-derived feedback flows ONLY into reflection (train split).
Test split is never touched here — see eval_test.py.
"""
import argparse
import json
import os
import time

import gepa
from gepa.core.adapter import EvaluationBatch, GEPAAdapter

import harness

HERE = os.path.dirname(os.path.abspath(__file__))


class BlockDebugAdapter(GEPAAdapter):
    def __init__(self, data, client):
        self.data = data
        self.client = client

    def evaluate(self, batch, candidate, capture_traces=False):
        instruction = candidate["instruction"]
        outputs, scores, trajs = [], [], ([] if capture_traces else None)
        for ep in batch:
            r = harness.run_episode(self.client, instruction, ep,
                                    self.data, tag=f"gepa:{ep['episode_id']}")
            outputs.append({"raw": r["raw"], "indices": r["indices"]})
            scores.append(r["score"])
            if trajs is not None:
                fb = harness.feedback_text(ep, self.data, r["props"],
                                           r["hits"], r["targets"],
                                           r["parse_error"])
                trajs.append({"ep": ep, "raw": r["raw"], "feedback": fb,
                              "user_message": r["user_message"]})
        return EvaluationBatch(outputs=outputs, scores=scores,
                               trajectories=trajs)

    def make_reflective_dataset(self, candidate, eval_batch,
                                components_to_update):
        comp = components_to_update[0]
        recs = []
        for t in eval_batch.trajectories:
            recs.append({
                "Inputs": t["user_message"][:6000],
                "Generated Outputs": t["raw"][:2000],
                "Feedback": t["feedback"],
            })
        return {comp: recs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default=os.path.join(
        HERE, os.pardir, "runs", "lambdafold", "amazon-google", "K15",
        "episodes.json"))
    ap.add_argument("--model", default="qwen3:8b")
    ap.add_argument("--budget", type=int, default=600)
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()

    run_dir = args.run_dir or os.path.join(
        HERE, os.pardir, "runs", "gepa",
        time.strftime("%Y%m%d_%H%M%S") + "_" + args.model.replace(":", "_"))
    os.makedirs(run_dir, exist_ok=True)

    data = harness.load_episodes(args.episodes)
    eps = data["episodes"]
    train = [e for e in eps if e["split"] == "train"]
    val = [e for e in eps if e["split"] == "val"]

    log = harness.EventLog(os.path.join(run_dir, "llm_events.jsonl"))
    task_client = harness.OllamaClient(args.model, log, think=False)
    refl_client = harness.OllamaClient(args.model, log, think=True,
                                       temperature=0.7, max_tokens=8192)

    def reflection_lm(prompt):
        if isinstance(prompt, str):
            msgs = [{"role": "user", "content": prompt}]
        else:
            msgs = prompt
        return refl_client.chat(msgs, tag="reflection")

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({"episodes_file": args.episodes, "model": args.model,
                   "budget": args.budget, "n_train": len(train),
                   "n_val": len(val), "think_task": False,
                   "think_reflection": True, "temperature_task": 0,
                   "seed_instruction": harness.SEED_INSTRUCTION}, f, indent=2)

    adapter = BlockDebugAdapter(data, task_client)
    result = gepa.optimize(
        seed_candidate={"instruction": harness.SEED_INSTRUCTION},
        trainset=train, valset=val, adapter=adapter,
        reflection_lm=reflection_lm,
        reflection_minibatch_size=3,
        max_metric_calls=args.budget,
        run_dir=run_dir, seed=0, display_progress_bar=False,
        raise_on_exception=False,
    )
    best = result.best_candidate
    summary = {
        "best_score_val": result.best_score,
        "seed_score_val": result.val_aggregate_scores[0],
        "n_candidates": len(result.candidates),
        "total_metric_calls": result.total_metric_calls,
        "best_instruction": best["instruction"],
        "all_val_scores": result.val_aggregate_scores,
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items()
                      if k != "best_instruction"}, indent=2))
    print("BEST INSTRUCTION:\n" + best["instruction"])


if __name__ == "__main__":
    main()
