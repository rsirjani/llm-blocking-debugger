"""GEPA run2: co-evolve {instruction, sampling_spec}.

The sampling_spec JSON controls EVERYTHING the model is shown (sampler
choice incl. size-conditional rules, text truncation, sim display,
metadata blocks, char budget). GEPA mutates it as text, round-robin
with the instruction. Score = target recall (per-episode PC recovery)
- 0.02 per false positive, so richer prompts must earn their tokens
via coverage and judgment, not bulk.
"""
import argparse
import json
import os
import time

import gepa
from gepa.core.adapter import EvaluationBatch, GEPAAdapter

import harness
import harness2

HERE = os.path.dirname(os.path.abspath(__file__))

DSL_DOC = (
    "sampling_spec DSL — the ONLY valid keys: default_sampler, rules, "
    "max_text (40..400), show_sim, show_method_metadata, "
    "show_block_stats, show_seed, char_budget (4000..60000). Samplers: "
    '{"name":"full"} | {"name":"global_topm","m":1..120} | '
    '{"name":"per_record_topk","k":1..5} | '
    '{"name":"hybrid","m":..,"k":..}. rules: up to 4 of '
    '{"if_max_side_le": N, "sampler": {...}} — first match (by '
    "max(|A-side|,|B-side|)) overrides default_sampler. Output MUST be "
    "pure JSON, no comments. Tradeoff to optimize: coverage (true pairs "
    "present in the shown list — reported in feedback) vs list length "
    "(long lists dilute the judge and can hit char_budget truncation).")


class BlockDebugAdapterV2(GEPAAdapter):
    def __init__(self, data, client):
        self.data = data
        self.client = client

    def evaluate(self, batch, candidate, capture_traces=False):
        instr = candidate["instruction"]
        spec_text = candidate["sampling_spec"]
        outputs, scores, trajs = [], [], ([] if capture_traces else None)
        for ep in batch:
            r = harness2.run_episode_v2(self.client, instr, spec_text, ep,
                                        self.data,
                                        tag=f"gepa2:{ep['episode_id']}")
            outputs.append({"raw": r["raw"], "indices": r["indices"],
                            "info": r["info"]})
            scores.append(r["score"])
            if trajs is not None:
                fb = harness2.feedback_v2(
                    ep, self.data, r["props"], r["hits"], r["targets"],
                    r["shown"], r["info"], r["spec_error"],
                    r["parse_error"])
                trajs.append({"ep": ep, "raw": r["raw"], "feedback": fb,
                              "user_message": r["user_message"],
                              "spec_text": spec_text})
        return EvaluationBatch(outputs=outputs, scores=scores,
                               trajectories=trajs)

    def make_reflective_dataset(self, candidate, eval_batch,
                                components_to_update):
        comp = components_to_update[0]
        recs = []
        for t in eval_batch.trajectories:
            fb = t["feedback"]
            if comp == "sampling_spec":
                inputs = ("Episode block sizes: "
                          f"{len(t['ep']['block1_a_ids'])} A-side x "
                          f"{len(t['ep']['block2_b_ids'])} B-side. "
                          "Current spec:\n" + t["spec_text"])
                fb = fb + " || " + DSL_DOC
            else:
                inputs = t["user_message"][:6000]
            recs.append({"Inputs": inputs,
                         "Generated Outputs": t["raw"][:1500],
                         "Feedback": fb})
        return {comp: recs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default=os.path.join(
        HERE, os.pardir, "runs", "lambdafold", "amazon-google", "K15",
        "episodes.json"))
    ap.add_argument("--model", default="qwen3:8b")
    ap.add_argument("--budget", type=int, default=800)
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.run_dir, exist_ok=True)

    data = harness.load_episodes(args.episodes)
    eps = data["episodes"]
    train = [e for e in eps if e["split"] == "train"]
    val = [e for e in eps if e["split"] == "val"]

    log = harness.EventLog(os.path.join(args.run_dir, "llm_events.jsonl"))
    task_client = harness.OllamaClient(args.model, log, think=False)
    refl_client = harness.OllamaClient(args.model, log, think=True,
                                       temperature=0.7, max_tokens=8192)

    def reflection_lm(prompt):
        msgs = ([{"role": "user", "content": prompt}]
                if isinstance(prompt, str) else prompt)
        return refl_client.chat(msgs, tag="reflection")

    seed_candidate = {"instruction": harness.SEED_INSTRUCTION,
                      "sampling_spec": harness2.SEED_SPEC_TEXT}
    with open(os.path.join(args.run_dir, "config.json"), "w") as f:
        json.dump({"episodes_file": args.episodes, "model": args.model,
                   "budget": args.budget, "score":
                   "recall - 0.02*FP, cap 25 proposals",
                   "components": list(seed_candidate),
                   "seed_candidate": seed_candidate,
                   "started": time.strftime("%Y-%m-%d %H:%M:%S")},
                  f, indent=2)

    adapter = BlockDebugAdapterV2(data, task_client)
    result = gepa.optimize(
        seed_candidate=seed_candidate,
        trainset=train, valset=val, adapter=adapter,
        reflection_lm=reflection_lm,
        reflection_minibatch_size=3,
        module_selector="round_robin",
        max_metric_calls=args.budget,
        run_dir=args.run_dir, seed=0, display_progress_bar=False,
        raise_on_exception=False,
    )
    best = result.best_candidate
    with open(os.path.join(args.run_dir, "best_instruction.txt"), "w",
              encoding="utf-8") as f:
        f.write(best["instruction"])
    with open(os.path.join(args.run_dir, "best_spec.json"), "w",
              encoding="utf-8") as f:
        f.write(best["sampling_spec"])
    print("FINISHED", result.total_metric_calls, "metric calls,",
          result.num_candidates, "candidates")
    print("BEST SPEC:\n" + best["sampling_spec"])
    print("BEST INSTRUCTION:\n" + best["instruction"])


if __name__ == "__main__":
    main()
