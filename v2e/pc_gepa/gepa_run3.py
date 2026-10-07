"""GEPA run3: rule-induction arm. Candidate = {instruction (monolithic
prompt template incl. {SEED_BLOCK}/{SAMPLES} placeholders), sampling_spec}.
Score = target recall via executed predicates − 0.0005 × added non-gold
pairs. Also usable as evaluator: --eval-split test --no-optimize.
"""
import argparse
import csv
import json
import os
import time

import harness
import harness3

HERE = os.path.dirname(os.path.abspath(__file__))


def load_assignments(K):
    path = os.path.join(HERE, os.pardir, "runs", "lambdafold",
                        "amazon-google", f"K{K}", "assignments.csv")
    out = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["record_id"]] = row["block"]
    return out


def load_gold():
    import sys
    sys.path.insert(0, os.path.join(HERE, os.pardir, "src"))
    import loaders
    _, _, gold, _ = loaders.load_amazon_google()
    return {(str(a), str(b)) for a, b in gold}


class RuleAdapter:
    propose_new_texts = None  # required by GEPA's proposer protocol

    def __init__(self, eps_data, recs, assignments, gold, client):
        self.data = eps_data
        self.recs = recs
        self.assign = assignments
        self.gold = gold
        self.client = client

    def evaluate(self, batch, candidate, capture_traces=False):
        from gepa.core.adapter import EvaluationBatch
        outputs, scores, trajs = [], [], ([] if capture_traces else None)
        for ep in batch:
            r = harness3.run_episode_v3(
                self.client, candidate["instruction"],
                candidate["sampling_spec"], ep, self.recs, self.assign,
                self.gold, tag=f"gepa3:{ep['episode_id']}")
            outputs.append({"raw": r["raw"]})
            scores.append(r["score"])
            if trajs is not None:
                trajs.append({"ep": ep, "raw": r["raw"],
                              "feedback": harness3.feedback_v3(ep, r),
                              "user_message": r["user_message"],
                              "spec_text": candidate["sampling_spec"]})
        return EvaluationBatch(outputs=outputs, scores=scores,
                               trajectories=trajs)

    def make_reflective_dataset(self, candidate, eval_batch,
                                components_to_update):
        comp = components_to_update[0]
        recs = []
        for t in eval_batch.trajectories:
            fb = t["feedback"]
            if comp == "sampling_spec":
                inputs = "Current spec:\n" + t["spec_text"]
                fb += (" || spec DSL keys: n_sample_cap (10..80), "
                       "proportional (bool), show_fields (sublist of "
                       "title/manufacturer/price/description, title "
                       "required), max_desc (0..300). Pure JSON only.")
            else:
                inputs = t["user_message"][:6000]
                fb += (" || instruction must keep the literal "
                       "{SEED_BLOCK} and {SAMPLES} placeholders.")
            recs.append({"Inputs": inputs,
                         "Generated Outputs": t["raw"][:1500],
                         "Feedback": fb})
        return {comp: recs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default=os.path.join(
        HERE, os.pardir, "runs", "lambdafold", "amazon-google", "K15",
        "episodes.json"))
    ap.add_argument("--K", type=int, default=15)
    ap.add_argument("--model", default="qwen3:8b")
    ap.add_argument("--budget", type=int, default=800)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--eval-split", default=None,
                    help="skip optimization; evaluate on this split")
    ap.add_argument("--instruction-file", default=None)
    ap.add_argument("--spec-file", default=None)
    args = ap.parse_args()
    os.makedirs(args.run_dir, exist_ok=True)

    data = harness.load_episodes(args.episodes)
    recs = harness3.build_record_fields(None)
    assignments = load_assignments(args.K)
    gold = load_gold()
    log = harness.EventLog(os.path.join(args.run_dir, "llm_events.jsonl"))
    task_client = harness.OllamaClient(args.model, log, think=False,
                                       num_ctx=24576)

    with open(args.instruction_file
              or os.path.join(HERE, "seed_instruction3.txt"),
              encoding="utf-8") as f:
        instr = f.read()
    spec_text = harness3.SEED_SPEC3_TEXT
    if args.spec_file:
        with open(args.spec_file, encoding="utf-8") as f:
            spec_text = f.read()

    if args.eval_split:
        eps = [e for e in data["episodes"] if e["split"] == args.eval_split]
        rows, t0 = [], time.time()
        for ep in eps:
            r = harness3.run_episode_v3(task_client, instr, spec_text, ep,
                                        recs, assignments, gold,
                                        tag=f"eval3:{ep['episode_id']}")
            rows.append({"episode_id": ep["episode_id"],
                         "score": r["score"], "n_hit": len(r["hits"]),
                         "n_targets": len(r["targets"]),
                         "added_nongold": r["added_nongold"],
                         "remedy": (r["reply"] or {}).get("remedy"),
                         "failure_mode": (r["reply"] or {}).get(
                             "failure_mode"),
                         "notes": r["notes"],
                         "spec_error": r["spec_error"],
                         "parse_error": r["parse_error"]})
            print(f"{ep['episode_id']}: score={r['score']:.3f} "
                  f"hit={len(r['hits'])}/{len(r['targets'])} "
                  f"added={r['added_nongold']} "
                  f"remedy={(r['reply'] or {}).get('remedy')}", flush=True)
        th = sum(x["n_hit"] for x in rows)
        tt = sum(x["n_targets"] for x in rows)
        ta = sum(x["added_nongold"] for x in rows)
        out = {"split": args.eval_split, "n_episodes": len(eps),
               "micro_recall_PCrecovery": th / tt if tt else 0,
               "targets_recovered": f"{th}/{tt}",
               "added_nongold_pairs": ta,
               "PC_points_recovered": round(100.0 * th / 1300, 2),
               "mean_score": sum(x["score"] for x in rows) / len(rows),
               "runtime_s": round(time.time() - t0, 1), "rows": rows}
        with open(os.path.join(args.run_dir,
                               f"eval_{args.eval_split}.json"), "w") as f:
            json.dump(out, f, indent=2)
        print(json.dumps({k: v for k, v in out.items() if k != "rows"},
                         indent=2))
        return

    import gepa
    train = [e for e in data["episodes"] if e["split"] == "train"]
    val = [e for e in data["episodes"] if e["split"] == "val"]
    refl_client = harness.OllamaClient(args.model, log, think=True,
                                       temperature=0.7, max_tokens=8192)

    def reflection_lm(prompt):
        msgs = ([{"role": "user", "content": prompt}]
                if isinstance(prompt, str) else prompt)
        return refl_client.chat(msgs, tag="reflection")

    adapter = RuleAdapter(data, recs, assignments, gold, task_client)
    with open(os.path.join(args.run_dir, "config.json"), "w") as f:
        json.dump({"arm": "rule-induction", "model": args.model,
                   "budget": args.budget, "K": args.K,
                   "score": "recall - 0.0005*added_nongold",
                   "seed_spec": spec_text}, f, indent=2)
    result = gepa.optimize(
        seed_candidate={"instruction": instr, "sampling_spec": spec_text},
        trainset=train, valset=val, adapter=adapter,
        reflection_lm=reflection_lm, reflection_minibatch_size=3,
        module_selector="round_robin", max_metric_calls=args.budget,
        run_dir=args.run_dir, seed=0, display_progress_bar=False,
        raise_on_exception=False)
    best = result.best_candidate
    for name, fn in (("best_instruction.txt", "instruction"),
                     ("best_spec.json", "sampling_spec")):
        with open(os.path.join(args.run_dir, name), "w",
                  encoding="utf-8") as f:
            f.write(best[fn])
    print("FINISHED", result.total_metric_calls, "calls,",
          result.num_candidates, "candidates")


if __name__ == "__main__":
    main()
