"""Smoke (N train episodes) or frozen eval (val/test) of one instruction.

Usage:
  python smoke_and_eval.py --n 3                       # smoke on train
  python smoke_and_eval.py --split test --instruction-file best.txt
"""
import argparse
import json
import os
import statistics
import time

import harness

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default=os.path.join(
        HERE, os.pardir, "runs", "lambdafold", "amazon-google", "K15",
        "episodes.json"))
    ap.add_argument("--model", default="qwen3:8b")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=0, help="0 = whole split")
    ap.add_argument("--instruction-file", default=None)
    ap.add_argument("--tag", default="smoke")
    args = ap.parse_args()

    data = harness.load_episodes(args.episodes)
    eps = [e for e in data["episodes"] if e["split"] == args.split]
    if args.n:
        eps = eps[:args.n]
    instr = harness.SEED_INSTRUCTION
    if args.instruction_file:
        with open(args.instruction_file, encoding="utf-8") as f:
            instr = f.read()

    od = os.path.join(HERE, os.pardir, "runs", "evals",
                      time.strftime("%Y%m%d_%H%M%S") + "_" + args.tag)
    log = harness.EventLog(os.path.join(od, "llm_events.jsonl"))
    client = harness.OllamaClient(args.model, log, think=False)

    scores, rows = [], []
    t0 = time.time()
    for ep in eps:
        r = harness.run_episode(client, instr, ep, data,
                                tag=f"{args.tag}:{ep['episode_id']}")
        scores.append(r["score"])
        rows.append({"episode_id": ep["episode_id"], "score": r["score"],
                     "n_proposed": len(r["props"]), "n_hit": len(r["hits"]),
                     "n_targets": len(r["targets"]),
                     "parse_error": r["parse_error"]})
        print(f"{ep['episode_id']}: F1={r['score']:.3f} "
              f"hit={len(r['hits'])}/{len(r['targets'])} "
              f"proposed={len(r['props'])} err={r['parse_error']}",
              flush=True)
    tot_t = sum(x["n_targets"] for x in rows)
    tot_h = sum(x["n_hit"] for x in rows)
    tot_p = sum(x["n_proposed"] for x in rows)
    micro_p = tot_h / tot_p if tot_p else 0.0
    micro_r = tot_h / tot_t if tot_t else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)
                if micro_p + micro_r else 0.0)
    out = {"split": args.split, "n_episodes": len(eps), "model": args.model,
           "mean_F1": statistics.mean(scores) if scores else 0.0,
           "micro_precision": micro_p, "micro_recall": micro_r,
           "micro_F1": micro_f1,
           "targets_recovered": f"{tot_h}/{tot_t}",
           "runtime_s": round(time.time() - t0, 1),
           "instruction": instr, "rows": rows}
    with open(os.path.join(od, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("rows", "instruction")}, indent=2))


if __name__ == "__main__":
    main()
