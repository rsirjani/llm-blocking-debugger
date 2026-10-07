"""Frozen eval of a {instruction, sampling_spec} candidate on a split.

Reports recall (PC recovery), precision, F1, coverage ceiling of the
spec, and PC-points recovered if applied to the whole dataset split.
"""
import argparse
import json
import os
import time

import harness
import harness2

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default=os.path.join(
        HERE, os.pardir, "runs", "lambdafold", "amazon-google", "K15",
        "episodes.json"))
    ap.add_argument("--model", default="qwen3:8b")
    ap.add_argument("--split", default="test")
    ap.add_argument("--instruction-file", default=None)
    ap.add_argument("--spec-file", default=None)
    ap.add_argument("--tag", default="eval2")
    args = ap.parse_args()

    data = harness.load_episodes(args.episodes)
    eps = [e for e in data["episodes"] if e["split"] == args.split]
    instr = harness.SEED_INSTRUCTION
    spec_text = harness2.SEED_SPEC_TEXT
    if args.instruction_file:
        with open(args.instruction_file, encoding="utf-8") as f:
            instr = f.read()
    if args.spec_file:
        with open(args.spec_file, encoding="utf-8") as f:
            spec_text = f.read()

    od = os.path.join(HERE, os.pardir, "runs", "evals",
                      time.strftime("%Y%m%d_%H%M%S") + "_" + args.tag)
    log = harness.EventLog(os.path.join(od, "llm_events.jsonl"))
    client = harness.OllamaClient(args.model, log, think=False)

    rows = []
    t0 = time.time()
    for ep in eps:
        r = harness2.run_episode_v2(client, instr, spec_text, ep, data,
                                    tag=f"{args.tag}:{ep['episode_id']}")
        cov_hit, cov_tot = harness2.coverage(ep, r["shown"])
        rows.append({
            "episode_id": ep["episode_id"], "score": r["score"],
            "n_hit": len(r["hits"]), "n_fp": len(r["props"] - r["hits"]),
            "n_targets": len(r["targets"]), "coverage": cov_hit,
            "info": r["info"], "spec_error": r["spec_error"],
            "parse_error": r["parse_error"]})
        print(f"{ep['episode_id']}: score={r['score']:.3f} "
              f"hit={len(r['hits'])}/{len(r['targets'])} "
              f"fp={len(r['props'] - r['hits'])} cov={cov_hit}/{cov_tot} "
              f"{r['info'].get('sampler_used')}"
              f"@{r['info'].get('n_shown')}", flush=True)
    tot_t = sum(x["n_targets"] for x in rows)
    tot_h = sum(x["n_hit"] for x in rows)
    tot_fp = sum(x["n_fp"] for x in rows)
    tot_cov = sum(x["coverage"] for x in rows)
    prec = tot_h / (tot_h + tot_fp) if tot_h + tot_fp else 0.0
    rec = tot_h / tot_t if tot_t else 0.0
    n_gold_dataset = 1300
    out = {"split": args.split, "n_episodes": len(eps),
           "model": args.model,
           "micro_recall_PCrecovery": rec, "micro_precision": prec,
           "micro_F1": (2 * prec * rec / (prec + rec)
                        if prec + rec else 0.0),
           "targets_recovered": f"{tot_h}/{tot_t}",
           "false_positives": tot_fp,
           "coverage_ceiling": f"{tot_cov}/{tot_t}",
           "PC_points_recovered_this_split": round(
               100.0 * tot_h / n_gold_dataset, 2),
           "runtime_s": round(time.time() - t0, 1),
           "instruction": instr, "spec": spec_text, "rows": rows}
    with open(os.path.join(od, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("rows", "instruction", "spec")},
                     indent=2))


if __name__ == "__main__":
    main()
