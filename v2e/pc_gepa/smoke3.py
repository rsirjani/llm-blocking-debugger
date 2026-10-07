"""3-episode smoke of the rule-induction arm."""
import json
import os
import sys

import harness
import harness3
from gepa_run3 import load_assignments, load_gold

HERE = os.path.dirname(os.path.abspath(__file__))
n = int(sys.argv[1]) if len(sys.argv) > 1 else 3

data = harness.load_episodes(os.path.join(
    HERE, os.pardir, "runs", "lambdafold", "amazon-google", "K15",
    "episodes.json"))
recs = harness3.build_record_fields(None)
assignments = load_assignments(15)
gold = load_gold()
log = harness.EventLog(os.path.join(HERE, os.pardir, "runs", "evals",
                                    "smoke3", "llm_events.jsonl"))
client = harness.OllamaClient("qwen3:8b", log, think=False, num_ctx=24576)
with open(os.path.join(HERE, "seed_instruction3.txt"),
          encoding="utf-8") as f:
    instr = f.read()

eps = [e for e in data["episodes"] if e["split"] == "train"][:n]
for ep in eps:
    r = harness3.run_episode_v3(client, instr, harness3.SEED_SPEC3_TEXT,
                                ep, recs, assignments, gold,
                                tag=f"smoke3:{ep['episode_id']}")
    rep = r["reply"] or {}
    print(f"{ep['episode_id']}: score={r['score']:.3f} "
          f"hit={len(r['hits'])}/{len(r['targets'])} "
          f"added_nongold={r['added_nongold']} "
          f"remedy={rep.get('remedy')} mode={rep.get('failure_mode')} "
          f"notes={r['notes']} err={r['spec_error'] or r['parse_error']}",
          flush=True)
    if rep:
        print("  predicate:", json.dumps(rep.get("move_specification"))[:400],
              flush=True)
