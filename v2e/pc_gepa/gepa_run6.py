"""GEPA run6: block-contents scaffold, evolvable FILTER, BLIND reflection.

Differences from run5 (user-directed):
- Judge sees blocks as record lists; filter_code selects/orders records
  per side; judge returns [a_number, b_number] pairs (harness6).
- Reflection is BLIND: it is never told the dataset or the blocker,
  and all record text it sees is content-masked (mask.ContentLexicon,
  placeholders consistent per reflective example). The judge still
  sees everything (test-time-legitimate info).
- Leakage gate: each Sonnet proposal is audited for dataset-content
  tokens (minus a whitelist drawn from our own seed texts); one masked
  re-ask on violation, then hard-strip by rejection feedback.
"""
import argparse
import json
import os
import subprocess
import time
from collections import defaultdict

import harness
import harness5
import harness6
import mask
from gepa_run5 import load_banks, split_items

HERE = os.path.dirname(os.path.abspath(__file__))

CODE_TEMPLATE = """You are evolving the FILTER STAGE of an entity-blocking
debug pipeline that must work across many unknown datasets and blocking
methods (you are deliberately not told which — record text in the
feedback is content-masked with <Tn> placeholders; rely on structure,
field roles and generic vocabulary only).

The filter is a pure Python function executed automatically. Its input
is the FULL contents of two blocks; it returns which records of each
side the judge LLM gets to see, in what order. The harness renders the
two lists until a shared 20,000-character budget runs out — records
past the cutoff are lost, and a true pair with an unshown record can
never be recovered.

Current filter source:
```
<curr_param>
```

Execution feedback from recent episodes:
<side_info>

Contract (MUST hold):
- define sample(recs_x, recs_y, seed_pair, sim, rng)
- recs_x: list of dicts, key "id" plus dataset-specific string fields
  (schemas vary; inspect keys, never hardcode a schema); recs_y same
- seed_pair: (a_id, b_id) confirmed split match; sim(a_id,b_id)->float
  char-3gram TF-IDF cosine; rng: seeded random.Random
- return (a_ids, b_ids) lists drawn from the given blocks. Available
  modules: re, math, itertools, Counter, defaultdict. No imports, no
  IO, 20s limit (blocks can be 400 x 15,000 records).

Write the improved COMPLETE function source inside a single ``` block."""

INSTR_TEMPLATE = """I provided an assistant with the instructions below to
find wrongly-separated matching record pairs between two lists of
records. The deployment spans many datasets and blocking methods you
are deliberately not told about; the example texts below are
content-masked with <Tn> placeholders (same placeholder = same token).
Write rules that work for ANY domain: about structure, shared rare
tokens, variant/edition markers, field roles — never about specific
names, brands, or domains.

Current instructions:
```
<curr_param>
```
Examples of the assistant's behavior with feedback:
<side_info>

The assistant must return JSON {"matches": [[a_number, b_number],
...]}. Provide the new instructions inside a single ``` block."""

SEED_INSTRUCTION6 = (
    "You are debugging the output of a blocking system for entity "
    "resolution. Two blocks that wrongly separated at least one true "
    "match are shown: list A holds one block's records from table A, "
    "list B holds the other block's records from table B. A confirmed "
    "true match split across the two blocks is given as the SEED. "
    "Find OTHER true matches between list A and list B: pairs "
    "referring to the same real-world entity. Return JSON "
    '{"matches": [[a_number, b_number], ...]} using the list numbers. '
    "Do not include the seed pair. If none, return an empty list.")


def claude_reflection_raw(prompt):
    for attempt in range(3):
        try:
            p = subprocess.run(
                ["claude", "-p", "--model", "sonnet"],
                input=prompt, capture_output=True, text=True,
                timeout=600)
            if p.returncode == 0 and p.stdout.strip():
                return p.stdout
        except subprocess.TimeoutExpired:
            pass
        time.sleep(5 * (attempt + 1))
    raise RuntimeError("claude CLI reflection failed 3x")


class BlindAdapter:
    propose_new_texts = None

    def __init__(self, banks, client, lex):
        self.banks = banks
        self.client = client
        self.lex = lex

    def _bank(self, item):
        return self.banks[tuple(item["bank_key"])]

    def evaluate(self, batch, candidate, capture_traces=False):
        from gepa.core.adapter import EvaluationBatch
        outputs, scores, trajs = [], [], ([] if capture_traces else None)
        for item in batch:
            bank, ep = self._bank(item), item["ep"]
            r = harness6.run_episode_v6(
                self.client, candidate["instruction"],
                candidate["filter_code"], bank, ep,
                tag=f"gepa6:{ep['episode_id']}")
            outputs.append({"raw": r["raw"], "coverage": r["coverage"]})
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
            mapping = {}
            fb = harness6.feedback_v6(bank, ep, r, comp, self.lex,
                                      mapping)
            if comp == "instruction" and r["user_message"]:
                # blind excerpt: drop the Context header (dataset +
                # blocker identity), mask the rest
                msg = r["user_message"]
                cut = msg.find("## SEED")
                blind = self.lex.mask_text(msg[cut:], mapping)
                inputs = ("[dataset and blocker hidden; record text "
                          "content-masked]\n" + blind[:4500])
            else:
                inputs = (f"episode with blocks {len(ep['a_ids'])}x"
                          f"{len(ep['b_ids'])} records "
                          "[dataset and blocker hidden]")
            out.append({"Inputs": inputs,
                        "Generated Outputs": r["raw"][:800],
                        "Feedback": fb})
        return {comp: out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blockers", nargs="+", default=["lambdafold"])
    ap.add_argument("--model", default="qwen3:8b")
    ap.add_argument("--budget", type=int, default=5000)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--eval-split", default=None)
    ap.add_argument("--instruction-file", default=None)
    ap.add_argument("--filter-file", default=None)
    args = ap.parse_args()
    os.makedirs(args.run_dir, exist_ok=True)

    banks = load_banks(args.blockers)
    lex = mask.ContentLexicon(banks)
    log = harness.EventLog(os.path.join(args.run_dir,
                                        "llm_events.jsonl"))
    client = harness.OllamaClient(args.model, log, think=False,
                                  num_ctx=16384)

    instr = SEED_INSTRUCTION6
    code = harness6.SEED_FILTER
    if args.instruction_file:
        with open(args.instruction_file, encoding="utf-8") as f:
            instr = f.read()
    if args.filter_file:
        with open(args.filter_file, encoding="utf-8") as f:
            code = f.read()

    if args.eval_split:
        per = defaultdict(lambda: [0, 0, 0, 0])
        t0 = time.time()
        for item in split_items(banks, args.eval_split):
            bank = banks[tuple(item["bank_key"])]
            ep = item["ep"]
            r = harness6.run_episode_v6(client, instr, code, bank, ep,
                                        tag=f"eval6:{args.eval_split}")
            per[bank["dataset"]][0] += len(r["hits"])
            per[bank["dataset"]][1] += len(r["targets"])
            per[bank["dataset"]][2] += len(r["props"] - r["hits"])
            per[bank["dataset"]][3] += r["coverage"]
            print(f"{ep['episode_id']}: "
                  f"{len(r['hits'])}/{len(r['targets'])} "
                  f"fp={len(r['props'] - r['hits'])} "
                  f"cov={r['coverage']}", flush=True)
        th = sum(v[0] for v in per.values())
        tt = sum(v[1] for v in per.values())
        tag = "best" if args.instruction_file else "seed"
        out = {"split": args.eval_split,
               "micro_recall": th / tt if tt else 0.0,
               "targets_recovered": f"{th}/{tt}",
               "false_positives": sum(v[2] for v in per.values()),
               "coverage": f"{sum(v[3] for v in per.values())}/{tt}",
               "per_dataset": {k: {"recovered": f"{v[0]}/{v[1]}",
                                   "fp": v[2], "coverage": v[3]}
                               for k, v in sorted(per.items())},
               "runtime_s": round(time.time() - t0, 1)}
        with open(os.path.join(args.run_dir,
                               f"eval_{args.eval_split}_{tag}.json"),
                  "w") as f:
            json.dump(out, f, indent=2)
        print(json.dumps({k: v for k, v in out.items()
                          if k != "per_dataset"}, indent=2))
        return

    import gepa
    train = split_items(banks, "train")
    val = split_items(banks, "val")

    whitelist = set(lex.audit_prompt(
        SEED_INSTRUCTION6 + " " + harness6.SEED_FILTER + " "
        + CODE_TEMPLATE + " " + INSTR_TEMPLATE))

    def reflection_lm(prompt):
        if not isinstance(prompt, str):
            prompt = "\n\n".join(m.get("content", "") for m in prompt)
        out = claude_reflection_raw(prompt)
        leaked = [t for t in lex.audit_prompt(out) if t not in whitelist]
        gate_note = None
        if leaked:
            gate_note = leaked[:20]
            out = claude_reflection_raw(
                prompt + "\n\nIMPORTANT: your previous draft contained "
                "dataset-specific content tokens that will not exist "
                f"at deployment: {leaked[:20]}. Rewrite the component "
                "WITHOUT any dataset-specific names or strings — "
                "structural, domain-generic rules only. Same output "
                "format, single ``` block.")
        log.append({"ts": time.time(), "tag": "reflection-sonnet",
                    "raw_response": out, "messages": prompt,
                    "leakage_gate_triggered": gate_note})
        return out

    with open(os.path.join(args.run_dir, "config.json"), "w") as f:
        json.dump({"arm": "run6-blind-blockcontents",
                   "blockers": args.blockers, "task_model": args.model,
                   "reflection": "claude sonnet CLI, BLIND + gated",
                   "budget": args.budget, "n_train": len(train),
                   "n_val": len(val),
                   "seed_candidate": {"instruction": instr,
                                      "filter_code": code}},
                  f, indent=2)
    adapter = BlindAdapter(banks, client, lex)
    result = gepa.optimize(
        seed_candidate={"instruction": instr, "filter_code": code},
        trainset=train, valset=val, adapter=adapter,
        reflection_lm=reflection_lm,
        reflection_minibatch_size=4,
        module_selector="round_robin",
        reflection_prompt_template={"instruction": INSTR_TEMPLATE,
                                    "filter_code": CODE_TEMPLATE},
        max_metric_calls=args.budget,
        run_dir=args.run_dir, seed=0, display_progress_bar=False,
        raise_on_exception=False)
    best = result.best_candidate
    for name, key in (("best_instruction.txt", "instruction"),
                      ("best_filter.py", "filter_code")):
        with open(os.path.join(args.run_dir, name), "w",
                  encoding="utf-8") as f:
            f.write(best[key])
    print("FINISHED", result.total_metric_calls, "calls,",
          result.num_candidates, "candidates")


if __name__ == "__main__":
    main()
