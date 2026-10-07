"""Block-contents baseline (the dumb start as originally specified):
prompt = ALL records of block X as one numbered list, ALL records of
block Y as another, plus the seed. Judge returns index PAIRS across the
two lists: {"matches": [[a_idx, b_idx], ...]}. O(|X|+|Y|) prompt cost
instead of the pair-list scaffold's O(|X|*|Y|).

Same physics as harness5: 20k-char budget (lists truncated evenly,
truncation reported), <=25 proposals, score = recall - 0.02*FP.

Usage: python records_baseline.py --split test [--datasets ...]
"""
import argparse
import json
import time
from collections import defaultdict

import harness
import harness5
from gepa_run5 import load_banks, split_items

PAIRS_SCHEMA = {
    "type": "object",
    "properties": {
        "matches": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "integer"},
                      "minItems": 2, "maxItems": 2},
        },
    },
    "required": ["matches"],
}

INSTRUCTION = (
    "You are debugging the output of a blocking system for entity "
    "resolution. Two blocks that wrongly separated at least one true "
    "match are shown IN FULL (possibly truncated at a size budget): "
    "list A holds block X's records from table A, list B holds block "
    "Y's records from table B. A confirmed true match that the blocker "
    "split across the two blocks is given as the SEED. Find OTHER true "
    "matches between list A and list B: pairs of records that refer to "
    "the same real-world entity. Return JSON {\"matches\": [[a_number, "
    "b_number], ...]} using the list numbers. Do not include the seed "
    "pair. If none, return an empty list.")


def render_records(bank, ep):
    recs = bank["records"]
    sa, sb = ep["seed_pair"]
    head = [
        "## Context",
        f"Dataset: {bank['dataset_metadata']}.",
        bank["method_metadata"],
        f"Block X: {len(ep['a_ids'])} table-A records. "
        f"Block Y: {len(ep['b_ids'])} table-B records.",
        "",
        "## SEED: confirmed true match wrongly split by the blocker",
        f"A: {harness5.fmt_fields(recs[sa])}",
        f"B: {harness5.fmt_fields(recs[sb])}",
        "",
    ]
    used = sum(len(x) + 1 for x in head) + 200
    budget = harness5.CHAR_BUDGET
    # split remaining budget between the two lists proportionally
    share_a = len(ep["a_ids"]) / (len(ep["a_ids"]) + len(ep["b_ids"]))
    lists, shown = {}, {}
    for side, ids, share in (("A", ep["a_ids"], share_a),
                             ("B", ep["b_ids"], 1 - share_a)):
        cap = (budget - used) * share
        lines, spent, kept = [], 0, []
        for n, rid in enumerate(ids, 1):
            entry = f"{n}. {harness5.fmt_fields(recs[rid])}"
            if spent + len(entry) + 1 > cap:
                break
            lines.append(entry)
            spent += len(entry) + 1
            kept.append(rid)
        lists[side] = lines
        shown[side] = kept
    trunc = (len(shown["A"]) < len(ep["a_ids"])
             or len(shown["B"]) < len(ep["b_ids"]))
    body = (head
            + [f"## List A — block X records ({len(shown['A'])} of "
               f"{len(ep['a_ids'])} shown)"] + lists["A"] + [""]
            + [f"## List B — block Y records ({len(shown['B'])} of "
               f"{len(ep['b_ids'])} shown)"] + lists["B"]
            + ["", 'Return JSON {"matches": [[a_number, b_number], ...]}'])
    return "\n".join(body), shown["A"], shown["B"], trunc


def run_episode_records(client, bank, ep, tag=""):
    user, ids_a, ids_b, trunc = render_records(bank, ep)
    targets = {tuple(t) for t in ep["targets"]}
    raw = client.chat([{"role": "system", "content": INSTRUCTION},
                       {"role": "user", "content": user}],
                      fmt=PAIRS_SCHEMA, tag=tag)
    try:
        matches = json.loads(raw).get("matches", [])
        perr = None
    except Exception as e:  # noqa: BLE001
        matches, perr = [], f"{type(e).__name__}: {e}"
    seed = tuple(ep["seed_pair"])
    props = set()
    for m in matches[:harness5.MAX_PROPS]:
        if (isinstance(m, list) and len(m) == 2
                and isinstance(m[0], int) and isinstance(m[1], int)
                and 1 <= m[0] <= len(ids_a) and 1 <= m[1] <= len(ids_b)):
            p = (ids_a[m[0] - 1], ids_b[m[1] - 1])
            if p != seed:
                props.add(p)
    hits = props & targets
    sa, sb = set(ids_a), set(ids_b)
    cov = sum(1 for a, b in targets if a in sa and b in sb)
    recall = len(hits) / len(targets) if targets else 0.0
    score = max(0.0, recall - harness5.FP_PENALTY * len(props - hits))
    return {"score": score, "hits": hits, "targets": targets,
            "fp": len(props - hits), "coverage": cov,
            "truncated": trunc, "parse_error": perr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", default="qwen3:8b")
    ap.add_argument("--out", default="../runs/gepa/run5_general_sonnet/"
                                     "eval_test_recordsbaseline.json")
    args = ap.parse_args()
    banks = load_banks(["lambdafold"])
    log = harness.EventLog("../runs/evals/records_baseline/"
                           "llm_events.jsonl")
    client = harness.OllamaClient(args.model, log, think=False,
                                  num_ctx=16384)
    per = defaultdict(lambda: [0, 0, 0, 0])
    t0 = time.time()
    for item in split_items(banks, args.split):
        bank = banks[tuple(item["bank_key"])]
        ep = item["ep"]
        r = run_episode_records(client, bank, ep,
                                tag=f"recbase:{ep['episode_id']}")
        per[bank["dataset"]][0] += len(r["hits"])
        per[bank["dataset"]][1] += len(r["targets"])
        per[bank["dataset"]][2] += r["fp"]
        per[bank["dataset"]][3] += r["coverage"]
        print(f"{ep['episode_id']}: {len(r['hits'])}/{len(r['targets'])}"
              f" fp={r['fp']} cov={r['coverage']} trunc={r['truncated']}",
              flush=True)
    th = sum(v[0] for v in per.values())
    tt = sum(v[1] for v in per.values())
    out = {"arm": "records-baseline (block contents, as specified)",
           "split": args.split,
           "micro_recall": th / tt if tt else 0.0,
           "targets_recovered": f"{th}/{tt}",
           "false_positives": sum(v[2] for v in per.values()),
           "coverage": f"{sum(v[3] for v in per.values())}/{tt}",
           "per_dataset": {k: {"recovered": f"{v[0]}/{v[1]}", "fp": v[2],
                               "coverage": v[3]}
                           for k, v in sorted(per.items())},
           "runtime_s": round(time.time() - t0, 1)}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: v for k, v in out.items()
                      if k != "per_dataset"}, indent=2))


if __name__ == "__main__":
    main()
