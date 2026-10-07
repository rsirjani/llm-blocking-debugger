"""run7 - corrected rerun of the run6 experiment.

Four defects in run6 are fixed here (all user-identified):

1. SYMMETRIC INSTANCES. R = R_A u R_B: clean-clean and dirty are treated
   identically. A block is a subset of R, a block pair is unordered
   {X, Y}, both blocks are shown WHOLE, and the targets are every gold
   pair the blocker separated between them, in either direction. run6
   used ordered pairs and showed only one source-side of each block,
   which made half the failures inexpressible and leaked the source
   split into the setup.

2. NOTHING THE JUDGE SEES IS HAND-WRITTEN. The evolved code returns the
   entire message: statistics, framing, seed presentation, formatting,
   closing instruction. The harness contributes no text at all. The
   dataset description and the blocker description are gone entirely --
   telling the judge which blocker produced the split is the same
   category of leak as showing the reviewer unmasked records.

3. EQUALIZED VALIDATION. A fixed number of instances per cell, rather
   than a proportion of each cell, so selection is not weighted by how
   many instances a dataset happens to contribute.

4. UNBIASED SEED CANDIDATE. The seed prompt states the task and the
   answer format; the seed code shows both blocks with basic counts.
   Neither hints at how to improve: no similarity primitive, no
   suggestion to rank, filter, prefilter or summarize, no mention of
   which record fields exist or that schemas vary. The reflection
   templates state the contract and the objective, not a strategy.

Fixed (the measurement instrument): the parse contract, the token
ceiling, the sandbox, and the score.

SECURITY: runs LLM-written Python through exec() in a restricted
namespace (whitelisted builtins, no imports, no file or network access,
SIGALRM timeout) - same posture as harness4/5/6.
"""
import argparse
import json
import os
import random as _random
import re
import signal
import subprocess
import time
import traceback
from collections import defaultdict

import bankio
import harness
import harness5
import mask

HERE = os.path.dirname(os.path.abspath(__file__))
BANK_ROOT = os.path.join(HERE, os.pardir, "runs", "bank2")
MAX_CTX = 32768
# Score = (hits - ALPHA * wrong) / |targets|. Both terms share the same
# denominator, so the precision a proposal must clear to be worth making
# is ALPHA/(1+ALPHA) regardless of how many pairs an instance holds; a
# fixed proposal cap would instead punish the instances with the most to
# find. Nothing here is visible to the pipeline: |targets| is used to
# grade, never to decide. There is no cap on proposals -- spamming is
# already priced by the penalty.
ALPHA = 0.2
# The penalty is floored so that one pathological instance cannot swamp
# a minibatch sum, and failing to produce a usable message sits below
# any answer that was actually produced.
WORST_ANSWER = -2.0
FAIL_SCORE = -3.0
CODE_TIMEOUT_S = 25

SEED_PROMPT = (
    "Two groups of records are shown, group X and group Y. One pair of "
    "records, one from each group, is known to describe the same "
    "real-world entity. Find the other pairs, one record from group X "
    "and one from group Y, that describe the same real-world entity. "
    'Answer with JSON of the form {"matches": [[x_number, y_number], '
    "...]} using the numbers shown beside the records. Do not include "
    'the pair that was given. If there are none, answer {"matches": '
    "[]}.")
SEED_CODE = '''def build(recs_x, recs_y, given_pair, rng):
    """Build the message shown to the judge model.

    recs_x: list of dicts, one per record of group X. Each dict has an
            "id" key; the remaining keys are that record's fields.
    recs_y: same, for group Y.
    given_pair: (x_id, y_id) -- a pair known to describe the same
            entity, one record from each group.
    rng: seeded random.Random.

    Return (text, x_ids, y_ids):
      text   -- the message the judge reads.
      x_ids  -- record ids in the order they are numbered in text, so
                answer index i refers to x_ids[i-1]; same for y_ids.
    The numbering in text must correspond to these lists. The message
    plus the system prompt must fit in the model's context window.
    """
    def fields(r):
        return " | ".join("%s: %s" % (k, v) for k, v in r.items()
                          if k != "id")

    by_id = {}
    for r in recs_x + recs_y:
        by_id[r["id"]] = r
    x_ids = [r["id"] for r in recs_x]
    y_ids = [r["id"] for r in recs_y]

    lines = []
    lines.append("Group X has %d records, group Y has %d records."
                 % (len(recs_x), len(recs_y)))
    gx, gy = given_pair
    if gx in by_id and gy in by_id:
        lines.append("Known same-entity pair:")
        lines.append("  X: " + fields(by_id[gx]))
        lines.append("  Y: " + fields(by_id[gy]))
    lines.append("")
    lines.append("Group X records:")
    for n, rid in enumerate(x_ids, 1):
        lines.append("%d. %s" % (n, fields(by_id[rid])))
    lines.append("")
    lines.append("Group Y records:")
    for n, rid in enumerate(y_ids, 1):
        lines.append("%d. %s" % (n, fields(by_id[rid])))
    return "\\n".join(lines), x_ids, y_ids
'''


_TOK = None


def token_count(text):
    """Exact count with the judge's own tokenizer; conservative
    fallback if it cannot be loaded (reported, never silent)."""
    global _TOK
    if _TOK is None:
        try:
            from transformers import AutoTokenizer
            _TOK = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
        except Exception as e:  # noqa: BLE001
            print(f"[warn] tokenizer unavailable ({e}); using chars/3.2",
                  flush=True)
            _TOK = False
    return int(len(text) / 3.2) if _TOK is False else len(_TOK.encode(text))


class _Timeout(Exception):
    pass


def _alarm(_sig, _frm):
    raise _Timeout(f"code exceeded {CODE_TIMEOUT_S}s")


def run_code(code, bank, inst):
    """Execute the evolved builder. Returns (text, x_ids, y_ids, err)."""
    recs = bank["records"]
    rx = [{"id": i, **recs[i]} for i in inst["a_ids"]]
    ry = [{"id": i, **recs[i]} for i in inst["b_ids"]]
    rng = _random.Random(f"build:{inst['episode_id']}")
    env = dict(harness5._SANDBOX_GLOBALS)
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(CODE_TIMEOUT_S)
    try:
        # sandboxed execution of LLM-written code, see module docstring
        exec(compile(code, "<builder>", "exec"), env)  # noqa: S102
        fn = env.get("build")
        if not callable(fn):
            return None, [], [], "the code must define build(...)"
        out = fn(rx, ry, tuple(inst["seed_pair"]), rng)
    except _Timeout as e:
        return None, [], [], str(e)
    except Exception:  # noqa: BLE001 -- traceback is the feedback
        return None, [], [], ("the code raised:\n"
                              + traceback.format_exc(limit=3))
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)

    if not isinstance(out, (list, tuple)) or len(out) != 3:
        return None, [], [], "build() must return (text, x_ids, y_ids)"
    text, x_ids, y_ids = out
    if not isinstance(text, str) or not text.strip():
        return None, [], [], "the first return value must be a non-empty str"
    vx, vy = set(inst["a_ids"]), set(inst["b_ids"])

    def clean(lst, valid, side):
        if not isinstance(lst, (list, tuple)):
            return None, f"{side}_ids must be a list"
        seen, kept = set(), []
        for i in lst:
            if i not in valid:
                return None, (f"{side}_ids contains an id that is not in "
                              f"group {side.upper()}")
            if i in seen:
                return None, f"{side}_ids contains a duplicate id"
            seen.add(i)
            kept.append(i)
        return kept, None

    x_ids, err = clean(x_ids, vx, "x")
    if err:
        return None, [], [], err
    y_ids, err = clean(y_ids, vy, "y")
    if err:
        return None, [], [], err
    return text, x_ids, y_ids, None


def parse_matches(reply):
    """Pull {"matches": [[x, y], ...]} out of a free-form reply."""
    if not reply or not reply.strip():
        return None, "the model returned nothing"
    cand = None
    for m in re.finditer(r"\{", reply):
        chunk = reply[m.start():]
        for end in range(len(chunk), 0, -1):
            if chunk[end - 1] != "}":
                continue
            try:
                obj = json.loads(chunk[:end])
            except Exception:  # noqa: BLE001
                continue
            if isinstance(obj, dict) and "matches" in obj:
                cand = obj
                break
        if cand is not None:
            break
    if cand is None:
        return None, ('no JSON object with a "matches" key was found '
                      f"(the reply began: {reply[:120]!r})")
    ms = cand.get("matches")
    if not isinstance(ms, list):
        return None, '"matches" must be a list of [x_number, y_number]'
    out = []
    for m in ms:
        if (isinstance(m, (list, tuple)) and len(m) == 2
                and all(isinstance(v, int) for v in m)):
            out.append((m[0], m[1]))
    return out, None


def run_instance(client, prompt, code, bank, inst, tag=""):
    targets = {tuple(t) for t in inst["targets"]}
    base = {"score": FAIL_SCORE, "targets": targets, "props": set(),
            "hits": set(), "coverage": 0, "n_tokens": 0, "raw": "",
            "code_error": None, "parse_error": None, "text": "",
            "x_ids": [], "y_ids": []}
    text, x_ids, y_ids, cerr = run_code(code, bank, inst)
    if cerr:
        return {**base, "code_error": cerr}
    ntok = token_count(text) + token_count(prompt)
    if ntok > MAX_CTX:
        return {**base, "n_tokens": ntok, "text": text, "x_ids": x_ids,
                "y_ids": y_ids,
                "code_error": (f"the message came to {ntok} tokens, over "
                               f"the {MAX_CTX} token limit, so nothing "
                               "was sent to the model")}
    raw = client.chat([{"role": "system", "content": prompt},
                       {"role": "user", "content": text}], tag=tag)
    ms, perr = parse_matches(raw)
    if perr:
        return {**base, "raw": raw, "n_tokens": ntok, "text": text,
                "x_ids": x_ids, "y_ids": y_ids, "parse_error": perr}
    given = tuple(inst["seed_pair"])
    props = set()
    for x, y in ms:
        if 1 <= x <= len(x_ids) and 1 <= y <= len(y_ids):
            p = (x_ids[x - 1], y_ids[y - 1])
            if p != given:
                props.add(p)
    hits = props & targets
    sx, sy = set(x_ids), set(y_ids)
    cov = sum(1 for a, b in targets if a in sx and b in sy)
    n_t = len(targets)
    recall = len(hits) / n_t if n_t else 0.0
    penalty = ALPHA * len(props - hits) / (n_t if n_t else 1)
    score = max(WORST_ANSWER, recall - penalty)
    return {"score": score, "targets": targets, "props": props,
            "hits": hits, "coverage": cov, "n_tokens": ntok, "raw": raw,
            "code_error": None, "parse_error": None, "text": text,
            "x_ids": x_ids, "y_ids": y_ids}


def feedback(bank, inst, r, component, lex, mapping):
    recs = bank["records"]
    tgt = r["targets"]

    def m(rid):
        return lex.mask_text(
            " ".join(str(v) for v in recs[rid].values())[:80], mapping)

    if r["code_error"]:
        return ("this instance scored 0. " + r["code_error"]
                + f" The two groups hold {len(inst['a_ids'])} and "
                f"{len(inst['b_ids'])} records.")
    if r["parse_error"]:
        return ("this instance scored 0: the reply could not be read. "
                + r["parse_error"] + ' The reply must contain {"matches":'
                " [[x_number, y_number], ...]}.")
    parts = [
        f"score={r['score']:.2f}: {len(r['hits'])} of {len(tgt)} pairs "
        f"found, {len(r['props'] - r['hits'])} proposals were wrong. The "
        f"message was {r['n_tokens']} tokens (limit {MAX_CTX}) and "
        f"contained {len(r['x_ids'])} of {len(inst['a_ids'])} group-X "
        f"records and {len(r['y_ids'])} of {len(inst['b_ids'])} group-Y "
        f"records, so {r['coverage']}/{len(tgt)} of the pairs to be "
        "found had both of their records present in it."]
    if component == "code":
        missing = [t for t in sorted(tgt)
                   if not (t[0] in set(r["x_ids"])
                           and t[1] in set(r["y_ids"]))]
        if missing:
            parts.append("Pairs whose records were not both in the "
                         "message: " + "; ".join(
                             f"'{m(a)}' with '{m(b)}'"
                             for a, b in missing[:3]))
    else:
        fp = sorted(r["props"] - r["hits"])[:3]
        shown = {(a, b) for a in r["x_ids"] for b in r["y_ids"]}
        fn = sorted((tgt & shown) - r["hits"])[:3]
        if fp:
            parts.append("Proposed but not the same entity: " + "; ".join(
                f"'{m(a)}' with '{m(b)}'" for a, b in fp))
        if fn:
            parts.append("Same entity but not proposed, both records "
                         "were in the message: " + "; ".join(
                             f"'{m(a)}' with '{m(b)}'" for a, b in fn))
    return " ".join(parts)


CODE_TEMPLATE = """The Python function below is given two groups of
records and must return the message that a small language model reads,
together with the mapping from the numbers used in that message to the
record ids. The model's answers are scored against pairs of records,
one from each group, that describe the same entity; its answer can only
refer to records the message contains.

You may change the function however you like within the contract below.
Nothing about the current version is privileged.

You are not told what these records are, where they come from, or how
they came to be grouped, and the record text in the feedback is
content-masked (<Tn> placeholders; the same placeholder means the same
original token).

Current source:
```
<curr_param>
```

Execution feedback from recent instances:
<side_info>

Contract:
- define build(recs_x, recs_y, given_pair, rng)
- recs_x / recs_y: lists of dicts, each with an "id" key plus that
  record's own fields
- given_pair: (x_id, y_id), a pair known to describe the same entity;
  rng: seeded random.Random
- return (text, x_ids, y_ids); the numbering inside text must
  correspond to those id lists
- available: re, math, itertools, Counter, defaultdict. No imports, no
  file or network access, 25 second limit

How an instance is scored: the fraction of the same-entity pairs that
the model finds, minus 0.2 for every pair it proposes that is not one,
both divided by how many same-entity pairs the instance holds. A pair
can only be found if both of its records are in the message. If the
message plus the system prompt exceeds 32768 tokens, or the code fails,
nothing is sent to the model and the instance scores -3.0, which is
worse than any answer it could have given.

Return the complete function inside a single ``` block."""

PROMPT_TEMPLATE = """The instructions below are given to a small
language model that is shown two groups of records and must find pairs,
one record from each group, that describe the same entity.

You are not told what these records are or where they come from, and
the record text below is content-masked (<Tn> placeholders; the same
placeholder means the same original token). Do not write instructions
that mention particular names, brands or subject areas: they will not
exist in the data this is deployed on.

Current instructions:
```
<curr_param>
```

Examples of the model's behaviour with feedback:
<side_info>

The reply is read by a program that looks for
{"matches": [[x_number, y_number], ...]} in the model's output. An
instance scores the fraction of the same-entity pairs the model finds,
minus 0.2 for every proposed pair that is not one, both divided by how
many same-entity pairs the instance holds; a reply that cannot be read
at all scores -3.0, which is worse than any readable answer.

Return the complete instructions inside a single ``` block."""


def claude_reflection(prompt):
    for attempt in range(3):
        try:
            p = subprocess.run(["claude", "-p", "--model", "sonnet"],
                               input=prompt, capture_output=True,
                               text=True, timeout=900)
            if p.returncode == 0 and p.stdout.strip():
                return p.stdout
        except subprocess.TimeoutExpired:
            pass
        time.sleep(5 * (attempt + 1))
    raise RuntimeError("claude CLI reflection failed 3x")


def load_cells(blockers, datasets):
    cells = {}
    for blk in blockers:
        for ds in datasets:
            path = os.path.join(BANK_ROOT, blk, f"{ds}.json")
            if os.path.exists(path):
                cells[(blk, ds)] = json.load(open(path))
    return cells


def build_splits(cells, train_ds, test_ds, val_per_cell, draw=0,
                 min_val=3):
    """Instances per split. Validation takes a FIXED number per cell
    (equalized), the rest of that cell's instances go to training."""
    train, val, test, notes = [], [], [], []
    for (blk, ds), bank in sorted(cells.items()):
        inst = bankio.derive(bank, "per_pair", draw=draw, min_case=3)
        for i in inst:
            i["cell"] = [blk, ds]
        if ds in test_ds:
            test += inst
            continue
        if ds not in train_ds:
            continue
        rng = _random.Random(f"split:{blk}:{ds}:{draw}")
        rng.shuffle(inst)
        # equalized: exactly val_per_cell from every cell that can spare
        # them, so selection is not weighted by cell size
        n_val = (val_per_cell if len(inst) >= 3 * val_per_cell
                 else max(1, len(inst) // 4))
        if n_val < min_val:
            notes.append(f"{blk}/{ds}: only {len(inst)} instances, "
                         f"contributes {n_val} to validation")
        val += inst[:n_val]
        train += inst[n_val:]
    return train, val, test, notes


class Adapter:
    propose_new_texts = None

    def __init__(self, cells, client, lex, reseed_ids=(), n_train=1):
        self.cells = cells
        self.client = client
        self.lex = lex
        # training instances redraw which failure is revealed once per
        # epoch: the same block pair teaches from a different seed each
        # pass, so neither component can be tuned to one draw. The epoch
        # advances only on minibatch evaluations (capture_traces=True),
        # so the pre- and post-mutation runs of an iteration always use
        # identical seeds and the acceptance test stays paired.
        self.reseed_ids = set(reseed_ids)
        self.n_train = max(1, n_train)
        self.seen = 0
        self.epoch = 0

    def _reseed(self, inst):
        if inst["episode_id"] not in self.reseed_ids:
            return inst
        failed = [tuple(f) for f in inst.get("failed", [])]
        if len(failed) < 2:
            return inst
        rng = _random.Random(f"seed:{inst['episode_id']}:{self.epoch}")
        given = failed[rng.randrange(len(failed))]
        return {**inst, "seed_pair": list(given),
                "targets": [list(f) for f in failed if f != given]}

    def evaluate(self, batch, candidate, capture_traces=False):
        from gepa.core.adapter import EvaluationBatch
        outputs, scores, trajs = [], [], ([] if capture_traces else None)
        if capture_traces:
            self.seen += len(batch)
            self.epoch = self.seen // self.n_train
        batch = [self._reseed(i) for i in batch]
        for inst in batch:
            bank = self.cells[tuple(inst["cell"])]
            r = run_instance(self.client, candidate["prompt"],
                             candidate["code"], bank, inst,
                             tag=f"run7:{inst['episode_id']}")
            outputs.append({"raw": r["raw"], "tokens": r["n_tokens"],
                            "coverage": r["coverage"]})
            scores.append(r["score"])
            if trajs is not None:
                trajs.append({"inst": inst, "r": r})
        return EvaluationBatch(outputs=outputs, scores=scores,
                               trajectories=trajs)

    def make_reflective_dataset(self, candidate, eval_batch,
                                components_to_update):
        comp = components_to_update[0]
        out = []
        for t in eval_batch.trajectories:
            inst = t["inst"]
            bank = self.cells[tuple(inst["cell"])]
            r = t["r"]
            mapping = {}
            fb = feedback(bank, inst, r, comp, self.lex, mapping)
            if comp == "prompt":
                body = (self.lex.mask_text(r["text"][:4500], mapping)
                        if r["text"] else "(no message was produced)")
                inputs = "[record text content-masked]\n" + body
            else:
                inputs = (f"groups of {len(inst['a_ids'])} and "
                          f"{len(inst['b_ids'])} records")
            out.append({"Inputs": inputs,
                        "Generated Outputs": (r["raw"][:800]
                                              or "(no reply)"),
                        "Feedback": fb})
        return {comp: out}


def evaluate_split(client, cells, instances, prompt, code, path, draws=1):
    """Census over the given instances, repeated over `draws`
    independent seed draws; reports mean and spread per cell."""
    per_draw = []
    t0 = time.time()
    for d in range(draws):
        per = defaultdict(lambda: [0, 0, 0, 0])
        fails = defaultdict(int)
        insts = instances if d == 0 else None
        if insts is None:
            insts = []
            for key, bank in sorted(cells.items()):
                got = bankio.derive(bank, "per_pair", draw=d, min_case=3)
                for i in got:
                    i["cell"] = list(key)
                insts += [i for i in got
                          if any(x["episode_id"] == i["episode_id"]
                                 for x in instances)]
        for inst in insts:
            bank = cells[tuple(inst["cell"])]
            r = run_instance(client, prompt, code, bank, inst,
                             tag=f"eval7:d{d}")
            k = f"{inst['cell'][0]}/{inst['cell'][1]}"
            per[k][0] += len(r["hits"])
            per[k][1] += len(r["targets"])
            per[k][2] += len(r["props"] - r["hits"])
            per[k][3] += r["coverage"]
            if r["code_error"]:
                fails["code_or_token_limit"] += 1
            if r["parse_error"]:
                fails["unparseable_reply"] += 1
            print(f"d{d} {inst['episode_id']}: "
                  f"{len(r['hits'])}/{len(r['targets'])} "
                  f"fp={len(r['props'] - r['hits'])} "
                  f"cov={r['coverage']} tok={r['n_tokens']}", flush=True)
        per_draw.append({"per_cell": {k: v[:] for k, v in per.items()},
                         "failures": dict(fails)})
    agg = {}
    for k in per_draw[0]["per_cell"]:
        rec = [d["per_cell"][k][0] for d in per_draw]
        tgt = per_draw[0]["per_cell"][k][1]
        fp = [d["per_cell"][k][2] for d in per_draw]
        cov = [d["per_cell"][k][3] for d in per_draw]
        agg[k] = {"recovered_per_draw": rec, "targets": tgt,
                  "recall_mean": sum(r / tgt for r in rec) / len(rec),
                  "recall_spread": (max(rec) - min(rec)) / tgt,
                  "fp_mean": sum(fp) / len(fp),
                  "coverage_mean": sum(cov) / len(cov)}
    macro = sum(v["recall_mean"] for v in agg.values()) / len(agg)
    tot_r = sum(sum(v["recovered_per_draw"]) / len(v["recovered_per_draw"])
                for v in agg.values())
    tot_t = sum(v["targets"] for v in agg.values())
    out = {"draws": draws, "macro_recall": macro,
           "micro_recall": tot_r / tot_t, "targets_total": tot_t,
           "recovered_mean": tot_r, "per_cell": agg,
           "failures": [d["failures"] for d in per_draw],
           "runtime_s": round(time.time() - t0, 1)}
    json.dump(out, open(path, "w"), indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "per_cell"},
                     indent=2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blockers", nargs="+", default=["lambdafold"])
    ap.add_argument("--train-datasets", nargs="+",
                    default=["amazon-google", "walmart-amazon",
                             "dblp-acm", "fodors-zagats"])
    ap.add_argument("--test-datasets", nargs="+",
                    default=["abt-buy", "dblp-scholar"])
    ap.add_argument("--val-per-cell", type=int, default=12)
    ap.add_argument("--model", default="qwen3:8b")
    ap.add_argument("--budget", type=int, default=5000)
    ap.add_argument("--minibatch", type=int, default=4)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--eval-split", default=None,
                    choices=[None, "test", "val", "train"])
    ap.add_argument("--eval-draws", type=int, default=3)
    ap.add_argument("--prompt-file", default=None)
    ap.add_argument("--code-file", default=None)
    ap.add_argument("--tag", default="best")
    args = ap.parse_args()
    os.makedirs(args.run_dir, exist_ok=True)

    cells = load_cells(args.blockers,
                       args.train_datasets + args.test_datasets)
    if not cells:
        raise SystemExit("no bank cells found")
    train, val, test, notes = build_splits(
        cells, set(args.train_datasets), set(args.test_datasets),
        args.val_per_cell)
    for n in notes:
        print("[split] " + n, flush=True)
    # the lexicon decides which tokens get masked in reflection
    # feedback; building it over held-out cells would let their
    # vocabulary statistics shape training, so it sees train cells only
    train_cells = {k: v for k, v in cells.items()
                   if k[1] in set(args.train_datasets)}
    lex = mask.ContentLexicon(train_cells)
    log = harness.EventLog(os.path.join(args.run_dir,
                                        "llm_events.jsonl"))
    client = harness.OllamaClient(args.model, log, think=False,
                                  num_ctx=MAX_CTX)

    prompt, code = SEED_PROMPT, SEED_CODE
    if args.prompt_file:
        prompt = open(args.prompt_file, encoding="utf-8").read()
    if args.code_file:
        code = open(args.code_file, encoding="utf-8").read()

    if args.eval_split:
        insts = {"train": train, "val": val, "test": test}[args.eval_split]
        evaluate_split(client, cells, insts, prompt, code,
                       os.path.join(args.run_dir,
                                    f"eval_{args.eval_split}_"
                                    f"{args.tag}.json"),
                       draws=args.eval_draws)
        return

    import gepa
    print(f"train={len(train)} val={len(val)} test={len(test)} "
          f"cells={sorted(cells)}", flush=True)
    whitelist = set(lex.audit_prompt(
        SEED_PROMPT + " " + SEED_CODE + " " + CODE_TEMPLATE + " "
        + PROMPT_TEMPLATE))

    def reflection_lm(p):
        if not isinstance(p, str):
            p = "\n\n".join(m.get("content", "") for m in p)
        out = claude_reflection(p)
        leaked = [t for t in lex.audit_prompt(out) if t not in whitelist]
        gate = None
        if leaked:
            gate = leaked[:20]
            out = claude_reflection(
                p + "\n\nYour draft contained tokens taken from the data "
                f"itself: {leaked[:20]}. Those will not exist in the data "
                "this is deployed on. Rewrite without them, same output "
                "format.")
        log.append({"ts": time.time(), "tag": "reflection-sonnet",
                    "messages": p, "raw_response": out,
                    "leakage_gate_triggered": gate})
        return out

    json.dump({"arm": "run7-symmetric-unbiased",
               "blockers": args.blockers,
               "train_datasets": args.train_datasets,
               "test_datasets": args.test_datasets,
               "val_per_cell": args.val_per_cell,
               "task_model": args.model, "num_ctx": MAX_CTX,
               "reflection": "claude sonnet CLI, blind + leakage gate",
               "budget": args.budget, "minibatch": args.minibatch,
               "score": "(hits - 0.2*wrong)/|targets|, no proposal cap, floored at -2.0; unusable message or reply = -3.0",
               "n_train": len(train), "n_val": len(val),
               "n_test": len(test), "split_notes": notes,
               "seed_candidate": {"prompt": prompt, "code": code}},
              open(os.path.join(args.run_dir, "config.json"), "w"),
              indent=2)

    result = gepa.optimize(
        seed_candidate={"prompt": prompt, "code": code},
        trainset=train, valset=val,
        adapter=Adapter(cells, client, lex,
                        reseed_ids=[i["episode_id"] for i in train],
                        n_train=len(train)),
        reflection_lm=reflection_lm,
        reflection_minibatch_size=args.minibatch,
        module_selector="round_robin",
        reflection_prompt_template={"prompt": PROMPT_TEMPLATE,
                                    "code": CODE_TEMPLATE},
        max_metric_calls=args.budget,
        run_dir=args.run_dir, seed=0, display_progress_bar=False,
        raise_on_exception=False)
    best = result.best_candidate
    open(os.path.join(args.run_dir, "best_prompt.txt"), "w",
         encoding="utf-8").write(best["prompt"])
    open(os.path.join(args.run_dir, "best_code.py"), "w",
         encoding="utf-8").write(best["code"])
    print("FINISHED", result.total_metric_calls, "rollouts,",
          result.num_candidates, "candidates")


if __name__ == "__main__":
    main()
