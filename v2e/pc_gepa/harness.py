"""Block-debug episode harness: rendering, Ollama client, scoring, feedback.

Invariants honored:
- Gold reaches the TASK prompt only via the preregistered seed pair of each
  episode. Targets are used post-hoc for scoring and (train-split only) in
  GEPA's reflection feedback — never in the task prompt.
- Every LLM call (prompt + raw response) is appended to llm_events.jsonl.
"""
import itertools
import json
import os
import threading
import time
import urllib.request

# One endpoint, or several comma-separated. Several is how the three
# GPU-pinned servers on the compute box are used: Ollama's multi-GPU
# mode is a layer split, so one model spread over three cards makes them
# take turns; three independent servers, one per card, actually compute
# in parallel. Calls are handed out round-robin.
OLLAMA_URLS = [u.strip().rstrip("/") for u in
               os.environ.get("OLLAMA_URL",
                              "http://localhost:11434").split(",")
               if u.strip()]
OLLAMA = OLLAMA_URLS[0]
# Least-loaded dispatch, not round-robin. Round-robin ignores what is
# already in flight, so with N workers over 3 servers the assignment
# drifts and 4-5 requests land on a server with 3 slots while another
# idles; the queued ones then wait a whole generation. Measured p50 of
# 82s on the unlucky endpoint against 19s on the lucky one, and ~1.6x
# on aggregate throughput. Handing each call to the endpoint with the
# fewest outstanding requests removes the drift.
_inflight = {u: 0 for u in OLLAMA_URLS}
_rr_lock = threading.Lock()


def _acquire_endpoint():
    if len(OLLAMA_URLS) == 1:
        return OLLAMA_URLS[0]
    with _rr_lock:
        u = min(OLLAMA_URLS, key=lambda k: _inflight[k])
        _inflight[u] += 1
        return u


def _release_endpoint(u):
    if len(OLLAMA_URLS) == 1:
        return
    with _rr_lock:
        _inflight[u] = max(0, _inflight[u] - 1)

MATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "match_indices": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["match_indices"],
}

SEED_INSTRUCTION = (
    "You are debugging the output of a blocking system for entity "
    "resolution. Two blocks are shown that should not have been separated "
    "for at least some records: a confirmed example of a true match that "
    "the blocker wrongly split across these two blocks is given as the "
    "SEED. You get a list of numbered candidate record pairs (one record "
    "from each block). Decide which candidate pairs are TRUE MATCHES, "
    "i.e. the two records refer to the same real-world product. Return "
    'JSON: {"match_indices": [<numbers of the true-match pairs>]}. Do not '
    "include the seed pair. If none, return an empty list.")


class EventLog:
    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def append(self, rec):
        with self.lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


class OllamaClient:
    def __init__(self, model, event_log, think=False, temperature=0.0,
                 num_ctx=16384, max_tokens=4096):
        self.model = model
        self.log = event_log
        self.think = think
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.max_tokens = max_tokens

    def chat(self, messages, fmt=None, tag=""):
        body = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": self.think,
            "options": {"temperature": self.temperature,
                        "num_ctx": self.num_ctx,
                        "num_predict": self.max_tokens},
        }
        if fmt is not None:
            body["format"] = fmt
        endpoint = _acquire_endpoint()
        req = urllib.request.Request(
            endpoint + "/api/chat", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        t0 = time.time()
        last_err = None
        # Long-running campaigns reach the model over an ssh tunnel that
        # a watchdog rebuilds within ~3 minutes of a drop. Retry patiently
        # enough to ride that out instead of failing the rollout: eight
        # attempts with capped exponential backoff covers about six
        # minutes of downtime.
        #
        # Six minutes was not enough. 97 of this campaign's 98 dead GEPA
        # iterations were "connection refused" -- the tunnel was gone for
        # hours, not minutes, and once the budget ran out every call in
        # the rollout raised, GEPA logged an exception pass, and the next
        # iteration started and died the same way. The loop span burning
        # iteration numbers while doing no work.
        #
        # A refused connection means nothing is listening: infrastructure
        # is absent, the model has not been asked anything and cannot be
        # blamed. That is a reason to WAIT, not to discard a rollout, so
        # it does not consume the attempt budget. Real failures -- a bad
        # request, a model error, a timeout mid-generation -- still get
        # eight attempts and then surface.
        def absent(e):
            reason = getattr(e, "reason", e)
            return isinstance(reason, (ConnectionRefusedError,
                                       ConnectionResetError)) or (
                isinstance(reason, OSError)
                and reason.errno in (111, 104, 61))

        try:
            attempt = 0
            waited = announced = 0.0
            while attempt < 8:
                try:
                    with urllib.request.urlopen(req, timeout=600) as r:
                        resp = json.loads(r.read())
                    break
                except Exception as e:  # noqa: BLE001 — retry then surface
                    last_err = e
                    if absent(e):
                        # Capped at 60s so recovery is picked up within a
                        # minute of the watchdog rebuilding the tunnel,
                        # and announced every five minutes so an outage is
                        # visible in the log rather than silent.
                        nap = min(60, 2 ** min(attempt, 6))
                        time.sleep(nap)
                        waited += nap
                        if waited - announced >= 300:
                            announced = waited
                            print(f"[endpoint] {endpoint} refusing "
                                  f"connections for {waited / 60:.0f} min; "
                                  "holding the rollout open", flush=True)
                        attempt = min(attempt + 1, 6)
                        continue
                    time.sleep(min(60, 2 ** attempt))
                    attempt += 1
            else:
                raise RuntimeError(f"ollama call failed 8x over ~6 min: "
                                   f"{last_err}")
        finally:
            _release_endpoint(endpoint)
        content = resp["message"]["content"]
        self.log.append({
            "ts": time.time(), "tag": tag, "model": self.model,
            "endpoint": endpoint,
            "think": self.think, "messages": messages, "format_used":
            fmt is not None, "raw_response": content,
            "thinking": resp["message"].get("thinking"),
            "eval_count": resp.get("eval_count"),
            "prompt_eval_count": resp.get("prompt_eval_count"),
            "latency_s": round(time.time() - t0, 2)})
        return content


def load_episodes(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data


def render_user_message(ep, data, max_text=180):
    ta, tb = data["record_texts_a"], data["record_texts_b"]
    sa, sb = ep["seed_pair"]
    lines = [
        "## Blocking method",
        data["method_metadata"],
        "",
        "## The two blocks being debugged",
        f"Block X (signature {ep['block1_sig']}): "
        f"{ep['n_block1_total']} records total, "
        f"{len(ep['block1_a_ids'])} from table A.",
        f"Block Y (signature {ep['block2_sig']}): "
        f"{ep['n_block2_total']} records total, "
        f"{len(ep['block2_b_ids'])} from table B.",
        "",
        "## SEED: confirmed true match wrongly split by the blocker",
        f"A-record (in block X): {ta[sa][:max_text]}",
        f"B-record (in block Y): {tb[sb][:max_text]}",
        "",
        "## Candidate pairs (A-record from block X vs B-record from "
        "block Y), ranked by char-3gram TF-IDF similarity",
    ]
    for i, p in enumerate(ep["top_pairs"], 1):
        lines.append(f"{i}. [sim {p['sim']:.2f}]")
        lines.append(f"   A: {ta[p['a']][:max_text]}")
        lines.append(f"   B: {tb[p['b']][:max_text]}")
    lines.append("")
    lines.append('Return JSON: {"match_indices": [...]}')
    return "\n".join(lines)


def score_episode(ep, match_indices, max_props=25):
    """F1 of proposed pairs vs target (unseen missed gold) pairs."""
    top = ep["top_pairs"]
    seed = tuple(ep["seed_pair"])
    props = set()
    for i in match_indices[:max_props]:
        if isinstance(i, int) and 1 <= i <= len(top):
            p = (top[i - 1]["a"], top[i - 1]["b"])
            if p != seed:
                props.add(p)
    targets = {tuple(t) for t in ep["targets"]}
    hits = props & targets
    prec = len(hits) / len(props) if props else 0.0
    rec = len(hits) / len(targets) if targets else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return f1, props, hits, targets


def feedback_text(ep, data, props, hits, targets, parse_error=None):
    """Gold-aware feedback — reflection-side only, train split only."""
    ta, tb = data["record_texts_a"], data["record_texts_b"]
    if parse_error:
        return (f"Reply unusable: {parse_error}. Emit only JSON "
                '{"match_indices": [...]} with integers from the list.')
    top_set = {(p["a"], p["b"]) for p in ep["top_pairs"]}
    fp = props - targets
    fn = targets - props
    fn_shown = [p for p in fn if p in top_set]
    fn_absent = len(fn) - len(fn_shown)
    parts = [f"Score(F1)={2*len(hits)/(len(props)+len(targets)) if props or targets else 0:.2f}. "
             f"{len(hits)} correct of {len(props)} proposed; "
             f"{len(targets)} true split pairs existed."]
    if hits:
        parts.append("CORRECT: " + "; ".join(
            f"{ta[a][:80]} == {tb[b][:80]}" for a, b in sorted(hits)[:5]))
    if fp:
        parts.append("WRONG (not the same product): " + "; ".join(
            f"{ta[a][:80]} != {tb[b][:80]}" for a, b in sorted(fp)[:5]))
    if fn_shown:
        parts.append("MISSED although shown in the candidate list: "
                     + "; ".join(f"{ta[a][:80]} == {tb[b][:80]}"
                                 for a, b in sorted(fn_shown)[:5]))
    if fn_absent:
        parts.append(f"{fn_absent} true pairs were absent from the "
                     "candidate list (not recoverable here).")
    return " | ".join(parts)


def run_episode(client, instruction, ep, data, tag=""):
    user = render_user_message(ep, data)
    messages = [{"role": "system", "content": instruction},
                {"role": "user", "content": user}]
    raw = client.chat(messages, fmt=MATCH_SCHEMA, tag=tag)
    try:
        idx = json.loads(raw).get("match_indices", [])
        if not isinstance(idx, list):
            raise ValueError("match_indices not a list")
        idx = [i for i in idx if isinstance(i, int)]
        err = None
    except Exception as e:  # noqa: BLE001 — parse failure is signal
        idx, err = [], f"{type(e).__name__}: {e}"
    f1, props, hits, targets = score_episode(ep, idx)
    return {"raw": raw, "indices": idx, "parse_error": err, "score": f1,
            "props": props, "hits": hits, "targets": targets,
            "user_message": user}
