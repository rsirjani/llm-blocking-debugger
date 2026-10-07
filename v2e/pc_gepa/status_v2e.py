#!/usr/bin/env python3
"""Live progress page for the v2e campaign, served over Tailscale.

http://100.72.5.105:8899 -- Tailscale-only, no public exposure, no
credentials, renders on a phone. Regenerates per request.

Reads GEPA's pickled state for per-candidate validation means. That file
is written by our own optimiser process on local disk on every
iteration; it is not untrusted input, and the validation scores exist
nowhere else (run_log.json carries minibatch scores only). Requested
explicitly so the score-vs-iteration curve can be plotted live.
"""
import http.server
import json
import os
import re
import pickle
import statistics
import threading
import time

RUN = ("/home/tin/projects/entity-matching-llm-blocking/"
       "runs/gepa/v2e_qwen38")
PORT = 8899
PAIRS = 64            # judge calls per instance
MINIBATCH = 24        # instances per rollout, and again per retest
VAL_N = 40            # instances in the validation panel
BUDGET = 5000         # instances

ROLLOUT_CALLS = MINIBATCH * PAIRS          # 1536
RETEST_CALLS = MINIBATCH * PAIRS           # 1536
VAL_CALLS = VAL_N * PAIRS                  # 2560


def driver_pid():
    """The search process, not an eval -- both take --blockers."""
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cl = f.read().decode("utf-8", "replace")
        except OSError:
            continue
        if "v2.py" in cl and "v2e_qwen38" in cl and "--eval-split" not in cl:
            return int(pid)
    return None


def stage_of(calls_in_iter, refl_since_state, split_counts):
    """Which leg of the iteration we are on.

    Derived from what the calls actually ARE, not from how many have
    happened. Counting calls against a fixed budget breaks whenever the
    state file is stale: GEPA writes state at the TOP of an iteration, so
    after an acceptance the counter keeps climbing against a boundary
    that has not moved, and the arithmetic becomes meaningless. Tags do
    not have that problem -- a val: tag can only be produced by the
    validation panel, which only runs after a proposal has been accepted.
    """
    val = split_counts.get("val", 0)
    train = split_counts.get("train", 0)
    if val:
        # Only an accepted candidate is scored on the panel, so the
        # presence of val calls is itself the accept signal.
        return ("accepted - scoring on the val panel", val, VAL_CALLS)
    if refl_since_state == 0:
        if train < ROLLOUT_CALLS:
            return ("rolling out parent on minibatch", train,
                    ROLLOUT_CALLS)
        return ("waiting on reflection", ROLLOUT_CALLS, ROLLOUT_CALLS)
    after = max(0, train - ROLLOUT_CALLS)
    if after < RETEST_CALLS:
        return ("retesting proposal on the same minibatch", after,
                RETEST_CALLS)
    return ("finishing iteration", after, RETEST_CALLS)


# Incremental tail of llm_events.jsonl. This used to re-read and
# re-parse the whole file on every request, and the page carries a
# 120-second meta refresh -- so one phone left on the tracker meant
# re-reading 711 MB and running ~56,000 json.loads every two minutes,
# forever, against a file that only grows. That is a lot of page-cache
# and allocator churn to recompute numbers that never change for events
# already seen. Only bytes appended since the last scan are parsed now.
_SCAN = {"off": 0, "ts": [], "refl_ts": [], "failed": 0,
         "split": [], "sorted": True}
_SCAN_LOCK = threading.Lock()


def _tail_events():
    path = f"{RUN}/llm_events.jsonl"
    with _SCAN_LOCK:
        size = os.path.getsize(path)
        if size < _SCAN["off"]:
            # truncated or replaced (a fresh run in the same dir)
            _SCAN.update(off=0, ts=[], refl_ts=[], failed=0, split=[])
        if size > _SCAN["off"]:
            # Read in bounded chunks. Slurping the gap in one f.read()
            # is fine for the 8 KB an incremental pass sees, but the
            # first pass after a restart faces the whole 700+ MB file
            # and would allocate it twice -- once as the blob, once as
            # the split list. That is worse than the full re-parse this
            # replaced. 4 MB at a time keeps peak RSS flat either way.
            CHUNK = 4 << 20
            rest = b""
            pos = _SCAN["off"]
            with open(path, "rb") as f:
                f.seek(pos)
                while pos < size:
                    blob = rest + f.read(min(CHUNK, size - pos))
                    pos += min(CHUNK, size - pos)
                    # The driver appends while we read, so the last line
                    # of the final chunk is often half-written. Keep the
                    # remainder for the next pass instead of dropping it.
                    cut = blob.rfind(b"\n") + 1
                    rest = blob[cut:]
                    for line in blob[:cut].split(b"\n"):
                        if not line:
                            continue
                        try:
                            e = json.loads(line)
                        except Exception:  # noqa: BLE001 -- tolerate it
                            continue
                        tag = e.get("tag")
                        if tag == "reflection-sonnet":
                            _SCAN["refl_ts"].append(e["ts"])
                        elif tag == "reflection-failed":
                            _SCAN["failed"] += 1
                        else:
                            _SCAN["ts"].append(e["ts"])
                            if tag and tag.startswith("v2:"):
                                _SCAN["split"].append(
                                    (e["ts"], tag.split(":")[1]))
                    del blob
            _SCAN["off"] = size - len(rest)
            _SCAN["sorted"] = False
        if not _SCAN["sorted"]:
            _SCAN["ts"].sort()
            _SCAN["sorted"] = True
        return (list(_SCAN["ts"]), list(_SCAN["refl_ts"]),
                _SCAN["failed"], list(_SCAN["split"]))


def collect():
    ts, refl_ts, failed, split = _tail_events()
    now = time.time()
    state_path = f"{RUN}/gepa_state.bin"
    st = os.path.getmtime(state_path)
    with open(state_path, "rb") as f:
        d = pickle.load(f)
    means = [statistics.mean(list(s.values()))
             for s in d["prog_candidate_val_subscores"]]
    log = json.load(open(f"{RUN}/run_log.json"))

    # Count only iterations that did work. GEPA's loop counter
    # increments on every pass, including ones that threw immediately --
    # a 5-hour network outage inflated it by 97 passes that evaluated
    # nothing, called no reflection, and consumed no budget. Plotting
    # against the raw counter makes dead loop-passes look like search.
    # An iteration only counts if it produced a proposal that got judged.
    # Two kinds don't: passes that threw immediately (a network outage
    # inflated the raw counter by 97 of these), and passes where the
    # parent rollout ran but reflection then failed -- those burned a
    # full 24-instance minibatch, 1,536 judge calls, and returned
    # nothing. 8 of the latter cost 192 instances, 30% of all budget
    # spent so far. Counting either as progress overstates the search.
    failed_refl = set()
    try:
        with open(f"{RUN}/run_log.txt") as f:
            for line in f:
                m = re.match(r"Iteration (\d+): .*did not propose", line)
                if m:
                    failed_refl.add(int(m.group(1)) - 1)
    except OSError:
        pass
    # Having a minibatch is not enough to count. Two more kinds slipped
    # through and inflated this from 6 to 9:
    #
    #  * MERGE attempts. Crossover passes carry subsample_ids too, but
    #    they are five val ids rather than a 24-instance minibatch, run no
    #    reflection, and propose nothing new -- i=2 and i=6 were both
    #    "worse than both parents, skipping merge". Merge is disabled now,
    #    but the two in the history were being reported as search.
    #  * ABANDONED iterations. i=113 recorded selecting a parent and
    #    drawing a batch, then the machine froze before any result was
    #    written. It has no scores of any kind.
    #
    # So require positive evidence that a proposal was actually judged:
    # either the child was scored on the minibatch (new_subsample_scores)
    # or it was accepted onto the frontier (new_program_idx). A merge is
    # identified by its own bookkeeping keys and excluded outright.
    def _is_merge(e):
        return "merged" in e or "id1_subsample_scores" in e

    def _was_judged(e):
        return ("new_subsample_scores" in e or "new_program_idx" in e
                or "new_program_indices" in e)

    worked = [e for e in log
              if e.get("subsample_ids") and e["i"] not in failed_refl
              and not _is_merge(e) and _was_judged(e)]
    n_wasted = len([e for e in log
                    if e.get("subsample_ids") and e["i"] in failed_refl])
    n_merge = len([e for e in log if e.get("subsample_ids")
                   and _is_merge(e)])
    n_abandoned = len([e for e in log
                       if e.get("subsample_ids") and e["i"] not in failed_refl
                       and not _is_merge(e) and not _was_judged(e)])
    ordinal = {e["i"]: n for n, e in enumerate(worked, start=1)}
    n_worked = len(worked)

    # Iteration at which each candidate entered the val panel, indexed
    # among working iterations. The seed is not born in an iteration at
    # all -- it is the starting point, so it sits at 0.
    born = {0: 0}
    for e in log:
        for key in ("new_program_idx", "new_program_indices"):
            v = e.get(key)
            if v is None:
                continue
            for idx in ([v] if isinstance(v, int) else v):
                if idx != 0:
                    born.setdefault(idx, ordinal.get(e["i"], 0))
    curve = sorted((born.get(i, 0), i, m) for i, m in enumerate(means))

    hour = [t for t in ts if t > now - 3600]
    spc = 3600 / max(1, len(hour))
    calls_iter = len([t for t in ts if t > st])
    refl_since = len([t for t in refl_ts if t > st])
    split_counts = {}
    for t_, sp in split:
        if t_ > st:
            split_counts[sp] = split_counts.get(sp, 0) + 1
    stage, s_done, s_total = stage_of(calls_iter, refl_since, split_counts)
    remaining = max(0, (ROLLOUT_CALLS + RETEST_CALLS) - calls_iter)
    return {
        "pid": driver_pid(),
        # Iterations that actually did work -- the honest progress
        # number. The raw loop counter is reported separately.
        "iteration": n_worked,
        "raw_iteration": (log[-1]["i"] + 1) if log else 0,
        "dead_iterations": ((log[-1]["i"] + 1) - n_worked) if log else 0,
        "wasted_iterations": n_wasted,
        "wasted_instances": n_wasted * MINIBATCH,
        "merge_iterations": n_merge,
        "abandoned_iterations": n_abandoned,
        "means": means, "curve": curve,
        "parents": [p[0] for p in d["parent_program_for_candidate"]],
        "frontier": statistics.mean(
            list(d["pareto_front_valset"].values())),
        "evals": d["total_num_evals"],
        "last_call_min": (now - ts[-1]) / 60 if ts else 1e9,
        "spc": spc, "calls_hour": len(hour), "total_calls": len(ts),
        "reflections": len(refl_ts), "refl_failed": failed,
        "stage": stage, "stage_done": s_done, "stage_total": s_total,
        "calls_iter": calls_iter, "state_age_h": (now - st) / 3600,
        "idle_min": (now - max(ts)) / 60 if ts else 1e9,
        "accepted_pending": split_counts.get("val", 0) > 0,
        "iter_total": (ROLLOUT_CALLS + RETEST_CALLS
                       + (VAL_CALLS if split_counts.get("val", 0) else 0)),
        "eta_h": remaining * spc / 3600,
    }


def sparkline(curve, cur_iter, w=560, h=250):
    """Best-so-far val score as a step function against iteration.

    A step, not a connected line: the score does not drift upward
    between accepts. It holds flat at the incumbent until a candidate
    beats it, jumps, and holds again -- and the last step extends to the
    current iteration, because that is still the best we have. A
    polyline through the candidate points would imply improvement during
    stretches where nothing was accepted, which is exactly backwards:
    those stretches are proposals that were rejected, or (from iteration
    6 onward here) a proposer that was down.

    Candidates are also drawn individually. A candidate that scores
    below the incumbent is a real event worth seeing -- it was accepted
    on its minibatch and then failed to beat the best on the full val
    panel -- and it sits below the step line rather than moving it.
    """
    if not curve:
        return "<div class=k>no candidates yet</div>"
    xs = [c[0] for c in curve]
    x0 = min(xs)
    x1 = max(max(xs), cur_iter) or 1
    if x1 == x0:
        x1 = x0 + 1
    ys = [c[2] for c in curve]
    lo, hi = min(ys), max(ys)
    span = (hi - lo) or abs(hi) or 1.0
    y0, y1 = lo - span * 0.30, hi + span * 0.30
    L, R, T, B = 52, 22, 26, 40

    def px(x):
        return L + (w - L - R) * ((x - x0) / (x1 - x0))

    def py(y):
        return T + (h - T - B) * (1 - (y - y0) / (y1 - y0))

    # best-so-far step path, extended to the current iteration
    best = None
    pts = []
    for x, _i, y in curve:
        if best is None:
            best = y
            pts.append((px(x), py(best)))
        elif y > best:
            pts.append((px(x), py(best)))      # hold to the jump
            best = y
            pts.append((px(x), py(best)))      # jump
    pts.append((px(x1), py(best)))             # flat to now
    path = " ".join(f"{a:.1f},{b:.1f}" for a, b in pts)

    dots, labels, ticks = "", "", ""
    seen_x, running = set(), None
    for x, i, y in curve:
        cx, cy = px(x), py(y)
        is_best = running is None or y > running
        running = y if is_best else running
        colour = "#4fc3f7" if is_best else "#7a7a7a"
        dots += (f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4.5" '
                 f'fill="{colour}"><title>cand {i} @ iteration {x}: '
                 f'{y:+.4f}{"" if is_best else " (below best)"}'
                 f'</title></circle>')
        anchor = ("start" if cx < L + 30 else
                  "end" if cx > w - R - 30 else "middle")
        labels += (f'<text x="{cx:.1f}" y="{cy - 11:.1f}" fill="#9ad" '
                   f'font-size="11" text-anchor="{anchor}">c{i} {y:+.3f}'
                   f'</text>')
        if x not in seen_x:
            seen_x.add(x)
            ticks += (f'<line x1="{cx:.1f}" y1="{h - B:.1f}" '
                      f'x2="{cx:.1f}" y2="{h - B + 5:.1f}" stroke="#555"/>'
                      f'<text x="{cx:.1f}" y="{h - B + 18:.1f}" '
                      f'fill="#999" font-size="11" text-anchor="middle">'
                      f'{x}</text>')
    # mark "now" on the axis
    nx = px(x1)
    ticks += (f'<line x1="{nx:.1f}" y1="{T}" x2="{nx:.1f}" '
              f'y2="{h - B:.1f}" stroke="#333" stroke-dasharray="2 4"/>'
              f'<text x="{nx:.1f}" y="{h - B + 18:.1f}" fill="#777" '
              f'font-size="11" text-anchor="end">now ({cur_iter})</text>')
    grid = ""
    for gy in (lo, hi):
        yy = py(gy)
        grid += (f'<line x1="{L}" y1="{yy:.1f}" x2="{w - R}" y2="{yy:.1f}" '
                 f'stroke="#2a2a2a" stroke-dasharray="3 3"/>'
                 f'<text x="{L - 6}" y="{yy + 4:.1f}" fill="#888" '
                 f'font-size="11" text-anchor="end">{gy:+.3f}</text>')
    return f"""<svg viewBox="0 0 {w} {h}" width="100%"
 preserveAspectRatio="xMidYMid meet" role=img style="overflow:visible">
{grid}
<line x1="{L}" y1="{h - B}" x2="{w - R}" y2="{h - B}" stroke="#555"/>
<line x1="{L}" y1="{T}" x2="{L}" y2="{h - B}" stroke="#555"/>
<polyline points="{path}" fill="none" stroke="#4fc3f7" stroke-width="2"/>
{dots}{labels}{ticks}
<text x="{(L + w - R) / 2:.0f}" y="{h - 6}" fill="#999" font-size="12"
 text-anchor="middle">search iteration</text>
<text x="14" y="{(T + h - B) / 2:.0f}" fill="#999" font-size="12"
 text-anchor="middle" transform="rotate(-90 14 {(T + h - B) / 2:.0f})">
best val score</text></svg>"""


def render(s):
    best_i = max(range(len(s["means"])), key=lambda i: s["means"][i])
    best = s["means"][best_i]
    healthy = s["pid"] is not None and s["last_call_min"] < 30
    colour = "#2e7d32" if healthy else "#c62828"
    verdict = ("RUNNING" if healthy else
               "DRIVER NOT RUNNING" if s["pid"] is None else
               "STALLED - no recent calls")
    pct = 100 * s["stage_done"] / max(1, s["stage_total"])
    warn = ""
    if s["refl_failed"]:
        warn += (f"<p class=w>{s['refl_failed']} reflection failure(s). "
                 "Retries forever, so this costs time, not work.</p>")
    # An old state file is not by itself a problem: GEPA writes state at
    # the TOP of an iteration, so a long-running iteration -- especially
    # one that got through the accept gate and is now on the 40-instance
    # val panel -- legitimately leaves it untouched for many hours. What
    # would be a problem is no judge calls. Warn on that instead.
    if s["idle_min"] > 45:
        warn += (f"<p class=w>no judge call in {s['idle_min']:.0f} min "
                 f"-- the run may be stalled.</p>")
    elif s["state_age_h"] > 16 and not s["accepted_pending"]:
        warn += (f"<p class=w>iteration running {s['state_age_h']:.1f} h "
                 f"without closing (expect ~7).</p>")
    if s["last_call_min"] > 30 and s["pid"]:
        warn += ("<p class=w>no judge call recently - tunnels may be "
                 "down; liveness timer retries every 15 min.</p>")
    rows = "".join(
        f"<tr><td>cand {i}{' &larr; best' if i == best_i else ''}</td>"
        f"<td>{m:+.4f}</td>"
        f"<td>{s['parents'][i] if s['parents'][i] is not None else 'seed'}"
        f"</td></tr>"
        for i, m in enumerate(s["means"]))
    return f"""<!doctype html><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<meta http-equiv=refresh content=120>
<title>v2e {verdict} {best:+.3f}</title>
<style>
body{{font:16px/1.5 -apple-system,system-ui,sans-serif;margin:0;
padding:18px;background:#111;color:#eee}}
h1{{font-size:21px;margin:0 0 2px;color:{colour}}}
h2{{font-size:15px;margin:20px 0 4px;color:#9ad;font-weight:600}}
table{{border-collapse:collapse;width:100%;margin:8px 0}}
td{{padding:7px 8px;border-bottom:1px solid #2a2a2a}}
td:last-child{{text-align:right;font-variant-numeric:tabular-nums}}
.k{{color:#888;font-size:13px}}
.w{{background:#4a1c1c;padding:9px;border-radius:6px;font-size:14px;
margin:10px 0}}
.big{{font-size:30px;font-weight:650;margin:6px 0 0}}
.bar{{background:#222;border-radius:5px;height:11px;overflow:hidden;
margin:7px 0}}
.bar>i{{display:block;height:100%;background:#4fc3f7;width:{pct:.1f}%}}
</style>
<h1>{verdict}</h1>
<div class=k>{time.strftime('%a %d %b %H:%M:%S')} &middot; refreshes
every 2 min</div>
{warn}

<p class=big>best {best:+.4f} <span class=k>cand {best_i}</span></p>
<div class=k>frontier {s['frontier']:+.4f} &middot; headroom
{s['frontier'] - best:+.4f} &middot; {len(s['means'])} candidates</div>

<h2>val score by iteration</h2>
{sparkline(s['curve'], s['iteration'])}
<div class=k>step = best so far, held flat until beaten and extended to now. blue = new best, grey = accepted but below the incumbent.</div>

<h2>current iteration</h2>
<div>{s['stage']}</div>
<div class=bar><i></i></div>
<div class=k>{s['stage_done']:,} / {s['stage_total']:,} calls in this
stage ({pct:.0f}%)</div>
<table>
<tr><td>completed reflective iterations</td><td>{s['iteration']}</td></tr>
<tr><td class=k>wasted (rollout, then no proposal)</td>
<td class=k>{s['wasted_iterations']} = {s['wasted_instances']}
instances</td></tr>
<tr><td class=k>merge attempts (not reflective; merge now off)</td>
<td class=k>{s['merge_iterations']}</td></tr>
<tr><td class=k>abandoned mid-iteration (no result written)</td>
<td class=k>{s['abandoned_iterations']}</td></tr>
<tr><td class=k>raw loop counter</td><td class=k>{s['raw_iteration']}
({s['dead_iterations']} non-search passes)</td></tr>
<tr><td>judge calls this iteration</td><td>{s['calls_iter']:,} of
{s['iter_total']:,} calls</td></tr>
<tr><td>est. to close iteration</td><td>{s['eta_h']:.1f} h</td></tr>
<tr><td>since last state write</td><td>{s['state_age_h']:.1f} h</td></tr>
</table>

<h2>candidates</h2>
<table><tr><td class=k>candidate</td><td class=k>val</td>
<td class=k>parent</td></tr>{rows}</table>

<h2>throughput</h2>
<table>
<tr><td>last judge call</td><td>{s['last_call_min']:.1f} min ago</td></tr>
<tr><td>rate</td><td>{s['spc']:.1f} s/call</td></tr>
<tr><td>calls last hour</td><td>{s['calls_hour']:,}</td></tr>
<tr><td>total judge calls</td><td>{s['total_calls']:,}</td></tr>
<tr><td>instances</td><td>{s['evals']:,} / {BUDGET:,}</td></tr>
<tr><td>reflections</td><td>{s['reflections']} ok / {s['refl_failed']}
failed</td></tr>
<tr><td>driver pid</td><td>{s['pid'] or 'none'}</td></tr>
</table>"""


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            body = render(collect()).encode()
            code = 200
        except Exception as e:  # noqa: BLE001 -- always serve something
            body = (f"<meta http-equiv=refresh content=120>"
                    f"<h1 style='color:#c62828'>status error</h1>"
                    f"<pre>{type(e).__name__}: {e}</pre>").encode()
            code = 500
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    http.server.ThreadingHTTPServer(("0.0.0.0", PORT),
                                    Handler).serve_forever()
