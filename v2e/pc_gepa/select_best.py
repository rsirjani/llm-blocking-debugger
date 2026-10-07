#!/usr/bin/env python3
"""Write prompts/seed.txt and prompts/best.txt for the harvest.

Reads only JSON and text that the driver already writes: candidate texts
from candidates.json, val scores from run_log.txt. Nothing here has to
deserialise GEPA's binary state, so the harvest can run unattended from
a timer with no approval step in the way.

"Best" is the candidate that held the highest val score, not the newest.
GEPA's accept gate is a minibatch comparison, so a later candidate is
not necessarily a better one -- five proposals in a row were rolled out
without beating candidate 3 on the val panel.
"""
import json
import os
import re
import sys

RUN = sys.argv[1] if len(sys.argv) > 1 else "../runs/gepa/v2e_qwen38"
OUT = os.path.join(RUN, "prompts")

BEST = re.compile(r"Iteration (\d+): Best score on valset: ([-\d.]+)")
FRONT = re.compile(r"Iteration (\d+): Linear pareto front program index: (\d+)")


def load_state_means():
    """Per-candidate val means from GEPA's state, or None if unreadable.

    Optional by design: the harvest must still run when the state cannot
    be deserialised (a different gepa version, a truncated write). The
    log-only path is enough to pick the frontier holder; this is a
    cross-check, not the source of truth.
    """
    import pickle
    import statistics
    path = os.path.join(RUN, "gepa_state.bin")
    try:
        with open(path, "rb") as f:
            state = pickle.load(f)
        return [statistics.mean(s.values())
                for s in state["prog_candidate_val_subscores"]]
    except Exception as exc:                       # noqa: BLE001
        print(f"  note: {os.path.basename(path)} not readable ({exc}); "
              f"falling back to run_log only")
        return None


def main():
    with open(os.path.join(RUN, "candidates.json"), encoding="utf-8") as f:
        cands = json.load(f)
    text = [list(c.values())[0] if isinstance(c, dict) else str(c)
            for c in cands]

    scores, idx_at = {}, {}
    with open(os.path.join(RUN, "run_log.txt"), encoding="utf-8",
              errors="replace") as f:
        for line in f:
            m = BEST.search(line)
            if m:
                scores[int(m.group(1))] = float(m.group(2))
            m = FRONT.search(line)
            if m:
                idx_at[int(m.group(1))] = int(m.group(2))

    # "Best score on valset" is the running frontier best, so it must be
    # paired with the frontier index, NOT with the candidate introduced
    # in the same iteration. Those agree only while every accepted
    # candidate is also the new best; candidate 8 was accepted on the
    # minibatch gate, scored 0.3871, and left the frontier at 7. Pairing
    # on "New program candidate index" therefore relabelled 8 with 7's
    # 0.5920 and dumped the wrong prompt for the harvest.
    pairs = [(s, idx_at[i]) for i, s in scores.items() if i in idx_at]
    if not pairs:
        sys.exit("no scored candidates found in run_log.txt")
    # Ties go to the earlier candidate: it reached the score first and is
    # the one the frontier actually holds.
    best_score, best_idx = max(pairs, key=lambda p: (p[0], -p[1]))
    if best_idx >= len(text):
        sys.exit(f"run_log names candidate {best_idx} but candidates.json "
                 f"holds only {len(text)}")

    # The log only ever names the frontier holder, so it cannot report a
    # score for a candidate that was accepted without taking the
    # frontier. GEPA's state carries every candidate's own val mean; use
    # it when it loads, and refuse to dump a prompt the two disagree on
    # -- harvesting the wrong arm is silent and unrecoverable.
    means = load_state_means()
    if means:
        state_idx = max(range(len(means)), key=lambda i: (means[i], -i))
        if state_idx != best_idx:
            sys.exit(f"run_log says best is candidate {best_idx} but "
                     f"gepa_state says {state_idx} "
                     f"({means[state_idx]:+.4f} vs {means[best_idx]:+.4f}). "
                     f"Refusing to dump an ambiguous best.")
        best_score = means[best_idx]

    os.makedirs(OUT, exist_ok=True)
    for name, i in (("seed", 0), ("best", best_idx)):
        p = os.path.join(OUT, f"{name}.txt")
        with open(p, "w", encoding="utf-8") as f:
            f.write(text[i])
        print(f"  {name}: candidate {i}, {len(text[i])} chars -> {p}")
    print(f"  best val {best_score:+.4f} = candidate {best_idx} "
          f"(of 0..{len(text) - 1})")
    if means:
        for i, m in sorted(enumerate(means), key=lambda p: -p[1]):
            print(f"    cand{i}  val {m:+.4f}"
                  f"{'   <- best' if i == best_idx else ''}")
    else:
        print("    (gepa_state unreadable; per-candidate means unavailable)")
        for s, i in sorted(set(pairs), reverse=True):
            print(f"    frontier held by cand{i} at {s:+.4f}")


if __name__ == "__main__":
    main()
