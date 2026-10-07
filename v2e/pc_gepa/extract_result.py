"""Extract best candidate + val scores from a finished GEPA run_dir.

Uses gepa's own state-loading APIs (the state file is this run's own
artifact, written by gepa in gepa_run.py).
"""
import inspect
import json
import os
import sys

from gepa.core.result import GEPAResult

run_dir = sys.argv[1]

state = None
try:
    from gepa.core.state import GEPAState
    for name in ("load", "from_run_dir", "load_from_dir", "init_from_dir"):
        fn = getattr(GEPAState, name, None)
        if fn is not None:
            print(f"trying GEPAState.{name}{inspect.signature(fn)}")
            try:
                state = fn(run_dir)
                break
            except Exception as e:  # noqa: BLE001
                print(f"  {name} failed: {e!r}")
except ImportError as e:
    print("no gepa.core.state:", e)

if state is None:
    # fall back to gepa's own serializer used at run time
    import pickle  # noqa: S403 — our own run artifact
    with open(os.path.join(run_dir, "gepa_state.bin"), "rb") as f:
        state = pickle.load(f)  # noqa: S301
    print("loaded via pickle, type:", type(state))

result = GEPAResult.from_state(state)
d = result.to_dict()
aggs = d.get("val_aggregate_scores")
if not aggs:
    subs = d.get("val_subscores") or []
    aggs = [(sum(s.values()) / len(s)) if isinstance(s, dict) else s
            for s in subs]
best_idx = result.best_idx
summary = {
    "n_candidates": result.num_candidates,
    "best_idx": best_idx,
    "val_aggregate_scores": aggs,
    "seed_val_score": aggs[0] if aggs else None,
    "best_val_score": aggs[best_idx] if aggs else None,
    "total_metric_calls": result.total_metric_calls,
    "num_full_val_evals": result.num_full_val_evals,
}
with open(os.path.join(run_dir, "summary.json"), "w") as f:
    json.dump({**summary,
               "best_instruction": result.best_candidate["instruction"]},
              f, indent=2)
with open(os.path.join(run_dir, "best.txt"), "w", encoding="utf-8") as f:
    f.write(result.best_candidate["instruction"])
print(json.dumps(summary, indent=2))
print("BEST INSTRUCTION:")
print(result.best_candidate["instruction"])
