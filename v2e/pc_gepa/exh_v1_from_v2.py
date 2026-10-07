"""Exhaustive-sweep replicate at variant v1 (ordering/disclosed-seed
replicate of jobs 80/81, 2026-10-06).

Runs v2.main() with the CLI args passed through unchanged, except that
frozen_specs is pinned to variant=1, so the eval instances are
val:<blocker>:<dataset>:v1:0 instead of v0. make_instance seeds every
draw from the instance id, and --pairs-per-instance 700 exceeds every
pool, so the boundary set per cell is identical to v0 by construction;
only visit order and the disclosed seed pair per boundary change
(verified in preflight before launch: same_pool=True in all 10 cells).

Invoked as "python -u exh_v1_from_v2.py --blockers ..." so the cmdline
contains the literal "v2.py --blockers" and the existing queue
wait_idle / health-poller / chainer pgrep patterns all match it.

NOTE: evaluate_panel stores rows[i]["variant"] = 0 (its loop index);
the authoritative variant is in instance_id (":v1:").
"""
import sys
sys.path.insert(0, ".")
import v2

_orig = v2.frozen_specs
def _frozen_v1(keys, per_cell, prefix, variant=1):
    return _orig(keys, per_cell, prefix, 1)
v2.frozen_specs = _frozen_v1
print("[exh_v1] frozen_specs pinned to variant=1", flush=True)
v2.main()
