"""
ARM B — Single-sample bit-sampling LSH (blocklib, lambda-fold), pilot run
on fodors-zagats (dev set)

Source of truth for every constraint below: BLOCKING_PROJECT_OPTIONS.txt

STATUS: PILOT / SMOKE TEST ONLY. Same caveats as Arm A: fodors-zagats is
DEV/CALIBRATION ONLY, never used for evaluation claims.

REQUIRED CONFIG PER PROJECT DOC (do not change without checking with your
mentor first — these aren't arbitrary defaults):
  - Method: PPRLIndexLambdaFold (blocklib's "lambda-fold" blocking type)
  - Lambda: 1 — NOT a free parameter. The doc is explicit: "Lambda=1 is a
    restriction imposed here, not either paper's proposal: Lambda-fold as
    published uses Lambda tables and fails P1; Lambda=1 discards the fold
    amplification." I.e. Lambda>1 would violate the same one-block-per-
    record constraint the whole project cares about — do not "improve"
    recall by raising it without discussing that tradeoff explicitly.
  - blocklib is a DORMANT package (no release since Jul 2023) per the doc —
    if `pip install blocklib` pulls something unexpectedly different,
    that's worth flagging, not silently working around.

KNOWN UNRESOLVED ISSUE — confirm before trusting output:
  The project doc flags: "blocklib's signature filter DELETES out-of-range
  blocks - it drops records rather than regrouping them, silently producing
  zero-block records. Disable it, or assert every input id appears in
  exactly one block."
  I could not confirm the exact parameter name for this filter against the
  currently-installed blocklib version from documentation alone (versions
  have drifted since the package went dormant). The assertion below
  (`assert_p1`) is the doc's own suggested fallback — it will tell you
  AFTER THE FACT whether records silently vanished, even if the filter
  itself isn't explicitly disabled. Do not skip this check.
"""

import pandas as pd
from blocklib import generate_candidate_blocks

# ---- 1. Load data (same real files as Arm A) ----------------------------
tableA = pd.read_csv("datasets/fodors-zagats/tableA.csv")
tableB = pd.read_csv("datasets/fodors-zagats/tableB.csv")
gold = pd.read_csv("datasets/fodors-zagats/matches.csv").rename(
    columns={"fodors_id": "id_A", "zagats_id": "id_B"}
)

print(f"Loaded |A|={len(tableA)}, |B|={len(tableB)}, gold pairs={len(gold)}")

# ---- 2. Prep data for blocklib -------------------------------------------
# blocklib requires list-of-lists/tuples input, not a DataFrame. Column 0
# must be the record id per blocklib's convention.
records_A = tableA[["id", "name", "addr", "city", "phone", "type"]].astype(str).values.tolist()
records_B = tableB[["id", "name", "addr", "city", "phone", "type"]].astype(str).values.tolist()

# ---- 3. Blocking config ---------------------------------------------------
# blocking-features uses column INDICES into the record lists above
# (0=id, 1=name, 2=addr, 3=city, 4=phone, 5=type). Using city (index 3) to
# mirror Arm A's blocking key choice for a fair comparison between arms —
# TODO: confirm with your mentor that using the SAME feature across arms is
# the right comparison to draw, or whether each arm should use its
# method-appropriate default instead.
blocking_config = {
    "type": "lambda-fold",
    "version": 1,
    "config": {
        "blocking-features": [3],  # city
        "Lambda": 1,               # REQUIRED value per project doc — do not change
        "bf-len": 2048,
        "num-hash-funcs": 5,
        "K": 20,
        "input-clks": False,
        "random_state": 0,         # fixed seed for reproducibility
    },
}

# ---- 4. Generate candidate blocks for each party independently ----------
result_A = generate_candidate_blocks(records_A, blocking_config)
result_B = generate_candidate_blocks(records_B, blocking_config)

blocks_A = result_A.blocks  # dict: signature -> list of record ids
blocks_B = result_B.blocks

print(f"Party A: {len(blocks_A)} distinct block signatures")
print(f"Party B: {len(blocks_B)} distinct block signatures")


# ---- 5. Validate P1 (per project doc's validation protocol) -------------
# "assert blocks-per-record == 1, never validate on pair counts" — this is
# the doc's own suggested fallback for the signature-filter issue above.
def assert_p1(blocks: dict, table_name: str):
    seen = {}
    violations = []
    for sig, ids in blocks.items():
        for rec_id in ids:
            if rec_id in seen:
                violations.append((rec_id, seen[rec_id], sig))
            seen[rec_id] = sig
    if violations:
        print(f"WARNING [{table_name}]: {len(violations)} records appear in "
              f"more than one block — P1 VIOLATED. Sample: {violations[:3]}")
    else:
        print(f"[{table_name}] P1 holds: every record in exactly one block "
              f"(or zero — check separately for dropped records).")
    return violations


assert_p1(blocks_A, "Party A")
assert_p1(blocks_B, "Party B")

# Separately check for records that vanished entirely (the doc's specific
# "silently producing zero-block records" warning) — P1-holds is NOT the
# same as "no records were dropped":
all_ids_A = set(tableA["id"].astype(str))
blocked_ids_A = {rec_id for ids in blocks_A.values() for rec_id in ids}
missing_A = all_ids_A - blocked_ids_A
if missing_A:
    print(f"WARNING: {len(missing_A)} Party A records got NO block at all "
          f"(dropped, not regrouped). Sample: {list(missing_A)[:5]}")

all_ids_B = set(tableB["id"].astype(str))
blocked_ids_B = {rec_id for ids in blocks_B.values() for rec_id in ids}
missing_B = all_ids_B - blocked_ids_B
if missing_B:
    print(f"WARNING: {len(missing_B)} Party B records got NO block at all "
          f"(dropped, not regrouped). Sample: {list(missing_B)[:5]}")

# ---- 6. Form candidate pairs: same signature in both A and B ------------
common_sigs = set(blocks_A) & set(blocks_B)
pairs = []
for sig in common_sigs:
    for id_a in blocks_A[sig]:
        for id_b in blocks_B[sig]:
            pairs.append((id_a, id_b))

print(f"Candidate pairs after blocking: {len(pairs)}")

# ---- 7. Compute PC / RR / |C| against gold -------------------------------
cand_keys = {(int(a), int(b)) for a, b in pairs}
gold_keys = set(zip(gold["id_A"], gold["id_B"]))

PC = len(gold_keys & cand_keys) / len(gold_keys) if gold_keys else float("nan")
RR = 1 - len(pairs) / (len(tableA) * len(tableB))

print(f"PC = {PC:.4f}, RR = {RR:.4f}, |C| = {len(pairs)}")

# ---- 8. Write candidate pairs ---------------------------------------------
pd.DataFrame(pairs, columns=["id_A", "id_B"]).to_csv(
    "pilot_candidate_pairs_B.csv", index=False
)
print("Wrote pilot_candidate_pairs_B.csv")
