"""
ARM A — Standard blocking, pilot run on fodors-zagats (dev set)

STATUS: PILOT / SMOKE TEST ONLY.
  - fodors-zagats is DEV/CALIBRATION ONLY per BLOCKING_PROJECT_OPTIONS.txt —
    never used for evaluation claims. This exists to prove the mechanics
    work, not to produce a reportable number.
  - Real file structure (confirmed on server, no longer a guess):
      datasets/fodors-zagats/tableA.csv   -> id,name,addr,city,phone,type,class
      datasets/fodors-zagats/tableB.csv   -> id,name,addr,city,phone,type,class
      datasets/fodors-zagats/matches.csv  -> fodors_id,zagats_id

KNOWN DATA ISSUE — read before trusting PC/RR numbers:
  tableA and tableB use DIFFERENT city naming conventions for the same
  restaurant (e.g. tableA: 'los angeles', tableB: 'west la' for a matching
  pair). Blocking on exact `city` equality will therefore MISS true matches
  whose city strings don't line up — this is not a bug in the script, it's
  a real property of this dataset. Expect PC to come in lower than a naive
  "provably non-overlapping negatives" reading of the project doc might
  suggest. Confirmed by inspecting the real data, not assumed.

IMPLEMENTATION NOTE — deviation from the project doc:
  The project doc names `reclin2` (R) as Arm A's implementation. This
  reimplements the same method (deterministic, conjunctive equi-join — what
  pair_blocking() does under the hood) in pandas instead, so Arm A shares
  the Python loader convention with the rest of the pipeline. The method is
  the same; the package is not. Flag this to your mentor before treating
  results as final.
"""

import pandas as pd

# ---- 1. Load data ------------------------------------------------------
# Real paths, confirmed on the server (no longer placeholders):
tableA = pd.read_csv("datasets/fodors-zagats/tableA.csv")
tableB = pd.read_csv("datasets/fodors-zagats/tableB.csv")
gold = pd.read_csv("datasets/fodors-zagats/matches.csv")
# gold columns are fodors_id, zagats_id — normalize to id_A/id_B below

gold = gold.rename(columns={"fodors_id": "id_A", "zagats_id": "id_B"})

ID_COL_A = "id"
ID_COL_B = "id"

print(f"Loaded |A|={len(tableA)}, |B|={len(tableB)}, gold pairs={len(gold)}")

# ---- 2. Block ------------------------------------------------------
# Semantics matching reclin2's pair_blocking(on = <single set>):
#   - ONE blocking key set, conjunctive (all columns must match).
#   - Equivalent to an inner merge on those columns.
#   - Do NOT do a k-of-n / "match on any of these" union — that's exactly
#     what pair_minsim() and rbind()-of-pairs do in R, and the project doc
#     flags both as P1 violations (a record ending up in more than one
#     block). A plain merge here can't accidentally do that as long as
#     `on` stays a single conjunctive set.
#
# Blocking key choice: the project doc calls out `city` specifically for
# fodors-zagats ("city field gives provably non-overlapping negatives").
# This is Arm A's one human-judgement call, and the doc notes it as Arm
# A's unique confound versus B/C/D. TODO: confirm with your mentor before
# treating this as final — don't let my choice stand in for theirs.
BLOCKING_KEY = ["city"]

pairs = tableA.merge(
    tableB,
    on=BLOCKING_KEY,
    suffixes=("_A", "_B"),
    how="inner",
)

# Recover the actual id pairing (merge above returns full joined rows;
# we only need id_A / id_B for the candidate set)
pairs = pairs[[f"{ID_COL_A}_A", f"{ID_COL_B}_B"]].rename(
    columns={f"{ID_COL_A}_A": "id_A", f"{ID_COL_B}_B": "id_B"}
)

print(f"Candidate pairs after blocking: {len(pairs)}")

# ---- 3. Validate (per project doc's validation protocol, note 3) ----
# "assert blocks-per-record == 1, never validate on pair counts" — a
# P1-violating configuration can still pass a pair-count check if something
# downstream dedupes. A plain equi-join on ONE conjunctive key can't itself
# put a record in two blocks, but assert it rather than assume it, since
# this is exactly the failure mode the doc warns about:
block_membership_A = pairs.groupby("id_A")["id_B"].apply(set)
# TODO: real P1 check depends on how "block" is defined for this method
# (city value == block here). This just confirms the merge behaved as an
# equi-join; it does not yet assert one-block-per-record formally.

# ---- 4. Compute PC / RR / |C| against gold ----
# PC (pairs completeness) = fraction of true gold pairs retained by blocking
# RR (reduction ratio)    = 1 - |C| / (|A| * |B|)
# |C|                     = size of candidate set

cand_keys = set(zip(pairs["id_A"], pairs["id_B"]))
gold_keys = set(zip(gold["id_A"], gold["id_B"]))

PC = len(gold_keys & cand_keys) / len(gold_keys) if gold_keys else float("nan")
RR = 1 - len(pairs) / (len(tableA) * len(tableB))

print(f"PC = {PC:.4f}, RR = {RR:.4f}, |C| = {len(pairs)}")

# ---- 5. Write candidate pairs for the LLM adjudication step ----
pairs.to_csv("pilot_candidate_pairs_A.csv", index=False)
print("Wrote pilot_candidate_pairs_A.csv")

# ---- 6. Interpretation note ----
# If PC comes in low here, that's likely the tableA/tableB city-naming
# mismatch described at the top of this file — check a few missed gold
# pairs manually before assuming the blocking logic itself is broken:
missed = gold_keys - cand_keys
if missed:
    sample_missed = list(missed)[:3]
    print(f"\n{len(missed)} gold pairs NOT captured by blocking. Sample:")
    for id_a, id_b in sample_missed:
        row_a = tableA[tableA[ID_COL_A] == id_a][["city"]].values
        row_b = tableB[tableB[ID_COL_B] == id_b][["city"]].values
        print(f"  id_A={id_a} city={row_a}, id_B={id_b} city={row_b}")
