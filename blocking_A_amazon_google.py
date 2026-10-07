"""
ARM A — Standard blocking, run on amazon-google.

STATUS: APPROXIMATION of a specific documented baseline, not a confirmed
replication. Read this before trusting any number below.

BLOCKING_PROJECT_OPTIONS.txt states:
  "PRIMARY BASE for all prelim/v3/v5 work: Fisher @ cmax=250 ->
   PC 0.6938, |C| = 137,986, 398 missing gold pairs"

"Fisher" is not defined anywhere else in the doc — this is the ONLY
mention. I could not confirm what specific blocking method/key "Fisher"
refers to. What's below is a reconstruction based on two clues, not a
confirmed match:
  1. The seed prompt's own worked example (Blocking_prompt_1.txt) shows a
     block of EXACTLY 250 records — matching "cmax=250" too precisely to
     be coincidence.
  2. amazon-google's schema (id, name, description, manufacturer, price)
     has exactly one natural categorical blocking field: `manufacturer`.
     The seed prompt's example block is visibly dominated by "adobe",
     "microsoft" products, consistent with blocking on manufacturer.

CONFIRM WITH YOUR MENTOR before treating this as the real baseline:
  - Is "manufacturer" actually the right blocking key, or something else?
  - What does "cmax=250" actually do — truncate oversized blocks, split
    them, or something else? This script TRUNCATES (keeps first 250
    records per block, arbitrary cutoff) as the simplest interpretation —
    this is a guess, not a confirmed mechanism.
  - This run's PC/RR/|C| are NOT expected to exactly match the doc's
    PC=0.6938, |C|=137,986 unless the guesses above happen to be correct.
    Treat any mismatch as informative, not necessarily as a bug.

KNOWN DATA ISSUE — different from fodors-zagats:
  IDs here are strings, not integers: Amazon side uses short alphanumeric
  codes (e.g. "b000jz4hqo"), Google side uses full URLs. Also,
  `manufacturer` has missing values for some records — those records will
  end up in NO block under exact-match blocking (a real, expected gap,
  not a bug).
"""

import os
import pandas as pd

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

CMAX = 250  # per project doc — see caveats above about what this means

# ---- 1. Load data --------------------------------------------------------
# encoding="latin-1" per BLOCKING_PROJECT_OPTIONS.txt: "Both tables latin-1"
tableA = pd.read_csv("datasets/amazon-google/tableA.csv", encoding="latin-1")
tableB = pd.read_csv("datasets/amazon-google/tableB.csv", encoding="latin-1")
gold = pd.read_csv("datasets/amazon-google/matches.csv").rename(
    columns={"idAmazon": "id_A", "idGoogleBase": "id_B"}
)

print(f"Loaded |A|={len(tableA)}, |B|={len(tableB)}, gold pairs={len(gold)}")

n_missing_mfr_A = tableA["manufacturer"].isna().sum()
n_missing_mfr_B = tableB["manufacturer"].isna().sum()
print(f"Missing manufacturer: A={n_missing_mfr_A}/{len(tableA)}, "
      f"B={n_missing_mfr_B}/{len(tableB)} (these records get NO block)")

# ---- 2. Block on manufacturer, cap block size at CMAX --------------------
cand = tableA.merge(tableB, on="manufacturer", suffixes=("_A", "_B"))
print(f"Candidate pairs BEFORE size cap: {len(cand)}")

# Apply cmax: group by manufacturer, keep only the first CMAX pairs per
# group. This is the simplest possible interpretation of "cmax" — flagged
# above as unconfirmed. TODO: confirm actual truncation/splitting logic.
cand_capped = cand.groupby("manufacturer", group_keys=False).apply(
    lambda g: g.head(CMAX)
)
print(f"Candidate pairs AFTER size cap ({CMAX}): {len(cand_capped)}")

pairs = cand_capped[["id_A", "id_B"]].reset_index(drop=True)

# ---- 3. Compute PC / RR / |C| against gold --------------------------------
cand_keys = set(zip(pairs["id_A"], pairs["id_B"]))
gold_keys = set(zip(gold["id_A"], gold["id_B"]))

PC = len(gold_keys & cand_keys) / len(gold_keys) if gold_keys else float("nan")
RR = 1 - len(pairs) / (len(tableA) * len(tableB))

print(f"\nPC = {PC:.4f}, RR = {RR:.4f}, |C| = {len(pairs)}")
print(f"Doc's reported baseline: PC = 0.6938, |C| = 137,986")
print(f"(If these don't match closely, the 'Fisher'/'cmax' reconstruction "
      f"above is likely wrong in some way — worth discussing, not just "
      f"silently accepting whichever number came out.)")

# ---- 4. Write candidate pairs ----------------------------------------------
pairs.to_csv(f"{RESULTS_DIR}/pilot_candidate_pairs_A_amazon_google.csv", index=False)
print(f"\nWrote {RESULTS_DIR}/pilot_candidate_pairs_A_amazon_google.csv")

# ---- 5. Missed matches, with full block contents (for GEPA examples) -----
missed = gold_keys - cand_keys
print(f"{len(missed)} gold pairs NOT captured by blocking (out of {len(gold_keys)})")
