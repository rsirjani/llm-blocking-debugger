"""
Generates GEPA training examples for the blocking-diagnosis prompt, using
Arm A's missed matches on fodors-zagats — the same failure mode
Blocking_prompt_1.txt was written for (a true match split across two
blocks), but generated programmatically instead of hand-picked, and for a
different dataset than the original amazon-google example.

Each example bundles: the missed pair, PLUS the full contents of both
blocks the two records ended up in (blocking key = city, matching
blocking_A.py) — this is what the prompt template needs to fill its
{block_a_contents} / {block_b_contents} placeholders.

Run from the project root:
    python3 generate_examples.py
Writes: gepa_examples_fodors_zagats.json
"""

import json
import os
import pandas as pd

DATASET_NAME = "fodors-zagats"
BLOCKING_KEY = "city"  # must match blocking_A.py — if that changes, this must too
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

tableA = pd.read_csv(f"datasets/{DATASET_NAME}/tableA.csv")
tableB = pd.read_csv(f"datasets/{DATASET_NAME}/tableB.csv")
gold = pd.read_csv(f"datasets/{DATASET_NAME}/matches.csv").rename(
    columns={"fodors_id": "id_A", "zagats_id": "id_B"}
)

# Recompute candidate pairs the same way blocking_A.py does, to find missed
# gold pairs without depending on a separate output file being present:
cand = tableA.merge(tableB, on=BLOCKING_KEY, suffixes=("_A", "_B"))
cand_keys = set(zip(cand["id_A"], cand["id_B"]))
gold_keys = set(zip(gold["id_A"], gold["id_B"]))
missed = gold_keys - cand_keys

print(f"{len(missed)} missed gold pairs found for {DATASET_NAME}")


def format_record(row, id_col="id", fields=("name", "addr", "city", "phone", "type")):
    parts = [f"{f}: {row[f]}" for f in fields]
    return f"[{row[id_col]}] " + " | ".join(parts)


def format_block(df_subset, id_col="id"):
    lines = [format_record(row, id_col=id_col) for _, row in df_subset.iterrows()]
    return "\n".join(lines)


examples = []
for id_a, id_b in missed:
    row_a = tableA[tableA["id"] == id_a].iloc[0]
    row_b = tableB[tableB["id"] == id_b].iloc[0]

    block_a_df = tableA[tableA[BLOCKING_KEY] == row_a[BLOCKING_KEY]]
    block_b_df = tableB[tableB[BLOCKING_KEY] == row_b[BLOCKING_KEY]]

    example = {
        "source_a": "Fodors",
        "source_b": "Zagats",
        "id_a": int(id_a),
        "id_b": int(id_b),
        "record_a": format_record(row_a).split("] ", 1)[1],  # strip leading [id]
        "record_b": format_record(row_b).split("] ", 1)[1],
        "n_records_a": len(block_a_df),
        "n_records_b": len(block_b_df),
        "block_a_contents": format_block(block_a_df),
        "block_b_contents": format_block(block_b_df),
    }
    examples.append(example)

out_path = f"{RESULTS_DIR}/gepa_examples_{DATASET_NAME.replace('-', '_')}.json"
with open(out_path, "w") as f:
    json.dump(examples, f, indent=2)

print(f"Wrote {len(examples)} examples to {out_path}")
print("\nNote: unlike the original amazon-google example, this dataset's")
print("missed matches are almost all city-naming mismatches (e.g. 'los")
print("angeles' vs 'west la'), not the failure modes the prompt's schema")
print("was designed around (vocabulary_mismatch, description_swamping,")
print("size_cap_split, etc.). Worth checking whether the LLM's diagnosis")
print("output even fits this dataset's actual failure pattern, or whether")
print("this dataset needs its own failure-mode taxonomy — flag before using.")
