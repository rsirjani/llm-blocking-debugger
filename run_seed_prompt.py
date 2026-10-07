"""
Runs the SEED prompt (unmodified, no GEPA optimization) against every
example in results/gepa_examples_fodors_zagats.json, and saves what the LLM
actually outputs for each one.

This is a look-before-you-optimize step: see what the current prompt
produces on real data before spending calls on GEPA search. Also useful
for figuring out what a real scoring metric should look like, since you
can eyeball actual outputs here.

Run from project root:
    python3 run_seed_prompt.py
Writes: results/seed_prompt_results_fodors_zagats.jsonl (one JSON object
per line, one line per example)
"""

import json
import time

from openai import OpenAI

from blocking_prompt_template import SEED_INSTRUCTIONS, format_prompt

OLLAMA_BASE_URL = "http://127.0.0.1:11434"  # confirm matches actual setup
MODEL_TAG = "llama3.1:8b"  # confirm this is actually pulled/available
EXAMPLES_PATH = "results/gepa_examples_fodors_zagats.json"
OUT_PATH = "results/seed_prompt_results_fodors_zagats.json"

client = OpenAI(base_url=f"{OLLAMA_BASE_URL}/v1", api_key="ollama")

with open(EXAMPLES_PATH) as f:
    examples = json.load(f)

print(f"Loaded {len(examples)} examples from {EXAMPLES_PATH}")
print(f"Running seed prompt against all {len(examples)} examples...")

results = []
for i, example in enumerate(examples, 1):
    prompt_text = format_prompt(SEED_INSTRUCTIONS, example)

    t0 = time.time()
    try:
        response = client.chat.completions.create(
            model=MODEL_TAG,
            messages=[{"role": "user", "content": prompt_text}],
            temperature=0,
        )
        raw_output = response.choices[0].message.content
        error = None
    except Exception as e:
        raw_output = None
        error = str(e)
    latency_s = time.time() - t0

    # Try to parse as JSON, since the prompt asks for JSON-only output —
    # record whether it actually complied, don't assume it did.
    parsed = None
    malformed_json = False
    if raw_output is not None:
        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError:
            malformed_json = True

    result = {
        "example_index": i - 1,
        "id_a": example["id_a"],
        "id_b": example["id_b"],
        "raw_output": raw_output,
        "parsed": parsed,
        "malformed_json": malformed_json,
        "error": error,
        "latency_s": latency_s,
    }
    results.append(result)

    status = "OK" if not error and not malformed_json else (
        "MALFORMED" if malformed_json else "ERROR"
    )
    print(f"  [{i}/{len(examples)}] id_a={example['id_a']} id_b={example['id_b']} -> {status}")

with open(OUT_PATH, "w") as f:
    json.dump(results, f, indent=2)

n_ok = sum(1 for r in results if not r["error"] and not r["malformed_json"])
n_malformed = sum(1 for r in results if r["malformed_json"])
n_error = sum(1 for r in results if r["error"])

print(f"\nDone. {n_ok} OK, {n_malformed} malformed JSON, {n_error} call errors.")
print(f"Wrote {OUT_PATH}")
print("\nWorth eyeballing a few outputs manually before deciding on a real")
print("scoring metric — open the .jsonl file and read some 'parsed' fields.")
