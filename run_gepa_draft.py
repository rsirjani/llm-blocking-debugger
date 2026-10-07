"""
DRAFT GEPA optimization run for the blocking-diagnosis prompt.

STATUS: PROVISIONAL. This uses a placeholder scoring metric (schema
compliance only) because the real question — what counts as a GOOD
diagnosis, not just a well-formatted one — is still open and needs
confirmation before treating any result here as final. See the
"PROVISIONAL METRIC" note below.

Uses GEPA's optimize_anything API (seed_candidate = starting prompt
instructions, evaluate() = scoring function GEPA calls repeatedly while
searching for better instructions).
"""

import json
import re

import gepa.optimize_anything as oa
from gepa.optimize_anything import optimize_anything, GEPAConfig, EngineConfig
from openai import OpenAI

from blocking_prompt_template import SEED_INSTRUCTIONS, format_prompt

# ---- Config ---------------------------------------------------------------
OLLAMA_BASE_URL = "http://127.0.0.1:11434"  # confirm this matches wherever
                                              # the LLMs are actually running
MODEL_TAG = "llama3.1:8b"  # confirm this is actually pulled/available
EXAMPLES_PATH = "results/gepa_examples_fodors_zagats.json"
MAX_METRIC_CALLS = 30  # small budget for a first draft run — raise once
                        # this is confirmed working end-to-end

VALID_FAILURE_MODES = {
    "vocabulary_mismatch", "description_swamping", "size_cap_split",
    "brand_line_granularity", "typo_corruption", "missing_attribute",
    "generic_title", "other",
}
VALID_REMEDIES_KEYWORDS = ("merge", "move", "split")

client = OpenAI(base_url=f"{OLLAMA_BASE_URL}/v1", api_key="ollama")


def strip_code_fences(text: str) -> str:
    """Models often wrap JSON responses in markdown code fences despite
    being asked for JSON only — strip before parsing, confirmed necessary
    from the seed prompt run (52/52 outputs were fenced)."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    return text.strip()

with open(EXAMPLES_PATH) as f:
    examples = json.load(f)

print(f"Loaded {len(examples)} examples from {EXAMPLES_PATH}")


def run_prompt(instructions: str, example: dict) -> str:
    prompt_text = format_prompt(instructions, example)
    response = client.chat.completions.create(
        model=MODEL_TAG,
        messages=[{"role": "user", "content": prompt_text}],
        temperature=0,
    )
    return response.choices[0].message.content


def score_output(raw_text: str) -> tuple[float, str]:
    """
    PROVISIONAL METRIC — schema compliance only, NOT diagnostic quality.
    Returns (score in [0,1], explanation string for GEPA's reflection).

    This intentionally does NOT judge whether the classification or
    remedy is actually correct — there's no ground truth for that yet.
    Confirm the real scoring approach before trusting optimization
    results produced with this metric; it will happily reward a prompt
    that produces well-formed nonsense.
    """
    checks = []
    score = 0.0

    try:
        parsed = json.loads(strip_code_fences(raw_text))
        checks.append("valid JSON: yes (+0.4)")
        score += 0.4
    except json.JSONDecodeError:
        return 0.0, "Output was not valid JSON (after fence-stripping) — automatic score of 0."

    failure_mode = parsed.get("failure_mode") or parsed.get("classification")
    if failure_mode in VALID_FAILURE_MODES:
        checks.append(f"failure mode '{failure_mode}' is valid (+0.2)")
        score += 0.2
    else:
        checks.append(f"failure mode '{failure_mode}' NOT in allowed set (+0)")

    rationale = parsed.get("rationale", "")
    if len(rationale.split()) >= 10:
        checks.append("rationale present and substantive (+0.2)")
        score += 0.2
    else:
        checks.append("rationale missing or too short (+0)")

    remedy_text = json.dumps(parsed).lower()
    if any(kw in remedy_text for kw in VALID_REMEDIES_KEYWORDS):
        checks.append("remedy present and recognizable (+0.2)")
        score += 0.2
    else:
        checks.append("no recognizable remedy (+0)")

    return score, "; ".join(checks)


def evaluate(candidate_instructions: str) -> tuple[float, dict]:
    """GEPA calls this repeatedly with different candidate instruction
    texts, scoring each against all loaded examples."""
    scores = []
    feedback_lines = []

    for i, example in enumerate(examples):
        try:
            raw_output = run_prompt(candidate_instructions, example)
        except Exception as e:
            scores.append(0.0)
            feedback_lines.append(f"Example {i}: LLM call failed ({e})")
            continue

        s, explanation = score_output(raw_output)
        scores.append(s)
        feedback_lines.append(f"Example {i}: score={s:.2f} — {explanation}")

    avg_score = sum(scores) / len(scores) if scores else 0.0
    oa.log(f"Average score: {avg_score:.3f} across {len(examples)} examples")

    return avg_score, {"per_example_feedback": "\n".join(feedback_lines)}


if __name__ == "__main__":
    result = optimize_anything(
        seed_candidate=SEED_INSTRUCTIONS,
        evaluator=evaluate,
        objective=(
            "Produce a prompt that reliably gets the LLM to output valid "
            "JSON diagnosing why an entity-resolution blocker separated "
            "two known-matching records, following the required schema "
            "(failure_mode, rationale, evidence_tokens, remedy)."
        ),
        config=GEPAConfig(engine=EngineConfig(max_metric_calls=MAX_METRIC_CALLS)),
    )

    print("\n=== RESULT ===")
    print("Best score:", result.best_candidate)
    print("\nBest instructions:\n", result.best_candidate)

    with open("results/gepa_result_fodors_zagats.json", "w") as f:
        json.dump({
            "best_candidate": result.best_candidate,
            "note": "PROVISIONAL run — schema-compliance metric only, "
                    "not diagnostic-quality. Do not treat as a final "
                    "optimized prompt until the real scoring approach "
                    "is confirmed.",
        }, f, indent=2)
    print("\nWrote results/gepa_result_fodors_zagats.json")
