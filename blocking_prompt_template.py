"""
Reusable version of Blocking_prompt_1.txt.

The original file is ONE frozen example (amazon-google, records A538/B2200,
block 3 with 250 records). This turns it into a template GEPA can actually
optimize: same instructions, same output schema, but the record/block
content is filled in per-example instead of hardcoded.

GEPA optimizes the parts marked SEED_INSTRUCTIONS below — treat this as the
starting point for optimization, not a finished, tuned prompt.
"""

SEED_INSTRUCTIONS = """You are diagnosing errors made by an automatic blocking algorithm for entity resolution between two product catalogs (source A = {source_a}, source B = {source_b}). The two records below are KNOWN to describe the same product, but the blocker placed them in different blocks, so they will never be compared by the matcher.

Record 1: [{id_a}] {record_a}
Record 2: [{id_b}] {record_b}

Below are the FULL contents of both blocks.

Block containing Record 1, all {n_records_a} records:
{block_a_contents}

Block containing Record 2, all {n_records_b} records:
{block_b_contents}

Why did the blocker separate these two records? Classify the PRIMARY
failure mode as exactly one of:
- "vocabulary_mismatch": same product expressed with different tokens/abbreviations (e.g. 'cs3' vs '3', 'win' vs 'windows')
- "description_swamping": long description/extra text dominates the character n-grams so titles no longer drive similarity
- "size_cap_split": one large product family was forced apart by the block size cap
- "brand_line_granularity": records grouped by brand or product line at different granularity in the two blocks
- "typo_corruption": misspelling or corrupted text broke surface similarity
- "missing_attribute": one record lacks the attribute(s) the other was grouped by
- "generic_title": title too generic/short to carry signal
- "other": none of the above (explain in rationale)

Also give: "rationale" (2-3 sentences tied to the records and block contents above),
"evidence_tokens" (the specific tokens that mismatch or mislead),

you must select exactly one of these as a remedy:
- Merge the two blocks entirely.
- Move a few records from one block to the other, and tell me which records should be moved.
- Split the blocks to create a third block, and tell me which records should be moved to the new block.

Respond with JSON only."""


def format_prompt(instructions: str, example: dict) -> str:
    """Fill the template with one diagnostic example's data.

    `instructions` is the (possibly GEPA-optimized) instruction text —
    passed separately from `example` so GEPA can swap it out between
    rollouts while the example data stays fixed.

    `example` must have keys:
      source_a, source_b, id_a, id_b, record_a, record_b,
      n_records_a, n_records_b, block_a_contents, block_b_contents
    (block_*_contents are pre-formatted strings, one record per line,
    matching the [id] name | field: value | ... style in the original.)
    """
    return instructions.format(**example)
