# run4 pipeline — complete before/after

Run: `runs/gepa/run4_pipeline_sonnet` (2026-08-18→19). Candidate =
`{instruction, sampler_code}`. Judge: qwen3:8b (think=false, temp 0,
structured output). Reflection: Claude Sonnet via `claude -p`.
2030 rollouts, 46 candidates. Val score (recall − 0.02·FP): 0.115 → 0.383.
Frozen test: 40/95 → 45/95 targets, FP 268 → 254, coverage 82 → 74.

Frozen (NOT evolved) pipeline parts, for completeness: episode
construction (`build_episodes.py`), user-message render template
(`harness4.render_v4` — seed + numbered pairs with sim scores), judge
JSON schema (`harness.MATCH_SCHEMA`), score function (recall − 0.02·FP,
≤25 proposals), reflection prompt templates (`gepa_run4.CODE_TEMPLATE`,
`INSTR_TEMPLATE`), sandbox contract (`harness4`).

---

## Component 1: `instruction` (system prompt for the judge LLM)

### BEFORE (seed)

```
You are debugging the output of a blocking system for entity resolution.
Two blocks are shown that should not have been separated for at least
some records: a confirmed example of a true match that the blocker
wrongly split across these two blocks is given as the SEED. You get a
list of numbered candidate record pairs (one record from each block).
Decide which candidate pairs are TRUE MATCHES, i.e. the two records
refer to the same real-world product. Return JSON:
{"match_indices": [<numbers of the true-match pairs>]}. Do not include
the seed pair. If none, return an empty list.
```

### AFTER (evolved by Sonnet, candidate 14 of 46)

```
You are debugging the output of a blocking system for entity resolution. Two blocks are shown that should not have been separated for at least some records: a confirmed example of a true match that the blocker wrongly split across these two blocks is given as the SEED. You get a list of numbered candidate record pairs (one record from each block). Decide which candidate pairs are TRUE MATCHES, i.e. the two records refer to the same real-world product (same edition/variant/version, not just the same franchise or brand).

Judge each candidate independently, in isolation — do not assume the list is sorted by relevance, do not skip low-ranked or low-similarity pairs, and do not let a nearby decoy pair change your answer on a different candidate.

Two records are a TRUE MATCH only if the CORE PRODUCT IDENTITY is the same once you account for surface differences. The following are surface noise and must NEVER block a match by themselves, as long as the core product name, edition, and version agree:
- Word order, spelling/punctuation variants (apostrophes, hyphens, "&" vs "and", missing spaces, e.g. "trudy's" = "trudys").
- Vendor-name prefixes or reseller boilerplate glued onto an otherwise identical title (e.g. "aspyr", "topics entertainment", "sos aggregation company", or a manufacturer name repeated twice as a listing prefix like "microsoft microsoft office 2004 software suite").
- Missing/extra generic words ("software", "for pc", "edition", "windows/mac", platform or bundle add-ons like "and ipod") when no specific named edition is introduced.
- Truncated titles, and packaging/format notes such as "(jewel case)", "(small box)", "(win 98 me 2000 xp)" — these describe packaging, not a different product.
- A matching model/part/SKU code appearing in both titles (e.g. "pxe-11433", "01-ssc-2574") is a STRONG positive signal even if the surrounding wording or manufacturer field differs (one side may be a reseller re-listing the same SKU) — do not dismiss these as different products just because the mfr fields differ.

Treat these as PART OF THE CORE IDENTITY — if they differ, it is NOT a match, even if the base product name and manufacturer are identical:
- Edition/variant name (e.g. "university" vs "nightlife" vs "seasons" vs "pets" vs "open for business" expansion packs; "standard" vs "pro" vs "deluxe"; "accountants' edition" vs plain).
- Version/release number or year (e.g. "6" vs "7", "2004" vs "2007", "cs2" vs "cs3", "10.0" vs "17"), UNLESS one side is clearly an "upgrade" label for the same version as the other.
- Product sub-line or distinct named product from the same manufacturer (e.g. "peachtree premium accounting" vs "act! 7.0" — both from Sage/mfr overlap, but different products; a "printmaster gold" listing vs another "printmaster gold" listing from a different publisher/price).
- An accessory/consumable/add-on sold FOR a product (e.g. "laser checks for peachtree accounting") is NOT the product itself — do not match it to the software.
- Platform/brand mismatch stated explicitly on either side (e.g. one title says "norton"/"symantec", the other says "panda" — a shared word like "antivirus 2007" is not enough if the actual brand/product names disagree).

Do NOT infer a match from shared manufacturer, shared brand word, or shared generic template alone (e.g. two different "Sims 2 X expansion pack" entries, two different "webroot spy sweeper" SKUs at different prices with different SKU numbers, two different "printmaster gold" listings from different publishers/prices) — these are frequently distinct catalog entries. Verify the specific distinguishing word/number/SKU lines up on both sides.

Every match you report must be grounded in actual overlapping text: the same product name, same version/edition, or same SKU code appearing in both titles. Never report a match based on vague similarity, shared price, or shared category alone — if you cannot point to the specific overlapping words/numbers that justify it, do not include it.

When several candidates share a near-identical title on one side but differ by price or manufacturer, treat that as a signal they are separate real listings — match each to the specific counterpart whose distinguishing details (variant name, version, mfr, SKU) agree, not just any plausible pairing.

Return JSON: {"match_indices": [<numbers of the true-match pairs>]}. Do not include the seed pair. If none, return an empty list.
```

What changed: "same product" sharpened to same edition/variant/version;
explicit surface-noise whitelist (word order, reseller prefixes,
packaging notes, SKU-match override); explicit identity blacklist
(edition names, version years, sub-lines, accessories, brand mismatch);
anti-heuristics against the judge's observed failure modes (matching on
franchise/brand alone, trusting sim rank, decoy interference); a
grounding requirement (must point to overlapping text).

---

## Component 2: `sampler_code` (pipeline stage, executed — never shown to the judge)

### BEFORE (seed)

```python
def sample(recs_x, recs_y, seed_pair, sim, rng):
    """Return up to 60 (a_id, b_id) candidate pairs for the judge.

    recs_x: list of dicts {"id","title","manufacturer","price",
            "description"} — records of block X from table A (Amazon).
    recs_y: same for block Y from table B (Google).
    seed_pair: (a_id, b_id) — confirmed true match the blocker split.
    sim(a_id, b_id) -> float — char-3gram TF-IDF cosine similarity.
    rng: seeded random.Random — use it for any randomness.
    """
    pairs = [(a["id"], b["id"]) for a in recs_x for b in recs_y]
    pairs.sort(key=lambda p: -sim(p[0], p[1]))
    return pairs[:40]
```

### AFTER (evolved by Sonnet)

```python
def sample(recs_x, recs_y, seed_pair, sim, rng):
    """Return up to 40 (a_id, b_id) candidate pairs for the judge."""
    x_ids = [a["id"] for a in recs_x]
    y_ids = [b["id"] for b in recs_y]
    x_by_id = {a["id"]: a for a in recs_x}
    y_by_id = {b["id"]: b for b in recs_y}
    seed_a, seed_b = seed_pair

    def manu(rec):
        return (rec.get("manufacturer") or "").strip().lower()

    seed_manu = ""
    if seed_a in x_by_id:
        seed_manu = manu(x_by_id[seed_a])
    if not seed_manu and seed_b in y_by_id:
        seed_manu = manu(y_by_id[seed_b])

    scored = []
    for ax in x_ids:
        a = x_by_id[ax]
        ma = manu(a)
        for by in y_ids:
            b = y_by_id[by]
            mb = manu(b)
            s = sim(ax, by)
            bonus = 0.0
            if ma and mb and ma == mb:
                bonus += 0.15
            if seed_manu and (ma == seed_manu or mb == seed_manu):
                bonus += 0.05
            scored.append((s + bonus, ax, by))

    scored.sort(key=lambda t: -t[0])

    result = []
    seen = set()

    if seed_a in x_by_id and seed_b in y_by_id:
        result.append((seed_a, seed_b))
        seen.add((seed_a, seed_b))

    # best candidate per a_id, ranked by score — row-coverage safety net
    best_per_a = {}
    for score, ax, by in scored:
        if ax not in best_per_a:
            best_per_a[ax] = (score, by)
    for ax, (score, by) in sorted(best_per_a.items(), key=lambda kv: -kv[1][0]):
        pair = (ax, by)
        if pair not in seen and len(result) < 40:
            result.append(pair)
            seen.add(pair)

    # fill remaining budget by score, capping per-id repeats so
    # near-duplicate distractors don't crowd out distinct candidates
    a_count = Counter(p[0] for p in result)
    b_count = Counter(p[1] for p in result)
    for score, ax, by in scored:
        if len(result) >= 40:
            break
        pair = (ax, by)
        if pair in seen:
            continue
        if a_count[ax] >= 3 or b_count[by] >= 3:
            continue
        result.append(pair)
        seen.add(pair)
        a_count[ax] += 1
        b_count[by] += 1

    return result[:40]
```

What changed (Sonnet independently rediscovered our design candidates,
then composed them): (1) manufacturer-agreement score bonus (+0.15) and
seed-manufacturer prior (+0.05) on top of TF-IDF; (2) per-A-record
best-match pass — the "record-coverage / per_record_topk" idea, as a
safety net before score-fill; (3) per-id repeat caps (≤3) so hub records
with many near-duplicates can't crowd the list; (4) seed pair pinned
first as an in-context anchor.

Sources: seed = `config.json.seed_candidate` (== `harness.SEED_INSTRUCTION`,
`harness4.SEED_SAMPLER`); final = `best_instruction.txt`,
`best_sampler.py` in the run dir. Full lineage of all 46 candidates:
`candidates.json` + `gepa_state.bin`; every rollout + reflection in
`llm_events.jsonl`.
