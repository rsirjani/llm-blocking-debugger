# LLM blocker-debugging: optimized pipeline transfers to unseen datasets

Status report, 2026-08-22. Companion files: `pc_vs_rollouts.png`,
`EVOLUTION.md` (checkpoint-by-checkpoint prompt + algorithm, with real
retrieval samples), `checkpoints.json`, `../results_run5/` (previous
campaign), `../RUN5_PLAN.md` (preregistered protocol).

## 1. What the experiment is

A blocker partitions two record tables into mutually exclusive blocks;
at a realistic operating point it splits many true matches across
blocks (pair completeness ≈ 0.6). Assume a few of those failures are
known — **seed pairs**. Each seed names a block-pair that provably
contains blocker errors. We give a cheap local LLM the two blocks plus
the seed and ask it to recover the OTHER split matches hidden there.
Everything is measured on the candidate-set plane: recovered gold pairs
= pair-completeness (PC) recovery. No matcher is ever run.

The system that does this is a two-stage pipeline:

```
seed → block-pair → [FILTER: which records of each block to show]
                  → [JUDGE LLM: which cross-list pairs are the same entity]
                  → recovered pairs
```

**Both stages are optimized automatically by GEPA** (reflective prompt
evolution). The candidate is a pair of text components — the judge's
instruction and the filter's Python source — and the reflection LLM
(Claude Sonnet, via CLI) rewrites either one from execution feedback.
The judge is qwen3:8b running locally on one consumer GPU: cheap model
does the volume work (5,000 rollouts), strong model does the ~200
engineering decisions.

**Blinding (this run's key control).** The reflection LLM is never told
which dataset or which blocker produced an episode, and every record
excerpt it sees has dataset-specific content tokens replaced by
placeholders (`<T7>`), leaving generic vocabulary intact. A leakage
gate audits each proposal and forces a rewrite if dataset content
appears. The judge, by contrast, sees real records — that is
information it will legitimately have at deployment. Purpose: force the
learned policy to be about *structure and evidence*, not memorized
domain content.

## 2. Setup

- Blocker: λ-fold LSH, Λ=1 (mutually exclusive blocks; per-record
  membership asserted). Granularity K auto-tuned per dataset so
  PC ≈ 0.6.
- Train datasets (mixed in one bank, minibatches drawn across them):
  amazon-google, walmart-amazon, dblp-acm, fodors-zagats —
  158 train / 55 val episodes.
- **Held-out datasets, never seen in training**: abt-buy (consumer
  electronics), dblp-scholar (bibliographic, noisy) — 233 episodes,
  **2,277 hidden target pairs**.
- Score per episode: recall of hidden targets − 0.02 × false positives.
- Seed candidate deliberately naive: instruction = 6-line generic task
  statement; filter = *show every record of both blocks, id order*
  (truncated only by the 20k-character prompt budget).

## 3. Result

**Held-out test bank (frozen, single pass, 2,277 targets):**

| pipeline | targets recovered | recall | coverage | false positives |
|---|---|---|---|---|
| naive seed (all records, generic prompt) | 48 | 2.1% | 198 | 1,021 |
| midpoint checkpoint (#16, rollout 1,364) | 553 | 24.3% | 989 | 3,485 |
| **optimized (blind, #41)** | **1,037** | **45.5%** | 1,609 | 3,276 |

Held-out performance tracks the val curve monotonically (2.1% → 24.3% →
45.5%), i.e. the optimizer's in-distribution selection signal was a
faithful proxy for out-of-distribution gain — no evidence of val
overfitting across the run. Coverage (true pairs whose BOTH records
reached the prompt) rises in step, 198 → 989 → 1,609, confirming the
filter, not just the wording, is what moved the ceiling. Note the
midpoint is *better on abt-buy* (186/376) than the final candidate
(152/376) but far worse on the large, noisy dblp-scholar episodes
(367 vs 885 of 1,901) — the later filter rewrites specifically bought
scale robustness.

Per dataset, expressed as pair completeness on the full dataset:

| dataset | recovered | base PC | PC after recovery | ΔPC |
|---|---|---|---|---|
| abt-buy | 152/376 | 0.552 | 0.691 | **+13.9 pts** |
| dblp-scholar | 885/1,901 | 0.592 | 0.758 | **+16.6 pts** |

21× more blocker errors repaired than the naive pipeline, on data the
optimizer never saw, using an 8B model.

**Optimization curve** (`pc_vs_rollouts.png`): mixed-val score
0.033 → 0.355 over 4,300 rollouts, 49 candidates.

## 4. How the pipeline evolved (checkpoints by relative gain)

Every candidate that set a new best val score. "Mutated" = which
component the reflection rewrote. Full text, diffs and real retrieval
samples for each: `EVOLUTION.md`.

| # | val | relative gain | rollout | mutated | what changed |
|---|---|---|---|---|---|
| 0 | 0.033 | — | 0 | seed | show everything, id order; budget truncates blindly |
| 1 | 0.087 | **+160%** | 103 | filter | stop dumping: order records by relevance so the budget is spent on plausible matches |
| 2 | 0.196 | **+125%** | 174 | filter | rank by *cross-block* evidence (rare-token overlap with the other block), not by seed similarity alone |
| 5 | 0.254 | +30% | 363 | filter | model the render cost per record; balance the two lists inside the budget |
| 8 | 0.266 | +5% | 664 | instruction | first judging rubric: identity vs surface noise |
| 14 | 0.274 | +3% | 1,122 | instruction | variant/edition/version mismatch ⇒ not a match |
| 16 | 0.298 | +9% | 1,364 | instruction | reject matches justified only by a shared secondary field or boilerplate |
| 17 | 0.300 | +1% | 1,443 | filter | de-duplicate near-identical distractors |
| 21 | 0.329 | +10% | 1,735 | filter | IDF weighting of shared tokens; posting-list caps for huge blocks |
| 28 | 0.335 | +2% | 2,420 | filter | water-fill leftover budget from the shorter list to the longer |
| 41 | 0.355 | +6% | 3,631 | filter | symmetric best-partner scoring both directions |

Two structural findings:

1. **The retrieval stage carries the early gains** (+160%, +125%, +30%
   from three filter rewrites in the first 400 rollouts), and the
   judging prompt carries the middle (+5%, +3%, +9%). Prompt wording
   cannot recover a pair the filter never showed — a coverage ceiling
   the optimizer had to lift first. This is the empirical case for
   optimizing the *pipeline*, not just the prompt.
2. **What it invented, blind**: the final filter is a small
   meta-blocking engine — schema-agnostic field concatenation, an
   inverted index over rare tokens, IDF-weighted best-partner scoring
   in both directions, an exact model of the prompt's own rendering
   cost, and water-fill budget allocation. It contains zero domain
   references; it was written by a model that never learned what the
   records were about.

## 5. Content-leakage audit

Rare dataset tokens appearing in the final judge prompt (a proxy for
memorized content):

| prompt | leaked content tokens |
|---|---|
| previous campaign (sighted reflection) | 7 (incl. `reseller`, `sku`) |
| **this campaign (blind reflection)** | **3** (`boilerplate`, `debugging`, `descriptor`) — all generic method vocabulary, zero dataset entities |

The blind prompt speaks only in roles ("core identifying field",
"variant/edition marker", "distinctive tokens"), which is why it
transfers to electronics and bibliographic data it never saw.

## 6. Comparison with the prior campaign

| campaign | reflection sees | scaffold | held-out recall |
|---|---|---|---|
| run5 | dataset name, blocker, real record text | pair list from a sampler | 918/2,277 = 40.3% |
| **run6 (this)** | **nothing identifying; masked text** | **block contents + record filter** | **1,037/2,277 = 45.5%** |

Blinding did not cost transfer — it improved it (+5.2 points of
recall), consistent with the memorization hypothesis: content-specific
rules are useless or harmful off-distribution. Caveat: run6 also
proposes more aggressively (3,276 vs 942 false positives), i.e. it
trades precision for recall; both remain within the addition budget
(≈1% of candidate-set size) but the operating points differ.

## 7. Limitations (state these)

- **One blocker family.** All banks are λ-fold LSH; blocker-generality
  is untested. SimCSE/Louvain, reclin2 and klsh cells are the next
  build.
- **Triage is seed-driven.** We debug block-pairs a seed points at; the
  gold-blind triage of unseeded block-pairs is a separate stage, not
  yet built.
- **Selection noise.** The winner is best-of-49 on a 55-episode val;
  the test-side margin over the naive baseline (21×) is far outside
  noise, but the run5-vs-run6 gap (5 points) is not confirmatory.
- **Judge model fixed** at qwen3:8b; larger judges untested.
- This run is at 4,300/5,000 rollouts — numbers above are the current
  pool leader (candidate 41, discovered at rollout 3,631); the last
  600 rollouts may or may not improve it.

## 8. Reproducibility

Every rollout and every reflection call (prompts + raw responses) is
logged in `runs/gepa/run6_blind_blockcontents/llm_events.jsonl`; all 49
candidates with lineage in `candidates.json`; filters are deterministic
per episode, so the retrieval samples in `EVOLUTION.md` are exact
re-renders, not reconstructions.

# Appendix A — the pipeline, worst to best

Verbatim artifacts at three points of the run: the naive seed, the midpoint, and the final pool leader. Both components are shown in full; nothing is paraphrased.

| | val | judge prompt | filter | held-out recall |
|---|---|---|---|---|
| A.1 seed (#0) | 0.033 | 550 chars | 17 lines | 48/2,277 = 2.1% |
| A.2 midpoint (#16) | 0.298 | 3,694 chars | 125 lines | 553/2,277 = 24.3% |
| A.3 best (#41) | 0.355 | 2,462 chars | 197 lines | 1,037/2,277 = 45.5% |

Note the final judge prompt is *shorter* than the midpoint one: the reflection compressed nine verbose rules into a tighter rubric while the filter absorbed the retrieval work.

---

## A.1 — Seed (candidate #0, rollout 0)

*val score 0.033 — the naive start: show every record of both blocks, id order, generic task statement*

### Judge instruction

```
You are debugging the output of a blocking system for entity resolution. Two blocks that wrongly separated at least one true match are shown: list A holds one block's records from table A, list B holds the other block's records from table B. A confirmed true match split across the two blocks is given as the SEED. Find OTHER true matches between list A and list B: pairs referring to the same real-world entity. Return JSON {"matches": [[a_number, b_number], ...]} using the list numbers. Do not include the seed pair. If none, return an empty list.
```

### Filter algorithm

```python
def sample(recs_x, recs_y, seed_pair, sim, rng):
    """FILTER the two blocks: decide which records the judge sees.

    recs_x: list of dicts — ALL records of block X (table A side).
            Every dict has "id" plus dataset-specific string fields.
    recs_y: same for block Y (table B side).
    seed_pair: (a_id, b_id) — confirmed true match the blocker split.
    sim(a_id, b_id) -> float — char-3gram TF-IDF cosine similarity.
    rng: seeded random.Random.

    Return (a_ids, b_ids): the record ids of each block to show, in
    display order. The harness renders each list until a shared
    character budget runs out — records beyond the cutoff are not
    shown, and record pairs where either side is unshown can never be
    recovered.
    """
    return [r["id"] for r in recs_x], [r["id"] for r in recs_y]
```


---

## A.2 — Midpoint (candidate #16, rollout 1,364)

*val score 0.298 — filter now ranks records by best-partner evidence with a sampling fallback for huge blocks; instruction has grown a 9-rule rubric*

### Judge instruction

```
You are debugging output of a blocking system for entity resolution. Two blocks wrongly separated at least one true match: list A holds records from table A, list B holds records from table B. A confirmed true match split across the two blocks given as SEED. Find OTHER true matches between list A and list B — pairs referring to same real-world entity. Return JSON {"matches": [[a_number, b_number], ...]} using list numbers. Do not include seed pair. If none, return empty list.

How to judge a candidate pair:

1. Identify which fields carry the entity's identity (e.g. name/title) versus fields that only describe context, category, or provenance (e.g. venue, source, publisher, format, channel). Two records can have near-identical identity content yet totally different context fields — expected, does NOT disqualify a match; SEED usually shows this pattern.

2. A shared rare or unusual token (distinctive word, code, number, proper name only a few records in block share) is much stronger evidence than shared common/generic words. Scan for overlaps across ALL fields, not just primary identity field. But single shared token outside identity field not enough alone — see rule 7.

3. Field order and role can vary between the two lists (multiple co-authors, attributes, values listed in different order, or split across differently-named fields). Compare by matching each field's role/content to its counterpart, not by raw string position.

4. Treat different surface forms of the same underlying content as equivalent, not as evidence against a match: different capitalization, punctuation, whitespace, encoding of special characters, or spelled-out vs symbolic form of the same value. Do not let a surface-formatting difference cause you to reject an otherwise strong identity-field match, and do not let it cause you to miss one either — check whether the two fields would read the same once such differences are stripped away, before ruling a pair out.

5. Watch for variant or edition markers: version numbers, revision letters, model/part numbers, year/date stamps, size/quantity, or "based on / derived from X" wording. Two records with same base name but DIFFERENT variant/edition markers usually refer to DIFFERENT real-world entities — do not match just because most text overlaps.

6. Do not infer a match from partial identity-field overlap alone when other fields (numbers, dates, secondary names, quantities, counts) point to a different specific item. A short shared phrase or category word across otherwise unrelated records is not evidence.

7. When a candidate pair's strongest evidence is a single field outside the identity field (e.g. one shared secondary name, code, or category), only accept it if the identity fields are also at least plausibly consistent with same entity. A shared secondary field alone — especially short, common, or frequently-repeated value recurring across many unrelated records in block — is not sufficient by itself; prefer pairs where multiple independent fields agree over pairs with only one strong-looking overlap.

8. Work through ENTIRE list A against ENTIRE list B systematically. Never match records because they occupy same or nearby position/index in their lists, never chain a guess off an already-matched pair's position — position carries zero evidence. Every candidate pair must be independently justified by identity-field and token evidence alone.

9. Do not pad output to hit a fixed or "expected" number of matches. Normal and correct for many list-A records to have no counterpart in list B — leave unmatched rather than assign low-confidence guess. Only include a pair when evidence in rules 1–7 clearly supports it.
```

### Filter algorithm

```python
def sample(recs_x, recs_y, seed_pair, sim, rng):
    """FILTER the two blocks: decide which records the judge sees.

    recs_x: list of dicts — ALL records of block X (table A side).
            Every dict has "id" plus dataset-specific string fields.
    recs_y: same for block Y (table B side).
    seed_pair: (a_id, b_id) — confirmed true match the blocker split.
    sim(a_id, b_id) -> float — char-3gram TF-IDF cosine similarity.
    rng: seeded random.Random.

    Return (a_ids, b_ids): the record ids of each block to show, in
    display order. The harness renders each list until a shared
    character budget runs out — records beyond the cutoff are not
    shown, and record pairs where either side is unshown can never be
    recovered.
    """
    BUDGET = 20000

    def rec_chars(r):
        return sum(len(str(k)) + len(str(v)) + 3 for k, v in r.items()) + 8

    seed_a, seed_b = seed_pair

    by_id_x = {r["id"]: r for r in recs_x}
    by_id_y = {r["id"]: r for r in recs_y}

    others_x = [r for r in recs_x if r["id"] != seed_a]
    others_y = [r for r in recs_y if r["id"] != seed_b]

    n_x, n_y = len(others_x), len(others_y)

    # Best-match score per record: does it look like it has a partner
    # on the other side at all (not just near the seed)? Full
    # bipartite scoring when cheap; bounded random sampling otherwise
    # to stay safely inside the time limit on very large blocks.
    MAX_CALLS = 150000

    best_x = {r["id"]: 0.0 for r in others_x}
    best_y = {r["id"]: 0.0 for r in others_y}

    if n_x and n_y:
        full_cost = n_x * n_y
        if full_cost <= MAX_CALLS:
            for rx in others_x:
                xid = rx["id"]
                best = 0.0
                for ry in others_y:
                    yid = ry["id"]
                    s = sim(xid, yid)
                    if s > best:
                        best = s
                    if s > best_y[yid]:
                        best_y[yid] = s
                best_x[xid] = best
        else:
            k_y = max(1, MAX_CALLS // (2 * n_x))
            k_x = max(1, MAX_CALLS // (2 * n_y))
            sample_y = rng.sample(others_y, min(k_y, n_y))
            sample_x = rng.sample(others_x, min(k_x, n_x))
            for rx in others_x:
                xid = rx["id"]
                best = 0.0
                for ry in sample_y:
                    s = sim(xid, ry["id"])
                    if s > best:
                        best = s
                best_x[xid] = best
            for ry in others_y:
                yid = ry["id"]
                best = 0.0
                for rx in sample_x:
                    s = sim(rx["id"], yid)
                    if s > best:
                        best = s
                best_y[yid] = best

    # Fold in direct relevance to the confirmed seed's counterpart —
    # still a useful boost for near-duplicates of the known match.
    seed_score_x = {r["id"]: sim(r["id"], seed_b) for r in others_x}
    seed_score_y = {r["id"]: sim(seed_a, r["id"]) for r in others_y}

    def score_x(r):
        rid = r["id"]
        return max(best_x.get(rid, 0.0), seed_score_x.get(rid, 0.0))

    def score_y(r):
        rid = r["id"]
        return max(best_y.get(rid, 0.0), seed_score_y.get(rid, 0.0))

    scored_x = sorted(others_x, key=score_x, reverse=True)
    scored_y = sorted(others_y, key=score_y, reverse=True)

    ordered_x = ([by_id_x[seed_a]] if seed_a in by_id_x else []) + scored_x
    ordered_y = ([by_id_y[seed_b]] if seed_b in by_id_y else []) + scored_y

    sizes_x = [rec_chars(r) for r in ordered_x]
    sizes_y = [rec_chars(r) for r in ordered_y]

    def fit_count(sizes, budget):
        used = 0
        n = 0
        for s in sizes:
            if used + s > budget:
                break
            used += s
            n += 1
        return n, used

    half = BUDGET / 2.0
    count_x, used_x = fit_count(sizes_x, half)
    count_y, used_y = fit_count(sizes_y, half)

    # Water-fill: give unused budget from a side that ran out of
    # records to the side that still has more to show.
    if used_x < half and count_x == len(sizes_x) and used_y >= half:
        y_budget = half + (half - used_x)
        count_y, used_y = fit_count(sizes_y, y_budget)
    elif used_y < half and count_y == len(sizes_y) and used_x >= half:
        x_budget = half + (half - used_y)
        count_x, used_x = fit_count(sizes_x, x_budget)

    a_ids = [r["id"] for r in ordered_x[:max(count_x, 1)]]
    b_ids = [r["id"] for r in ordered_y[:max(count_y, 1)]]

    return a_ids, b_ids
```


---

## A.3 — Final best (candidate #41, rollout 3,631)

*val score 0.355 — inverted-index prefilter, IDF-weighted symmetric partnerability, exact render-cost model, water-fill budget; instruction compressed to a role-based rubric*

### Judge instruction

```
You are debugging output of blocking system for entity resolution. Two blocks wrongly split ≥1 true match. List A = table A records, list B = table B records. SEED = confirmed true match split across blocks. Find OTHER true matches: pairs naming same real-world entity. Return JSON {"matches": [[a_number, b_number], ...]}. Skip seed pair. Empty list if none.

Match test — same entity if core identifying fields agree, even when:
- case differs, punctuation differs, minor spelling/encoding differs
- words appear in different order (e.g. multi-name list on A written first-to-last, on B last-to-first — same set of names still counts)
- one side abbreviates and other spells out (short form vs long form of same field)
- one venue/category label is a synonym or acronym of the other
- record is truncated/cut off — judge only on visible shared text, don't reject for missing tail

Do NOT match on:
- shared secondary field alone (same one co-author, same manufacturer, same category) without the core identifying field (title/name) also agreeing
- same core identifying field text but differing in a variant/edition/version marker (different version number, platform tag, year, revision letter, size/quantity) — these are distinct entities, not the same record
- generic/boilerplate field values that recur across many unrelated records in the block (common publisher, common category, common short description opener) — these give no evidence alone

Method:
1. Use SEED to learn which field(s) carry identity signal in this pair (e.g. title text, or name+descriptor combo) and what kind of noise differs between A and B sides (case, order, abbreviation) — apply same tolerance to other candidates.
2. For each list-A record, compare against list-B records sharing rare/distinctive tokens (proper nouns, numbers, uncommon words) — ignore pairs sharing only common/filler words.
3. Confirm candidate pairs field-by-field: core identifying field must align under noise rules above; check for a variant/edition/version marker mismatch that would flip a near-match into a false one.
4. Every visible record pair is a candidate — do not skip later-numbered records; truncated list tail still holds valid matches for what's visible.
5. When unsure between two same-scoring candidates, prefer the one with more matching distinctive tokens; do not guess sequential pairing (a_number ≈ b_number) as a shortcut — check text every time.

Output only JSON, no explanation.
```

### Filter algorithm

```python
def sample(recs_x, recs_y, seed_pair, sim, rng):
    """FILTER the two blocks: decide which records the judge sees.

    recs_x: list of dicts — ALL records of block X (table A side).
            Every dict has "id" plus dataset-specific string fields.
    recs_y: same for block Y (table B side).
    seed_pair: (a_id, b_id) — confirmed true match the blocker split.
    sim(a_id, b_id) -> float — char-3gram TF-IDF cosine similarity.
    rng: seeded random.Random.

    Return (a_ids, b_ids): the record ids of each block to show, in
    display order. The harness renders each list until a shared
    character budget runs out — records beyond the cutoff are not
    shown, and record pairs where either side is unshown can never be
    recovered.
    """
    BUDGET = 20000

    def rec_chars(r):
        return sum(len(str(k)) + len(str(v)) + 3 for k, v in r.items()) + 8

    def words_of(r):
        out = []
        for k, v in r.items():
            if k == "id":
                continue
            s = str(v).lower()
            out.extend(re.findall(r"[a-z0-9]{2,}", s))
        return out

    seed_a, seed_b = seed_pair

    by_id_x = {r["id"]: r for r in recs_x}
    by_id_y = {r["id"]: r for r in recs_y}

    others_x = [r for r in recs_x if r["id"] != seed_a]
    others_y = [r for r in recs_y if r["id"] != seed_b]

    n_x = len(others_x) or 1
    n_y = len(others_y) or 1

    word_sets_x = {r["id"]: set(words_of(r)) for r in others_x}
    word_sets_y = {r["id"]: set(words_of(r)) for r in others_y}

    def doc_count(word_sets):
        count = defaultdict(int)
        for ws in word_sets.values():
            for w in ws:
                count[w] += 1
        return count

    count_x = doc_count(word_sets_x)
    count_y = doc_count(word_sets_y)

    # Sub-linear cap: small blocks get a generous cutoff (so moderately
    # shared, still-distinguishing terms stay indexed and reciprocal
    # true pairs get found), large blocks get bounded growth (keeps
    # worst-case index/lookup cost in check).
    cap_words_x = max(5, int(2 * math.sqrt(n_x)))
    cap_words_y = max(5, int(2 * math.sqrt(n_y)))

    def build_index(word_sets, count, cap):
        index = defaultdict(list)
        for rid, ws in word_sets.items():
            for w in ws:
                if count[w] <= cap:
                    index[w].append(rid)
        return index

    index_x = build_index(word_sets_x, count_x, cap_words_x)
    index_y = build_index(word_sets_y, count_y, cap_words_y)

    def idf(count, n, w):
        return math.log((n + 1) / (count[w] + 1)) + 1.0

    def best_matches(word_sets, other_index, other_count, other_n):
        # for each record, its single strongest opposite-side match
        # (id + score) — used both for relevance ranking and to find
        # reciprocal (mutual nearest-neighbor) pairs below.
        best_id = {}
        best_score = {}
        for rid, ws in word_sets.items():
            acc = defaultdict(float)
            for w in ws:
                postings = other_index.get(w)
                if not postings:
                    continue
                weight = idf(other_count, other_n, w)
                for other_id in postings:
                    acc[other_id] += weight
            if acc:
                top_id = max(acc, key=acc.get)
                best_id[rid] = top_id
                best_score[rid] = acc[top_id]
            else:
                best_id[rid] = None
                best_score[rid] = 0.0
        return best_id, best_score

    best_id_x, best_score_x = best_matches(word_sets_x, index_y, count_y, n_y)
    best_id_y, best_score_y = best_matches(word_sets_y, index_x, count_x, n_x)

    max_score_x = max(best_score_x.values()) if best_score_x else 0.0
    max_score_y = max(best_score_y.values()) if best_score_y else 0.0

    def relevance(best_score, max_score, rid, seed_similarity):
        norm = (best_score.get(rid, 0.0) / max_score) if max_score > 0 else 0.0
        return norm + seed_similarity

    rel_x = {r["id"]: relevance(best_score_x, max_score_x, r["id"], sim(r["id"], seed_b)) for r in others_x}
    rel_y = {r["id"]: relevance(best_score_y, max_score_y, r["id"], sim(seed_a, r["id"])) for r in others_y}

    # mutual nearest-neighbor pairs: when x's top opposite-side match
    # is y and y's top opposite-side match is x, that's a strong
    # signal of a true pair unrelated to the seed. Surface both
    # members early on each side's list so they land together inside
    # the shared budget window instead of being ranked independently
    # and split apart.
    mutual_pairs = []
    for x_id, y_id in best_id_x.items():
        if y_id is not None and best_id_y.get(y_id) == x_id:
            score = best_score_x.get(x_id, 0.0) + best_score_y.get(y_id, 0.0)
            mutual_pairs.append((score, x_id, y_id))
    mutual_pairs.sort(key=lambda t: t[0], reverse=True)

    mutual_x_ids = []
    mutual_y_ids = []
    seen_x = {seed_a}
    seen_y = {seed_b}
    for _, x_id, y_id in mutual_pairs:
        if x_id in seen_x or y_id in seen_y:
            continue
        seen_x.add(x_id)
        seen_y.add(y_id)
        mutual_x_ids.append(x_id)
        mutual_y_ids.append(y_id)

    rest_x = sorted(
        (r for r in others_x if r["id"] not in seen_x),
        key=lambda r: rel_x[r["id"]],
        reverse=True,
    )
    rest_y = sorted(
        (r for r in others_y if r["id"] not in seen_y),
        key=lambda r: rel_y[r["id"]],
        reverse=True,
    )

    ordered_x = ([by_id_x[seed_a]] if seed_a in by_id_x else []) \
        + [by_id_x[i] for i in mutual_x_ids] + rest_x
    ordered_y = ([by_id_y[seed_b]] if seed_b in by_id_y else []) \
        + [by_id_y[i] for i in mutual_y_ids] + rest_y

    # Hard ceiling on how many records get shown per side, independent
    # of budget. A block with thousands of small records can otherwise
    # fill the whole budget with distractors — every extra shown
    # record is a chance for the judge to emit a false positive, while
    # only records near the top of the relevance order have any real
    # chance of being the missing true pair. Ceiling grows sub-linearly
    # with block size so small blocks are barely touched but large
    # ones get sharply trimmed.
    show_ceiling_x = max(8, int(4 * math.sqrt(n_x)))
    show_ceiling_y = max(8, int(4 * math.sqrt(n_y)))

    sizes_x = [rec_chars(r) for r in ordered_x]
    sizes_y = [rec_chars(r) for r in ordered_y]

    def fit_count(sizes, budget, ceiling):
        used = 0
        n = 0
        limit = min(ceiling, len(sizes))
        for s in sizes[:limit]:
            if used + s > budget:
                break
            used += s
            n += 1
        return n, used

    half = BUDGET / 2.0
    num_x, used_x = fit_count(sizes_x, half, show_ceiling_x)
    num_y, used_y = fit_count(sizes_y, half, show_ceiling_y)

    # give unused budget from a side that ran out (of records or its
    # ceiling) to the side that still has more to show.
    limit_x_hit = num_x == min(show_ceiling_x, len(sizes_x))
    limit_y_hit = num_y == min(show_ceiling_y, len(sizes_y))
    if used_x < half and limit_x_hit and used_y >= half:
        y_budget = half + (half - used_x)
        num_y, used_y = fit_count(sizes_y, y_budget, show_ceiling_y)
    elif used_y < half and limit_y_hit and used_x >= half:
        x_budget = half + (half - used_y)
        num_x, used_x = fit_count(sizes_x, x_budget, show_ceiling_x)

    a_ids = [r["id"] for r in ordered_x[:max(num_x, 1)]]
    b_ids = [r["id"] for r in ordered_y[:max(num_y, 1)]]

    return a_ids, b_ids
```
