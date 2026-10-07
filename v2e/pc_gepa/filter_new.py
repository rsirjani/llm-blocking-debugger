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
    TOPK = 5

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

    size_x = {r["id"]: rec_chars(r) for r in recs_x}
    size_y = {r["id"]: rec_chars(r) for r in recs_y}

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

    cap_x = max(5, n_x // 20)
    cap_y = max(5, n_y // 20)

    def build_index(word_sets, count, cap):
        index = defaultdict(list)
        for rid, ws in word_sets.items():
            for w in ws:
                if count[w] <= cap:
                    index[w].append(rid)
        return index

    index_x = build_index(word_sets_x, count_x, cap_x)
    index_y = build_index(word_sets_y, count_y, cap_y)

    def idf(count, n, w):
        return math.log((n + 1) / (count[w] + 1)) + 1.0

    def topk_matches(word_sets, other_index, other_count, other_n, k):
        # for each record, its top-k strongest cross-block candidates
        # (id, score) — used for relevance ranking and to find soft
        # mutual (reciprocal-ish) pairs below. Top-k instead of top-1
        # so reciprocal near-neighbors aren't missed just because the
        # single best match isn't perfectly symmetric.
        out = {}
        for rid, ws in word_sets.items():
            acc = defaultdict(float)
            for w in ws:
                postings = other_index.get(w)
                if not postings:
                    continue
                weight = idf(other_count, other_n, w)
                for cand_id in postings:
                    acc[cand_id] += weight
            if acc:
                out[rid] = sorted(acc.items(), key=lambda t: t[1], reverse=True)[:k]
            else:
                out[rid] = []
        return out

    topk_x = topk_matches(word_sets_x, index_y, count_y, n_y, TOPK)
    topk_y = topk_matches(word_sets_y, index_x, count_x, n_x, TOPK)

    best_score_x = {rid: (lst[0][1] if lst else 0.0) for rid, lst in topk_x.items()}
    best_score_y = {rid: (lst[0][1] if lst else 0.0) for rid, lst in topk_y.items()}

    max_score_x = max(best_score_x.values()) if best_score_x else 0.0
    max_score_y = max(best_score_y.values()) if best_score_y else 0.0

    def relevance(best_score, max_score, rid, seed_similarity):
        norm = (best_score.get(rid, 0.0) / max_score) if max_score > 0 else 0.0
        return norm + seed_similarity

    rel_x = {r["id"]: relevance(best_score_x, max_score_x, r["id"], sim(r["id"], seed_b)) for r in others_x}
    rel_y = {r["id"]: relevance(best_score_y, max_score_y, r["id"], sim(seed_a, r["id"])) for r in others_y}

    # soft mutual nearest-neighbor pairs: x is in y's top-k AND y is in
    # x's top-k (not necessarily each other's #1). Strong signal of a
    # true pair unrelated to the seed. Surface both members early on
    # each side's list so they land together inside the shared budget
    # window instead of being ranked independently and split apart.
    score_map_y = {rid: dict(lst) for rid, lst in topk_y.items()}
    mutual_candidates = []
    for x_id, lst in topk_x.items():
        for y_id, score_x in lst:
            y_scores = score_map_y.get(y_id)
            if y_scores and x_id in y_scores:
                combined = score_x + y_scores[x_id]
                density = combined / (size_x.get(x_id, 1) + size_y.get(y_id, 1))
                mutual_candidates.append((density, combined, x_id, y_id))
    mutual_candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)

    mutual_x_ids = []
    mutual_y_ids = []
    seen_x = {seed_a}
    seen_y = {seed_b}
    for _, _, x_id, y_id in mutual_candidates:
        if x_id in seen_x or y_id in seen_y:
            continue
        seen_x.add(x_id)
        seen_y.add(y_id)
        mutual_x_ids.append(x_id)
        mutual_y_ids.append(y_id)

    # value-density packing: rank remaining records by relevance per
    # char, not raw relevance, so the tight shared budget favors many
    # cheap-relevant records over a few large ones that would starve
    # the rest of the list (large description/URL fields are common
    # and can eat the whole budget in 5-6 records otherwise).
    rest_x = sorted(
        (r for r in others_x if r["id"] not in seen_x),
        key=lambda r: rel_x[r["id"]] / max(size_x[r["id"]], 1),
        reverse=True,
    )
    rest_y = sorted(
        (r for r in others_y if r["id"] not in seen_y),
        key=lambda r: rel_y[r["id"]] / max(size_y[r["id"]], 1),
        reverse=True,
    )

    ordered_x = ([by_id_x[seed_a]] if seed_a in by_id_x else []) \
        + [by_id_x[i] for i in mutual_x_ids] + rest_x
    ordered_y = ([by_id_y[seed_b]] if seed_b in by_id_y else []) \
        + [by_id_y[i] for i in mutual_y_ids] + rest_y

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
    num_x, used_x = fit_count(sizes_x, half)
    num_y, used_y = fit_count(sizes_y, half)

    # give unused budget from a side that ran out of records to the
    # side that still has more to show.
    if used_x < half and num_x == len(sizes_x) and used_y >= half:
        y_budget = half + (half - used_x)
        num_y, used_y = fit_count(sizes_y, y_budget)
    elif used_y < half and num_y == len(sizes_y) and used_x >= half:
        x_budget = half + (half - used_y)
        num_x, used_x = fit_count(sizes_x, x_budget)

    a_ids = [r["id"] for r in ordered_x[:max(num_x, 1)]]
    b_ids = [r["id"] for r in ordered_y[:max(num_y, 1)]]

    return a_ids, b_ids
