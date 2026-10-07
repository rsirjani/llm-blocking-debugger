def build(recs_x, recs_y, given_pair, rng):
    """Build the message shown to the judge model.

    recs_x: list of dicts, one per record of group X. Each dict has an
            "id" key; the remaining keys are that record's fields.
    recs_y: same, for group Y.
    given_pair: (x_id, y_id) -- a pair known to describe the same
            entity, one record from each group.
    rng: seeded random.Random.

    Return (text, x_ids, y_ids):
      text   -- the message the judge reads.
      x_ids  -- record ids in the order they are numbered in text, so
                answer index i refers to x_ids[i-1]; same for y_ids.
    The numbering in text must correspond to these lists. The message
    plus the system prompt must fit in the model's context window.
    """

    def fmt_record(r, cap):
        parts = []
        for k, v in r.items():
            if k == "id":
                continue
            parts.append(str(k) + ": " + str(v))
        s = " | ".join(parts)
        if len(s) > cap:
            s = s[:max(0, cap - 3)] + "..."
        return s

    def raw_text(r):
        parts = []
        for k, v in r.items():
            if k == "id":
                continue
            parts.append(str(k) + ": " + str(v))
        return " | ".join(parts)

    # Stay well clear of the 32768-token limit. Real tokenizers rarely
    # go below ~3 chars/token for this kind of text, so this is a safe
    # lower bound that still leaves plenty of budget to use.
    TOKEN_LIMIT = 32768
    RESERVE_TOKENS = 1500
    CHARS_PER_TOKEN = 3.0
    CHAR_BUDGET = int((TOKEN_LIMIT - RESERVE_TOKENS) * CHARS_PER_TOKEN)

    by_id_x = {r["id"]: r for r in recs_x}
    by_id_y = {r["id"]: r for r in recs_y}

    gx, gy = given_pair
    header_lines = []
    header_lines.append("Group X has %d records, group Y has %d records."
                         % (len(recs_x), len(recs_y)))
    if gx in by_id_x and gy in by_id_y:
        header_lines.append("Known same-entity pair:")
        header_lines.append("  X: " + fmt_record(by_id_x[gx], 300))
        header_lines.append("  Y: " + fmt_record(by_id_y[gy], 300))
    header_lines.append("")
    header_lines.append("Group X records:")
    header_text = "\n".join(header_lines)

    mid_text = "\nGroup Y records:"
    fixed_overhead = len(header_text) + len(mid_text) + 100
    remaining = max(0, CHAR_BUDGET - fixed_overhead)

    total_n = max(1, len(recs_x) + len(recs_y))
    budget_x = int(remaining * len(recs_x) / total_n)
    budget_y = remaining - budget_x

    LINE_OVERHEAD = 8  # numbering prefix + newline, upper bound
    LO_CAP, HI_CAP = 15, 400

    def select_group(records, budget):
        n = len(records)
        if n == 0 or budget <= 0:
            return [], HI_CAP

        order = list(range(n))
        rng.shuffle(order)
        texts = [raw_text(records[i]) for i in range(n)]

        def cost_all(cap):
            total = 0
            for i in order:
                total += min(len(texts[i]), cap) + LINE_OVERHEAD
                if total > budget:
                    return None
            return total

        if cost_all(HI_CAP) is not None:
            cap = HI_CAP
            chosen = order
        elif cost_all(LO_CAP) is None:
            cap = LO_CAP
            chosen = []
            total = 0
            for i in order:
                c = min(len(texts[i]), cap) + LINE_OVERHEAD
                if total + c > budget:
                    if not chosen:
                        continue
                    break
                chosen.append(i)
                total += c
        else:
            lo, hi = LO_CAP, HI_CAP
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if cost_all(mid) is not None:
                    lo = mid
                else:
                    hi = mid
            cap = lo
            chosen = order

        ids = [records[i]["id"] for i in chosen]
        return ids, cap

    x_ids, cap_x = select_group(recs_x, budget_x)
    y_ids, cap_y = select_group(recs_y, budget_y)

    lines = [header_text]
    for n, rid in enumerate(x_ids, 1):
        lines.append("%d. %s" % (n, fmt_record(by_id_x[rid], cap_x)))
    lines.append("")
    lines.append("Group Y records:")
    for n, rid in enumerate(y_ids, 1):
        lines.append("%d. %s" % (n, fmt_record(by_id_y[rid], cap_y)))

    return "\n".join(lines), x_ids, y_ids
