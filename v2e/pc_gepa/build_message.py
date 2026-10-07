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
    def fmt_val(v, maxlen):
        s = str(v)
        if len(s) > maxlen:
            s = s[:maxlen] + "..."
        return s

    def fields(r, maxlen):
        return " | ".join("%s: %s" % (k, fmt_val(v, maxlen))
                          for k, v in r.items() if k != "id")

    x_by_id = {r["id"]: r for r in recs_x}
    y_by_id = {r["id"]: r for r in recs_y}
    gx, gy = given_pair
    gx_rec = x_by_id.get(gx)
    gy_rec = y_by_id.get(gy)

    def order(records, keep_id):
        # known-pair record first (if any), rest shuffled so large
        # groups get a representative sample rather than a fixed prefix
        head = [r for r in records if r["id"] == keep_id]
        rest = [r for r in records if r["id"] != keep_id]
        rng.shuffle(rest)
        return head + rest

    header_lines = []
    header_lines.append("Group X has %d records, group Y has %d records."
                       % (len(recs_x), len(recs_y)))
    if gx_rec is not None and gy_rec is not None:
        header_lines.append("Known same-entity pair (calibration example"
                             " only, do not repeat it in your answer):")
        header_lines.append("  X: " + fields(gx_rec, 300))
        header_lines.append("  Y: " + fields(gy_rec, 300))
    header_lines.append("")
    header_lines.append("Only report a pair if the two records almost"
                         " certainly describe the same real-world entity."
                         " Answer using the numbers shown below.")
    header_lines.append("")
    header_text = "\n".join(header_lines)

    # Conservative char budget for the whole message, leaving headroom
    # for the system prompt and for tokenization denser than 1 token
    # per ~2.5 characters.
    CHAR_BUDGET = 60000
    remaining = max(0, CHAR_BUDGET - len(header_text))
    total_n = max(1, len(recs_x) + len(recs_y))
    budget_x = int(remaining * len(recs_x) / total_n)
    budget_y = remaining - budget_x

    def select(records, keep_id, budget, maxlen):
        chosen = []
        used = 0
        for r in order(records, keep_id):
            line = fields(r, maxlen)
            cost = len(line) + 10
            if used + cost > budget:
                continue
            chosen.append(r["id"])
            used += cost
        return chosen

    maxlen = 300
    x_ids = select(recs_x, gx if gx_rec is not None else None, budget_x, maxlen)
    y_ids = select(recs_y, gy if gy_rec is not None else None, budget_y, maxlen)

    shrink = maxlen
    while not x_ids and recs_x and shrink > 20:
        shrink //= 2
        x_ids = select(recs_x, gx if gx_rec is not None else None, budget_x, shrink)
    shrink = maxlen
    while not y_ids and recs_y and shrink > 20:
        shrink //= 2
        y_ids = select(recs_y, gy if gy_rec is not None else None, budget_y, shrink)

    lines = [header_text, "Group X records:"]
    for n, rid in enumerate(x_ids, 1):
        lines.append("%d. %s" % (n, fields(x_by_id[rid], maxlen)))
    lines.append("")
    lines.append("Group Y records:")
    for n, rid in enumerate(y_ids, 1):
        lines.append("%d. %s" % (n, fields(y_by_id[rid], maxlen)))

    if len(x_ids) < len(recs_x) or len(y_ids) < len(recs_y):
        lines.append("")
        lines.append(
            "Note: list is shortened; only %d of %d group X and %d of %d "
            "group Y records are shown." % (
                len(x_ids), len(recs_x), len(y_ids), len(recs_y)))

    return "\n".join(lines), x_ids, y_ids
