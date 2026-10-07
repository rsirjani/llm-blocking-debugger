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
    FIELD_CAP = 300
    CHARS_PER_TOKEN = 3
    MAX_TOKENS = 32768
    RESERVE_TOKENS = 4000
    budget_chars = (MAX_TOKENS - RESERVE_TOKENS) * CHARS_PER_TOKEN

    def fields(r):
        parts = []
        for k, v in r.items():
            if k == "id":
                continue
            s = str(v)
            if len(s) > FIELD_CAP:
                s = s[:FIELD_CAP] + "..."
            parts.append("%s: %s" % (k, s))
        return " | ".join(parts)

    # keep group namespaces separate -- x and y ids may collide as raw
    # strings even though they belong to different groups
    by_id_x = {r["id"]: r for r in recs_x}
    by_id_y = {r["id"]: r for r in recs_y}

    gx, gy = given_pair
    has_given = gx in by_id_x and gy in by_id_y

    all_x_ids = list(by_id_x.keys())
    all_y_ids = list(by_id_y.keys())

    rest_x = [i for i in all_x_ids if not (has_given and i == gx)]
    rest_y = [i for i in all_y_ids if not (has_given and i == gy)]
    rng.shuffle(rest_x)
    rng.shuffle(rest_y)

    ordered_x = ([gx] if has_given else []) + rest_x
    ordered_y = ([gy] if has_given else []) + rest_y

    header = "Group X has %d records, group Y has %d records." % (
        len(recs_x), len(recs_y))
    running = len(header) + 100

    given_lines = []
    if has_given:
        given_block = ("Known same-entity pair:\n  X: %s\n  Y: %s"
                        % (fields(by_id_x[gx]), fields(by_id_y[gy])))
        given_lines = [given_block]
        running += len(given_block)

    running_holder = [running]

    def fit(ids, by_id):
        kept = []
        for rid in ids:
            line = fields(by_id[rid])
            cost = len(line) + 10
            if running_holder[0] + cost > budget_chars:
                continue
            running_holder[0] += cost
            kept.append(rid)
        return kept

    x_ids = fit(ordered_x, by_id_x)
    y_ids = fit(ordered_y, by_id_y)

    instructions = (
        "Each numbered X record describes at most one same entity among "
        "the numbered Y records, and vice versa. Most records have no "
        "match in the other group -- do not force a guess for every "
        "record. Only report a pair when you are confident the X record "
        "and the Y record describe the same real-world entity. "
        "Respond with a single JSON object and nothing else, in the form "
        "{\"matches\": [[x_number, y_number], ...]}, listing only the "
        "confident matches (an empty list if none)."
    )

    lines = [header, "", instructions, ""]
    lines.extend(given_lines)
    lines.append("")
    lines.append("Group X records:")
    for n, rid in enumerate(x_ids, 1):
        lines.append("%d. %s" % (n, fields(by_id_x[rid])))
    lines.append("")
    lines.append("Group Y records:")
    for n, rid in enumerate(y_ids, 1):
        lines.append("%d. %s" % (n, fields(by_id_y[rid])))
    return "\n".join(lines), x_ids, y_ids
