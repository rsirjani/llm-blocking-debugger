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
    # conservative chars-per-token so char budget stays inside the real
    # token limit even for dense/unicode text; reserve room for the
    # system prompt and the model's JSON answer.
    CHARS_PER_TOKEN = 3
    MAX_TOKENS = 32768
    RESERVE_TOKENS = 2500
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

    by_id = {}
    for r in recs_x + recs_y:
        by_id[r["id"]] = r

    gx, gy = given_pair
    all_x_ids = [r["id"] for r in recs_x]
    all_y_ids = [r["id"] for r in recs_y]
    x_id_set = set(all_x_ids)
    y_id_set = set(all_y_ids)
    has_given = gx in x_id_set and gy in y_id_set

    rest_x = [i for i in all_x_ids if not (has_given and i == gx)]
    rest_y = [i for i in all_y_ids if not (has_given and i == gy)]
    rng.shuffle(rest_x)
    rng.shuffle(rest_y)

    instructions = (
        "Task: match records that describe the same real-world entity.\n"
        "Each X record may match at most one Y record and vice versa.\n"
        "Record numbers do NOT align across groups -- X#3 and Y#3 are "
        "unrelated unless their fields say otherwise. Do not assume "
        "positional pairing. Only report a pair if the fields give you "
        "real confidence it is the same entity; when unsure, omit it. "
        "Not every X record has a match in this message, and not every "
        "Y record has a match in this message -- omit anything without "
        "a confident match rather than guessing."
    )

    header = "Group X has %d records, group Y has %d records." % (
        len(recs_x), len(recs_y))
    running = len(header) + len(instructions) + 150  # slack for headers/newlines

    given_lines = []
    if has_given:
        given_block = ("Known same-entity pair (example only, not part of "
                        "the numbered lists below):\n  X: %s\n  Y: %s"
                        % (fields(by_id[gx]), fields(by_id[gy])))
        given_lines = [given_block]
        running += len(given_block)

    running_holder = [running]

    def try_add(rid, kept):
        line = fields(by_id[rid])
        cost = len(line) + 10
        if running_holder[0] + cost > budget_chars:
            return False
        running_holder[0] += cost
        kept.append(rid)
        return True

    # Interleave X/Y so one large group can't starve the other's budget.
    x_kept, y_kept = [], []
    i = 0
    while i < len(rest_x) or i < len(rest_y):
        if i < len(rest_x):
            try_add(rest_x[i], x_kept)
        if i < len(rest_y):
            try_add(rest_y[i], y_kept)
        i += 1

    x_ids = ([gx] if has_given else []) + x_kept
    y_ids = ([gy] if has_given else []) + y_kept
    if has_given:
        # given pair's records get their own numbered entry too, so the
        # model can cite them by index like any other record.
        running_holder[0] += len(fields(by_id[gx])) + len(fields(by_id[gy])) + 20

    lines = [header, "", instructions, ""]
    lines.extend(given_lines)
    lines.append("")
    lines.append("Group X records:")
    for n, rid in enumerate(x_ids, 1):
        lines.append("%d. %s" % (n, fields(by_id[rid])))
    lines.append("")
    lines.append("Group Y records:")
    for n, rid in enumerate(y_ids, 1):
        lines.append("%d. %s" % (n, fields(by_id[rid])))
    return "\n".join(lines), x_ids, y_ids
