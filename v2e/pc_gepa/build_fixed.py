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
    TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

    def fmt_val(v, maxlen):
        s = str(v)
        if len(s) > maxlen:
            s = s[:maxlen] + "..."
        return s

    def fields(r, maxlen):
        return " | ".join("%s: %s" % (k, fmt_val(v, maxlen))
                          for k, v in r.items() if k != "id")

    def tokenize(r):
        toks = []
        for k, v in r.items():
            if k == "id":
                continue
            toks.extend(t.lower() for t in TOKEN_RE.findall(str(v)))
        return toks

    # Separate id->record maps per group. Ids can collide across X and Y
    # (e.g. both tables use small integer row ids), and merging them into
    # one dict would silently swap in the wrong group's content when
    # rendering a record by id.
    by_id_x = {r["id"]: r for r in recs_x}
    by_id_y = {r["id"]: r for r in recs_y}

    gx, gy = given_pair

    # --- Cheap cross-group token overlap scoring (blocking-style), so
    # truncation for huge groups keeps records more likely to connect to
    # the other side, instead of just taking a prefix. ---
    x_tokens = {}
    y_tokens = {}
    token_doc_freq = Counter()
    for r in recs_x:
        toks = set(tokenize(r))
        x_tokens[r["id"]] = toks
        for t in toks:
            token_doc_freq[t] += 1
    for r in recs_y:
        toks = set(tokenize(r))
        y_tokens[r["id"]] = toks
        for t in toks:
            token_doc_freq[t] += 1

    total_records = max(1, len(recs_x) + len(recs_y))
    max_df = max(2, int(total_records * 0.05))

    x_index = defaultdict(list)
    for xid, toks in x_tokens.items():
        for t in toks:
            if token_doc_freq[t] <= max_df:
                x_index[t].append(xid)
    y_index = defaultdict(list)
    for yid, toks in y_tokens.items():
        for t in toks:
            if token_doc_freq[t] <= max_df:
                y_index[t].append(yid)

    score_x = Counter()
    score_y = Counter()
    for t in (set(x_index) & set(y_index)):
        w = 1.0 / token_doc_freq[t]
        for xid in x_index[t]:
            score_x[xid] += w
        for yid in y_index[t]:
            score_y[yid] += w

    def prioritize(records, given_id, score_map):
        ranked = sorted(records, key=lambda r: -score_map.get(r["id"], 0.0))
        if given_id is not None:
            idx = None
            for i, r in enumerate(ranked):
                if r["id"] == given_id:
                    idx = i
                    break
            if idx is not None and idx != 0:
                rec = ranked.pop(idx)
                ranked.insert(0, rec)
        return ranked

    ranked_x = prioritize(recs_x, gx, score_x)
    ranked_y = prioritize(recs_y, gy, score_y)

    CHAR_BUDGET = 60000

    header = []
    header.append("Group X has %d records, group Y has %d records."
                 % (len(recs_x), len(recs_y)))
    if gx in by_id_x and gy in by_id_y:
        header.append("Known same-entity pair:")
        header.append("  X: " + fields(by_id_x[gx], 300))
        header.append("  Y: " + fields(by_id_y[gy], 300))
    header.append("")
    header.append("Only report a pair if you are confident the two records "
                   "describe the same real-world entity. Do not guess.")
    header.append("")
    header_text = "\n".join(header)

    remaining = CHAR_BUDGET - len(header_text)
    if remaining < 0:
        remaining = 0

    total_n = max(1, len(recs_x) + len(recs_y))
    budget_x = int(remaining * len(recs_x) / total_n)
    budget_y = remaining - budget_x

    def select(records, budget, maxlen):
        chosen = []
        used = 0
        for r in records:
            line = fields(r, maxlen)
            cost = len(line) + 10
            if used + cost > budget and chosen:
                break
            chosen.append(r["id"])
            used += cost
            if used >= budget:
                break
        return chosen

    maxlen = 300
    x_ids = select(ranked_x, budget_x, maxlen)
    y_ids = select(ranked_y, budget_y, maxlen)

    shrink = maxlen
    while not x_ids and recs_x and shrink > 20:
        shrink //= 2
        x_ids = select(ranked_x, budget_x, shrink)
    shrink = maxlen
    while not y_ids and recs_y and shrink > 20:
        shrink //= 2
        y_ids = select(ranked_y, budget_y, shrink)

    lines = [header_text, "Group X records:"]
    for n, rid in enumerate(x_ids, 1):
        lines.append("%d. %s" % (n, fields(by_id_x[rid], maxlen)))
    lines.append("")
    lines.append("Group Y records:")
    for n, rid in enumerate(y_ids, 1):
        lines.append("%d. %s" % (n, fields(by_id_y[rid], maxlen)))

    return "\n".join(lines), x_ids, y_ids
