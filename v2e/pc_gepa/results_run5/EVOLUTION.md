# run5 evolution: seed -> jumps -> best

Reference episode for all retrieval samples: `amazon-google_K8_000100000_000100010` (blocks 100x227 records, 6 hidden targets). Samplers are deterministic, so these re-renders reproduce exactly what the judge saw during the run.

**Scaffold note / deviation from the original spec:** the run5 scaffold renders PAIRS (the sampler picks (a,b) pairs; the judge answers with pair indices). The user-specified dumb start was block CONTENTS (all records of X as one list, all of Y as another — O(|X|+|Y|) prompt cost, not O(|X|*|Y|)). The seed below is therefore 'all pairs, id order', which truncates far harder than the intended baseline (e.g. 43 of 22,700 pairs shown on the reference episode). A faithful block-contents baseline (records_baseline.py, judge returns [a_number, b_number] pairs over the two lists) is evaluated separately in REPORT.md §4.

---

## Candidate #0  (val 0.000, discovered at rollout 0, seed)

**What its sampler actually retrieved on the reference episode:** 43 pairs shown to the judge, covering 0/6 hidden targets, list TRUNCATED at prompt budget.

Prompt excerpt as fed to the judge (first pairs):
```
1.
   A: title: suretrak project manager 3.0 | description: communicate with professional-quality reports and graphics. suretrak project manager combines ease of use powe | manufacturer: primavera systems | price: 499
   B: name: hallmark blank greeting cards half-fold matte premium 20 count (2050xf) | description: hallmark blank greeting cards half-fold matte premium 20 count brand: nova development mpn: 2050xf variant nam | price: 14.39
2.
   A: title: suretrak project manager 3.0 | description: communicate with professional-quality reports and graphics. suretrak project manager combines ease of use powe | manufacturer: primavera systems | price: 499
   B: name: microsoft office 2004 for students and teachers (mac) | description: key features: for mac for student and teachers full-featured programs word processing spreadsheets create pres | price: 134.99
3.
   A: title: suretrak project manager 3.0 | description: communicate with professional-quality reports and graphics. suretrak project manager combines ease of use powe | manufacturer: primavera systems | price: 499
   B: name: software engineering measurement and analysis by steven strauss 9780849319303 | description: a common perception is that software measurement is a complicated system activity. a corresponding perception  | price: 89.95
4.
   A: title: suretrak project manager 3.0 | description: communicate with professional-quality reports and graphics. suretrak project manager combines ease of use powe | manufacturer: primavera systems | price: 499
   B: name: quicken home and business 2007 (pc) intuit | description: key features: manage personal finances manage business finances simplify taxes save time on your taxes ... | price: 89.99
5.
   A: title: suretrak project manager 3.0 | description: communicate with professional-quality reports and graphics. suretrak project manager combines ease of use powe | manufacturer: primavera systems | price: 499
   B: name: freeverse software 4001 northland | description: stand-alone real-time strategy game based on viking mythology description: stand-alone real-time strategy game | manufacturer: freeverse software | price: 19.99
6.
   A: title: suretrak project manager 3.0 | description: communicate with professional-quality reports and graphics. suretrak project manager combines ease of use powe | manufacturer: primavera systems | price: 499
   B: name: dragon tales learn and fly with dragons | description: explore dragon land as you help cassie's brothers and sisters learn to fly. earn a dragon badge by using your  | price: 17.9
7.
   A: title: suretrak project manager 3.0 | description: communicate with professional-quality reports and graphics. suretrak project manager combines ease of use powe | manufacturer: primavera systems | price: 499
   B: name: statistical methods in software engineering: reliability and risk 9780387988238 | description: this book establishes a framework for dealing with uncertainties in software engineering and for using quantit | price: 99
8.
   A: title: suretrak project manager 3.0 | description: communicate with professional-quality reports and graphics. suretrak project manager combines ease of use powe | manufacturer: primavera systems | price: 499
   B: name: quicken(r) quickbooks(r) microsoft money(r) and simply money(r) elegant laser checks | description: add style to your laser or inkjet checks!choose from several designs. all the great features of the laser chec | price: 98.49
9.
   A: title: suretrak project manager 3.0 | description: communicate with professional-quality reports and graphics. suretrak project manager combines ease of use powe | manufacturer: primavera systems | price: 499
   B: name: software product lines: practices and patterns by paul clements 9780201703320 | description: long a standard practice in traditional manufacturing the concept of product lines is relatively new to the so | price: 64.99
10.
   A: title: suretrak project manager 3.0 | description: communicate with professional-quality reports and graphics. suretrak project manager combines ease of use powe | manufacturer: primavera systems | price: 499
   B: name: punch software 25100 punch! master landscape and home design | description: punch! master landscape & home design offers an extensive plantings database deck topography and new technolog | manufacturer: punch software | price: 45.99
11.
   A: title: suretrak project manager 3.0 | description: communicate with professional-quality reports and graphics. suretrak project manager combines ease of use powe | manufacturer: primavera systems | price: 499
   B: name: jumpstart advanced 1st grade knowledge adventure | description: key features: four cd set enhances education over 50 skills reviewed special features ... | price: 19.99
```

### instruction (full)
```
You are debugging the output of a blocking system for entity resolution. Two blocks are shown that should not have been separated for at least some records: a confirmed example of a true match that the blocker wrongly split across these two blocks is given as the SEED. You get a list of numbered candidate record pairs (one record from each block). Decide which candidate pairs are TRUE MATCHES, i.e. the two records refer to the same real-world product. Return JSON: {"match_indices": [<numbers of the true-match pairs>]}. Do not include the seed pair. If none, return an empty list.
```

### sampler_code (full)
```python
def sample(recs_x, recs_y, seed_pair, sim, rng):
    """Return candidate (a_id, b_id) pairs for the judge.

    recs_x: list of dicts — records of block X (table A side). Every
            dict has "id" plus dataset-specific text fields (products:
            title/manufacturer/price/...; bibliographic: title/authors/
            venue/year; restaurants: name/addr/city/...).
    recs_y: same for block Y (table B side).
    seed_pair: (a_id, b_id) — confirmed true match the blocker split.
    sim(a_id, b_id) -> float — char-3gram TF-IDF cosine similarity.
    rng: seeded random.Random — use for any randomness.
    The harness shows at most 250 pairs to the judge (fewer if the
    prompt character budget is hit) — pairs beyond that are dropped.
    """
    pairs = [(a["id"], b["id"]) for a in recs_x for b in recs_y]
    pairs.sort()
    return pairs
```

---

## Candidate #1  (val 0.227, discovered at rollout 87, parent #0, mutated: sampler_code)

**What its sampler actually retrieved on the reference episode:** 41 pairs shown to the judge, covering 6/6 hidden targets, list TRUNCATED at prompt budget.

Prompt excerpt as fed to the judge (first pairs):
```
1.
   A: title: punch! master landscape & home design | description: master landscape & home design offers nine programs on one easy-to-use interface to help you design home exter | manufacturer: punch! software | price: 59.99
   B: name: punch software 26100 punch! master landscape and home design (small box) | description: punch! master landscape & home design offers an extensive plantings database deck topography and new technolog | manufacturer: punch software | price: 45.99
2.
   A: title: pinnacle mobile media organizer | description: package contents: mobile media organizer cd media mobile master quick install guide pinnacle's mobile media or | manufacturer: pinnacle | price: 49.99
   B: name: pinnacle mobile media organizer software for windows ipod software | description: mobile media organizer software for windows the mobile media organizer software for windows from pinnacle allo | price: 39.95
3.
   A: title: punch! master landscape & home design | description: master landscape & home design offers nine programs on one easy-to-use interface to help you design home exter | manufacturer: punch! software | price: 59.99
   B: name: punch software 26100 punch! master landscape and home design (small box) | description: punch! master landscape & home design offers an extensive plantings database deck topography and new technolog | manufacturer: punch software | price: 45.99
4.
   A: title: punch! master landscape & home design | description: master landscape & home design offers nine programs on one easy-to-use interface to help you design home exter | manufacturer: punch! software | price: 59.99
   B: name: punch software 26100 punch! master landscape and home design (small box) | description: punch! master landscape & home design offers an extensive plantings database deck topography and new technolog | manufacturer: punch software | price: 45.99
5.
   A: title: punch! master landscape & home design | description: master landscape & home design offers nine programs on one easy-to-use interface to help you design home exter | manufacturer: punch! software | price: 59.99
   B: name: punch software 25100 punch! master landscape and home design | description: punch! master landscape & home design offers an extensive plantings database deck topography and new technolog | manufacturer: punch software | price: 45.99
6.
   A: title: punch! master landscape & home design | description: master landscape & home design offers nine programs on one easy-to-use interface to help you design home exter | manufacturer: punch! software | price: 59.99
   B: name: punch software 25100 punch! master landscape and home design | description: punch! master landscape & home design offers an extensive plantings database deck topography and new technolog | manufacturer: punch software | price: 45.99
7.
   A: title: punch! master landscape & home design | description: master landscape & home design offers nine programs on one easy-to-use interface to help you design home exter | manufacturer: punch! software | price: 59.99
   B: name: punch software 25100 punch! master landscape and home design | description: punch! master landscape & home design offers an extensive plantings database deck topography and new technolog | manufacturer: punch software | price: 45.99
8.
   A: title: dk rainbow fish most beautiful fish in the ocean | description: one stormy night while rainbow fish slept on the ocean floor his beautiful sparkling scales were stolen by thr | manufacturer: encore | price: 9.99
   B: name: rainbow fish and the whale | description: rainbow fish is trapped inside the whale and needs help getting out. in this adventure children will encounter | price: 6.95
9.
   A: title: project x project management software | description: project x is truly project management software for the rest of us. its for those of us that want to spend more | manufacturer: marware | price: 199.95
   B: name: microsoft(r) project standard 2007 | description: microsoft project standard 2007 helps you manage projects and to address the work and people management needs  | price: 599.95
10.
   A: title: project x project management software | description: project x is truly project management software for the rest of us. its for those of us that want to spend more | manufacturer: marware | price: 199.95
   B: name: microsoft(r) project standard 2007 | description: microsoft project standard 2007 helps you manage projects and to address the work and people management needs  | price: 349.95
11.
   A: title: punch! master landscape & home design | description: master landscape & home design offers nine programs on one easy-to-use interface to help you design home exter | manufacturer: punch! software | price: 59.99
   B: name: individual software inc total 3d home & landscape deluxe 9 | description: home and landscape design software designed for homeowners. whether you are decorating a room landscaping your | price: 33.5
```

### instruction (full)
```
You are debugging the output of a blocking system for entity resolution. Two blocks are shown that should not have been separated for at least some records: a confirmed example of a true match that the blocker wrongly split across these two blocks is given as the SEED. You get a list of numbered candidate record pairs (one record from each block). Decide which candidate pairs are TRUE MATCHES, i.e. the two records refer to the same real-world product. Return JSON: {"match_indices": [<numbers of the true-match pairs>]}. Do not include the seed pair. If none, return an empty list.
```

### sampler_code (full)
```python
def sample(recs_x, recs_y, seed_pair, sim, rng):
    """Return candidate (a_id, b_id) pairs for the judge.

    recs_x: list of dicts — records of block X (table A side). Every
            dict has "id" plus dataset-specific text fields (products:
            title/manufacturer/price/...; bibliographic: title/authors/
            venue/year; restaurants: name/addr/city/...).
    recs_y: same for block Y (table B side).
    seed_pair: (a_id, b_id) — confirmed true match the blocker split.
    sim(a_id, b_id) -> float — char-3gram TF-IDF cosine similarity.
    rng: seeded random.Random — use for any randomness.
    The harness shows at most 250 pairs to the judge (fewer if the
    prompt character budget is hit) — pairs beyond that are dropped.
    """
    def text_of(r):
        return " ".join(str(v) for k, v in r.items() if k != "id" and isinstance(v, str)).lower()

    def tokens_of(r):
        return set(re.findall(r"[a-z0-9]+", text_of(r)))

    id_x = [r["id"] for r in recs_x]
    id_y = [r["id"] for r in recs_y]
    nx, ny = len(id_x), len(id_y)
    total = nx * ny
    MAX_SIM_CALLS = 8000
    MAX_RETURN = 250

    if total == 0:
        candidate_pairs = []
    elif total <= MAX_SIM_CALLS:
        candidate_pairs = [(a, b) for a in id_x for b in id_y]
    else:
        # cheap token-overlap prefilter — avoid calling sim() on the full
        # cross product for big blocks (can be 400 x 15000 records)
        tok_x = {r["id"]: tokens_of(r) for r in recs_x}
        tok_y = {r["id"]: tokens_of(r) for r in recs_y}

        index = defaultdict(list)
        max_posting = max(20, ny // 10)
        for bid, toks in tok_y.items():
            for t in toks:
                lst = index[t]
                if len(lst) < max_posting:
                    lst.append(bid)

        per_a_k = max(3, MAX_SIM_CALLS // max(1, nx))
        seen = set()
        candidate_pairs = []
        for aid, toks in tok_x.items():
            overlap = Counter()
            for t in toks:
                for bid in index.get(t, ()):
                    overlap[bid] += 1
            for bid, _ in overlap.most_common(per_a_k):
                key = (aid, bid)
                if key not in seen:
                    seen.add(key)
                    candidate_pairs.append(key)
            if len(candidate_pairs) >= MAX_SIM_CALLS:
                break

        if not candidate_pairs:
            # degenerate: no token overlap anywhere — fall back to a
            # bounded diagonal sample so sim() still has something to rank
            candidate_pairs = [(id_x[i], id_y[i % ny]) for i in range(min(nx, MAX_SIM_CALLS))]

    scored = []
    for a, b in candidate_pairs:
        try:
            s = sim(a, b)
        except Exception:
            s = 0.0
        scored.append((s, a, b))
    scored.sort(key=lambda t: -t[0])

    result = []
    seen_pairs = set()
    ax, bx = seed_pair
    if ax in id_x and bx in id_y:
        result.append((ax, bx))
        seen_pairs.add((ax, bx))
    elif bx in id_x and ax in id_y:
        result.append((bx, ax))
        seen_pairs.add((bx, ax))

    for s, a, b in scored:
        if (a, b) in seen_pairs:
            continue
        seen_pairs.add((a, b))
        result.append((a, b))
        if len(result) >= MAX_RETURN:
            break

    return result
```

### sampler_code diff vs parent #0 (truncated)
```diff
--- 
+++ 
@@ -14,4 +14,81 @@
     """
-    pairs = [(a["id"], b["id"]) for a in recs_x for b in recs_y]
-    pairs.sort()
-    return pairs
+    def text_of(r):
+        return " ".join(str(v) for k, v in r.items() if k != "id" and isinstance(v, str)).lower()
+
+    def tokens_of(r):
+        return set(re.findall(r"[a-z0-9]+", text_of(r)))
+
+    id_x = [r["id"] for r in recs_x]
+    id_y = [r["id"] for r in recs_y]
+    nx, ny = len(id_x), len(id_y)
+    total = nx * ny
+    MAX_SIM_CALLS = 8000
+    MAX_RETURN = 250
+
+    if total == 0:
+        candidate_pairs = []
+    elif total <= MAX_SIM_CALLS:
+        candidate_pairs = [(a, b) for a in id_x for b in id_y]
+    else:
+        # cheap token-overlap prefilter — avoid calling sim() on the full
+        # cross product for big blocks (can be 400 x 15000 records)
+        tok_x = {r["id"]: tokens_of(r) for r in recs_x}
+        tok_y = {r["id"]: tokens_of(r) for r in recs_y}
+
+        index = defaultdict(list)
+        max_posting = max(20, ny // 10)
+        for bid, toks in tok_y.items():
+            for t in toks:
+                lst = index[t]
+                if len(lst) < max_posting:
+                    lst.append(bid)
+
+        per_a_k = max(3, MAX_SIM_CALLS // max(1, nx))
+        seen = set()
+        candidate_pairs = []
+        for aid, toks in tok_x.items():
+            overlap = Counter()
+            for t in toks:
+                for bid in index.get(t, ()):
+                    overlap[bid] += 1
+            for bid, _ in overlap.most_common(per_a_k):
+                key = (aid, bid)
+                if key not in seen:
+                    seen.add(key)
+                    candidate_pairs.append(key)
+            if len(candidate_pairs) >= MAX_SIM_CALLS:
+                break
+
+        if not candidate_pairs:
+            # degenerate: no token overlap anywhere — fall back to a
+            # bounded diagonal sample so sim() still has something to rank
+            candidate_pairs = [(id_x[i], id_y[i % ny]) for i in range(min(nx, MAX_SIM_CALLS))]
+
+    scored = []
```

---

## Candidate #2  (val 0.281, discovered at rollout 222, parent #1, mutated: sampler_code)

**What its sampler actually retrieved on the reference episode:** 44 pairs shown to the judge, covering 2/6 hidden targets, list TRUNCATED at prompt budget.

Prompt excerpt as fed to the judge (first pairs):
```
1.
   A: title: punch! master landscape & home design | description: master landscape & home design offers nine programs on one easy-to-use interface to help you design home exter | manufacturer: punch! software | price: 59.99
   B: name: punch software 26100 punch! master landscape and home design (small box) | description: punch! master landscape & home design offers an extensive plantings database deck topography and new technolog | manufacturer: punch software | price: 45.99
2.
   A: title: pinnacle mobile media organizer | description: package contents: mobile media organizer cd media mobile master quick install guide pinnacle's mobile media or | manufacturer: pinnacle | price: 49.99
   B: name: pinnacle mobile media organizer software for windows ipod software | description: mobile media organizer software for windows the mobile media organizer software for windows from pinnacle allo | price: 39.95
3.
   A: title: punch! master landscape & home design | description: master landscape & home design offers nine programs on one easy-to-use interface to help you design home exter | manufacturer: punch! software | price: 59.99
   B: name: punch software 26100 punch! master landscape and home design (small box) | description: punch! master landscape & home design offers an extensive plantings database deck topography and new technolog | manufacturer: punch software | price: 45.99
4.
   A: title: dk rainbow fish most beautiful fish in the ocean | description: one stormy night while rainbow fish slept on the ocean floor his beautiful sparkling scales were stolen by thr | manufacturer: encore | price: 9.99
   B: name: rainbow fish and the whale | description: rainbow fish is trapped inside the whale and needs help getting out. in this adventure children will encounter | price: 6.95
5.
   A: title: project x project management software | description: project x is truly project management software for the rest of us. its for those of us that want to spend more | manufacturer: marware | price: 199.95
   B: name: microsoft(r) project standard 2007 | description: microsoft project standard 2007 helps you manage projects and to address the work and people management needs  | price: 599.95
6.
   A: title: punch! master landscape & home design | description: plan out your perfect home & garden with the set of applications available here! / for windows take a virtual  | manufacturer: punch! software | price: 99.99
   B: name: individual software inc total 3d home & landscape deluxe 9 | description: home and landscape design software designed for homeowners. whether you are decorating a room landscaping your | price: 33.5
7.
   A: title: adobe photoshop cs2 fundamental techniques by julieanne kost | description: learn essential skills that will become a foundation for a creative and efficient workflow. you will gain a co | manufacturer: software cinema | price: 0
   B: name: software cinema dvd-rom: dvdrom: photoshop cs2 advanced techniques (training) photoshop software | description: dvd-rom: photoshop cs2 advanced techniques (training) by julieanne kost software cinema - photoshop cs2 advanc | price: 159.95
8.
   A: title: instant play electric guitar (jewel box) | description: instant play electric guitar hits the right note for flexible computer-based instruction. this 2-cd set takes  | manufacturer: topics entertainment | price: 6.99
   B: name: jorma kaukonen homespun the electric guitar of jorma kaukonen - blues rock & roll and beyond | description: a formula for instant success on the electric guitar! jorma kaukonen teaches licks solos lead lines and rhythm | price: 22.13
9.
   A: title: pdf converter 4 | description: scansoft pdf converter 4 instantly converts pdf files into documents spreadsheets and forms that look exactly  | manufacturer: nuance | price: 49.95
   B: name: adobe acrobat 7.0 standard academic mac | description: transform pdf files into intelligent documents | price: 98.99
10.
   A: title: individual small business advantage deluxe 2006 | description: with small business advantage deluxe 2006 is a complete set of marketing tools for entrepreneurs & small busin | manufacturer: individual | price: 49.99
   B: name: encore software 10476 - encore home and business lawyer 2006 deluxe - complete product - legal reference - 1 u | description: encore software 10476 : protect your family and small business with home and business lawyer deluxe 2006. this | price: 17.97
11.
   A: title: instant landscaping 3.0 [lb] | description: instant landscape design 3.0 is the perfect tool for creating the perfect garden yard or landscape. draw the p | manufacturer: topics entertainment | price: 19.99
   B: name: punch software 25100 punch! master landscape and home design | description: punch! master landscape & home design offers an extensive plantings database deck topography and new technolog | manufacturer: punch software | price: 45.99
```

### instruction (full)
```
You are debugging the output of a blocking system for entity resolution. Two blocks are shown that should not have been separated for at least some records: a confirmed example of a true match that the blocker wrongly split across these two blocks is given as the SEED. You get a list of numbered candidate record pairs (one record from each block). Decide which candidate pairs are TRUE MATCHES, i.e. the two records refer to the same real-world product. Return JSON: {"match_indices": [<numbers of the true-match pairs>]}. Do not include the seed pair. If none, return an empty list.
```

### sampler_code (full)
```python
def sample(recs_x, recs_y, seed_pair, sim, rng):
    """Return candidate (a_id, b_id) pairs for the judge.

    recs_x: list of dicts — records of block X (table A side). Every
            dict has "id" plus dataset-specific text fields (products:
            title/manufacturer/price/...; bibliographic: title/authors/
            venue/year; restaurants: name/addr/city/...).
    recs_y: same for block Y (table B side).
    seed_pair: (a_id, b_id) — confirmed true match the blocker split.
    sim(a_id, b_id) -> float — char-3gram TF-IDF cosine similarity.
    rng: seeded random.Random — use for any randomness.
    The harness shows at most 250 pairs to the judge (fewer if the
    prompt character budget is hit) — pairs beyond that are dropped.
    """
    def text_of(r):
        return " ".join(str(v) for k, v in r.items() if k != "id" and isinstance(v, str)).lower()

    def tokens_of(r):
        return set(re.findall(r"[a-z0-9]+", text_of(r)))

    id_x = [r["id"] for r in recs_x]
    id_y = [r["id"] for r in recs_y]
    nx, ny = len(id_x), len(id_y)
    total = nx * ny
    MAX_SIM_CALLS = 8000
    MAX_RETURN = 250

    if total == 0:
        return []

    if total <= MAX_SIM_CALLS:
        candidate_pairs = [(a, b) for a in id_x for b in id_y]
    else:
        # cheap token-overlap prefilter — avoid calling sim() on the full
        # cross product for big blocks (can be 400 x 15000 records)
        tok_x = {r["id"]: tokens_of(r) for r in recs_x}
        tok_y = {r["id"]: tokens_of(r) for r in recs_y}

        index = defaultdict(list)
        max_posting = max(20, ny // 10)
        for bid, toks in tok_y.items():
            for t in toks:
                lst = index[t]
                if len(lst) < max_posting:
                    lst.append(bid)

        per_a_k = max(6, MAX_SIM_CALLS // max(1, nx))
        seen = set()
        candidate_pairs = []
        for aid, toks in tok_x.items():
            overlap = Counter()
            for t in toks:
                for bid in index.get(t, ()):
                    overlap[bid] += 1
            for bid, _ in overlap.most_common(per_a_k):
                key = (aid, bid)
                if key not in seen:
                    seen.add(key)
                    candidate_pairs.append(key)
            if len(candidate_pairs) >= MAX_SIM_CALLS:
                break

        if not candidate_pairs:
            # degenerate: no token overlap anywhere — fall back to a
            # bounded diagonal sample so sim() still has something to rank
            candidate_pairs = [(id_x[i], id_y[i % ny]) for i in range(min(nx, MAX_SIM_CALLS))]

    # make sure the confirmed seed pair is always scored, even if the
    # prefilter happened to drop it
    ax, bx = seed_pair
    seed = None
    if ax in id_x and bx in id_y:
        seed = (ax, bx)
    elif bx in id_x and ax in id_y:
        seed = (bx, ax)
    if seed is not None and seed not in candidate_pairs:
        candidate_pairs.append(seed)

    scores = {}
    for a, b in candidate_pairs:
        try:
            scores[(a, b)] = sim(a, b)
        except Exception:
            scores[(a, b)] = 0.0

    # per-row and per-column ranked partner lists
    ranked_by_a = defaultdict(list)
    ranked_by_b = defaultdict(list)
    for (a, b), s in scores.items():
        ranked_by_a[a].append((s, b))
        ranked_by_b[b].append((s, a))
    for a in ranked_by_a:
        ranked_by_a[a].sort(key=lambda t: -t[0])
    for b in ranked_by_b:
        ranked_by_b[b].sort(key=lambda t: -t[0])

    # visit rows/columns best-top-score-first (shuffle first to break ties
    # without bias), so the entities most likely to have a real match get
    # their slot before entities that clearly have nothing similar
    a_order = list(ranked_by_a.keys())
    b_order = list(ranked_by_b.keys())
    rng.shuffle(a_order)
    rng.shuffle(b_order)
    a_order.sort(key=lambda a: -ranked_by_a[a][0][0])
    b_order.sort(key=lambda b: -ranked_by_b[b][0][0])

    result = []
    seen_pairs = set()
    if seed is not None:
        result.append(seed)
        seen_pairs.add(seed)

    # breadth-first round robin: hand out each row's best partner, then
    # each column's best partner, before anyone gets a 2nd-best pair.
    # This is the key fix — a plain global sim-sort lets a handful of
    # near-duplicate pairs (e.g. product variants) burn most of the
    # char-budget-limited visible slots, pushing a *different* entity's
    # true-but-moderate-similarity match past the truncation cutoff
    # where it can never be recovered. Round-robin guarantees every
    # entity gets a shot at its best candidate first.
    max_rank = max((len(v) for v in ranked_by_a.values()), default=0)
    max_rank = max(max_rank, max((len(v) for v in ranked_by_b.values()), default=0))

    for rank in range(max_rank):
        for a in a_order:
            lst = ranked_by_a[a]
            if rank < len(lst):
                _, b = lst[rank]
                pair = (a, b)
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    result.append(pair)
        for b in b_order:
            lst = ranked_by_b[b]
            if rank < len(lst):
                _, a = lst[rank]
                pair = (a, b)
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    result.append(pair)
        if len(result) >= MAX_RETURN:
            break

    return result[:MAX_RETURN]
```

### sampler_code diff vs parent #1 (truncated)
```diff
--- 
+++ 
@@ -28,4 +28,5 @@
     if total == 0:
-        candidate_pairs = []
-    elif total <= MAX_SIM_CALLS:
+        return []
+
+    if total <= MAX_SIM_CALLS:
         candidate_pairs = [(a, b) for a in id_x for b in id_y]
@@ -45,3 +46,3 @@
 
-        per_a_k = max(3, MAX_SIM_CALLS // max(1, nx))
+        per_a_k = max(6, MAX_SIM_CALLS // max(1, nx))
         seen = set()
@@ -66,10 +67,40 @@
 
-    scored = []
+    # make sure the confirmed seed pair is always scored, even if the
+    # prefilter happened to drop it
+    ax, bx = seed_pair
+    seed = None
+    if ax in id_x and bx in id_y:
+        seed = (ax, bx)
+    elif bx in id_x and ax in id_y:
+        seed = (bx, ax)
+    if seed is not None and seed not in candidate_pairs:
+        candidate_pairs.append(seed)
+
+    scores = {}
     for a, b in candidate_pairs:
         try:
-            s = sim(a, b)
+            scores[(a, b)] = sim(a, b)
         except Exception:
-            s = 0.0
-        scored.append((s, a, b))
-    scored.sort(key=lambda t: -t[0])
+            scores[(a, b)] = 0.0
+
+    # per-row and per-column ranked partner lists
+    ranked_by_a = defaultdict(list)
+    ranked_by_b = defaultdict(list)
+    for (a, b), s in scores.items():
+        ranked_by_a[a].append((s, b))
+        ranked_by_b[b].append((s, a))
+    for a in ranked_by_a:
+        ranked_by_a[a].sort(key=lambda t: -t[0])
+    for b in ranked_by_b:
+        ranked_by_b[b].sort(key=lambda t: -t[0])
+
+    # visit rows/columns best-top-score-first (shuffle first to break ties
+    # without bias), so the entities most likely to have a real match get
+    # their slot before entities that clearly have nothing similar
+    a_order = list(ranked_by_a.keys())
+    b_order = list(ranked_by_b.keys())
+    rng.shuffle(a_order)
+    rng.shuffle(b_order)
+    a_order.sort(key=lambda a: -ranked_by_a[a][0][0])
+    b_order.sort(key=lambda b: -ranked_by_b[b][0][0])
```

---

## Candidate #6  (val 0.309, discovered at rollout 474, parent #1, mutated: sampler_code)

**What its sampler actually retrieved on the reference episode:** 25 pairs shown to the judge, covering 2/6 hidden targets.

Prompt excerpt as fed to the judge (first pairs):
```
1.
   A: title: punch! master landscape & home design | description: master landscape & home design offers nine programs on one easy-to-use interface to help you design home exter | manufacturer: punch! software | price: 59.99
   B: name: punch software 26100 punch! master landscape and home design (small box) | description: punch! master landscape & home design offers an extensive plantings database deck topography and new technolog | manufacturer: punch software | price: 45.99
2.
   A: title: pinnacle mobile media organizer | description: package contents: mobile media organizer cd media mobile master quick install guide pinnacle's mobile media or | manufacturer: pinnacle | price: 49.99
   B: name: pinnacle mobile media organizer software for windows ipod software | description: mobile media organizer software for windows the mobile media organizer software for windows from pinnacle allo | price: 39.95
3.
   A: title: punch! master landscape & home design | description: master landscape & home design offers nine programs on one easy-to-use interface to help you design home exter | manufacturer: punch! software | price: 59.99
   B: name: punch software 26100 punch! master landscape and home design (small box) | description: punch! master landscape & home design offers an extensive plantings database deck topography and new technolog | manufacturer: punch software | price: 45.99
4.
   A: title: dk rainbow fish most beautiful fish in the ocean | description: one stormy night while rainbow fish slept on the ocean floor his beautiful sparkling scales were stolen by thr | manufacturer: encore | price: 9.99
   B: name: rainbow fish and the whale | description: rainbow fish is trapped inside the whale and needs help getting out. in this adventure children will encounter | price: 6.95
5.
   A: title: project x project management software | description: project x is truly project management software for the rest of us. its for those of us that want to spend more | manufacturer: marware | price: 199.95
   B: name: microsoft(r) project standard 2007 | description: microsoft project standard 2007 helps you manage projects and to address the work and people management needs  | price: 599.95
6.
   A: title: punch! master landscape & home design | description: plan out your perfect home & garden with the set of applications available here! / for windows take a virtual  | manufacturer: punch! software | price: 99.99
   B: name: individual software inc total 3d home & landscape deluxe 9 | description: home and landscape design software designed for homeowners. whether you are decorating a room landscaping your | price: 33.5
7.
   A: title: adobe photoshop cs2 fundamental techniques by julieanne kost | description: learn essential skills that will become a foundation for a creative and efficient workflow. you will gain a co | manufacturer: software cinema | price: 0
   B: name: software cinema dvd-rom: dvdrom: photoshop cs2 advanced techniques (training) photoshop software | description: dvd-rom: photoshop cs2 advanced techniques (training) by julieanne kost software cinema - photoshop cs2 advanc | price: 159.95
8.
   A: title: instant play electric guitar (jewel box) | description: instant play electric guitar hits the right note for flexible computer-based instruction. this 2-cd set takes  | manufacturer: topics entertainment | price: 6.99
   B: name: jorma kaukonen homespun the electric guitar of jorma kaukonen - blues rock & roll and beyond | description: a formula for instant success on the electric guitar! jorma kaukonen teaches licks solos lead lines and rhythm | price: 22.13
9.
   A: title: pdf converter 4 | description: scansoft pdf converter 4 instantly converts pdf files into documents spreadsheets and forms that look exactly  | manufacturer: nuance | price: 49.95
   B: name: adobe acrobat 7.0 standard academic mac | description: transform pdf files into intelligent documents | price: 98.99
10.
   A: title: individual small business advantage deluxe 2006 | description: with small business advantage deluxe 2006 is a complete set of marketing tools for entrepreneurs & small busin | manufacturer: individual | price: 49.99
   B: name: encore software 10476 - encore home and business lawyer 2006 deluxe - complete product - legal reference - 1 u | description: encore software 10476 : protect your family and small business with home and business lawyer deluxe 2006. this | price: 17.97
11.
   A: title: instant landscaping 3.0 [lb] | description: instant landscape design 3.0 is the perfect tool for creating the perfect garden yard or landscape. draw the p | manufacturer: topics entertainment | price: 19.99
   B: name: punch software 25100 punch! master landscape and home design | description: punch! master landscape & home design offers an extensive plantings database deck topography and new technolog | manufacturer: punch software | price: 45.99
```

### instruction (full)
```
You are debugging the output of a blocking system for entity resolution. Two blocks are shown that should not have been separated for at least some records: a confirmed example of a true match that the blocker wrongly split across these two blocks is given as the SEED. You get a list of numbered candidate record pairs (one record from each block). Decide which candidate pairs are TRUE MATCHES, i.e. the two records refer to the same real-world product. Return JSON: {"match_indices": [<numbers of the true-match pairs>]}. Do not include the seed pair. If none, return an empty list.
```

### sampler_code (full)
```python
def sample(recs_x, recs_y, seed_pair, sim, rng):
    """Return candidate (a_id, b_id) pairs for the judge.

    recs_x: list of dicts — records of block X (table A side). Every
            dict has "id" plus dataset-specific text fields (products:
            title/manufacturer/price/...; bibliographic: title/authors/
            venue/year; restaurants: name/addr/city/...).
    recs_y: same for block Y (table B side).
    seed_pair: (a_id, b_id) — confirmed true match the blocker split.
    sim(a_id, b_id) -> float — char-3gram TF-IDF cosine similarity.
    rng: seeded random.Random — use for any randomness.
    The harness shows at most 250 pairs to the judge (fewer if the
    prompt character budget is hit) — pairs beyond that are dropped.
    """
    def text_of(r):
        return " ".join(str(v) for k, v in r.items() if k != "id" and isinstance(v, str)).lower()

    def tokens_of(text):
        return set(re.findall(r"[a-z0-9]+", text))

    text_x = {r["id"]: text_of(r) for r in recs_x}
    text_y = {r["id"]: text_of(r) for r in recs_y}
    tok_x = {aid: tokens_of(t) for aid, t in text_x.items()}
    tok_y = {bid: tokens_of(t) for bid, t in text_y.items()}

    id_x = list(text_x.keys())
    id_y = list(text_y.keys())
    nx, ny = len(id_x), len(id_y)
    total = nx * ny
    MAX_SIM_CALLS = 8000
    MAX_RETURN = 250
    # harness hard-truncates rendered prompt at 20000 chars; leave margin
    # so every pair we return actually survives the render, none wasted
    # past a cutoff we can't see.
    CHAR_BUDGET = 17000

    if total == 0:
        candidate_pairs = []
    elif total <= MAX_SIM_CALLS:
        candidate_pairs = [(a, b) for a in id_x for b in id_y]
    else:
        # cheap token-overlap prefilter — avoid calling sim() on the full
        # cross product for big blocks (can be 400 x 15000 records)
        index = defaultdict(list)
        max_posting = max(20, ny // 10)
        for bid, toks in tok_y.items():
            for t in toks:
                lst = index[t]
                if len(lst) < max_posting:
                    lst.append(bid)

        per_a_k = max(3, MAX_SIM_CALLS // max(1, nx))
        seen = set()
        candidate_pairs = []
        for aid, toks in tok_x.items():
            overlap = Counter()
            for t in toks:
                for bid in index.get(t, ()):
                    overlap[bid] += 1
            for bid, _ in overlap.most_common(per_a_k):
                key = (aid, bid)
                if key not in seen:
                    seen.add(key)
                    candidate_pairs.append(key)
            if len(candidate_pairs) >= MAX_SIM_CALLS:
                break

        if not candidate_pairs:
            # degenerate: no token overlap anywhere — fall back to a
            # bounded diagonal sample so sim() still has something to rank
            candidate_pairs = [(id_x[i], id_y[i % ny]) for i in range(min(nx, MAX_SIM_CALLS))]

    scored = []
    for a, b in candidate_pairs:
        try:
            s = sim(a, b)
        except Exception:
            s = 0.0
        scored.append((s, a, b))

    # group by a-id: a single a-id with many high-sim candidates must not
    # crowd out other a-ids' true pairs before the char budget runs out
    by_a = defaultdict(list)
    for s, a, b in scored:
        by_a[a].append((s, b))
    for a in by_a:
        by_a[a].sort(key=lambda t: -t[0])

    # visit a-ids in order of their own best candidate, round-robin one
    # candidate per a-id per round — spreads coverage across distinct
    # a-ids instead of one global top-N dominated by a single a
    a_order = sorted(by_a.keys(), key=lambda a: -by_a[a][0][0])

    result = []
    seen_pairs = set()
    used_chars = 0

    def try_add(a, b):
        nonlocal used_chars
        if (a, b) in seen_pairs:
            return False
        cost = len(text_x.get(a, "")) + len(text_y.get(b, "")) + 40
        if result and used_chars + cost > CHAR_BUDGET:
            return False
        seen_pairs.add((a, b))
        result.append((a, b))
        used_chars += cost
        return True

    ax, bx = seed_pair
    if ax in text_x and bx in text_y:
        try_add(ax, bx)
    elif bx in text_x and ax in text_y:
        try_add(bx, ax)

    ptrs = {a: 0 for a in a_order}
    progress = True
    while progress and len(result) < MAX_RETURN:
        progress = False
        for a in a_order:
            i = ptrs[a]
            lst = by_a[a]
            if i >= len(lst):
                continue
            s, b = lst[i]
            ptrs[a] = i + 1
            if try_add(a, b):
                progress = True
            if len(result) >= MAX_RETURN:
                break

    return result
```

### sampler_code diff vs parent #1 (truncated)
```diff
--- 
+++ 
@@ -17,7 +17,12 @@
 
-    def tokens_of(r):
-        return set(re.findall(r"[a-z0-9]+", text_of(r)))
+    def tokens_of(text):
+        return set(re.findall(r"[a-z0-9]+", text))
 
-    id_x = [r["id"] for r in recs_x]
-    id_y = [r["id"] for r in recs_y]
+    text_x = {r["id"]: text_of(r) for r in recs_x}
+    text_y = {r["id"]: text_of(r) for r in recs_y}
+    tok_x = {aid: tokens_of(t) for aid, t in text_x.items()}
+    tok_y = {bid: tokens_of(t) for bid, t in text_y.items()}
+
+    id_x = list(text_x.keys())
+    id_y = list(text_y.keys())
     nx, ny = len(id_x), len(id_y)
@@ -26,2 +31,6 @@
     MAX_RETURN = 250
+    # harness hard-truncates rendered prompt at 20000 chars; leave margin
+    # so every pair we return actually survives the render, none wasted
+    # past a cutoff we can't see.
+    CHAR_BUDGET = 17000
 
@@ -34,5 +43,2 @@
         # cross product for big blocks (can be 400 x 15000 records)
-        tok_x = {r["id"]: tokens_of(r) for r in recs_x}
-        tok_y = {r["id"]: tokens_of(r) for r in recs_y}
-
         index = defaultdict(list)
@@ -73,3 +79,15 @@
         scored.append((s, a, b))
-    scored.sort(key=lambda t: -t[0])
+
+    # group by a-id: a single a-id with many high-sim candidates must not
+    # crowd out other a-ids' true pairs before the char budget runs out
+    by_a = defaultdict(list)
+    for s, a, b in scored:
+        by_a[a].append((s, b))
+    for a in by_a:
+        by_a[a].sort(key=lambda t: -t[0])
+
+    # visit a-ids in order of their own best candidate, round-robin one
+    # candidate per a-id per round — spreads coverage across distinct
+    # a-ids instead of one global top-N dominated by a single a
+    a_order = sorted(by_a.keys(), key=lambda a: -by_a[a][0][0])
 
@@ -77,17 +95,37 @@
     seen_pairs = set()
-    ax, bx = seed_pair
-    if ax in id_x and bx in id_y:
-        result.append((ax, bx))
-        seen_pairs.add((ax, bx))
-    elif bx in id_x and ax in id_y:
-        result.append((bx, ax))
-        seen_pairs.add((bx, ax))
+    used_chars = 0
 
```

---

## Candidate #13  (val 0.425, discovered at rollout 947, parent #5, mutated: sampler_code)

**What its sampler actually retrieved on the reference episode:** 17 pairs shown to the judge, covering 3/6 hidden targets.

Prompt excerpt as fed to the judge (first pairs):
```
1.
   A: title: punch! master landscape & home design | description: master landscape & home design offers nine programs on one easy-to-use interface to help you design home exter | manufacturer: punch! software | price: 59.99
   B: name: punch software 26100 punch! master landscape and home design (small box) | description: punch! master landscape & home design offers an extensive plantings database deck topography and new technolog | manufacturer: punch software | price: 45.99
2.
   A: title: punch! master landscape & home design | description: master landscape & home design offers nine programs on one easy-to-use interface to help you design home exter | manufacturer: punch! software | price: 59.99
   B: name: punch software 26100 punch! master landscape and home design (small box) | description: punch! master landscape & home design offers an extensive plantings database deck topography and new technolog | manufacturer: punch software | price: 45.99
3.
   A: title: punch! master landscape & home design | description: master landscape & home design offers nine programs on one easy-to-use interface to help you design home exter | manufacturer: punch! software | price: 59.99
   B: name: punch software 26100 punch! master landscape and home design (small box) | description: punch! master landscape & home design offers an extensive plantings database deck topography and new technolog | manufacturer: punch software | price: 45.99
4.
   A: title: punch! master landscape & home design | description: master landscape & home design offers nine programs on one easy-to-use interface to help you design home exter | manufacturer: punch! software | price: 59.99
   B: name: punch software 25100 punch! master landscape and home design | description: punch! master landscape & home design offers an extensive plantings database deck topography and new technolog | manufacturer: punch software | price: 45.99
5.
   A: title: punch! master landscape & home design | description: master landscape & home design offers nine programs on one easy-to-use interface to help you design home exter | manufacturer: punch! software | price: 59.99
   B: name: punch software 25100 punch! master landscape and home design | description: punch! master landscape & home design offers an extensive plantings database deck topography and new technolog | manufacturer: punch software | price: 45.99
6.
   A: title: punch! master landscape & home design | description: plan out your perfect home & garden with the set of applications available here! / for windows take a virtual  | manufacturer: punch! software | price: 99.99
   B: name: punch software 26100 punch! master landscape and home design (small box) | description: punch! master landscape & home design offers an extensive plantings database deck topography and new technolog | manufacturer: punch software | price: 45.99
7.
   A: title: instant home design 3.0 [lb] | description: software suite helps you create customized before & after visuals of your home renovation in both photographic | manufacturer: topics entertainment | price: 19.99
   B: name: punch software 26100 punch! master landscape and home design (small box) | description: punch! master landscape & home design offers an extensive plantings database deck topography and new technolog | manufacturer: punch software | price: 45.99
8.
   A: title: instant landscaping 3.0 [lb] | description: instant landscape design 3.0 is the perfect tool for creating the perfect garden yard or landscape. draw the p | manufacturer: topics entertainment | price: 19.99
   B: name: punch software 26100 punch! master landscape and home design (small box) | description: punch! master landscape & home design offers an extensive plantings database deck topography and new technolog | manufacturer: punch software | price: 45.99
9.
   A: title: instant home design (jewel case) | description: instant home design (win 95 98 me nt 2000 xp) | manufacturer: topics entertainment | price: 9.99
   B: name: punch software 26100 punch! master landscape and home design (small box) | description: punch! master landscape & home design offers an extensive plantings database deck topography and new technolog | manufacturer: punch software | price: 45.99
10.
   A: title: pinnacle mobile media organizer | description: package contents: mobile media organizer cd media mobile master quick install guide pinnacle's mobile media or | manufacturer: pinnacle | price: 49.99
   B: name: pinnacle mobile media organizer software for windows ipod software | description: mobile media organizer software for windows the mobile media organizer software for windows from pinnacle allo | price: 39.95
11.
   A: title: dk rainbow fish most beautiful fish in the ocean | description: one stormy night while rainbow fish slept on the ocean floor his beautiful sparkling scales were stolen by thr | manufacturer: encore | price: 9.99
   B: name: rainbow fish and the whale | description: rainbow fish is trapped inside the whale and needs help getting out. in this adventure children will encounter | price: 6.95
```

### instruction (full)
```
You are debugging the output of a blocking system for entity resolution. Two blocks are shown that should not have been separated for at least some records: a confirmed example of a true match that the blocker wrongly split across these two blocks is given as the SEED. You get a list of numbered candidate record pairs (one record from each block). Decide which candidate pairs are TRUE MATCHES, i.e. the two records refer to the same real-world product. Return JSON: {"match_indices": [<numbers of the true-match pairs>]}. Do not include the seed pair. If none, return an empty list.
```

### sampler_code (full)
```python
def sample(recs_x, recs_y, seed_pair, sim, rng):
    """Return candidate (a_id, b_id) pairs for the judge.

    recs_x: list of dicts — records of block X (table A side). Every
            dict has "id" plus dataset-specific text fields (products:
            title/manufacturer/price/...; bibliographic: title/authors/
            venue/year; restaurants: name/addr/city/...).
    recs_y: same for block Y (table B side).
    seed_pair: (a_id, b_id) — confirmed true match the blocker split.
    sim(a_id, b_id) -> float — char-3gram TF-IDF cosine similarity.
    rng: seeded random.Random — use for any randomness.
    The harness shows at most 250 pairs to the judge (fewer if the
    prompt character budget is hit) — pairs beyond that are dropped.
    """
    def text_of(r):
        return " ".join(str(v) for k, v in r.items() if k != "id" and isinstance(v, str)).lower()

    def tokens_of(r):
        return set(re.findall(r"[a-z0-9]+", text_of(r)))

    def rec_render_len(r):
        return sum(len(str(v)) for v in r.values())

    id_x = [r["id"] for r in recs_x]
    id_y = [r["id"] for r in recs_y]
    nx, ny = len(id_x), len(id_y)
    total = nx * ny
    MAX_SIM_CALLS = 8000
    HARD_MAX_RETURN = 250

    if total == 0:
        return []

    # ---- budget-aware cap: estimate chars/pair from ACTUAL record text
    # length so we don't blindly hand back 250 pairs when the render
    # truncates at ~10-50 on long-text datasets (walmart-amazon: titles
    # + descriptions routinely blow the 20k prompt budget after a
    # handful of pairs). Sizing the return list to what can actually be
    # SHOWN beats over-supplying and hoping list order survives an
    # unknown truncation point.
    sample_x = recs_x if nx <= 200 else [recs_x[i] for i in range(0, nx, max(1, nx // 200))]
    sample_y = recs_y if ny <= 200 else [recs_y[i] for i in range(0, ny, max(1, ny // 200))]
    avg_len_x = sum(rec_render_len(r) for r in sample_x) / max(1, len(sample_x))
    avg_len_y = sum(rec_render_len(r) for r in sample_y) / max(1, len(sample_y))
    EFFECTIVE_BUDGET = 14000  # headroom under 20k for instructions/schema/labels
    per_pair_cost = (avg_len_x + avg_len_y) * 1.15 + 30
    MAX_RETURN = max(15, min(HARD_MAX_RETURN, int(EFFECTIVE_BUDGET / max(per_pair_cost, 1))))

    if total <= MAX_SIM_CALLS:
        candidate_pairs = [(a, b) for a in id_x for b in id_y]
    else:
        # cheap token-overlap prefilter — avoid calling sim() on the full
        # cross product for big blocks (can be 400 x 15000 records)
        tok_x = {r["id"]: tokens_of(r) for r in recs_x}
        tok_y = {r["id"]: tokens_of(r) for r in recs_y}

        # document frequency of each y-side token, computed BEFORE capping
        # postings — used to down-weight generic words so rare/distinctive
        # tokens (model numbers, exact editions) drive candidate selection
        # instead of being drowned out by raw overlap counts.
        df = Counter()
        for toks in tok_y.values():
            for t in toks:
                df[t] += 1

        index = defaultdict(list)
        max_posting = max(20, ny // 10)
        for bid, toks in tok_y.items():
            for t in toks:
                lst = index[t]
                if len(lst) < max_posting:
                    lst.append(bid)

        per_a_k = max(6, MAX_SIM_CALLS // max(1, nx))
        seen = set()
        candidate_pairs = []

        # process the seed's a-record FIRST so it never gets starved by
        # an early MAX_SIM_CALLS break — its neighborhood (other listings
        # of the same entity) is the highest-value source of additional
        # true pairs beyond the seed itself.
        ax0, bx0 = seed_pair
        seed_a_first = ax0 if ax0 in tok_x else (bx0 if bx0 in tok_x else None)
        order = list(tok_x.keys())
        if seed_a_first is not None:
            order.remove(seed_a_first)
            order.insert(0, seed_a_first)

        for aid in order:
            toks = tok_x[aid]
            overlap = Counter()
            for t in toks:
                w = 1.0 / df[t] if df.get(t) else 0.0
                for bid in index.get(t, ()):
                    overlap[bid] += w
            for bid, _ in overlap.most_common(per_a_k):
                key = (aid, bid)
                if key not in seen:
                    seen.add(key)
                    candidate_pairs.append(key)
            if len(candidate_pairs) >= MAX_SIM_CALLS:
                break

        if not candidate_pairs:
            # degenerate: no token overlap anywhere — fall back to a
            # bounded diagonal sample so sim() still has something to rank
            candidate_pairs = [(id_x[i], id_y[i % ny]) for i in range(min(nx, MAX_SIM_CALLS))]

    # make sure the confirmed seed pair is always scored, even if the
    # prefilter happened to drop it
    ax, bx = seed_pair
    seed = None
    if ax in id_x and bx in id_y:
        seed = (ax, bx)
    elif bx in id_x and ax in id_y:
        seed = (bx, ax)
    if seed is not None and seed not in candidate_pairs:
        candidate_pairs.append(seed)

    scores = {}
    for a, b in candidate_pairs:
        try:
            scores[(a, b)] = sim(a, b)
        except Exception:
            scores[(a, b)] = 0.0

    # per-row and per-column ranked partner lists
    ranked_by_a = defaultdict(list)
    ranked_by_b = defaultdict(list)
    for (a, b), s in scores.items():
        ranked_by_a[a].append((s, b))
        ranked_by_b[b].append((s, a))
    for a in ranked_by_a:
        ranked_by_a[a].sort(key=lambda t: -t[0])
    for b in ranked_by_b:
        ranked_by_b[b].sort(key=lambda t: -t[0])

    result = []
    seen_pairs = set()
    if seed is not None:
        result.append(seed)
        seen_pairs.add(seed)

        # seed's row/column neighbors — other B candidates for the same
        # A entity and other A candidates for the same B entity. These
        # are the cheapest, highest-probability source of ADDITIONAL true
        # pairs (duplicate listings of the entity that was already
        # confirmed to match) and cost only two extra small lookups, so
        # they go in right after the seed, ahead of generic ranking.
        sa, sb = seed
        NEIGHBOR_K = 5
        for s, b in ranked_by_a.get(sa, [])[:NEIGHBOR_K]:
            pair = (sa, b)
            if pair not in seen_pairs and len(result) < MAX_RETURN:
                seen_pairs.add(pair)
                result.append(pair)
        for s, a in ranked_by_b.get(sb, [])[:NEIGHBOR_K]:
            pair = (a, sb)
            if pair not in seen_pairs and len(result) < MAX_RETURN:
                seen_pairs.add(pair)
                result.append(pair)

    # mutual top-1 pairs — a's best partner is b AND b's best partner is
    # a. Strong true-match signal independent of row/column rank order;
    # front-loaded right after the seed (+ its neighbors) so it survives
    # even when the prompt char budget only fits a handful of pairs.
    mutual = []
    for a, lst in ranked_by_a.items():
        if not lst:
            continue
        s, b = lst[0]
        blst = ranked_by_b.get(b)
        if blst and blst[0][1] == a:
            pair = (a, b)
            if pair not in seen_pairs:
                mutual.append((s, pair))
    mutual.sort(key=lambda t: -t[0])
    for s, pair in mutual:
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            result.append(pair)
        if len(result) >= MAX_RETURN:
            break

    # visit rows/columns best-top-score-first (shuffle first to break ties
    # without bias), so the entities most likely to have a real match get
    # their slot before entities that clearly have nothing similar
    a_order = list(ranked_by_a.keys())
    b_order = list(ranked_by_b.keys())
    rng.shuffle(a_order)
    rng.shuffle(b_order)
    a_order.sort(key=lambda a: -ranked_by_a[a][0][0])
    b_order.sort(key=lambda b: -ranked_by_b[b][0][0])

    # zippered round robin: at each rank, alternate ONE pair at a time
    # between the a-side and b-side orders (not a full block-dump of one
    # side before the other) so both sides get slots even when only the
    # first handful of pairs survive truncation.
    max_rank = max((len(v) for v in ranked_by_a.values()), default=0)
    max_rank = max(max_rank, max((len(v) for v in ranked_by_b.values()), default=0))

    for rank in range(max_rank):
        if len(result) >= MAX_RETURN:
            break
        ai = bi = 0
        while ai < len(a_order) or bi < len(b_order):
            if ai < len(a_order):
                a = a_order[ai]
                ai += 1
                lst = ranked_by_a[a]
                if rank < len(lst):
                    _, b = lst[rank]
                    pair = (a, b)
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        result.append(pair)
                        if len(result) >= MAX_RETURN:
                            break
            if bi < len(b_order):
                b = b_order[bi]
                bi += 1
                lst = ranked_by_b[b]
                if rank < len(lst):
                    _, a = lst[rank]
                    pair = (a, b)
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        result.append(pair)
                        if len(result) >= MAX_RETURN:
                            break
            if len(result) >= MAX_RETURN:
                break

    return result[:MAX_RETURN]
```

### sampler_code diff vs parent #5 (truncated)
```diff
--- 
+++ 
@@ -20,2 +20,5 @@
 
+    def rec_render_len(r):
+        return sum(len(str(v)) for v in r.values())
+
     id_x = [r["id"] for r in recs_x]
@@ -25,3 +28,3 @@
     MAX_SIM_CALLS = 8000
-    MAX_RETURN = 250
+    HARD_MAX_RETURN = 250
 
@@ -29,2 +32,17 @@
         return []
+
+    # ---- budget-aware cap: estimate chars/pair from ACTUAL record text
+    # length so we don't blindly hand back 250 pairs when the render
+    # truncates at ~10-50 on long-text datasets (walmart-amazon: titles
+    # + descriptions routinely blow the 20k prompt budget after a
+    # handful of pairs). Sizing the return list to what can actually be
+    # SHOWN beats over-supplying and hoping list order survives an
+    # unknown truncation point.
+    sample_x = recs_x if nx <= 200 else [recs_x[i] for i in range(0, nx, max(1, nx // 200))]
+    sample_y = recs_y if ny <= 200 else [recs_y[i] for i in range(0, ny, max(1, ny // 200))]
+    avg_len_x = sum(rec_render_len(r) for r in sample_x) / max(1, len(sample_x))
+    avg_len_y = sum(rec_render_len(r) for r in sample_y) / max(1, len(sample_y))
+    EFFECTIVE_BUDGET = 14000  # headroom under 20k for instructions/schema/labels
+    per_pair_cost = (avg_len_x + avg_len_y) * 1.15 + 30
+    MAX_RETURN = max(15, min(HARD_MAX_RETURN, int(EFFECTIVE_BUDGET / max(per_pair_cost, 1))))
 
@@ -39,5 +57,4 @@
         # document frequency of each y-side token, computed BEFORE capping
-        # postings — used to down-weight generic words (e.g. "adobe",
-        # "acrobat" in a software catalog) so rare/distinctive tokens
-        # (model numbers, exact editions) drive candidate selection
+        # postings — used to down-weight generic words so rare/distinctive
+        # tokens (model numbers, exact editions) drive candidate selection
         # instead of being drowned out by raw overlap counts.
@@ -59,3 +76,16 @@
         candidate_pairs = []
-        for aid, toks in tok_x.items():
+
+        # process the seed's a-record FIRST so it never gets starved by
+        # an early MAX_SIM_CALLS break — its neighborhood (other listings
+        # of the same entity) is the highest-value source of additional
+        # true pairs beyond the seed itself.
+        ax0, bx0 = seed_pair
+        seed_a_first = ax0 if ax0 in tok_x else (bx0 if bx0 in tok_x else None)
+        order = list(tok_x.keys())
+        if seed_a_first is not None:
+            order.remove(seed_a_first)
+            order.insert(0, seed_a_first)
+
+        for aid in order:
+            toks = tok_x[aid]
             overlap = Counter()
@@ -113,7 +143,25 @@
 
+        # seed's row/column neighbors — other B candidates for the same
```

---

## Candidate #54  (val 0.440, discovered at rollout 4102, parent #32, mutated: sampler_code)

**What its sampler actually retrieved on the reference episode:** 12 pairs shown to the judge, covering 2/6 hidden targets.

Prompt excerpt as fed to the judge (first pairs):
```
1.
   A: title: punch! master landscape & home design | description: master landscape & home design offers nine programs on one easy-to-use interface to help you design home exter | manufacturer: punch! software | price: 59.99
   B: name: punch software 26100 punch! master landscape and home design (small box) | description: punch! master landscape & home design offers an extensive plantings database deck topography and new technolog | manufacturer: punch software | price: 45.99
2.
   A: title: punch! master landscape & home design | description: master landscape & home design offers nine programs on one easy-to-use interface to help you design home exter | manufacturer: punch! software | price: 59.99
   B: name: punch software 26100 punch! master landscape and home design (small box) | description: punch! master landscape & home design offers an extensive plantings database deck topography and new technolog | manufacturer: punch software | price: 45.99
3.
   A: title: punch! master landscape & home design | description: plan out your perfect home & garden with the set of applications available here! / for windows take a virtual  | manufacturer: punch! software | price: 99.99
   B: name: punch software 26100 punch! master landscape and home design (small box) | description: punch! master landscape & home design offers an extensive plantings database deck topography and new technolog | manufacturer: punch software | price: 45.99
4.
   A: title: pinnacle mobile media organizer | description: package contents: mobile media organizer cd media mobile master quick install guide pinnacle's mobile media or | manufacturer: pinnacle | price: 49.99
   B: name: pinnacle mobile media organizer software for windows ipod software | description: mobile media organizer software for windows the mobile media organizer software for windows from pinnacle allo | price: 39.95
5.
   A: title: dk rainbow fish most beautiful fish in the ocean | description: one stormy night while rainbow fish slept on the ocean floor his beautiful sparkling scales were stolen by thr | manufacturer: encore | price: 9.99
   B: name: rainbow fish and the whale | description: rainbow fish is trapped inside the whale and needs help getting out. in this adventure children will encounter | price: 6.95
6.
   A: title: project x project management software | description: project x is truly project management software for the rest of us. its for those of us that want to spend more | manufacturer: marware | price: 199.95
   B: name: microsoft(r) project standard 2007 | description: microsoft project standard 2007 helps you manage projects and to address the work and people management needs  | price: 599.95
7.
   A: title: adobe photoshop cs2 fundamental techniques by julieanne kost | description: learn essential skills that will become a foundation for a creative and efficient workflow. you will gain a co | manufacturer: software cinema | price: 0
   B: name: software cinema dvd-rom: dvdrom: photoshop cs2 advanced techniques (training) photoshop software | description: dvd-rom: photoshop cs2 advanced techniques (training) by julieanne kost software cinema - photoshop cs2 advanc | price: 159.95
8.
   A: title: instant play electric guitar (jewel box) | description: instant play electric guitar hits the right note for flexible computer-based instruction. this 2-cd set takes  | manufacturer: topics entertainment | price: 6.99
   B: name: jorma kaukonen homespun the electric guitar of jorma kaukonen - blues rock & roll and beyond | description: a formula for instant success on the electric guitar! jorma kaukonen teaches licks solos lead lines and rhythm | price: 22.13
9.
   A: title: pdf converter 4 | description: scansoft pdf converter 4 instantly converts pdf files into documents spreadsheets and forms that look exactly  | manufacturer: nuance | price: 49.95
   B: name: adobe acrobat 7.0 standard academic mac | description: transform pdf files into intelligent documents | price: 98.99
10.
   A: title: individual small business advantage deluxe 2006 | description: with small business advantage deluxe 2006 is a complete set of marketing tools for entrepreneurs & small busin | manufacturer: individual | price: 49.99
   B: name: encore software 10476 - encore home and business lawyer 2006 deluxe - complete product - legal reference - 1 u | description: encore software 10476 : protect your family and small business with home and business lawyer deluxe 2006. this | price: 17.97
11.
   A: title: instant play electric guitar deluxe | description: with play electric guitar beginners who want a fast flexible computer-based teaching tool have the in-depth in | manufacturer: topics entertainment | price: 39.99
   B: name: frederic hand homespun classical guitar - dvd | description: the dvd lesson frederic hand has made for homespun should be of interest to all guitarists who want to discove | price: 22.13
```

### instruction (full)
```
You are debugging the output of a blocking system for entity resolution. Two blocks are shown that should not have been separated for at least some records: a confirmed example of a true match that the blocker wrongly split across these two blocks is given as the SEED. You get a list of numbered candidate record pairs (one record from each block, drawn from the same two blocks as the SEED).

Decide which candidate pairs are TRUE MATCHES: the two records refer to the same specific real-world entity (the same product, the same publication, the same restaurant, etc.), not merely a similar or related one.

How to judge each pair:

1. Use the SEED only to calibrate what kind of surface variation the blocker is prone to (word order, abbreviations, formatting, extra boilerplate, differing source/vendor). Do not treat "same brand/manufacturer as the seed" or "same block" as evidence of a match on its own — every candidate is judged independently against every other candidate, not against the seed's identity.

2. Separate noise from signal in the text:
   - Treat as noise (ignore when comparing): reseller/distributor/publisher prefixes or suffixes, catalog/SKU/part numbers, packaging or license descriptors (e.g. "CD/DVD", "OEM", "no box"), currency/price differences, punctuation, word order, and boilerplate marketing text.
   - Treat as signal (must match, or the pair is NOT a match): the core identity of the entity — specific product line + edition/version/tier (e.g. Home vs Professional, Upgrade vs Full, Standard vs Plus/Deluxe), model or part number when it is the primary identifier, and any other attribute that distinguishes one real-world item from another close relative in the same family (e.g. a different edition of the same software, a different chapter/volume of the same publication, a different location of the same restaurant chain).
   - A rebrand, reseller listing, or re-release of the identical underlying item under a different storefront name/prefix is still a match — do not reject solely because the surface text differs in vendor framing.
   - Sharing a manufacturer, category, generic keyword, or price range is not sufficient evidence by itself; require overlap on the specific identifying attributes above.

3. When two records could plausibly be the same entity but a key distinguishing attribute (edition, version, model, size/capacity, location) is missing from one side, do not guess — only mark it a match if nothing you can see contradicts it and the specific identifiers that are present agree.

4. Consider every candidate pair shown, even ones that look like a "near miss" of the seed's product family — do not anchor only on the first few or the most seed-similar entries. Do not default to an empty list out of caution when a candidate's identifying attributes genuinely line up; equally, do not include a pair just because it resembles other accepted pairs in the same batch.

Return JSON: {"match_indices": [<numbers of the true-match pairs, using only the numbers shown in the candidate list>]}. Do not include the seed pair. If none of the candidates are true matches, return an empty list.
```

### sampler_code (full)
```python
def sample(recs_x, recs_y, seed_pair, sim, rng):
    """Return candidate (a_id, b_id) pairs for the judge.

    recs_x: list of dicts — records of block X (table A side). Every
            dict has "id" plus dataset-specific text fields (products:
            title/manufacturer/price/...; bibliographic: title/authors/
            venue/year; restaurants: name/addr/city/...).
    recs_y: same for block Y (table B side).
    seed_pair: (a_id, b_id) — confirmed true match the blocker split.
    sim(a_id, b_id) -> float — char-3gram TF-IDF cosine similarity.
    rng: seeded random.Random — use for any randomness.
    The harness shows at most 250 pairs to the judge (fewer if the
    prompt character budget is hit) — pairs beyond that are dropped.
    """
    def text_of(r):
        return " ".join(str(v) for k, v in r.items() if k != "id" and isinstance(v, str)).lower()

    def tokens_of(r):
        return set(re.findall(r"[a-z0-9]+", text_of(r)))

    def rec_render_len(r):
        # include key names + per-field formatting overhead (quotes,
        # colons, commas, newlines) — raw value length alone
        # undercounts actual rendered cost and caused real truncation
        # on long-field datasets (walmart-amazon: 15 sampled, only 10
        # survived the 20k prompt budget).
        return sum(len(str(k)) + len(str(v)) + 4 for k, v in r.items())

    id_x = [r["id"] for r in recs_x]
    id_y = [r["id"] for r in recs_y]
    nx, ny = len(id_x), len(id_y)
    total = nx * ny
    MAX_SIM_CALLS = 8000
    HARD_MAX_RETURN = 250

    if total == 0:
        return []

    # ---- budget-aware cap: estimate chars/pair from ACTUAL record text
    # length so we don't blindly hand back 250 pairs when the render
    # truncates at ~10-50 on long-text datasets. Extra safety margin
    # (lower effective budget, higher per-pair multiplier/overhead)
    # since underestimating here silently drops shown pairs mid-list.
    sample_x = recs_x if nx <= 200 else [recs_x[i] for i in range(0, nx, max(1, nx // 200))]
    sample_y = recs_y if ny <= 200 else [recs_y[i] for i in range(0, ny, max(1, ny // 200))]
    avg_len_x = sum(rec_render_len(r) for r in sample_x) / max(1, len(sample_x))
    avg_len_y = sum(rec_render_len(r) for r in sample_y) / max(1, len(sample_y))
    EFFECTIVE_BUDGET = 12000  # extra headroom under 20k vs instructions/schema/labels
    per_pair_cost = (avg_len_x + avg_len_y) * 1.3 + 50
    MAX_RETURN = max(12, min(HARD_MAX_RETURN, int(EFFECTIVE_BUDGET / max(per_pair_cost, 1))))

    if total <= MAX_SIM_CALLS:
        candidate_pairs = [(a, b) for a in id_x for b in id_y]
    else:
        # cheap token-overlap prefilter — avoid calling sim() on the full
        # cross product for big blocks (can be 400 x 15000 records)
        tok_x = {r["id"]: tokens_of(r) for r in recs_x}
        tok_y = {r["id"]: tokens_of(r) for r in recs_y}

        # document frequency of each y-side token, computed BEFORE capping
        # postings — used to down-weight generic words so rare/distinctive
        # tokens (model numbers, exact editions) drive candidate selection
        # instead of being drowned out by raw overlap counts.
        df = Counter()
        for toks in tok_y.values():
            for t in toks:
                df[t] += 1

        index = defaultdict(list)
        max_posting = max(20, ny // 10)
        for bid, toks in tok_y.items():
            for t in toks:
                lst = index[t]
                if len(lst) < max_posting:
                    lst.append(bid)

        per_a_k = max(6, MAX_SIM_CALLS // max(1, nx))
        seen = set()
        candidate_pairs = []

        # process the seed's a-record FIRST so it never gets starved by
        # an early MAX_SIM_CALLS break — its neighborhood (other listings
        # of the same entity) is the highest-value source of additional
        # true pairs beyond the seed itself.
        ax0, bx0 = seed_pair
        seed_a_first = ax0 if ax0 in tok_x else (bx0 if bx0 in tok_x else None)
        order = list(tok_x.keys())
        if seed_a_first is not None:
            order.remove(seed_a_first)
            order.insert(0, seed_a_first)

        for aid in order:
            toks = tok_x[aid]
            overlap = Counter()
            for t in toks:
                w = 1.0 / df[t] if df.get(t) else 0.0
                for bid in index.get(t, ()):
                    overlap[bid] += w
            for bid, _ in overlap.most_common(per_a_k):
                key = (aid, bid)
                if key not in seen:
                    seen.add(key)
                    candidate_pairs.append(key)
            if len(candidate_pairs) >= MAX_SIM_CALLS:
                break

        if not candidate_pairs:
            # degenerate: no token overlap anywhere — fall back to a
            # bounded diagonal sample so sim() still has something to rank
            candidate_pairs = [(id_x[i], id_y[i % ny]) for i in range(min(nx, MAX_SIM_CALLS))]

    # make sure the confirmed seed pair is always scored, even if the
    # prefilter happened to drop it
    ax, bx = seed_pair
    seed = None
    if ax in id_x and bx in id_y:
        seed = (ax, bx)
    elif bx in id_x and ax in id_y:
        seed = (bx, ax)
    if seed is not None and seed not in candidate_pairs:
        candidate_pairs.append(seed)

    scores = {}
    for a, b in candidate_pairs:
        try:
            scores[(a, b)] = sim(a, b)
        except Exception:
            scores[(a, b)] = 0.0

    # per-row and per-column ranked partner lists
    ranked_by_a = defaultdict(list)
    ranked_by_b = defaultdict(list)
    for (a, b), s in scores.items():
        ranked_by_a[a].append((s, b))
        ranked_by_b[b].append((s, a))
    for a in ranked_by_a:
        ranked_by_a[a].sort(key=lambda t: -t[0])
    for b in ranked_by_b:
        ranked_by_b[b].sort(key=lambda t: -t[0])

    result = []
    seen_pairs = set()
    if seed is not None:
        result.append(seed)
        seen_pairs.add(seed)

        # seed's row/column neighbors — other B candidates for the same
        # A entity and other A candidates for the same B entity. Scale
        # NEIGHBOR_K to the available budget: with a tight char budget
        # (a handful of pairs) burning 10 of 16 slots on the seed's own
        # neighborhood starves coverage of OTHER true pairs elsewhere in
        # the block (observed: amazon-google block with 8 true split
        # pairs, only 3 recovered because the sampler never looked past
        # the seed's row/col). Only go deep on seed neighbors when the
        # budget can afford breadth elsewhere too.
        sa, sb = seed
        NEIGHBOR_K = max(1, min(5, MAX_RETURN // 6))
        for s, b in ranked_by_a.get(sa, [])[:NEIGHBOR_K]:
            pair = (sa, b)
            if pair not in seen_pairs and len(result) < MAX_RETURN:
                seen_pairs.add(pair)
                result.append(pair)
        for s, a in ranked_by_b.get(sb, [])[:NEIGHBOR_K]:
            pair = (a, sb)
            if pair not in seen_pairs and len(result) < MAX_RETURN:
                seen_pairs.add(pair)
                result.append(pair)

    # mutual top-1 pairs — a's best partner is b AND b's best partner is
    # a. Strong true-match signal independent of row/column rank order,
    # and naturally spreads across DIFFERENT entities in the block (not
    # just the seed's), so it's the main breadth source under a tight
    # budget; front-loaded right after the seed(+neighbors) so it
    # survives even when the prompt char budget only fits a handful of
    # pairs.
    mutual = []
    for a, lst in ranked_by_a.items():
        if not lst:
            continue
        s, b = lst[0]
        blst = ranked_by_b.get(b)
        if blst and blst[0][1] == a:
            pair = (a, b)
            if pair not in seen_pairs:
                mutual.append((s, pair))
    mutual.sort(key=lambda t: -t[0])
    for s, pair in mutual:
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            result.append(pair)
        if len(result) >= MAX_RETURN:
            break

    # visit rows/columns best-top-score-first (shuffle first to break ties
    # without bias), so the entities most likely to have a real match get
    # their slot before entities that clearly have nothing similar
    a_order = list(ranked_by_a.keys())
    b_order = list(ranked_by_b.keys())
    rng.shuffle(a_order)
    rng.shuffle(b_order)
    a_order.sort(key=lambda a: -ranked_by_a[a][0][0])
    b_order.sort(key=lambda b: -ranked_by_b[b][0][0])

    # zippered round robin: at each rank, alternate ONE pair at a time
    # between the a-side and b-side orders (not a full block-dump of one
    # side before the other) so both sides get slots even when only the
    # first handful of pairs survive truncation.
    max_rank = max((len(v) for v in ranked_by_a.values()), default=0)
    max_rank = max(max_rank, max((len(v) for v in ranked_by_b.values()), default=0))

    for rank in range(max_rank):
        if len(result) >= MAX_RETURN:
            break
        ai = bi = 0
        while ai < len(a_order) or bi < len(b_order):
            if ai < len(a_order):
                a = a_order[ai]
                ai += 1
                lst = ranked_by_a[a]
                if rank < len(lst):
                    _, b = lst[rank]
                    pair = (a, b)
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        result.append(pair)
                        if len(result) >= MAX_RETURN:
                            break
            if bi < len(b_order):
                b = b_order[bi]
                bi += 1
                lst = ranked_by_b[b]
                if rank < len(lst):
                    _, a = lst[rank]
                    pair = (a, b)
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        result.append(pair)
                        if len(result) >= MAX_RETURN:
                            break
            if len(result) >= MAX_RETURN:
                break

    return result[:MAX_RETURN]
```

### sampler_code diff vs parent #32 (truncated)
```diff
--- 
+++ 
@@ -21,3 +21,8 @@
     def rec_render_len(r):
-        return sum(len(str(v)) for v in r.values())
+        # include key names + per-field formatting overhead (quotes,
+        # colons, commas, newlines) — raw value length alone
+        # undercounts actual rendered cost and caused real truncation
+        # on long-field datasets (walmart-amazon: 15 sampled, only 10
+        # survived the 20k prompt budget).
+        return sum(len(str(k)) + len(str(v)) + 4 for k, v in r.items())
 
@@ -35,7 +40,5 @@
     # length so we don't blindly hand back 250 pairs when the render
-    # truncates at ~10-50 on long-text datasets (walmart-amazon: titles
-    # + descriptions routinely blow the 20k prompt budget after a
-    # handful of pairs). Sizing the return list to what can actually be
-    # SHOWN beats over-supplying and hoping list order survives an
-    # unknown truncation point.
+    # truncates at ~10-50 on long-text datasets. Extra safety margin
+    # (lower effective budget, higher per-pair multiplier/overhead)
+    # since underestimating here silently drops shown pairs mid-list.
     sample_x = recs_x if nx <= 200 else [recs_x[i] for i in range(0, nx, max(1, nx // 200))]
@@ -44,5 +47,5 @@
     avg_len_y = sum(rec_render_len(r) for r in sample_y) / max(1, len(sample_y))
-    EFFECTIVE_BUDGET = 14000  # headroom under 20k for instructions/schema/labels
-    per_pair_cost = (avg_len_x + avg_len_y) * 1.15 + 30
-    MAX_RETURN = max(15, min(HARD_MAX_RETURN, int(EFFECTIVE_BUDGET / max(per_pair_cost, 1))))
+    EFFECTIVE_BUDGET = 12000  # extra headroom under 20k vs instructions/schema/labels
+    per_pair_cost = (avg_len_x + avg_len_y) * 1.3 + 50
+    MAX_RETURN = max(12, min(HARD_MAX_RETURN, int(EFFECTIVE_BUDGET / max(per_pair_cost, 1))))
 
@@ -144,9 +147,12 @@
         # seed's row/column neighbors — other B candidates for the same
-        # A entity and other A candidates for the same B entity. These
-        # are the cheapest, highest-probability source of ADDITIONAL true
-        # pairs (duplicate listings of the entity that was already
-        # confirmed to match) and cost only two extra small lookups, so
-        # they go in right after the seed, ahead of generic ranking.
+        # A entity and other A candidates for the same B entity. Scale
+        # NEIGHBOR_K to the available budget: with a tight char budget
+        # (a handful of pairs) burning 10 of 16 slots on the seed's own
+        # neighborhood starves coverage of OTHER true pairs elsewhere in
+        # the block (observed: amazon-google block with 8 true split
+        # pairs, only 3 recovered because the sampler never looked past
+        # the seed's row/col). Only go deep on seed neighbors when the
+        # budget can afford breadth elsewhere too.
         sa, sb = seed
-        NEIGHBOR_K = 5
+        NEIGHBOR_K = max(1, min(5, MAX_RETURN // 6))
         for s, b in ranked_by_a.get(sa, [])[:NEIGHBOR_K]:
@@ -163,5 +169,8 @@
     # mutual top-1 pairs — a's best partner is b AND b's best partner is
-    # a. Strong true-match signal independent of row/column rank order;
-    # front-loaded right after the seed (+ its neighbors) so it survives
-    # even when the prompt char budget only fits a handful of pairs.
+    # a. Strong true-match signal independent of row/column rank order,
+    # and naturally spreads across DIFFERENT entities in the block (not
+    # just the seed's), so it's the main breadth source under a tight
+    # budget; front-loaded right after the seed(+neighbors) so it
```
