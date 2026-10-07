# run6 evolution — checkpoints by relative val gain

Reference episode for retrieval samples: `amazon-google_K8_000100000_000100010` (blocks 100x227 records, 6 hidden targets). Filters are deterministic, so these re-renders are exactly what the judge saw.

| # | val | rel. gain | rollout | mutated |
|---|---|---|---|---|
| 0 | 0.033 | — (seed) | 0 | seed |
| 1 | 0.087 | +160% | 103 | filter_code |
| 2 | 0.196 | +125% | 174 | filter_code |
| 5 | 0.254 | +30% | 363 | filter_code |
| 8 | 0.266 | +5% | 664 | instruction |
| 14 | 0.274 | +3% | 1122 | instruction |
| 16 | 0.298 | +9% | 1364 | instruction |
| 17 | 0.300 | +1% | 1443 | filter_code |
| 21 | 0.329 | +10% | 1735 | filter_code |
| 28 | 0.335 | +2% | 2420 | filter_code |
| 41 | 0.355 | +6% | 3631 | filter_code |

---

## Checkpoint #0 — val 0.033 (seed), rollout 0, mutated: seed

**Filter output on the reference episode:** showed 26 A-records and 60 B-records, covering 2/6 hidden targets (render truncated at budget).

```
## List A — records from block X (26 of 100 in block)
1. title: suretrak project manager 3.0 | description: communicate with professional-quality reports and graphics. suretrak project manager combines ease of use powe | manufacturer: primavera systems | price: 499
2. title: punch! master landscape & home design | description: master landscape & home design offers nine programs on one easy-to-use interface to help you design home exter | manufacturer: punch! software | price: 59.99
3. title: microsoft data analyzer 2002 | description: model- ms-cd87800wi vendor- microsoft corporation features- data analyzer microsoft data analyzer is an easy-t | manufacturer: microsoft | price: 179
4. title: punch! master landscape & home design | description: plan out your perfect home & garden with the set of applications available here! / for windows take a virtual  | manufacturer: punch! software | price: 99.99
5. title: talk now! learn afrikaans - beginning level | description: talk now! beginners afrikaans is the perfect way to study a new language. it operates on multiple levels to cr | manufacturer: eurotalk | price: 29.99
6. title: encyclopedia britannica profiles: dinosaurs (jewel case) | description: the captivating world of dinosaurs at yourfingertips product informationwhat makes dinosaurs so interesting -  | manufacturer: atari | price: 9.99
7. title: instant home cooking (jewel case) | description: with over 23 600 recipes and access to over 100 000 more instant home cooking is your complete reference sourc | manufacturer: topics entertainment | price: 9.99
8. title: fisher-price rescue heroes: lava landslide | description: join the rescue heroes and become a hero when avolcano erupts!product information&nbsp;emergency! this is a re | manufacturer: knowledge adventure | price: 0
9. title: norton internet security mac 3.0 [antivirus firewall privacy controls iclean] | description: norton internet security 2003 professional is the complete set of tools for guarding your business from intern | manufacturer: symantec | price: 99.95
10. title: power director 3 | description: powerdirector 3 - it's everything you need to enhance your camcorder videos! turn simple home movies into high | manufacturer: avanquest publishing usa inc. | price: 79.95
11. title: the human body | description: in topics presents the human body you'll uncover the deepest mysteries and complexities of the human form in t | manufacturer: topics entertainment | price: 19.99
```

### instruction
```
You are debugging the output of a blocking system for entity resolution. Two blocks that wrongly separated at least one true match are shown: list A holds one block's records from table A, list B holds the other block's records from table B. A confirmed true match split across the two blocks is given as the SEED. Find OTHER true matches between list A and list B: pairs referring to the same real-world entity. Return JSON {"matches": [[a_number, b_number], ...]} using the list numbers. Do not include the seed pair. If none, return an empty list.
```

### filter_code
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

## Checkpoint #1 — val 0.087 (+160% relative), rollout 103, mutated: filter_code

**Filter output on the reference episode:** showed 21 A-records and 23 B-records, covering 5/6 hidden targets.

```
## List A — records from block X (21 of 100 in block)
1. title: punch! master landscape & home design | description: master landscape & home design offers nine programs on one easy-to-use interface to help you design home exter | manufacturer: punch! software | price: 59.99
2. title: punch! master landscape & home design | description: plan out your perfect home & garden with the set of applications available here! / for windows take a virtual  | manufacturer: punch! software | price: 99.99
3. title: instant home design 3.0 [lb] | description: software suite helps you create customized before & after visuals of your home renovation in both photographic | manufacturer: topics entertainment | price: 19.99
4. title: instant landscaping 3.0 [lb] | description: instant landscape design 3.0 is the perfect tool for creating the perfect garden yard or landscape. draw the p | manufacturer: topics entertainment | price: 19.99
5. title: instant home design (jewel case) | description: instant home design (win 95 98 me nt 2000 xp) | manufacturer: topics entertainment | price: 9.99
6. title: intuit quicken home inventory manager - windows | description: quicken home inventory manager makes it easy to keep an organized record of your belongings so you are ready f | manufacturer: intuit | price: 34.95
7. title: instant architect design suite | description: instant architect design suite (win 98 me nt 2000 xp) | manufacturer: imsi design | price: 29.99
8. title: instant home cooking (jewel case) | description: with over 23 600 recipes and access to over 100 000 more instant home cooking is your complete reference sourc | manufacturer: topics entertainment | price: 9.99
9. title: ca antivirus 2007 | description: anti-virus 2007 is the latest antivirus protection for your home and home office pc. take comfort while you're | manufacturer: computer associates | price: 39.95
10. title: solitaire antics deluxe | description: 50 different games of solitaire includes tours double deck games customizable rules and options high score and | manufacturer: masque publishing | price: 0
11. title: power director 3 | description: powerdirector 3 - it's everything you need to enhance your camcorder videos! turn simple home movies into high | manufacturer: avanquest publishing usa inc. | price: 79.95
```

### instruction
```
You are debugging the output of a blocking system for entity resolution. Two blocks that wrongly separated at least one true match are shown: list A holds one block's records from table A, list B holds the other block's records from table B. A confirmed true match split across the two blocks is given as the SEED. Find OTHER true matches between list A and list B: pairs referring to the same real-world entity. Return JSON {"matches": [[a_number, b_number], ...]} using the list numbers. Do not include the seed pair. If none, return an empty list.
```

### filter_code
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

    # Rank each side by relevance to the confirmed seed record on the
    # other side — records that resemble the seed's counterpart are
    # the most likely to belong to other true (unshown-so-far) pairs.
    others_x = [r for r in recs_x if r["id"] != seed_a]
    others_y = [r for r in recs_y if r["id"] != seed_b]

    scored_x = sorted(
        others_x, key=lambda r: sim(r["id"], seed_b), reverse=True
    )
    scored_y = sorted(
        others_y, key=lambda r: sim(seed_a, r["id"]), reverse=True
    )

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
    cnt_x, used_x = fit_count(sizes_x, half)
    cnt_y, used_y = fit_count(sizes_y, half)

    # Water-fill: give unused budget from a side that ran out of
    # records to the side that still has more to show.
    if used_x < half and cnt_x == len(sizes_x) and used_y >= half:
        y_budget = half + (half - used_x)
        cnt_y, used_y = fit_count(sizes_y, y_budget)
    elif used_y < half and cnt_y == len(sizes_y) and used_x >= half:
        x_budget = half + (half - used_y)
        cnt_x, used_x = fit_count(sizes_x, x_budget)

    a_ids = [r["id"] for r in ordered_x[:max(cnt_x, 1)]]
    b_ids = [r["id"] for r in ordered_y[:max(cnt_y, 1)]]

    return a_ids, b_ids
```

### filter_code: diff vs parent #0
```diff
--- 
+++ 
@@ -16,2 +16,57 @@
     """
-    return [r["id"] for r in recs_x], [r["id"] for r in recs_y]
+    BUDGET = 20000
+
+    def rec_chars(r):
+        return sum(len(str(k)) + len(str(v)) + 3 for k, v in r.items()) + 8
+
+    seed_a, seed_b = seed_pair
+
+    by_id_x = {r["id"]: r for r in recs_x}
+    by_id_y = {r["id"]: r for r in recs_y}
+
+    # Rank each side by relevance to the confirmed seed record on the
+    # other side — records that resemble the seed's counterpart are
+    # the most likely to belong to other true (unshown-so-far) pairs.
+    others_x = [r for r in recs_x if r["id"] != seed_a]
+    others_y = [r for r in recs_y if r["id"] != seed_b]
+
+    scored_x = sorted(
+        others_x, key=lambda r: sim(r["id"], seed_b), reverse=True
+    )
+    scored_y = sorted(
+        others_y, key=lambda r: sim(seed_a, r["id"]), reverse=True
+    )
+
+    ordered_x = ([by_id_x[seed_a]] if seed_a in by_id_x else []) + scored_x
+    ordered_y = ([by_id_y[seed_b]] if seed_b in by_id_y else []) + scored_y
+
+    sizes_x = [rec_chars(r) for r in ordered_x]
+    sizes_y = [rec_chars(r) for r in ordered_y]
+
+    def fit_count(sizes, budget):
+        used = 0
+        n = 0
+        for s in sizes:
+            if used + s > budget:
+                break
+            used += s
+            n += 1
+        return n, used
+
+    half = BUDGET / 2.0
+    cnt_x, used_x = fit_count(sizes_x, half)
+    cnt_y, used_y = fit_count(sizes_y, half)
+
+    # Water-fill: give unused budget from a side that ran out of
+    # records to the side that still has more to show.
+    if used_x < half and cnt_x == len(sizes_x) and used_y >= half:
+        y_budget = half + (half - used_x)
+        cnt_y, used_y = fit_count(sizes_y, y_budget)
+    elif used_y < half and cnt_y == len(sizes_y) and used_x >= half:
+        x_budget = half + (half - used_y)
+        cnt_x, used_x = fit_count(sizes_x, x_budget)
+
+    a_ids = [r["id"] for r in ordered_x[:max(cnt_x, 1)]]
+    b_ids = [r["id"] for r in ordered_y[:max(cnt_y, 1)]]
+
+    return a_ids, b_ids
```

---

## Checkpoint #2 — val 0.196 (+125% relative), rollout 174, mutated: filter_code

**Filter output on the reference episode:** showed 23 A-records and 23 B-records, covering 6/6 hidden targets.

```
## List A — records from block X (23 of 100 in block)
1. title: punch! master landscape & home design | description: master landscape & home design offers nine programs on one easy-to-use interface to help you design home exter | manufacturer: punch! software | price: 59.99
2. title: pinnacle mobile media organizer | description: package contents: mobile media organizer cd media mobile master quick install guide pinnacle's mobile media or | manufacturer: pinnacle | price: 49.99
3. title: dk rainbow fish most beautiful fish in the ocean | description: one stormy night while rainbow fish slept on the ocean floor his beautiful sparkling scales were stolen by thr | manufacturer: encore | price: 9.99
4. title: project x project management software | description: project x is truly project management software for the rest of us. its for those of us that want to spend more | manufacturer: marware | price: 199.95
5. title: punch! master landscape & home design | description: plan out your perfect home & garden with the set of applications available here! / for windows take a virtual  | manufacturer: punch! software | price: 99.99
6. title: adobe photoshop cs2 fundamental techniques by julieanne kost | description: learn essential skills that will become a foundation for a creative and efficient workflow. you will gain a co | manufacturer: software cinema | price: 0
7. title: instant play electric guitar (jewel box) | description: instant play electric guitar hits the right note for flexible computer-based instruction. this 2-cd set takes  | manufacturer: topics entertainment | price: 6.99
8. title: pdf converter 4 | description: scansoft pdf converter 4 instantly converts pdf files into documents spreadsheets and forms that look exactly  | manufacturer: nuance | price: 49.95
9. title: individual small business advantage deluxe 2006 | description: with small business advantage deluxe 2006 is a complete set of marketing tools for entrepreneurs & small busin | manufacturer: individual | price: 49.99
10. title: instant landscaping 3.0 [lb] | description: instant landscape design 3.0 is the perfect tool for creating the perfect garden yard or landscape. draw the p | manufacturer: topics entertainment | price: 19.99
11. title: instant play electric guitar deluxe | description: with play electric guitar beginners who want a fast flexible computer-based teaching tool have the in-depth in | manufacturer: topics entertainment | price: 39.99
```

### instruction
```
You are debugging the output of a blocking system for entity resolution. Two blocks that wrongly separated at least one true match are shown: list A holds one block's records from table A, list B holds the other block's records from table B. A confirmed true match split across the two blocks is given as the SEED. Find OTHER true matches between list A and list B: pairs referring to the same real-world entity. Return JSON {"matches": [[a_number, b_number], ...]} using the list numbers. Do not include the seed pair. If none, return an empty list.
```

### filter_code
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

### filter_code: diff vs parent #1
```diff
--- 
+++ 
@@ -26,5 +26,2 @@
 
-    # Rank each side by relevance to the confirmed seed record on the
-    # other side — records that resemble the seed's counterpart are
-    # the most likely to belong to other true (unshown-so-far) pairs.
     others_x = [r for r in recs_x if r["id"] != seed_a]
@@ -32,8 +29,64 @@
 
-    scored_x = sorted(
-        others_x, key=lambda r: sim(r["id"], seed_b), reverse=True
-    )
-    scored_y = sorted(
-        others_y, key=lambda r: sim(seed_a, r["id"]), reverse=True
-    )
+    n_x, n_y = len(others_x), len(others_y)
+
+    # Best-match score per record: does it look like it has a partner
+    # on the other side at all (not just near the seed)? Full
+    # bipartite scoring when cheap; bounded random sampling otherwise
+    # to stay safely inside the time limit on very large blocks.
+    MAX_CALLS = 150000
+
+    best_x = {r["id"]: 0.0 for r in others_x}
+    best_y = {r["id"]: 0.0 for r in others_y}
+
+    if n_x and n_y:
+        full_cost = n_x * n_y
+        if full_cost <= MAX_CALLS:
+            for rx in others_x:
+                xid = rx["id"]
+                best = 0.0
+                for ry in others_y:
+                    yid = ry["id"]
+                    s = sim(xid, yid)
+                    if s > best:
+                        best = s
+                    if s > best_y[yid]:
+                        best_y[yid] = s
+                best_x[xid] = best
+        else:
+            k_y = max(1, MAX_CALLS // (2 * n_x))
+            k_x = max(1, MAX_CALLS // (2 * n_y))
+            sample_y = rng.sample(others_y, min(k_y, n_y))
+            sample_x = rng.sample(others_x, min(k_x, n_x))
+            for rx in others_x:
+                xid = rx["id"]
+                best = 0.0
+                for ry in sample_y:
+                    s = sim(xid, ry["id"])
+                    if s > best:
+                        best = s
+                best_x[xid] = best
+            for ry in others_y:
+                yid = ry["id"]
+                best = 0.0
+                for rx in sample_x:
+                    s = sim(rx["id"], yid)
+                    if s > best:
+                        best = s
+                best_y[yid] = best
+
+    # Fold in direct relevance to the confirmed seed's counterpart —
+    # still a useful boost for near-duplicates of the known match.
+    seed_score_x = {r["id"]: sim(r["id"], seed_b) for r in others_x}
+    seed_score_y = {r["id"]: sim(seed_a, r["id"]) for r in others_y}
+
+    def score_x(r):
+        rid = r["id"]
+        return max(best_x.get(rid, 0.0), seed_score_x.get(rid, 0.0))
+
+    def score_y(r):
+        rid = r["id"]
+        return max(best_y.get(rid, 0.0), seed_score_y.get(rid, 0.0))
+
+    scored_x = sorted(others_x, key=score_x, reverse=True)
+    scored_y = sorted(others_y, key=score_y, reverse=True)
 
@@ -56,4 +109,4 @@
```

---

## Checkpoint #5 — val 0.254 (+30% relative), rollout 363, mutated: filter_code

**Filter output on the reference episode:** showed 20 A-records and 23 B-records, covering 6/6 hidden targets.

```
## List A — records from block X (20 of 100 in block)
1. title: punch! master landscape & home design | description: master landscape & home design offers nine programs on one easy-to-use interface to help you design home exter | manufacturer: punch! software | price: 59.99
2. title: pinnacle mobile media organizer | description: package contents: mobile media organizer cd media mobile master quick install guide pinnacle's mobile media or | manufacturer: pinnacle | price: 49.99
3. title: dk rainbow fish most beautiful fish in the ocean | description: one stormy night while rainbow fish slept on the ocean floor his beautiful sparkling scales were stolen by thr | manufacturer: encore | price: 9.99
4. title: punch! master landscape & home design | description: plan out your perfect home & garden with the set of applications available here! / for windows take a virtual  | manufacturer: punch! software | price: 99.99
5. title: adobe photoshop cs2 fundamental techniques by julieanne kost | description: learn essential skills that will become a foundation for a creative and efficient workflow. you will gain a co | manufacturer: software cinema | price: 0
6. title: instant play electric guitar (jewel box) | description: instant play electric guitar hits the right note for flexible computer-based instruction. this 2-cd set takes  | manufacturer: topics entertainment | price: 6.99
7. title: pdf converter 4 | description: scansoft pdf converter 4 instantly converts pdf files into documents spreadsheets and forms that look exactly  | manufacturer: nuance | price: 49.95
8. title: instant play electric guitar deluxe | description: with play electric guitar beginners who want a fast flexible computer-based teaching tool have the in-depth in | manufacturer: topics entertainment | price: 39.99
9. title: business plan writer deluxe 8.0 2005 | description: business plan writer deluxe 8 helps you get your ideas down turn them into a definite business plan and commit | manufacturer: nova development | price: 99.99
10. title: instant landscaping 3.0 [lb] | description: instant landscape design 3.0 is the perfect tool for creating the perfect garden yard or landscape. draw the p | manufacturer: topics entertainment | price: 19.99
11. title: myinvoices & estimates deluxe | description: myinvoices and estimates deluxe makes ite asier to handle one of the most important elements of your business: | manufacturer: avanquest software | price: 39.95
```

### instruction
```
You are debugging the output of a blocking system for entity resolution. Two blocks that wrongly separated at least one true match are shown: list A holds one block's records from table A, list B holds the other block's records from table B. A confirmed true match split across the two blocks is given as the SEED. Find OTHER true matches between list A and list B: pairs referring to the same real-world entity. Return JSON {"matches": [[a_number, b_number], ...]} using the list numbers. Do not include the seed pair. If none, return an empty list.

Use the SEED only to learn which fields carry identity in this schema (e.g. which field holds a unique code/identifier, which holds the name/title, which holds a secondary descriptor). Then apply the same field roles to judge every other candidate pair independently. Do not assume any other record resembles the seed's content.

A pair is a match only if BOTH of these hold:
1. Any field that acts as a unique code, ID, or reference number (e.g. an identifier repeated in two places in one record, like a URL that embeds the same code as an adjacent field) must be IDENTICAL between the two records if both records expose that field. A mismatch on such a field is disqualifying even if surrounding text looks similar — do not treat "same field structure/template" as "same value."
2. The core name/title/description text must refer to the same specific entity: same category of thing AND same distinguishing specifics (variant, edition, size/capacity, model, configuration, revision). Overlapping category words alone (same broad product type, same general subject) are NOT sufficient.

Distinguish true variant relationships from false ones:
- Two records CAN match despite differing surface wording if one is a bundle/kit/hybrid/repackaged listing of the exact same underlying item, or differs only in incidental packaging/channel language, provided every distinguishing spec (edition number, size, capacity, version) still agrees.
- Two records must NOT match if any distinguishing marker differs: different version/edition/revision numbers, different upgrade-source ("upgrade from X" vs "upgrade from Y"), different size/capacity/dimensions, different color/model variant — even when the rest of the wording is near-identical. Near-duplicate records that differ only in one such marker are DIFFERENT entities (e.g. two records that are otherwise textually identical but state different predecessor-version numbers are not a match).

Rare shared tokens are strong evidence: if a distinctive multi-word phrase, code, or number sequence appears in both records and is too specific to be generic category language, weight it heavily. Conversely, shared generic/boilerplate phrasing (category names, standard descriptive filler, common measurement units) is weak evidence and should not drive a match by itself.

Do not default to matching many-to-one or one-to-many: in the common case each entity in list A has at most one true counterpart in list B. If several list B records look superficially similar to one list A record, prefer zero matches over guessing among them unless one is clearly distinguished by matching specifics.

Do not produce a match for every, or nearly every, row in a list — that indicates you are matching on generic category/template similarity rather than entity identity. A typical correct answer is a small number of pairs, often zero.

Before finalizing each pair, check it against the disqualifying rule in point 1 above and re-read both full records side by side for any distinguishing spec that differs. If uncertain, exclude the pair.
```

### filter_code
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

    # cheap seed-relative relevance (one sim call per record)
    seed_score_x = {r["id"]: sim(r["id"], seed_b) for r in others_x}
    seed_score_y = {r["id"]: sim(seed_a, r["id"]) for r in others_y}

    # structural cross-cluster candidate discovery: char-3gram shingles,
    # independent of the seed pair, to surface OTHER true pairs that
    # don't resemble the seed lexically.
    def text_of(r):
        return " ".join(str(v) for k, v in r.items() if k != "id").lower()

    def shingles(s, n=3):
        s = re.sub(r"\s+", " ", s).strip()
        if len(s) < n:
            return {s} if s else set()
        return {s[i:i + n] for i in range(len(s) - n + 1)}

    shingles_x = {r["id"]: shingles(text_of(r)) for r in others_x}
    shingles_y = {r["id"]: shingles(text_of(r)) for r in others_y}

    doc_freq = Counter()
    for sset in shingles_y.values():
        for sh in sset:
            doc_freq[sh] += 1
    n_y = max(len(shingles_y), 1)
    common_cutoff = max(5, n_y // 20)  # ignore very common shingles (structural, generic)

    inv_index = defaultdict(list)
    for yid, sset in shingles_y.items():
        for sh in sset:
            if doc_freq[sh] <= common_cutoff:
                inv_index[sh].append(yid)

    overlap = defaultdict(int)
    max_ops = 200000
    ops = 0
    stop = False
    for xid, sset in shingles_x.items():
        if stop:
            break
        for sh in sset:
            if doc_freq.get(sh, 0) > common_cutoff:
                continue
            for yid in inv_index.get(sh, ()):
                overlap[(xid, yid)] += 1
                ops += 1
                if ops >= max_ops:
                    stop = True
                    break
            if stop:
                break

    top_pairs = sorted(overlap.items(), key=lambda kv: kv[1], reverse=True)[:400]
    scored_pairs = sorted(
        ((sim(xid, yid), xid, yid) for (xid, yid), _ov in top_pairs),
        key=lambda t: t[0],
        reverse=True,
    )

    order_x_ids = [seed_a] if seed_a in by_id_x else []
    order_y_ids = [seed_b] if seed_b in by_id_y else []
    seen_x = set(order_x_ids)
    seen_y = set(order_y_ids)

    for _score, xid, yid in scored_pairs:
        if xid not in seen_x:
            order_x_ids.append(xid)
            seen_x.add(xid)
        if yid not in seen_y:
            order_y_ids.append(yid)
            seen_y.add(yid)

    remainder_x = sorted(others_x, key=lambda r: seed_score_x[r["id"]], reverse=True)
    remainder_y = sorted(others_y, key=lambda r: seed_score_y[r["id"]], reverse=True)

    for r in remainder_x:
        if r["id"] not in seen_x:
            order_x_ids.append(r["id"])
            seen_x.add(r["id"])
    for r in remainder_y:
        if r["id"] not in seen_y:
            order_y_ids.append(r["id"])
            seen_y.add(r["id"])

    ordered_x = [by_id_x[i] for i in order_x_ids]
    ordered_y = [by_id_y[i] for i in order_y_ids]

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

    # water-fill: give unused budget from a side that ran out of
    # records to the side that still has more to show.
    if used_x < half and count_x == len(sizes_x) and used_y >= half:
        budget_y = half + (half - used_x)
        count_y, used_y = fit_count(sizes_y, budget_y)
    elif used_y < half and count_y == len(sizes_y) and used_x >= half:
        budget_x = half + (half - used_y)
        count_x, used_x = fit_count(sizes_x, budget_x)

    a_ids = [r["id"] for r in ordered_x[:max(count_x, 1)]]
    b_ids = [r["id"] for r in ordered_y[:max(count_y, 1)]]

    return a_ids, b_ids
```

### filter_code: diff vs parent #3
```diff
--- 
+++ 
@@ -26,5 +26,2 @@
 
-    # Rank each side by relevance to the confirmed seed record on the
-    # other side — records that resemble the seed's counterpart are
-    # the most likely to belong to other true (unshown-so-far) pairs.
     others_x = [r for r in recs_x if r["id"] != seed_a]
@@ -32,11 +29,87 @@
 
-    scored_x = sorted(
-        others_x, key=lambda r: sim(r["id"], seed_b), reverse=True
-    )
-    scored_y = sorted(
-        others_y, key=lambda r: sim(seed_a, r["id"]), reverse=True
+    # cheap seed-relative relevance (one sim call per record)
+    seed_score_x = {r["id"]: sim(r["id"], seed_b) for r in others_x}
+    seed_score_y = {r["id"]: sim(seed_a, r["id"]) for r in others_y}
+
+    # structural cross-cluster candidate discovery: char-3gram shingles,
+    # independent of the seed pair, to surface OTHER true pairs that
+    # don't resemble the seed lexically.
+    def text_of(r):
+        return " ".join(str(v) for k, v in r.items() if k != "id").lower()
+
+    def shingles(s, n=3):
+        s = re.sub(r"\s+", " ", s).strip()
+        if len(s) < n:
+            return {s} if s else set()
+        return {s[i:i + n] for i in range(len(s) - n + 1)}
+
+    shingles_x = {r["id"]: shingles(text_of(r)) for r in others_x}
+    shingles_y = {r["id"]: shingles(text_of(r)) for r in others_y}
+
+    doc_freq = Counter()
+    for sset in shingles_y.values():
+        for sh in sset:
+            doc_freq[sh] += 1
+    n_y = max(len(shingles_y), 1)
+    common_cutoff = max(5, n_y // 20)  # ignore very common shingles (structural, generic)
+
+    inv_index = defaultdict(list)
+    for yid, sset in shingles_y.items():
+        for sh in sset:
+            if doc_freq[sh] <= common_cutoff:
+                inv_index[sh].append(yid)
+
+    overlap = defaultdict(int)
+    max_ops = 200000
+    ops = 0
+    stop = False
+    for xid, sset in shingles_x.items():
+        if stop:
+            break
+        for sh in sset:
+            if doc_freq.get(sh, 0) > common_cutoff:
+                continue
+            for yid in inv_index.get(sh, ()):
+                overlap[(xid, yid)] += 1
+                ops += 1
+                if ops >= max_ops:
+                    stop = True
+                    break
+            if stop:
+                break
+
+    top_pairs = sorted(overlap.items(), key=lambda kv: kv[1], reverse=True)[:400]
+    scored_pairs = sorted(
+        ((sim(xid, yid), xid, yid) for (xid, yid), _ov in top_pairs),
+        key=lambda t: t[0],
+        reverse=True,
     )
 
-    ordered_x = ([by_id_x[seed_a]] if seed_a in by_id_x else []) + scored_x
-    ordered_y = ([by_id_y[seed_b]] if seed_b in by_id_y else []) + scored_y
+    order_x_ids = [seed_a] if seed_a in by_id_x else []
+    order_y_ids = [seed_b] if seed_b in by_id_y else []
+    seen_x = set(order_x_ids)
+    seen_y = set(order_y_ids)
+
```

---

## Checkpoint #8 — val 0.266 (+5% relative), rollout 664, mutated: instruction

**Filter output on the reference episode:** showed 23 A-records and 23 B-records, covering 6/6 hidden targets.

```
## List A — records from block X (23 of 100 in block)
1. title: punch! master landscape & home design | description: master landscape & home design offers nine programs on one easy-to-use interface to help you design home exter | manufacturer: punch! software | price: 59.99
2. title: pinnacle mobile media organizer | description: package contents: mobile media organizer cd media mobile master quick install guide pinnacle's mobile media or | manufacturer: pinnacle | price: 49.99
3. title: dk rainbow fish most beautiful fish in the ocean | description: one stormy night while rainbow fish slept on the ocean floor his beautiful sparkling scales were stolen by thr | manufacturer: encore | price: 9.99
4. title: project x project management software | description: project x is truly project management software for the rest of us. its for those of us that want to spend more | manufacturer: marware | price: 199.95
5. title: punch! master landscape & home design | description: plan out your perfect home & garden with the set of applications available here! / for windows take a virtual  | manufacturer: punch! software | price: 99.99
6. title: adobe photoshop cs2 fundamental techniques by julieanne kost | description: learn essential skills that will become a foundation for a creative and efficient workflow. you will gain a co | manufacturer: software cinema | price: 0
7. title: instant play electric guitar (jewel box) | description: instant play electric guitar hits the right note for flexible computer-based instruction. this 2-cd set takes  | manufacturer: topics entertainment | price: 6.99
8. title: pdf converter 4 | description: scansoft pdf converter 4 instantly converts pdf files into documents spreadsheets and forms that look exactly  | manufacturer: nuance | price: 49.95
9. title: individual small business advantage deluxe 2006 | description: with small business advantage deluxe 2006 is a complete set of marketing tools for entrepreneurs & small busin | manufacturer: individual | price: 49.99
10. title: instant landscaping 3.0 [lb] | description: instant landscape design 3.0 is the perfect tool for creating the perfect garden yard or landscape. draw the p | manufacturer: topics entertainment | price: 19.99
11. title: instant play electric guitar deluxe | description: with play electric guitar beginners who want a fast flexible computer-based teaching tool have the in-depth in | manufacturer: topics entertainment | price: 39.99
```

### instruction
```
You are debugging the output of a blocking system for entity resolution. Two blocks that wrongly separated at least one true match are shown: list A holds one block's records from table A, list B holds the other block's records from table B. A confirmed true match split across the two blocks is given as the SEED. Find OTHER true matches between list A and list B: pairs referring to the same real-world entity. Return JSON {"matches": [[a_number, b_number], ...]} using the list numbers. Do not include the seed pair. If none, return an empty list.

How to judge a candidate pair:
1. Identify which fields carry the entity's identity (e.g. a name/title field) versus fields that only describe context, category, or provenance (e.g. venue, source, publisher, format, listing channel). Two records can have identical or near-identical identity fields yet come from completely different context fields — that is expected and does NOT disqualify a match; the SEED itself usually shows this pattern (same identity, different context field values, different formatting/capitalization/encoding).
2. A shared rare or unusual token (a distinctive word, code, identifier, or name that only a few records in the block share) is much stronger evidence than shared common/generic words. Prioritize scanning for these overlaps across ALL fields, not just the primary identity field.
3. Field order and role can vary between the two lists (e.g. multiple co-authors or attributes listed in different orders, or split across differently-named fields). Compare by matching each field's role/content to its counterpart in the other record, not by comparing raw string position or concatenated text.
4. Watch for variant or edition markers: version numbers, model numbers, revision letters, year/date stamps, size/quantity, or "upgrade from X" phrasing. Two records with the same base name but DIFFERENT variant/edition markers usually refer to DIFFERENT real-world entities (e.g. version 6 vs version 7 of the same product line) — do not match them just because most of the text overlaps. Conversely, minor formatting differences in the SAME marker (capitalization, punctuation, encoding artifacts, HTML entities) do not change the entity.
5. Do not infer a match from partial title/name overlap alone when the surrounding fields (numeric identifiers, dates, secondary names, quantities) diverge in a way that indicates a different specific item. A short shared phrase or category word across otherwise unrelated records is not evidence.
6. Work through the ENTIRE list A against the ENTIRE list B systematically — do not stop after finding a plausible run of matches early in the lists, and do not default to matching by matching list position/index. Every candidate pair must be independently justified by identity-field and rare-token evidence, not by proximity to the seed or to already-matched pairs.
7. When uncertain between two similarly plausible candidates in list B for the same list A record, prefer the one with more independently corroborating field overlaps (multiple fields agree) over the one with only a single strong-looking overlap.
```

### filter_code
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

### instruction: diff vs parent #2
```diff
--- 
+++ 
@@ -1 +1,10 @@
 You are debugging the output of a blocking system for entity resolution. Two blocks that wrongly separated at least one true match are shown: list A holds one block's records from table A, list B holds the other block's records from table B. A confirmed true match split across the two blocks is given as the SEED. Find OTHER true matches between list A and list B: pairs referring to the same real-world entity. Return JSON {"matches": [[a_number, b_number], ...]} using the list numbers. Do not include the seed pair. If none, return an empty list.
+
+How to judge a candidate pair:
+1. Identify which fields carry the entity's identity (e.g. a name/title field) versus fields that only describe context, category, or provenance (e.g. venue, source, publisher, format, listing channel). Two records can have identical or near-identical identity fields yet come from completely different context fields — that is expected and does NOT disqualify a match; the SEED itself usually shows this pattern (same identity, different context field values, different formatting/capitalization/encoding).
+2. A shared rare or unusual token (a distinctive word, code, identifier, or name that only a few records in the block share) is much stronger evidence than shared common/generic words. Prioritize scanning for these overlaps across ALL fields, not just the primary identity field.
+3. Field order and role can vary between the two lists (e.g. multiple co-authors or attributes listed in different orders, or split across differently-named fields). Compare by matching each field's role/content to its counterpart in the other record, not by comparing raw string position or concatenated text.
+4. Watch for variant or edition markers: version numbers, model numbers, revision letters, year/date stamps, size/quantity, or "upgrade from X" phrasing. Two records with the same base name but DIFFERENT variant/edition markers usually refer to DIFFERENT real-world entities (e.g. version 6 vs version 7 of the same product line) — do not match them just because most of the text overlaps. Conversely, minor formatting differences in the SAME marker (capitalization, punctuation, encoding artifacts, HTML entities) do not change the entity.
+5. Do not infer a match from partial title/name overlap alone when the surrounding fields (numeric identifiers, dates, secondary names, quantities) diverge in a way that indicates a different specific item. A short shared phrase or category word across otherwise unrelated records is not evidence.
+6. Work through the ENTIRE list A against the ENTIRE list B systematically — do not stop after finding a plausible run of matches early in the lists, and do not default to matching by matching list position/index. Every candidate pair must be independently justified by identity-field and rare-token evidence, not by proximity to the seed or to already-matched pairs.
+7. When uncertain between two similarly plausible candidates in list B for the same list A record, prefer the one with more independently corroborating field overlaps (multiple fields agree) over the one with only a single strong-looking overlap.
```

---

## Checkpoint #14 — val 0.274 (+3% relative), rollout 1122, mutated: instruction

**Filter output on the reference episode:** showed 23 A-records and 23 B-records, covering 6/6 hidden targets.

```
## List A — records from block X (23 of 100 in block)
1. title: punch! master landscape & home design | description: master landscape & home design offers nine programs on one easy-to-use interface to help you design home exter | manufacturer: punch! software | price: 59.99
2. title: pinnacle mobile media organizer | description: package contents: mobile media organizer cd media mobile master quick install guide pinnacle's mobile media or | manufacturer: pinnacle | price: 49.99
3. title: dk rainbow fish most beautiful fish in the ocean | description: one stormy night while rainbow fish slept on the ocean floor his beautiful sparkling scales were stolen by thr | manufacturer: encore | price: 9.99
4. title: project x project management software | description: project x is truly project management software for the rest of us. its for those of us that want to spend more | manufacturer: marware | price: 199.95
5. title: punch! master landscape & home design | description: plan out your perfect home & garden with the set of applications available here! / for windows take a virtual  | manufacturer: punch! software | price: 99.99
6. title: adobe photoshop cs2 fundamental techniques by julieanne kost | description: learn essential skills that will become a foundation for a creative and efficient workflow. you will gain a co | manufacturer: software cinema | price: 0
7. title: instant play electric guitar (jewel box) | description: instant play electric guitar hits the right note for flexible computer-based instruction. this 2-cd set takes  | manufacturer: topics entertainment | price: 6.99
8. title: pdf converter 4 | description: scansoft pdf converter 4 instantly converts pdf files into documents spreadsheets and forms that look exactly  | manufacturer: nuance | price: 49.95
9. title: individual small business advantage deluxe 2006 | description: with small business advantage deluxe 2006 is a complete set of marketing tools for entrepreneurs & small busin | manufacturer: individual | price: 49.99
10. title: instant landscaping 3.0 [lb] | description: instant landscape design 3.0 is the perfect tool for creating the perfect garden yard or landscape. draw the p | manufacturer: topics entertainment | price: 19.99
11. title: instant play electric guitar deluxe | description: with play electric guitar beginners who want a fast flexible computer-based teaching tool have the in-depth in | manufacturer: topics entertainment | price: 39.99
```

### instruction
```
You are debugging output of a blocking system for entity resolution. Two blocks wrongly separated at least one true match: list A holds records from table A, list B holds records from table B. A confirmed true match split across the two blocks given as SEED. Find OTHER true matches between list A and list B — pairs referring to same real-world entity. Return JSON {"matches": [[a_number, b_number], ...]} using list numbers. Do not include seed pair. If none, return empty list.

How to judge a candidate pair:

1. Identify which fields carry the entity's identity (e.g. name/title) versus fields that only describe context, category, or provenance (e.g. venue, source, publisher, format, channel). Two records can have near-identical identity content yet totally different context fields — expected, does NOT disqualify a match; SEED usually shows this pattern.

2. A shared rare or unusual token (distinctive word, code, number, proper name only a few records in block share) is much stronger evidence than shared common/generic words. Scan for overlaps across ALL fields, not just primary identity field. But single shared token outside identity field not enough alone — see rule 7.

3. Field order and role can vary between the two lists (multiple co-authors, attributes, values listed in different order, or split across differently-named fields). Compare by matching each field's role/content to its counterpart, not by raw string position.

4. Treat different surface renditions of the same underlying content as equivalent, not as evidence against a match: different capitalization, punctuation, whitespace, escaping/encoding of special characters, or spelled-out vs symbolic form of the same value. Do not let a surface-formatting difference cause you to reject an otherwise strong identity-field match, and do not let it cause you to miss one either — check whether the two fields would read the same once such differences are stripped away, before ruling a pair out.

5. Watch for variant or edition markers: version numbers, revision letters, model/part numbers, year/date stamps, size/quantity, or "based on / derived from X" wording. Two records with same base name but DIFFERENT variant/edition markers usually refer to DIFFERENT real-world entities — do not match just because most text overlaps.

6. Do not infer a match from partial identity-field overlap alone when other fields (numbers, dates, secondary names, quantities, counts) point to a different specific item. A short shared phrase or category word across otherwise unrelated records is not evidence.

7. When a candidate pair's strongest evidence is a single field outside the identity field (e.g. one shared secondary name, code, or category), only accept it if the identity fields are also at least plausibly consistent with same entity. A shared secondary field alone — especially short, common, or frequently-repeated value recurring across many unrelated records in block — is not sufficient by itself; prefer pairs where multiple independent fields agree over pairs with only one strong-looking overlap.

8. Work through ENTIRE list A against ENTIRE list B systematically. Never match records because they occupy same or nearby position/index in their lists, never chain a guess off an already-matched pair's position — position carries zero evidence. Every candidate pair must be independently justified by identity-field and token evidence alone.

9. Do not pad output to hit a fixed or "expected" number of matches. Normal and correct for many list-A records to have no counterpart in list B — leave unmatched rather than assign low-confidence guess. Only include a pair when evidence in rules 1–7 clearly supports it.
```

### filter_code
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

### instruction: diff vs parent #11
```diff
--- 
+++ 
@@ -1,13 +1,14 @@
-You are debugging the output of a blocking system for entity resolution. Two blocks that wrongly separated at least one true match are shown: list A holds one block's records from table A, list B holds the other block's records from table B. A confirmed true match split across the two blocks is given as the SEED. Find OTHER true matches between list A and list B: pairs referring to the same real-world entity. Return JSON {"matches": [[a_number, b_number], ...]} using the list numbers. Do not include the seed pair. If none, return an empty list.
+You are debugging output of a blocking system for entity resolution. Two blocks wrongly separated at least one true match: list A holds records from table A, list B holds records from table B. A confirmed true match split across the two blocks given as SEED. Find OTHER true matches between list A and list B — pairs referring to same real-world entity. Return JSON {"matches": [[a_number, b_number], ...]} using list numbers. Do not include seed pair. If none, return empty list.
 
 How to judge a candidate pair:
-1. Identify which fields carry the entity's identity (e.g. a name/title field) versus fields that only describe context, category, or provenance (e.g. venue, source, publisher, format, listing channel). Two records can have identical or near-identical identity fields yet come from completely different context fields — that is expected and does NOT disqualify a match; the SEED itself usually shows this pattern.
 
-2. A shared rare or unusual token (a distinctive word, code, number, or proper name that only a few records in the block share) is much stronger evidence than shared common/generic words. Scan for these overlaps across ALL fields, not just the primary identity field. But a single shared token outside the identity field is not enough on its own — see rule 7.
+1. Identify which fields carry the entity's identity (e.g. name/title) versus fields that only describe context, category, or provenance (e.g. venue, source, publisher, format, channel). Two records can have near-identical identity content yet totally different context fields — expected, does NOT disqualify a match; SEED usually shows this pattern.
 
-3. Field order and role can vary between the two lists (multiple co-authors, attributes, or values listed in different order, or split across differently-named fields). Compare by matching each field's role/content to its counterpart, not by raw string position.
+2. A shared rare or unusual token (distinctive word, code, number, proper name only a few records in block share) is much stronger evidence than shared common/generic words. Scan for overlaps across ALL fields, not just primary identity field. But single shared token outside identity field not enough alone — see rule 7.
 
-4. Treat different renderings of the same underlying text as equivalent, not as evidence against a match: different capitalization, punctuation, whitespace, escaped or encoded special characters, or spelled-out vs. symbolic forms of the same character. Do not let a surface-formatting difference cause you to reject an otherwise strong identity-field match, and do not let it cause you to miss one either — check whether decoding/normalizing the text would make two fields read the same before ruling a pair out.
+3. Field order and role can vary between the two lists (multiple co-authors, attributes, values listed in different order, or split across differently-named fields). Compare by matching each field's role/content to its counterpart, not by raw string position.
 
-5. Watch for variant or edition markers: version numbers, revision letters, model/part numbers, year/date stamps, size/quantity, or "based on / derived from X" phrasing. Two records with the same base name but DIFFERENT variant/edition markers usually refer to DIFFERENT real-world entities — do not match them just because most of the text overlaps.
+4. Treat different surface renditions of the same underlying content as equivalent, not as evidence against a match: different capitalization, punctuation, whitespace, escaping/encoding of special characters, or spelled-out vs symbolic form of the same value. Do not let a surface-formatting difference cause you to reject an otherwise strong identity-field match, and do not let it cause you to miss one either — check whether the two fields would read the same once such differences are stripped away, before ruling a pair out.
+
+5. Watch for variant or edition markers: version numbers, revision letters, model/part numbers, year/date stamps, size/quantity, or "based on / derived from X" wording. Two records with same base name but DIFFERENT variant/edition markers usually refer to DIFFERENT real-world entities — do not match just because most text overlaps.
 
@@ -15,6 +16,6 @@
 
-7. When a candidate pair's strongest evidence is a single field outside the identity field (e.g. one shared secondary name, code, or category), only accept it if the identity fields are also at least plausibly consistent with the same entity. A shared secondary field alone — especially a short, common, or frequently-repeated value that recurs across many unrelated records in the block — is not sufficient by itself; prefer pairs where multiple independent fields agree over pairs with only one strong-looking overlap.
+7. When a candidate pair's strongest evidence is a single field outside the identity field (e.g. one shared secondary name, code, or category), only accept it if the identity fields are also at least plausibly consistent with same entity. A shared secondary field alone — especially short, common, or frequently-repeated value recurring across many unrelated records in block — is not sufficient by itself; prefer pairs where multiple independent fields agree over pairs with only one strong-looking overlap.
 
-8. Work through the ENTIRE list A against the ENTIRE list B systematically. Never match records because they occupy the same or a nearby position/index in their lists, and never chain a guess off an already-matched pair's position — position carries zero evidence. Every candidate pair must be independently justified by identity-field and token evidence alone.
+8. Work through ENTIRE list A against ENTIRE list B systematically. Never match records because they occupy same or nearby position/index in their lists, never chain a guess off an already-matched pair's position — position carries zero evidence. Every candidate pair must be independently justified by identity-field and token evidence alone.
 
-9. Do not pad the output to produce a fixed or "expected" number of matches. It is normal and correct for many list-A records to have no counterpart in list B — leave them unmatched rather than assigning a low-confidence guess. Only include a pair when the evidence in rules 1–7 clearly supports it.
+9. Do not pad output to hit a fixed or "expected" number of matches. Normal and correct for many list-A records to have no counterpart in list B — leave unmatched rather than assign low-confidence guess. Only include a pair when evidence in rules 1–7 clearly supports it.
```

---

## Checkpoint #16 — val 0.298 (+9% relative), rollout 1364, mutated: instruction

**Filter output on the reference episode:** showed 23 A-records and 23 B-records, covering 6/6 hidden targets.

```
## List A — records from block X (23 of 100 in block)
1. title: punch! master landscape & home design | description: master landscape & home design offers nine programs on one easy-to-use interface to help you design home exter | manufacturer: punch! software | price: 59.99
2. title: pinnacle mobile media organizer | description: package contents: mobile media organizer cd media mobile master quick install guide pinnacle's mobile media or | manufacturer: pinnacle | price: 49.99
3. title: dk rainbow fish most beautiful fish in the ocean | description: one stormy night while rainbow fish slept on the ocean floor his beautiful sparkling scales were stolen by thr | manufacturer: encore | price: 9.99
4. title: project x project management software | description: project x is truly project management software for the rest of us. its for those of us that want to spend more | manufacturer: marware | price: 199.95
5. title: punch! master landscape & home design | description: plan out your perfect home & garden with the set of applications available here! / for windows take a virtual  | manufacturer: punch! software | price: 99.99
6. title: adobe photoshop cs2 fundamental techniques by julieanne kost | description: learn essential skills that will become a foundation for a creative and efficient workflow. you will gain a co | manufacturer: software cinema | price: 0
7. title: instant play electric guitar (jewel box) | description: instant play electric guitar hits the right note for flexible computer-based instruction. this 2-cd set takes  | manufacturer: topics entertainment | price: 6.99
8. title: pdf converter 4 | description: scansoft pdf converter 4 instantly converts pdf files into documents spreadsheets and forms that look exactly  | manufacturer: nuance | price: 49.95
9. title: individual small business advantage deluxe 2006 | description: with small business advantage deluxe 2006 is a complete set of marketing tools for entrepreneurs & small busin | manufacturer: individual | price: 49.99
10. title: instant landscaping 3.0 [lb] | description: instant landscape design 3.0 is the perfect tool for creating the perfect garden yard or landscape. draw the p | manufacturer: topics entertainment | price: 19.99
11. title: instant play electric guitar deluxe | description: with play electric guitar beginners who want a fast flexible computer-based teaching tool have the in-depth in | manufacturer: topics entertainment | price: 39.99
```

### instruction
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

### filter_code
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

### instruction: diff vs parent #14
```diff
--- 
+++ 
@@ -10,3 +10,3 @@
 
-4. Treat different surface renditions of the same underlying content as equivalent, not as evidence against a match: different capitalization, punctuation, whitespace, escaping/encoding of special characters, or spelled-out vs symbolic form of the same value. Do not let a surface-formatting difference cause you to reject an otherwise strong identity-field match, and do not let it cause you to miss one either — check whether the two fields would read the same once such differences are stripped away, before ruling a pair out.
+4. Treat different surface forms of the same underlying content as equivalent, not as evidence against a match: different capitalization, punctuation, whitespace, encoding of special characters, or spelled-out vs symbolic form of the same value. Do not let a surface-formatting difference cause you to reject an otherwise strong identity-field match, and do not let it cause you to miss one either — check whether the two fields would read the same once such differences are stripped away, before ruling a pair out.
 
```

---

## Checkpoint #17 — val 0.300 (+1% relative), rollout 1443, mutated: filter_code

**Filter output on the reference episode:** showed 22 A-records and 26 B-records, covering 1/6 hidden targets.

```
## List A — records from block X (22 of 100 in block)
1. title: punch! master landscape & home design | description: master landscape & home design offers nine programs on one easy-to-use interface to help you design home exter | manufacturer: punch! software | price: 59.99
2. title: pinnacle mobile media organizer | description: package contents: mobile media organizer cd media mobile master quick install guide pinnacle's mobile media or | manufacturer: pinnacle | price: 49.99
3. title: dk rainbow fish most beautiful fish in the ocean | description: one stormy night while rainbow fish slept on the ocean floor his beautiful sparkling scales were stolen by thr | manufacturer: encore | price: 9.99
4. title: project x project management software | description: project x is truly project management software for the rest of us. its for those of us that want to spend more | manufacturer: marware | price: 199.95
5. title: punch! master landscape & home design | description: plan out your perfect home & garden with the set of applications available here! / for windows take a virtual  | manufacturer: punch! software | price: 99.99
6. title: adobe photoshop cs2 fundamental techniques by julieanne kost | description: learn essential skills that will become a foundation for a creative and efficient workflow. you will gain a co | manufacturer: software cinema | price: 0
7. title: instant play electric guitar (jewel box) | description: instant play electric guitar hits the right note for flexible computer-based instruction. this 2-cd set takes  | manufacturer: topics entertainment | price: 6.99
8. title: pdf converter 4 | description: scansoft pdf converter 4 instantly converts pdf files into documents spreadsheets and forms that look exactly  | manufacturer: nuance | price: 49.95
9. title: individual small business advantage deluxe 2006 | description: with small business advantage deluxe 2006 is a complete set of marketing tools for entrepreneurs & small busin | manufacturer: individual | price: 49.99
10. title: instant play electric guitar deluxe | description: with play electric guitar beginners who want a fast flexible computer-based teaching tool have the in-depth in | manufacturer: topics entertainment | price: 39.99
11. title: intuit quicken home inventory manager - windows | description: quicken home inventory manager makes it easy to keep an organized record of your belongings so you are ready f | manufacturer: intuit | price: 34.95
```

### instruction
```
You are debugging the output of a blocking system for entity resolution. Two blocks that wrongly separated at least one true match are shown: list A holds one block's records from table A, list B holds the other block's records from table B. A confirmed true match split across the two blocks is given as the SEED. Find OTHER true matches between list A and list B: pairs referring to the same real-world entity. Return JSON {"matches": [[a_number, b_number], ...]} using the list numbers. Do not include the seed pair. If none, return an empty list.

How to judge a candidate pair:
1. Identify which fields carry the entity's identity (e.g. a name/title field) versus fields that only describe context, category, or provenance (e.g. venue, source, publisher, format, listing channel). Two records can have identical or near-identical identity fields yet come from completely different context fields — that is expected and does NOT disqualify a match; the SEED itself usually shows this pattern.

2. A shared rare or unusual token (a distinctive word, code, number, or proper name that only a few records in the block share) is much stronger evidence than shared common/generic words. Scan for these overlaps across ALL fields, not just the primary identity field. But a single shared token outside the identity field is not enough on its own — see rule 7.

3. Field order and role can vary between the two lists (multiple co-authors, attributes, or values listed in different order, or split across differently-named fields). Compare by matching each field's role/content to its counterpart, not by raw string position.

4. Treat different renderings of the same underlying text as equivalent, not as evidence against a match: different capitalization, punctuation, whitespace, escaped or encoded special characters, or spelled-out vs. symbolic forms of the same character. Do not let a surface-formatting difference cause you to reject an otherwise strong identity-field match, and do not let it cause you to miss one either — check whether decoding/normalizing the text would make two fields read the same before ruling a pair out.

5. Watch for variant or edition markers: version numbers, revision letters, model/part numbers, year/date stamps, size/quantity, or "based on / derived from X" phrasing. Two records with the same base name but DIFFERENT variant/edition markers usually refer to DIFFERENT real-world entities — do not match them just because most of the text overlaps.

6. Do not infer a match from partial identity-field overlap alone when other fields (numbers, dates, secondary names, quantities, counts) point to a different specific item. A short shared phrase or category word across otherwise unrelated records is not evidence.

7. When a candidate pair's strongest evidence is a single field outside the identity field (e.g. one shared secondary name, code, or category), only accept it if the identity fields are also at least plausibly consistent with the same entity. A shared secondary field alone — especially a short, common, or frequently-repeated value that recurs across many unrelated records in the block — is not sufficient by itself; prefer pairs where multiple independent fields agree over pairs with only one strong-looking overlap.

8. Work through the ENTIRE list A against the ENTIRE list B systematically. Never match records because they occupy the same or a nearby position/index in their lists, and never chain a guess off an already-matched pair's position — position carries zero evidence. Every candidate pair must be independently justified by identity-field and token evidence alone.

9. Do not pad the output to produce a fixed or "expected" number of matches. It is normal and correct for many list-A records to have no counterpart in list B — leave them unmatched rather than assigning a low-confidence guess. Only include a pair when the evidence in rules 1–7 clearly supports it.
```

### filter_code
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

    # Best-match score AND best-match partner id per record. Raw
    # similarity can be inflated by generic shared structure across
    # many unrelated records (common field shapes, shared prefixes),
    # which makes absolute score alone a weak discriminator when many
    # candidates tie near the top. Tracking *which* record produced
    # the best score lets us detect reciprocal best-matches (rx's top
    # pick is ry AND ry's top pick is rx), a much sharper signal that
    # a pair is a genuine correspondence rather than shared noise.
    MAX_CALLS = 150000

    best_x_score = {r["id"]: 0.0 for r in others_x}
    best_x_partner = {r["id"]: None for r in others_x}
    best_y_score = {r["id"]: 0.0 for r in others_y}
    best_y_partner = {r["id"]: None for r in others_y}

    if n_x and n_y:
        full_cost = n_x * n_y
        if full_cost <= MAX_CALLS:
            for rx in others_x:
                xid = rx["id"]
                best = 0.0
                best_id = None
                for ry in others_y:
                    yid = ry["id"]
                    s = sim(xid, yid)
                    if s > best:
                        best = s
                        best_id = yid
                    if s > best_y_score[yid]:
                        best_y_score[yid] = s
                        best_y_partner[yid] = xid
                best_x_score[xid] = best
                best_x_partner[xid] = best_id
        else:
            k_y = max(1, MAX_CALLS // (2 * n_x))
            k_x = max(1, MAX_CALLS // (2 * n_y))
            sample_y = rng.sample(others_y, min(k_y, n_y))
            sample_x = rng.sample(others_x, min(k_x, n_x))
            for rx in others_x:
                xid = rx["id"]
                best = 0.0
                best_id = None
                for ry in sample_y:
                    s = sim(xid, ry["id"])
                    if s > best:
                        best = s
                        best_id = ry["id"]
                best_x_score[xid] = best
                best_x_partner[xid] = best_id
            for ry in others_y:
                yid = ry["id"]
                best = 0.0
                best_id = None
                for rx in sample_x:
                    s = sim(rx["id"], yid)
                    if s > best:
                        best = s
                        best_id = rx["id"]
                best_y_score[yid] = best
                best_y_partner[yid] = best_id

    # Fold in direct relevance to the confirmed seed's counterpart —
    # still a useful boost for near-duplicates of the known match.
    seed_score_x = {r["id"]: sim(r["id"], seed_b) for r in others_x}
    seed_score_y = {r["id"]: sim(seed_a, r["id"]) for r in others_y}

    def is_reciprocal_x(xid):
        yid = best_x_partner.get(xid)
        if yid is None:
            return False
        return best_y_partner.get(yid) == xid

    def is_reciprocal_y(yid):
        xid = best_y_partner.get(yid)
        if xid is None:
            return False
        return best_x_partner.get(xid) == yid

    def score_x(r):
        rid = r["id"]
        base = max(best_x_score.get(rid, 0.0), seed_score_x.get(rid, 0.0))
        recip = 1.0 if is_reciprocal_x(rid) else 0.0
        return (recip, base)

    def score_y(r):
        rid = r["id"]
        base = max(best_y_score.get(rid, 0.0), seed_score_y.get(rid, 0.0))
        recip = 1.0 if is_reciprocal_y(rid) else 0.0
        return (recip, base)

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

### filter_code: diff vs parent #11
```diff
--- 
+++ 
@@ -31,10 +31,16 @@
 
-    # Best-match score per record: does it look like it has a partner
-    # on the other side at all (not just near the seed)? Full
-    # bipartite scoring when cheap; bounded random sampling otherwise
-    # to stay safely inside the time limit on very large blocks.
+    # Best-match score AND best-match partner id per record. Raw
+    # similarity can be inflated by generic shared structure across
+    # many unrelated records (common field shapes, shared prefixes),
+    # which makes absolute score alone a weak discriminator when many
+    # candidates tie near the top. Tracking *which* record produced
+    # the best score lets us detect reciprocal best-matches (rx's top
+    # pick is ry AND ry's top pick is rx), a much sharper signal that
+    # a pair is a genuine correspondence rather than shared noise.
     MAX_CALLS = 150000
 
-    best_x = {r["id"]: 0.0 for r in others_x}
-    best_y = {r["id"]: 0.0 for r in others_y}
+    best_x_score = {r["id"]: 0.0 for r in others_x}
+    best_x_partner = {r["id"]: None for r in others_x}
+    best_y_score = {r["id"]: 0.0 for r in others_y}
+    best_y_partner = {r["id"]: None for r in others_y}
 
@@ -46,2 +52,3 @@
                 best = 0.0
+                best_id = None
                 for ry in others_y:
@@ -51,5 +58,8 @@
                         best = s
-                    if s > best_y[yid]:
-                        best_y[yid] = s
-                best_x[xid] = best
+                        best_id = yid
+                    if s > best_y_score[yid]:
+                        best_y_score[yid] = s
+                        best_y_partner[yid] = xid
+                best_x_score[xid] = best
+                best_x_partner[xid] = best_id
         else:
@@ -62,2 +72,3 @@
                 best = 0.0
+                best_id = None
                 for ry in sample_y:
@@ -66,3 +77,5 @@
                         best = s
-                best_x[xid] = best
+                        best_id = ry["id"]
+                best_x_score[xid] = best
+                best_x_partner[xid] = best_id
             for ry in others_y:
@@ -70,2 +83,3 @@
                 best = 0.0
+                best_id = None
                 for rx in sample_x:
@@ -74,3 +88,5 @@
                         best = s
-                best_y[yid] = best
+                        best_id = rx["id"]
+                best_y_score[yid] = best
+                best_y_partner[yid] = best_id
 
@@ -81,5 +97,19 @@
 
+    def is_reciprocal_x(xid):
+        yid = best_x_partner.get(xid)
+        if yid is None:
+            return False
+        return best_y_partner.get(yid) == xid
+
+    def is_reciprocal_y(yid):
+        xid = best_y_partner.get(yid)
+        if xid is None:
+            return False
+        return best_x_partner.get(xid) == yid
+
     def score_x(r):
         rid = r["id"]
-        return max(best_x.get(rid, 0.0), seed_score_x.get(rid, 0.0))
```

---

## Checkpoint #21 — val 0.329 (+10% relative), rollout 1735, mutated: filter_code

**Filter output on the reference episode:** showed 20 A-records and 24 B-records, covering 1/6 hidden targets.

```
## List A — records from block X (20 of 100 in block)
1. title: punch! master landscape & home design | description: master landscape & home design offers nine programs on one easy-to-use interface to help you design home exter | manufacturer: punch! software | price: 59.99
2. title: route 66 route canada (french) | description: door to door route planning for canada and the us. over 4 000 000 points-of-interest more than 11 500 000 kilo | manufacturer: csdc | price: 0
3. title: pinnacle mobile media organizer | description: package contents: mobile media organizer cd media mobile master quick install guide pinnacle's mobile media or | manufacturer: pinnacle | price: 49.99
4. title: adobe photoshop cs2 fundamental techniques by julieanne kost | description: learn essential skills that will become a foundation for a creative and efficient workflow. you will gain a co | manufacturer: software cinema | price: 0
5. title: dk rainbow fish most beautiful fish in the ocean | description: one stormy night while rainbow fish slept on the ocean floor his beautiful sparkling scales were stolen by thr | manufacturer: encore | price: 9.99
6. title: instant play electric guitar (jewel box) | description: instant play electric guitar hits the right note for flexible computer-based instruction. this 2-cd set takes  | manufacturer: topics entertainment | price: 6.99
7. title: kaplan sat/act/psat platinum 2007 win/mac | description: kaplan has helped more than 3 million students score higher on admission exams and get into the nation's top c | manufacturer: topics entertainment | price: 49.99
8. title: sentinel: descendants in time | description: sentinel: descendants in time gives you a chance to rediscover the treasures of an ancient people. the tastan  | manufacturer: dreamcatcher interactive | price: 19.99
9. title: instant landscaping 3.0 [lb] | description: instant landscape design 3.0 is the perfect tool for creating the perfect garden yard or landscape. draw the p | manufacturer: topics entertainment | price: 19.99
10. title: panda internet security 2007 3-user | description: format: win 98 me 2000 xp | manufacturer: panda software | price: 69.95
11. title: instant home design (jewel case) | description: instant home design (win 95 98 me nt 2000 xp) | manufacturer: topics entertainment | price: 9.99
```

### instruction
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

### filter_code
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

    def best_matches(word_sets, other_index, other_count, other_n):
        # for each record, its single strongest cross-block candidate
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
                for cand_id in postings:
                    acc[cand_id] += weight
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

    # mutual nearest-neighbor pairs: when x's top cross-block match is
    # y and y's top cross-block match is x, that's a strong signal of
    # a true pair unrelated to the seed. Surface both members early on
    # each side's list so they land together inside the shared budget
    # window instead of being ranked independently and split apart.
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
```

### filter_code: diff vs parent #9
```diff
--- 
+++ 
@@ -44,11 +44,11 @@
 
-    def doc_freq(word_sets):
-        df = defaultdict(int)
+    def doc_count(word_sets):
+        count = defaultdict(int)
         for ws in word_sets.values():
             for w in ws:
-                df[w] += 1
-        return df
+                count[w] += 1
+        return count
 
-    df_x = doc_freq(word_sets_x)
-    df_y = doc_freq(word_sets_y)
+    count_x = doc_count(word_sets_x)
+    count_y = doc_count(word_sets_y)
 
@@ -57,3 +57,3 @@
 
-    def build_index(word_sets, df, cap):
+    def build_index(word_sets, count, cap):
         index = defaultdict(list)
@@ -61,3 +61,3 @@
             for w in ws:
-                if df[w] <= cap:
+                if count[w] <= cap:
                     index[w].append(rid)
@@ -65,10 +65,14 @@
 
-    index_x = build_index(word_sets_x, df_x, cap_x)
-    index_y = build_index(word_sets_y, df_y, cap_y)
+    index_x = build_index(word_sets_x, count_x, cap_x)
+    index_y = build_index(word_sets_y, count_y, cap_y)
 
-    def idf(df, n, w):
-        return math.log((n + 1) / (df[w] + 1)) + 1.0
+    def idf(count, n, w):
+        return math.log((n + 1) / (count[w] + 1)) + 1.0
 
-    def best_overlap_scores(word_sets, other_index, other_df, other_n):
-        scores = {}
+    def best_matches(word_sets, other_index, other_count, other_n):
+        # for each record, its single strongest cross-block candidate
+        # (id + score) — used both for relevance ranking and to find
+        # reciprocal (mutual nearest-neighbor) pairs below.
+        best_id = {}
+        best_score = {}
         for rid, ws in word_sets.items():
@@ -79,29 +83,59 @@
                     continue
-                weight = idf(other_df, other_n, w)
-                for oid in postings:
-                    acc[oid] += weight
-            scores[rid] = max(acc.values()) if acc else 0.0
-        return scores
+                weight = idf(other_count, other_n, w)
+                for cand_id in postings:
+                    acc[cand_id] += weight
+            if acc:
+                top_id = max(acc, key=acc.get)
+                best_id[rid] = top_id
+                best_score[rid] = acc[top_id]
+            else:
+                best_id[rid] = None
+                best_score[rid] = 0.0
+        return best_id, best_score
 
-    # relevance of each x record = strength of its best word-overlap
-    # match on the y side (and symmetrically for y) — catches true
-    # pairs unrelated to the seed, not just seed-adjacent ones.
-    relevance_x = best_overlap_scores(word_sets_x, index_y, df_y, n_y)
-    relevance_y = best_overlap_scores(word_sets_y, index_x, df_x, n_x)
+    best_id_x, best_score_x = best_matches(word_sets_x, index_y, count_y, n_y)
+    best_id_y, best_score_y = best_matches(word_sets_y, index_x, count_x, n_x)
 
-    max_rel_x = max(relevance_x.values()) if relevance_x else 0.0
-    max_rel_y = max(relevance_y.values()) if relevance_y else 0.0
```

---

## Checkpoint #28 — val 0.335 (+2% relative), rollout 2420, mutated: filter_code

**Filter output on the reference episode:** showed 22 A-records and 26 B-records, covering 1/6 hidden targets.

```
## List A — records from block X (22 of 100 in block)
1. title: punch! master landscape & home design | description: master landscape & home design offers nine programs on one easy-to-use interface to help you design home exter | manufacturer: punch! software | price: 59.99
2. title: pinnacle mobile media organizer | description: package contents: mobile media organizer cd media mobile master quick install guide pinnacle's mobile media or | manufacturer: pinnacle | price: 49.99
3. title: dk rainbow fish most beautiful fish in the ocean | description: one stormy night while rainbow fish slept on the ocean floor his beautiful sparkling scales were stolen by thr | manufacturer: encore | price: 9.99
4. title: project x project management software | description: project x is truly project management software for the rest of us. its for those of us that want to spend more | manufacturer: marware | price: 199.95
5. title: punch! master landscape & home design | description: plan out your perfect home & garden with the set of applications available here! / for windows take a virtual  | manufacturer: punch! software | price: 99.99
6. title: adobe photoshop cs2 fundamental techniques by julieanne kost | description: learn essential skills that will become a foundation for a creative and efficient workflow. you will gain a co | manufacturer: software cinema | price: 0
7. title: instant play electric guitar (jewel box) | description: instant play electric guitar hits the right note for flexible computer-based instruction. this 2-cd set takes  | manufacturer: topics entertainment | price: 6.99
8. title: pdf converter 4 | description: scansoft pdf converter 4 instantly converts pdf files into documents spreadsheets and forms that look exactly  | manufacturer: nuance | price: 49.95
9. title: individual small business advantage deluxe 2006 | description: with small business advantage deluxe 2006 is a complete set of marketing tools for entrepreneurs & small busin | manufacturer: individual | price: 49.99
10. title: instant play electric guitar deluxe | description: with play electric guitar beginners who want a fast flexible computer-based teaching tool have the in-depth in | manufacturer: topics entertainment | price: 39.99
11. title: intuit quicken home inventory manager - windows | description: quicken home inventory manager makes it easy to keep an organized record of your belongings so you are ready f | manufacturer: intuit | price: 34.95
```

### instruction
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

### filter_code
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
    partner_x = {}  # xid -> id of best-matching y record seen so far
    partner_y = {}  # yid -> id of best-matching x record seen so far

    full_mode = False
    if n_x and n_y:
        full_cost = n_x * n_y
        if full_cost <= MAX_CALLS:
            full_mode = True
            for rx in others_x:
                xid = rx["id"]
                best = 0.0
                best_pid = None
                for ry in others_y:
                    yid = ry["id"]
                    s = sim(xid, yid)
                    if s > best:
                        best = s
                        best_pid = yid
                    if s > best_y[yid]:
                        best_y[yid] = s
                        partner_y[yid] = xid
                best_x[xid] = best
                if best_pid is not None:
                    partner_x[xid] = best_pid
        else:
            k_y = max(1, MAX_CALLS // (2 * n_x))
            k_x = max(1, MAX_CALLS // (2 * n_y))
            sample_y = rng.sample(others_y, min(k_y, n_y))
            sample_x = rng.sample(others_x, min(k_x, n_x))
            for rx in others_x:
                xid = rx["id"]
                best = 0.0
                best_pid = None
                for ry in sample_y:
                    yid = ry["id"]
                    s = sim(xid, yid)
                    if s > best:
                        best = s
                        best_pid = yid
                best_x[xid] = best
                if best_pid is not None:
                    partner_x[xid] = best_pid
            for ry in others_y:
                yid = ry["id"]
                best = 0.0
                best_pid = None
                for rx in sample_x:
                    xid = rx["id"]
                    s = sim(xid, yid)
                    if s > best:
                        best = s
                        best_pid = xid
                best_y[yid] = best
                if best_pid is not None:
                    partner_y[yid] = best_pid

    # Reciprocal best-match bonus: a record that names another record
    # as its strongest counterpart, and is named back, is structurally
    # more likely to be a true pair than raw aggregate score alone
    # suggests — especially useful when the aggregate score itself
    # came from thin sampling on a very large side.
    mutual_x = set()
    mutual_y = set()
    for xid, yid in partner_x.items():
        if partner_y.get(yid) == xid:
            mutual_x.add(xid)
            mutual_y.add(yid)

    # Fold in direct relevance to the confirmed seed's counterpart —
    # still a useful boost for near-duplicates of the known match.
    # This is always computed exactly (one sim call per record), so
    # when the aggregate best-match score is only sample-based, trust
    # the seed-relative score more.
    seed_score_x = {r["id"]: sim(r["id"], seed_b) for r in others_x}
    seed_score_y = {r["id"]: sim(seed_a, r["id"]) for r in others_y}

    agg_weight = 1.0 if full_mode else 0.4
    seed_weight = 1.0 if full_mode else 0.9
    MUTUAL_BONUS = 1.0

    def score_x(r):
        rid = r["id"]
        s = max(agg_weight * best_x.get(rid, 0.0),
                seed_weight * seed_score_x.get(rid, 0.0))
        if rid in mutual_x:
            s += MUTUAL_BONUS
        return s

    def score_y(r):
        rid = r["id"]
        s = max(agg_weight * best_y.get(rid, 0.0),
                seed_weight * seed_score_y.get(rid, 0.0))
        if rid in mutual_y:
            s += MUTUAL_BONUS
        return s

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

### filter_code: diff vs parent #16
```diff
--- 
+++ 
@@ -39,3 +39,6 @@
     best_y = {r["id"]: 0.0 for r in others_y}
+    partner_x = {}  # xid -> id of best-matching y record seen so far
+    partner_y = {}  # yid -> id of best-matching x record seen so far
 
+    full_mode = False
     if n_x and n_y:
@@ -43,2 +46,3 @@
         if full_cost <= MAX_CALLS:
+            full_mode = True
             for rx in others_x:
@@ -46,2 +50,3 @@
                 best = 0.0
+                best_pid = None
                 for ry in others_y:
@@ -51,5 +56,9 @@
                         best = s
+                        best_pid = yid
                     if s > best_y[yid]:
                         best_y[yid] = s
+                        partner_y[yid] = xid
                 best_x[xid] = best
+                if best_pid is not None:
+                    partner_x[xid] = best_pid
         else:
@@ -62,7 +71,12 @@
                 best = 0.0
+                best_pid = None
                 for ry in sample_y:
-                    s = sim(xid, ry["id"])
+                    yid = ry["id"]
+                    s = sim(xid, yid)
                     if s > best:
                         best = s
+                        best_pid = yid
                 best_x[xid] = best
+                if best_pid is not None:
+                    partner_x[xid] = best_pid
             for ry in others_y:
@@ -70,7 +84,24 @@
                 best = 0.0
+                best_pid = None
                 for rx in sample_x:
-                    s = sim(rx["id"], yid)
+                    xid = rx["id"]
+                    s = sim(xid, yid)
                     if s > best:
                         best = s
+                        best_pid = xid
                 best_y[yid] = best
+                if best_pid is not None:
+                    partner_y[yid] = best_pid
+
+    # Reciprocal best-match bonus: a record that names another record
+    # as its strongest counterpart, and is named back, is structurally
+    # more likely to be a true pair than raw aggregate score alone
+    # suggests — especially useful when the aggregate score itself
+    # came from thin sampling on a very large side.
+    mutual_x = set()
+    mutual_y = set()
+    for xid, yid in partner_x.items():
+        if partner_y.get(yid) == xid:
+            mutual_x.add(xid)
+            mutual_y.add(yid)
 
@@ -78,2 +109,5 @@
     # still a useful boost for near-duplicates of the known match.
+    # This is always computed exactly (one sim call per record), so
+    # when the aggregate best-match score is only sample-based, trust
+    # the seed-relative score more.
     seed_score_x = {r["id"]: sim(r["id"], seed_b) for r in others_x}
@@ -81,5 +115,13 @@
 
+    agg_weight = 1.0 if full_mode else 0.4
+    seed_weight = 1.0 if full_mode else 0.9
+    MUTUAL_BONUS = 1.0
+
     def score_x(r):
```

---

## Checkpoint #41 — val 0.355 (+6% relative), rollout 3631, mutated: filter_code

**Filter output on the reference episode:** showed 21 A-records and 24 B-records, covering 1/6 hidden targets.

```
## List A — records from block X (21 of 100 in block)
1. title: punch! master landscape & home design | description: master landscape & home design offers nine programs on one easy-to-use interface to help you design home exter | manufacturer: punch! software | price: 59.99
2. title: route 66 route canada (french) | description: door to door route planning for canada and the us. over 4 000 000 points-of-interest more than 11 500 000 kilo | manufacturer: csdc | price: 0
3. title: adobe photoshop cs2 fundamental techniques by julieanne kost | description: learn essential skills that will become a foundation for a creative and efficient workflow. you will gain a co | manufacturer: software cinema | price: 0
4. title: pinnacle mobile media organizer | description: package contents: mobile media organizer cd media mobile master quick install guide pinnacle's mobile media or | manufacturer: pinnacle | price: 49.99
5. title: dk rainbow fish most beautiful fish in the ocean | description: one stormy night while rainbow fish slept on the ocean floor his beautiful sparkling scales were stolen by thr | manufacturer: encore | price: 9.99
6. title: instant architect design suite | description: instant architect design suite (win 98 me nt 2000 xp) | manufacturer: imsi design | price: 29.99
7. title: kaplan sat/act/psat platinum 2007 win/mac | description: kaplan has helped more than 3 million students score higher on admission exams and get into the nation's top c | manufacturer: topics entertainment | price: 49.99
8. title: instant landscaping 3.0 [lb] | description: instant landscape design 3.0 is the perfect tool for creating the perfect garden yard or landscape. draw the p | manufacturer: topics entertainment | price: 19.99
9. title: intellimover transfer your pc deluxe | description: intellimover deluxe simplifies pc file migration saving you time and eliminating frustration. transfer customi | manufacturer: nova development | price: 69.99
10. title: panda internet security 2007 3-user | description: format: win 98 me 2000 xp | manufacturer: panda software | price: 69.95
11. title: encyclopedia britannica deluxe 2008 win/mac | description: designed for adults and students alike encyclopaedia britannica deluxe is a comprehensive reference resource t | manufacturer: avanquest | price: 29.95
```

### instruction
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

### filter_code
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

### filter_code: diff vs parent #34
```diff
--- 
+++ 
@@ -54,4 +54,8 @@
 
-    cap_words_x = max(5, n_x // 20)
-    cap_words_y = max(5, n_y // 20)
+    # Sub-linear cap: small blocks get a generous cutoff (so moderately
+    # shared, still-distinguishing terms stay indexed and reciprocal
+    # true pairs get found), large blocks get bounded growth (keeps
+    # worst-case index/lookup cost in check).
+    cap_words_x = max(5, int(2 * math.sqrt(n_x)))
+    cap_words_y = max(5, int(2 * math.sqrt(n_y)))
 
```
