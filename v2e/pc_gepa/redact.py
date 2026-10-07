"""Similarity-preserving redaction of episode records.

Purpose: stop the evolved prompt from memorizing dataset content
(brands, product lines, domain vocabulary) by making a fraction of
training episodes semantics-free while preserving the string-overlap
structure that blocker-split matches are actually found by.

Modes
-----
cipher : per-episode random character permutation. Lowercase letters
         map through one permutation, uppercase mirrors it, digits map
         through a digit permutation; punctuation/whitespace unchanged.
         EXACTLY similarity-isomorphic: every edit distance, n-gram
         overlap, token boundary and length is preserved; meaning is
         destroyed. "mavis beacon 17" -> e.g. "kwzoj xqwnyr 82".
pseudo : token-level pseudonyms. Each distinct alphabetic token maps to
         a deterministic fake word of similar length (episode-
         consistent), digits are permuted. Readable; preserves token
         overlap and rarity; breaks sub-token variant structure.

Both modes also mask field names to attr_1..attr_n (episode-consistent
order) so schema-specific rules can't form.

Determinism: everything is seeded by (global seed, episode_id) — the
same episode always redacts identically, across processes.

Usage: redacted_bank_view(bank, ep, mode, seed) -> (records_view, ep)
where records_view covers exactly the ids the episode touches; feed it
wherever bank["records"] was used for THIS episode (render, feedback,
sim -- sim on ciphered text is isomorphic to sim on the original for
shared tokens and near-variants).
"""
import random
import re
import string

_WORDLIKE = re.compile(r"[A-Za-z]+|[0-9]+")

_CONSONANTS = "bcdfghjklmnpqrstvwz"
_VOWELS = "aeiou"


def _char_cipher(rng):
    letters = list(string.ascii_lowercase)
    perm = letters[:]
    rng.shuffle(perm)
    lmap = dict(zip(letters, perm))
    digits = list(string.digits)
    dperm = digits[:]
    rng.shuffle(dperm)
    dmap = dict(zip(digits, dperm))

    def enc(text):
        out = []
        for ch in text:
            lo = ch.lower()
            if lo in lmap:
                sub = lmap[lo]
                out.append(sub.upper() if ch.isupper() else sub)
            elif ch in dmap:
                out.append(dmap[ch])
            else:
                out.append(ch)
        return "".join(out)
    return enc


def _fake_word(rng, length):
    out = []
    for i in range(max(2, length)):
        out.append(rng.choice(_CONSONANTS if i % 2 == 0 else _VOWELS))
    return "".join(out[:max(2, length)])


def _token_pseudo(rng):
    tok_map = {}
    digits = list(string.digits)
    dperm = digits[:]
    rng.shuffle(dperm)
    dmap = dict(zip(digits, dperm))

    def enc(text):
        def sub(m):
            t = m.group(0)
            if t.isdigit():
                return "".join(dmap[c] for c in t)
            key = t.lower()
            if key not in tok_map:
                w = _fake_word(rng, len(key))
                tok_map[key] = w
            w = tok_map[key]
            return w.capitalize() if t[0].isupper() else w
        return _WORDLIKE.sub(sub, text)
    return enc


def redacted_bank_view(bank, ep, mode="cipher", seed=0):
    """Return (bank_view) whose records for this episode's ids are
    redacted; other bank fields (metadata) pass through unchanged.
    mode None/'none' returns the bank untouched."""
    if not mode or mode == "none":
        return bank
    rng = random.Random(f"redact:{seed}:{ep['episode_id']}:{mode}")
    enc = _char_cipher(rng) if mode == "cipher" else _token_pseudo(rng)
    ids = set(ep["a_ids"]) | set(ep["b_ids"]) | set(ep["seed_pair"])
    field_names = {}

    def mask_fields(fields):
        out = {}
        for k, v in fields.items():
            if k not in field_names:
                field_names[k] = f"attr_{len(field_names) + 1}"
            out[field_names[k]] = enc(str(v))
        return out

    records = dict(bank["records"])
    for rid in ids:
        if rid in records:
            records[rid] = mask_fields(records[rid])
    view = dict(bank)
    view["records"] = records
    view["dataset_metadata"] = ("REDACTED dataset: record content has "
                                "been consistently pseudonymized; match "
                                "by string structure (shared rare "
                                "tokens, near-identical spellings, "
                                "numbers), not by meaning")
    return view


def should_redact(ep, fraction, seed=0):
    """Deterministic per-episode coin flip for augmentation."""
    if fraction <= 0:
        return False
    if fraction >= 1:
        return True
    return (random.Random(f"flip:{seed}:{ep['episode_id']}").random()
            < fraction)


if __name__ == "__main__":
    import json
    import harness5
    from gepa_run5 import load_banks

    banks = load_banks(["lambdafold"])
    bank = banks[("lambdafold", "amazon-google")]
    ep = next(e for e in bank["episodes"] if e["split"] == "train"
              and len(e["targets"]) >= 3)
    sa, sb = ep["seed_pair"]
    tgt = ep["targets"][0]
    print("=== ORIGINAL seed pair")
    print("A:", harness5.fmt_fields(bank["records"][sa]))
    print("B:", harness5.fmt_fields(bank["records"][sb]))
    print("=== ORIGINAL target pair")
    print("A:", harness5.fmt_fields(bank["records"][tgt[0]]))
    print("B:", harness5.fmt_fields(bank["records"][tgt[1]]))
    for mode in ("cipher", "pseudo"):
        v = redacted_bank_view(bank, ep, mode=mode, seed=0)
        print(f"=== {mode.upper()} seed pair")
        print("A:", harness5.fmt_fields(v["records"][sa]))
        print("B:", harness5.fmt_fields(v["records"][sb]))
        print(f"=== {mode.upper()} target pair")
        print("A:", harness5.fmt_fields(v["records"][tgt[0]]))
        print("B:", harness5.fmt_fields(v["records"][tgt[1]]))
