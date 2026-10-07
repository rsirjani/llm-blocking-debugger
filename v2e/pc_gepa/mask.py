"""Reflection-side content masking + prompt leakage audit.

Content token = alphabetic-bearing token that is RARE in its dataset's
record corpus (document frequency below threshold). These are brands,
product/venue/author names, SKUs — memorizable content. Generic tokens
(document-frequent: "deluxe", "edition", "software", "vol") and pure
numbers are kept — they are the domain-general marker vocabulary that
generalizable rules are legitimately about.

Masking is applied ONLY to what the reflection LLM sees; the judge
always gets real records. Placeholders are consistent within one
reflective example (<T1>, <T2>, ...) so cross-record correspondence
stays visible.

Also: audit_prompt(text) -> content tokens leaked into a prompt, the
enforcement basis for the leakage gate.
"""
import re
from collections import defaultdict

_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9'&-]*")

DF_FRACTION = 0.01   # token generic if in >=1% of a dataset's records
DF_MIN = 8           # ...and at least this many records


class ContentLexicon:
    """generic = document-frequent in >=2 DIFFERENT datasets (cross-
    domain vocabulary: 'deluxe', 'edition', 'because', ...). Everything
    else that occurs in records is content ('aspyr' is frequent in
    amazon-google only -> content -> masked)."""

    def __init__(self, banks):
        self.content = set()
        generic_votes = defaultdict(set)
        for (blk, ds), bank in banks.items():
            df = defaultdict(int)
            n = len(bank["records"])
            for fields in bank["records"].values():
                toks = set()
                for v in fields.values():
                    toks.update(t.lower() for t in _TOKEN.findall(str(v)))
                for t in toks:
                    df[t] += 1
            cut = max(DF_MIN, int(n * DF_FRACTION))
            for t, c in df.items():
                if c >= cut:
                    generic_votes[t].add(ds)
                self.content.add(t)
        self.generic = {t for t, dss in generic_votes.items()
                        if len(dss) >= 2}
        self.content -= self.generic

    def is_content(self, token):
        return token.lower() in self.content

    def mask_text(self, text, mapping):
        """Replace content tokens with consistent <Tn> placeholders.
        A token is masked iff it is dataset-rare (content set) AND
        uncommon English (zipf < 3.0) — mirrors audit_prompt, keeps
        scaffold/structural/common vocabulary readable.
        mapping: dict carried across one reflective example."""
        from wordfreq import zipf_frequency

        def sub(m):
            t = m.group(0)
            key = t.lower()
            if (key not in self.content
                    or zipf_frequency(key, "en") >= 3.0):
                return t
            if key not in mapping:
                mapping[key] = f"<T{len(mapping) + 1}>"
            return mapping[key]
        return _TOKEN.sub(sub, text)

    def audit_prompt(self, text):
        """Dataset-content tokens leaked into a prompt: token occurs in
        the record corpora as dataset-rare AND is uncommon English
        (wordfreq zipf < 3.0). Sorted unique list."""
        from wordfreq import zipf_frequency
        hits = {t.lower() for t in _TOKEN.findall(text)
                if self.is_content(t)
                and zipf_frequency(t.lower(), "en") < 3.0}
        return sorted(hits)


if __name__ == "__main__":
    import sys
    from gepa_run5 import load_banks
    banks = load_banks(["lambdafold"])
    lex = ContentLexicon(banks)
    print(f"lexicon: {len(lex.content)} content / "
          f"{len(lex.generic)} generic tokens")
    demo = ("broderbund mavis beacon teaches typing deluxe 17 upgrade "
            "(win/mac) by aspyr media")
    mp = {}
    print("masked:", lex.mask_text(demo, mp))
    if len(sys.argv) > 1:
        text = open(sys.argv[1], encoding="utf-8").read()
        hits = lex.audit_prompt(text)
        print(f"\nAUDIT {sys.argv[1]}: {len(hits)} leaked content "
              f"tokens:\n{hits[:60]}")
