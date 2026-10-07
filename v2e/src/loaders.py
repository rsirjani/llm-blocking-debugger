"""Dataset loaders for all 13 staged blocking benchmarks.

Every loader returns a 4-tuple:

    (tableA_df, tableB_df_or_None, gold_pairs_set, cluster_ids_or_None)

- tableA_df / tableB_df: pandas DataFrames with the record id column named as in
  the source files (see ID_COLUMNS). Id values are coerced to str.
- tableB_df_or_None: None for dirty-ER / multi-source datasets (cora,
  musicbrainz-*, ncvr).
- gold_pairs_set: set of (idA, idB) tuples (str, str). For two-table datasets
  the tuple order is (tableA id, tableB id). For single-table datasets pairs
  are normalized with sorted() so each undirected pair appears exactly once.
- cluster_ids_or_None: pandas Series mapping record id (str, index) ->
  cluster id (str), only for cluster-ground-truth datasets (musicbrainz-*,
  ncvr); None otherwise. wdc-block deliberately returns None: its tables carry
  a cluster_id column but the manifest flags it as inconsistent with the pair
  labels -- ONLY the pair labels are gold.

All quirks are taken from each dataset's DOWNLOAD_MANIFEST.json:
- Leipzig tableA files (Abt, Amazon, DBLP1, DBLP2) are latin-1;
  GoogleProducts is also latin-1; Scholar.csv is utf-8-sig (BOM); rest utf-8.
- walmart-amazon: gold id1/id2 reference the custom_id column, NOT id.
- cora: pipe-delimited with a trailing '|' producing an empty 14th column
  (dropped); gold is pipe-delimited id pairs without header.
- musicbrainz-*: CID column is the cluster ground truth; TID is the record id.
- ncvr: 5 duplicate-free sources; recid is the cluster id; records with the
  same recid across sources are duplicates.
- beer / itunes-amazon: INCOMPLETE GOLD. Their deepmatcher raw_data ships no
  matches.csv; gold is the positive subset of labeled_data.csv, which labels a
  sample of a Magellan-produced candidate set. Recall measured against it is
  recall over a prior blocker's output, not PC over all true matches -- do not
  pool it with the complete-gold datasets. Tables are latin-1 (beer text is
  already mojibake at source); labeled_data.csv has '#' metadata lines above
  the header. exp_data/ holds a SECOND id space (0-based row index into
  differently ordered tables) that is NOT interchangeable with the raw ids.
- wdc-block: pair labels only, NEVER table cluster_id. Blocking gold is the
  COMPLETE positive union over train_l+valid_l+test (10,126 pairs, identical
  for block_s and block_m per manifest gold_structure/verification_pass2) --
  it does NOT depend on the train_size chosen for supervision. train/valid
  splits of the chosen train_size are supervision-only and available via
  load_wdc_block_splits(), with known test-set leakage rows removed from
  train/valid (4/33/133 rows for train_{s,m,l}). Fields can exceed csv's
  default field limit -> csv.field_size_limit(sys.maxsize).
"""

from __future__ import annotations

import csv
import glob
import io
import itertools
import os
import sys

import pandas as pd

csv.field_size_limit(sys.maxsize)

STAGING = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       os.pardir, "datasets_staging")
STAGING = os.path.abspath(STAGING)

# record-id column per dataset (tableA, tableB) -- for reference by callers
ID_COLUMNS = {
    "abt-buy": ("id", "id"),
    "amazon-google": ("id", "id"),
    "beer": ("Label", "Label"),
    "itunes-amazon": ("Sno", "Sno"),
    "dblp-acm": ("id", "id"),
    "dblp-scholar": ("id", "id"),
    "fodors-zagats": ("id", "id"),
    "walmart-amazon": ("custom_id", "custom_id"),
    "cora": ("Entity Id", None),
    "musicbrainz-20k": ("TID", None),
    "musicbrainz-200k": ("TID", None),
    "ncvr": ("uid", None),
    "wdc-block": ("id", "id"),
}


def _read_csv(path, encoding="utf-8", sep=",", **kw):
    df = pd.read_csv(path, encoding=encoding, sep=sep, dtype=str,
                     keep_default_na=False, **kw)
    return df


def _two_table(dirname, file_a, file_b, gold_file, enc_a, enc_b,
               id_a="id", id_b="id", gold_cols=None):
    root = os.path.join(STAGING, dirname)
    dfa = _read_csv(os.path.join(root, file_a), encoding=enc_a)
    dfb = _read_csv(os.path.join(root, file_b), encoding=enc_b)
    gold_df = _read_csv(os.path.join(root, gold_file), encoding="utf-8")
    if gold_cols is None:
        gold_cols = list(gold_df.columns[:2])
    gold = set(zip(gold_df[gold_cols[0]].astype(str),
                   gold_df[gold_cols[1]].astype(str)))
    dfa[id_a] = dfa[id_a].astype(str)
    dfb[id_b] = dfb[id_b].astype(str)
    return dfa, dfb, gold, None


def load_abt_buy():
    return _two_table("abt-buy", "Abt.csv", "Buy.csv",
                      "abt_buy_perfectMapping.csv", "latin-1", "utf-8")


def load_amazon_google():
    return _two_table("amazon-google", "Amazon.csv", "GoogleProducts.csv",
                      "Amzon_GoogleProducts_perfectMapping.csv",
                      "latin-1", "latin-1")


def load_dblp_acm():
    # tableA = DBLP2.csv (latin-1), tableB = ACM.csv (utf-8)
    return _two_table("dblp-acm", "DBLP2.csv", "ACM.csv",
                      "DBLP-ACM_perfectMapping.csv", "latin-1", "utf-8")


def load_dblp_scholar():
    # Scholar.csv has a UTF-8 BOM -> utf-8-sig
    return _two_table("dblp-scholar", "DBLP1.csv", "Scholar.csv",
                      "DBLP-Scholar_perfectMapping.csv", "latin-1", "utf-8-sig")


def load_fodors_zagats():
    return _two_table("fodors-zagats", "tableA.csv", "tableB.csv",
                      "matches.csv", "utf-8", "utf-8")


def load_walmart_amazon():
    # matches.csv id1/id2 reference custom_id (verified in manifest), not id.
    return _two_table("walmart-amazon", "tableA.csv", "tableB.csv",
                      "matches.csv", "utf-8", "utf-8",
                      id_a="custom_id", id_b="custom_id")


def _magellan_labeled(dirname, file_a, file_b, id_a, id_b, enc="latin-1"):
    """deepmatcher Structured/* sets whose raw_data ships labeled_data.csv.

    Unlike fodors-zagats / walmart-amazon (complete matches.csv), beer and
    itunes-amazon have NO complete gold: labeled_data.csv labels only a sample
    of a Magellan-produced candidate set. Gold = its positive rows (manifest
    gold_structure). Magellan writes '#'-prefixed metadata lines above the
    header, which must be skipped. exp_data/ carries a SECOND, non-interchange-
    able id space (0-based row index into different tables) -- not used here.
    """
    root = os.path.join(STAGING, dirname)
    dfa = _read_csv(os.path.join(root, file_a), encoding=enc)
    dfb = _read_csv(os.path.join(root, file_b), encoding=enc)
    with io.open(os.path.join(root, "labeled_data.csv"),
                 encoding=enc, newline="") as fh:
        body = "".join(ln for ln in fh if not ln.startswith("#"))
    lab = _read_csv(io.StringIO(body))
    gcol = "gold" if "gold" in lab.columns else "label"
    pos = lab[lab[gcol] == "1"]
    gold = set(zip(pos[f"ltable.{id_a}"].astype(str),
                   pos[f"rtable.{id_b}"].astype(str)))
    dfa[id_a] = dfa[id_a].astype(str)
    dfb[id_b] = dfb[id_b].astype(str)
    return dfa, dfb, gold, None


def load_beer():
    return _magellan_labeled("beer", "tableA.csv", "tableB.csv",
                             "Label", "Label")


def load_itunes_amazon():
    return _magellan_labeled("itunes-amazon", "tableA.csv", "tableB.csv",
                             "Sno", "Sno")


def load_cora():
    root = os.path.join(STAGING, "cora")
    df = _read_csv(os.path.join(root, "cora.csv"), sep="|")
    # trailing '|' creates an empty unnamed 14th column -> drop it
    empty_cols = [c for c in df.columns
                  if c.strip() == "" or c.startswith("Unnamed")]
    df = df.drop(columns=empty_cols)
    df["Entity Id"] = df["Entity Id"].astype(str)
    gold = set()
    with open(os.path.join(root, "cora_gt.csv"), encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="|"):
            if not row or not row[0].strip():
                continue
            a, b = row[0].strip(), row[1].strip()
            gold.add(tuple(sorted((a, b))))
    return df, None, gold, None


def _cluster_gold(cluster_series):
    """Undirected within-cluster pairs from a Series: record id -> cluster id."""
    gold = set()
    for _, ids in cluster_series.groupby(cluster_series):
        members = ids.index.tolist()
        if len(members) > 1:
            for a, b in itertools.combinations(sorted(members), 2):
                gold.add((a, b))
    return gold


def _load_musicbrainz(dirname, fname):
    root = os.path.join(STAGING, dirname)
    df = _read_csv(os.path.join(root, "raw", fname))
    df["TID"] = df["TID"].astype(str)
    clusters = pd.Series(df["CID"].astype(str).values,
                         index=df["TID"].values, name="cluster_id")
    gold = _cluster_gold(clusters)
    return df, None, gold, clusters


def load_musicbrainz_20k():
    return _load_musicbrainz("musicbrainz-20k", "musicbrainz-20-A01.csv.dapo")


def load_musicbrainz_200k():
    return _load_musicbrainz("musicbrainz-200k", "musicbrainz-200-A01.csv.dapo")


def load_ncvr():
    """5-party NCVR: concat all sources; uid = 'p<party>_<recid>'; cluster =
    recid; gold = all cross-source pairs sharing a recid."""
    root = os.path.join(STAGING, "ncvr", "5Party-ocp20")
    paths = sorted(glob.glob(os.path.join(root, "ncvr_*_nump_5.csv")))
    parts = []
    for p in paths:
        party = p.split("_myp_")[1].split("_")[0]
        d = _read_csv(p)
        d.insert(0, "uid", "p" + party + "_" + d["recid"].astype(str))
        d.insert(1, "source", party)
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    clusters = pd.Series(df["recid"].astype(str).values,
                         index=df["uid"].values, name="cluster_id")
    gold = _cluster_gold(clusters)
    return df, None, gold, clusters


def _wdc_root(block_size, train_size):
    name = f"wdcproducts80cc20rnd050un_block_{block_size}_train_{train_size}"
    return os.path.join(STAGING, "wdc-block", name)


def _wdc_read_big(path):
    """Read a wdc-block CSV via the csv module (fields > pandas' C limits)."""
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    return pd.DataFrame(rows[1:], columns=rows[0])


def load_wdc_block(block_size="s", train_size="s"):
    """WDC-Block. Gold = the COMPLETE set of true matches in A x B.

    Blocking evaluation (PC) needs the maximal known gold, which by benchmark
    construction is the union of distinct label==1 pairs over the train_l
    config's train+valid+test = 10,126 pairs (manifest gold_structure;
    identical for block_s and block_m -- filler records are non-matching).
    Gold therefore does NOT depend on `train_size`: the train_s/train_m
    positive unions (897 / 2,827) are down-sampled SUPERVISION sets, and using
    them as the PC denominator would grossly overstate recall. `train_size`
    only selects which supervision splits load_wdc_block_splits() returns.

    tableA/tableB/test are byte-identical across train_* configs of the same
    block size, so tables are read from the requested config directory.

    NEVER derive gold from the cluster_id columns (manifest: inconsistent
    between pair files and tables; cluster 2432 alone would inject 305k bogus
    pairs); cluster_ids return value is None.
    """
    root = os.path.join(_wdc_root(block_size, train_size))
    dfa = _wdc_read_big(os.path.join(root, "tableA.csv"))
    dfb = _wdc_read_big(os.path.join(root, "tableB.csv"))

    gold_root = _wdc_root(block_size, "l")  # complete positive union lives here
    gold = set()
    pos_per_file = {}
    for split in ("train", "valid", "test"):
        d = _read_csv(os.path.join(gold_root, f"{split}.csv"))
        pos = d[d["label"] == "1"]
        pos_per_file[split] = len(pos)
        gold |= set(zip(pos["ltable_id"].astype(str),
                        pos["rtable_id"].astype(str)))
    load_wdc_block.last_pos_per_file = pos_per_file  # for auditing
    return dfa, dfb, gold, None


def load_wdc_block_splits(block_size="s", train_size="s"):
    """Supervision splits for a wdc-block config: {'train','valid','test'}.

    Manifest verification_pass2 flags TEST-SET LEAKAGE: some test.csv pairs
    reappear as rows in train/valid of the same block (4 / 33 / 133 rows for
    train_{s,m,l}; labels consistent). Those rows are DROPPED from train and
    valid here so the supervision splits are disjoint from test.
    """
    root = _wdc_root(block_size, train_size)
    splits = {s: _read_csv(os.path.join(root, f"{s}.csv"))
              for s in ("train", "valid", "test")}
    test_pairs = set(zip(splits["test"]["ltable_id"],
                         splits["test"]["rtable_id"]))
    dropped = 0
    for s in ("train", "valid"):
        d = splits[s]
        leak = [pair in test_pairs
                for pair in zip(d["ltable_id"], d["rtable_id"])]
        dropped += sum(leak)
        splits[s] = d[[not x for x in leak]].reset_index(drop=True)
    load_wdc_block_splits.last_leaked_rows = dropped  # for auditing
    return splits


LOADERS = {
    "abt-buy": load_abt_buy,
    "amazon-google": load_amazon_google,
    "beer": load_beer,
    "cora": load_cora,
    "dblp-acm": load_dblp_acm,
    "dblp-scholar": load_dblp_scholar,
    "fodors-zagats": load_fodors_zagats,
    "itunes-amazon": load_itunes_amazon,
    "musicbrainz-20k": load_musicbrainz_20k,
    "musicbrainz-200k": load_musicbrainz_200k,
    "ncvr": load_ncvr,
    "walmart-amazon": load_walmart_amazon,
    "wdc-block": load_wdc_block,
}


# ---------------------------------------------------------------------------
# self-test: load every dataset and assert manifest counts
# ---------------------------------------------------------------------------

# (nA, nB or None, gold count or None, cluster count or None)
_EXPECTED = {
    "abt-buy": (1081, 1092, 1097, None),
    "amazon-google": (1363, 3226, 1300, None),
    # beer / itunes-amazon gold = positives of labeled_data.csv, a labeled
    # SAMPLE of a Magellan candidate set -- NOT a complete match list
    # (manifest gold_structure). Matches the published |M| for both.
    "beer": (4345, 3000, 68, None),
    "cora": (1295, None, 17184, None),
    "dblp-acm": (2616, 2294, 2224, None),
    "dblp-scholar": (2616, 64263, 5347, None),
    "fodors-zagats": (533, 331, 112, None),
    "itunes-amazon": (6907, 55923, 132, None),
    "musicbrainz-20k": (19375, None, 16250, 10000),
    "musicbrainz-200k": (193750, None, 162500, 100000),
    "ncvr": (5000000, None, 3331384, 3500840),
    "walmart-amazon": (2554, 22074, 1154, None),
    # wdc-block gold = complete positive union from train_l config (manifest:
    # 10,126, identical for block_s/block_m); per-file positives also asserted
    "wdc-block": (5000, 5000, 10126, None),
}

_WDC_POS_PER_FILE = {"train": 6454, "valid": 3226, "test": 500}  # train_l
_WDC_SPLIT_LEAKS = {"s": 4, "m": 33, "l": 133}  # rows dropped from train/valid


def _check(name):
    exp_a, exp_b, exp_gold, exp_clusters = _EXPECTED[name]
    dfa, dfb, gold, clusters = LOADERS[name]()

    assert len(dfa) == exp_a, f"{name}: tableA {len(dfa)} != {exp_a}"
    if exp_b is None:
        assert dfb is None, f"{name}: expected single-table, got tableB"
    else:
        assert dfb is not None and len(dfb) == exp_b, \
            f"{name}: tableB {None if dfb is None else len(dfb)} != {exp_b}"

    assert len(gold) == exp_gold, f"{name}: gold {len(gold)} != {exp_gold}"
    gold_note = f"gold={len(gold)}"
    if name == "wdc-block":
        ppf = load_wdc_block.last_pos_per_file
        assert ppf == _WDC_POS_PER_FILE, \
            f"{name}: pos-per-file {ppf} != {_WDC_POS_PER_FILE}"
        # supervision splits: leakage vs test removed, counts per manifest
        for ts, exp_leak in _WDC_SPLIT_LEAKS.items():
            sp = load_wdc_block_splits("s", ts)
            leaked = load_wdc_block_splits.last_leaked_rows
            assert leaked == exp_leak, \
                f"{name}: train_{ts} leaked rows {leaked} != {exp_leak}"
            test_pairs = set(zip(sp["test"]["ltable_id"],
                                 sp["test"]["rtable_id"]))
            for s in ("train", "valid"):
                overlap = test_pairs & set(zip(sp[s]["ltable_id"],
                                               sp[s]["rtable_id"]))
                assert not overlap, f"{name}: {s} still overlaps test"
        gold_note += (f" pos-per-file(train_l)={ppf}"
                      f" split-leaks-removed={_WDC_SPLIT_LEAKS}")

    if exp_clusters is None:
        assert clusters is None, f"{name}: unexpected cluster ids"
        cl_note = ""
    else:
        n_cl = clusters.nunique()
        assert n_cl == exp_clusters, \
            f"{name}: clusters {n_cl} != {exp_clusters}"
        cl_note = f" clusters={n_cl}"

    # referential integrity of gold vs id columns (skip huge ncvr scan: gold
    # ids there are constructed from the very rows they reference)
    if name not in ("ncvr", "musicbrainz-200k"):
        ida, idb = ID_COLUMNS[name]
        ids_a = set(dfa[ida])
        ids_b = set(dfb[idb]) if dfb is not None else ids_a
        bad = [p for p in itertools.islice(iter(gold), 500)
               if p[0] not in ids_a and p[0] not in ids_b
               or p[1] not in ids_b and p[1] not in ids_a]
        assert not bad, f"{name}: gold ids not in tables, e.g. {bad[:3]}"

    nb = "-" if dfb is None else len(dfb)
    print(f"PASS {name}: A={len(dfa)} B={nb} {gold_note}{cl_note}")


if __name__ == "__main__":
    failed = []
    for _name in LOADERS:
        try:
            _check(_name)
        except Exception as e:  # noqa: BLE001
            failed.append((_name, e))
            print(f"FAIL {_name}: {e}")
    print(f"\n{len(LOADERS) - len(failed)}/{len(LOADERS)} datasets passed")
    sys.exit(1 if failed else 0)
