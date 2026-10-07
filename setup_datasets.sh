#!/bin/bash
# ==============================================================================
# Dataset + method scaffolding for entity-matching-llm-blocking
#
# RUN THIS ON THE SERVER, inside tmux/screen (downloads can take a while,
# and an SSH session dying mid-download will kill a bare `bash` run):
#
#   ssh FPATEL59@dblab-gpu.csd.uwo.ca
#   tmux new -s dataset_setup
#   cd /data/project/dblab/rsirjani/entity-matching-llm-blocking
#   bash setup_datasets.sh
#
# (Detach with Ctrl+B then D; reattach later with `tmux attach -t dataset_setup`)
#
# STATUS: covers the datasets I have CONFIRMED working URLs for. The rest
# (cora, musicbrainz, ncvr, wdc-block, abt-buy) come from different sources
# (Leipzig, FAMER, SC-Block/ESWC) that need separate lookup — left as TODOs
# below rather than guessed at.
# ==============================================================================

set -e  # stop on first error, so a bad download doesn't silently continue

BASE="datasets"
mkdir -p "$BASE"

download_and_unzip () {
  local name="$1"
  local url="$2"
  local dir="$BASE/$name"

  if [ -d "$dir" ] && [ "$(ls -A "$dir" 2>/dev/null)" ]; then
    echo "[$name] already exists, skipping"
    return
  fi

  echo "[$name] downloading..."
  mkdir -p "$dir"
  wget -q --show-progress -O "$dir/raw_data.zip" "$url"
  unzip -q "$dir/raw_data.zip" -d "$dir"
  rm "$dir/raw_data.zip"
  echo "[$name] done -> $dir"
}

# ---- Structured / Magellan-deepmatcher datasets (confirmed URLs) ----
download_and_unzip "fodors-zagats"     "http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/Fodors-Zagats/fodors_zagat_raw_data.zip"
download_and_unzip "amazon-google"     "http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/Amazon-Google/amazon_google_raw_data.zip"
download_and_unzip "walmart-amazon"    "http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/Walmart-Amazon/walmart_amazon_raw_data.zip"
download_and_unzip "dblp-acm"          "http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/DBLP-ACM/dblp_acm_raw_data.zip"
download_and_unzip "dblp-scholar"      "http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/DBLP-GoogleScholar/dblp_scholar_raw_data.zip"

# ---- From BLOCKING_PROJECT_OPTIONS.txt (marked as "not yet downloaded") ----
# TODO: confirm these two URLs against the project doc before running —
# copying the pattern from the confirmed ones above, not independently verified:
download_and_unzip "beer"              "http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/Beer/beer_raw_data.zip"
download_and_unzip "itunes-amazon"     "http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/iTunes-Amazon/itunes_amazon_raw_data.zip"

# ---- Abt-Buy (Magellan Textual category, not Structured) ----
download_and_unzip "abt-buy"           "http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Textual/Abt-Buy/abt_buy_raw_data.zip"

# ---- NOT YET RESOLVED — different sources, need lookup before downloading ----
# - cora            (likely Leipzig DBS group repository)
# - musicbrainz     (likely FAMER / Leipzig, dirty-cluster format)
# - ncvr            (NCVR Leipzig 5-party — real voter PII, needs its own
#                     access/handling process, do NOT bulk-download blindly)
# - wdc-block       (WDC / SC-Block, ESWC-related)
echo ""
echo "Remaining unresolved: cora, musicbrainz, ncvr, wdc-block — see script comments."

# ---- Method scaffolding (empty dirs, code goes in per-arm later) ----
# Named per BLOCKING_PROJECT_OPTIONS.txt's actual method names, not bare
# letters, so the folder itself tells you what's inside:
#   A = Standard Blocking, exact-equality key match       -> reclin2
#   B = Bit-sampling LSH, randomised code / exact collision -> blocklib
#   C = KLSH, vector-based geometric proximity              -> klsh
#   D = SimCSE k-NN graph -> Louvain/Leiden community detect -> Mugeni & Amagasa
mkdir -p blocking_methods/A_standard_reclin2
mkdir -p blocking_methods/B_lsh_blocklib
mkdir -p blocking_methods/C_klsh
mkdir -p blocking_methods/D_simcse_louvain

echo ""
echo "Done. Structure so far:"
find "$BASE" -maxdepth 1 -type d
find blocking_methods -maxdepth 1 -type d
