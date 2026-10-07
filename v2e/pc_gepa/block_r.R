#!/usr/bin/env Rscript
# Blockers A (reclin2 standard blocking) and C (klsh) for the run7 bank.
#
# Reads a CSV with columns: id, key, text  (key = derived blocking key
# for method A; text = concatenated record text for method C).
# Writes a CSV with columns: id, block.
#
# usage: block_r.R <method> <in.csv> <out.csv> [param]
#   method A : blocks are the equivalence classes of `key`; reclin2's
#              pair_blocking(on="key") is executed on the same key so the
#              partition and the certified implementation agree.
#   method C : klsh(num.blocks = param); blocks are the k-means LSH
#              clusters over shingled record text.
.libPaths("~/R/library")
args <- commandArgs(trailingOnly = TRUE)
method <- args[1]; infile <- args[2]; outfile <- args[3]
param <- if (length(args) > 3) as.numeric(args[4]) else NA

d <- read.csv(infile, colClasses = "character", quote = "\"",
              stringsAsFactors = FALSE)

if (method == "A") {
  suppressMessages(library(reclin2))
  suppressMessages(library(data.table))
  d$block <- d$key
  # execute the certified blocking on the same key so the partition we
  # export is exactly what reclin2 would join on
  x <- data.table(id = d$id, bkey = d$key)
  p <- pair_blocking(x, x, on = "bkey")
  cat("reclin2 pairs:", nrow(p), "blocks:", length(unique(d$block)), "\n")
} else if (method == "C") {
  suppressMessages(library(klsh))
  set.seed(0)
  r.set <- data.frame(text = d$text, stringsAsFactors = FALSE)
  b <- klsh(r.set, p = 20, num.blocks = as.integer(param), k = 2,
            quiet = TRUE)
  ids <- klsh::block.ids.from.blocking(b)
  d$block <- paste0("c", ids)
  cat("klsh blocks:", length(unique(d$block)), "\n")
} else {
  stop("unknown method")
}

stopifnot(!any(is.na(d$block)), length(d$block) == nrow(d))
write.csv(data.frame(id = d$id, block = d$block), outfile,
          row.names = FALSE, quote = TRUE)
