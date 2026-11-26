#!/usr/bin/env Rscript
# Extract the simulated lab shifts and noise scales used in the five-client LOCO comparison.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop("Supply the informative runs directory and output TSV")
rows <- lapply(1:10, function(root) {
  d <- readRDS(file.path(args[1], sprintf("root%03d_inputs.rds", root)))
  stopifnot(d$root == root, identical(d$variant, "informative"))
  b <- attr(d$observed, "batch_params")
  stopifnot(setequal(names(b$noise_sd), unique(d$meta$lab)),
            identical(names(b$noise_sd), names(b$additive)))
  data.frame(root, lab = names(b$noise_sd), additive = unname(b$additive),
             noise_sd = unname(b$noise_sd))
})
x <- do.call(rbind, rows)
write.table(x[order(-x$noise_sd), ], args[2], sep = "\t", quote = FALSE, row.names = FALSE)
