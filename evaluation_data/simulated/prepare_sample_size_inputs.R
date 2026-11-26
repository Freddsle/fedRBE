#!/usr/bin/env Rscript
# Export balanced run 1 for generate_fedrbe_corrected_datasets.py.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("Usage: Rscript prepare_sample_size_inputs.R 600,3000")
script <- normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)))
study <- new.env()
source(file.path(dirname(script), "sample_size_scaling.R"), local = study)
opts <- study$parse_options(c("--samples", args[[1]], "--runs", "1"))

prepare <- function(samples) {
  scenario <- file.path(dirname(script), "sample_size_scaling", sprintf("n%05d_run001", samples))
  before <- file.path(scenario, "before")
  if (file.exists(before)) stop("Input directory already exists: ", before)
  staging <- tempfile(".before-", tmpdir = scenario)
  dir.create(staging, recursive = TRUE)
  on.exit(unlink(staging, recursive = TRUE))

  d <- study$simulate(samples, 1L)
  labs <- sort(unique(d$meta$lab))
  descriptor <- jsonlite::read_json(file.path(dirname(script), "balanced", "before", "datainfo.json"))
  descriptor$data_name <- sprintf("Simulated Balanced (n=%d)", samples)
  descriptor$datafile$filename <- "intensities.tsv.gz"
  for (i in seq_along(labs)) {
    lab <- labs[[i]]
    folder <- file.path(staging, lab)
    dir.create(folder)
    design <- d$meta[d$meta$lab == lab, c("file", "condition", "lab")]
    design$A <- as.integer(design$condition == "B")
    output <- gzfile(file.path(folder, descriptor$datafile$filename), "wt")
    tryCatch(write.table(data.frame(rowname = rownames(d$observed), d$observed[, design$file],
                                    check.names = FALSE),
                         output, sep = "\t", row.names = FALSE), finally = close(output))
    write.table(design, file.path(folder, "design.tsv"), sep = "\t", row.names = FALSE)
    descriptor$cohorts[[i]]$name <- descriptor$cohorts[[i]]$folder <- lab
  }
  jsonlite::write_json(descriptor, file.path(staging, "datainfo.json"),
                       auto_unbox = TRUE, null = "null", pretty = TRUE)
  if (!file.rename(staging, before)) stop("Could not publish input directory: ", before)
  cat("Prepared", before, "\n")
}

for (samples in opts$samples) prepare(samples)
