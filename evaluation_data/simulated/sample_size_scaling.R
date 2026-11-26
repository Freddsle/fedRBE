#!/usr/bin/env Rscript
# Design, requirements and usage: README.md, "Sample-size scaling of the fedRBE-limma
# agreement". Heavy at the larger sample counts.

suppressPackageStartupMessages(library(dplyr))

script <- if (sys.nframe() == 0L) {
  sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE))
} else sys.frame(1)$ofile
script <- normalizePath(script)
repo <- normalizePath(file.path(dirname(script), "..", ".."))
# The study's generator and its local fedRBE-limma comparison, used unchanged.
study <- new.env()
source(file.path(dirname(script), "robustness_simulation.R"), local = study)

sample_counts <- c(600L, 3000L, 12000L, 50004L)
base_seed <- study$params$seed
release_dir <- file.path(dirname(script), "sample_size_scaling")

parse_options <- function(args) {
  flags <- args[seq_along(args) %% 2L == 1L]
  values <- args[seq_along(args) %% 2L == 0L]
  if (length(args) %% 2L || anyDuplicated(flags) ||
      any(!flags %in% c("--samples", "--runs", "--out", "--workers"))) {
    stop("Use each option once with a value: --samples 600,3000 --runs 1,2 --out DIR ",
         "--workers 2")
  }
  integers <- function(flag, default) {
    if (!flag %in% flags) return(default)
    text <- values[flags == flag]
    if (!grepl("^[0-9]+(,[0-9]+)*$", text)) {
      stop(flag, " requires comma-separated integers")
    }
    result <- suppressWarnings(as.integer(strsplit(text, ",", fixed = TRUE)[[1]]))
    if (anyDuplicated(result)) stop(flag, " requires distinct values")
    result
  }
  opts <- list(samples = integers("--samples", sample_counts),
               runs = integers("--runs", seq_len(study$params$runs)),
               workers = integers("--workers", 1L),
               out = if ("--out" %in% flags) values[flags == "--out"] else {
                 file.path(dirname(script), "sample_size_scaling_work")
               })
  if (anyNA(c(opts$samples, opts$runs, opts$workers)) || length(opts$workers) != 1L ||
      !all(opts$samples %in% sample_counts) || opts$workers < 1L ||
      !all(opts$runs %in% seq_len(study$params$runs))) {
    stop("--samples must be among ", paste(sample_counts, collapse = ","),
         ", --runs within 1:", study$params$runs, " and --workers a positive integer")
  }
  if (!nzchar(opts$out) || startsWith(opts$out, "--")) stop("--out requires a directory")
  if (identical(normalizePath(opts$out, mustWork = FALSE),
                normalizePath(release_dir, mustWork = FALSE))) {
    stop("Use a separate --out directory to preserve the released results.")
  }
  opts
}

# The balanced three-client dataset of the study at another sample count. The first
# sample count keeps the study's seeds and reproduces its balanced runs.
simulate <- function(samples, run) {
  study$params$samples <- samples
  study$params$seed <- base_seed + 100L * (match(samples, sample_counts) - 1L)
  d <- study$simulate_base("balanced", run)
  d$truth <- NULL
  d
}

# The study's comparison of one dataset in both correction arms: central limma against
# the app protocol run locally with plaintext aggregation.
run_dataset <- function(job, out) {
  d <- simulate(job$samples, job$run)
  rows <- lapply(study$arms, function(arm) {
    compared <- study$correct_both(d$observed, d$meta, arm)
    data.frame(samples = job$samples, run = job$run, arm,
               features = nrow(compared$corrected), cells = sum(!is.na(compared$corrected)),
               min_diff = compared$differences[["min_diff"]],
               mean_diff = compared$differences[["mean_diff"]],
               max_diff = compared$differences[["max_diff"]])
  })
  prefix <- file.path(out, "runs", sprintf("n%05d_run%03d", job$samples, job$run))
  # The differences file marks a completed dataset, so it is published last.
  write.table(do.call(rbind, rows), paste0(prefix, ".partial"), sep = "\t", row.names = FALSE,
              quote = FALSE)
  if (!file.rename(paste0(prefix, ".partial"), paste0(prefix, ".tsv"))) {
    stop("Could not publish completed dataset: ", prefix)
  }
  cat(sprintf("%s done at %s\n", basename(prefix), format(Sys.time(), "%H:%M")))
  TRUE
}

fingerprint <- function() {
  own <- file.path("evaluation_data", "simulated", "sample_size_scaling.R")
  c(paste("sample counts:", paste(sample_counts, collapse = ", ")),
    "scenario: balanced; seeds: study seed + 1000 + run + 100 * (sample-count index - 1)",
    paste("md5", own, unname(tools::md5sum(file.path(repo, own)))),
    study$fingerprint())
}

# Mean of the dataset means and extremes over the datasets, as in the study's agreement table.
summarise <- function(out) {
  files <- list.files(file.path(out, "runs"), pattern = "_run[0-9]{3}\\.tsv$", full.names = TRUE)
  results <- do.call(rbind, lapply(files, read.delim))
  summary <- results %>%
    dplyr::group_by(samples, arm) %>%
    dplyr::summarise(datasets = dplyr::n(), min_diff = min(min_diff),
                     mean_diff = mean(mean_diff), max_diff = max(max_diff), .groups = "drop")
  write.table(results, file.path(out, "differences.tsv"), sep = "\t", row.names = FALSE,
              quote = FALSE)
  write.table(summary, file.path(out, "summary.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
  print(as.data.frame(summary))
}

main <- function(args = commandArgs(trailingOnly = TRUE)) {
  opts <- parse_options(args)
  dir.create(file.path(opts$out, "runs"), recursive = TRUE, showWarnings = FALSE)
  provenance <- file.path(opts$out, "provenance.txt")
  current <- fingerprint()
  if (file.exists(provenance)) {
    if (!identical(readLines(provenance, n = length(current)), current)) {
      stop("Parameters or code changed since ", opts$out,
           " was created; use a new --out directory.")
    }
  } else {
    writeLines(c(current, "", study$session_lines()), provenance)
  }
  jobs <- list()
  for (samples in opts$samples) for (run in opts$runs) {
    if (!file.exists(file.path(opts$out, "runs", sprintf("n%05d_run%03d.tsv", samples, run)))) {
      jobs[[length(jobs) + 1L]] <- list(samples = samples, run = run)
    }
  }
  # Datasets are independent; with several workers each runs in a forked process and a
  # failed dataset does not stop the others.
  done <- parallel::mclapply(jobs, run_dataset, out = opts$out, mc.cores = opts$workers,
                             mc.preschedule = FALSE)
  failed <- !vapply(done, isTRUE, logical(1))
  if (any(failed)) {
    stop(sum(failed), " dataset(s) failed, the first with: ", done[failed][[1]])
  }
  summarise(opts$out)
}

if (sys.nframe() == 0L) main()
