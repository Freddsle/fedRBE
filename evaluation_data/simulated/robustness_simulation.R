#!/usr/bin/env Rscript
# Model, parameters and usage: README.md, "Robustness simulations".

suppressPackageStartupMessages(library(limma))
suppressPackageStartupMessages(library(variancePartition))

script <- if (sys.nframe() == 0L) {
  sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE))
} else sys.frame(1)$ofile
script <- normalizePath(script)
repo <- normalizePath(file.path(dirname(script), "..", ".."))
for (helper in c("simulation_func.R", "filtering.R", "evaluation_funcs.R")) {
  suppressPackageStartupMessages(source(file.path(repo, "evaluation_utils", helper)))
}
fedrbe_script <- file.path(repo, "evaluation_utils", "fedRBE_simulation_scrip_simdata.py")

params <- list(
  seed = 20261016L,
  runs = 10L,
  samples = 600L, mu_1 = 1.45, mu_4 = 1.25, frac_1 = 0.45, frac_7 = 0.15,
  # Raw-scale offsets per lab, as quantiles of the run's batch-free raw intensities.
  offset_quantiles = c(batch2 = 0.10, batch3 = 0.25),
  missing_alpha = c(batch1 = 0.1, batch2 = 0.2, batch3 = 0.3), missing_beta = 0.8,
  panel_fraction = c(0.1, 0.2, 0.7), # features observed in one, two, three labs
  verify_runs = c(1L, 5L, 10L),
  tolerance = 1e-8
)
scenarios <- c("balanced", "mild_imbalanced", "strong_imbalanced",
               "intensity_dependent", "missingness")
# Scenarios generated from each base composition; derived ones reuse the balanced run.
derived <- list(balanced = c("balanced", "intensity_dependent", "missingness"),
                mild_imbalanced = "mild_imbalanced", strong_imbalanced = "strong_imbalanced")
arms <- c("with_condition", "without_condition")

parse_options <- function(args) {
  flags <- args[seq_along(args) %% 2L == 1L]
  values <- args[seq_along(args) %% 2L == 0L]
  if (length(args) %% 2L || any(!flags %in% c("--runs", "--scenarios", "--out")) ||
      anyDuplicated(flags) || any(!nzchar(args)) ||
      any(startsWith(values, "--"))) {
    stop("Use each option once with a value: --runs 1,5 --scenarios balanced --out DIR")
  }
  value <- function(flag, default) {
    i <- match(flag, flags)
    if (is.na(i)) default else values[i]
  }
  selected <- strsplit(value("--scenarios", paste(scenarios, collapse = ",")),
                       ",", fixed = TRUE)[[1]]
  if (!length(selected) || length(setdiff(selected, scenarios)) ||
      grepl("(^,|,$|,,)", value("--scenarios", ""))) {
    stop("Use --scenarios from ", paste(scenarios, collapse = ","))
  }
  runs <- value("--runs", paste(seq_len(params$runs), collapse = ","))
  if (!grepl("^[1-9][0-9]*(,[1-9][0-9]*)*$", runs)) {
    stop("--runs requires positive integers, e.g. --runs 1,5")
  }
  opts <- list(runs = unique(suppressWarnings(as.integer(strsplit(runs, ",", fixed = TRUE)[[1]]))),
               scenarios = unique(selected),
               out = value("--out", file.path(dirname(script), "robustness")))
  if (anyNA(opts$runs) || any(opts$runs > params$runs)) {
    stop("--runs must be within 1:", params$runs)
  }
  opts
}

run_seed <- function(mode, run) params$seed + 1000L * match(mode, scenarios) + run

simulate_base <- function(mode, run) {
  set.seed(run_seed(mode, run))
  meta <- data.frame(file = paste("s", seq_len(params$samples), sep = "."))
  rownames(meta) <- meta$file
  meta <- generate_metadata(meta, mode)
  frac_A <- mean(meta$condition == "A")
  truth <- generate_data(frac_A, 1 - frac_A, params$frac_1, params$frac_7,
                         mu_1 = params$mu_1, mu_4 = params$mu_4, batch_info = meta,
                         mode_version = mode, m = params$samples)
  role <- attr(truth, "feature_role")
  meta$latent <- meta$file %in% attr(truth, "latent_samples")
  meta$lab <- as.character(meta$batch)
  truth <- matrix(truth, nrow(truth),
                  dimnames = list(paste0("prt", seq_len(nrow(truth))), meta$file))
  stopifnot(identical(as.vector(table(factor(role, c("A_DE", "B_DE", "latent", "null")))),
                      c(1125L, 1125L, 150L, 3600L)))
  list(meta = meta, truth = truth, observed = add_batch_effect(truth, meta), role = role)
}

# Batch-specific additive offset on the raw scale (calibration model of Huber et al. 2002).
add_intensity_offset <- function(d) {
  raw_truth <- 2^d$truth
  for (lab in names(params$offset_quantiles)) {
    offset <- quantile(raw_truth, params$offset_quantiles[[lab]], names = FALSE)
    cols <- d$meta$lab == lab
    d$observed[, cols] <- log2(2^d$observed[, cols] + offset)
  }
  d
}

# Abundance-dependent and random missingness, then lab-specific feature absence.
add_missingness <- function(d, run) {
  set.seed(run_seed("balanced", run) + 500L)
  labs <- names(params$missing_alpha)
  for (lab in labs) {
    cols <- d$meta$lab == lab
    d$observed[, cols] <- simulateMissingValues(d$observed[, cols], params$missing_alpha[[lab]],
                                                params$missing_beta)
  }
  # Repeating 1,2,2,3,...,3 over role-sorted, randomly ordered features gives the panel
  # fractions overall and within each truth role.
  n <- nrow(d$observed)
  d$n_labs <- integer(n)
  d$n_labs[order(d$role, runif(n))] <- rep_len(rep(1:3, round(10 * params$panel_fraction)), n)
  present <- t(apply(matrix(runif(3 * n), n), 1, rank)) <= d$n_labs
  for (i in seq_along(labs)) d$observed[!present[, i], d$meta$lab == labs[i]] <- NA
  d$missing_fraction <- sapply(labs, function(lab) mean(is.na(d$observed[, d$meta$lab == lab])))
  d
}

correct_central <- function(x, meta, arm) {
  covariates <- if (arm == "with_condition") "condition" else character()
  input <- suppressMessages(filter_fedrbe_client(x, meta, "file", "lab", num_batches = 3,
                                                 covariates = covariates))
  corrected <- if (arm == "with_condition") {
    limma::removeBatchEffect(input, batch = meta$lab, design = model.matrix(~condition, meta))
  } else {
    limma::removeBatchEffect(input, batch = meta$lab)
  }
  stopifnot(identical(is.na(input), is.na(corrected)))
  list(input = input, corrected = corrected)
}

# Production fedRBE code run locally on per-lab inputs written as for the FeatureCloud app.
correct_fedrbe <- function(x, meta, arm) {
  dir <- tempfile("fedrbe_")
  on.exit(unlink(dir, recursive = TRUE))
  labs <- sort(unique(meta$lab))
  for (i in seq_along(labs)) {
    lab_dir <- file.path(dir, labs[i])
    dir.create(lab_dir, recursive = TRUE)
    lab_meta <- meta[meta$lab == labs[i], c("file", "condition", "lab")]
    lab_meta$A <- as.integer(lab_meta$condition == "B")
    write.table(data.frame(rowname = rownames(x), x[, lab_meta$file], check.names = FALSE),
                file.path(lab_dir, "intensities.tsv"), sep = "\t", row.names = FALSE)
    write.table(lab_meta, file.path(lab_dir, "design.tsv"), sep = "\t", row.names = FALSE)
    writeLines(c("flimmaBatchCorrection:", "  batch_col: null",
                 if (arm == "with_condition") c("  covariates:", "  - A") else "  covariates: []",
                 "  data_filename: intensities.tsv", "  design_filename: design.tsv",
                 '  design_separator: "\\t"', "  expression_file_flag: true",
                 "  index_col: rowname", "  min_samples: 0", "  normalizationMethod: null",
                 paste0("  position: ", i - 1), '  separator: "\\t"', "  smpc: true"),
               file.path(lab_dir, "config.yml"))
  }
  out <- file.path(dir, "corrected.tsv")
  log <- system2("python", shQuote(c(fedrbe_script, dir, out)), stdout = TRUE, stderr = TRUE)
  if (!is.null(attr(log, "status")) || !file.exists(out)) {
    stop("Local fedRBE failed:\n", paste(tail(log, 20), collapse = "\n"))
  }
  as.matrix(read.delim(out, row.names = 1, check.names = FALSE))
}

# The two matrices may have different NA masks when measuring selection effects.
contrast_recovery <- function(x, reference, meta, role) {
  stopifnot(identical(dimnames(x), dimnames(reference)),
            identical(colnames(x), meta$file), length(role) == nrow(x))
  contrast <- function(m, group, control) {
    rowMeans(m[, group, drop = FALSE], na.rm = TRUE) -
      rowMeans(m[, control, drop = FALSE], na.rm = TRUE)
  }
  attenuation <- function(group, control, rows) {
    estimate <- contrast(x, group, control)
    target <- contrast(reference, group, control)
    ok <- rows & is.finite(estimate) & is.finite(target)
    denominator <- sum(target[ok]^2)
    if (denominator == 0) NA_real_ else sum(estimate[ok] * target[ok]) / denominator
  }
  in_b <- meta$condition == "B"
  c(attenuation_condition = attenuation(in_b, !in_b, role %in% c("A_DE", "B_DE")),
    attenuation_latent = attenuation(in_b & meta$latent, in_b & !meta$latent, role == "latent"))
}

# Score every stage on the retained features and the same observed cells.
recovery_metrics <- function(x, truth, meta, role) {
  stopifnot(identical(dimnames(x), dimnames(truth)))
  truth[is.na(x)] <- NA
  residual <- x - truth
  residual <- residual - rowMeans(residual, na.rm = TRUE)
  c(rmse_to_truth = sqrt(mean(residual^2, na.rm = TRUE)),
    contrast_recovery(x, truth, meta, role))
}

evaluate_stage <- function(x, truth, meta, role) {
  # Sequential sums of squares would depend on term order under confounding.
  variance <- if (anyNA(x)) NULL else variance_partition_long(x, meta, ~condition + lab)
  medians <- numeric()
  if (!is.null(variance)) {
    medians <- tapply(variance$value, variance$variable, median, na.rm = TRUE)
    medians <- setNames(as.vector(medians), paste0("median_variance_", names(medians)))
  }
  list(metrics = c(recovery_metrics(x, truth, meta, role), medians), variance = variance)
}

evaluate_dataset <- function(scenario, run, d, baseline = NULL) {
  tidy <- function(arm, stage, metric, value, feature = NA_character_) {
    data.frame(scenario, run, arm, stage, metric, feature, value)
  }
  metrics <- variance <- list()
  if (!is.null(d$n_labs)) {
    metrics[[1]] <- tidy("none", "input",
                         c(paste0("missing_fraction_", names(d$missing_fraction)),
                           paste0("features_in_", 1:3, "_labs")),
                         c(d$missing_fraction, tabulate(d$n_labs, 3)))
  }
  shared <- NULL
  for (arm in arms) {
    central <- if (scenario == "balanced" && !is.null(baseline)) baseline[[arm]] else {
      correct_central(d$observed, d$meta, arm)
    }
    kept <- match(rownames(central$input), rownames(d$observed))
    stopifnot(is.null(d$n_labs) || all(d$n_labs[kept] == 3L))
    truth <- d$truth[kept, , drop = FALSE]
    role <- d$role[kept]
    # Before-correction and truth stages are shared when both arms retain the same cells.
    if (is.null(shared) || !identical(is.na(shared$input), is.na(central$input))) {
      masked_truth <- truth
      masked_truth[is.na(central$input)] <- NA
      shared <- list(input = central$input, stages = list(
        before = evaluate_stage(central$input, truth, d$meta, role),
        truth = evaluate_stage(masked_truth, truth, d$meta, role)),
        selection = contrast_recovery(masked_truth, truth, d$meta, role))
    }
    stages <- c(shared$stages,
                list(after = evaluate_stage(central$corrected, truth, d$meta, role)))
    for (stage in names(stages)) {
      m <- stages[[stage]]$metrics
      v <- stages[[stage]]$variance
      metrics[[length(metrics) + 1]] <- tidy(arm, stage, names(m), m)
      if (!is.null(v)) {
        variance[[length(variance) + 1]] <- tidy(arm, stage, paste0("variance_", v$variable),
                                                 signif(v$value, 6), v$feature)
      }
    }
    metrics[[length(metrics) + 1L]] <- tidy(arm, "selection", names(shared$selection),
                                            shared$selection)
    if (scenario %in% c("intensity_dependent", "missingness")) {
      stopifnot(!is.null(baseline))
      # Fit the complete baseline first; apply the scenario's mask only for scoring.
      matched <- baseline[[arm]]$corrected[rownames(central$input), colnames(central$input),
                                          drop = FALSE]
      matched[is.na(central$input)] <- NA
      stopifnot(identical(is.na(matched), is.na(central$input)))
      baseline_metrics <- recovery_metrics(matched, truth, d$meta, role)
      delta <- stages$after$metrics[names(baseline_metrics)] - baseline_metrics
      metrics[[length(metrics) + 1L]] <- tidy(arm, "baseline_after", names(baseline_metrics),
                                              baseline_metrics)
      metrics[[length(metrics) + 1L]] <- tidy(arm, "paired_after", names(delta), delta)
    }
    excluded <- nrow(d$observed) - length(kept)
    panel_excluded <- if (is.null(d$n_labs)) 0L else sum(d$n_labs < 3L)
    metrics[[length(metrics) + 1]] <- tidy(arm, "input",
      c("retained_features", "excluded_features", "panel_excluded_features",
        "eligibility_excluded_features"),
      c(length(kept), excluded, panel_excluded, excluded - panel_excluded))
    if (run %in% params$verify_runs) {
      federated <- correct_fedrbe(d$observed, d$meta, arm)
      stopifnot(setequal(rownames(federated), rownames(central$corrected)))
      federated <- federated[rownames(central$corrected), colnames(central$corrected)]
      stopifnot(all(is.na(federated) == is.na(central$corrected)))
      differences <- calculated_differences(central$corrected, federated)
      stopifnot(differences[["max_diff"]] <= params$tolerance)
      metrics[[length(metrics) + 1]] <- tidy(arm, "fedrbe_vs_central", names(differences),
                                             differences)
    }
  }
  list(metrics = do.call(rbind, metrics), variance = do.call(rbind, variance))
}

fingerprint <- function() {
  files <- c(file.path("evaluation_data", "simulated", "robustness_simulation.R"),
             file.path("evaluation_utils", c("simulation_func.R", "filtering.R",
                                             "evaluation_funcs.R",
                                             "fedRBE_simulation_scrip_simdata.py")),
             file.path("batchcorrection", "classes", c("client.py", "coordinator_utils.py")),
             "pixi.lock")
  files <- files[file.exists(file.path(repo, files))]
  c(paste("params:", paste(deparse(params), collapse = "")),
    "seeds: seed + 1000 * scenario index + run; missingness draws: balanced seed + 500",
    paste("md5", files, unname(tools::md5sum(file.path(repo, files)))))
}

write_run <- function(result, out, scenario, run) {
  prefix <- file.path(out, "runs", sprintf("%s_run%03d", scenario, run))
  if (!is.null(result$variance)) {
    connection <- gzfile(paste0(prefix, "_variance.tsv.gz"), "w")
    write.table(result$variance, connection, sep = "\t", row.names = FALSE, quote = FALSE)
    close(connection)
  }
  # The metrics file marks a completed run, so it is published last.
  write.table(result$metrics, paste0(prefix, ".partial"), sep = "\t", row.names = FALSE,
              quote = FALSE)
  if (!file.rename(paste0(prefix, ".partial"), paste0(prefix, ".tsv"))) {
    stop("Could not publish completed run: ", prefix)
  }
}

summarise_values <- function(v) {
  finite <- v[is.finite(v)]
  n <- length(finite)
  average <- if (n) mean(finite) else NA_real_
  mcse <- if (n > 1L) sd(finite) / sqrt(n) else NA_real_
  half_width <- if (n > 1L) qt(0.975, df = n - 1L) * mcse else NA_real_
  data.frame(n_runs = n, n_missing = length(v) - n, mean = average, mcse = mcse,
             ci_low = average - half_width, ci_high = average + half_width)
}

summarise <- function(out) {
  files <- list.files(file.path(out, "runs"), pattern = "_run[0-9]{3}\\.tsv$",
                      full.names = TRUE)
  if (!length(files)) stop("No completed runs in ", out)
  results <- do.call(rbind, lapply(files, read.delim))
  equivalence <- results[results$stage == "fedrbe_vs_central", ]
  results <- results[results$stage != "fedrbe_vs_central", ]
  summary <- results %>%
    dplyr::group_by(scenario, arm, stage, metric) %>%
    dplyr::group_modify(~summarise_values(.x$value)) %>%
    dplyr::ungroup()
  write.table(summary, file.path(out, "summary.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
  write.table(equivalence, file.path(out, "equivalence.tsv"), sep = "\t", row.names = FALSE,
              quote = FALSE)
  grDevices::pdf(file.path(out, "summary.pdf"), width = 12, height = 8)
  on.exit(grDevices::dev.off())
  pages <- list(recovery = c("before", "after", "truth"),
                paired_change = "paired_after", selection = "selection")
  for (page in names(pages)) {
    shown <- summary[summary$stage %in% pages[[page]] & summary$n_runs > 0, ]
    if (!nrow(shown)) next
    plot <- ggplot2::ggplot(shown, ggplot2::aes(scenario, mean, ymin = ci_low, ymax = ci_high,
                                              colour = stage, shape = arm)) +
      ggplot2::geom_point(position = ggplot2::position_dodge(width = 0.6)) +
      ggplot2::geom_errorbar(position = ggplot2::position_dodge(width = 0.6),
                            width = 0.2, na.rm = TRUE) +
      ggplot2::facet_wrap(~metric, scales = "free_y") +
      ggplot2::labs(title = page) + ggplot2::theme_bw(base_size = 8) +
      ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 45, hjust = 1))
    if (page != "recovery") {
      plot <- plot + ggplot2::geom_hline(yintercept = if (page == "selection") 1 else 0,
                                        linetype = "dashed", colour = "grey50")
    }
    print(plot)
  }
  cat("Completed runs per scenario:\n")
  print(table(unique(results[c("scenario", "run")])$scenario))
  cat("Passed equivalence checks:", nrow(unique(equivalence[c("scenario", "run", "arm")])), "\n")
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
    writeLines(c(current, "", capture.output(sessionInfo())), provenance)
  }

  for (mode in names(derived)) {
    for (run in opts$runs) {
      todo <- intersect(derived[[mode]], opts$scenarios)
      todo <- todo[!file.exists(file.path(opts$out, "runs", sprintf("%s_run%03d.tsv", todo, run)))]
      if (!length(todo)) next
      base <- simulate_base(mode, run)
      baseline <- if (any(todo %in% c("intensity_dependent", "missingness"))) {
        setNames(lapply(arms, function(arm) correct_central(base$observed, base$meta, arm)), arms)
      } else NULL
      for (scenario in todo) {
        started <- Sys.time()
        d <- switch(scenario, intensity_dependent = add_intensity_offset(base),
                    missingness = add_missingness(base, run), base)
        lab1 <- d$meta$lab == "batch1"
        stopifnot(scenario != "intensity_dependent" ||
                    identical(d$observed[, lab1], base$observed[, lab1]))
        stopifnot(scenario != "missingness" ||
                    identical(d$observed[!is.na(d$observed)], base$observed[!is.na(d$observed)]))
        write_run(evaluate_dataset(scenario, run, d, baseline), opts$out, scenario, run)
        cat(sprintf("%s run %d: %.1f s\n", scenario, run,
                    as.numeric(difftime(Sys.time(), started, units = "secs"))))
      }
    }
  }
  summarise(opts$out)
}

if (sys.nframe() == 0L) main()
