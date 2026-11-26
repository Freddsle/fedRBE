#!/usr/bin/env Rscript
# Export matched inputs from the existing confounding and missingness simulations.

script <- if (sys.nframe() == 0L) {
  sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE))
} else sys.frame(1)$ofile
script <- normalizePath(script)
repo <- normalizePath(file.path(dirname(script), "..", ".."))
simulation <- new.env()
source(file.path(repo, "evaluation_data", "simulated", "robustness_simulation.R"),
       local = simulation)
missingness <- new.env()
source(file.path(repo, "evaluation_data", "simulated", "missingness_overlap.R"),
       local = missingness)

parse_options <- function(args) {
  flags <- args[seq_along(args) %% 2L == 1L]
  values <- args[seq_along(args) %% 2L == 0L]
  required <- c("--scenario", "--run", "--out", "--reference")
  allowed <- c(required, "--analysis", "--target", "--panel", "--inputs", "--dcombat")
  if (length(args) %% 2L || anyDuplicated(flags) || !all(required %in% flags) ||
      any(!flags %in% allowed) ||
      any(!nzchar(values)) || any(startsWith(values, "--"))) {
    stop("Use --scenario NAME --run N --out NEW_DIR --reference RETAINED_TSV; ",
         "missingness also needs --analysis missingness_overlap --target F --panel P --inputs RDS ",
         "and takes --dcombat UPSTREAM_DIR")
  }
  opts <- as.list(setNames(values, sub("^--", "", flags)))
  if (is.null(opts$analysis)) opts$analysis <- "confounding"
  if (!opts$analysis %in% c("confounding", "missingness_overlap")) stop("Invalid --analysis")
  if (opts$analysis == "confounding" &&
      !opts$scenario %in% names(simulation$confounded_b_fraction)) {
    stop("--scenario must be one of: ",
         paste(names(simulation$confounded_b_fraction), collapse = ", "))
  }
  extra <- c("--target", "--panel", "--inputs")
  if (opts$analysis == "confounding" && any(c(extra, "--dcombat") %in% flags)) {
    stop("Missingness options require --analysis missingness_overlap")
  }
  if (opts$analysis == "missingness_overlap") {
    if (!all(extra %in% flags) || !opts$panel %in% c("full", "overlap") ||
        !opts$target %in% c("0", "0.1", "0.3", "0.5")) stop("Invalid missingness options")
    opts$target <- as.numeric(opts$target)
    expected <- sprintf("missingness_%s_%02d", opts$panel, as.integer(100 * opts$target))
    if (opts$scenario != expected) stop("--scenario must match missingness panel and target: ", expected)
    if (!file.exists(opts$inputs)) stop("Missing saved root: ", opts$inputs)
  }
  if (!grepl("^([1-9]|10)$", opts$run)) stop("--run must be an integer in 1:10")
  opts$run <- as.integer(opts$run)
  if (file.exists(opts$out)) stop("--out must not exist; use a fresh directory")
  if (!file.exists(opts$reference)) stop("Missing retained metrics: ", opts$reference)
  opts
}

compare_reference <- function(current, path, scenario, run, tolerance) {
  reference <- read.delim(path, check.names = FALSE, stringsAsFactors = FALSE)
  keys <- c("scenario", "run", "arm", "stage", "metric", "feature")
  if (!setequal(names(reference), c(keys, "value"))) {
    stop("Reference must contain the retained run-level metric schema")
  }
  reference <- reference[!is.na(reference$scenario) & !is.na(reference$run) &
                           reference$scenario == scenario & reference$run == run, , drop = FALSE]
  if (!nrow(reference)) {
    stop("Reference contains no metrics for ", scenario, " run ", run)
  }
  key <- function(x) do.call(paste, c(lapply(x[keys], function(v) {
    ifelse(is.na(v), "<NA>", as.character(v))
  }), sep = "\r"))
  actual_keys <- key(current)
  expected_keys <- key(reference)
  if (anyDuplicated(actual_keys) || anyDuplicated(expected_keys) ||
      !setequal(actual_keys, expected_keys)) {
    stop("Generated and retained metric keys are duplicated or do not match")
  }
  reference <- reference[match(actual_keys, expected_keys), , drop = FALSE]
  actual <- current$value
  expected <- reference$value
  if (any(is.na(actual) != is.na(expected)) ||
      any(is.finite(actual) != is.finite(expected))) {
    stop("Generated and retained metrics have different finite/missing masks")
  }
  infinite <- is.infinite(actual)
  if (!identical(actual[infinite], expected[infinite])) {
    stop("Generated and retained infinite metrics differ")
  }
  finite <- is.finite(actual)
  differences <- abs(actual - expected)
  failed <- finite & differences > tolerance
  if (any(failed)) {
    i <- which.max(replace(differences, !finite, -Inf))
    stop("Retained simulation mismatch: ", current$arm[i], "/", current$stage[i],
         "/", current$metric[i], "; absolute difference ", differences[i])
  }
  non_equivalence <- finite & current$stage != "fedrbe_vs_central"
  list(matched_metrics = nrow(current), finite_metrics = sum(finite),
       missing_metrics = sum(is.na(actual)), tolerance = tolerance,
       max_absolute_difference = max(c(0, differences[finite])),
       max_non_equivalence_difference = max(c(0, differences[non_equivalence])),
       reference_md5 = unname(tools::md5sum(path)))
}

write_matrix <- function(x, path) {
  connection <- gzfile(path, "w")
  on.exit(close(connection))
  write.table(data.frame(feature = rownames(x), x, check.names = FALSE), connection,
              sep = "\t", row.names = FALSE, quote = FALSE)
}

export_confounding <- function(opts) {
started <- proc.time()[["elapsed"]]
d <- simulation$simulate_base(opts$scenario, opts$run)

# No-label correction receives no condition information. All labs' unlabeled
# expression and batch identities enter this transductive correction.
redacted_meta <- d$meta
redacted_meta$condition <- rep("redacted", nrow(redacted_meta))
corrections <- list(
  without_condition = simulation$correct_both(d$observed, redacted_meta,
                                              "without_condition"),
  with_condition = simulation$correct_both(d$observed, d$meta, "with_condition")
)

# Reuse the original scoring code with the already validated corrections.
simulation$correct_both <- function(x, meta, arm) {
  if (!identical(x, d$observed) || !identical(meta, d$meta)) {
    stop("Cached correction requested for different inputs")
  }
  corrections[[arm]]
}
metrics <- simulation$evaluate_dataset(opts$scenario, opts$run, d)$metrics
comparison <- compare_reference(metrics, opts$reference, opts$scenario, opts$run,
                                simulation$params$tolerance)

features <- rownames(d$truth)
samples <- d$meta$file
# fedRBE output for the federated learners, central limma for the central ones.
matrices <- list(truth = d$truth, uncorrected = d$observed,
                 no_label = corrections$without_condition$corrected,
                 label_aware = corrections$with_condition$corrected,
                 no_label_central = corrections$without_condition$central,
                 label_aware_central = corrections$with_condition$central)
for (state in names(matrices)) {
  x <- matrices[[state]]
  if (anyDuplicated(rownames(x)) || anyDuplicated(colnames(x)) ||
      !identical(rownames(x), features) || !identical(colnames(x), samples) ||
      any(!is.finite(x))) {
    stop("Nonfinite values or feature/sample alignment failure in ", state)
  }
}
if (length(features) != 6000L || length(samples) != 600L ||
    anyDuplicated(samples) || anyDuplicated(features) ||
    !identical(corrections$without_condition$input, corrections$with_condition$input)) {
  stop("The complete, matched feature/sample universe was not preserved")
}

if (!dir.create(opts$out, recursive = TRUE)) stop("Could not create output directory")
for (state in names(matrices)) {
  write_matrix(matrices[[state]], file.path(opts$out, paste0(state, ".tsv.gz")))
}
write.table(d$meta, file.path(opts$out, "metadata.tsv"), sep = "\t", row.names = FALSE,
            quote = FALSE)
write.table(data.frame(feature = features, role = d$role),
            file.path(opts$out, "features.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
write.table(metrics, file.path(opts$out, "generation_metrics.tsv"), sep = "\t",
            row.names = FALSE, quote = FALSE)
files <- list.files(opts$out, full.names = TRUE)
counts <- as.data.frame(table(lab = d$meta$lab, condition = d$meta$condition))
validation <- list(
  scenario = opts$scenario, run = opts$run,
  data_seed = simulation$run_seed(opts$scenario, opts$run),
  eta2_condition = simulation$eta_squared(d$meta$condition == "B", d$meta$lab),
  samples = length(samples), features = length(features), lab_condition_counts = counts,
  aligned_complete_inputs = TRUE,
  no_label_held_out_access = "Unlabeled expression and lab identity from every lab",
  label_aware_scope = "Oracle: all samples' condition labels enter correction",
  retained_comparison = comparison,
  correction_equivalence = lapply(corrections, function(x) as.list(x$differences)),
  generation_fingerprint = simulation$fingerprint(),
  exporter_md5 = unname(tools::md5sum(script)),
  output_md5 = as.list(setNames(unname(tools::md5sum(files)), basename(files))),
  elapsed_seconds = proc.time()[["elapsed"]] - started,
  session_info = simulation$session_lines()
)
jsonlite::write_json(validation, file.path(opts$out, "validation.json"),
                     auto_unbox = TRUE, pretty = TRUE, digits = 17, na = "null")
cat(sprintf("%s run %d: %d matched metrics; %.1f seconds\n", opts$scenario, opts$run,
            comparison$matched_metrics, validation$elapsed_seconds))
invisible(validation)
}

compare_missingness_reference <- function(current, path) {
  reference <- read.delim(path, check.names = FALSE, stringsAsFactors = FALSE)
  if (!setequal(names(current), names(reference))) stop("Invalid missingness reference schema")
  keys <- c("root", "panel", "target", "lab")
  key <- function(x) do.call(paste, c(x[keys], sep = "\r"))
  actual_keys <- key(current)
  expected_keys <- key(reference)
  if (anyDuplicated(expected_keys) || anyDuplicated(actual_keys) ||
      !all(actual_keys %in% expected_keys)) stop("Missing or duplicate retained mask records")
  selected <- reference[match(actual_keys, expected_keys), names(current), drop = FALSE]
  values <- setdiff(names(current), keys)
  differences <- abs(as.matrix(current[values]) - as.matrix(selected[values]))
  if (any(!is.finite(differences)) || any(differences > 1e-12)) {
    stop("Reconstructed masks differ from retained missingness records")
  }
  list(matched_metrics = nrow(current) * length(values), tolerance = 1e-12,
       max_absolute_difference = max(differences),
       reference_md5 = unname(tools::md5sum(path)))
}

# d-ComBat reference code (Chen et al. 2022): distributedCombat.R, neuroCombat.R and
# neuroCombat_helpers.R of github.com/andy1764/Distributed-ComBat at commit 45b5a25,
# read from a local directory because the repository states no licence.
dcombat_md5 <- c(distributedCombat.R = "bc50f42c70bf27646ed0a704e2ae533f",
                 neuroCombat.R = "1fef27eae2764498a9677a624b2c9e6a",
                 neuroCombat_helpers.R = "6d3ac0d4ee78f51ab7a8dde2f0e2f54f")

load_dcombat <- function(directory) {
  suppressPackageStartupMessages(library(matrixStats))  # the reference code calls rowVars
  upstream <- new.env(parent = globalenv())
  for (name in names(dcombat_md5)) {
    path <- file.path(directory, name)
    if (!file.exists(path) || unname(tools::md5sum(path)) != dcombat_md5[[name]]) {
      stop("--dcombat needs ", name, " of Distributed-ComBat commit 45b5a25 with MD5 ",
           dcombat_md5[[name]])
    }
    sys.source(path, envir = upstream)
  }
  upstream
}

# The reference code's documented protocol with each lab as a site: sites send summaries
# for the coordinator's least-squares fit, then for the pooled variance, and harmonize
# their own samples. No covariates enter, so condition labels never do; the defaults give
# the parametric empirical Bayes location and scale adjustment. Returns the harmonized
# output of every site.
dcombat_sites <- function(x, lab, upstream) {
  sites <- sort(unique(lab))
  batch <- factor(lab, levels = sites)
  # Results are returned in memory; the reference code warns that no file was named.
  quiet <- function(expr) withCallingHandlers(expr, warning = function(w) {
    if (startsWith(conditionMessage(w), "Must specify filename")) invokeRestart("muffleWarning")
  })
  site <- function(central) lapply(sites, function(s) quiet(upstream$distributedCombat_site(
    x[, lab == s, drop = FALSE], batch[lab == s], central.out = central, verbose = FALSE)))
  coordinator <- function(outs) quiet(upstream$distributedCombat_central(outs))
  setNames(site(coordinator(site(coordinator(site(NULL))))), sites)
}

# Centralized ComBat (the reference code's neuroCombat) with the same defaults.
combat_correct <- function(x, lab, upstream) {
  upstream$neuroCombat(x, factor(lab, levels = sort(unique(lab))), verbose = FALSE)$dat.combat
}

dcombat_correct <- function(x, lab, upstream) {
  corrected <- do.call(cbind, lapply(unname(dcombat_sites(x, lab, upstream)), `[[`,
                                     "dat.combat"))
  if (!setequal(colnames(corrected), colnames(x)) || anyDuplicated(colnames(corrected))) {
    stop("d-ComBat output samples differ from its input")
  }
  corrected <- corrected[, colnames(x), drop = FALSE]
  if (!identical(dimnames(corrected), dimnames(x))) stop("d-ComBat changed the features")
  if (any(!is.finite(corrected))) {
    stop("d-ComBat gave non-finite values for ", sum(rowSums(!is.finite(corrected)) > 0),
         " features")
  }
  corrected
}

# Each method of a route is scored with the native recovery metrics on the route's
# features and originally observed cells: against the truth on those cells and, for the
# contrasts, against the complete truth.
correction_diagnostics <- function(routes, observed, d) {
  do.call(rbind, lapply(names(routes), function(route) {
    do.call(rbind, lapply(names(routes[[route]]), function(method) {
      y <- routes[[route]][[method]]
      y[is.na(observed[rownames(y), colnames(y), drop = FALSE])] <- NA_real_
      truth <- d$truth[rownames(y), colnames(y), drop = FALSE]
      role <- d$role[match(rownames(y), rownames(d$truth))]
      values <- list(masked_truth = missingness$score_metrics(y, truth, d$meta, role),
                     complete_truth = simulation$contrast_recovery(y, truth, d$meta, role))
      do.call(rbind, lapply(names(values), function(stage) data.frame(
        route, method, features = nrow(y), observed_cells = sum(!is.na(y)), stage,
        metric = names(values[[stage]]), value = unname(values[[stage]]))))
    }))
  }))
}

export_missingness <- function(opts) {
  started <- proc.time()[["elapsed"]]
  upstream <- if (!is.null(opts$dcombat)) load_dcombat(opts$dcombat)
  d <- readRDS(opts$inputs)
  if (!identical(d, missingness$make_root(opts$run))) {
    stop("Saved root does not exactly match the fixed-seed simulation inputs")
  }
  input_sha256 <- strsplit(system2("sha256sum", shQuote(opts$inputs), stdout = TRUE), " +")[[1]][1]
  if (!grepl("^[0-9a-f]{64}$", input_sha256)) stop("Could not hash the saved root")
  x <- missingness$panel_data(missingness$mask_root(d, opts$target), d, opts$panel)
  metrics <- missingness$missingness_rates(x, d, opts$panel, opts$target)
  comparison <- compare_missingness_reference(metrics, opts$reference)
  exported <- missingness$downstream_inputs(d, opts$target, opts$panel)
  matrices <- exported$matrices
  features <- rownames(matrices$truth)
  eligibility <- exported$eligibility
  if (!dir.create(opts$out, recursive = TRUE)) stop("Could not create output directory")
  for (state in names(matrices)) {
    write_matrix(matrices[[state]], file.path(opts$out, paste0(state, ".tsv.gz")))
  }
  for (state in names(exported$impute)) {
    write_matrix(exported$impute[[state]],
                 file.path(opts$out, paste0("impute_", state, ".tsv.gz")))
  }
  if (!is.null(upstream)) {
    # d-ComBat corrects the raw drop and impute matrices itself, jointly over all labs;
    # centralized ComBat on the pooled matrix is the central learners' counterpart.
    complete <- rowSums(is.na(matrices$uncorrected)) == 0L
    routes <- list(drop = list(uncorrected = matrices$uncorrected[complete, , drop = FALSE],
                               fedrbe = matrices$no_label[complete, , drop = FALSE]),
                   impute = list(uncorrected = exported$impute$uncorrected,
                                 fedrbe = exported$impute$no_label))
    for (route in names(routes)) {
      raw <- routes[[route]]$uncorrected
      fit <- dcombat_correct(raw, d$meta$lab, upstream)
      central <- combat_correct(raw, d$meta$lab, upstream)
      if (max(abs(fit - central)) > simulation$params$tolerance) {
        stop("d-ComBat and ComBat differ on the ", route, " route")
      }
      routes[[route]]$dcombat <- fit
      write_matrix(fit, file.path(opts$out, paste0(route, "_dcombat.tsv.gz")))
      write_matrix(central, file.path(opts$out, paste0(route, "_dcombat_central.tsv.gz")))
    }
    write.table(cbind(root = d$root, panel = opts$panel, target = opts$target,
                      correction_diagnostics(routes, x, d)),
                file.path(opts$out, "dcombat_diagnostics.tsv"), sep = "\t",
                row.names = FALSE, quote = FALSE, na = "NA")
  }
  for (name in c("metadata", "features", "generation_metrics", "learner_eligibility",
                 "sample_minimum")) {
    data <- switch(name, metadata = d$meta,
      sample_minimum = data.frame(file = d$meta$file, minimum = exported$sample_minimum),
      features = data.frame(feature = features, role = d$role[match(features, rownames(d$truth))]),
      generation_metrics = metrics, learner_eligibility = eligibility)
    write.table(data, file.path(opts$out, paste0(name, ".tsv")), sep = "\t",
                row.names = FALSE, quote = FALSE, na = "NA")
  }
  files <- list.files(opts$out, full.names = TRUE)
  validation <- list(
    analysis = opts$analysis, scenario = opts$scenario, run = opts$run, root = d$root,
    variant = d$variant, data_seed = d$seed, panel = opts$panel, target = opts$target,
    eta2_condition = simulation$eta_squared(d$meta$condition == "B", d$meta$lab),
    samples = nrow(d$meta), features = length(features),
    complete_features = sum(eligibility$complete_case),
    impute_features = sum(eligibility$impute_retained), imputed_cells = exported$imputed_cells,
    impute_policy = "observed_ge6_every_client_sample_minimum",
    lab_condition_counts = as.data.frame(table(lab = d$meta$lab, condition = d$meta$condition)),
    saved_input_sha256 = input_sha256,
    aligned_matched_inputs = TRUE, feature_mask_policy = "observed_ge6_clients_ge3_native",
    matching_min_observations = 6L, matching_min_clients = 3L,
    native_retention = list(without_condition = with(exported$corrections$without_condition,
      list(features = nrow(input), observed_cells = sum(!is.na(input)),
           matching_excluded_features = nrow(input) - length(features),
           matching_excluded_cells = sum(!is.na(input)) - sum(!is.na(matrices$uncorrected))))),
    no_label_held_out_access = "Unlabeled expression and lab identity from every lab",
    retained_comparison = comparison,
    correction_equivalence = lapply(exported$corrections, function(fit) as.list(fit$differences)),
    generation_fingerprint = c(simulation$fingerprint(),
      paste("md5 evaluation_data/simulated/missingness_overlap.R", missingness$campaign_source_md5)),
    exporter_md5 = unname(tools::md5sum(script)),
    output_md5 = as.list(setNames(unname(tools::md5sum(files)), basename(files))),
    elapsed_seconds = proc.time()[["elapsed"]] - started,
    session_info = simulation$session_lines()
  )
  if (!is.null(upstream)) {
    validation$dcombat <- list(upstream_md5 = as.list(dcombat_md5), labels_used = FALSE)
  }
  jsonlite::write_json(validation, file.path(opts$out, "validation.json"),
                       auto_unbox = TRUE, pretty = TRUE, digits = 17, na = "null")
  cat(sprintf("%s root %d: %d matched, %d complete, %d imputed features; %.1f seconds\n",
              opts$scenario, opts$run, length(features), validation$complete_features,
              validation$impute_features, validation$elapsed_seconds))
  invisible(validation)
}

if (sys.nframe() == 0L) {
  opts <- parse_options(commandArgs(trailingOnly = TRUE))
  if (opts$analysis == "confounding") export_confounding(opts) else export_missingness(opts)
}
