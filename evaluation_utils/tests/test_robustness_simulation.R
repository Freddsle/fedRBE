# Run from the repository root: pixi run Rscript evaluation_utils/tests/test_robustness_simulation.R
simulation <- new.env()
source("evaluation_data/simulated/robustness_simulation.R", local = simulation)
same <- function(actual, expected) {
  isTRUE(all.equal(actual, expected, tolerance = 1e-12, check.attributes = FALSE))
}
fails <- function(expr) inherits(tryCatch(force(expr), error = identity), "error")

# Five-client data keep balanced labels and a 0.6 latent fraction per batch.
small_metadata <- data.frame(file = paste0("small", 1:100))
five_meta <- simulation$generate_metadata(small_metadata, "balanced", num_batches = 5L)
stopifnot(all(table(five_meta$condition, five_meta$batch) == 10L),
          fails(simulation$generate_metadata(small_metadata, "balanced", num_batches = 3L)),
          fails(simulation$generate_metadata(small_metadata, "mild_imbalanced", 5L)))
set.seed(921)
five_truth <- simulation$generate_data(0.5, 0.5, 0.45, 0.15, batch_info = five_meta,
                                       m = 100)
stopifnot(identical(dim(five_truth), c(6000L, 100L)),
          identical(as.vector(table(factor(attr(five_truth, "feature_role"),
                                            c("A_DE", "B_DE", "latent", "null")))),
                    c(1125L, 1125L, 150L, 3600L)),
          identical(as.integer(table(factor(five_meta$batch[five_meta$file %in%
                                              attr(five_truth, "latent_samples")],
                                             levels = paste0("batch", 1:5)))), rep(6L, 5L)))

# Named batch parameters follow batch names, not metadata order.
small_truth <- five_truth[1:10, ]
batch_levels <- unique(as.character(five_meta$batch))
batch_means <- setNames(c(-2, -1, 0, 1, 2), paste0("batch", 1:5))
set.seed(922)
named_batch <- simulation$add_batch_effect(small_truth, five_meta, batch_means,
                                          sd_additive = rep(0.1, 5),
                                          shape_multiplicative = rep(3, 5),
                                          mult_scale = 1, noise_sd = 0.25, min_pc1_var = NULL)
set.seed(922)
ordered_batch <- simulation$add_batch_effect(small_truth, five_meta,
                                            unname(batch_means[batch_levels]),
                                            sd_additive = rep(0.1, 5),
                                            shape_multiplicative = rep(3, 5),
                                            mult_scale = 1, noise_sd = 0.25, min_pc1_var = NULL)
stopifnot(identical(named_batch, ordered_batch),
          identical(attr(named_batch, "batch_params")$noise_sd,
                    0.25 * attr(named_batch, "batch_params")$multiplicative))

stopifnot(all(simulation$params$runs == 10L),
          identical(simulation$parse_options(character())$runs, 1:10),
          identical(simulation$parse_options(c("--scenarios", "missingness"))$runs, 1:10),
          identical(simulation$parse_options(c("--scenarios", "confounded_p05"))$scenarios,
                    "confounded_p05"))
valid <- simulation$parse_options(c("--runs", "1,5,10", "--out", "/tmp/data,cohort"))
stopifnot(identical(valid$runs, c(1L, 5L, 10L)), identical(valid$out, "/tmp/data,cohort"))
for (args in list(c("--run", "1"), "--runs", c("--out", "--runs"),
                  c("--runs", "1", "--runs", "2"), c("--scenarios", "unknown"),
                  c("--runs", "1.5"), c("--runs", "0"), c("--runs", "11"),
                  c("--runs", "1,,2"), c("--runs", ""))) {
  stopifnot(fails(simulation$parse_options(args)))
}

# Graded confounding: condition-B fractions p, 1 - p and p over 150/300/150 samples.
set.seed(1)
confounded <- simulation$generate_metadata(data.frame(file = paste("s", 1:600, sep = ".")), "confounded_p05")
stopifnot(identical(as.vector(table(confounded$batch)), c(150L, 300L, 150L)),
          identical(as.vector(table(confounded$batch[confounded$condition == "B"])),
                    c(8L, 285L, 8L)),
          abs(simulation$eta_squared(confounded$condition == "B", confounded$batch) - 0.804) < 0.001,
          simulation$eta_squared(rep(c(TRUE, FALSE), 9), rep(1:3, each = 6)) == 0)

meta <- data.frame(file = paste0("s", 1:18), lab = rep(paste0("lab", 1:3), each = 6),
                   condition = factor(rep(c("A", "A", "A", "B", "B", "B"), 3)),
                   latent = rep(c(FALSE, FALSE, FALSE, FALSE, TRUE, TRUE), 3))
rownames(meta) <- meta$file
truth <- rbind(condition = rep(c(0, 0, 0, 0, 2, 4), 3),
               condition_a = rep(c(3, 2, 1, 0, 0, 0), 3),
               latent = rep(c(0, 0, 0, 0, 2, 2), 3), null = rep(c(-1, 0, 1, 0, -1, 1), 3))
colnames(truth) <- meta$file
role <- c("B_DE", "A_DE", "latent", "null")
selected <- truth
selected["condition", c(6, 12, 18)] <- NA
original <- truth
slopes <- c("attenuation_condition", "attenuation_condition_a_de",
            "attenuation_condition_b_de", "attenuation_latent")
for (gain in c(1, 0.5, -0.5)) {
  x <- sweep(gain * selected, 1, c(7, -4, 3, 1), "+")
  result <- simulation$recovery_metrics(x, truth, meta, role)
  stopifnot(same(result[slopes], rep(gain, 4)),
            same(result[c("shift_condition", "shift_latent")], c(0, 0)),
            same(result["residual_lab_rms"], 0),
            same(result["within_lab_rmse"], result["rmse_to_truth"]),
            identical(truth, original))
  if (gain == 1) stopifnot(same(result["rmse_to_truth"], 0))
  # Centred residual sums of squares per row: 9.6, 24, 16 and 12 over 69 observed cells.
  if (gain == 0.5) stopifnot(same(result["rmse_to_truth"], sqrt(0.25 * 61.6 / 69)))
}
# A lab offset is lab structure: it is the whole residual and leaves no within-lab error.
offset <- sweep(selected, 2, c(0, 1, -1)[as.integer(factor(meta$lab))], "+")
result <- simulation$recovery_metrics(offset, truth, meta, role)
stopifnot(same(result["within_lab_rmse"], 0),
          same(result["residual_lab_rms"], result["rmse_to_truth"]),
          same(result[c("shift_condition", "shift_latent")], c(0, 0)))
# A shift shared by every feature moves the mean contrast but partly cancels in the pooled
# slope, because A_DE and B_DE targets have opposite signs.
shifted <- selected
shifted[, meta$condition == "B"] <- shifted[, meta$condition == "B"] + 0.5
result <- simulation$contrast_recovery(shifted, selected, meta, role)
stopifnot(same(result["shift_condition"], 0.5), same(result["shift_latent"], 0),
          same(result["attenuation_condition_b_de"], 1.5),
          same(result["attenuation_condition_a_de"], 0.75),
          same(result["attenuation_condition"], 0.9))
# Selection changes the B_DE contrast from 2 to 1, even with perfect observed data.
selection <- simulation$contrast_recovery(selected, truth, meta, role)
stopifnot(same(selection["attenuation_condition"], 0.75),
          same(selection["attenuation_condition_b_de"], 0.5),
          same(selection["attenuation_latent"], 1),
          same(simulation$recovery_metrics(selected, truth, meta, role)["rmse_to_truth"], 0))
no_contrast <- selected
no_contrast["condition", meta$condition == "B"] <- NA
undefined <- simulation$recovery_metrics(no_contrast, truth, meta, role)
stopifnot(is.na(undefined["attenuation_condition_b_de"]))
stopifnot(fails(simulation$recovery_metrics(selected[, 18:1], truth, meta, role)),
          fails(simulation$recovery_metrics(selected[4:1, ], truth, meta, role)),
          fails(simulation$recovery_metrics(selected, truth, meta[18:1, ], role)))

simulation$correct_central <- function(x, meta, arm) list(input = x, corrected = x)
simulation$correct_fedrbe <- function(x, meta, arm) x[nrow(x):1, ncol(x):1]
baseline <- setNames(lapply(simulation$arms, function(arm) {
  simulation$correct_both(truth, meta, arm)
}), simulation$arms)
observed <- 0.5 * selected
attr(observed, "batch_params") <- list(additive = c(lab1 = 0, lab2 = 1, lab3 = -1),
                                       multiplicative = c(lab1 = 2, lab2 = 1, lab3 = 4),
                                       noise_sd = c(lab1 = 1, lab2 = 0.5, lab3 = 2))
data <- list(truth = truth, observed = observed, meta = meta, role = role)
output <- simulation$evaluate_dataset("missingness", 2L, data, baseline = baseline)$metrics
metric <- function(stage, name) {
  values <- output$value[output$stage == stage & output$metric == name]
  stopifnot(length(values) == 2L)
  values
}
input <- function(name) {
  values <- output$value[output$stage == "input" & output$metric == name]
  stopifnot(length(values) == 1L)
  values
}
stopifnot(input("eta2_condition") == 0, input("eta2_latent_in_b") == 0,
          input("offset_lab2") == 1, input("noise_sd_lab3") == 2)
stopifnot(same(metric("baseline_after", "rmse_to_truth"), c(0, 0)),
          same(metric("baseline_after", "attenuation_condition"), c(1, 1)),
          same(metric("paired_after", "attenuation_condition"), c(-0.5, -0.5)),
          same(metric("paired_after", "attenuation_latent"), c(-0.5, -0.5)),
          same(metric("observed_truth", "attenuation_condition"), c(0.75, 0.75)),
          same(metric("after_vs_complete", "attenuation_condition"), c(0.375, 0.375)),
          same(metric("before", "gain_condition_lab2"), c(0.5, 0.5)),
          same(metric("before", "gain_latent_lab3"), c(0.5, 0.5)),
          all(metric("truth", "de_condition_fp") == 0), all(metric("truth", "de_condition_fn") == 0),
          all(metric("joint", "de_condition_fp") == 0), all(metric("after", "de_condition_recall") == 1))
identity_baseline <- lapply(baseline, function(arm) {
  simulation$correct_both(data$observed, meta, "with_condition")
})
control <- simulation$evaluate_dataset("missingness", 2L, data, identity_baseline)$metrics
stopifnot(any(control$stage == "paired_after"),
          all(abs(control$value[control$stage == "paired_after"]) < 1e-12))

# Every run and arm must be compared; scoring must use fedRBE, including paired baselines.
calls <- character()
simulation$correct_fedrbe <- function(x, meta, arm) {
  calls <<- c(calls, arm)
  x[, meta$condition == "B"] <- x[, meta$condition == "B"] + 5e-9
  x[nrow(x):1, ncol(x):1]
}
federated_baseline <- setNames(lapply(simulation$arms, function(arm) {
  simulation$correct_both(data$observed, meta, arm)
}), simulation$arms)
calls <- character()
fed <- simulation$evaluate_dataset("missingness", 2L, data, federated_baseline)$metrics
stopifnot(identical(calls, simulation$arms),
          sum(fed$stage == "fedrbe_vs_central" & fed$metric == "max_diff") == 2L,
          all(fed$value[fed$stage == "fedrbe_vs_central" & fed$metric == "max_diff"] > 0),
          all(abs(fed$value[fed$stage == "paired_after"]) < 1e-12))
for (stage in c("after", "after_vs_complete")) {
  recovered <- fed$value[fed$stage == stage & fed$metric == "attenuation_condition_b_de"]
  central <- output$value[output$stage == stage &
                            output$metric == "attenuation_condition_b_de"]
  stopifnot(all(recovered > central + 1e-9))
}
calls <- character()
balanced <- simulation$evaluate_dataset("balanced", 2L, data, federated_baseline)$metrics
stopifnot(length(calls) == 0L,
          sum(balanced$stage == "fedrbe_vs_central" & balanced$metric == "max_diff") == 2L,
          all(balanced$value[balanced$stage == "after" &
                               balanced$metric == "attenuation_condition_b_de"] > 0.5))
for (bad in list(truth[-1, ], truth[, -1], replace(truth, 1, NA_real_), truth + 1e-6)) {
  simulation$correct_fedrbe <- function(...) bad
  stopifnot(fails(simulation$correct_both(truth, meta, "with_condition")))
}

summary <- simulation$summarise_values(c(1, 2, 3, NA_real_, Inf), deviation = TRUE)
half_width <- qt(0.975, df = 2) / sqrt(3)
stopifnot(summary$n_runs == 3L, summary$n_missing == 2L, summary$mean == 2, summary$median == 2,
          same(summary$mean_abs_dev_1, 1), same(summary$mcse, 1 / sqrt(3)),
          same(summary$ci_low, 2 - half_width), same(summary$ci_high, 2 + half_width),
          is.na(simulation$summarise_values(c(1, 2, 3))$mean_abs_dev_1))
for (values in list(numeric(), NA_real_, c(4, NA_real_))) {
  summary <- simulation$summarise_values(values)
  stopifnot(is.na(summary$mcse), is.na(summary$ci_low), is.na(summary$ci_high))
}
constant <- simulation$summarise_values(rep(4, 3))
stopifnot(constant$mcse == 0, constant$ci_low == 4, constant$ci_high == 4)

# Unit-gain deviations apply to slope levels, not scenario-minus-baseline differences.
summary_dir <- tempfile("robustness_summary_")
dir.create(file.path(summary_dir, "runs"), recursive = TRUE)
for (run in 1:2) {
  metrics <- data.frame(scenario = "confounded_p50", run, arm = "with_condition",
                        stage = c("input", "before", "after", "paired_after", "observed_truth",
                                  "fedrbe_vs_central"),
                        metric = c("eta2_condition", "gain_condition_lab1", "attenuation_condition",
                                   "attenuation_condition", "shift_condition", "max_diff"),
                        feature = NA_character_,
                        value = c(0, c(0.5, 1.5)[run], c(0.75, 1.25)[run], 0, 0, 5e-9))
  simulation$write_run(list(metrics = metrics), summary_dir, "confounded_p50", run)
}
simulation$summarise(summary_dir)
summary <- read.delim(file.path(summary_dir, "summary.tsv"))
equivalence <- read.delim(file.path(summary_dir, "equivalence.tsv"))
stopifnot(identical(equivalence$run, 1:2), all(equivalence$value == 5e-9))
stopifnot(same(summary$mean_abs_dev_1[summary$stage == "before"], 0.5),
          same(summary$mean_abs_dev_1[summary$stage == "after"], 0.25),
          is.na(summary$mean_abs_dev_1[summary$stage == "paired_after"]),
          summary$mean[summary$stage == "paired_after"] == 0)
unlink(summary_dir, recursive = TRUE)

# Complete 6,000-by-600 fixtures check the offset arithmetic, the abundance tertiles and the
# balanced closed form without running the simulation.
pipeline <- new.env()
source("evaluation_data/simulated/robustness_simulation.R", local = pipeline)
close_numeric <- function(actual, expected, tolerance = 1e-8) {
  length(actual) == length(expected) && all(is.finite(actual)) && all(is.finite(expected)) &&
    max(abs(actual - expected)) <= tolerance
}
generated <- c("balanced", "mild_imbalanced", "strong_imbalanced", "intensity_dependent",
               "missingness", "confounded_p50", "confounded_p30", "confounded_p15", "confounded_p05")
stopifnot(identical(pipeline$scenarios, c(generated, "intensity_lower")),
          identical(pipeline$parse_options(character())$scenarios, pipeline$scenarios),
          identical(pipeline$parse_options(c("--scenarios", "intensity_lower"))$scenarios,
                    "intensity_lower"),
          identical(pipeline$derived$balanced,
                    c("balanced", "intensity_lower", "intensity_dependent", "missingness")),
          identical(vapply(generated, pipeline$run_seed, integer(1), run = 1L),
                    setNames(20261017L + 1000L * 1:9, generated)),
          pipeline$run_seed("balanced", 10L) == 20262026L)

# A tiny complete input pins the pooled-quantile and direct-log arithmetic at both severities.
offset_meta <- meta
offset_meta$lab <- sub("lab", "batch", offset_meta$lab)
offset_data <- list(truth = truth, observed = truth + 0.25, meta = offset_meta, role = role)
attr(offset_data$observed, "batch_params") <- list(
  additive = c(batch1 = 0, batch2 = 1, batch3 = -1),
  noise_sd = c(batch1 = 1, batch2 = 0.5, batch3 = 2))
quantiles <- quantile(2^truth, c(0.1, 0.25), names = FALSE)
for (severity in c(0.5, 1)) {
  transformed <- pipeline$add_intensity_offset(offset_data, severity)
  expected <- offset_data$observed
  for (i in 1:2) {
    cols <- offset_meta$lab == c("batch2", "batch3")[i]
    expected[, cols] <- log2(2^expected[, cols] + severity * quantiles[i])
  }
  stopifnot(identical(transformed$observed, expected), identical(transformed$truth, truth),
            identical(transformed$observed[, offset_meta$lab == "batch1"],
                      offset_data$observed[, offset_meta$lab == "batch1"]))
}
stopifnot(identical(pipeline$add_intensity_offset(offset_data),
                    pipeline$add_intensity_offset(offset_data, 1)))

# Full role counts and deterministic values; no generator or correction runs.
fixture_meta <- data.frame(file = paste0("fixture", 1:600),
                           lab = rep(paste0("batch", 1:3), each = 200),
                           condition = factor(rep(rep(c("A", "B"), each = 100), 3)),
                           latent = rep(c(rep(FALSE, 100), rep(TRUE, 60), rep(FALSE, 40)), 3))
rownames(fixture_meta) <- fixture_meta$file
fixture_role <- rep(c("A_DE", "B_DE", "latent", "null"), c(1125, 1125, 150, 3600))
fixture_abundance <- rep(c(2, 4, 6, 8, 10), length.out = 6000)
fixture_truth <- matrix(fixture_abundance, 6000, 600,
                        dimnames = list(paste0("fixture_feature", 1:6000), fixture_meta$file))
fixture_truth[fixture_role == "A_DE", fixture_meta$condition == "A"] <-
  fixture_truth[fixture_role == "A_DE", fixture_meta$condition == "A"] + 1.5
fixture_truth[fixture_role == "B_DE", fixture_meta$condition == "B"] <-
  fixture_truth[fixture_role == "B_DE", fixture_meta$condition == "B"] + 1.75
fixture_truth[fixture_role == "latent", fixture_meta$latent] <-
  fixture_truth[fixture_role == "latent", fixture_meta$latent] - 1.25
fixture_observed <- fixture_truth + outer(cos(1:6000), sin(1:600)) * 0.1
fixture_observed <- sweep(fixture_observed, 2,
                          c(0, 0.75, -0.5)[match(fixture_meta$lab, paste0("batch", 1:3))], "+")
attr(fixture_observed, "batch_params") <- list(additive = c(batch1 = 0, batch2 = 0.75, batch3 = -0.5),
                                              noise_sd = c(batch1 = 0.1, batch2 = 0.2, batch3 = 0.3))
fixture <- list(truth = fixture_truth, observed = fixture_observed,
                meta = fixture_meta, role = fixture_role)
tertiles <- pipeline$abundance_tertiles(fixture_truth, fixture_meta, fixture_role)
stopifnot(identical(names(tertiles), rownames(fixture_truth)),
          identical(as.vector(table(factor(fixture_role, c("A_DE", "B_DE", "latent", "null")),
                                    tertiles)), rep(c(375L, 375L, 50L, 1200L), 3)))
for (role_name in unique(fixture_role)) {
  rows <- which(fixture_role == role_name)
  sorted <- rows[order(fixture_abundance[rows], rownames(fixture_truth)[rows])]
  stopifnot(identical(unname(tertiles[sorted]), rep(1:3, each = length(rows) / 3)))
}
# A role without truth contrast gives NA tertile slopes for that role only.
no_signal <- fixture_truth
no_signal[fixture_role == "A_DE", ] <- fixture_abundance[fixture_role == "A_DE"]
no_signal_bins <- pipeline$tertile_recovery(no_signal, no_signal, fixture_meta, fixture_role, tertiles)
stopifnot(all(is.na(no_signal_bins[paste0("attenuation_condition_a_de_bin", 1:3)])),
          same(no_signal_bins[paste0("attenuation_condition_b_de_bin", 1:3)], rep(1, 3)))

# Both central arms are the same balanced lab-mean projection; all features remain.
closed_form <- fixture_observed
for (lab in unique(fixture_meta$lab)) {
  cols <- fixture_meta$lab == lab
  closed_form[, cols] <- fixture_observed[, cols] - rowMeans(fixture_observed[, cols]) +
    rowMeans(fixture_observed)
}
for (arm in pipeline$arms) {
  control <- pipeline$correct_central(fixture_observed, fixture_meta, arm)
  stopifnot(identical(dim(control$corrected), c(6000L, 600L)),
            max(abs(control$corrected - closed_form)) < 1e-8,
            close_numeric(pipeline$contrast_recovery(control$corrected, fixture_truth,
                                               fixture_meta, fixture_role),
                 pipeline$contrast_recovery(fixture_observed, fixture_truth,
                                               fixture_meta, fixture_role)))
}
# Tertile metrics equal whole-set recovery within each tertile, follow feature names rather than
# row order, and appear only on complete data.
stage <- pipeline$evaluate_stage(fixture_observed, fixture_truth, fixture_meta, fixture_role,
                                 partition = FALSE, tertiles = tertiles)
for (b in 1:3) {
  rows <- tertiles == b
  expected <- pipeline$contrast_recovery(fixture_observed[rows, ], fixture_truth[rows, ],
                                         fixture_meta, fixture_role[rows])
  stopifnot(same(stage$metrics[paste0(names(expected), "_bin", b)], expected))
}
bin_names <- function(metrics) grep("_bin[1-3]$", names(metrics), value = TRUE)
reversed <- 6000:1
incomplete <- fixture_observed
incomplete[1, 1] <- NA
stopifnot(length(bin_names(stage$metrics)) == 18L,
          same(pipeline$tertile_recovery(fixture_observed[reversed, ], fixture_truth[reversed, ],
                                         fixture_meta, fixture_role[reversed], tertiles),
               stage$metrics[bin_names(stage$metrics)]),
          !length(bin_names(pipeline$evaluate_stage(incomplete, fixture_truth, fixture_meta,
                                                     fixture_role, partition = FALSE,
                                                     tertiles = tertiles)$metrics)),
          !length(bin_names(pipeline$evaluate_stage(fixture_observed, fixture_truth, fixture_meta,
                                                     fixture_role, partition = FALSE)$metrics)))

# Both offset severities execute the real diagnostic on a small complete fixture.
diagnostic <- new.env()
source("evaluation_data/simulated/robustness_simulation.R", local = diagnostic)
stopifnot(requireNamespace("variancePartition", quietly = TRUE))
diagnostic_meta <- offset_meta
diagnostic_meta$lab <- factor(diagnostic_meta$lab)
diagnostic_truth <- truth[rep(seq_len(nrow(truth)), 2), ]
rownames(diagnostic_truth) <- paste0("diagnostic", seq_len(nrow(diagnostic_truth)))
diagnostic_truth <- diagnostic_truth +
  outer(seq_len(nrow(diagnostic_truth)), seq_len(ncol(diagnostic_truth)),
        function(i, j) 0.2 * sin(i * j))
diagnostic_observed <- sweep(diagnostic_truth, 2,
                              c(0, 0.75, -0.5)[diagnostic_meta$lab], "+")
attr(diagnostic_observed, "batch_params") <- list(
  additive = c(batch1 = 0, batch2 = 0.75, batch3 = -0.5),
  noise_sd = c(batch1 = 0.1, batch2 = 0.2, batch3 = 0.3))
diagnostic_data <- list(truth = diagnostic_truth, observed = diagnostic_observed,
                        meta = diagnostic_meta, role = rep(role, 2))
# Keep real central filtering/correction, with only the expensive app protocol doubled.
diagnostic$correct_fedrbe <- function(x, meta, arm) {
  corrected <- diagnostic$correct_central(x, meta, arm)$corrected
  corrected[nrow(corrected):1, ncol(corrected):1]
}
diagnostic_baseline <- setNames(lapply(diagnostic$arms, function(arm) {
  diagnostic$correct_both(diagnostic_observed, diagnostic_meta, arm)
}), diagnostic$arms)
real_partition <- diagnostic$variance_partition_long
partition_calls <- list()
diagnostic$variance_partition_long <- function(data, metadata, form) {
  partition_calls[[length(partition_calls) + 1L]] <<- list(data = data, form = form)
  real_partition(data, metadata, form)
}
diagnostic_dir <- tempfile("robustness_variance_")
dir.create(file.path(diagnostic_dir, "runs"), recursive = TRUE)
for (scenario in c("intensity_lower", "intensity_dependent")) {
  severity <- if (scenario == "intensity_lower") 0.5 else 1
  offset <- diagnostic$add_intensity_offset(diagnostic_data, severity)
  partition_calls <- list()
  result <- diagnostic$evaluate_dataset(scenario, 1L, offset, diagnostic_baseline)
  stopifnot(length(partition_calls) == 4L,
            all(vapply(partition_calls, function(call) {
              identical(attr(terms(call$form), "term.labels"), c("condition", "lab")) &&
                identical(dimnames(call$data), dimnames(diagnostic_truth)) &&
                !anyNA(call$data)
            }, logical(1))))
  expected <- expand.grid(arm = diagnostic$arms, stage = c("before", "after", "truth"),
                          metric = c("variance_condition", "variance_lab"),
                          feature = rownames(diagnostic_truth), stringsAsFactors = FALSE)
  coverage_key <- function(table) do.call(paste, c(table[names(expected)], sep = ":"))
  stopifnot(nrow(result$variance) == nrow(expected),
            setequal(coverage_key(result$variance), coverage_key(expected)))
  direct <- variancePartition::fitExtractVarPartModel(offset$observed, ~condition + lab,
                                                      diagnostic_meta)
  for (arm in diagnostic$arms) {
    for (stage in c("before", "after", "truth")) {
      for (term in c("condition", "lab")) {
        values <- result$variance[result$variance$arm == arm &
                                    result$variance$stage == stage &
                                    result$variance$metric == paste0("variance_", term), ]
        median_value <- result$metrics$value[result$metrics$arm == arm &
                            result$metrics$stage == stage &
                            result$metrics$metric == paste0("median_variance_", term)]
        stopifnot(length(median_value) == 1L,
                  abs(median_value - median(values$value)) < 1e-5)
        if (stage == "before") {
          stopifnot(same(values$value,
                         signif(direct[values$feature, term], 6)))
        }
      }
    }
  }
  diagnostic$write_run(result, diagnostic_dir, scenario, 1L)
}
diagnostic$summarise(diagnostic_dir)
diagnostic_summary <- read.delim(file.path(diagnostic_dir, "summary.tsv"))
median_summary <- diagnostic_summary[
  startsWith(diagnostic_summary$metric, "median_variance_"), ]
stopifnot(nrow(median_summary) == 24L, all(median_summary$n_runs == 1L))

unlink(diagnostic_dir, recursive = TRUE)

# The shared helper must fail explicitly when variancePartition is unavailable.
diagnostic$requireNamespace <- function(...) FALSE
diagnostic$fixed_effect_variance_partition <- function(...) stop("Forbidden fallback")
missing_package <- tryCatch(real_partition(diagnostic_truth, diagnostic_meta,
                                           ~condition + lab), error = identity)
stopifnot(inherits(missing_package, "error"),
          grepl("variancePartition", conditionMessage(missing_package)),
          !grepl("Forbidden fallback", conditionMessage(missing_package)))

# Interaction-model variance partitioning on synthetic features: lab, condition and
# lab-condition effects, a cell-balanced feature whose group components are exactly zero,
# missing observations and failures.
vp <- new.env()
source("evaluation_data/simulated/variance_partition.R", local = vp)
vp_meta <- data.frame(lab = rep(paste0("batch", 1:3), each = 40),
                      condition = rep(c("A", "B"), 60))
rownames(vp_meta) <- paste0("v", 1:120)
cell <- as.integer(factor(paste(vp_meta$lab, vp_meta$condition)))
set.seed(41)
noise <- matrix(rnorm(5 * 120, sd = 0.5), 5)
vp_x <- rbind(lab = c(-3, 0, 3)[(cell + 1L) %/% 2L] + noise[1, ],
              condition = c(-2, 2)[2L - cell %% 2L] + noise[2, ],
              # Cell means 2, -2, -2, 2, 0, 0: no lab or condition main effect.
              interaction = c(2, -2, -2, 2, 0, 0)[cell] + noise[3, ],
              balanced = ave(seq_len(120), cell, FUN = function(i) rep(c(-1, 1), 10)),
              partial = c(-3, 0, 3)[(cell + 1L) %/% 2L] + noise[4, ],
              incomplete = c(2, -2, -2, 2, 0, 0)[cell] + noise[5, ],
              one_condition = noise[5, ], flat = rep(1, 120))
vp_x["partial", 1:30] <- NA
vp_x["incomplete", cell == 2L] <- NA
vp_x["one_condition", vp_meta$condition == "B"] <- NA
colnames(vp_x) <- rownames(vp_meta)
fit <- vp$fit_fractions(vp_x, vp_meta)
ok <- fit$status == "ok"
components <- as.matrix(fit[names(vp$vp_components)])
rownames(components) <- fit$feature
stopifnot(identical(fit$feature, rownames(vp_x)),
          identical(names(vp$vp_components), c("lab", "condition", "interaction", "residual")),
          identical(ok, !rownames(vp_x) %in% c("one_condition", "flat")),
          all(abs(rowSums(components[ok, ]) - 1) < 1e-8),
          components["lab", "lab"] > 0.5, components["condition", "condition"] > 0.5,
          components["interaction", "interaction"] > 0.5,
          components["lab", "interaction"] < 0.1, components["interaction", "lab"] < 0.1,
          all(components["balanced", c("lab", "condition", "interaction")] < vp$vp_zero),
          grepl("response variance", fit$error[fit$feature == "flat"]),
          nzchar(fit$error[fit$feature == "one_condition"]), all(is.na(fit$error[ok])),
          fails(vp$fit_fractions(vp_x, vp_meta[120:1, ])))
# Missing observations: the feature is fitted on its observed samples only.
keep <- !is.na(vp_x["partial", ])
alone <- vp$fit_fractions(vp_x[c("partial", "lab"), keep], vp_meta[keep, ])
stopifnot(identical(alone$status, c("ok", "ok")),
          max(abs(unlist(alone[1, names(vp$vp_components)]) -
                    components["partial", ])) < 1e-4)
structure <- vp$observation_structure(vp_x, vp_meta)
stopifnot(identical(structure$crossed, rownames(vp_x) != "incomplete"),
          identical(structure$estimable, rownames(vp_x) != "one_condition"),
          identical(structure$n_obs[structure$feature == "partial"], 90))

# Matching keeps the features and cells retained by both arms, aligned by name.
truth <- matrix(1:24 + 0.5, 4, dimnames = list(paste0("t", 1:4), paste0("s", 1:6)))
no_label <- truth[1:3, ]
no_label["t2", "s1"] <- NA
label_aware <- truth[2:4, 6:1]
label_aware["t3", "s2"] <- NA
native <- list(no_label = list(input = no_label, corrected = no_label + 10),
               label_aware = list(input = label_aware, corrected = label_aware + 20))
matched <- vp$matched_states(native, truth)
expected_mask <- matrix(FALSE, 2, 6, dimnames = list(c("t2", "t3"), colnames(truth)))
expected_mask["t2", "s1"] <- expected_mask["t3", "s2"] <- TRUE
expected_losses <- c(3, 1, 17, 6, 1)
stopifnot(identical(names(matched$states), vp$vp_states),
          all(vapply(matched$states, function(x) identical(is.na(x), expected_mask),
                     logical(1))),
          same(matched$states$no_label[!expected_mask], truth[2:3, ][!expected_mask] + 10),
          same(matched$states$label_aware[!expected_mask], truth[2:3, ][!expected_mask] + 20),
          same(matched$states$uncorrected[!expected_mask], truth[2:3, ][!expected_mask]),
          same(matched$matching, c(expected_losses, expected_losses, 2, 10)),
          identical(names(matched$matching)[c(1, 5, 11, 12)],
                    c("native_features_no_label", "matching_blanked_cells_no_label",
                      "common_features", "common_observed_cells")))

# Summaries use the features fitted in every available state; a state whose fit fails
# entirely is reported as unavailable instead of shrinking the others.
states <- setNames(rep(list(vp_x), 4), vp$vp_states)
states$no_label["interaction", ] <- 1
states$truth[!is.na(states$truth)] <- 1
result <- vp$analyse_dataset(states, vp_meta, rep(c("A_DE", "null"), 4))
summary <- result$summary
value <- function(state, component, statistic) {
  summary$value[summary$state == state & summary$component == component &
                  summary$statistic == statistic]
}
valid <- setdiff(rownames(vp_x), c("interaction", "one_condition", "flat"))
stopifnot(nrow(result$features) == 4L * nrow(vp_x),
          value("truth", "all", "available") == 0, value("truth", "lab", "n_features") == 0,
          is.na(value("truth", "lab", "mean")),
          value("uncorrected", "all", "excluded_by_other_states") == 1,
          value("no_label", "all", "failed_features") == 3,
          value("no_label", "all", "failed_estimable_features") == 2,
          value("uncorrected", "all", "failed_estimable_features") == 1,
          value("matched", "all", "valid_features") == length(valid),
          value("matched", "all", "unestimable_features") == 1,
          value("uncorrected", "lab", "n_features") == length(valid),
          abs(value("uncorrected", "lab", "mean") - mean(components[valid, "lab"])) < 1e-12,
          abs(value("label_aware", "interaction", "median") -
                median(components[valid, "interaction"])) < 1e-12,
          value("uncorrected", "all", "max_sum_error") < 1e-8,
          value("uncorrected", "all", "zero_lab") >= 1,
          !any(grepl("[\t\n]", result$features$error)))
# With no state fitted, no feature is valid.
failed <- setNames(rep(list(states$truth), 4), vp$vp_states)
none <- vp$analyse_dataset(failed, vp_meta, rep(c("A_DE", "null"), 4))$summary
stopifnot(none$value[none$state == "matched" & none$statistic == "valid_features"] == 0)

# Every fit reaches the maximum likelihood, compared with an exhaustive theta grid refined
# locally. Samples ordered by condition with unequal lab compositions make the template
# refits of variancePartition 1.40.2 stop early under lme4 2.0.6; lab-centred features with
# lab-dependent condition shares (as after a no-label correction) give lme4's default start
# a lower local maximum.
grid_deviance <- function(y, meta) {
  deviance <- lme4::lmer(update(vp$vp_formula, y ~ .), cbind(meta, y = y), REML = FALSE,
                         control = vp$vp_control, devFunOnly = TRUE)
  grid <- as.matrix(expand.grid(rep(list(c(0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1, 1.5, 2,
                                           3, 5)), 3)))
  on_grid <- apply(grid, 1, deviance)
  min(on_grid, minqa::bobyqa(grid[which.min(on_grid), ], deviance, lower = rep(0, 3),
                             control = list(rhobeg = 0.05, rhoend = 1e-8))$fval)
}
set.seed(4)
ordered <- do.call(rbind, lapply(1:3, function(l) {
  data.frame(lab = paste0("batch", l),
             condition = rep(c("A", "B"), list(c(26, 4), c(9, 51), c(26, 4))[[l]]))
}))
ordered <- ordered[order(ordered$condition, sample(nrow(ordered))), ]
rownames(ordered) <- paste0("o", seq_len(nrow(ordered)))
lab_index <- as.integer(factor(ordered$lab))
template_x <- t(replicate(10, rnorm(3, 0, 1.5)[lab_index] +
                            (ordered$condition == "B") * rnorm(1, 0, 0.5) + rnorm(120)))
set.seed(5)
centred <- data.frame(lab = rep(paste0("batch", 1:3), each = 80))
centred$condition <- ifelse(runif(240) < c(batch1 = 0.8, batch2 = 0.5,
                                           batch3 = 0.2)[centred$lab], "B", "A")
rownames(centred) <- paste0("c", 1:240)
centred_x <- t(replicate(12, {
  y <- (centred$condition == "B") * rnorm(1, 0, 1.5) +
    rnorm(3, 0, 2)[as.integer(factor(centred$lab))] + rnorm(240)
  y - ave(y, centred$lab)
}))
for (case in list(list(template_x, ordered), list(centred_x, centred))) {
  x <- case[[1]]
  dimnames(x) <- list(paste0("f", seq_len(nrow(x))), rownames(case[[2]]))
  fit <- vp$fit_fractions(x, case[[2]])
  gaps <- -2 * fit$log_likelihood -
    vapply(seq_len(nrow(x)), function(i) grid_deviance(x[i, ], case[[2]]), numeric(1))
  stopifnot(all(fit$status == "ok"), all(gaps < 1e-3))
}
cat("Robustness simulation numerical and variance diagnostic checks passed.\n")
