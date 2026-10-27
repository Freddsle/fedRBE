# Run from the repository root: pixi run Rscript evaluation_utils/tests/test_robustness_simulation.R
simulation <- new.env()
source("evaluation_data/simulated/robustness_simulation.R", local = simulation)
same <- function(actual, expected) {
  isTRUE(all.equal(actual, expected, tolerance = 1e-12, check.attributes = FALSE))
}
fails <- function(expr) inherits(tryCatch(force(expr), error = identity), "error")

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
confounded <- generate_metadata(data.frame(file = paste("s", 1:600, sep = ".")), "confounded_p05")
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
simulation$correct_fedrbe <- function(...) stop("Unexpected verification in numerical fixture")
baseline <- setNames(lapply(simulation$arms, function(arm) {
  list(input = truth, corrected = truth)
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
  list(input = data$observed, corrected = data$observed)
})
control <- simulation$evaluate_dataset("missingness", 2L, data, identity_baseline)$metrics
stopifnot(any(control$stage == "paired_after"),
          all(abs(control$value[control$stage == "paired_after"]) < 1e-12))

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
                        stage = c("input", "before", "after", "paired_after", "observed_truth"),
                        metric = c("eta2_condition", "gain_condition_lab1", "attenuation_condition",
                                   "attenuation_condition", "shift_condition"),
                        feature = NA_character_,
                        value = c(0, c(0.5, 1.5)[run], c(0.75, 1.25)[run], 0, 0))
  simulation$write_run(list(metrics = metrics), summary_dir, "confounded_p50", run)
}
simulation$summarise(summary_dir)
summary <- read.delim(file.path(summary_dir, "summary.tsv"))
stopifnot(same(summary$mean_abs_dev_1[summary$stage == "before"], 0.5),
          same(summary$mean_abs_dev_1[summary$stage == "after"], 0.25),
          is.na(summary$mean_abs_dev_1[summary$stage == "paired_after"]),
          summary$mean[summary$stage == "paired_after"] == 0)
unlink(summary_dir, recursive = TRUE)
cat("Robustness simulation numerical checks passed.\n")
