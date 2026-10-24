# Run from the repository root: pixi run Rscript evaluation_utils/tests/test_robustness_simulation.R
simulation <- new.env()
source("evaluation_data/simulated/robustness_simulation.R", local = simulation)
same <- function(actual, expected) {
  isTRUE(all.equal(actual, expected, tolerance = 1e-12, check.attributes = FALSE))
}
fails <- function(expr) inherits(tryCatch(force(expr), error = identity), "error")

stopifnot(all(simulation$params$runs == 10L),
          identical(simulation$parse_options(character())$runs, 1:10),
          identical(simulation$parse_options(c("--scenarios", "missingness"))$runs, 1:10))
valid <- simulation$parse_options(c("--runs", "1,5,10", "--out", "/tmp/data,cohort"))
stopifnot(identical(valid$runs, c(1L, 5L, 10L)), identical(valid$out, "/tmp/data,cohort"))
for (args in list(c("--run", "1"), "--runs", c("--out", "--runs"),
                  c("--runs", "1", "--runs", "2"), c("--scenarios", "unknown"),
                  c("--runs", "1.5"), c("--runs", "0"), c("--runs", "11"),
                  c("--runs", "1,,2"), c("--runs", ""))) {
  stopifnot(fails(simulation$parse_options(args)))
}

meta <- data.frame(file = paste0("s", 1:18), lab = rep(paste0("lab", 1:3), each = 6),
                   condition = factor(rep(c("A", "A", "A", "B", "B", "B"), 3)),
                   latent = rep(c(FALSE, FALSE, FALSE, FALSE, TRUE, TRUE), 3))
rownames(meta) <- meta$file
truth <- rbind(condition = rep(c(0, 0, 0, 0, 2, 4), 3),
               latent = rep(c(0, 0, 0, 0, 2, 2), 3), null = rep(c(-1, 0, 1, -1, 0, 1), 3))
colnames(truth) <- meta$file
role <- c("B_DE", "latent", "null")
selected <- truth
selected["condition", c(6, 12, 18)] <- NA
original <- truth
for (gain in c(1, 0.5, -0.5)) {
  x <- sweep(gain * selected, 1, c(7, -4, 1), "+")
  result <- simulation$recovery_metrics(x, truth, meta, role)
  stopifnot(same(result[c("attenuation_condition", "attenuation_latent")], rep(gain, 2)),
            identical(truth, original))
  if (gain == 1) stopifnot(same(result["rmse_to_truth"], 0))
  if (gain == 0.5) stopifnot(same(result["rmse_to_truth"], sqrt(47 / 255)))
}
# Selection changes the condition contrast from 2 to 1, even with perfect observed data.
selection <- simulation$contrast_recovery(selected, truth, meta, role)
stopifnot(same(selection["attenuation_condition"], 0.5),
          same(selection["attenuation_latent"], 1),
          same(simulation$recovery_metrics(selected, truth, meta, role)["rmse_to_truth"], 0))
no_contrast <- selected
no_contrast["condition", meta$condition == "B"] <- NA
undefined <- simulation$recovery_metrics(no_contrast, truth, meta, role)
stopifnot(is.na(undefined["attenuation_condition"]))
stopifnot(fails(simulation$recovery_metrics(selected[, 18:1], truth, meta, role)),
          fails(simulation$recovery_metrics(selected[3:1, ], truth, meta, role)),
          fails(simulation$recovery_metrics(selected, truth, meta[18:1, ], role)))

simulation$correct_central <- function(x, meta, arm) list(input = x, corrected = x)
simulation$correct_fedrbe <- function(...) stop("Unexpected verification in numerical fixture")
baseline <- setNames(lapply(simulation$arms, function(arm) {
  list(input = truth, corrected = truth)
}), simulation$arms)
data <- list(truth = truth, observed = 0.5 * selected, meta = meta, role = role)
output <- simulation$evaluate_dataset("missingness", 2L, data, baseline = baseline)$metrics
stopifnot(!any(grepl("^de_", output$metric)))
metric <- function(stage, name) {
  values <- output$value[output$stage == stage & output$metric == name]
  stopifnot(length(values) == 2L)
  values
}
stopifnot(same(metric("baseline_after", "rmse_to_truth"), c(0, 0)),
          same(metric("baseline_after", "attenuation_condition"), c(1, 1)),
          same(metric("paired_after", "attenuation_condition"), c(-0.5, -0.5)),
          same(metric("paired_after", "attenuation_latent"), c(-0.5, -0.5)),
          same(metric("selection", "attenuation_condition"), c(0.5, 0.5)))
identity_baseline <- lapply(baseline, function(arm) {
  list(input = data$observed, corrected = data$observed)
})
control <- simulation$evaluate_dataset("missingness", 2L, data, identity_baseline)$metrics
stopifnot(any(control$stage == "paired_after"),
          all(abs(control$value[control$stage == "paired_after"]) < 1e-12))

summary <- simulation$summarise_values(c(1, 2, 3, NA_real_, Inf))
half_width <- qt(0.975, df = 2) / sqrt(3)
stopifnot(summary$n_runs == 3L, summary$n_missing == 2L, summary$mean == 2,
          same(summary$mcse, 1 / sqrt(3)), same(summary$ci_low, 2 - half_width),
          same(summary$ci_high, 2 + half_width))
for (values in list(numeric(), NA_real_, c(4, NA_real_))) {
  summary <- simulation$summarise_values(values)
  stopifnot(is.na(summary$mcse), is.na(summary$ci_low), is.na(summary$ci_high))
}
constant <- simulation$summarise_values(rep(4, 3))
stopifnot(constant$mcse == 0, constant$ci_low == 4, constant$ci_high == 4)
cat("Robustness simulation numerical checks passed.\n")
