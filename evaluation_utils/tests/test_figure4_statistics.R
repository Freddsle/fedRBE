# Run from the repository root: pixi run Rscript evaluation_utils/tests/test_figure4_statistics.R
# Check the Figure 4/S7 rank tests on toy data, then rebuild the label-aware panel
# figures into a temporary directory and check the statistics against the source scores.
source(file.path("evaluation_utils", "figure_aggregate_utils.R"))
source(file.path("evaluation_utils", "figure4_statistics.R"))
same <- function(actual, expected) {
  isTRUE(all.equal(actual, expected, tolerance = 1e-12, check.attributes = FALSE))
}

# Brute-force references: every sign vector and every split, one at a time.
wilcoxon_reference <- function(diff) {
  ranks <- rank(abs(diff))
  signed <- (ranks * sign(diff))[diff != 0]
  n <- length(signed)
  null <- vapply(0:(2^n - 1), function(bits) {
    abs(sum(ifelse(bitwAnd(bits, 2^(0:(n - 1))) > 0, -1, 1) * abs(signed)))
  }, numeric(1))
  mean(null >= abs(sum(signed)) - 1e-10)
}
mann_whitney_reference <- function(a, b) {
  ranks <- rank(c(a, b))
  center <- length(a) * (length(ranks) + 1) / 2
  null <- vapply(utils::combn(length(ranks), length(a), simplify = FALSE), function(index) {
    abs(sum(ranks[index]) - center)
  }, numeric(1))
  mean(null >= abs(sum(ranks[seq_along(a)]) - center) - 1e-10)
}

# Constant groups reach the smallest attainable p; identical groups give p = 1.
stopifnot(same(wilcoxon_pratt_exact(rep(0.4, 10)), 2 / 2^10),
          same(mann_whitney_exact(rep(0, 10), rep(1, 10)), 2 / choose(20, 10)),
          wilcoxon_pratt_exact(rep(0, 10)) == 1,
          mann_whitney_exact(rep(0.5, 10), rep(0.5, 10)) == 1)

set.seed(3)
diff <- c(round(stats::rnorm(7), 2), 0)
a <- c(0.2, 0.5, 0.4, 0.9, 0.1)
b <- c(0.7, 0.8, 0.6, 0.65)
tied_a <- c(1, 1, 0.5, 0.2)
tied_b <- c(1, 0.5, 0.5, 0.1, 0.1)
stopifnot(same(wilcoxon_pratt_exact(diff), wilcoxon_reference(diff)),
          same(mann_whitney_exact(a, b), mann_whitney_reference(a, b)),
          same(mann_whitney_exact(b, a), mann_whitney_reference(b, a)),
          same(mann_whitney_exact(tied_a, tied_b), mann_whitney_reference(tied_a, tied_b)))

# Pratt: 0, 0, 0.1, 0.2, -0.3 have signed ranks 3, 4, -5, so the sum is +2, not the
# 0 that dropping zeros before ranking gives.
pratt_example <- c(0, 0, 0.1, 0.2, -0.3)
stopifnot(same(wilcoxon_pratt_exact(pratt_example), mean(abs(c(12, 6, 4, 2, -2, -4, -6, -12)) >= 2)))

# Bootstrap CIs collapse for constant data, contain the estimate otherwise and leave
# the global RNG stream unchanged.
stopifnot(same(bootstrap_ci(rep(0, 10), rep(0.3, 10), paired = TRUE), c(0.3, 0.3)),
          same(bootstrap_ci(rep(0, 10), rep(0.3, 10), paired = FALSE), c(0.3, 0.3)))
ci <- bootstrap_ci(a, b, paired = FALSE)
stopifnot(ci[[1]] <= mean(b) - mean(a), mean(b) - mean(a) <= ci[[2]])
set.seed(9)
expected_draw <- stats::runif(1)
set.seed(9)
invisible(bootstrap_ci(a, b, paired = FALSE))
stopifnot(same(stats::runif(1), expected_draw))
cat("Rank tests match brute-force references; bootstrap CIs behave\n")

stopifnot(identical(figure4_marker(c(0.002, 0.3)), c("*", "n.s.")))

# Canonical label-aware panel results, written to a temporary Figure 4 output.
output <- file.path(tempdir(), "figure4_test.png")
Sys.setenv(CORRECTION_ARM = "label_aware", FIGURE4_FEATURE_SPACE = "panel", FIGURE4_OUTPUT = output)
script <- new.env()
source(file.path("evaluation", "aggregate_figure4.R"), local = script)
Sys.unsetenv(c("CORRECTION_ARM", "FIGURE4_FEATURE_SPACE", "FIGURE4_OUTPUT"))
statistics_file <- file.path(tempdir(), "figure4_test_statistics.tsv")
stopifnot(file.exists(output), file.exists(statistics_file),
          file.exists(file.path(tempdir(), "figureS7_label_aware.png")))

stats <- script$figure4_stats
data <- script$figure4_data
loo <- data$leave_one_cohort_out
cohort_seeds <- table(paste(loo$Task, loo$predicted_client_name), loo$Scenario)
stopifnot(nrow(cohort_seeds) == 17L, all(cohort_seeds == 10L),
          identical(as.vector(table(stats$panel)[c("4a ARI", "4c train/test", "S7 LOCO")]),
                    c(16L, 16L, 68L)),
          all(stats$n_a == 10L), all(stats$n_b == 10L),
          nrow(utils::read.delim(statistics_file)) == nrow(stats))

# Effect sizes equal group means computed directly from the plotted scores.
expected_diff <- function(rows, scores, value, unit) {
  means <- tapply(scores[[value]], paste(scores[[unit]], scores$Scenario), mean)
  as.vector(means[paste(rows$x_level, rows$group_b)] - means[paste(rows$x_level, rows$group_a)])
}
for (check in list(list("4a ARI", data$ari, "ARI", "DatasetLabel"),
                   list("4c train/test", data$train_test_split, "metric_value", "Task"),
                   list("S7 LOCO", loo, "metric_value", "predicted_client_name"))) {
  rows <- stats[stats$panel == check[[1]], , drop = FALSE]
  stopifnot(same(rows$diff, expected_diff(rows, check[[2]], check[[3]], check[[4]])))
}

# Known canonical outcomes: ARI improves in every seed with correction; the
# centralized RF is at the ceiling in three train/test tasks; Quartet LOCO is 1.0 throughout.
ari_primary <- stats[stats$panel == "4a ARI" & stats$role == "primary", ]
split_c1 <- stats[stats$panel == "4c train/test" & stats$contrast == "C1" &
                    !startsWith(stats$dataset, "ccRCC"), ]
quartet_loo <- stats[stats$panel == "S7 LOCO" & startsWith(stats$dataset, "Quartet"), ]
stopifnot(nrow(ari_primary) == 8L, all(ari_primary$p_raw == 2 / 2^10),
          nrow(split_c1) == 3L, all(split_c1$p_raw == 1),
          all(split_c1$note == "all pairs identical"),
          nrow(quartet_loo) == 12L, all(quartet_loo$p_raw == 1))

# Negligible differences between constant groups stay significant and are listed.
ccrcc_c4 <- stats[stats$panel == "4a ARI" & startsWith(stats$dataset, "ccRCC") &
                    stats$contrast == "C4", ]
negligible <- figure4_negligible_effects(stats)
stopifnot(nrow(ccrcc_c4) == 1L, ccrcc_c4$marker == "*", !ccrcc_c4$relevant,
          any(negligible$panel == "4a ARI" & startsWith(negligible$dataset, "ccRCC") &
                negligible$contrast == "C4"),
          all(negligible$p_raw < 0.05), all(abs(negligible$diff) < figure4_min_difference),
          all(ari_primary$marker == "*"))

# Group SDs equal the SDs of the plotted scores.
ari_sd <- tapply(data$ari$ARI, paste(data$ari$DatasetLabel, data$ari$Scenario), stats::sd)
ari_rows <- stats[stats$panel == "4a ARI", ]
stopifnot(same(ari_rows$sd_a, as.vector(ari_sd[paste(ari_rows$x_level, ari_rows$group_a)])),
          same(ari_rows$sd_b, as.vector(ari_sd[paste(ari_rows$x_level, ari_rows$group_b)])))

annotations <- figure4_annotation_data(stats, "4a ARI")
stopifnot(nrow(annotations) == 16L, setequal(annotations$level, 1:3),
          identical(annotations$label, ari_rows$marker))
cat("Label-aware panel statistics: 100 contrasts, effect sizes and canonical outcomes match\n")
