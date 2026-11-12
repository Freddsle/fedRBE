# Run from the repository root: pixi run Rscript evaluation_utils/tests/test_figure4_statistics.R
# Check exact rank tests and paired/unpaired intervals, then render both arms
# temporarily and compare the summaries with their source scores.
source(file.path("evaluation_utils", "figure_aggregate_utils.R"))
source(file.path("evaluation_utils", "figure4_statistics.R"))
same <- function(actual, expected) {
  isTRUE(all.equal(actual, expected, tolerance = 1e-12, check.attributes = FALSE))
}

# Independent small-sample references: enumerate every sign vector or group split.
wilcoxon_reference <- function(diff) {
  diff[abs(diff) <= figure4_statistics_tolerance] <- 0
  diff <- round(diff, 10)
  ranks <- rank(abs(diff))
  signed <- (ranks * sign(diff))[diff != 0]
  n <- length(signed)
  if (!n) return(1)
  null <- vapply(0:(2^n - 1), function(bits) {
    abs(sum(ifelse(bitwAnd(bits, 2^(0:(n - 1))) > 0, -1, 1) * abs(signed)))
  }, numeric(1))
  mean(null >= abs(sum(signed)) - 1e-10)
}
mann_whitney_reference <- function(a, b) {
  ranks <- rank(round(c(a, b), 10))
  center <- length(a) * (length(ranks) + 1) / 2
  null <- vapply(utils::combn(length(ranks), length(a), simplify = FALSE), function(index) {
    abs(sum(ranks[index]) - center)
  }, numeric(1))
  mean(null >= abs(sum(ranks[seq_along(a)]) - center) - 1e-10)
}

stopifnot(same(wilcoxon_pratt_exact(rep(0.4, 10)), 2 / 2^10),
          same(mann_whitney_exact(rep(0, 10), rep(1, 10)), 2 / choose(20, 10)),
          wilcoxon_pratt_exact(rep(0, 10)) == 1,
          mann_whitney_exact(rep(0.5, 10), rep(0.5, 10)) == 1)
a <- c(0.2, 0.5, 0.4, 0.9, 0.1)
b <- c(0.7, 0.8, 0.6, 0.65)
tied_a <- c(1, 1, 0.5, 0.2)
tied_b <- c(1, 0.5, 0.5, 0.1, 0.1)
for (diff in list(c(0, 0, 0.1, 0.2, -0.3), c(0.2, -0.1, 0.2, -0.4))) {
  stopifnot(same(wilcoxon_pratt_exact(diff), wilcoxon_reference(diff)))
}
stopifnot(same(mann_whitney_exact(a, b), mann_whitney_reference(a, b)),
          same(mann_whitney_exact(b, a), mann_whitney_reference(b, a)),
          same(mann_whitney_exact(tied_a, tied_b), mann_whitney_reference(tied_a, tied_b)),
          same(mann_whitney_exact(tied_b, tied_a), mann_whitney_exact(tied_a, tied_b)))
# This case distinguishes Pratt's zero-inclusive ranks from dropping zeros first.
pratt_example <- c(0, 0.1, 0.2, -0.3, -0.4)
stopifnot(same(wilcoxon_pratt_exact(pratt_example), 0.75),
          same(wilcoxon_reference(pratt_example[pratt_example != 0]), 0.625))

# Numerical equality is decided before rounding can split almost-identical values.
stopifnot(wilcoxon_pratt_exact(rep(9e-11, 10)) == 1,
          wilcoxon_pratt_exact(rep(1e-10, 10)) == 1,
          same(wilcoxon_pratt_exact(rep(1.1e-10, 10)), 2 / 2^10),
          wilcoxon_pratt_exact(rep(0.5 + 9e-11, 10) - 0.5) == 1,
          mann_whitney_exact(rep(0.50000000004, 10), rep(0.50000000006, 10)) == 1,
          same(mann_whitney_exact(rep(0, 10), rep(1.1e-10, 10)), 2 / choose(20, 10)),
          identical(figure4_marker(c(0.002, 0.3, NA)), c("*", "n.s.", NA_character_)))

# Scalar resampling must sample positions, not R's numeric 1:x shorthand.
stopifnot(same(bootstrap_ci(0, 2, paired = TRUE), c(2, 2)),
          same(bootstrap_ci(2, 3, paired = FALSE), c(1, 1)),
          same(bootstrap_ci(c(0, 1), c(0.1, 1.1), paired = TRUE), c(0.1, 0.1)),
          same(bootstrap_ci(c(0, 1), c(0.1, 1.1), paired = FALSE), c(-0.9, 1.1)))
for (paired in c(TRUE, FALSE)) {
  set.seed(9)
  before <- .Random.seed
  invisible(bootstrap_ci(c(-0.2, 0.1, 0.7), c(-0.1, 0.4, 0.8), paired))
  stopifnot(identical(.Random.seed, before))
  rm(".Random.seed", envir = globalenv())
  invisible(bootstrap_ci(c(-0.2, 0.1, 0.7), c(-0.1, 0.4, 0.8), paired))
  stopifnot(!exists(".Random.seed", envir = globalenv()))
}

# Only C1/C2 match seed sets. Sorting each group makes all CIs row-order invariant.
scenarios <- figure4_scenarios()$order
toy <- data.frame(
  Scenario = rep(scenarios, each = 3),
  seed = c(1, 2, 3, 2, 1, 3, 3, 1, 2, 1, 3, 2),
  score = c(0.1, 0.4, 0.8, 0.3, 0.2, 0.1, 0.5, 0.2, 0.9, 0.3, 0.1, 0.5)
)
toy_stats <- unit_statistics(toy, "score", "seed")
independent_seeds <- toy
federated <- independent_seeds$Scenario %in% scenarios[c(2, 4)]
independent_seeds$seed[federated] <- independent_seeds$seed[federated] + 100
stopifnot(same(unit_statistics(toy[nrow(toy):1, ], "score", "seed"), toy_stats),
          same(unit_statistics(independent_seeds, "score", "seed"), toy_stats),
          same(toy_stats$diff, c(0.1, 0.1, -7 / 30, -7 / 30)),
          identical(toy_stats$paired, c(TRUE, TRUE, FALSE, FALSE)),
          same(toy_stats$p_holm, stats::p.adjust(toy_stats$p_raw, "holm")))

# The descriptive cutoff includes both signs and does not gate adjusted-p markers.
threshold_data <- data.frame(
  Scenario = rep(scenarios, each = 10), seed = rep(1:10, 4),
  score = rep(c(0, 0.04, 0.05, -0.01), each = 10)
)
threshold_stats <- unit_statistics(threshold_data, "score", "seed")
stopifnot(identical(threshold_stats$relevant, c(TRUE, TRUE, FALSE, TRUE)),
          any(!threshold_stats$relevant & threshold_stats$p_holm < 0.05),
          identical(threshold_stats$marker, figure4_marker(threshold_stats$p_holm)))

# Seeds must uniquely identify runs.
expect_seed_error <- function(data) {
  error <- tryCatch(unit_statistics(data, "score", "seed"), error = identity)
  stopifnot(inherits(error, "error"), grepl("seed", conditionMessage(error), ignore.case = TRUE))
}
for (scenario in scenarios) {
  duplicate <- toy[toy$Scenario == scenario, ][1, ]
  expect_seed_error(rbind(toy, duplicate))
}
mismatched <- toy
mismatched$seed[mismatched$Scenario == scenarios[[3]] & mismatched$seed == 3] <- 4
expect_seed_error(mismatched)
missing_seed <- toy
missing_seed$seed[[1]] <- NA
expect_seed_error(missing_seed)
cat("Exact rank tests, numerical ties, paired/unpaired intervals and seed handling verified\n")

verified_outputs <- character()
for (arm in c("label_aware", "no_label")) {
  previous_hashes <- tools::md5sum(verified_outputs)
  output <- file.path(tempdir(), paste0("figure4_test_", arm, ".png"))
  Sys.setenv(CORRECTION_ARM = arm, FIGURE4_OUTPUT = output)
  script <- new.env()
  source(file.path("evaluation", "aggregate_figure4.R"), local = script)
  stopifnot(identical(script$output_prefix, "figure4_"))
  Sys.unsetenv(c("CORRECTION_ARM", "FIGURE4_OUTPUT"))
  statistics_file <- sub("\\.png$", "_statistics.tsv", output)
  annotated_file <- sub("\\.png$", "_statistics.png", output)
  cohort_files <- file.path(tempdir(), paste0(c("figureS7_", "figureS8_"), arm, ".png"))
  current_outputs <- c(output, annotated_file, statistics_file, cohort_files)
  stopifnot(all(file.exists(current_outputs)),
            !any(current_outputs %in% verified_outputs),
            identical(tools::md5sum(verified_outputs), previous_hashes))
  verified_outputs <- c(verified_outputs, current_outputs)

  summary <- script$figure4_stats
  data <- script$figure4_data
  loo <- data$leave_one_cohort_out
  split <- data$train_test_split
  cohort_seeds <- table(paste(loo$Task, loo$predicted_client_name), loo$Scenario)
  split_seeds <- table(paste(split$Task, split$predicted_client_name), split$Scenario)
  saved <- utils::read.delim(statistics_file, check.names = FALSE)
  inference_columns <- c("p_raw", "p_holm", "ci95_low", "ci95_high", "marker")
  family <- paste(summary$panel, summary$dataset,
                  ifelse(is.na(summary$cohort), "", summary$cohort), sep = "|")
  families <- split(seq_len(nrow(summary)), family)
  stopifnot(nrow(cohort_seeds) == 17L, all(cohort_seeds == 10L),
            nrow(split_seeds) == 17L, all(split_seeds == 10L),
            identical(as.vector(table(summary$panel)[c("4a ARI", "S7 LOCO", "S8 train/test")]),
                      c(16L, 68L, 68L)),
            all(summary$n_a == 10L), all(summary$n_b == 10L), nrow(saved) == 152L,
            identical(summary$paired, summary$contrast %in% c("C1", "C2")),
            !anyNA(summary[inference_columns]),
            identical(summary$marker, figure4_marker(summary$p_holm)),
            same(saved$p_raw, summary$p_raw), same(saved$p_holm, summary$p_holm),
            same(saved$ci95_low, summary$ci95_low), same(saved$ci95_high, summary$ci95_high),
            length(families) == 38L, all(lengths(families) == 4L),
            identical(saved$relevant, summary$relevant),
            identical(summary$relevant,
                      abs(summary$diff) >= figure4_min_difference - figure4_statistics_tolerance),
            nrow(figure4_annotation_data(summary, "4a ARI")) == 16L,
            nrow(figure4_annotation_data(summary, "S7 LOCO")) == 68L,
            nrow(figure4_annotation_data(summary, "S8 train/test")) == 68L)
  stopifnot(
    identical(names(script$figure4_annotations), "ari"),
    identical(script$figure4_annotations$ari$label,
              summary$marker[summary$panel == "4a ARI"]),
    length(script$figureS7_annotations) == 4L,
    length(script$figureS8_annotations) == 4L
  )
  for (figure in c("S7 LOCO", "S8 train/test")) {
    annotations <- if (figure == "S7 LOCO") script$figureS7_annotations else script$figureS8_annotations
    for (task in names(annotations)) {
      expected <- summary$marker[summary$panel == figure & summary$dataset == one_line(task)]
      stopifnot(identical(annotations[[task]]$label, expected))
    }
  }
  for (indices in families) {
    stopifnot(same(summary$p_holm[indices], stats::p.adjust(summary$p_raw[indices], "holm")))
  }

  # Match each output row to its actual plotted scores, including dataset/cohort keys.
  panels <- list(
    "4a ARI" = list(data = data$ari, label = "DatasetLabel", value = "ARI", seed = "Seed"),
    "S7 LOCO" = list(data = loo, label = "predicted_client_name",
                      value = "metric_value", seed = "seed"),
    "S8 train/test" = list(data = split, label = "predicted_client_name",
                           value = "metric_value", seed = "seed")
  )
  for (i in seq_len(nrow(summary))) {
    row <- summary[i, ]
    panel <- panels[[row$panel]]
    scores <- panel$data[as.character(panel$data[[panel$label]]) == row$x_level, , drop = FALSE]
    if (row$panel %in% c("S7 LOCO", "S8 train/test")) {
      scores <- scores[one_line(as.character(scores$Task)) == row$dataset, , drop = FALSE]
    }
    a <- scores[scores$Scenario == row$group_a, , drop = FALSE]
    b <- scores[scores$Scenario == row$group_b, , drop = FALSE]
    a <- a[order(a[[panel$seed]]), , drop = FALSE]
    b <- b[order(b[[panel$seed]]), , drop = FALSE]
    values_a <- a[[panel$value]]
    values_b <- b[[panel$value]]
    stopifnot(same(row$mean_a, mean(values_a)), same(row$mean_b, mean(values_b)),
              same(row$sd_a, stats::sd(values_a)), same(row$sd_b, stats::sd(values_b)),
              same(row$diff, mean(values_b) - mean(values_a)))
    if (row$paired) {
      b <- b[match(a[[panel$seed]], b[[panel$seed]]), , drop = FALSE]
      diff <- b[[panel$value]] - values_a
      p <- wilcoxon_pratt_exact(diff)
      bounds <- range(diff)
    } else {
      p <- mann_whitney_exact(values_a, values_b)
      bounds <- c(min(values_b) - max(values_a), max(values_b) - min(values_a))
    }
    stopifnot(same(row$p_raw, p), row$ci95_low >= bounds[[1]] - 1e-12,
              row$ci95_high <= bounds[[2]] + 1e-12, row$ci95_low <= row$ci95_high)
  }
  cat(arm, ": 152 rank-test contrasts in 38 Holm families and all figures verified\n")
}
stopifnot(length(verified_outputs) == 8L, !anyDuplicated(verified_outputs))
