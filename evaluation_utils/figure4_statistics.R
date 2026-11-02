# Exact rank comparisons for Figure 4 (ARI and train/test MCC) and Figure S7
# (leave-one-cohort-out MCC per held-out cohort).
# C1/C2 pair correction states by run seed; their Pratt signed-rank null assumes
# symmetric paired differences. C3/C4 use unpaired Mann-Whitney tests, assuming
# independent run groups and exchangeable scores under their null. Different
# centralized/federated split schemes remain part of the pipeline comparison.
# Inference concerns run variation on fixed data, not new patients or cohorts.
# Requires figure_aggregate_utils.R for figure4_scenarios().

# Numerical zeros use this tolerance; rank ties use ten-decimal rounding.
# Effect estimates and bootstrap intervals retain source precision.
figure4_statistics_tolerance <- 1e-10

# Descriptive absolute mean-difference cutoff; independent of p-values and stars.
figure4_min_difference <- 0.05


figure4_marker <- function(p) {
  ifelse(p < 0.05, "*", "n.s.")
}


# C1/C2: correction effect per learning setup. C3/C4: federated vs central.
# `level` is the bracket height in the plots (1 is lowest).
figure4_contrasts <- function() {
  scenarios <- figure4_scenarios()$order
  data.frame(
    contrast = c("C1", "C2", "C3", "C4"),
    group_a = scenarios[c(1, 2, 1, 3)],
    group_b = scenarios[c(3, 4, 2, 4)],
    paired = c(TRUE, TRUE, FALSE, FALSE),
    level = c(2L, 3L, 1L, 1L),
    stringsAsFactors = FALSE
  )
}


# Exact Wilcoxon signed-rank test with Pratt's method: zero differences take part in
# the ranking, but only the ranks of non-zero differences are summed. Rounding keeps
# floating-point noise from breaking ties. coin cannot test all-zero differences.
wilcoxon_pratt_exact <- function(diff) {
  diff[abs(diff) <= figure4_statistics_tolerance] <- 0
  diff <- round(diff, 10)
  if (all(diff == 0)) return(1)
  as.numeric(coin::pvalue(coin::wilcoxsign_test(
    diff ~ numeric(length(diff)), zero.method = "Pratt", distribution = "exact"
  )))
}


# Exact Mann-Whitney U test on the pooled midranks. coin cannot test identical values.
mann_whitney_exact <- function(a, b) {
  if (max(c(a, b)) - min(c(a, b)) <= figure4_statistics_tolerance) return(1)
  values <- round(c(a, b), 10)
  if (length(unique(values)) == 1) return(1)
  group <- factor(rep(c("a", "b"), c(length(a), length(b))))
  as.numeric(coin::pvalue(coin::wilcox_test(values ~ group, distribution = "exact")))
}


# Percentile bootstrap CI of mean(b) - mean(a). Paired data resample the pairs.
# The global RNG state is left unchanged.
bootstrap_ci <- function(a, b, paired, level = 0.95, resamples = 10000, seed = 1) {
  if (exists(".Random.seed", envir = globalenv())) {
    old_seed <- get(".Random.seed", envir = globalenv())
    on.exit(assign(".Random.seed", old_seed, envir = globalenv()), add = TRUE)
  } else {
    on.exit(rm(".Random.seed", envir = globalenv()), add = TRUE)
  }
  set.seed(seed)
  resample_means <- function(x) {
    indices <- sample.int(length(x), length(x) * resamples, replace = TRUE)
    colMeans(matrix(x[indices], nrow = length(x)))
  }
  estimates <- if (paired) resample_means(b - a) else resample_means(b) - resample_means(a)
  alpha <- 1 - level
  unname(stats::quantile(estimates, c(alpha / 2, 1 - alpha / 2)))
}


is_constant <- function(x) {
  max(x) - min(x) <= figure4_statistics_tolerance
}


contrast_note <- function(a, b, paired) {
  if (paired && all(abs(b - a) <= figure4_statistics_tolerance)) return("all pairs identical")
  if (is_constant(a) && is_constant(b)) return("both groups constant")
  if (is_constant(a) || is_constant(b)) return("one group constant")
  ""
}


# One score per scenario/seed within a dataset or held-out cohort.
# All four contrasts in this unit and arm form one Holm family.
unit_statistics <- function(data, value_col, seed_col) {
  contrasts <- figure4_contrasts()
  rows <- lapply(seq_len(nrow(contrasts)), function(i) {
    contrast <- contrasts[i, ]
    a <- data[data$Scenario == contrast$group_a, , drop = FALSE]
    b <- data[data$Scenario == contrast$group_b, , drop = FALSE]
    if (!nrow(a) || !nrow(b)) {
      stop("Missing scenario rows for ", contrast$contrast, call. = FALSE)
    }
    if (anyNA(c(a[[seed_col]], b[[seed_col]])) ||
        anyDuplicated(a[[seed_col]]) || anyDuplicated(b[[seed_col]])) {
      stop("Each scenario needs unique, nonmissing run seeds", call. = FALSE)
    }
    if (contrast$paired && !setequal(a[[seed_col]], b[[seed_col]])) {
      stop("Paired contrast ", contrast$contrast, " needs the same unique seeds in both groups",
           call. = FALSE)
    }
    # Fixed within-group order makes finite bootstrap CIs invariant to row shuffles.
    a <- a[order(a[[seed_col]]), , drop = FALSE]
    b <- b[order(b[[seed_col]]), , drop = FALSE]
    if (contrast$paired) b <- b[match(a[[seed_col]], b[[seed_col]]), , drop = FALSE]
    values_a <- a[[value_col]]
    values_b <- b[[value_col]]
    ci <- bootstrap_ci(values_a, values_b, contrast$paired)
    p <- if (contrast$paired) {
      wilcoxon_pratt_exact(values_b - values_a)
    } else {
      mann_whitney_exact(values_a, values_b)
    }
    diff <- mean(values_b) - mean(values_a)
    data.frame(
      contrast = contrast$contrast,
      group_a = contrast$group_a, group_b = contrast$group_b, paired = contrast$paired,
      n_a = length(values_a), n_b = length(values_b),
      mean_a = mean(values_a), mean_b = mean(values_b),
      sd_a = stats::sd(values_a), sd_b = stats::sd(values_b),
      diff = diff, ci95_low = ci[[1]], ci95_high = ci[[2]], p_raw = p,
      relevant = abs(diff) >= figure4_min_difference - figure4_statistics_tolerance,
      note = contrast_note(values_a, values_b, contrast$paired),
      stringsAsFactors = FALSE
    )
  })
  stats <- do.call(rbind, rows)
  stats$p_holm <- stats::p.adjust(stats$p_raw, method = "holm")
  stats$marker <- figure4_marker(stats$p_holm)
  stats
}


# `units` gives one row per tested unit: its plot x level, dataset label and cohort.
panel_statistics <- function(data, units, unit_rows, value_col, seed_col, figure, panel, unit) {
  rows <- lapply(seq_len(nrow(units)), function(i) {
    stats <- unit_statistics(data[unit_rows(data, units[i, ]), , drop = FALSE],
                             value_col, seed_col)
    data.frame(figure = figure, panel = panel, dataset = units$dataset[[i]],
               cohort = units$cohort[[i]], x_level = units$x_level[[i]], unit = unit,
               stats, stringsAsFactors = FALSE)
  })
  do.call(rbind, rows)
}


one_line <- function(label) gsub("\n", " ", label, fixed = TRUE)


# One row per unit and contrast. `x_level` is the plot's x label used for brackets;
# write_figure4_statistics() drops it from the saved table.
compute_figure4_statistics <- function(figure4_data, arm) {
  ari <- figure4_data$ari
  ari_levels <- levels(droplevels(ari$DatasetLabel))
  split <- figure4_data$train_test_split
  split_levels <- levels(droplevels(split$Task))
  loo <- figure4_data$leave_one_cohort_out
  loo_units <- unique(data.frame(task = as.character(loo$Task),
                                 cohort = loo$predicted_client_name,
                                 stringsAsFactors = FALSE))
  loo_units <- loo_units[order(match(loo_units$task, levels(droplevels(loo$Task)))), ]

  by_dataset <- function(levels) {
    data.frame(x_level = levels, dataset = one_line(levels), cohort = NA_character_,
               stringsAsFactors = FALSE)
  }
  stats <- rbind(
    panel_statistics(
      ari, by_dataset(ari_levels),
      function(data, unit) as.character(data$DatasetLabel) == unit$x_level,
      "ARI", "Seed", "figure4", "4a ARI", "k-means run (seed)"
    ),
    panel_statistics(
      split, by_dataset(split_levels),
      function(data, unit) as.character(data$Task) == unit$x_level,
      "metric_value", "seed", "figure4", "4c train/test", "RF run (seed), pooled 70:30 test set"
    ),
    panel_statistics(
      loo,
      data.frame(x_level = loo_units$cohort, dataset = one_line(loo_units$task),
                 cohort = loo_units$cohort, task = loo_units$task, stringsAsFactors = FALSE),
      function(data, unit) {
        as.character(data$Task) == unit$task & data$predicted_client_name == unit$cohort
      },
      "metric_value", "seed", "figureS7", "S7 LOCO", "RF run (seed), held-out cohort"
    )
  )
  rownames(stats) <- NULL
  data.frame(arm = arm, stats, stringsAsFactors = FALSE)
}


write_figure4_statistics <- function(stats, output_path) {
  dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
  table <- stats[, setdiff(names(stats), "x_level"), drop = FALSE]
  utils::write.table(table, output_path, sep = "\t", quote = FALSE, row.names = FALSE, na = "")
  normalizePath(output_path, mustWork = TRUE)
}


# Significant contrasts whose mean difference is below figure4_min_difference.
figure4_negligible_effects <- function(stats) {
  columns <- c("panel", "dataset", "cohort", "contrast", "diff", "p_raw", "p_holm")
  stats[stats$p_holm < 0.05 & !stats$relevant, columns, drop = FALSE]
}


# Brackets for one panel (and, for S7, one dataset), labelled with the table's marker.
figure4_annotation_data <- function(stats, panel, dataset = NULL) {
  rows <- stats[stats$panel == panel, , drop = FALSE]
  if (!is.null(dataset)) rows <- rows[rows$dataset == dataset, , drop = FALSE]
  if (!nrow(rows)) stop("No statistics for panel ", panel, call. = FALSE)
  levels <- figure4_contrasts()$level[match(rows$contrast, figure4_contrasts()$contrast)]
  data.frame(
    x_level = rows$x_level, contrast = rows$contrast, level = levels,
    group_a = rows$group_a, group_b = rows$group_b,
    label = rows$marker,
    stringsAsFactors = FALSE
  )
}


figure4_statistics_caption <- function() {
  paste("Exact paired Wilcoxon signed-rank (Pratt) and unpaired Mann-Whitney tests.",
        "Holm per dataset/cohort (four contrasts): * p < 0.05, n.s. otherwise; inference on fixed data.",
        sep = "\n")
}
