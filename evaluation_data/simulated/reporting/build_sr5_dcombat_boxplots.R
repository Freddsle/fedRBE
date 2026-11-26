#!/usr/bin/env Rscript
# Figures SR5 and SR6 and the training-only d-ComBat comparison. Five-client RF after FedRBE
# or d-ComBat, corrected on all samples (SR5, SR6) or on the training samples of each split.

here <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE))))
repo <- normalizePath(file.path(here, "..", "..", ".."))
results <- file.path(repo, "evaluation_data/simulated/results/missingness_overlap")
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("Supply an output directory")
out <- args[1]
dir.create(out, recursive = TRUE, showWarnings = FALSE)
suppressPackageStartupMessages(library(ggplot2))
source(file.path(repo, "evaluation_utils/figure_aggregate_utils.R"))
source(file.path(here, "figure4_panels.R"))
grDevices::pdf(NULL)
truth_label <- "limmaRBE / FedRBE keep (median)"

levels <- c("0" = "0%", "0.1" = "10%", "0.3" = "30%", "0.5" = "50%")
panels <- c(overlap = "Partial overlap", full = "Full panels")
methods <- c(no_label = "limmaRBE / FedRBE", dcombat = "ComBat / d-ComBat")
# One figure per RF protocol, from the scores of its scoring population.
figures <- list(
  list(stem = "sr5_dcombat_learning", values = "sr5_plotted_values.tsv",
       table = "informative_dcombat_generated_run_means.tsv", labs = paste0("batch", 1:5),
       metric = "MCC, mean of five labs", lab = "held-out labs"),
  list(stem = "sr6_dcombat_learning_70_30", values = "sr6_plotted_values.tsv",
       table = "informative_dcombat_generated_run_means_within_lab_70_30.tsv", labs = paste0("batch", 1:5),
       metric = "MCC, mean of five labs", lab = "test samples of five labs"),
  list(stem = "dcombat_learning_training_only",
       values = "dcombat_training_only_plotted_values.tsv",
       table = "informative_training_only_generated_run_means.tsv", labs = paste0("batch", 1:5),
       metric = "MCC, mean of five labs", lab = "held-out labs"),
  list(stem = "dcombat_learning_70_30_training_only",
       values = "dcombat_70_30_training_only_plotted_values.tsv",
       table = "informative_training_only_generated_run_means_within_lab_70_30.tsv",
       labs = paste0("batch", 1:5), metric = "MCC, mean of five labs",
       lab = "test samples of five labs"))

for (figure in figures) {
  scores <- read.delim(file.path(results, figure$table), check.names = FALSE)
  scores <- rf_dataset_means(scores, figure$labs)
  stopifnot(all(table(scores$panel, scores$target, scores$state, scores$learner) == 10L))
  scores$metric <- figure$metric
  scores$lab <- figure$lab

  rows <- list()
  for (panel in names(panels)) for (route in c("drop", "impute")) for (method in names(methods)) {
    # The keep-route scores of the same datasets give each learner's reference bar.
    selected <- scores[scores$panel == panel &
                         scores$state %in% paste0(route, c("_uncorrected", paste0("_", method))) |
                         scores$panel == panel & scores$state == "no_label", ]
    selected$state[selected$state == "no_label"] <- "keep_no_label"
    selected$scenario <- paste(panel, route, selected$target)
    letter <- letters[length(rows) + 1L]
    title <- sprintf("%s) %s, %s route: %s", letter, panels[[panel]], route, methods[[method]])
    rows[[length(rows) + 1L]] <- list(
      row = panel_row(letter, title, selected, "target", levels),
      after = paste0(route, "_", method), before = paste0(route, "_uncorrected"))
  }
  drawn <- lapply(rows, function(item) {
    figure4_panel(item$row, after = item$after, before = item$before, truth = "keep_no_label")
  })

  width <- 180
  height <- 210
  stem <- file.path(out, figure$stem)
  save_result_png(paste0(stem, ".png"), width, height, draw_columns(drawn))
  values <- do.call(rbind, lapply(seq_along(drawn), function(i) {
    cbind(panel = rows[[i]]$row$letter, drawn[[i]]$values)
  }))
  write_values(values, file.path(out, figure$values))
}
