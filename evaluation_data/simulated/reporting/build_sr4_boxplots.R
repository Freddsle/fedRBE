#!/usr/bin/env Rscript
# Figure SR4. Five-client RF by missing-value route.

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
truth_label <- "Masked truth (median)"  # truth on the features and observed cells of the route

read <- function(name) read.delim(file.path(results, name), check.names = FALSE)
count <- function(x) format(round(x), big.mark = ",", trim = TRUE)

# A point is the mean MCC of a dataset over its five held-out labs (leave-one-lab-out,
# panels a-b) or over the test samples of its five labs (within-lab 70:30, panels c-d),
# with the correction fitted on the training samples: the three routes at 50%, and the
# drop route at 10-50% by the dataset's count of A/B features complete in all samples.
protocol_scores <- function(table, labs, metric, lab) {
  scores <- rf_dataset_means(read(table), labs)
  scores <- scores[scores$panel == "overlap" & !grepl("dcombat", scores$state), ]
  scores$panel <- NULL
  scores$metric <- metric
  scores$lab <- lab
  scores$route <- ifelse(grepl("^drop_", scores$state), "drop",
                         ifelse(grepl("^impute_", scores$state), "impute", "keep"))
  scores
}
protocols <- list(
  loco = protocol_scores("informative_training_only_generated_run_means.tsv",
                         paste0("batch", 1:5), "MCC, mean of five labs", "held-out labs"),
  split = protocol_scores("informative_training_only_generated_run_means_within_lab_70_30.tsv",
                          paste0("batch", 1:5), "MCC, mean of five labs",
                          "test samples of five labs"))
tables <- c(loco = "informative_training_only_metrics.tsv.gz",
            split = "informative_training_only_metrics_within_lab_70_30.tsv.gz")
by_route <- function(scores) {
  scores <- scores[scores$target == 0.5, ]
  scores$state <- sub("^(drop|impute)_", "", scores$state)
  scores$scenario <- paste0("informative_overlap_50_", scores$route)
  scores
}
# Features of a route at 50%: the mean over the training splits of the ten datasets, five
# held-out labs or five learner seeds each.
route_labels <- function(protocol) {
  rows <- read.delim(gzfile(file.path(results, tables[[protocol]])), check.names = FALSE)
  rows <- rows[rows$panel == "overlap" & rows$target == 0.5 & rows$lab != "pooled" &
                 !grepl("dcombat", rows$state), ]
  route <- ifelse(grepl("^drop_", rows$state), "drop",
                  ifelse(grepl("^impute_", rows$state), "impute", "keep"))
  means <- tapply(rows$n_features, route, mean)
  setNames(sprintf("%s\n%s features", names(means), count(means)), names(means))
}
features <- read("informative_learner_eligibility.tsv")
features <- features[features$panel == "overlap", ]
onoff <- features[features$role %in% c("A_DE", "B_DE"), ]
onoff <- aggregate(kmeans_features ~ root + target, onoff, sum)
names(onoff)[names(onoff) == "root"] <- "run"
by_count <- function(scores) {
  scores <- merge(scores[scores$route == "drop" & scores$target > 0, ], onoff)
  scores$group <- as.character(cut(scores$kmeans_features, c(-Inf, 0, 1, Inf),
                                   labels = c("none", "one", "two or more")))
  scores$scenario <- paste("informative_overlap", scores$group)
  scores
}
groups <- by_count(protocols$loco)
group_counts <- table(unique(groups[c("target", "run", "group")])$group)
group_labels <- setNames(sprintf("%s\n%d of %d", names(group_counts), group_counts,
                                sum(group_counts)), names(group_counts))
group_labels <- group_labels[c("none", "one", "two or more")]
# One row per RF protocol: routes at 50% (left), drop route by complete A/B features (right).
titles <- c(loco = "Leave-one-lab-out", split = "Within-lab 70:30")
rows <- list()
for (protocol in names(protocols)) {
  letter <- letters[length(rows) + 1:2]
  rows <- c(rows, list(
    list(row = panel_row(letter[1], sprintf("%s) %s RF by route at 50%%", letter[1],
                                            titles[[protocol]]),
                         by_route(protocols[[protocol]]), "route",
                         route_labels(protocol)[c("keep", "drop", "impute")]),
         subtitle = "Route labels show mean feature counts of the training splits",
         after = "no_label", before = "uncorrected", truth = "truth", runs = 1:10),
    list(row = panel_row(letter[2], sprintf("%s) %s RF, drop route", letter[2],
                                            titles[[protocol]]),
                         by_count(protocols[[protocol]]), "group", group_labels),
         subtitle = "Complete A/B features, 10 datasets × 3 levels (10, 30, 50%)",
         after = "drop_no_label", before = "drop_uncorrected", truth = "drop_truth",
         runs = NULL)))
}

panels <- lapply(rows, function(item) {
  panel <- figure4_panel(item$row, after = item$after, before = item$before,
                         truth = item$truth, runs = item$runs)
  panel$plot <- panel$plot + labs(subtitle = item$subtitle)
  panel
})

width <- 180
height <- 172
stem <- file.path(out, "sr4_five_client_learning")
save_result_png(paste0(stem, ".png"), width, height, draw_columns(panels))
values <- do.call(rbind, lapply(seq_along(panels), function(i) {
  cbind(panel = rows[[i]]$row$letter, panels[[i]]$values)
}))
write_values(values, file.path(out, "sr4_plotted_values.tsv"))
