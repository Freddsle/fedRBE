#!/usr/bin/env Rscript
# Figure SR2. RF and k-means under confounding.

here <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE))))
repo <- normalizePath(file.path(here, "..", "..", ".."))
results <- file.path(repo, "evaluation_data/simulated/results/confounding_downstream")
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("Supply an output directory")
out <- args[1]
dir.create(out, recursive = TRUE, showWarnings = FALSE)
suppressPackageStartupMessages(library(ggplot2))
source(file.path(repo, "evaluation_utils/figure_aggregate_utils.R"))
source(file.path(here, "figure4_panels.R"))
grDevices::pdf(NULL)

designs <- c(confounded_p50 = "p = 0.50\nη² = 0",
             confounded_p30 = "p = 0.30\nη² = 0.16",
             confounded_p15 = "p = 0.15\nη² = 0.49",
             confounded_p05 = "p = 0.05\nη² = 0.80")
arms <- c(no_label = "No-label correction", label_aware = "Label-aware correction")
labs <- paste0("batch", 1:3)
dataset_means <- function(file, task, metric, lab) {
  scores <- read.delim(file.path(results, file), check.names = FALSE)
  scores <- scores[scores$task == task & scores$metric == metric & scores$lab %in% lab, ]
  stopifnot(all(scores$n_learner_seeds == 5L), all(is.finite(scores$mean)))
  scores
}
kmeans <- dataset_means("generated_run_means.tsv", "kmeans", "ARI", "pooled")
loco <- dataset_means("generated_run_means.tsv", "rf", "MCC", labs)
split <- dataset_means("generated_run_means_within_lab_70_30.tsv", "rf", "MCC", labs)
# A within-lab 70:30 point is a dataset's mean MCC over the test samples of its labs.
stopifnot(all(table(split[c("scenario", "run", "learner", "state")]) == length(labs)))
split <- aggregate(mean ~ scenario + run + task + learner + state + metric, split, mean)
split$lab <- "mean of labs"
# Held-out labs are labelled with their B samples among all their samples.
sizes <- read.delim(gzfile(file.path(results, "metrics.tsv.gz")), check.names = FALSE)
sizes <- unique(sizes[sizes$task == "rf" & sizes$lab %in% labs,
                      c("scenario", "lab", "n_samples", "n_b")])
sizes <- sizes[order(sizes$lab), ]

loco_title <- function(i, detail) {
  paste0(if (i == 1) "b) " else "", "RF leave-one-lab-out, ", detail)
}
by_design <- lapply(seq_along(designs), function(i) {
  current <- sizes[sizes$scenario == names(designs)[i], ]
  lab_labels <- setNames(sprintf("lab %s (%d/%d B)", sub("batch", "", current$lab),
                                 current$n_b, current$n_samples), current$lab)
  panel_row("b", loco_title(i, sub("\n", ", ", designs[[i]])),
            loco[loco$scenario == names(designs)[i], ], "lab", lab_labels)
})
first <- list(panel_row("a", "a) k-means clustering", kmeans, "scenario", designs))
last <- list(panel_row("c", "c) RF within-lab 70:30", split, "scenario", designs))
figure <- c(first, by_design, last)

# One panel of an arm. The uncorrected boxes recur in both arms.
panel <- function(row, arm) {
  built <- figure4_panel(row, after = arm,
                         title = if (arm == names(arms)[1]) row$title else " ")
  list(plot = built$plot, values = cbind(panel = row$letter, arm = arm, built$values))
}

# Arm headers, one stack of panels per arm with equal plotting heights, shared legend.
draw <- function(panels) {
  grid::grid.newpage()
  rows <- grid::unit.c(grid::unit(6, "mm"), grid::unit(1, "null"), grid::unit(8, "mm"))
  grid::pushViewport(grid::viewport(layout = grid::grid.layout(3, length(arms), heights = rows)))
  for (column in seq_along(arms)) {
    grid::pushViewport(grid::viewport(layout.pos.row = 1, layout.pos.col = column))
    grid::grid.text(arms[[column]], gp = grid::gpar(fontsize = 9.5, fontface = "bold"))
    grid::popViewport()
    grid::pushViewport(grid::viewport(layout.pos.row = 2, layout.pos.col = column))
    grid::grid.draw(do.call(rbind, lapply(panels[[column]], function(item) {
      ggplotGrob(item$plot + theme(legend.position = "none"))
    })))
    grid::popViewport()
  }
  grid::pushViewport(grid::viewport(layout.pos.row = 3))
  legend_plot <- panels[[1]][[1]]$plot
  # Keep the four correction states before the batch-free reference.
  fill_guide <- legend_plot$scales$get_scales("fill")$guide
  fill_guide$params$order <- 1L
  grid::grid.draw(extract_plot_legend(legend_plot + guides(fill = fill_guide)))
  grid::popViewport(2)
}

width <- 180
stem <- "sr2_confounding_learning"
panels <- lapply(names(arms), function(arm) lapply(figure, panel, arm = arm))
save_result_png(file.path(out, paste0(stem, ".png")), width, 214, draw(panels))
values <- do.call(rbind, lapply(panels, function(arm) {
  do.call(rbind, lapply(arm, `[[`, "values"))
}))
write_values(values, file.path(out, "sr2_plotted_values.tsv"))
