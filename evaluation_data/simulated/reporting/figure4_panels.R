# Panels in the boxplot format of the paper's Figure 4, shared by build_sr2_boxplots.R,
# build_sr4_boxplots.R and build_sr5_dcombat_boxplots.R. Source
# evaluation_utils/figure_aggregate_utils.R first.

truth_label <- "Batch-free truth (median)"
style <- figure4_scenarios()
key <- c("scenario", "task", "learner", "lab", "state", "metric")

# Five held-out labs form one LOCO dataset mean; within-lab 70:30 averages the scores of
# the labs' test samples. Each source score already averages the same five seeds.
rf_dataset_means <- function(scores, labs) {
  x <- scores[scores$task == "rf" & scores$metric == "MCC" & scores$lab %in% labs, ]
  unit <- c("panel", "target", "run", "task", "learner", "state")
  stopifnot(nrow(x) > 0L, all(x$n_learner_seeds == 5L), all(is.finite(x$mean)),
            !anyDuplicated(x[c(unit, "lab")]))
  counts <- aggregate(x$lab, x[unit], length)
  if (any(counts$x != length(labs))) stop("Incomplete lab coverage in RF dataset means")
  aggregate(mean ~ panel + target + run + task + learner + state, x, mean)
}

# A row of panels: letter, title, dataset means, the column shown on x with its labels,
# and whether the labels are drawn.
panel_row <- function(letter, title, scores, x, labels, labelled = TRUE) {
  scores$x <- labels[as.character(scores[[x]])]
  list(letter = letter, title = title, scores = scores, labels = unname(labels),
       labelled = labelled)
}

# One panel with its plotted statistics: boxes of the states `before` and `after` for
# the central and the federated learner, and the median of `truth` as a bar across each
# corrected box. `runs` are the runs every box must hold (NULL: boxes of any size).
figure4_panel <- function(row, after, before = "uncorrected", truth = "truth",
                          title = row$title, runs = 1:10) {
  item <- row$scores[row$scores$state %in% c(before, after), ]
  item$Scenario <- style$order[1L + (item$learner == "federated") + 2L * (item$state == after)]
  plot <- plot_figure4_box_panel(
    data = item, x_col = "x", y_col = "mean", x_levels = row$labels,
    y_label = item$metric[1], panel_tag = title,
    y_limits = c(-0.25, 1.05), y_breaks = c(-0.2, 0, 0.25, 0.5, 0.75, 1),
    scenario_order = style$order, scenario_colors = style$colors, point_size = 0.9)
  built <- ggplot_build(plot)
  boxes <- built$data[[3]]
  points <- built$data[[4]]
  drawn <- plot$data
  groups <- lapply(boxes$group, function(group) drawn[as.integer(drawn$Figure4Group) == group, ])
  # Every box holds the dataset means of one x category, learner and state.
  stopifnot(!anyNA(item$x), nrow(boxes) == 4L * length(row$labels),
            vapply(seq_along(groups), function(i) {
              nrow(unique(groups[[i]][key])) == 1L &&
                (is.null(runs) || identical(sort(groups[[i]]$run), runs)) &&
                identical(sort(groups[[i]]$mean), sort(points$y[points$group == boxes$group[i]]))
            }, logical(1)))
  statistics <- c(ymin = "whisker_low", lower = "q1", middle = "median", upper = "q3",
                  ymax = "whisker_high")
  values <- do.call(rbind, lapply(seq_along(groups), function(i) {
    data.frame(groups[[i]][1, key], statistic = unname(statistics),
               value = unlist(boxes[i, names(statistics)]), row.names = NULL)
  }))
  reference <- row$scores[row$scores$state == truth, ]
  id <- intersect(c("scenario", "task", "learner", "lab", "run", "panel", "target"), names(item))
  expected <- item[item$state == after, , drop = FALSE]
  # SR4 has masked-truth bars only for keep and drop. SR5–SR6 use a keep-fit
  # reference for both complete routes.
  if (truth == "truth" && "route" %in% names(expected)) {
    expected <- expected[expected$route != "impute", ]
  }
  matched <- expected[id]
  stopifnot(!anyDuplicated(matched), !anyDuplicated(reference[id]),
            nrow(reference) == nrow(matched),
            nrow(merge(matched, reference[id], by = id)) == nrow(matched))
  if (nrow(reference)) {
    truth <- aggregate(reference$mean, reference[key], median)
    values <- rbind(values, data.frame(truth[key], statistic = "median", value = truth$x))
    # The truth bar of a learner lies across that learner's corrected box, below the
    # points so that datasets equal to truth stay visible.
    position <- unique(drawn[drawn$state == after,
                             c("scenario", "lab", "learner", "Figure4XPosition")])
    truth <- merge(truth, position)
    half <- (boxes$xmax[1] - boxes$xmin[1]) / 2 + 0.012
    bars <- geom_segment(data = truth, inherit.aes = FALSE, color = "#CC79A7", linewidth = 1,
                         alpha = 0.8,
                         aes(x = Figure4XPosition - half, xend = Figure4XPosition + half,
                             y = x, yend = x, linetype = truth_label))
    plot$layers <- append(plot$layers, bars, after = 3L)
    plot <- plot + scale_linetype_manual(values = setNames("solid", truth_label), name = NULL)
  }
  plot <- plot +
    theme(plot.title = element_text(size = 9), legend.key.width = grid::unit(0.45, "cm"),
          legend.text = element_text(size = 7.5),
          axis.text.x = if (row$labelled) element_text() else element_blank())
  list(plot = plot, values = values)
}


# Use the same vector graphics device and 300 dpi rasterization as the supplement.
save_result_png <- function(path, width, height, plot) {
  prefix <- tempfile(tmpdir = dirname(path))
  pdf <- paste0(prefix, ".pdf")
  png <- paste0(prefix, ".png")
  on.exit(unlink(c(pdf, png)))
  grDevices::cairo_pdf(pdf, width = width / 25.4, height = height / 25.4, family = "sans")
  tryCatch(force(plot), finally = grDevices::dev.off())
  status <- system2("pdftoppm", c("-png", "-singlefile", "-cropbox", "-r", "300",
                                  shQuote(pdf), shQuote(prefix)))
  if (status != 0L || !file.rename(png, path)) stop("Could not export ", path)
}

# Two columns of panels in reading order and one shared legend.
draw_columns <- function(panels) {
  grid::grid.newpage()
  n <- length(panels) %/% 2L
  heights <- grid::unit.c(rep(grid::unit(1, "null"), n), grid::unit(8, "mm"))
  grid::pushViewport(grid::viewport(layout = grid::grid.layout(n + 1L, 2, heights = heights)))
  for (i in seq_along(panels)) {
    grid::pushViewport(grid::viewport(layout.pos.row = (i + 1L) %/% 2L,
                                      layout.pos.col = 2L - i %% 2L))
    grid::grid.draw(ggplotGrob(panels[[i]]$plot + theme(legend.position = "none")))
    grid::popViewport()
  }
  # Boxes first and truth last, as in Figure SR2.
  boxes <- guide_legend(order = 1, override.aes = list(alpha = 1, color = "#333333",
                                                       linewidth = 0.28))
  grid::pushViewport(grid::viewport(layout.pos.row = n + 1L))
  grid::grid.draw(extract_plot_legend(
    panels[[1]]$plot + guides(fill = boxes, linetype = guide_legend(order = 2))))
  grid::popViewport(2)
}

# Plotted statistics at full precision, in a stable row order.
write_values <- function(values, path) {
  values$value <- format(values$value, digits = 17, trim = TRUE)
  write.table(values[do.call(order, values), ], path, sep = "\t", quote = FALSE,
              row.names = FALSE)
}
