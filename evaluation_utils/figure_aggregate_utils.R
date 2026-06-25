require_figure_packages <- function(required = c("tidyverse", "ggplot2", "grid", "gtable", "rlang", "dplyr", "tidyr", "tibble", "glue")) {
  missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing) > 0) {
    stop("Missing required R packages: ", paste(missing, collapse = ", "), call. = FALSE)
  }
}


require_figure4_packages <- function(required = c("ggplot2", "grid", "gtable", "rlang")) {
  require_figure_packages(required)
}


find_repo_root <- function(helper_path = file.path("evaluation_utils", "figure_aggregate_utils.R")) {
  candidates <- c(
    getwd(),
    file.path(getwd(), "..")
  )
  for (candidate in candidates) {
    normalized <- normalizePath(candidate, mustWork = FALSE)
    if (file.exists(file.path(normalized, helper_path))) {
      return(normalizePath(normalized, mustWork = TRUE))
    }
  }
  stop("Run this notebook from the repository root or from the evaluation/ directory.", call. = FALSE)
}


source_plotting_utils <- function(repo_root) {
  source(file.path(repo_root, "evaluation_utils", "plots_eda.R"))
  source(file.path(repo_root, "evaluation_utils", "evaluation_funcs.R"))
}


read_matrix_table <- function(path, zip_inner_file = NULL, use_fread = TRUE) {
  read_with_base <- function(table_path, inner_file = NULL) {
    if (file.exists(table_path)) {
      return(utils::read.delim(table_path, header = TRUE, row.names = 1, check.names = FALSE))
    }

    zip_candidates <- unique(c(
      paste0(table_path, ".zip"),
      sub("\\.[^.]+$", ".zip", table_path)
    ))
    zip_path <- zip_candidates[file.exists(zip_candidates)][1]
    if (!is.null(inner_file) && !is.na(zip_path)) {
      connection <- unz(zip_path, inner_file)
      return(utils::read.delim(connection, header = TRUE, row.names = 1, check.names = FALSE))
    }

    stop("Matrix file not found: ", table_path, call. = FALSE)
  }

  if (!isTRUE(use_fread) || !requireNamespace("data.table", quietly = TRUE)) {
    return(read_with_base(path, zip_inner_file))
  }

  if (file.exists(path)) {
    table <- data.table::fread(path, data.table = FALSE, check.names = FALSE)
  } else {
    zip_candidates <- unique(c(
      paste0(path, ".zip"),
      sub("\\.[^.]+$", ".zip", path)
    ))
    zip_path <- zip_candidates[file.exists(zip_candidates)][1]
    if (is.null(zip_inner_file) || is.na(zip_path) || !nzchar(Sys.which("unzip"))) {
      return(read_with_base(path, zip_inner_file))
    }
    table <- data.table::fread(
      cmd = paste("unzip -p", shQuote(zip_path), shQuote(zip_inner_file)),
      data.table = FALSE,
      check.names = FALSE
    )
  }

  if (ncol(table) < 2) {
    stop("Matrix table must contain row ids and at least one sample column: ", path, call. = FALSE)
  }
  row_ids <- table[[1]]
  table[[1]] <- NULL
  rownames(table) <- row_ids
  table
}


align_matrices_to_metadata <- function(uncorrected, central_corrected = NULL, fed_corrected, metadata, sample_col = "file") {
  if (!sample_col %in% names(metadata)) {
    metadata[[sample_col]] <- rownames(metadata)
  }
  sample_order <- rownames(metadata)
  matrices <- list(uncorrected = uncorrected)
  if (!is.null(central_corrected)) {
    matrices$central_corrected <- central_corrected
  }
  matrices$fed_corrected <- fed_corrected

  for (matrix_name in names(matrices)) {
    missing_samples <- setdiff(sample_order, colnames(matrices[[matrix_name]]))
    if (length(missing_samples) > 0) {
      stop(
        matrix_name,
        " matrix is missing metadata samples: ",
        paste(head(missing_samples), collapse = ", "),
        call. = FALSE
      )
    }
  }

  shared_features <- Reduce(intersect, lapply(matrices, rownames))
  if (length(shared_features) == 0) {
    stop("No shared features across Figure 3 matrices.", call. = FALSE)
  }

  aligned <- lapply(matrices, function(matrix_data) {
    matrix_data[shared_features, sample_order, drop = FALSE]
  })
  aligned$metadata <- metadata
  aligned
}


prepare_plot_matrices <- function(data, matrix_names = c("uncorrected", "fed_corrected")) {
  missing_matrices <- setdiff(matrix_names, names(data))
  if (length(missing_matrices) > 0) {
    stop("Missing matrix/matrices for plotting: ", paste(missing_matrices, collapse = ", "), call. = FALSE)
  }

  complete_features <- lapply(matrix_names, function(matrix_name) {
    rownames(data[[matrix_name]])[stats::complete.cases(data[[matrix_name]])]
  })
  names(complete_features) <- matrix_names
  reference_features <- complete_features[[1]]
  for (matrix_name in names(complete_features)[-1]) {
    if (!identical(reference_features, complete_features[[matrix_name]])) {
      stop(
        "Complete plotted feature sets differ between matrices; refusing to silently drop data for ",
        matrix_name,
        ".",
        call. = FALSE
      )
    }
  }
  if (length(reference_features) == 0) {
    stop("No complete features available for plotting.", call. = FALSE)
  }

  data$plot_matrices <- lapply(matrix_names, function(matrix_name) {
    data[[matrix_name]][reference_features, , drop = FALSE]
  })
  names(data$plot_matrices) <- matrix_names
  data
}


make_dataset_diagnostic_panel <- function(title, pca_before, pca_after, variance_before, variance_after) {
  list(
    title = title,
    plots = list(
      pca_before = pca_before,
      pca_after = pca_after,
      variance_before = variance_before,
      variance_after = variance_after
    )
  )
}


extract_plot_legend <- function(plot) {
  plot_grob <- ggplot2::ggplotGrob(plot)
  legend_indices <- which(vapply(plot_grob$grobs, function(grob) {
    grepl("^guide-box", grob$name) && !inherits(grob, "zeroGrob")
  }, logical(1)))

  if (length(legend_indices) == 0) {
    return(grid::zeroGrob())
  }

  plot_grob$grobs[[legend_indices[1]]]
}


legend_is_present <- function(legend) {
  !inherits(legend, "zeroGrob")
}


plot_legend_width <- function(legend) {
  if (!legend_is_present(legend)) {
    return(grid::unit(0, "cm"))
  }
  if (!is.null(legend$widths)) {
    return(sum(legend$widths))
  }
  grid::grobWidth(legend)
}


max_plot_legend_width <- function(legends, padding = grid::unit(0.12, "cm")) {
  present <- legends[vapply(legends, legend_is_present, logical(1))]
  if (length(present) == 0) {
    return(grid::unit(0, "cm"))
  }

  Reduce(grid::unit.pmax, lapply(present, plot_legend_width)) + padding
}


draw_dataset_diagnostic_panel <- function(
  panel,
  title_fontsize = 12,
  subplot_row_gap = grid::unit(0, "cm"),
  subplot_column_gap = grid::unit(0, "cm"),
  legend_gap = grid::unit(0, "cm")
) {
  legends <- list(
    pca = extract_plot_legend(panel$plots$pca_after),
    variance = extract_plot_legend(panel$plots$variance_after)
  )
  legend_width <- max_plot_legend_width(legends)

  layout <- grid::grid.layout(
    nrow = 4,
    ncol = 5,
    heights = grid::unit.c(
      grid::unit(0.58, "cm"),
      grid::unit(1, "null"),
      subplot_row_gap,
      grid::unit(1, "null")
    ),
    widths = grid::unit.c(
      grid::unit(1, "null"),
      subplot_column_gap,
      grid::unit(1, "null"),
      legend_gap,
      legend_width
    )
  )
  grid::pushViewport(grid::viewport(layout = layout))
  grid::pushViewport(grid::viewport(layout.pos.row = 1, layout.pos.col = 1:5))
  grid::grid.text(
    panel$title,
    x = grid::unit(0, "npc"),
    y = grid::unit(0.48, "npc"),
    just = c("left", "center"),
    gp = grid::gpar(fontface = "bold", fontsize = title_fontsize)
  )
  grid::popViewport()

  plot_positions <- list(
    pca_before = c(2, 1),
    pca_after = c(2, 3),
    variance_before = c(4, 1),
    variance_after = c(4, 3)
  )
  for (plot_name in names(plot_positions)) {
    position <- plot_positions[[plot_name]]
    grid::pushViewport(grid::viewport(layout.pos.row = position[[1]], layout.pos.col = position[[2]]))
    grid::grid.draw(ggplot2::ggplotGrob(panel$plots[[plot_name]] + ggplot2::theme(legend.position = "none")))
    grid::popViewport()
  }

  legend_positions <- list(
    pca = c(2, 5),
    variance = c(4, 5)
  )
  for (legend_name in names(legend_positions)) {
    if (legend_is_present(legends[[legend_name]])) {
      position <- legend_positions[[legend_name]]
      grid::pushViewport(grid::viewport(layout.pos.row = position[[1]], layout.pos.col = position[[2]]))
      grid::grid.draw(legends[[legend_name]])
      grid::popViewport()
    }
  }

  grid::popViewport()
  invisible(panel)
}


interleave_null_units <- function(values, gap) {
  values <- as.numeric(values)
  if (length(values) == 0) {
    return(grid::unit(0, "null"))
  }

  units <- grid::unit(values[[1]], "null")
  if (length(values) == 1) {
    return(units)
  }

  for (index in seq(2, length(values))) {
    units <- grid::unit.c(units, gap, grid::unit(values[[index]], "null"))
  }
  units
}


draw_stacked_panels <- function(panels, panel_heights = rep(1, length(panels))) {
  grid::grid.newpage()
  layout <- grid::grid.layout(
    nrow = length(panels),
    ncol = 1,
    heights = grid::unit(panel_heights, "null")
  )
  grid::pushViewport(grid::viewport(layout = layout))
  for (index in seq_along(panels)) {
    grid::pushViewport(grid::viewport(layout.pos.row = index, layout.pos.col = 1))
    draw_dataset_diagnostic_panel(panels[[index]])
    grid::popViewport()
  }
  grid::popViewport()
  invisible(panels)
}


arrange_dataset_panels <- function(
  panels,
  ncol = 2,
  panel_heights = NULL,
  panel_widths = NULL,
  title_fontsize = 8,
  dataset_row_gap = grid::unit(0, "cm"),
  dataset_column_gap = grid::unit(0, "cm"),
  subplot_row_gap = grid::unit(0, "cm"),
  subplot_column_gap = grid::unit(0, "cm"),
  legend_gap = grid::unit(0, "cm")
) {
  nrow <- ceiling(length(panels) / ncol)
  if (is.null(panel_heights)) {
    panel_heights <- rep(1, nrow)
  }
  if (is.null(panel_widths)) {
    panel_widths <- rep(1, ncol)
  }
  list(
    panels = panels,
    nrow = nrow,
    ncol = ncol,
    panel_heights = panel_heights,
    panel_widths = panel_widths,
    title_fontsize = title_fontsize,
    dataset_row_gap = dataset_row_gap,
    dataset_column_gap = dataset_column_gap,
    subplot_row_gap = subplot_row_gap,
    subplot_column_gap = subplot_column_gap,
    legend_gap = legend_gap
  )
}


draw_dataset_panel_layout <- function(layout) {
  grid::grid.newpage()
  row_heights <- interleave_null_units(layout$panel_heights, layout$dataset_row_gap)
  col_widths <- interleave_null_units(layout$panel_widths, layout$dataset_column_gap)
  grid_layout <- grid::grid.layout(
    nrow = layout$nrow * 2 - 1,
    ncol = layout$ncol * 2 - 1,
    heights = row_heights,
    widths = col_widths
  )
  grid::pushViewport(grid::viewport(layout = grid_layout))
  for (index in seq_along(layout$panels)) {
    row_index <- ((index - 1) %/% layout$ncol) + 1
    col_index <- ((index - 1) %% layout$ncol) + 1
    grid::pushViewport(grid::viewport(layout.pos.row = row_index * 2 - 1, layout.pos.col = col_index * 2 - 1))
    draw_dataset_diagnostic_panel(
      layout$panels[[index]],
      title_fontsize = layout$title_fontsize,
      subplot_row_gap = layout$subplot_row_gap,
      subplot_column_gap = layout$subplot_column_gap,
      legend_gap = layout$legend_gap
    )
    grid::popViewport()
  }
  grid::popViewport()
  invisible(layout)
}


save_dataset_panel_layout <- function(layout, output_path, width, height, dpi = 600) {
  dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
  png_args <- list(filename = output_path, width = width, height = height, units = "in", res = dpi)
  if (isTRUE(capabilities("cairo"))) {
    png_args$type <- "cairo"
  }
  do.call(grDevices::png, png_args)
  on.exit(grDevices::dev.off(), add = TRUE)
  draw_dataset_panel_layout(layout)
  normalizePath(output_path, mustWork = TRUE)
}


save_stacked_panels <- function(panels, output_path, width, height, dpi = 600, panel_heights = rep(1, length(panels))) {
  dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
  png_args <- list(filename = output_path, width = width, height = height, units = "in", res = dpi)
  if (isTRUE(capabilities("cairo"))) {
    png_args$type <- "cairo"
  }
  do.call(grDevices::png, png_args)
  on.exit(grDevices::dev.off(), add = TRUE)
  draw_stacked_panels(panels, panel_heights = panel_heights)
  normalizePath(output_path, mustWork = TRUE)
}


figure_should_display_inline <- function() {
  interactive() || "IRkernel" %in% loadedNamespaces() || nzchar(Sys.getenv("JPY_PARENT_PID"))
}


assert_columns <- function(data, required_columns, source_name) {
  missing_columns <- setdiff(required_columns, names(data))
  if (length(missing_columns) > 0) {
    stop(
      source_name,
      " is missing columns: ",
      paste(missing_columns, collapse = ", "),
      call. = FALSE
    )
  }
}


prepare_figure4a_plot_data <- function(
  ari,
  dataset_order,
  dataset_labels,
  method_labels,
  scenario_order,
  seeds_per_group
) {
  assert_columns(ari, c("Dataset", "Target", "Method", "Seed", "ARI"), "ARI source")

  missing_dataset_labels <- setdiff(dataset_order, names(dataset_labels))
  if (length(missing_dataset_labels) > 0) {
    stop(
      "Missing Figure 4a dataset labels for: ",
      paste(missing_dataset_labels, collapse = ", "),
      call. = FALSE
    )
  }

  plot_data <- ari[ari$Target == "condition" & ari$Dataset %in% dataset_order, , drop = FALSE]
  if (nrow(plot_data) == 0) {
    stop("No condition-target rows found for Figure 4a", call. = FALSE)
  }

  missing_datasets <- setdiff(dataset_order, unique(plot_data$Dataset))
  if (length(missing_datasets) > 0) {
    stop("Figure 4a is missing ARI datasets: ", paste(missing_datasets, collapse = ", "), call. = FALSE)
  }

  unexpected_methods <- setdiff(unique(plot_data$Method), names(method_labels))
  if (length(unexpected_methods) > 0) {
    stop("Unexpected ARI methods: ", paste(unexpected_methods, collapse = ", "), call. = FALSE)
  }

  seed_counts <- stats::aggregate(
    Seed ~ Dataset + Method,
    data = plot_data,
    FUN = function(x) length(unique(x))
  )
  incomplete <- seed_counts[seed_counts$Seed != seeds_per_group, , drop = FALSE]
  if (nrow(incomplete) > 0) {
    stop(
      "Expected ",
      seeds_per_group,
      " seeds per Figure 4a dataset/method group; incomplete groups: ",
      paste(apply(incomplete, 1, paste, collapse = "/"), collapse = ", "),
      call. = FALSE
    )
  }

  plot_data$DatasetLabel <- factor(
    unname(dataset_labels[plot_data$Dataset]),
    levels = unname(dataset_labels[dataset_order]),
    ordered = TRUE
  )
  plot_data$Scenario <- factor(
    unname(method_labels[plot_data$Method]),
    levels = scenario_order,
    ordered = TRUE
  )

  plot_data[order(plot_data$DatasetLabel, plot_data$Scenario, plot_data$Seed), , drop = FALSE]
}


map_classification_task_labels <- function(report, task_labels) {
  keys <- paste(report$data_name, report$predicted_target, sep = "|")
  unname(task_labels[keys])
}


map_classification_scenario_labels <- function(report, scenario_labels) {
  keys <- paste(
    tolower(as.character(report$learning_type)),
    tolower(as.character(report$data_preprocessing_name)),
    sep = "|"
  )
  unname(scenario_labels[keys])
}


prepare_figure4_classification_data <- function(
  report,
  cv_method,
  task_labels,
  task_order,
  scenario_labels,
  scenario_order,
  simulated_datasets,
  excluded_datasets,
  average_client_name
) {
  assert_columns(
    report,
    c(
      "data_name",
      "data_preprocessing_name",
      "metric_name",
      "metric_value",
      "predicted_client_name",
      "cross_validation_method",
      "seed",
      "predicted_target",
      "learning_type"
    ),
    "classification source"
  )

  subset <- report[
    report$metric_name == "MCC" & report$cross_validation_method == cv_method,
    ,
    drop = FALSE
  ]
  subset$Task <- map_classification_task_labels(subset, task_labels)
  subset$Scenario <- map_classification_scenario_labels(subset, scenario_labels)

  unexpected_scenarios <- unique(subset[is.na(subset$Scenario), c("learning_type", "data_preprocessing_name"), drop = FALSE])
  if (nrow(unexpected_scenarios) > 0) {
    stop(
      "Unexpected classification scenarios in Figure 4 source data: ",
      paste(apply(unexpected_scenarios, 1, paste, collapse = "/"), collapse = ", "),
      call. = FALSE
    )
  }

  unmapped_real <- subset[
    is.na(subset$Task) &
      !(subset$data_name %in% simulated_datasets) &
      !(subset$data_name %in% excluded_datasets),
    c("data_name", "predicted_target"),
    drop = FALSE
  ]
  unmapped_real <- unique(unmapped_real)
  if (nrow(unmapped_real) > 0) {
    stop(
      "Unmapped real classification tasks in Figure 4 source data: ",
      paste(apply(unmapped_real, 1, paste, collapse = "/"), collapse = ", "),
      call. = FALSE
    )
  }

  subset <- subset[!is.na(subset$Task) & !is.na(subset$Scenario), , drop = FALSE]
  federated_average <- tolower(subset$learning_type) == "federated" &
    subset$predicted_client_name == average_client_name
  subset <- subset[!federated_average, , drop = FALSE]

  dedup_columns <- c(
    "Task",
    "Scenario",
    "metric_name",
    "metric_value",
    "predicted_client_name",
    "cross_validation_method",
    "seed",
    "predicted_target"
  )
  subset <- subset[!duplicated(subset[dedup_columns]), , drop = FALSE]

  subset$Task <- factor(subset$Task, levels = task_order, ordered = TRUE)
  subset$Scenario <- factor(subset$Scenario, levels = scenario_order, ordered = TRUE)

  subset[order(subset$Task, subset$Scenario, subset$predicted_client_name, subset$seed), , drop = FALSE]
}


prepare_figure4_plot_data <- function(
  ari,
  classification_report,
  dataset_order,
  dataset_labels,
  method_labels,
  task_labels,
  task_order,
  scenario_labels,
  scenario_order,
  simulated_datasets,
  excluded_datasets,
  average_client_name,
  seeds_per_group
) {
  ari_data <- prepare_figure4a_plot_data(
    ari = ari,
    dataset_order = dataset_order,
    dataset_labels = dataset_labels,
    method_labels = method_labels,
    scenario_order = scenario_order,
    seeds_per_group = seeds_per_group
  )
  loo_data <- prepare_figure4_classification_data(
    report = classification_report,
    cv_method = "Leave-One-Cohort-Out",
    task_labels = task_labels,
    task_order = task_order,
    scenario_labels = scenario_labels,
    scenario_order = scenario_order,
    simulated_datasets = simulated_datasets,
    excluded_datasets = excluded_datasets,
    average_client_name = average_client_name
  )
  split_data <- prepare_figure4_classification_data(
    report = classification_report,
    cv_method = "Train-Test-Split",
    task_labels = task_labels,
    task_order = task_order,
    scenario_labels = scenario_labels,
    scenario_order = scenario_order,
    simulated_datasets = simulated_datasets,
    excluded_datasets = excluded_datasets,
    average_client_name = average_client_name
  )

  if (nrow(loo_data) == 0) {
    stop("No Figure 4b MCC rows after filtering", call. = FALSE)
  }
  if (nrow(split_data) == 0) {
    stop("No Figure 4c MCC rows after filtering", call. = FALSE)
  }

  list(
    ari = ari_data,
    leave_one_cohort_out = loo_data,
    train_test_split = split_data
  )
}


summarize_figure4_data <- function(ari, classification_report, figure4_data, ari_dataset_order) {
  list(
    ari_source_rows = nrow(ari),
    figure4a_rows = nrow(figure4_data$ari),
    figure4a_datasets = levels(droplevels(figure4_data$ari$DatasetLabel)),
    ari_datasets_outside_main_figure = setdiff(unique(ari$Dataset), ari_dataset_order),
    classification_source_rows = nrow(classification_report),
    figure4b_rows = nrow(figure4_data$leave_one_cohort_out),
    figure4c_rows = nrow(figure4_data$train_test_split),
    figure4_classification_tasks = levels(droplevels(figure4_data$train_test_split$Task))
  )
}


observed_levels <- function(values, full_order) {
  full_order[full_order %in% as.character(stats::na.omit(values))]
}


figure4_panel_theme <- function(base_size = 8.5) {
  ggplot2::theme_bw(base_size = base_size) +
    ggplot2::theme(
      panel.grid.major.x = ggplot2::element_blank(),
      panel.grid.minor = ggplot2::element_blank(),
      panel.grid.major.y = ggplot2::element_line(color = "#D7D7D7", linewidth = 0.25),
      panel.border = ggplot2::element_rect(color = "#444444", fill = NA, linewidth = 0.28),
      axis.title.x = ggplot2::element_blank(),
      axis.text.x = ggplot2::element_text(color = "#222222", margin = ggplot2::margin(t = 3)),
      axis.text.y = ggplot2::element_text(color = "#222222"),
      axis.title.y = ggplot2::element_text(color = "#222222", margin = ggplot2::margin(r = 4)),
      axis.ticks.x = ggplot2::element_blank(),
      plot.margin = ggplot2::margin(4, 4, 4, 4),
      plot.title = ggplot2::element_text(face = "bold", size = 11, hjust = 0),
      plot.title.position = "plot",
      legend.position = "bottom",
      legend.title = ggplot2::element_blank(),
      legend.key.width = grid::unit(0.72, "cm"),
      legend.key.height = grid::unit(0.28, "cm"),
      legend.text = ggplot2::element_text(size = 8)
    )
}


plot_figure4_box_panel <- function(
  data,
  x_col,
  y_col,
  x_levels,
  y_label,
  panel_tag,
  y_limits,
  y_breaks,
  scenario_order,
  scenario_colors,
  point_size,
  scenario_col = "Scenario"
) {
  box_cluster_width <- 0.68
  within_pair_gap <- 0.03
  before_after_gap <- 0.06
  point_jitter_width <- 0.03

  if (length(scenario_order) != 4) {
    stop("Figure 4 box layout expects four scenarios", call. = FALSE)
  }

  box_width <- (box_cluster_width - (2 * within_pair_gap + before_after_gap)) / length(scenario_order)
  scenario_centers <- c(
    0,
    box_width + within_pair_gap,
    2 * box_width + within_pair_gap + before_after_gap,
    3 * box_width + 2 * within_pair_gap + before_after_gap
  )
  scenario_offsets <- stats::setNames(
    scenario_centers - mean(range(scenario_centers)),
    scenario_order
  )
  x_axis_margin <- max(abs(scenario_offsets)) + box_width / 2 + 0.08

  plot_data <- data
  plot_data$Figure4XIndex <- match(as.character(plot_data[[x_col]]), x_levels)
  if (anyNA(plot_data$Figure4XIndex)) {
    unexpected_x <- unique(as.character(plot_data[[x_col]][is.na(plot_data$Figure4XIndex)]))
    stop("Figure 4 x values outside configured order: ", paste(unexpected_x, collapse = ", "), call. = FALSE)
  }

  plot_data$Figure4Scenario <- factor(
    as.character(plot_data[[scenario_col]]),
    levels = scenario_order,
    ordered = TRUE
  )
  plot_data$Figure4XOffset <- unname(scenario_offsets[as.character(plot_data$Figure4Scenario)])
  if (anyNA(plot_data$Figure4XOffset)) {
    unexpected_scenarios <- unique(as.character(plot_data[[scenario_col]][is.na(plot_data$Figure4XOffset)]))
    stop("Figure 4 scenarios outside configured order: ", paste(unexpected_scenarios, collapse = ", "), call. = FALSE)
  }

  plot_data$Figure4XPosition <- plot_data$Figure4XIndex + plot_data$Figure4XOffset
  plot_data$Figure4Group <- interaction(plot_data$Figure4XIndex, plot_data$Figure4Scenario, drop = TRUE)

  separator_positions <- if (length(x_levels) > 1) {
    seq(1.5, length(x_levels) - 0.5, by = 1)
  } else {
    numeric(0)
  }
  x_limits <- c(1 - x_axis_margin, length(x_levels) + x_axis_margin)

  ggplot2::ggplot(
    plot_data,
    ggplot2::aes(
      x = .data$Figure4XPosition,
      y = .data[[y_col]],
      fill = .data[[scenario_col]],
      color = .data[[scenario_col]],
      group = .data$Figure4Group
    )
  ) +
    ggplot2::geom_hline(yintercept = 0, color = "#666666", linewidth = 0.22) +
    ggplot2::geom_vline(
      xintercept = separator_positions,
      color = "#6F6F6F",
      linewidth = 0.22,
      alpha = 0.32
    ) +
    ggplot2::geom_boxplot(
      width = box_width,
      outlier.shape = NA,
      alpha = 0.72,
      linewidth = 0.28,
      color = "#333333",
      key_glyph = "rect"
    ) +
    ggplot2::geom_point(
      position = ggplot2::position_jitter(width = point_jitter_width, height = 0, seed = 17),
      shape = 21,
      size = point_size,
      alpha = 0.65,
      stroke = 0.14,
      color = "white",
      show.legend = FALSE
    ) +
    ggplot2::scale_x_continuous(
      breaks = seq_along(x_levels),
      labels = x_levels,
      limits = x_limits,
      expand = ggplot2::expansion(add = 0)
    ) +
    ggplot2::scale_y_continuous(breaks = y_breaks) +
    ggplot2::scale_fill_manual(
      values = scenario_colors,
      limits = scenario_order,
      drop = FALSE,
      guide = ggplot2::guide_legend(
        override.aes = list(alpha = 1, color = "#333333", linewidth = 0.28)
      )
    ) +
    ggplot2::scale_color_manual(values = scenario_colors, limits = scenario_order, guide = "none") +
    ggplot2::coord_cartesian(ylim = y_limits) +
    ggplot2::labs(y = y_label, title = panel_tag) +
    figure4_panel_theme()
}


make_figure4_plot <- function(
  figure4_data,
  dataset_labels,
  dataset_order,
  task_order,
  scenario_order,
  scenario_colors,
  point_size = 1.05,
  panel_heights = c(2.7, 2.7, 2.7, 0.55),
  panel_gap = grid::unit(0.18, "cm"),
  legend_gap = grid::unit(0.08, "cm")
) {
  ari_levels <- observed_levels(
    figure4_data$ari$DatasetLabel,
    unname(dataset_labels[dataset_order])
  )
  loo_levels <- observed_levels(figure4_data$leave_one_cohort_out$Task, task_order)
  split_levels <- observed_levels(figure4_data$train_test_split$Task, task_order)

  panel_a <- plot_figure4_box_panel(
    data = figure4_data$ari,
    x_col = "DatasetLabel",
    y_col = "ARI",
    x_levels = ari_levels,
    y_label = "ARI",
    panel_tag = "a) k-means clustering",
    y_limits = c(-0.1, 1.05),
    y_breaks = c(-0.1, 0, 0.25, 0.5, 0.75, 1),
    scenario_order = scenario_order,
    scenario_colors = scenario_colors,
    point_size = point_size
  )
  panel_b <- plot_figure4_box_panel(
    data = figure4_data$leave_one_cohort_out,
    x_col = "Task",
    y_col = "metric_value",
    x_levels = loo_levels,
    y_label = "MCC",
    panel_tag = "b) RF leave-one-cohort-out",
    y_limits = c(-0.25, 1.05),
    y_breaks = c(-0.2, 0, 0.25, 0.5, 0.75, 1),
    scenario_order = scenario_order,
    scenario_colors = scenario_colors,
    point_size = point_size
  )
  panel_c <- plot_figure4_box_panel(
    data = figure4_data$train_test_split,
    x_col = "Task",
    y_col = "metric_value",
    x_levels = split_levels,
    y_label = "MCC",
    panel_tag = "c) RF 70:30 train-test split",
    y_limits = c(-0.25, 1.05),
    y_breaks = c(-0.2, 0, 0.25, 0.5, 0.75, 1),
    scenario_order = scenario_order,
    scenario_colors = scenario_colors,
    point_size = point_size
  )

  list(
    panels = list(panel_a, panel_b, panel_c),
    panel_heights = panel_heights,
    panel_gap = panel_gap,
    legend_gap = legend_gap
  )
}


draw_figure4 <- function(figure_plot) {
  grid::grid.newpage()
  panels <- figure_plot$panels
  panel_heights <- figure_plot$panel_heights
  legend_height <- if (length(panel_heights) > length(panels)) {
    panel_heights[[length(panels) + 1]]
  } else {
    0.55
  }
  panel_heights <- panel_heights[seq_along(panels)]
  panel_gap <- if (is.null(figure_plot$panel_gap)) grid::unit(0.18, "cm") else figure_plot$panel_gap
  legend_gap <- if (is.null(figure_plot$legend_gap)) grid::unit(0.08, "cm") else figure_plot$legend_gap
  legend <- if (inherits(panels[[1]], "ggplot")) {
    extract_plot_legend(panels[[1]])
  } else if (!is.null(figure_plot$legend)) {
    figure_plot$legend
  } else {
    grid::zeroGrob()
  }
  has_legend <- legend_is_present(legend)

  panel_grobs <- lapply(panels, function(plot) {
    if (inherits(plot, "ggplot")) {
      ggplot2::ggplotGrob(plot + ggplot2::theme(legend.position = "none"))
    } else {
      plot
    }
  })
  layout_heights <- interleave_null_units(panel_heights, panel_gap)
  if (has_legend) {
    layout_heights <- grid::unit.c(
      layout_heights,
      legend_gap,
      grid::unit(legend_height, "null")
    )
  }
  layout_rows <- length(panels) * 2 - 1 + ifelse(has_legend, 2, 0)
  layout <- grid::grid.layout(
    nrow = layout_rows,
    ncol = 1,
    heights = layout_heights
  )
  grid::pushViewport(grid::viewport(layout = layout))
  for (i in seq_along(panel_grobs)) {
    grid::pushViewport(grid::viewport(layout.pos.row = i * 2 - 1, layout.pos.col = 1))
    grid::grid.draw(panel_grobs[[i]])
    grid::popViewport()
  }
  if (has_legend) {
    grid::pushViewport(grid::viewport(layout.pos.row = layout_rows, layout.pos.col = 1))
    grid::grid.draw(legend)
    grid::popViewport()
  }
  grid::popViewport()
  invisible(figure_plot)
}


save_figure4_png <- function(figure_plot, output_path, width, height, dpi = 600, output_format = NULL) {
  if (is.null(output_format)) {
    output_format <- tools::file_ext(output_path)
  }
  output_format <- tolower(output_format)
  if (!nzchar(output_format)) {
    output_format <- "png"
  }
  output_format <- match.arg(output_format, c("png", "svg"))

  current_ext <- tolower(tools::file_ext(output_path))
  if (!identical(current_ext, output_format)) {
    output_path <- paste0(tools::file_path_sans_ext(output_path), ".", output_format)
  }

  dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)

  if (identical(output_format, "png")) {
    png_args <- list(filename = output_path, width = width, height = height, units = "in", res = dpi)
    if (isTRUE(capabilities("cairo"))) {
      png_args$type <- "cairo"
    }
    do.call(grDevices::png, png_args)
  } else {
    grDevices::svg(filename = output_path, width = width, height = height)
  }

  on.exit(grDevices::dev.off(), add = TRUE)
  draw_figure4(figure_plot)

  normalizePath(output_path, mustWork = TRUE)
}


figure4_should_display_inline <- function() {
  figure_should_display_inline()
}
