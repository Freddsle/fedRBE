require_figure_packages <- function(required = c("tidyverse", "ggplot2", "grid", "gtable", "rlang", "dplyr", "tidyr", "tibble", "glue")) {
  missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing) > 0) {
    stop("Missing required R packages: ", paste(missing, collapse = ", "), call. = FALSE)
  }
}


require_figure4_packages <- function(required = c("ggplot2", "grid", "gtable", "rlang")) {
  require_figure_packages(required)
}


# Paths in this file are relative to the repository root, the working directory
# of the evaluation scripts.
source_plotting_utils <- function() {
  source(file.path("evaluation_utils", "plots_eda.R"))
  source(file.path("evaluation_utils", "evaluation_funcs.R"))
}


correction_arm_titles <- c(label_aware = "Label-aware correction", no_label = "Label-free correction")


# CORRECTION_ARM selects the corrected outputs under after/<arm>/.
correction_arm <- function() {
  match.arg(Sys.getenv("CORRECTION_ARM", "label_aware"), names(correction_arm_titles))
}


read_matrix_table <- function(path, zip_inner_file = NULL) {
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

  if (!requireNamespace("data.table", quietly = TRUE)) {
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


align_matrices_to_metadata <- function(uncorrected, fed_corrected, metadata) {
  sample_order <- rownames(metadata)
  matrices <- list(uncorrected = uncorrected, fed_corrected = fed_corrected)

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


# Dataset loaders return the aligned uncorrected and fedRBE matrices of one
# correction arm, with sample metadata.
load_ecoli_data <- function(correction_arm) {
  data_path <- file.path("evaluation_data", "ecoli")
  after_path <- file.path(data_path, "after", correction_arm)
  metadata <- utils::read.delim(
    file.path(data_path, "before", "initial_data", "central_batch_info.tsv"),
    header = TRUE, row.names = 1, check.names = FALSE
  )
  metadata$file <- rownames(metadata)

  align_matrices_to_metadata(
    uncorrected = read_matrix_table(file.path(data_path, "before", "central_intensities_log_UNION.tsv")),
    fed_corrected = read_matrix_table(file.path(after_path, "FedApp_corrected_data.tsv")),
    metadata = metadata
  )
}


load_ovarian_data <- function(correction_arm) {
  data_path <- file.path("evaluation_data", "ovarian_cancer")
  after_path <- file.path(data_path, "after", correction_arm)
  metadata <- utils::read.delim(
    file.path(data_path, "before", "all_metadata.tsv"),
    header = TRUE, row.names = 1, check.names = FALSE
  )
  rownames(metadata) <- gsub("^X", "", rownames(metadata))
  metadata$file <- rownames(metadata)

  align_matrices_to_metadata(
    uncorrected = read_matrix_table(
      file.path(data_path, "before", "central_input.tsv"),
      zip_inner_file = "central_input.tsv"
    ),
    fed_corrected = read_matrix_table(file.path(after_path, "FedApp_corrected_data.tsv")),
    metadata = metadata
  )
}


load_ccrcc_data <- function(correction_arm) {
  data_path <- file.path("evaluation_data", "ccRCC_studies")
  after_path <- file.path(data_path, "after", correction_arm)
  metadata <- do.call(rbind, lapply(c("PDC000127", "PXD030344", "PXD042844"), function(site) {
    design <- utils::read.delim(
      file.path(data_path, "before", site, "design.tsv"), check.names = FALSE
    )
    data.frame(
      file = design$Sample,
      condition = ifelse(design$Tumor == 1, "Tumor", "Normal"),
      lab = site,
      stringsAsFactors = FALSE
    )
  }))
  rownames(metadata) <- metadata$file

  align_matrices_to_metadata(
    uncorrected = read_matrix_table(file.path(data_path, "before", "central_intensities_log_UNION.tsv")),
    fed_corrected = read_matrix_table(file.path(after_path, "FedApp_corrected_data.tsv")),
    metadata = metadata
  )
}


quartet_donor_levels <- c("D5", "D6", "F7", "M8")
quartet_batch_levels <- c("B01", "B02", "B03", "B04")

quartet_condition_palette <- c(D5 = "#44abe7", D6 = "#c55702", F7 = "#009E73", M8 = "#F0E442")


# Joint all-modality k-means matrices of the Quartet evaluation.
load_quartet_data <- function(correction_arm) {
  data_path <- file.path("evaluation_data", "quartet_multiomics")
  after_path <- file.path(data_path, "after", correction_arm)
  metadata <- utils::read.delim(
    file.path(data_path, "all_modalities_metadata.tsv"),
    header = TRUE, check.names = FALSE
  )
  metadata$file <- metadata$pseudo_sample
  metadata$lab <- metadata$client
  # The final pseudo-sample component identifies the batch within each client.
  metadata$batch <- factor(paste0(metadata$client, "_B", sub(".*_", "", metadata$pseudo_sample)))
  metadata$batch_code <- factor(metadata$batch, labels = paste0("B", seq_len(nlevels(metadata$batch))))
  metadata$condition <- factor(metadata$condition, levels = quartet_donor_levels)
  metadata$lab <- factor(metadata$lab)
  rownames(metadata) <- metadata$file

  align_matrices_to_metadata(
    uncorrected = read_matrix_table(file.path(data_path, "before", "all_modalities_before_kmeans_matrix.tsv")),
    fed_corrected = read_matrix_table(file.path(after_path, "all_modalities_fedapp_kmeans_matrix.tsv")),
    metadata = metadata
  )
}


load_quartet_modality_data <- function(modality, correction_arm) {
  modality_before <- file.path("evaluation_data", "quartet_multiomics", "before", modality)
  modality_after <- file.path("evaluation_data", "quartet_multiomics", "after", correction_arm, modality)

  metadata <- utils::read.delim(
    file.path(modality_before, "metadata.tsv"),
    header = TRUE,
    check.names = FALSE
  )
  required_columns <- c("file", "condition", "batch", "batch_code")
  assert_columns(metadata, required_columns, paste(modality, "metadata"))
  if (anyNA(metadata[required_columns]) || anyDuplicated(metadata$file) ||
      any(!metadata$condition %in% quartet_donor_levels) ||
      any(!metadata$batch_code %in% quartet_batch_levels)) {
    stop("Missing, duplicated or unexpected Quartet metadata for ", modality, call. = FALSE)
  }
  batch_codes <- unique(metadata[c("batch", "batch_code")])
  if (anyDuplicated(batch_codes$batch) || anyDuplicated(batch_codes$batch_code)) {
    stop("Each batch must have exactly one short batch code for ", modality, call. = FALSE)
  }
  metadata$file <- as.character(metadata$file)
  metadata$condition <- factor(metadata$condition, levels = quartet_donor_levels)
  metadata$batch <- factor(metadata$batch)
  metadata$batch_code <- factor(metadata$batch_code, levels = quartet_batch_levels)
  rownames(metadata) <- metadata$file

  align_matrices_to_metadata(
    uncorrected = read_matrix_table(file.path(modality_before, "central_intensities_log_UNION.tsv")),
    fed_corrected = read_matrix_table(file.path(modality_after, "FedApp_corrected_data.tsv")),
    metadata = metadata
  )
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


# Writes the grid drawing of draw() to a PNG, or an SVG for a .svg path; width and
# height are in inches. A path without extension is written as .png.
save_grid_png <- function(draw, output_path, width, height, dpi = 600) {
  output_format <- tolower(tools::file_ext(output_path))
  if (!nzchar(output_format)) {
    output_format <- "png"
    output_path <- paste0(output_path, ".png")
  }
  output_format <- match.arg(output_format, c("png", "svg"))

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
  draw()
  normalizePath(output_path, mustWork = TRUE)
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
  identity_columns <- c("Dataset", "Target", "Method", "Seed")
  if (anyNA(ari[identity_columns])) {
    stop("Missing ARI experiment keys", call. = FALSE)
  }

  plot_data <- ari[ari$Target == "condition" & ari$Dataset %in% dataset_order, , drop = FALSE]
  if (nrow(plot_data) == 0) {
    stop("No condition-target rows found for Figure 4a", call. = FALSE)
  }
  if (anyDuplicated(plot_data[identity_columns])) {
    stop("Duplicate ARI experiment keys", call. = FALSE)
  }
  plot_data$ARI <- suppressWarnings(as.numeric(plot_data$ARI))
  if (any(!is.finite(plot_data$ARI) | plot_data$ARI < -1 | plot_data$ARI > 1)) {
    stop("ARI scores must be finite and between -1 and 1", call. = FALSE)
  }

  missing_datasets <- setdiff(dataset_order, unique(plot_data$Dataset))
  if (length(missing_datasets) > 0) {
    stop("Figure 4a is missing ARI datasets: ", paste(missing_datasets, collapse = ", "), call. = FALSE)
  }

  unexpected_methods <- setdiff(unique(plot_data$Method), names(method_labels))
  if (length(unexpected_methods) > 0) {
    stop("Unexpected ARI methods: ", paste(unexpected_methods, collapse = ", "), call. = FALSE)
  }

  seed_counts <- table(
    factor(plot_data$Dataset, levels = dataset_order),
    factor(plot_data$Method, levels = names(method_labels))
  )
  if (any(seed_counts != seeds_per_group)) {
    stop(
      "Expected ", seeds_per_group, " seeds in every Figure 4a dataset/method group",
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
  average_client_name,
  metric_name = "MCC"
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

  identity_columns <- c(
    "data_name", "data_preprocessing_name", "metric_name", "predicted_client_name",
    "cross_validation_method", "seed", "predicted_target", "learning_type"
  )
  if (anyNA(report[identity_columns])) {
    stop("Missing classification experiment keys", call. = FALSE)
  }
  subset <- report[
    report$metric_name == metric_name & report$cross_validation_method == cv_method,
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
  if (anyDuplicated(subset[identity_columns])) {
    stop("Duplicate classification experiment keys", call. = FALSE)
  }
  subset$metric_value <- suppressWarnings(as.numeric(subset$metric_value))
  lower_bound <- if (metric_name == "MCC") -1 else 0
  if (any(!is.finite(subset$metric_value) |
          subset$metric_value < lower_bound | subset$metric_value > 1)) {
    stop("Classification scores must be finite and within the metric's range",
         call. = FALSE)
  }
  # Both learning methods contribute one pooled score per seed for train/test.
  # Their sample splits still differ. LOCO contributes individual held-out cohorts.
  pooled <- subset$predicted_client_name == average_client_name
  subset <- subset[if (cv_method == "Train-Test-Split") pooled else !pooled, , drop = FALSE]

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
      plot.subtitle = ggplot2::element_text(size = 8.5, color = "#444444",
                                           margin = ggplot2::margin(b = 3)),
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

  plot_data$Figure4Scenario <- factor(
    as.character(plot_data[[scenario_col]]),
    levels = scenario_order,
    ordered = TRUE
  )
  plot_data$Figure4XOffset <- unname(scenario_offsets[as.character(plot_data$Figure4Scenario)])

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


figure4_scenarios <- function() {
  colors <- c(
    "Central before BEC" = "#9E9E9E", "Federated before BEC" = "#D55E00",
    "Central after BEC" = "#0072B2", "Federated after BEC" = "#009E73"
  )
  list(order = names(colors), colors = colors)
}


# RF tasks as data_name|predicted_target, in the dataset order of Figure 4a.
classification_task_labels <- c(
  "E. coli|Pyr" = "E. coli\n(Pyr)",
  "Ovarian cancer|HGSC" = "Ovarian cancer\n(HGSC)",
  "ccRCC|Condition" = "ccRCC\n(condition)",
  "Quartet Multiomics|condition" = "Quartet multiomics\n(condition)"
)


# Maps learning_type|data_preprocessing_name to the four Figure 4 scenarios.
classification_scenario_labels <- function(corrected_name = "corrected") {
  keys <- paste(
    rep(c("centralized", "federated"), 2),
    rep(c("uncorrected", corrected_name), each = 2), sep = "|"
  )
  stats::setNames(figure4_scenarios()$order, keys)
}


export_classification_plots <- function(results_file, plots_dir) {
  require_figure4_packages()
  report <- utils::read.csv(results_file, stringsAsFactors = FALSE, check.names = FALSE)
  task_labels <- classification_task_labels
  required_columns <- c(
    "data_name", "data_preprocessing_name", "metric_name", "metric_value",
    "predicted_client_name", "cross_validation_method", "seed", "predicted_target",
    "learning_type"
  )
  assert_columns(report, required_columns, "RF metric report")
  report <- report[report$data_name %in% sub("\\|.*$", "", names(task_labels)), , drop = FALSE]
  if (!nrow(report)) stop("No canonical real-data RF tasks in the report", call. = FALSE)
  task_labels <- task_labels[sub("\\|.*$", "", names(task_labels)) %in% report$data_name]
  task_order <- unname(task_labels)
  arms <- c(label_aware = "corrected", no_label = "no_label")
  cv_titles <- c(
    "Leave-One-Cohort-Out" = "RF leave-one-cohort-out",
    "Train-Test-Split" = "RF 70:30 train/test split"
  )
  metric_labels <- c(MCC = "MCC", F1_score = "Weighted F1 score")
  if (anyNA(report[required_columns]) ||
      any(!report$data_preprocessing_name %in% c("uncorrected", unname(arms))) ||
      any(!report$learning_type %in% c("centralized", "federated")) ||
      any(!report$metric_name %in% names(metric_labels)) ||
      any(!report$cross_validation_method %in% names(cv_titles))) {
    stop("Missing or unexpected RF values in the report", call. = FALSE)
  }
  report$seed <- suppressWarnings(as.numeric(report$seed))
  if (any(!is.finite(report$seed) | report$seed != floor(report$seed))) {
    stop("RF seeds must contain finite integers", call. = FALSE)
  }
  seeds <- sort(unique(report$seed))
  arms <- arms[arms %in% report$data_preprocessing_name]
  if (!length(arms)) stop("No corrected RF arm found in the report", call. = FALSE)
  style <- figure4_scenarios()
  panels <- list()
  for (arm in names(arms)) {
    arm_data <- report[report$data_preprocessing_name %in% c("uncorrected", arms[[arm]]),
                       , drop = FALSE]
    scenario_labels <- classification_scenario_labels(arms[[arm]])
    for (metric in names(metric_labels)) {
      for (cv in names(cv_titles)) {
        data <- prepare_figure4_classification_data(
          arm_data, cv, task_labels, task_order, scenario_labels, style$order,
          simulated_datasets = character(), excluded_datasets = character(),
          average_client_name = "All", metric_name = metric
        )
        if (any(table(data$Task, data$Scenario) == 0L)) {
          stop("Missing RF task/scenario results for ", arm, "/", metric, "/", cv,
               call. = FALSE)
        }
        # Every recorded test cohort must have the same seeds in all scenarios.
        for (task in task_order) {
          task_data <- data[data$Task == task, , drop = FALSE]
          expected <- expand.grid(
            Scenario = style$order, predicted_client_name = unique(task_data$predicted_client_name),
            seed = seeds, stringsAsFactors = FALSE
          )
          key_columns <- c("Scenario", "predicted_client_name", "seed")
          keys <- function(frame) do.call(paste, c(frame[key_columns], sep = "|"))
          if (!setequal(keys(task_data), keys(expected))) {
            stop("Incomplete RF cohort/seed coverage for ", arm, "/", metric, "/", cv,
                 "/", task, "; expected every recorded seed in every scenario", call. = FALSE)
          }
        }
        path <- file.path(plots_dir, arm, paste0(
          "classification_", tolower(metric), "_", tolower(cv), "_boxplot.png"
        ))
        panels[[path]] <- list(data = data, arm = arm, metric = metric, cv = cv)
      }
    }
  }
  # Validate every panel before replacing any existing plots.
  for (path in names(panels)) {
    item <- panels[[path]]
    metric_min <- min(vapply(
      Filter(function(panel) panel$metric == item$metric, panels),
      function(panel) min(panel$data$metric_value), numeric(1)
    ))
    y_min <- if (item$metric == "MCC") min(-0.25, floor(metric_min * 4) / 4) else -0.02
    y_breaks <- if (item$metric == "MCC") {
      sort(unique(c(-0.2, seq(0, 1, 0.25), if (y_min < -0.25) y_min)))
    } else seq(0, 1, 0.25)
    panel <- plot_figure4_box_panel(
      data = item$data, x_col = "Task", y_col = "metric_value", x_levels = task_order,
      y_label = metric_labels[[item$metric]],
      panel_tag = paste(correction_arm_titles[[item$arm]], cv_titles[[item$cv]], sep = " · "),
      y_limits = c(y_min, 1.05), y_breaks = y_breaks,
      scenario_order = style$order, scenario_colors = style$colors, point_size = 1.05
    ) + ggplot2::guides(fill = ggplot2::guide_legend(nrow = 2, byrow = TRUE))
    dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
    ggplot2::ggsave(path, plot = panel, width = 7.5, height = 3.2,
                    units = "in", dpi = 600, bg = "white")
    message("Saved ", path, " (", nrow(item$data), " seed scores)")
  }
  invisible(names(panels))
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
  legend_gap = grid::unit(0.08, "cm"),
  clustering_feature_label = NULL,
  classification_feature_label = NULL
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
    y_limits = c(min(-0.25, floor(min(figure4_data$ari$ARI) * 4) / 4), 1.05),
    y_breaks = c(-0.2, 0, 0.25, 0.5, 0.75, 1),
    scenario_order = scenario_order,
    scenario_colors = scenario_colors,
    point_size = point_size
  ) + ggplot2::labs(subtitle = clustering_feature_label)
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
  ) + ggplot2::labs(subtitle = classification_feature_label)
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
  ) + ggplot2::labs(subtitle = classification_feature_label)

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
  # The height after the panel heights is the legend row.
  legend_height <- figure_plot$panel_heights[[length(panels) + 1]]
  panel_heights <- figure_plot$panel_heights[seq_along(panels)]
  legend <- extract_plot_legend(panels[[1]])
  has_legend <- legend_is_present(legend)

  panel_grobs <- lapply(panels, function(plot) {
    ggplot2::ggplotGrob(plot + ggplot2::theme(legend.position = "none"))
  })
  layout_heights <- interleave_null_units(panel_heights, figure_plot$panel_gap)
  if (has_legend) {
    layout_heights <- grid::unit.c(
      layout_heights,
      figure_plot$legend_gap,
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
