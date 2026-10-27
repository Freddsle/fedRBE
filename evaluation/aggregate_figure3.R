# Aggregate Figure 3
#
# Generate the four real-dataset Figure 3 panels from source matrices and metadata.
# Run from the repository root:
#   CORRECTION_ARM=label_aware pixi run Rscript evaluation/aggregate_figure3.R
# Set CORRECTION_ARM=label_aware (default) or CORRECTION_ARM=no_label. Each arm
# writes its own PNG and median variance summary under figures/.

source(file.path("evaluation_utils", "figure_aggregate_utils.R"))
require_figure_packages()
source_plotting_utils()

# Paths

CORRECTION_ARM <- correction_arm()
output_dir <- "figures"
output_path <- file.path(output_dir, paste0("figure3_", CORRECTION_ARM, ".png"))

# Labels and style: panel titles, palettes, model-term labels and output size.

panel_titles <- c(
  ecoli = "a) E. coli (proteomics)",
  ovarian = "b) Ovarian cancer (microarrays)",
  ccrcc = "c) ccRCC (proteomics)",
  quartet = "d) Quartet (multi-omics)"
)

two_condition_palette <- c("#44abe7", "#c55702")

common_term_labels <- c(
  condition = "Condition",
  lab = "Dataset",
  batch = "Batch",
  Status = "Status",
  Dataset = "Dataset"
)

figure_width_cm <- 18.5
figure_height_cm <- 18
cm_to_inches <- function(value_cm) value_cm / 2.54
figure_width <- cm_to_inches(figure_width_cm)
figure_height <- cm_to_inches(figure_height_cm)
figure_dpi <- 600

# Load data: the before and after-BEC matrices of each real-world dataset.

figure3_data <- list(
  ecoli = prepare_plot_matrices(load_ecoli_data(CORRECTION_ARM)),
  ovarian = prepare_plot_matrices(load_ovarian_data(CORRECTION_ARM)),
  ccrcc = prepare_plot_matrices(load_ccrcc_data(CORRECTION_ARM)),
  quartet = prepare_plot_matrices(load_quartet_data(CORRECTION_ARM))
)

# Build individual panels: the shared plot style and dataset-specific plotting
# settings, applied to each dataset panel through the same local helper.

compact_figure_theme <- ggplot2::theme(
  plot.title = ggplot2::element_text(size = 7, margin = ggplot2::margin(b = 1.5)),
  axis.title = ggplot2::element_text(size = 6.8),
  axis.text = ggplot2::element_text(size = 6.2),
  legend.title = ggplot2::element_text(size = 6.5),
  legend.text = ggplot2::element_text(size = 5.9),
  legend.key.width = grid::unit(0.46, "cm"),
  legend.key.height = grid::unit(0.24, "cm"),
  legend.spacing.y = grid::unit(0.06, "cm"),
  legend.margin = ggplot2::margin(0, 0, 0, 0),
  plot.margin = ggplot2::margin(2, 2, 2, 5)
)

compact_figure_guides <- ggplot2::guides(
  color = ggplot2::guide_legend(override.aes = list(size = 1.45), title.position = "top"),
  shape = ggplot2::guide_legend(override.aes = list(size = 1.45), title.position = "top"),
  fill = ggplot2::guide_legend(
    title.position = "top",
    keywidth = grid::unit(0.46, "cm"),
    keyheight = grid::unit(0.24, "cm")
  )
)

variance_median_label_size <- 2.15

dataset_panel_order <- c("ecoli", "ovarian", "ccrcc", "quartet")

dataset_plot_specs <- list(
  ecoli = list(
    color_col = "condition",
    shape_col = "lab",
    palette = two_condition_palette,
    color_title = "Condition",
    shape_title = "Dataset",
    form = ~ condition + lab
  ),
  ovarian = list(
    color_col = "Status",
    shape_col = "Dataset",
    palette = two_condition_palette,
    color_title = "Status",
    shape_title = "Dataset",
    form = ~ Status + Dataset
  ),
  ccrcc = list(
    color_col = "condition",
    shape_col = "lab",
    palette = two_condition_palette,
    color_title = "Condition",
    shape_title = "Dataset",
    form = ~ condition + lab
  ),
  quartet = list(
    color_col = "condition",
    shape_col = "batch_code",
    palette = quartet_condition_palette,
    color_title = "Condition",
    shape_title = "Batch",
    form = ~ condition + batch
  )
)

style_figure3_plot <- function(plot) {
  plot + compact_figure_theme + compact_figure_guides
}

build_dataset_panel <- function(dataset_key) {
  spec <- dataset_plot_specs[[dataset_key]]
  dataset_data <- figure3_data[[dataset_key]]
  plot_matrices <- dataset_data$plot_matrices
  metadata <- dataset_data$metadata

  pca_before <- pca_plot(
    df = plot_matrices$uncorrected,
    batch_info = metadata,
    title = "Before correction",
    quantitative_col_name = "file",
    col_col = spec$color_col,
    shape_col = spec$shape_col,
    show_legend = FALSE,
    cbPalette = spec$palette,
    color_title = spec$color_title,
    shape_title = spec$shape_title,
    point_size = 0.8,
    use_gram = TRUE
  )

  pca_after <- pca_plot(
    df = plot_matrices$fed_corrected,
    batch_info = metadata,
    title = "After fedRBE",
    quantitative_col_name = "file",
    col_col = spec$color_col,
    shape_col = spec$shape_col,
    show_legend = TRUE,
    cbPalette = spec$palette,
    color_title = spec$color_title,
    shape_title = spec$shape_title,
    point_size = 0.8,
    use_gram = TRUE
  )

  variance_before <- lmpv_plot(
    data = plot_matrices$uncorrected,
    metadata = metadata,
    title = "Before correction",
    form = spec$form,
    show_legend = FALSE,
    term_labels = common_term_labels,
    fill_title = NULL,
    median_label_size = variance_median_label_size
  )

  variance_after <- lmpv_plot(
    data = plot_matrices$fed_corrected,
    metadata = metadata,
    title = "After fedRBE",
    form = spec$form,
    show_legend = TRUE,
    term_labels = common_term_labels,
    fill_title = NULL,
    median_label_size = variance_median_label_size
  )

  make_dataset_diagnostic_panel(
    title = panel_titles[[dataset_key]],
    pca_before = style_figure3_plot(pca_before),
    pca_after = style_figure3_plot(pca_after),
    variance_before = style_figure3_plot(variance_before),
    variance_after = style_figure3_plot(variance_after)
  )
}

# Assemble panels: build all four dataset panels before arranging them.

figure3_dataset_panels <- lapply(dataset_panel_order, build_dataset_panel)
names(figure3_dataset_panels) <- dataset_panel_order

# Aggregate layout: the final two-column arrangement of the dataset panels.

figure3_layout <- arrange_dataset_panels(
  panels = figure3_dataset_panels,
  ncol = 2,
  title_fontsize = 8.8,
  subplot_row_gap = grid::unit(0.34, "cm"),
  subplot_column_gap = grid::unit(0.14, "cm"),
  legend_gap = grid::unit(0.18, "cm"),
  dataset_row_gap = grid::unit(0.34, "cm"),
  dataset_column_gap = grid::unit(0.18, "cm")
)

# Save the median variance summary and the paper-ready aggregate Figure 3 image.

variance_summary <- do.call(rbind, lapply(names(figure3_dataset_panels), function(dataset) {
  panel <- figure3_dataset_panels[[dataset]]
  do.call(rbind, lapply(c("before", "after"), function(state) {
    values <- panel$plots[[paste0("variance_", state)]]$data
    summary <- stats::aggregate(value ~ variable, values, stats::median)
    data.frame(
      dataset = dataset, state = state, term = summary$variable,
      features = nrow(figure3_data[[dataset]]$plot_matrices$uncorrected),
      median_fraction = summary$value
    )
  }))
}))
utils::write.table(
  variance_summary,
  file.path(output_dir, paste0("figure3_", CORRECTION_ARM, "_variance.tsv")),
  sep = "\t", row.names = FALSE, quote = FALSE
)

saved_path <- save_grid_png(
  function() draw_dataset_panel_layout(figure3_layout),
  output_path = output_path,
  width = figure_width,
  height = figure_height * figure3_layout$nrow / 2,
  dpi = figure_dpi
)

cat("Saved Figure 3 to", saved_path, "\n")
