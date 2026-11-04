# Quartet separated-modality diagnostics
#
# Run from the repository root after preparing and correcting the default layout
# (48 profiles, four batches and three clients per modality):
#   CORRECTION_ARM=label_aware pixi run Rscript \
#     evaluation/quartet_separated_modality_pca_fixed_effect_variance.R
# Set CORRECTION_ARM=label_aware (default) or no_label.
#
# Inputs under evaluation_data/quartet_multiomics/:
# before/<Modality>/central_intensities_log_UNION.tsv,
# before/<Modality>/metadata.tsv and after/<arm>/<Modality>/FedApp_corrected_data.tsv.
#
# The sequential model ~ condition + batch assigns variance to donor first, then
# batch. PCA batch codes are modality-specific. Outputs are
# figures/figureS5_<arm>.png and a matching _variance.tsv summary.

source(file.path("evaluation_utils", "figure_aggregate_utils.R"))
require_figure_packages()
source_plotting_utils()

CORRECTION_ARM <- correction_arm()
output_dir <- "figures"
output_path <- file.path(output_dir, paste0("figureS5_", CORRECTION_ARM, ".png"))
summary_path <- file.path(output_dir, paste0("figureS5_", CORRECTION_ARM, "_variance.tsv"))

modalities <- c("Transcriptomics", "Proteomics", "Metabolomics")
modality_titles <- c(
  Transcriptomics = "Quartet: Transcriptomics",
  Proteomics = "Quartet: Proteomics",
  Metabolomics = "Quartet: Metabolomics"
)

batch_shapes <- c(B01 = 16, B02 = 17, B03 = 3, B04 = 15)
term_labels <- c(condition = "Condition", batch = "Batch")

form <- ~ condition + batch
figure_width <- 6
figure_height <- 5
figure_dpi <- 600

quartet_modality_data <- setNames(lapply(modalities, function(modality) {
  prepare_plot_matrices(load_quartet_modality_data(modality, CORRECTION_ARM))
}), modalities)

compact_modality_theme <- ggplot2::theme(
  plot.title = ggplot2::element_text(size = 6.724, margin = ggplot2::margin(b = 1.5)),
  axis.title = ggplot2::element_text(size = 5.904),
  axis.text = ggplot2::element_text(size = 5.248),
  legend.title = ggplot2::element_text(size = 5.576),
  legend.text = ggplot2::element_text(size = 5.084),
  legend.key.width = grid::unit(0.3772, "cm"),
  legend.key.height = grid::unit(0.1968, "cm"),
  legend.spacing.y = grid::unit(0.05, "cm"),
  legend.margin = ggplot2::margin(0, 0, 0, 0),
  plot.margin = ggplot2::margin(2, 2, 2, 4)
)

compact_modality_guides <- ggplot2::guides(
  color = ggplot2::guide_legend(override.aes = list(size = 1.23), title.position = "top"),
  shape = ggplot2::guide_legend(override.aes = list(size = 1.23), title.position = "top"),
  fill = ggplot2::guide_legend(
    title.position = "top",
    keywidth = grid::unit(0.3772, "cm"),
    keyheight = grid::unit(0.1968, "cm")
  )
)

style_modality_plot <- function(plot) {
  plot + compact_modality_theme + compact_modality_guides
}

style_pca_plot <- function(plot) {
  style_modality_plot(plot) +
    ggplot2::scale_shape_manual(values = batch_shapes, limits = quartet_batch_levels, drop = FALSE)
}

draw_modality_diagnostic_row <- function(
  row,
  title_fontsize = 8.8,
  plot_column_gap = grid::unit(0.12, "cm")
) {
  layout <- grid::grid.layout(
    nrow = 2,
    ncol = 7,
    heights = grid::unit.c(grid::unit(0.38, "cm"), grid::unit(1, "null")),
    widths = grid::unit.c(
      grid::unit(1, "null"),
      plot_column_gap,
      grid::unit(1, "null"),
      plot_column_gap,
      grid::unit(1, "null"),
      plot_column_gap,
      grid::unit(1, "null")
    )
  )
  grid::pushViewport(grid::viewport(layout = layout))

  grid::pushViewport(grid::viewport(layout.pos.row = 1, layout.pos.col = 1:7))
  grid::grid.text(
    row$title,
    x = grid::unit(0, "npc"),
    y = grid::unit(0.55, "npc"),
    just = c("left", "center"),
    gp = grid::gpar(fontface = "bold", fontsize = title_fontsize)
  )
  grid::popViewport()

  plot_positions <- list(
    pca_before = c(2, 1),
    pca_after = c(2, 3),
    variance_before = c(2, 5),
    variance_after = c(2, 7)
  )
  for (plot_name in names(plot_positions)) {
    position <- plot_positions[[plot_name]]
    grid::pushViewport(grid::viewport(layout.pos.row = position[[1]], layout.pos.col = position[[2]]))
    grid::grid.draw(ggplot2::ggplotGrob(row$plots[[plot_name]] + ggplot2::theme(legend.position = "none")))
    grid::popViewport()
  }
  grid::popViewport()
  invisible(row)
}

extract_bottom_legend <- function(plot, legend_box = "horizontal") {
  extract_plot_legend(plot + ggplot2::theme(legend.position = "bottom", legend.box = legend_box))
}

draw_modality_bottom_legends <- function(
  rows,
  legend_source_index = length(rows),
  plot_column_gap = grid::unit(0.12, "cm")
) {
  legend_source <- rows[[legend_source_index]]
  legends <- list(
    pca = extract_bottom_legend(legend_source$plots$pca_after, legend_box = "vertical"),
    variance = extract_bottom_legend(legend_source$plots$variance_after)
  )

  layout <- grid::grid.layout(
    nrow = 1,
    ncol = 7,
    widths = grid::unit.c(
      grid::unit(1, "null"),
      plot_column_gap,
      grid::unit(1, "null"),
      plot_column_gap,
      grid::unit(1, "null"),
      plot_column_gap,
      grid::unit(1, "null")
    )
  )
  grid::pushViewport(grid::viewport(layout = layout))
  if (legend_is_present(legends$pca)) {
    grid::pushViewport(grid::viewport(layout.pos.row = 1, layout.pos.col = 1:3))
    grid::grid.draw(legends$pca)
    grid::popViewport()
  }
  if (legend_is_present(legends$variance)) {
    grid::pushViewport(grid::viewport(layout.pos.row = 1, layout.pos.col = 5:7))
    grid::grid.draw(legends$variance)
    grid::popViewport()
  }
  grid::popViewport()
  invisible(legends)
}

draw_modality_diagnostic_rows <- function(
  rows,
  row_gap = grid::unit(0.28, "cm"),
  legend_gap = grid::unit(0.12, "cm"),
  bottom_legend_height = grid::unit(1.35, "cm")
) {
  grid::grid.newpage()
  row_count <- length(rows) * 2 + 2
  layout <- grid::grid.layout(
    nrow = row_count,
    ncol = 1,
    heights = grid::unit.c(
      grid::unit(0.5, "cm"),
      interleave_null_units(rep(1, length(rows)), row_gap),
      legend_gap,
      bottom_legend_height
    )
  )
  grid::pushViewport(grid::viewport(layout = layout))
  grid::pushViewport(grid::viewport(layout.pos.row = 1, layout.pos.col = 1))
  grid::grid.text(
    correction_arm_titles[[CORRECTION_ARM]],
    gp = grid::gpar(fontface = "bold", fontsize = 10)
  )
  grid::popViewport()
  for (index in seq_along(rows)) {
    grid::pushViewport(grid::viewport(layout.pos.row = index * 2, layout.pos.col = 1))
    draw_modality_diagnostic_row(rows[[index]])
    grid::popViewport()
  }
  grid::pushViewport(grid::viewport(layout.pos.row = row_count, layout.pos.col = 1))
  draw_modality_bottom_legends(rows)
  grid::popViewport()
  grid::popViewport()
  invisible(rows)
}

build_modality_row <- function(modality) {
  modality_data <- quartet_modality_data[[modality]]
  plot_matrices <- modality_data$plot_matrices
  metadata <- modality_data$metadata

  pca_before <- pca_plot(
    df = plot_matrices$uncorrected,
    batch_info = metadata,
    title = "Before correction",
    quantitative_col_name = "file",
    col_col = "condition",
    shape_col = "batch_code",
    show_legend = FALSE,
    cbPalette = quartet_condition_palette,
    color_title = "Donor",
    shape_title = "Batch",
    point_size = 0.7,
    use_gram = TRUE
  )

  pca_after <- pca_plot(
    df = plot_matrices$fed_corrected,
    batch_info = metadata,
    title = "After fedRBE correction",
    quantitative_col_name = "file",
    col_col = "condition",
    shape_col = "batch_code",
    show_legend = TRUE,
    cbPalette = quartet_condition_palette,
    color_title = "Donor",
    shape_title = "Batch",
    point_size = 0.7,
    use_gram = TRUE
  )

  # Sequential fixed-effect variance fractions, without an x-axis title.
  variance_before <- lmpv_plot(
    data = plot_matrices$uncorrected,
    metadata = metadata,
    title = "Before correction",
    form = form,
    show_legend = FALSE,
    term_labels = term_labels,
    fill_title = "Effect",
    median_label_size = 2.2,
    partition = fixed_effect_variance_partition,
    box_args = list(width = 0.62, outlier.size = 0.25, linewidth = 0.25),
    median_nudge = -0.18
  ) + ggplot2::labs(x = NULL)

  variance_after <- lmpv_plot(
    data = plot_matrices$fed_corrected,
    metadata = metadata,
    title = "After fedRBE correction",
    form = form,
    show_legend = TRUE,
    term_labels = term_labels,
    fill_title = "Effect",
    median_label_size = 2.2,
    partition = fixed_effect_variance_partition,
    box_args = list(width = 0.62, outlier.size = 0.25, linewidth = 0.25),
    median_nudge = -0.18
  ) + ggplot2::labs(x = NULL)

  make_dataset_diagnostic_panel(
    title = unname(modality_titles[[modality]]),
    pca_before = style_pca_plot(pca_before),
    pca_after = style_pca_plot(pca_after),
    variance_before = style_modality_plot(variance_before),
    variance_after = style_modality_plot(variance_after)
  )
}

quartet_modality_rows <- setNames(lapply(modalities, build_modality_row), modalities)

saved_path <- save_grid_png(
  function() draw_modality_diagnostic_rows(quartet_modality_rows),
  output_path = output_path,
  width = figure_width,
  height = figure_height,
  dpi = figure_dpi
)

variance_summary <- do.call(rbind, lapply(modalities, function(modality) {
  do.call(rbind, lapply(c("before", "after"), function(state) {
    plot_data <- quartet_modality_rows[[modality]]$plots[[paste0("variance_", state)]]$data
    medians <- dplyr::summarize(
      dplyr::group_by(plot_data, variable),
      features = dplyr::n(),
      median_fraction = median(value, na.rm = TRUE),
      .groups = "drop"
    )
    data.frame(
      modality = modality, state = state, term = as.character(medians$variable),
      features = medians$features, median_fraction = medians$median_fraction
    )
  }))
}))
utils::write.table(variance_summary, summary_path, sep = "\t", quote = FALSE, row.names = FALSE)

cat("Saved Quartet separated-modality diagnostics to", saved_path, "\n")
cat("Saved variance summary to", summary_path, "\n")
