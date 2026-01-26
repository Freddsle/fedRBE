suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(stringr)
  library(ggplot2)
  library(patchwork)
})

source(file.path("..", "evaluation_utils", "plots_eda.R"))
source(file.path("..", "evaluation_utils", "rotation_utils.R"))

read_intensities <- function(path) {
  if (!file.exists(path)) {
    warning(paste("Missing intensities file:", path))
    return(NULL)
  }

  intensities <- read_delim(
    path,
    delim = "\t",
    col_names = TRUE,
    show_col_types = FALSE
  )
  if (!"rowname" %in% colnames(intensities)) {
    warning(paste("Missing rowname column in:", path))
    return(NULL)
  }
  tibble::column_to_rownames(intensities, "rowname")
}

replace_na_with_zero <- function(df) {
  df[is.na(df)] <- 0
  df
}

load_run_metadata <- function(run_dir, run_id) {
  cntrl_path <- file.path(run_dir, paste0(run_id, "_metadata_cntrl_kmeans_res.tsv"))
  before_fed_path <- file.path(run_dir, paste0(run_id, "_metadata_before_fedclusters.tsv"))
  after_fed_path <- file.path(run_dir, paste0(run_id, "_metadata_after_fedclusters.tsv"))

  missing <- c(cntrl_path, before_fed_path, after_fed_path)
  missing <- missing[!file.exists(missing)]
  if (length(missing) > 0) {
    warning(
      paste(
        "Skipping", run_id,
        "missing:", paste(basename(missing), collapse = ", ")
      )
    )
    return(NULL)
  }

  cntrl_res <- read_delim(
    cntrl_path,
    delim = "\t",
    col_names = TRUE,
    show_col_types = FALSE
  )
  before_fed <- read_delim(
    before_fed_path,
    delim = "\t",
    col_names = TRUE,
    show_col_types = FALSE
  )
  after_fed <- read_delim(
    after_fed_path,
    delim = "\t",
    col_names = TRUE,
    show_col_types = FALSE
  )

  key_cols <- c("file", "condition", "lab")
  if (!all(key_cols %in% colnames(cntrl_res)) ||
      !all(key_cols %in% colnames(before_fed)) ||
      !all(key_cols %in% colnames(after_fed))) {
    warning(paste("Skipping", run_id, "missing join columns"))
    return(NULL)
  }

  before_fed <- before_fed %>%
    rename(
      BeforeC_Fed_2clusters = Fed_2clusters,
      BeforeC_Fed_3clusters = Fed_3clusters
    )
  after_fed <- after_fed %>%
    rename(
      AfterC_Fed_2clusters = Fed_2clusters,
      AfterC_Fed_3clusters = Fed_3clusters
    )

  cntrl_res %>%
    inner_join(before_fed, by = key_cols) %>%
    inner_join(after_fed, by = key_cols) %>%
    mutate(
      BeforeC_Cntrl_2clusters = Before_CtrlKm_2clusters,
      AfterC_Cntrl_2clusters = Cor_CtrlKm_2clusters,
      BeforeC_Cntrl_3clusters = Before_CtrlKm_3clusters,
      AfterC_Cntrl_3clusters = Cor_CtrlKm_3clusters
    )
}

normalize_cluster_labels <- function(x, k, label = "") {
  if (all(is.na(x))) {
    warning(paste("All NA cluster values for", label))
    return(factor(x, levels = 0:(k - 1)))
  }

  uniq_vals <- unique(na.omit(x))
  if (length(uniq_vals) > k) {
    warning(paste("Found", length(uniq_vals), "cluster values for", label, "expected", k))
  }

  mapped <- as.integer(as.factor(x)) - 1L
  factor(mapped, levels = 0:(k - 1))
}

compute_ari <- function(truth, clusters, label = "") {
  mask <- !is.na(truth) & !is.na(clusters)
  if (!any(mask)) {
    warning(paste("No overlapping labels for ARI", label))
    return(NA_real_)
  }
  mclust::adjustedRandIndex(
    as.character(truth[mask]),
    as.character(clusters[mask])
  )
}

format_metric_line <- function(value, label = "ARI") {
  if (is.na(value)) {
    return(paste0(label, "=NA"))
  }
  paste0(label, "=", sprintf("%.3f", value))
}

make_pca_panel <- function(intensities,
                           plot_meta,
                           color_col,
                           shape_col,
                           title,
                           cb_palette,
                           show_legend) {
  pca_plot(
    intensities,
    plot_meta,
    title = title,
    quantitative_col_name = "file",
    col_col = color_col,
    shape_col = shape_col,
    show_legend = show_legend,
    cbPalette = cb_palette,
    point_size = 1.3
  )
}

build_cluster_plots <- function(intensities,
                                plot_meta,
                                cluster_cols,
                                panel_titles,
                                row_label,
                                cb_palette,
                                shape_col,
                                show_legend,
                                panel_metrics = NULL,
                                metric_label = "ARI") {
  if (!is.null(panel_metrics) && length(panel_metrics) != length(cluster_cols)) {
    warning("panel_metrics length does not match cluster_cols; skipping metric labels.")
    panel_metrics <- NULL
  }
  plots <- vector("list", length(cluster_cols))
  for (i in seq_along(cluster_cols)) {
    title_parts <- c(row_label, panel_titles[i])
    if (!is.null(panel_metrics)) {
      title_parts <- c(title_parts, format_metric_line(panel_metrics[i], metric_label))
    }
    plots[[i]] <- make_pca_panel(
      intensities,
      plot_meta,
      color_col = cluster_cols[i],
      shape_col = shape_col,
      title = paste(title_parts, collapse = "\n"),
      cb_palette = cb_palette,
      show_legend = show_legend && i == 1
    )
  }
  plots
}

plot_run_clusters <- function(intensities_before,
                              intensities_after,
                              metadata,
                              k,
                              mode,
                              run_id,
                              out_dir) {
  cluster_cols_before <- c(
    sprintf("BeforeC_Cntrl_%dclusters", k),
    sprintf("BeforeC_Fed_%dclusters", k)
  )
  cluster_cols_after <- c(
    sprintf("AfterC_Cntrl_%dclusters", k),
    sprintf("AfterC_Fed_%dclusters", k)
  )
  cluster_cols_all <- c(cluster_cols_before, cluster_cols_after)

  missing_cols <- cluster_cols_all[!cluster_cols_all %in% colnames(metadata)]
  if (length(missing_cols) > 0) {
    warning(
      paste(
        "Skipping", paste(mode, run_id, sep = "/"),
        "missing cluster columns:", paste(missing_cols, collapse = ", ")
      )
    )
    return(NULL)
  }

  plot_meta <- metadata %>%
    mutate(
      condition = factor(condition),
      lab = factor(lab)
    ) %>%
    mutate(across(
      all_of(cluster_cols_all),
      ~ normalize_cluster_labels(.x, k, label = cur_column())
    ))

  intensities_before <- replace_na_with_zero(intensities_before)
  intensities_after <- replace_na_with_zero(intensities_after)

  if (k == 2) {
    shape_col <- "lab"
    extra_color_col <- "condition"
    extra_title <- "True labels"
  } else {
    shape_col <- "condition"
    extra_color_col <- "lab"
    extra_title <- "Lab"
  }

  truth_vals <- plot_meta[[extra_color_col]]
  ari_before <- vapply(
    cluster_cols_before,
    function(col) compute_ari(truth_vals, plot_meta[[col]], label = col),
    numeric(1)
  )
  ari_after <- vapply(
    cluster_cols_after,
    function(col) compute_ari(truth_vals, plot_meta[[col]], label = col),
    numeric(1)
  )

  extra_levels <- levels(factor(plot_meta[[extra_color_col]]))
  extra_palette <- scales::hue_pal()(length(extra_levels))

  cb_palette <- scales::hue_pal()(k)
  panel_titles <- c(
    "Central clusters",
    "Federated clusters"
  )

  extra_before <- make_pca_panel(
    intensities_before,
    plot_meta,
    color_col = extra_color_col,
    shape_col = shape_col,
    title = paste("Before correction intensities", extra_title, sep = "\n"),
    cb_palette = extra_palette,
    show_legend = TRUE
  )
  extra_after <- make_pca_panel(
    intensities_after,
    plot_meta,
    color_col = extra_color_col,
    shape_col = shape_col,
    title = paste("After correction intensities", extra_title, sep = "\n"),
    cb_palette = extra_palette,
    show_legend = FALSE
  )

  plots_before <- c(list(extra_before), build_cluster_plots(
    intensities_before,
    plot_meta,
    cluster_cols_before,
    panel_titles,
    "Before correction intensities",
    cb_palette,
    shape_col,
    TRUE,
    panel_metrics = ari_before
  ))
  plots_after <- c(list(extra_after), build_cluster_plots(
    intensities_after,
    plot_meta,
    cluster_cols_after,
    panel_titles,
    "After correction intensities",
    cb_palette,
    shape_col,
    FALSE,
    panel_metrics = ari_after
  ))

  row_before <- wrap_plots(plots_before, nrow = 1)
  row_after <- wrap_plots(plots_after, nrow = 1)
  layout <- row_before / row_after
  layout <- layout +
    plot_layout(guides = "collect") &
    theme(legend.position = "right")

  layout <- layout +
    plot_annotation(
      title = paste0("Simulated ", mode, " run ", run_id, " (", k, " clusters)")
  )

  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  out_path <- file.path(out_dir, paste0(mode, "_run_", run_id, ".png"))
  ggsave(out_path, layout, width = 15, height = 7.5, dpi = 320)
  message("Saved: ", out_path)
}

base_data_root <- file.path("..", "evaluation_data", "simulated_rotation")
plot_root <- file.path("results")

for (mode in rotation_modes) {
  run_dir <- file.path(base_data_root, mode, "after", "kmeans_res", "runs")
  if (!dir.exists(run_dir)) {
    warning(paste("Run directory not found:", run_dir))
    next
  }

  run_files <- list.files(
    run_dir,
    pattern = "_metadata_cntrl_kmeans_res\\.tsv$",
    full.names = FALSE
  )
  run_ids <- str_replace(run_files, "_metadata_cntrl_kmeans_res\\.tsv$", "")
  run_ids <- sort(run_ids)

  if (length(run_ids) == 0) {
    warning(paste("No runs found for mode:", mode))
    next
  }

  for (run_id in run_ids) {
    before_path <- file.path(
      base_data_root,
      mode,
      "before",
      "intermediate",
      paste0(run_id, "_intensities_data.tsv")
    )
    after_path <- file.path(
      base_data_root,
      mode,
      "after",
      "runs",
      paste0(run_id, "_R_corrected.tsv")
    )
    intensities_before <- read_intensities(before_path)
    if (is.null(intensities_before)) {
      next
    }
    intensities_after <- read_intensities(after_path)
    if (is.null(intensities_after)) {
      next
    }

    metadata <- load_run_metadata(run_dir, run_id)
    if (is.null(metadata)) {
      next
    }

    plot_run_clusters(
      intensities_before,
      intensities_after,
      metadata,
      2,
      mode,
      run_id,
      file.path(plot_root, "plots_2")
    )
    plot_run_clusters(
      intensities_before,
      intensities_after,
      metadata,
      3,
      mode,
      run_id,
      file.path(plot_root, "plots_3")
    )
  }
}
