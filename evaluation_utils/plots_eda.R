
library(tidyverse)
library(grid)
suppressPackageStartupMessages(library(patchwork))

build_scatter_mapping <- function(x, y, color_col, shape_col = NULL) {
  mapping <- list(x = rlang::sym(x), y = rlang::sym(y), color = rlang::sym(color_col))
  if (!is.null(shape_col) && nzchar(shape_col)) {
    mapping$shape <- rlang::sym(shape_col)
  }
  do.call(ggplot2::aes, mapping)
}

prepare_intensity_plot_data <- function(matrix_like, metadata_df, quantitativeColumnName) {
  as.data.frame(matrix_like, check.names = FALSE) %>%
    tidyr::pivot_longer(
      cols = dplyr::everything(),
      names_to = "file",
      values_to = "Intensity"
    ) %>%
    dplyr::left_join(
      metadata_df,
      by = stats::setNames(quantitativeColumnName, "file")
    )
}

save_plot_if_requested <- function(plot_obj, path, width = NULL, height = NULL) {
  if (!is.null(path) && nzchar(path)) {
    ggsave(path, plot_obj, width = width, height = height)
  }
  plot_obj
}


gram_pca_scores <- function(df, pc_x = "PC1", pc_y = "PC2") {
  complete_data <- df[stats::complete.cases(df), , drop = FALSE]
  if (nrow(complete_data) == 0) {
    stop("PCA requires at least one complete feature row.", call. = FALSE)
  }

  ncomp <- max(as.integer(sub("PC", "", c(pc_x, pc_y))))
  response_matrix <- t(as.matrix(complete_data))
  storage.mode(response_matrix) <- "double"
  centered <- scale(response_matrix, center = TRUE, scale = FALSE)
  gram <- tcrossprod(centered)
  eig <- eigen(gram, symmetric = TRUE)
  eigen_values <- pmax(eig$values, 0)
  keep <- seq_len(min(ncomp, length(eigen_values)))
  scores <- eig$vectors[, keep, drop = FALSE] %*%
    diag(sqrt(eigen_values[keep]), nrow = length(keep))
  rownames(scores) <- colnames(complete_data)
  colnames(scores) <- paste0("PC", keep)
  var_expl <- eigen_values / sum(eigen_values)
  names(var_expl) <- paste0("PC", seq_along(var_expl))

  list(scores = scores, var_expl = var_expl)
}


pca_plot <- function(
    df, 
    batch_info, 
    title, 
    path = "", 
    quantitative_col_name = "Quantitative.column.name", 
    col_col = "Group", 
    shape_col = "",
    pc_x = "PC1",  # Default principal component for the x-axis
    pc_y = "PC2",  # Default principal component for the y-axis
    show_legend = TRUE,
    cbPalette = NULL,
    point_size = 2,
    use_gram = FALSE,
    color_title = NULL,
    shape_title = NULL
    ){
  if (use_gram) {
    pca <- gram_pca_scores(df, pc_x = pc_x, pc_y = pc_y)
    pca_df <- pca$scores %>%
      as.data.frame() %>%
      rownames_to_column(quantitative_col_name) %>%
      left_join(batch_info, by = quantitative_col_name)
    var_expl <- pca$var_expl
  } else {
    pca <- prcomp(t(na.omit(df)))
    pca_df <- pca$x %>%
      as.data.frame() %>%
      rownames_to_column(quantitative_col_name) %>% 
      left_join(batch_info, by = quantitative_col_name)
    var_expl <- pca$sdev^2 / sum(pca$sdev^2)
    names(var_expl) <- paste0("PC", 1:length(var_expl))
  }

  missing_pcs <- setdiff(c(pc_x, pc_y), colnames(pca_df))
  if (length(missing_pcs) > 0) {
    stop("Requested principal component(s) not available: ", paste(missing_pcs, collapse = ", "), call. = FALSE)
  }

  pca_plotting <- ggplot(
    pca_df,
    build_scatter_mapping(pc_x, pc_y, col_col, shape_col)
  )

  if (nzchar(shape_col) && length(unique(pca_df[[shape_col]])) > 6) {
    shapes_codes <- c(0, 1, 3, 8, 7, 15, 19)
    pca_plotting <- pca_plotting + scale_shape_manual(values = shapes_codes)
  }

  label_args <- list(
    title = title,
    x = glue::glue("{pc_x} [{round(var_expl[pc_x]*100, 2)}%]"),
    y = glue::glue("{pc_y} [{round(var_expl[pc_y]*100, 2)}%]")
  )
  if (!is.null(color_title)) {
    label_args$color <- color_title
  }
  if (!is.null(shape_title)) {
    label_args$shape <- shape_title
  }

  pca_plotting <- pca_plotting + 
    geom_point(size=point_size) +
    theme_classic() +
    do.call(ggplot2::labs, label_args)

  if(!show_legend){
    pca_plotting <- pca_plotting + 
      theme(legend.position = "none")
  }


  if (!is.null(cbPalette)) {
    pca_plotting <- pca_plotting + scale_color_manual(values = cbPalette)
  }

  save_plot_if_requested(pca_plotting, path, width = 5, height = 5)
}


plots_normtumor <- function(expr, metadata, i, dataset){
    pca_plot <- pca_plot(expr, metadata, 
        paste0(dataset , i, ", norm_vs_tumor"),
        col_col = 'Status', shape_col = "Dataset", quantitative_col_name = 'Sample_geo_ID')

    boxplot <- boxplot_plot(expr, metadata, 
        title = paste0(dataset , i, ", norm_vs_tumor"),
        color_col = 'Status', quantitativeColumnName = 'Sample_geo_ID', 
        path = '')

    density_plot <- plotIntensityDensity(expr, metadata, 
        quantitativeColumnName = 'Sample_geo_ID', 
        colorColumnName = 'Status',
        title = paste0(dataset , i, ", norm_vs_tumor"))

    return(list(pca_plot, density_plot, boxplot))
}



# boxplot
boxplot_plot <- function(matrix, metadata_df, quantitativeColumnName, color_col, title, path="",
                         remove_xnames=FALSE) {
  merged_data <- prepare_intensity_plot_data(matrix, metadata_df, quantitativeColumnName)
  
  # Log tranformed scale
  boxplot <- ggplot(merged_data, aes(x = file, y = Intensity, fill = .data[[color_col]])) + 
    geom_boxplot() +
    stat_summary(fun = mean, geom = "point", shape = 4, size = 3, color = "red") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1, size = 8)) +
    labs(title = title) 

  if(remove_xnames){
    boxplot <- boxplot + theme(axis.title.x=element_blank(), axis.text.x=element_blank(), axis.ticks.x=element_blank())
  }

  save_plot_if_requested(boxplot, path)
}

plotIntensityDensity <- function(
    intensities_df, metadata_df, quantitativeColumnName, colorColumnName, title, path=""
) {
  merged_data <- prepare_intensity_plot_data(intensities_df, metadata_df, quantitativeColumnName) %>%
    dplyr::rename(Sample = file)
  
  # Plot the data
  results <- ggplot(merged_data, aes(x = Intensity, color = .data[[colorColumnName]])) +  
    geom_density() +
    theme_minimal() +
    labs(title = paste(title, " by", colorColumnName),
         x = "Intensity",
         y = "Density")

  save_plot_if_requested(results, path)
}


plots_multiple <- function(intensities, metadata, name){

  pca_plot_study <- pca_plot(
      intensities, metadata,
      title = name,
      quantitative_col_name = 'file',
      col_col = "lab", shape_col = "condition")

  density_plot <- plotIntensityDensity(
      intensities, metadata,
      quantitativeColumnName = 'file',
      colorColumnName = 'lab',
      title = name)

  pca_plot_class <- pca_plot(
      intensities, metadata,
      title = name,
      quantitative_col_name = 'file',
      shape_col = "lab", col_col = "condition")

  boxplot <- boxplot_plot(
      intensities, metadata,
      title = name,
      color_col = 'lab', quantitativeColumnName = 'file',
      path = '')

  layout <- (pca_plot_class | pca_plot_study) /
              boxplot /
              density_plot

  return(layout)
}

# Saves a diagnostic panel per (modality, state): PCA by batch, PCA by donor,
# sample boxplot coloured by batch, density by batch. Metadata needs file,
# batch_code and condition columns (Quartet multiomics layout).
plot_diagnostics <- function(expr, metadata, modality, state_label, plots_dir) {
  pca_batch <- pca_plot(expr, metadata,
    title = paste(modality, state_label, "by batch"),
    quantitative_col_name = "file", col_col = "batch_code", shape_col = "condition",
    point_size = 2.5)

  pca_condition <- pca_plot(expr, metadata,
    title = paste(modality, state_label, "by donor"),
    quantitative_col_name = "file", col_col = "condition", shape_col = "batch_code",
    point_size = 2.5)

  box_by_batch <- boxplot_plot(expr, metadata,
    quantitativeColumnName = "file", color_col = "batch_code",
    title = paste(modality, state_label, "batch distributions"),
    remove_xnames = TRUE)

  density_by_batch <- plotIntensityDensity(expr, metadata,
    quantitativeColumnName = "file", colorColumnName = "batch_code",
    title = paste(modality, state_label))

  layout <- (pca_batch | pca_condition) / box_by_batch / density_by_batch
  out_path <- file.path(plots_dir, paste0(modality, "_limma_", state_label, "_diagnostics.png"))
  ggsave(out_path, layout, width = 14, height = 16, bg = "white")

  tibble(modality = modality, state = state_label, pca_features = nrow(na.omit(expr)))
}
