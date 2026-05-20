
library(tidyverse)
library(gridExtra)
library(patchwork)
library(grid)
library(umap)
library(ggsci)

library(viridis)

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
    use_nipals = FALSE
    ){
  if (use_nipals) {
    if (!requireNamespace("nipals", quietly = TRUE)) stop("Package 'nipals' is required for NIPALS PCA. Install it with install.packages('nipals').")
    ncomp <- max(as.integer(sub("PC", "", c(pc_x, pc_y))))
    res <- nipals::nipals(
      t(df),
      ncomp = ncomp,
      center = TRUE,
      scale = TRUE,
      maxiter = 500,
      startcol = function(x) sum(!is.na(x))
    )
    scores <- res$scores %*% diag(res$eig)
    colnames(scores) <- paste0("PC", seq_len(ncol(scores)))
    pca_df <- scores %>%
      as.data.frame() %>%
      rownames_to_column(quantitative_col_name) %>%
      left_join(batch_info, by = quantitative_col_name)
    var_expl <- res$eig^2 / sum(res$eig^2)
    names(var_expl) <- paste0("PC", seq_along(var_expl))
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

  pca_plotting <- pca_plotting + 
    geom_point(size=point_size) +
    theme_classic() +
    labs(title = title,
         x = glue::glue("{pc_x} [{round(var_expl[pc_x]*100, 2)}%]"),
         y = glue::glue("{pc_y} [{round(var_expl[pc_y]*100, 2)}%]"))

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



umap_plot <- function(
  df, metadata, 
  title = "UMAP Projection", 
  color_column = "study_accession", 
  shape_column = "class",
  quantitative_col_name = 'sample',
  show_legend = TRUE,
  path = "", cbPalette=NULL) {
  
  # Perform UMAP on the transposed data
  umap_result <- umap(t(na.omit(df)))
  
  # Convert the UMAP result into a data frame and merge with metadata
  umap_data <- umap_result$layout %>%
    as.data.frame() %>%
    setNames(c("X1", "X2")) %>% 
    rownames_to_column(quantitative_col_name) %>% 
    left_join(metadata, by = quantitative_col_name) %>%
    column_to_rownames(quantitative_col_name)

  plot_result <- ggplot(umap_data, build_scatter_mapping("X1", "X2", color_column, shape_column)) +
    geom_point(size = 2) +
    theme_minimal()

  if (!is.null(cbPalette)) {
    plot_result <- plot_result + scale_color_manual(values = cbPalette)

  }
  
  plot_result <- plot_result +
    labs(
      title = title, 
      x = "UMAP 1", 
      y = "UMAP 2", 
      color = "cohorts"  # Update legend label
    ) +
    guides(color = guide_legend(override.aes = list(size = 3))) # Ensure legend accurately represents centroids
  
  if (!show_legend) {
    plot_result <- plot_result + 
      theme(legend.position = "none")
  }
    
  save_plot_if_requested(plot_result, path)
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

boxplot_plot_groupped <- function(matrix, metadata_df, quantitativeColumnName, color_col, title, path="",
                         remove_xnames=FALSE, y_limits=NULL,
                         show_legend=TRUE, cbPalette=NULL) {
  merged_data <- prepare_intensity_plot_data(matrix, metadata_df, quantitativeColumnName)
  
  # Group by color_col
  merged_data_grouped <- merged_data %>%
    group_by(.data[[color_col]])
  
  # Log transformed scale
  boxplot <- ggplot(merged_data_grouped, aes(x = .data[[color_col]], y = Intensity, fill = .data[[color_col]])) + 
    geom_violin(trim = FALSE) +
    stat_summary(fun = median, geom = "crossbar", width = 0.35, color = "black", position = position_dodge(width = 0.2)) +
    stat_summary(fun = mean, geom = "point", shape = 4, size = 3, color = "darkred") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1)) +
    # adjust font size for the x-axis
    theme(axis.text.x = element_text(size = 8)) +
    labs(title = title) +
    guides(fill = guide_legend(override.aes = list(shape = NA, linetype = 0)))

  if(remove_xnames){
    boxplot <- boxplot + theme(axis.title.x=element_blank(), axis.text.x=element_blank(), axis.ticks.x=element_blank())
  }

  if (!is.null(y_limits)) {
    boxplot <- boxplot + coord_cartesian(ylim = y_limits)
  }

  if (!is.null(cbPalette)) {
    boxplot <- boxplot + scale_fill_manual(values = cbPalette)
  }


  if(!show_legend){
    boxplot <- boxplot + theme(legend.position = "none")
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


heatmap_plot <- function(pg_matrix, batch_info, name, condition="condition", lab="lab"){
    cor_matrix <- cor(na.omit(pg_matrix), use = "pairwise.complete.obs")
    resulting_plot <- ggpubr::as_ggplot(grid::grid.grabExpr(
        pheatmap::pheatmap(cor_matrix, 
                        annotation_col = select(batch_info, c(condition, lab)),
                        treeheight_row = 0, treeheight_col = 0, 
                        main = paste0(name, ' heatmap')
        )
      )
    )
    return(resulting_plot)
}


plots_multiple <- function(intensities, metadata, name, simulated = FALSE, use_nipals = FALSE){

  pca_plot_study <- pca_plot(
      intensities, metadata,
      title = name,
      quantitative_col_name = 'file',
      col_col = "lab", shape_col = "condition",
      use_nipals = use_nipals)

  density_plot <- plotIntensityDensity(
      intensities, metadata,
      quantitativeColumnName = 'file',
      colorColumnName = 'lab',
      title = name)

  if (simulated) {
    boxplot <- boxplot_plot_groupped(
        intensities, metadata,
        title = name,
        color_col = 'lab', quantitativeColumnName = 'file',
        path = '')

    layout <- pca_plot_study /
                boxplot /
                density_plot
  } else {
    pca_plot_class <- pca_plot(
        intensities, metadata,
        title = name,
        quantitative_col_name = 'file',
        shape_col = "lab", col_col = "condition",
        use_nipals = use_nipals)

    boxplot <- boxplot_plot(
        intensities, metadata,
        title = name,
        color_col = 'lab', quantitativeColumnName = 'file',
        path = '')

    layout <- (pca_plot_class | pca_plot_study) /
                boxplot /
                density_plot
  }

  return(layout)
}
