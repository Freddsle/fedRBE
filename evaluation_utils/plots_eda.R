
library(tidyverse)
library(gridExtra)
library(patchwork)
library(grid)
library(umap)
library(ggsci)


pca_plot <- function(
    df, 
    batch_info, 
    title, 
    path = "", 
    quantitative_col_name = "Quantitative.column.name", 
    col_col = "Group", 
    shape_col = "",
    show_legend = TRUE
    ){
  pca <- prcomp(t(na.omit(df)))
  pca_df <- pca$x %>%
    as.data.frame() %>%
    rownames_to_column(quantitative_col_name) %>% 
    left_join(batch_info, by = quantitative_col_name)
  var_expl <- pca$sdev^2 / sum(pca$sdev^2)
  names(var_expl) <- paste0("PC", 1:length(var_expl))

  if(shape_col != ""){
    pca_plot <- pca_df %>%
        ggplot(aes_string(x = "PC1", y = "PC2", color = col_col, shape = shape_col))
  } else {
    pca_plot <- pca_df %>%
        ggplot(aes_string(x = "PC1", y = "PC2", color = col_col))
  }

  pca_plot <- pca_plot + 
    geom_point(size=3) +
    theme_classic() +
    labs(title = title,
         x = glue::glue("PC1 [{round(var_expl['PC1']*100, 2)}%]"),
         y = glue::glue("PC2 [{round(var_expl['PC2']*100, 2)}%]"))

  if(!show_legend){
    pca_plot <- pca_plot + 
      theme(legend.position = "none")
  }

  if (path == "") {
    return(pca_plot)
  } else {
    ggsave(path, pca_plot, width = 5, height = 5)
    return(pca_plot)
  }
}


umap_plot <- function(
  df, metadata, 
  title = "UMAP Projection", 
  color_column = "study_accession", 
  quantitative_col_name = 'sample',
  path = "") {
  # Perform UMAP on the transposed data
  umap_result <- umap(t(na.omit(df)))
  
  # Convert the UMAP result into a data frame and merge with metadata
  umap_data <- umap_result$layout %>%
    as.data.frame() %>%
    setNames(c("X1", "X2")) %>% 
    rownames_to_column(quantitative_col_name) %>% 
    left_join(metadata, by = quantitative_col_name) %>%
    column_to_rownames(quantitative_col_name)

  plot_result <- ggplot(umap_data, aes_string(x = "X1", y = "X2", color = color_column)) +
    geom_point(aes_string(col = color_column), size = 0.7) +
    stat_ellipse(type = "t", level = 0.95) + # Add ellipses for each condition
    theme_minimal() +
    scale_color_lancet() + 
    labs(title = title, x = "UMAP 1", y = "UMAP 2") +
    guides(color = guide_legend(override.aes = list(size = 3))) # Ensure legend accurately represents centroids
  
    
    if (path == "") {
        return(plot_result)
  } else {
        ggsave(path, plot_result)
  }
}


# boxplot
boxplot_plot <- function(matrix, metadata_df, quantitativeColumnName, color_col, title, path="") {
  # Reshape data into long format
  long_data <- tidyr::gather(matrix, 
                             key = "file", value = "Intensity")
  merged_data <- merge(long_data, metadata_df, by.x = "file", by.y = quantitativeColumnName)
  
  # Log tranformed scale
  boxplot <- ggplot(merged_data, aes(x = file, y = Intensity, fill = .data[[color_col]])) + 
    geom_boxplot() +
    stat_summary(fun = mean, geom = "point", shape = 4, size = 3, color = "red") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1)) +
    # adjust fonsize for the x-axis
    theme(axis.text.x = element_text(size = 8)) +
    labs(title = title) 

  if(path == "") {
    return(boxplot)
  } else {
      ggsave(path, boxplot)
      return(boxplot)
  }
}


plotIntensityDensity <- function(
    intensities_df, metadata_df, quantitativeColumnName, colorColumnName, title, path=""
) {
  # Reshape the intensities_df from wide to long format
  long_intensities <- reshape2::melt(intensities_df, 
    variable.name = "Sample", value.name = "Intensity")
  
  # Adjust the merge function based on your metadata column names
  merged_data <- merge(long_intensities, metadata_df, by.x = "Sample", by.y = quantitativeColumnName)
  
  # Plot the data
  results <- ggplot(merged_data, aes(x = Intensity, color = .data[[colorColumnName]])) +  
    geom_density() +
    theme_minimal() +
    labs(title = paste(title, " by", colorColumnName),
         x = "Intensity",
         y = "Density")

  if(path == "") {
    return(results)
  } else {
    ggsave(path, results)
    return(results)
  }
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

