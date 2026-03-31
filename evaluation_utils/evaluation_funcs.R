library(variancePartition)


lmpv_plot <- function(
  data, metadata, title = "Variance explained by the model", 
  y_limits = c(0, 1), show_legend = TRUE, 
  form = NULL, max_yval = NULL,
  median_position = 0.05,
  only_table = FALSE
){
  if (is.null(form)){
    form <- ~ Status + Dataset
  }

  # Extract variance explained by the model
  varPart <- fitExtractVarPartModel(na.omit(data), form, metadata)
  model_terms <- setdiff(colnames(varPart), "Residuals")
  if (length(model_terms) == 0) {
    stop("No model terms available to plot after removing residual variance.", call. = FALSE)
  }
  variance_col <- as.matrix(varPart[, model_terms, drop = FALSE])

  # Convert data to long format for ggplot2
  df_long <- reshape2::melt(variance_col)

  if (isTRUE(only_table)) {
    return(df_long)
  }

  # Calculate the medians for each group
  medians <- df_long %>%
    dplyr::group_by(Var2) %>%
    dplyr::summarize(median_value = median(value, na.rm = TRUE), .groups = "drop") %>%
    dplyr::mutate(
      label_y = median_value + ifelse(median_value > 0.8, -0.05, median_position)
    )

  effective_y_limits <- y_limits
  if (!is.null(max_yval)) {
    effective_y_limits[2] <- max_yval
  }

  # Create the boxplot using ggplot2
  res_plots <- ggplot(df_long, aes(x = Var2, y = value, fill = Var2)) +
    geom_boxplot() +
    labs(title = title,
         y = "Variance explained, %") +
    scale_fill_discrete(name = "Column") +
    theme_minimal() +
    coord_cartesian(ylim = effective_y_limits) +
    # Add the median values to the plot with conditional positioning
    geom_text(
      data = medians,
      aes(
        x = Var2,
        y = label_y,
        label = sprintf("%.2f", median_value)
      ),
      nudge_x = -0.2, # Nudge text to the left
      size = 3, # Adjust text size as needed
      color = "black" # Text color
    ) +
     # no name for x-axis
    xlab("") 

  # Remove legend if specified
  if (!show_legend){
    res_plots <- res_plots + 
      theme(legend.position = "none")
  }

  return(res_plots)
}


calculated_differences <- function(central_results, fed_results){
    differences <- as.matrix(abs(central_results - fed_results))
    
    # max min mean 
    max_diff <- max(differences, na.rm = TRUE)
    min_diff <- min(differences, na.rm = TRUE)
    mean_diff <- mean(differences, na.rm = TRUE)

    return(c(max_diff = max_diff, min_diff = min_diff, mean_diff = mean_diff))
}
