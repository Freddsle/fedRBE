library(variancePartition)


lmpv_plot <- function(
  data, metadata, title, 
  y_limits = c(0, 1), show_legend = TRUE, 
  form = NULL, max_yval = NULL,
  median_position = 0.05
){
  if (is.null(form)){
    form <- ~ Status + Dataset
  }

  # Extract variance explained by the model
  varPart <- fitExtractVarPartModel(na.omit(data), form, metadata)
  variance_col <- as.matrix(varPart[,1:2])

  # Convert data to long format for ggplot2
  df_long <- reshape2::melt(variance_col)

  # Calculate the medians for each group
  medians <- df_long %>%
    dplyr::group_by(Var2) %>%
    dplyr::summarize(median_value = median(value, na.rm = TRUE))

  # Create the boxplot using ggplot2
  res_plots <- ggplot(df_long, aes(x = Var2, y = value, fill = Var2)) +
    geom_boxplot() +
    labs(title = title,
         y = "Variance explained, %") +
    scale_fill_discrete(name = "Column") +
    theme_minimal() +
    ylim(y_limits) +
    # Add the median values to the plot with conditional positioning
    geom_text(
      data = medians,
      aes(
        x = Var2,
        y = median_value,
        label = sprintf("%.2f", median_value)
      ),
      # Conditional y position adjustment
      nudge_y = ifelse(medians$median_value > 0.8, -0.05, median_position), # Adjust y position conditionally
      nudge_x = -0.2, # Nudge text to the left
      size = 3, # Adjust text size as needed
      color = "black" # Text color
    ) +
     # no name for x-axis
    xlab("") 

  # Adjust y-axis limits if max_yval is provided
  if (!is.null(max_yval)){
    res_plots <- res_plots + 
      coord_cartesian(ylim = c(0, max_yval))
  }

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
    max_diff <- max(differences, na.rm = T)
    min_diff <- min(differences, na.rm = T)
    mean_diff <- mean(differences, na.rm = T)

    return(c(max_diff, min_diff, mean_diff))
}