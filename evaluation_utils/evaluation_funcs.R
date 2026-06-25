variance_projection_matrix <- function(design) {
  decomposition <- qr(design)
  rank <- decomposition$rank
  if (rank == 0) {
    return(matrix(0, nrow = nrow(design), ncol = nrow(design)))
  }
  q_matrix <- qr.Q(decomposition)[, seq_len(rank), drop = FALSE]
  tcrossprod(q_matrix)
}


fixed_effect_variance_partition <- function(data, metadata, form) {
  complete_data <- data[stats::complete.cases(data), , drop = FALSE]
  metadata <- metadata[colnames(complete_data), , drop = FALSE]
  model_terms <- attr(stats::terms(form), "term.labels")
  if (length(model_terms) == 0) {
    stop("Variance partitioning requires at least one model term.", call. = FALSE)
  }

  response_matrix <- t(as.matrix(complete_data))
  storage.mode(response_matrix) <- "double"
  centered <- sweep(response_matrix, 2, colMeans(response_matrix), check.margin = FALSE)
  total_ss <- colSums(centered^2)
  total_ss[total_ss <= .Machine$double.eps] <- NA_real_

  previous_projection <- variance_projection_matrix(stats::model.matrix(~ 1, data = metadata))
  term_values <- vector("list", length(model_terms))

  for (index in seq_along(model_terms)) {
    current_formula <- stats::reformulate(model_terms[seq_len(index)])
    current_projection <- variance_projection_matrix(stats::model.matrix(current_formula, data = metadata))
    projection_delta <- current_projection - previous_projection
    term_ss <- colSums(response_matrix * (projection_delta %*% response_matrix))
    term_fraction <- term_ss / total_ss
    term_fraction[!is.finite(term_fraction)] <- NA_real_
    term_values[[index]] <- pmax(term_fraction, 0, na.rm = FALSE)
    previous_projection <- current_projection
  }

  data.frame(
    feature = rep(rownames(complete_data), times = length(model_terms)),
    variable = rep(model_terms, each = nrow(complete_data)),
    value = unlist(term_values, use.names = FALSE),
    stringsAsFactors = FALSE
  )
}


variance_partition_long <- function(data, metadata, form) {
  if (requireNamespace("variancePartition", quietly = TRUE)) {
    var_part <- variancePartition::fitExtractVarPartModel(na.omit(data), form, metadata)
    model_terms <- setdiff(colnames(var_part), "Residuals")
    if (length(model_terms) == 0) {
      stop("No model terms available to plot after removing residual variance.", call. = FALSE)
    }
    table <- as.data.frame(var_part[, model_terms, drop = FALSE], check.names = FALSE)
    table$feature <- rownames(table)
    return(tidyr::pivot_longer(
      table,
      cols = dplyr::all_of(model_terms),
      names_to = "variable",
      values_to = "value"
    ))
  }

  fixed_effect_variance_partition(data, metadata, form)
}


lmpv_plot <- function(
  data, metadata, title = "Variance explained by the model",
  y_limits = c(0, 1), show_legend = TRUE,
  form = NULL, max_yval = NULL,
  median_position = 0.05,
  median_label_size = 3,
  only_table = FALSE,
  term_labels = NULL,
  fill_title = "Column"
){
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' is required for lmpv_plot().", call. = FALSE)
  }
  if (is.null(form)){
    form <- ~ Status + Dataset
  }

  model_terms <- attr(stats::terms(form), "term.labels")
  if (is.null(term_labels)) {
    term_labels <- stats::setNames(model_terms, model_terms)
  }
  df_long <- variance_partition_long(data, metadata, form)
  df_long$variable <- factor(df_long$variable, levels = model_terms, ordered = TRUE)
  if (!"Var1" %in% names(df_long)) {
    df_long$Var1 <- df_long$feature
  }
  if (!"Var2" %in% names(df_long)) {
    df_long$Var2 <- df_long$variable
  }

  if (isTRUE(only_table)) {
    return(df_long)
  }

  medians <- dplyr::group_by(df_long, variable)
  medians <- dplyr::summarize(medians, median_value = median(value, na.rm = TRUE), .groups = "drop")
  medians <- dplyr::mutate(
    medians,
    label_y = median_value + ifelse(median_value > 0.8, -0.05, median_position)
  )

  effective_y_limits <- y_limits
  if (!is.null(max_yval)) {
    effective_y_limits[2] <- max_yval
  }

  # Create the boxplot using ggplot2
  res_plots <- ggplot2::ggplot(df_long, ggplot2::aes(x = variable, y = value, fill = variable)) +
    ggplot2::geom_boxplot() +
    ggplot2::labs(title = title,
                  y = "Variance explained, %",
                  fill = fill_title) +
    ggplot2::scale_x_discrete(labels = term_labels) +
    ggplot2::scale_fill_discrete(name = fill_title, labels = term_labels) +
    ggplot2::theme_minimal() +
    ggplot2::coord_cartesian(ylim = effective_y_limits) +
    # Add the median values to the plot with conditional positioning
    ggplot2::geom_text(
      data = medians,
      ggplot2::aes(
        x = variable,
        y = label_y,
        label = sprintf("%.2f", median_value)
      ),
      nudge_x = -0.2,
      size = median_label_size,
      color = "black"
    ) +
     # no name for x-axis
    ggplot2::xlab("") 

  # Remove legend if specified
  if (!show_legend){
    res_plots <- res_plots + 
      ggplot2::theme(legend.position = "none")
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
