library(tidyverse)
library(data.table)

#################################################################
# Filtering functions
#################################################################
filter_na_proteins <- function(dt, meta_data, quantitative_column_name) {
    message('Filtering out features that have NAs in all columns')
    message('\tBefore filtering: ', paste(dim(dt), collapse = " "))
    sample_columns <- meta_data[[quantitative_column_name]]
    all_na_rows <- rowSums(is.na(dt[, sample_columns, drop = FALSE])) == length(sample_columns)
    # Filter out proteins that have NAs in all columns - 2 (only two non NA)
    dt <- dt[!all_na_rows, , drop = FALSE]
    message('\tAfter filtering: ', paste(dim(dt), collapse = " "))
    return(dt)
}


filter_per_center <- function(intensities, metadata, quantitative_column_name, centers, center_column_name, min_samples = 2) {
  message('Filtering by ', center_column_name, ' - two not-NA per ', center_column_name)
  message('\tBefore filtering: ', paste(dim(intensities), collapse = " "))
  if (length(centers) == 0) {
    return(intensities)
  }

  center_samples <- stats::setNames(
    lapply(centers, function(center) {
      metadata[metadata[[center_column_name]] == center, ][[quantitative_column_name]]
    }),
    centers
  )

  # Determine rows with at least 2   non-NA values across each center's samples
  conditions <- lapply(center_samples, function(samples) {
    rowSums(!is.na(intensities[, samples, drop = FALSE])) >= min_samples
  })

  # Filter intensities where all conditions across centers are met
  keep_rows <- Reduce(`&`, conditions)
  filtered_intensities <- intensities[keep_rows, , drop = FALSE]

  message('\tAfter filtering: ', paste(dim(filtered_intensities), collapse = " "))
  return(filtered_intensities)
}
