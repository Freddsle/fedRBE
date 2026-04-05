library(tidyverse)
library(data.table)

#################################################################
# Filtering functions
#################################################################
filter_per_center <- function(intensities, metadata, quantitative_column_name, centers, center_column_name, min_samples = 2, drop_row = TRUE) {
  message('Filtering by ', center_column_name, ' - min ', min_samples, ' not-NA per ', center_column_name)
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

  if (drop_row) {
    # Hard filter: remove entire row if condition is not met in ANY group.
    conditions <- lapply(center_samples, function(samples) {
      rowSums(!is.na(intensities[, samples, drop = FALSE])) >= min_samples
    })
    keep_rows <- Reduce(`&`, conditions)
    filtered_intensities <- intensities[keep_rows, , drop = FALSE]
  } else {
    # Soft mask: set values to NA only in groups that do not meet the threshold.
    # Features that pass in other groups keep their original values.
    # Rows are never removed; the caller is responsible for any all-NA cleanup.
    for (center_name in names(center_samples)) {
      samples <- center_samples[[center_name]]
      not_met <- rowSums(!is.na(intensities[, samples, drop = FALSE])) < min_samples
      intensities[not_met, samples] <- NA
    }
    filtered_intensities <- intensities
  }

  message('\tAfter filtering: ', paste(dim(filtered_intensities), collapse = " "))
  return(filtered_intensities)
}
