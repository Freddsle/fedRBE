library(tidyverse)
library(data.table)

#################################################################
# Filtering functions
#################################################################
filter_per_center <- function(
  intensities,
  metadata,
  quantitative_column_name,
  centers,
  center_column_name,
  min_samples = 2,
  min_fraction = NULL,
  drop_row = TRUE
) {
  if (!is.null(min_fraction) && (min_fraction < 0 || min_fraction > 1)) {
    stop("min_fraction must be between 0 and 1.")
  }

  threshold_label <- if (is.null(min_fraction)) {
    paste0("min ", min_samples, " not-NA per ", center_column_name)
  } else {
    paste0(
      "min ",
      scales::percent(min_fraction, accuracy = 1),
      " not-NA per ",
      center_column_name,
      " (at least ",
      min_samples,
      " sample(s))"
    )
  }
  message("Filtering by ", center_column_name, " - ", threshold_label)
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

  center_thresholds <- vapply(
    center_samples,
    function(samples) {
      if (is.null(min_fraction)) {
        as.integer(min_samples)
      } else {
        as.integer(max(min_samples, ceiling(length(samples) * min_fraction)))
      }
    },
    integer(1)
  )

  if (drop_row) {
    # Hard filter: remove entire row if condition is not met in ANY group.
    conditions <- Map(function(samples, threshold) {
      rowSums(!is.na(intensities[, samples, drop = FALSE])) >= threshold
    }, center_samples, center_thresholds)
    keep_rows <- Reduce(`&`, conditions)
    filtered_intensities <- intensities[keep_rows, , drop = FALSE]
  } else {
    # Soft mask: set values to NA only in groups that do not meet the threshold.
    # Features that pass in other groups keep their original values.
    # Rows are never removed; the caller is responsible for any all-NA cleanup.
    for (center_name in names(center_samples)) {
      samples <- center_samples[[center_name]]
      threshold <- center_thresholds[[center_name]]
      not_met <- rowSums(!is.na(intensities[, samples, drop = FALSE])) < threshold
      intensities[not_met, samples] <- NA
    }
    filtered_intensities <- intensities
  }

  message('\tAfter filtering: ', paste(dim(filtered_intensities), collapse = " "))
  return(filtered_intensities)
}

fedrbe_min_samples <- function(num_batches, covariates = character(), min_samples = 0) {
  as.integer(max(num_batches + length(covariates) + 1, min_samples))
}

filter_fedrbe_client <- function(
  intensities,
  metadata,
  quantitative_column_name,
  batch_column_name,
  num_batches,
  covariates = character(),
  min_samples = 0,
  client_column_name = "lab"
) {
  effective_min_samples <- fedrbe_min_samples(
    num_batches = num_batches,
    covariates = covariates,
    min_samples = min_samples
  )
  message("Applying fedRBE-like client filtering with min_samples = ", effective_min_samples)

  filtered_intensities <- filter_per_center(
    intensities = intensities,
    metadata = metadata,
    quantitative_column_name = quantitative_column_name,
    centers = unique(metadata[[client_column_name]]),
    center_column_name = client_column_name,
    min_samples = 1,
    drop_row = TRUE
  )

  filtered_intensities <- filter_per_center(
    intensities = filtered_intensities,
    metadata = metadata,
    quantitative_column_name = quantitative_column_name,
    centers = unique(metadata[[batch_column_name]]),
    center_column_name = batch_column_name,
    min_samples = 2,
    drop_row = FALSE
  )

  filtered_intensities <- filter_per_center(
    intensities = filtered_intensities,
    metadata = metadata,
    quantitative_column_name = quantitative_column_name,
    centers = unique(metadata[[client_column_name]]),
    center_column_name = client_column_name,
    min_samples = effective_min_samples,
    drop_row = FALSE
  )

  return(filtered_intensities)
}
