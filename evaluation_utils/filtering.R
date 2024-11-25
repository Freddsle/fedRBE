library(tidyverse)
library(data.table)

#################################################################
# Filtering functions
#################################################################
filter_na_proteins <- function(dt, meta_data, quantitative_column_name) {
    cat('Filtering out features that have NAs in all columns\n')
    cat('\tBefore filtering:', dim(dt), '\n')
    # Filter out proteins that have NAs in all columns - 2 (only two non NA)
    dt <- dt[!rowSums(is.na(dt[, (meta_data[[quantitative_column_name]])])) == length(meta_data[[quantitative_column_name]]),]
    cat('\tAfter filtering:', dim(dt), '\n')
    return(dt)
}


filter_per_center <- function(intensities, metadata, quantitative_column_name, centers, center_column_name, min_samples = 2) {
  cat('Filtering by ', center_column_name, ' - two not-NA per', center_column_name,'\n')
  cat('\tBefore filtering:', dim(intensities), "\n")
  
  # Initialize a list to store the sample indices for each center
  center_samples <- list()
  
  # Loop through each center and extract relevant sample names
  for (center in centers) {
    center_samples[[center]] <- metadata[metadata[[center_column_name]] == center, ][[quantitative_column_name]]
  }
  # Determine rows with at least 2   non-NA values across each center's samples
  conditions <- sapply(center_samples, function(samples) {
    rowSums(!is.na(intensities[, samples, drop = FALSE])) >= min_samples
  })
  # Filter intensities where all conditions across centers are met
  filtered_intensities <- intensities[rowSums(conditions) == length(centers), ]

  cat('\tAfter filtering:', dim(filtered_intensities), "\n")
  return(filtered_intensities)
}