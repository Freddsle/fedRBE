suppressPackageStartupMessages({
  library(readr)
  library(stringr)
})

rd_normalize_token <- function(x) {
  out <- as.character(x)
  out <- str_trim(out)
  out <- str_replace_all(out, "^\"|\"$", "")
  out <- str_replace_all(out, "^'|'$", "")
  out
}

rd_parse_csv_list <- function(value) {
  parts <- unlist(str_split(value, ","))
  parts <- trimws(parts)
  parts[nzchar(parts)]
}

rd_get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- "^--file="
  match <- grep(file_arg, args)
  if (length(match) > 0) {
    return(dirname(normalizePath(sub(file_arg, "", args[[match[[1]]]]), mustWork = TRUE)))
  }
  normalizePath(getwd(), mustWork = TRUE)
}

rd_dataset_configs <- function(repo_root) {
  data_root <- file.path(repo_root, "evaluation_data")
  list(
    proteomics = list(
      name = "proteomics",
      before_matrix = file.path(data_root, "proteomics", "before", "central_intensities_log_UNION.tsv"),
      before_sites_root = file.path(data_root, "proteomics", "before"),
      before_site_glob = "lab_*",
      before_site_matrix_file = "intensities_log_UNION.tsv",
      corrected_central = file.path(data_root, "proteomics", "after", "intensities_log_Rcorrected_UNION.tsv"),
      corrected_federated = file.path(data_root, "proteomics", "after", "FedApp_corrected_data_smpc.tsv")
    ),
    microarray = list(
      name = "microarray",
      before_matrix = file.path(data_root, "microarray", "before", "all_expression_UNION.tsv"),
      before_sites_root = file.path(data_root, "microarray", "before"),
      before_site_glob = "GSE*",
      before_site_matrix_file = "expr_for_correction_UNION.tsv",
      corrected_central = file.path(data_root, "microarray", "after", "central_corrected_UNION.tsv"),
      corrected_federated = file.path(data_root, "microarray", "after", "FedApp_corrected_data.tsv")
    ),
    microbiome = list(
      name = "microbiome",
      before_matrix = file.path(data_root, "microbiome", "before", "logmin_normalized_counts_5C.tsv"),
      before_sites_root = file.path(data_root, "microbiome", "before"),
      before_site_glob = "*",
      before_site_matrix_file = "UQnorm_log_counts_for_corr.tsv",
      corrected_central = file.path(
        data_root, "microbiome", "after", "normalized_logmin_counts_5centers_Rcorrected.tsv"
      ),
      corrected_federated = file.path(data_root, "microbiome", "after", "FedApp_corrected_data_smpc.tsv")
    )
  )
}

rd_choose_corrected_path <- function(cfg, source) {
  if (source == "central") {
    if (!file.exists(cfg$corrected_central)) {
      stop("Missing central corrected matrix: ", cfg$corrected_central, call. = FALSE)
    }
    return(cfg$corrected_central)
  }
  if (source == "federated") {
    if (!file.exists(cfg$corrected_federated)) {
      stop("Missing federated corrected matrix: ", cfg$corrected_federated, call. = FALSE)
    }
    return(cfg$corrected_federated)
  }
  if (file.exists(cfg$corrected_federated)) {
    return(cfg$corrected_federated)
  }
  if (file.exists(cfg$corrected_central)) {
    return(cfg$corrected_central)
  }
  stop(
    "No corrected matrix found for ", cfg$name,
    " (checked ", cfg$corrected_central, " and ", cfg$corrected_federated, ")",
    call. = FALSE
  )
}

rd_load_feature_matrix <- function(path) {
  df <- read_delim(path, delim = "\t", show_col_types = FALSE, progress = FALSE)
  if (ncol(df) < 2) {
    stop("Unexpected matrix shape for ", path, call. = FALSE)
  }
  colnames(df)[1] <- "rowname"
  df$rowname <- rd_normalize_token(df$rowname)
  df <- as.data.frame(df, check.names = FALSE)
  rownames(df) <- df$rowname
  df$rowname <- NULL
  colnames(df) <- rd_normalize_token(colnames(df))

  numeric_df <- as.data.frame(
    lapply(df, function(col) suppressWarnings(as.numeric(col))),
    check.names = FALSE
  )
  rownames(numeric_df) <- rownames(df)
  numeric_df <- numeric_df[rowSums(!is.na(numeric_df)) > 0, , drop = FALSE]
  if (nrow(numeric_df) == 0) {
    stop("Matrix became empty after dropping all-NA rows: ", path, call. = FALSE)
  }
  numeric_df
}

rd_load_before_matrix_from_sites <- function(cfg) {
  if (is.null(cfg$before_sites_root) || is.null(cfg$before_site_glob) || is.null(cfg$before_site_matrix_file)) {
    return(rd_load_feature_matrix(cfg$before_matrix))
  }

  all_dirs <- list.dirs(cfg$before_sites_root, full.names = TRUE, recursive = FALSE)
  if (length(all_dirs) == 0) {
    stop("No site directories found in: ", cfg$before_sites_root, call. = FALSE)
  }

  is_match <- grepl(glob2rx(cfg$before_site_glob), basename(all_dirs))
  site_dirs <- sort(all_dirs[is_match])
  if (length(site_dirs) == 0) {
    stop(
      "No site directories matching ", cfg$before_site_glob,
      " found in ", cfg$before_sites_root,
      call. = FALSE
    )
  }

  matrix_paths <- file.path(site_dirs, cfg$before_site_matrix_file)
  missing_paths <- matrix_paths[!file.exists(matrix_paths)]
  if (length(missing_paths) > 0) {
    stop(
      "Missing per-site before matrices:\n - ",
      paste(missing_paths, collapse = "\n - "),
      call. = FALSE
    )
  }

  mats <- lapply(matrix_paths, rd_load_feature_matrix)
  all_rows <- unique(unlist(lapply(mats, rownames)))
  aligned <- lapply(
    mats,
    function(m) {
      missing_rows <- setdiff(all_rows, rownames(m))
      if (length(missing_rows) > 0) {
        add <- matrix(
          NA_real_,
          nrow = length(missing_rows),
          ncol = ncol(m),
          dimnames = list(missing_rows, colnames(m))
        )
        m <- rbind(m, add)
      }
      m[all_rows, , drop = FALSE]
    }
  )
  merged <- do.call(cbind, aligned)
  dup_cols <- unique(colnames(merged)[duplicated(colnames(merged))])
  if (length(dup_cols) > 0) {
    stop(
      "Duplicate sample columns while merging per-site before matrices. Examples: ",
      paste(utils::head(dup_cols, 5), collapse = ", "),
      call. = FALSE
    )
  }
  as.data.frame(merged, check.names = FALSE)
}

rd_align_matrix_to_files <- function(matrix, file_order, label) {
  missing <- setdiff(file_order, colnames(matrix))
  if (length(missing) > 0) {
    stop(
      label, ": ", length(missing), " samples are missing in matrix columns. Examples: ",
      paste(utils::head(missing, 5), collapse = ", "),
      call. = FALSE
    )
  }
  matrix[, file_order, drop = FALSE]
}

rd_replace_na_with_zero <- function(df) {
  df[is.na(df)] <- 0
  df
}

rd_drop_rows_with_any_na <- function(df, label = "matrix") {
  total_rows <- nrow(df)
  out <- df[stats::complete.cases(df), , drop = FALSE]
  removed <- total_rows - nrow(out)
  message(
    "[", label, "] remove_na=TRUE dropped ", removed,
    " of ", total_rows, " feature rows containing NA"
  )
  if (nrow(out) == 0) {
    stop(
      label, ": all feature rows were removed by remove_na=TRUE. ",
      "No data left for analysis.",
      call. = FALSE
    )
  }
  out
}

rd_permute_values <- function(values) {
  if (length(values) <= 1) {
    return(list(values))
  }
  out <- list()
  idx <- 1
  for (i in seq_along(values)) {
    current <- values[[i]]
    rest <- values[-i]
    for (sub in rd_permute_values(rest)) {
      out[[idx]] <- c(current, sub)
      idx <- idx + 1
    }
  }
  out
}

rd_align_predictions_to_truth <- function(predicted, truth) {
  pred_chr <- as.character(predicted)
  truth_chr <- as.character(truth)
  mask <- !is.na(pred_chr) & !is.na(truth_chr)
  aligned <- rep(NA_character_, length(pred_chr))
  if (!any(mask)) {
    return(aligned)
  }

  pred_levels <- sort(unique(pred_chr[mask]))
  truth_levels <- sort(unique(truth_chr[mask]))

  mapping <- c()
  if (length(pred_levels) == length(truth_levels) && length(pred_levels) <= 7) {
    best_acc <- -1
    for (perm in rd_permute_values(truth_levels)) {
      candidate <- setNames(as.character(perm), pred_levels)
      mapped <- candidate[pred_chr[mask]]
      acc <- mean(mapped == truth_chr[mask])
      if (acc > best_acc) {
        best_acc <- acc
        mapping <- candidate
      }
    }
  } else {
    for (level in pred_levels) {
      votes <- truth_chr[mask][pred_chr[mask] == level]
      if (length(votes) == 0) {
        next
      }
      mapping[[level]] <- names(sort(table(votes), decreasing = TRUE))[1]
    }
  }

  aligned[mask] <- mapping[pred_chr[mask]]
  aligned
}
