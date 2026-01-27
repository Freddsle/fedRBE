suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(stringr)
  library(ggplot2)
  library(scales)
  library(patchwork)
  library(grid)
  library(readr)
  library(mclust)
})

source(file.path("..", "evaluation_utils", "rotation_utils.R"))

calculate_metrics <- function(true_labels, predicted_labels) {
  mask <- !is.na(true_labels) & !is.na(predicted_labels)
  if (!any(mask)) {
    return(list(
      accuracy = NA_real_,
      precision = NA_real_,
      recall = NA_real_,
      f1_score = NA_real_,
      mcc = NA_real_,
      ari = NA_real_
    ))
  }

  true_labels <- factor(true_labels[mask], levels = c(0, 1))
  predicted_labels <- factor(predicted_labels[mask], levels = c(0, 1))

  cm <- table(True = true_labels, Predicted = predicted_labels)
  cm <- cm[levels(true_labels), levels(predicted_labels), drop = FALSE]

  TP <- as.numeric(cm[2, 2])
  TN <- as.numeric(cm[1, 1])
  FP <- as.numeric(cm[1, 2])
  FN <- as.numeric(cm[2, 1])

  total <- sum(cm)
  if (total == 0) {
    return(list(
      accuracy = NA_real_,
      precision = NA_real_,
      recall = NA_real_,
      f1_score = NA_real_,
      mcc = NA_real_,
      ari = NA_real_
    ))
  }

  accuracy <- (TP + TN) / total
  precision <- if (TP + FP == 0) 0 else TP / (TP + FP)
  recall <- if (TP + FN == 0) 0 else TP / (TP + FN)
  f1_score <- if (precision + recall == 0) 0 else 2 * precision * recall / (precision + recall)

  denom <- sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
  mcc <- if (denom == 0) NA_real_ else (TP * TN - FP * FN) / denom

  ari <- mclust::adjustedRandIndex(
    as.character(true_labels),
    as.character(predicted_labels)
  )

  list(
    accuracy = accuracy,
    precision = precision,
    recall = recall,
    f1_score = f1_score,
    mcc = mcc,
    ari = ari
  )
}

normalize_binary <- function(x, label = "value") {
  if (all(is.na(x))) {
    warning(paste("All NA values for", label))
    return(x)
  }
  uniq <- sort(unique(x[!is.na(x)]))
  if (all(uniq %in% c(0, 1))) {
    return(x)
  }
  if (all(uniq %in% c(1, 2))) {
    return(as.integer(as.factor(x)) - 1L)
  }
  warning(paste("Normalizing non-binary labels for", label, ":", paste(uniq, collapse = ",")))
  as.integer(as.factor(x)) - 1L
}

align_to_truth <- function(pred, truth, label = "pred") {
  mask <- !is.na(pred) & !is.na(truth)
  if (!any(mask)) {
    warning(paste("No overlapping non-NA values for", label))
    return(pred)
  }
  acc_as_is <- mean(pred[mask] == truth[mask])
  acc_flip <- mean((1L - pred[mask]) == truth[mask])
  if (is.nan(acc_as_is) || is.nan(acc_flip)) {
    warning(paste("Unable to align", label))
    return(pred)
  }
  if (acc_flip > acc_as_is) {
    pred <- 1L - pred
  }
  pred
}

modes <- rotation_modes
metrics_res_full <- NULL

for (mode in modes) {
  run_dir <- file.path(
    "..", "evaluation_data", "simulated_rotation", mode, "after", "kmeans_res", "runs"
  )
  run_files <- list.files(
    run_dir,
    pattern = "_metadata_cntrl_kmeans_res\\.tsv$",
    full.names = FALSE
  )
  run_ids <- str_replace(run_files, "_metadata_cntrl_kmeans_res\\.tsv$", "")
  run_ids <- sort(run_ids)

  if (length(run_ids) == 0) {
    warning(paste("No runs found for mode:", mode))
    next
  }

  mode_metrics <- NULL

  for (run_id in run_ids) {
    cntrl_path <- file.path(run_dir, paste0(run_id, "_metadata_cntrl_kmeans_res.tsv"))
    before_fed_path <- file.path(run_dir, paste0(run_id, "_metadata_before_fedclusters.tsv"))
    after_fed_path <- file.path(run_dir, paste0(run_id, "_metadata_after_fedclusters.tsv"))

    missing <- c(cntrl_path, before_fed_path, after_fed_path)
    missing <- missing[!file.exists(missing)]
    if (length(missing) > 0) {
      warning(
        paste(
          "Skipping", paste(mode, run_id, sep = "/"),
          "missing:", paste(basename(missing), collapse = ", ")
        )
      )
      next
    }

    cntrl_res <- read_delim(
      cntrl_path,
      delim = "\t",
      col_names = TRUE,
      show_col_types = FALSE
    )
    before_fed <- read_delim(
      before_fed_path,
      delim = "\t",
      col_names = TRUE,
      show_col_types = FALSE
    )
    after_fed <- read_delim(
      after_fed_path,
      delim = "\t",
      col_names = TRUE,
      show_col_types = FALSE
    )

    key_cols <- c("file", "condition", "lab")
    if (!all(key_cols %in% colnames(cntrl_res)) ||
        !all(key_cols %in% colnames(before_fed)) ||
        !all(key_cols %in% colnames(after_fed))) {
      warning(paste("Skipping", paste(mode, run_id, sep = "/"), "missing join columns"))
      next
    }

    before_fed <- before_fed %>%
      rename(
        BeforeC_Fed_2clusters = Fed_2clusters,
        BeforeC_Fed_3clusters = Fed_3clusters
      )
    after_fed <- after_fed %>%
      rename(
        AfterC_Fed_2clusters = Fed_2clusters,
        AfterC_Fed_3clusters = Fed_3clusters
      )

    merged <- cntrl_res %>%
      inner_join(before_fed, by = key_cols) %>%
      inner_join(after_fed, by = key_cols) %>%
      mutate(
        A = if_else(condition == "A", 0L, 1L),
        BeforeC_Cntrl_2clusters = Before_CtrlKm_2clusters,
        AfterC_Cntrl_2clusters = Cor_CtrlKm_2clusters
      )

    if (!all(c("BeforeC_Cntrl_2clusters", "AfterC_Cntrl_2clusters") %in% colnames(merged))) {
      warning(paste("Skipping", paste(mode, run_id, sep = "/"), "missing central K=2 columns"))
      next
    }

    merged$BeforeC_Cntrl_2clusters <- normalize_binary(
      merged$BeforeC_Cntrl_2clusters,
      paste(mode, run_id, "BeforeC_Cntrl_2clusters")
    )
    merged$AfterC_Cntrl_2clusters <- normalize_binary(
      merged$AfterC_Cntrl_2clusters,
      paste(mode, run_id, "AfterC_Cntrl_2clusters")
    )

    merged$BeforeC_Fed_2clusters <- normalize_binary(
      merged$BeforeC_Fed_2clusters,
      paste(mode, run_id, "BeforeC_Fed_2clusters")
    )
    merged$AfterC_Fed_2clusters <- normalize_binary(
      merged$AfterC_Fed_2clusters,
      paste(mode, run_id, "AfterC_Fed_2clusters")
    )

    merged$BeforeC_Cntrl_2clusters <- align_to_truth(
      merged$BeforeC_Cntrl_2clusters,
      merged$A,
      paste(mode, run_id, "BeforeC_Cntrl_2clusters")
    )
    merged$AfterC_Cntrl_2clusters <- align_to_truth(
      merged$AfterC_Cntrl_2clusters,
      merged$A,
      paste(mode, run_id, "AfterC_Cntrl_2clusters")
    )
    merged$BeforeC_Fed_2clusters <- align_to_truth(
      merged$BeforeC_Fed_2clusters,
      merged$A,
      paste(mode, run_id, "BeforeC_Fed_2clusters")
    )
    merged$AfterC_Fed_2clusters <- align_to_truth(
      merged$AfterC_Fed_2clusters,
      merged$A,
      paste(mode, run_id, "AfterC_Fed_2clusters")
    )

    metrics_res_before_cntrl <- calculate_metrics(
      merged$A,
      merged$BeforeC_Cntrl_2clusters
    )
    metrics_res_before_fed <- calculate_metrics(
      merged$A,
      merged$BeforeC_Fed_2clusters
    )
    metrics_res_cntrl <- calculate_metrics(
      merged$A,
      merged$AfterC_Cntrl_2clusters
    )
    metrics_res_fed <- calculate_metrics(
      merged$A,
      merged$AfterC_Fed_2clusters
    )

    metrics_res <- data.frame(
      run = run_id,
      Method = c(
        paste0(mode, "_BC_Cntrl_2cls"),
        paste0(mode, "_BC_Fed_2cls"),
        paste0(mode, "_AC_Cntrl_2cls"),
        paste0(mode, "_AC_Fed_2cls")
      ),
      Accuracy = c(
        metrics_res_before_cntrl$accuracy,
        metrics_res_before_fed$accuracy,
        metrics_res_cntrl$accuracy,
        metrics_res_fed$accuracy
      ),
      Precision = c(
        metrics_res_before_cntrl$precision,
        metrics_res_before_fed$precision,
        metrics_res_cntrl$precision,
        metrics_res_fed$precision
      ),
      Recall = c(
        metrics_res_before_cntrl$recall,
        metrics_res_before_fed$recall,
        metrics_res_cntrl$recall,
        metrics_res_fed$recall
      ),
      F1_Score = c(
        metrics_res_before_cntrl$f1_score,
        metrics_res_before_fed$f1_score,
        metrics_res_cntrl$f1_score,
        metrics_res_fed$f1_score
      ),
      MCC = c(
        metrics_res_before_cntrl$mcc,
        metrics_res_before_fed$mcc,
        metrics_res_cntrl$mcc,
        metrics_res_fed$mcc
      ),
      ARI = c(
        metrics_res_before_cntrl$ari,
        metrics_res_before_fed$ari,
        metrics_res_cntrl$ari,
        metrics_res_fed$ari
      )
    )

    mode_metrics <- rbind(mode_metrics, metrics_res)
  }

  if (is.null(mode_metrics) || nrow(mode_metrics) == 0) {
    warning(paste("No metrics computed for mode:", mode))
    next
  }

  out_dir <- file.path(
    "..", "evaluation_data", "simulated_rotation", mode, "after", "kmeans_res"
  )
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  write_delim(
    mode_metrics,
    file = file.path(out_dir, "metrics_res_runs.tsv"),
    delim = "\t",
    col_names = TRUE
  )

  metrics_res_full <- rbind(metrics_res_full, mode_metrics)
}

plot_dir <- file.path("results")
dir.create(plot_dir, showWarnings = FALSE, recursive = TRUE)

# save full metrics result into a file in the results directory
write_delim(
  metrics_res_full,
  file = file.path(plot_dir, "metrics_res_runs_full.tsv"),
  delim = "\t",
  col_names = TRUE
)

debug_plot_drop <- TRUE
debug_drop_path <- file.path(plot_dir, "plot_dropped_rows_2cls.tsv")
debug_na_summary <- TRUE

p <- plot_rotation_metrics(
  metrics_res_full,
  debug_drop = debug_plot_drop,
  debug_path = debug_drop_path,
  na_summary = debug_na_summary
)
print(head(metrics_res_full))
print(p)
ggsave(file.path(plot_dir, "metrics_runs_boxplot.png"), p, width = 13, height = 7.5, dpi = 320)

p_mccari <- plot_rotation_metrics(
  metrics_res_full,
  metrics = c("MCC", "ARI"),
  debug_drop = debug_plot_drop,
  debug_path = debug_drop_path,
  na_summary = debug_na_summary
)
print(p_mccari)
ggsave(
  file.path(plot_dir, "metrics_runs_boxplot_MCCARI.png"),
  p_mccari,
  width = 6,
  height = 7.5,
  dpi = 320
)
