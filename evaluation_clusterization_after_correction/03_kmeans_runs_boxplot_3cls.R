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

truth_col <- "condition"
min_truth_classes <- 2L

permute_vec <- function(x) {
  if (length(x) <= 1) {
    return(list(x))
  }
  out <- list()
  for (i in seq_along(x)) {
    rest <- x[-i]
    for (p in permute_vec(rest)) {
      out[[length(out) + 1]] <- c(x[i], p)
    }
  }
  out
}

align_to_truth <- function(pred, truth, label = "pred") {
  mask <- !is.na(pred) & !is.na(truth)
  if (!any(mask)) {
    warning(paste("No overlapping non-NA values for", label))
    return(pred)
  }

  pred_chr <- as.character(pred)
  truth_chr <- as.character(truth)
  pred_levels <- sort(unique(pred_chr[mask]))
  truth_levels <- sort(unique(truth_chr[mask]))

  mapping <- NULL
  if (length(pred_levels) == length(truth_levels) && length(pred_levels) <= 7) {
    perms <- permute_vec(truth_levels)
    best_acc <- -1
    for (perm in perms) {
      candidate <- setNames(perm, pred_levels)
      mapped <- candidate[pred_chr[mask]]
      acc <- mean(mapped == truth_chr[mask])
      if (acc > best_acc) {
        best_acc <- acc
        mapping <- candidate
      }
    }
  } else {
    mapping <- setNames(rep(NA_character_, length(pred_levels)), pred_levels)
    for (pl in pred_levels) {
      votes <- truth_chr[mask][pred_chr[mask] == pl]
      if (length(votes) == 0) {
        next
      }
      mapping[pl] <- names(which.max(table(votes)))
    }
  }

  mapped_pred <- pred_chr
  mapped_pred[!is.na(pred_chr)] <- mapping[pred_chr[!is.na(pred_chr)]]
  mapped_pred
}

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

  true_labels <- factor(true_labels[mask])
  predicted_labels <- factor(predicted_labels[mask], levels = levels(true_labels))

  cm <- table(True = true_labels, Predicted = predicted_labels)
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

  accuracy <- sum(diag(cm)) / total

  precisions <- numeric(length(levels(true_labels)))
  recalls <- numeric(length(levels(true_labels)))
  f1s <- numeric(length(levels(true_labels)))

  for (i in seq_along(levels(true_labels))) {
    tp <- cm[i, i]
    fp <- sum(cm[, i]) - tp
    fn <- sum(cm[i, ]) - tp
    precision <- if ((tp + fp) == 0) 0 else tp / (tp + fp)
    recall <- if ((tp + fn) == 0) 0 else tp / (tp + fn)
    f1 <- if ((precision + recall) == 0) 0 else 2 * precision * recall / (precision + recall)
    precisions[i] <- precision
    recalls[i] <- recall
    f1s[i] <- f1
  }

  precision_macro <- mean(precisions)
  recall_macro <- mean(recalls)
  f1_macro <- mean(f1s)

  p_k <- rowSums(cm)
  t_k <- colSums(cm)
  c_val <- sum(diag(cm))
  n <- sum(cm)
  numerator <- (c_val * n) - sum(p_k * t_k)
  denominator <- sqrt((n^2 - sum(t_k^2)) * (n^2 - sum(p_k^2)))
  mcc <- if (denominator == 0) 0 else numerator / denominator

  ari <- mclust::adjustedRandIndex(
    as.character(true_labels),
    as.character(predicted_labels)
  )

  list(
    accuracy = accuracy,
    precision = precision_macro,
    recall = recall_macro,
    f1_score = f1_macro,
    mcc = mcc,
    ari = ari
  )
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
        BeforeC_Fed_3clusters = Fed_3clusters
      )
    after_fed <- after_fed %>%
      rename(
        AfterC_Fed_3clusters = Fed_3clusters
      )

    merged <- cntrl_res %>%
      inner_join(before_fed, by = key_cols) %>%
      inner_join(after_fed, by = key_cols) %>%
      mutate(
        BeforeC_Cntrl_3clusters = Before_CtrlKm_3clusters,
        AfterC_Cntrl_3clusters = Cor_CtrlKm_3clusters
      )

    if (!truth_col %in% colnames(merged)) {
      warning(
        paste(
          "Skipping", paste(mode, run_id, sep = "/"),
          "missing truth column:", truth_col
        )
      )
      next
    }
    truth_vals <- unique(merged[[truth_col]])
    truth_vals <- truth_vals[!is.na(truth_vals)]
    if (length(truth_vals) < min_truth_classes) {
      warning(
        paste(
          "Skipping", paste(mode, run_id, sep = "/"),
          truth_col, "has <", min_truth_classes, "classes"
        )
      )
      next
    }
    merged$Truth <- merged[[truth_col]]

    needed_cols <- c(
      "Truth",
      "BeforeC_Cntrl_3clusters",
      "AfterC_Cntrl_3clusters",
      "BeforeC_Fed_3clusters",
      "AfterC_Fed_3clusters"
    )
    if (!all(needed_cols %in% colnames(merged))) {
      warning(paste("Skipping", paste(mode, run_id, sep = "/"), "missing 3-cluster columns"))
      next
    }

    merged$BeforeC_Cntrl_3clusters <- align_to_truth(
      merged$BeforeC_Cntrl_3clusters,
      merged$Truth,
      paste(mode, run_id, "BeforeC_Cntrl_3clusters")
    )
    merged$AfterC_Cntrl_3clusters <- align_to_truth(
      merged$AfterC_Cntrl_3clusters,
      merged$Truth,
      paste(mode, run_id, "AfterC_Cntrl_3clusters")
    )
    merged$BeforeC_Fed_3clusters <- align_to_truth(
      merged$BeforeC_Fed_3clusters,
      merged$Truth,
      paste(mode, run_id, "BeforeC_Fed_3clusters")
    )
    merged$AfterC_Fed_3clusters <- align_to_truth(
      merged$AfterC_Fed_3clusters,
      merged$Truth,
      paste(mode, run_id, "AfterC_Fed_3clusters")
    )

    metrics_res_before_cntrl <- calculate_metrics(
      merged$Truth,
      merged$BeforeC_Cntrl_3clusters
    )
    metrics_res_before_fed <- calculate_metrics(
      merged$Truth,
      merged$BeforeC_Fed_3clusters
    )
    metrics_res_cntrl <- calculate_metrics(
      merged$Truth,
      merged$AfterC_Cntrl_3clusters
    )
    metrics_res_fed <- calculate_metrics(
      merged$Truth,
      merged$AfterC_Fed_3clusters
    )

    metrics_res <- data.frame(
      run = run_id,
      Method = c(
        paste0(mode, "_BC_Cntrl_3cls"),
        paste0(mode, "_BC_Fed_3cls"),
        paste0(mode, "_AC_Cntrl_3cls"),
        paste0(mode, "_AC_Fed_3cls")
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
    file = file.path(out_dir, "metrics_res_runs_3cls.tsv"),
    delim = "\t",
    col_names = TRUE
  )

  metrics_res_full <- rbind(metrics_res_full, mode_metrics)
}

plot_dir <- file.path("results")
dir.create(plot_dir, showWarnings = FALSE, recursive = TRUE)

debug_plot_drop <- TRUE
debug_drop_path <- file.path(plot_dir, "plot_dropped_rows_3cls.tsv")
debug_na_summary <- TRUE

p <- plot_rotation_metrics(
  metrics_res_full,
  debug_drop = debug_plot_drop,
  debug_path = debug_drop_path,
  na_summary = debug_na_summary
)
print(p)
ggsave(file.path(plot_dir, "metrics_runs_boxplot_3cls.png"), p, width = 13, height = 7.5, dpi = 320)

p_mccari <- plot_rotation_metrics(
  metrics_res_full,
  metrics = c("MCC", "ARI"),
  debug_drop = debug_plot_drop,
  debug_path = debug_drop_path,
  na_summary = debug_na_summary
)
print(p_mccari)
ggsave(
  file.path(plot_dir, "metrics_runs_boxplot_3cls_MCCARI.png"),
  p_mccari,
  width = 6,
  height = 7.5,
  dpi = 320
)
