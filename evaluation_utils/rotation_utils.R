rotation_modes <- c("balanced", "mild_imbalanced", "strong_imbalanced")

report_plot_drops <- function(df, y_limits, context = "plot", debug_path = NULL) {
  if (is.null(df) || nrow(df) == 0) {
    return(invisible(NULL))
  }
  if (length(y_limits) != 2 || any(!is.finite(y_limits))) {
    return(invisible(NULL))
  }

  required <- intersect(
    c("value", "x_group", "condition", "approach", "imbalance", "metric"),
    colnames(df)
  )
  if (length(required) == 0) {
    return(invisible(NULL))
  }

  missing_mask <- !stats::complete.cases(df[, required, drop = FALSE])
  out_mask <- !is.na(df$value) & (df$value < y_limits[1] | df$value > y_limits[2])
  bad_mask <- missing_mask | out_mask

  if (!any(bad_mask)) {
    return(invisible(NULL))
  }

  reason <- ifelse(
    missing_mask & out_mask,
    "missing+outside",
    ifelse(missing_mask, "missing", "outside")
  )
  counts <- table(reason[bad_mask])

  if (!is.null(debug_path) && nzchar(debug_path)) {
    dir.create(dirname(debug_path), showWarnings = FALSE, recursive = TRUE)
    sample_cols <- intersect(
      c("run", "Method", "metric", "imbalance", "approach", "condition", "x_group", "value"),
      colnames(df)
    )
    drop_rows <- df[bad_mask, sample_cols, drop = FALSE]
    drop_rows$drop_reason <- reason[bad_mask]
    drop_rows$context <- context
    write_header <- !file.exists(debug_path)
    utils::write.table(
      drop_rows,
      file = debug_path,
      sep = "\t",
      row.names = FALSE,
      quote = FALSE,
      col.names = write_header,
      append = !write_header
    )
  }

  message(sprintf(
    "Plot debug (%s): dropping %d/%d rows. %s",
    context,
    sum(bad_mask),
    nrow(df),
    paste(paste0(names(counts), "=", counts), collapse = ", ")
  ))

  sample_cols <- intersect(
    c("run", "Method", "metric", "imbalance", "approach", "condition", "x_group", "value"),
    colnames(df)
  )
  if (length(sample_cols) > 0) {
    sample_rows <- utils::head(df[bad_mask, sample_cols, drop = FALSE], 20)
    message("Examples of dropped rows:")
    message(paste(capture.output(print(sample_rows)), collapse = "\n"))
  }

  invisible(NULL)
}

report_na_summary <- function(plot_df, context = "plot") {
  required <- c("imbalance", "x_group", "metric", "value")
  if (!all(required %in% colnames(plot_df))) {
    return(invisible(NULL))
  }

  summary <- plot_df %>%
    group_by(imbalance, x_group, metric) %>%
    summarise(
      total = dplyr::n(),
      na_count = sum(is.na(value)),
      na_pct = ifelse(total == 0, 0, round(100 * na_count / total, 2)),
      .groups = "drop"
    )

  if (nrow(summary) == 0) {
    return(invisible(NULL))
  }

  summary <- summary %>% filter(na_count > 0)
  if (nrow(summary) == 0) {
    message(sprintf("NA summary (%s): complete case", context))
    return(invisible(NULL))
  }

  for (imb in unique(summary$imbalance)) {
    message(sprintf("NA summary (%s) - %s", context, imb))
    print(summary %>% filter(imbalance == imb))
  }

  invisible(NULL)
}

build_rotation_plot_df <- function(metrics_res_full, metrics = NULL) {
  if (is.null(metrics_res_full) || nrow(metrics_res_full) == 0) {
    stop("No metrics collected across modes; nothing to plot.")
  }

  needed_cols <- c("run", "Method", "Accuracy", "Precision", "Recall", "F1_Score", "MCC", "ARI")
  stopifnot(all(needed_cols %in% colnames(metrics_res_full)))
  metric_levels <- c("Accuracy", "Precision", "Recall", "F1_Score", "MCC", "ARI")

  parsed <- metrics_res_full %>%
    mutate(
      Method = str_replace_all(as.character(Method), "\\s+", "")
    ) %>%
    tidyr::extract(
      Method,
      into = c("imbalance", "approach", "condition", "clusters"),
      regex = "^(balanced|mild_imbalanced|strong_imbalanced)_(BC|AC)_(Cntrl|Fed)_(\\d+cls)$",
      remove = FALSE
    )

  bad_methods <- parsed %>%
    filter(is.na(imbalance) | is.na(approach) | is.na(condition) | is.na(clusters)) %>%
    distinct(Method) %>%
    pull(Method)

  if (length(bad_methods) > 0) {
    stop(
      "These Method strings did not match the expected pattern:\n",
      paste0(" - ", bad_methods, collapse = "\n"),
      "\nExpected: (balanced|mild_imbalanced|strong_imbalanced)_(BC|AC)_(Cntrl|Fed)_(2cls|3cls|...)"
    )
  }

  plot_df <- parsed %>%
    pivot_longer(
      cols = all_of(metric_levels),
      names_to = "metric",
      values_to = "value"
    ) %>%
    mutate(
      imbalance = factor(imbalance, levels = rotation_modes),
      approach = factor(approach, levels = c("BC", "AC")),
      condition = factor(condition, levels = c("Cntrl", "Fed")),
      metric = factor(metric, levels = metric_levels),
      x_group = factor(
        paste0(approach, "__", condition),
        levels = c("BC__Cntrl", "BC__Fed", "AC__Cntrl", "AC__Fed")
      )
    )

  if (!is.null(metrics)) {
    metrics <- as.character(metrics)
    missing_metrics <- setdiff(metrics, metric_levels)
    if (length(missing_metrics) > 0) {
      stop(
        "Requested metrics not available: ",
        paste(missing_metrics, collapse = ", "),
        "\nAvailable: ",
        paste(metric_levels, collapse = ", ")
      )
    }
    plot_df <- plot_df %>%
      filter(metric %in% metrics) %>%
      mutate(metric = factor(metric, levels = metrics))
  }

  plot_df %>%
    filter(!is.na(value))
}

make_rotation_panel_plot <- function(df,
                                     facet_rows = TRUE,
                                     y_limits = NULL,
                                     y_breaks = NULL,
                                     debug_drop = FALSE,
                                     context = NULL,
                                     debug_path = NULL) {
  if (is.null(y_limits)) {
    y_limits <- range(df$value, na.rm = TRUE)
  }
  if (!all(is.finite(y_limits))) {
    stop("No finite metric values to plot.")
  }
  if (is.null(y_breaks)) {
    y_breaks <- scales::pretty_breaks(n = 5)(y_limits)
  }
  if (debug_drop) {
    if (is.null(context) || !nzchar(context)) {
      context <- "panel"
    }
    report_plot_drops(df, y_limits, context = context, debug_path = debug_path)
  }

  g <- ggplot(
    df,
    aes(
      x = x_group,
      y = value,
      fill = approach,
      color = approach,
      group = x_group
    )
  ) +
    geom_boxplot(
      width = 0.55,
      alpha = 0.4,
      outlier.shape = NA
    ) +
    geom_point(
      aes(shape = condition),
      position = position_jitter(width = 0.18),
      size = 1.6,
      alpha = 0.75,
      stroke = 0.4
    ) +
    scale_x_discrete(labels = c(
      BC__Cntrl = "Cntrl\nBC",
      BC__Fed = "Fed\nBC",
      AC__Cntrl = "Cntrl\nAC",
      AC__Fed = "Fed\nAC"
    )) +
    scale_y_continuous(
      limits = y_limits,
      breaks = y_breaks,
      labels = label_number(accuracy = 0.01),
      expand = expansion(mult = 0)
    ) +
    labs(x = NULL, y = "Score", color = "Approach", shape = "Condition") +
    theme_minimal(base_size = 12) +
    guides(fill = "none") +
    theme(
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_blank(),
      strip.text = element_text(face = "bold"),
      legend.position = "top"
    )

  if (facet_rows) {
    g <- g + facet_grid(imbalance ~ metric)
  } else {
    g <- g + facet_grid(. ~ metric)
  }
  g
}

compose_rotation_metrics_plot <- function(plot_df, debug_drop = FALSE, debug_path = NULL, debug_tag = NULL) {
  y_limits <- range(plot_df$value, na.rm = TRUE)
  if (!all(is.finite(y_limits))) {
    stop("No finite metric values to plot.")
  }
  y_breaks <- scales::pretty_breaks(n = 5)(y_limits)
  if (is.null(debug_tag) || !nzchar(debug_tag)) {
    debug_tag <- "plot"
  }

  p_bal <- make_rotation_panel_plot(
    plot_df %>% filter(imbalance == "balanced"),
    facet_rows = FALSE,
    y_limits = y_limits,
    y_breaks = y_breaks,
    debug_drop = debug_drop,
    context = paste0("balanced:", debug_tag),
    debug_path = debug_path
  ) +
    theme(strip.text.y = element_blank()) +
    labs(y = "Score\n(Balanced)")

  p_mild <- make_rotation_panel_plot(
    plot_df %>% filter(imbalance == "mild_imbalanced"),
    facet_rows = FALSE,
    y_limits = y_limits,
    y_breaks = y_breaks,
    debug_drop = debug_drop,
    context = paste0("mild_imbalanced:", debug_tag),
    debug_path = debug_path
  ) +
    theme(strip.text.y = element_blank()) +
    labs(y = "Score\n(Mildly Imbalanced)")

  p_strong <- make_rotation_panel_plot(
    plot_df %>% filter(imbalance == "strong_imbalanced"),
    facet_rows = FALSE,
    y_limits = y_limits,
    y_breaks = y_breaks,
    debug_drop = debug_drop,
    context = paste0("strong_imbalanced:", debug_tag),
    debug_path = debug_path
  ) +
    theme(strip.text.y = element_blank()) +
    labs(y = "Score\n(Strongly Imbalanced)")

  separator <- wrap_elements(
    full = rectGrob(gp = gpar(col = NA, fill = "grey60"))
  ) + theme_void()

  (p_bal / separator / p_mild / separator / p_strong) +
    plot_layout(heights = c(1, 0.035, 1, 0.035, 1), guides = "collect") &
    theme(legend.position = "top")
}

plot_rotation_metrics <- function(metrics_res_full,
                                  metrics = NULL,
                                  debug_drop = FALSE,
                                  debug_path = NULL,
                                  debug_tag = NULL,
                                  na_summary = FALSE) {
  plot_df <- build_rotation_plot_df(metrics_res_full, metrics = metrics)
  if (na_summary) {
    report_na_summary(
      plot_df,
      context = if (is.null(metrics)) "all_metrics" else paste(metrics, collapse = "+")
    )
  }
  if (is.null(debug_tag) || !nzchar(debug_tag)) {
    debug_tag <- if (is.null(metrics)) "all_metrics" else paste(metrics, collapse = "+")
  }
  compose_rotation_metrics_plot(
    plot_df,
    debug_drop = debug_drop,
    debug_path = debug_path,
    debug_tag = debug_tag
  )
}
