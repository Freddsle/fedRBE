# Simulated 30-run evaluation: FedSim-versus-R correction differences and
# variance partitioning aggregated over the 30 runs of each imbalance scenario.
# Run from the repository root:
#   pixi run Rscript evaluation/evaluation_simulated_30runs.R
#
# Differences are recalculated for scenarios whose
# evaluation_data/simulated/<mode>/after/runs/ hold all 30 FedSim and R corrections.
# The variance-partitioning plots in evaluation/eval_simulation/lmpv_plots/ read the
# committed summary cache evaluation/eval_simulation/precomputed/; without it,
# before-correction tables are computed from before/intermediate/ and the cache is
# built from the per-run tables in evaluation/eval_simulation/<mode>/lmvp/.

library(tidyverse)
library(data.table)
library(parallel)
library(patchwork)

source(file.path("evaluation_utils", "evaluation_funcs.R"))

# Load data

path_to_data <- file.path("evaluation_data", "simulated")
root_dir <- file.path("evaluation", "eval_simulation")
n_runs <- 30
modes <- c("balanced", "mild_imbalanced", "strong_imbalanced")

run_path <- function(mode, run, method) {
    file.path(path_to_data, mode, "after", "runs", paste0(run, "_", method, "_corrected.tsv"))
}

run_inputs_available <- function(mode) {
    all(vapply(1:n_runs, function(j) {
        file.exists(run_path(mode, j, "FedSim")) &&
            file.exists(run_path(mode, j, "R"))
    }, logical(1)))
}

modes_with_run_inputs <- modes[vapply(modes, run_inputs_available, logical(1))]
missing_run_input_modes <- setdiff(modes, modes_with_run_inputs)
if (length(missing_run_input_modes) > 0) {
    message(
        "Skipping raw 30-run recalculation for modes missing after/runs inputs: ",
        paste(missing_run_input_modes, collapse = ", "),
        ". Existing eval_simulation summary tables can still be used."
    )
}

read_simulated_metadata <- function(mode) {
    metadata_path <- file.path(path_to_data, mode, "all_metadata.tsv")
    first_line <- readLines(metadata_path, n = 1, warn = FALSE)
    header_names <- strsplit(first_line, "\t", fixed = TRUE)[[1]]
    header_names <- gsub('^"|"$', "", header_names)

    first_data <- fread(metadata_path, sep = "\t", header = FALSE, skip = 1, nrows = 1)
    if (length(header_names) == 3 && ncol(first_data) == 4) {
        metadata <- fread(
            metadata_path,
            sep = "\t",
            header = FALSE,
            skip = 1,
            col.names = c("metadata_row_id", header_names)
        )
        if (!identical(metadata$metadata_row_id, metadata$file)) {
            stop("Metadata row id differs from file column for mode: ", mode)
        }
        metadata[, metadata_row_id := NULL]
    } else {
        metadata <- fread(metadata_path, sep = "\t")
    }

    required_columns <- c("file", "condition", "lab")
    missing_columns <- setdiff(required_columns, names(metadata))
    if (length(missing_columns) > 0) {
        stop(
            "Missing metadata columns for mode ", mode, ": ",
            paste(missing_columns, collapse = ", ")
        )
    }

    metadata <- metadata[, ..required_columns]
    setDT(metadata)
    metadata
}

# Calculate raw correction differences: the data.table loop below calculates raw
# FedSim-versus-R differences for all 30 runs in parallel.

# Cache metadata once for all modes
metadata_list <- lapply(modes_with_run_inputs, function(mode) {
  list(mode = mode, metadata = read_simulated_metadata(mode))
})

detected_cores <- parallel::detectCores(logical = FALSE)
if (is.na(detected_cores)) {
  detected_cores <- 1
}
run_cores <- min(5, detected_cores, n_runs)

for (metadata_info in metadata_list) {
  mode <- metadata_info$mode
  metadata <- metadata_info$metadata
  sample_columns <- metadata$file

  print(paste0("Processing mode: ", mode))

  # Parallelize the loop over runs
  results <- mclapply(1:n_runs, function(j) {
    fedsim_path <- run_path(mode, j, "FedSim")
    r_corrected_path <- run_path(mode, j, "R")

    # Read FedSim and R-corrected files
    fedsim_results <- fread(fedsim_path, sep = "\t")
    r_corrected <- fread(r_corrected_path, sep = "\t")

    # Check for size mismatch
    if (nrow(r_corrected) != nrow(fedsim_results) || ncol(r_corrected) != ncol(fedsim_results)) {
      warning(paste0("Size mismatch for run ", j, " in mode ", mode))
      return(NULL)
    }

    # Align column and row order
    setkeyv(r_corrected, "rowname")
    setkeyv(fedsim_results, "rowname")
    fedsim_results <- fedsim_results[, sample_columns, with = FALSE]
    r_corrected <- r_corrected[, sample_columns, with = FALSE]

    # Calculate differences
    differences <- abs(as.matrix(r_corrected) - as.matrix(fedsim_results))
    c(min(differences, na.rm = TRUE),
      max(differences, na.rm = TRUE),
      mean(differences, na.rm = TRUE),
      sd(differences, na.rm = TRUE))
  }, mc.cores = run_cores)

  failed_runs <- which(vapply(results, is.null, logical(1)))
  if (length(failed_runs) > 0) {
    stop(
      "Difference calculation failed for mode ", mode,
      " run(s): ", paste(failed_runs, collapse = ", ")
    )
  }

  # Combine results and write to output
  differences_table <- do.call(rbind, results)
  # add colnames
  colnames(differences_table) <- c("min", "max", "mean", "mean_sd")
  output_path <- file.path(root_dir, mode, "differences_fedRBE_R.tsv")
  fwrite(data.frame(run = 1:n_runs, differences_table), output_path, sep = "\t")
}

# Before-correction LMPV

precomputed_lmpv_path <- file.path(root_dir, "precomputed", "simulated_30runs_lmpv_summary.tsv.gz")

if (file.exists(precomputed_lmpv_path)) {
    message("Skipping per-run before-correction LMPV generation because the precomputed cache exists: ", precomputed_lmpv_path)
} else {
for(mode in modes){
    print(paste0("Processing mode: ", mode))

    metadata <- read_simulated_metadata(mode) %>%
        as.data.frame()
    rownames(metadata) <- metadata$file

    # create directory for lmvp if it does not exist
    if(!dir.exists(file.path(root_dir, mode, "lmvp"))){
        dir.create(file.path(root_dir, mode, "lmvp"), recursive = TRUE)
    }

    for(j in 1:n_runs){
        lmpv_output_path <- file.path(root_dir, mode, "lmvp", paste0(j, "_before_correction_lmpv.tsv"))
        if (file.exists(lmpv_output_path)) {
            next
        }

        before_corr <- fread(
                file.path(path_to_data, mode, "before", "intermediate", paste0(j, "_intensities_data.tsv")),
                sep = "\t") %>%
            as.data.frame() %>% column_to_rownames("rowname")
        before_corr <- before_corr[, metadata$file]


        form <- ~ condition + lab
        # calculate before correction LMPV
        before_correction_lmpv <- lmpv_plot(before_corr, metadata,
                                           form = form, only_table = TRUE)
        # write to file
        write.table(before_correction_lmpv,
                    lmpv_output_path,
                    sep = "\t", quote = FALSE, row.names = FALSE)
    }
}
}

# Evaluation errors

combined_df <- data.frame() # Initialize an empty data frame for combining data

# Read and combine data for all modes
for (mode in modes) {
    print(paste0("Processing mode: ", mode))
    errors_df <- read.table(file.path(root_dir, mode, "differences_fedRBE_R.tsv"), sep = "\t", header = TRUE)
    errors_df$mode <- ifelse(mode == "balanced", "Balanced", ifelse(mode == "mild_imbalanced", "Mild Imbalanced", "Strong Imbalanced")) # Add a column for the mode
    combined_df <- rbind(combined_df, errors_df) # Combine data
}

# Calculate the maximum value for each mode
mean_values_df <- combined_df %>%
    group_by(mode) %>%
    summarise(mean_value = mean(mean, na.rm = TRUE),
              max_value = max(max, na.rm = TRUE))

print(mean_values_df)

## ============================================================================
## Variance-partitioning summary violin plots (aggregated over 30 runs)
## ============================================================================

out_dir  <- file.path(root_dir, "lmpv_plots")             # where final figures will be stored
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# ---------------------------------------------------------------------------
# helper: read one lmpv table and reshape it
# ---------------------------------------------------------------------------
read_lmpv <- function(file, mode, run, status){
  read_tsv(file, show_col_types = FALSE, progress = FALSE) |>
    transmute(
      mode, run, status,
      Category = Var2,
      Value    = value
    )
}

# ---------------------------------------------------------------------------
# helper: build or read the lightweight precomputed summary cache
# ---------------------------------------------------------------------------
expected_lmpv_files <- function(){
  expand_grid(mode = modes, run = 1:n_runs, status = c("Before", "After")) |>
    mutate(
      file = file.path(
        root_dir,
        mode,
        "lmvp",
        sprintf("%d_%s_correction_lmpv.tsv", run, tolower(status))
      )
    )
}

build_lmpv_summary_from_files <- function(){
  expected_files <- expected_lmpv_files()
  missing_files <- expected_files |> filter(!file.exists(file))

  if (nrow(missing_files) > 0) {
    stop(
      "Missing precomputed LMPV cache and ", nrow(missing_files),
      " per-run LMPV source files. Restore ", precomputed_lmpv_path,
      " or provide the per-run LMPV tables. First missing file: ",
      missing_files$file[[1]],
      call. = FALSE
    )
  }

  expected_files |>
    pmap_dfr(\(mode, run, status, file) read_lmpv(file, mode, run, status))
}

validate_lmpv_summary <- function(lmpv_summary){
  required_columns <- c("mode", "run", "status", "Category", "Value")
  missing_columns <- setdiff(required_columns, names(lmpv_summary))
  if (length(missing_columns) > 0) {
    stop("LMPV summary is missing columns: ", paste(missing_columns, collapse = ", "), call. = FALSE)
  }

  lmpv_summary <- lmpv_summary |>
    transmute(
      mode = as.character(mode),
      run = as.integer(run),
      status = as.character(status),
      Category = as.character(Category),
      Value = as.numeric(Value)
    )

  expected_groups <- expand_grid(mode = modes, run = 1:n_runs, status = c("Before", "After"))
  observed_groups <- lmpv_summary |> distinct(mode, run, status)
  missing_groups <- anti_join(expected_groups, observed_groups, by = c("mode", "run", "status"))
  if (nrow(missing_groups) > 0) {
    stop(
      "LMPV summary is missing mode/run/status groups. First missing group: ",
      paste(missing_groups[1, ], collapse = ", "),
      call. = FALSE
    )
  }

  if (any(!is.finite(lmpv_summary$Value))) {
    stop("LMPV summary contains non-finite values", call. = FALSE)
  }

  lmpv_summary
}

load_lmpv_summary <- function(){
  if (file.exists(precomputed_lmpv_path)) {
    message("Reading precomputed LMPV summary cache: ", precomputed_lmpv_path)
    lmpv_summary <- read_tsv(precomputed_lmpv_path, show_col_types = FALSE, progress = FALSE)
  } else {
    message("Precomputed LMPV summary cache not found; building it from existing per-run LMPV tables.")
    lmpv_summary <- build_lmpv_summary_from_files()
    dir.create(dirname(precomputed_lmpv_path), showWarnings = FALSE, recursive = TRUE)
    write_tsv(lmpv_summary, precomputed_lmpv_path)
  }

  validate_lmpv_summary(lmpv_summary)
}

lmpv_summary <- load_lmpv_summary()

# ---------------------------------------------------------------------------
# helper: annotate statistics for each Category x status
# ---------------------------------------------------------------------------
label_stats <- function(df){
  df |>
    group_by(status, Category) |>
    summarise(
      mean   = mean(Value),
      sd     = sd(Value),
      median = median(Value),
      mad    = mad(Value),
      .groups = "drop"
    ) |>
    mutate(
      label = sprintf("Mean = %.2g ± %.2g\nMedian = %.2g ± %.2g",
                      mean, sd, median, mad)
    )
}

# ---------------------------------------------------------------------------
# helper: build one plot object (returned invisibly for reuse)
# ---------------------------------------------------------------------------
plot_mode <- function(mode){
  message("processing mode: ", mode)

  # ingest -----------------------------------------------------
  df <- lmpv_summary |>
    filter(.data$mode == .env$mode)

  if (nrow(df) == 0) {
    stop("No LMPV summary rows found for mode: ", mode, call. = FALSE)
  }

  df$status <- factor(df$status, levels = c("Before", "After"))

  ## ---- statistics to annotate ------------------------------------------------
  stats_tbl <- label_stats(df)
  stats_tbl$status <- factor(stats_tbl$status, levels = levels(df$status))
  # ensure identical factor order across modes -----------------
  df$Category     <- factor(df$Category, levels = unique(df$Category))
  stats_tbl$Category <- factor(stats_tbl$Category, levels = levels(df$Category))

  mode_without_underscore <- gsub("_", " ", mode)

  ## ---- base violin plot ------------------------------------------------------
  p <- ggplot(df, aes(x = Category, y = Value, fill = Category)) +
        geom_violin(trim = FALSE, colour = "black", alpha = 0.75) +
        geom_boxplot(width = .12, outlier.shape = NA, alpha = 0.50) +
        facet_wrap(~ status, ncol = 1, scales = "fixed") +
        geom_text(data = stats_tbl,
                  aes(x = Category, y = Inf, label = label),
                  vjust   = 1.1,
                  size    = 3.25,
                  fontface= "bold",
                  colour  = "black",
                  inherit.aes = FALSE
        ) +
        scale_y_continuous(expand = expansion(mult = c(0.02, 0.12))) +
        scale_fill_brewer(palette = "Dark2") +
        labs(title = glue::glue("Variance Partitioning\n\t{mode_without_underscore} data"),
             subtitle = "Aggregated over 30 runs",
             x = "Category", y = "Variance explained (%)") +
        theme_bw(base_size = 11) +
        theme(legend.position = "none",
              strip.text = element_text(size = 11, face = "bold"),
              plot.title = element_text(size = 14, face = "bold"))

  ## ---- export ----------------------------------------------------------------
  ggsave(filename = file.path(out_dir,
                              sprintf("variance_partition_%s.png", mode)),
         plot = p, width = 4.5, height = 7, dpi = 600)

  return(invisible(p))
}

# ---------------------------------------------------------------------------
# main loop (one plot per mode)
# ---------------------------------------------------------------------------
plots <- map(modes, plot_mode)

# combine the three modes into a single multi-panel PNG
all_modes_plot <- wrap_plots(plots, nrow = 1) +
  plot_annotation(title = "Variance Partitioning across imbalance scenarios")
ggsave(file.path(out_dir, "variance_partition_all_modes.png"), plot = all_modes_plot,
       width = 15, height = 7)
