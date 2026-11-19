# Aggregate Figure 4
#
# Build Figure 4 from the clustering and classification tables. K-means uses all
# complete-case features and RF uses all union features.
# Run from the repository root:
#   pixi run Rscript evaluation/aggregate_figure4.R
# Without CORRECTION_ARM, the script runs once per arm (label_aware, then
# no_label). CORRECTION_ARM=label_aware or CORRECTION_ARM=no_label runs only that
# arm, as does sourcing the script (default label_aware).
# Each arm writes figures/figure4_<arm>.png without statistics and
# figures/figure4_<arm>_statistics.png with significance brackets in panel a.
# FIGURE4_ARI_TABLE, FIGURE4_RF_REPORT and FIGURE4_OUTPUT override the source and
# output paths; an output path ending in .svg writes an SVG. The ARI table and
# the output belong to one arm, so overriding them requires CORRECTION_ARM.
# The script also writes the test table (<output>_statistics.tsv), Figure S7
# (leave-one-cohort-out per held-out cohort with brackets, figureS7_<arm>) and
# Figure S8 (70:30 per test cohort with brackets, figureS8_<arm>) next to the
# Figure 4 output.
#
# What the panels show. Every panel has four scenarios per dataset: central or
# federated learning, each on uncorrected data ("before BEC") and on data
# corrected with the chosen arm ("after BEC"). Central learning uses central
# limma::removeBatchEffect, federated learning uses fedRBE.
# - 4a, k-means clustering: one dot per seed (10 seeds), giving the ARI between
#   the k-means clusters and the condition labels. K is the number of
#   conditions. K-means clusters all samples of a dataset on the complete-case
#   features, and the correction is fitted on all samples. With label_aware,
#   the correction also uses all of the labels.
# - 4b, random forest, leave-one-cohort-out: one dot per held-out cohort and
#   seed, giving the MCC on that cohort. The correction and the forest see only
#   the other cohorts. The held-out cohort is shifted onto the training level
#   without its labels.
# - 4c, random forest, 70:30 train/test split: one dot per cohort and seed,
#   giving the MCC on that cohort's 30% test part. The split is stratified by
#   label and batch. The correction and the forest see only the 70% training
#   part. The test samples are corrected with the training batch coefficients,
#   without their labels.
# The pooled RF score over all cohorts ("All" in the report) is not plotted.

source(file.path("evaluation_utils", "figure_aggregate_utils.R"))
source(file.path("evaluation_utils", "figure4_statistics.R"))
require_figure4_packages()

run_each_correction_arm("Figure 4", c("FIGURE4_ARI_TABLE", "FIGURE4_OUTPUT"))

# Source and output files: Figure 4a uses the multi-run clustering ARI table.
# Figure 4b and Figure 4c use the classification metric report and are filtered
# to real MCC tasks.

CORRECTION_ARM <- correction_arm()

ari_dir <- file.path(
  "evaluation_clusterization_after_correction",
  "real_datasets"
)
if (CORRECTION_ARM == "no_label") {
  ari_dir <- file.path(ari_dir, "no_label")
}
ari_path <- file.path(ari_dir, "metrics_ari_multirun.tsv")
classification_path <- file.path(
  "evaluation_classification_after_correction",
  "results",
  "classification_metric_report.csv"
)
output_dir <- "figures"
output_prefix <- "figure4_"
output_path <- file.path(output_dir, paste0(output_prefix, CORRECTION_ARM, ".png"))

# Path overrides are absolute or relative to the repository root.
resolve_figure4_path <- function(variable, default) {
  value <- Sys.getenv(variable, "")
  if (!nzchar(value)) return(default)
  value
}
ari_path <- resolve_figure4_path("FIGURE4_ARI_TABLE", ari_path)
classification_path <- resolve_figure4_path("FIGURE4_RF_REPORT", classification_path)
output_path <- resolve_figure4_path("FIGURE4_OUTPUT", output_path)

# Figure parameters: task ordering, scenario names, colors, seed counts and PNG size.

style <- figure4_scenarios()
scenario_order <- style$order
scenario_colors <- style$colors

ari_dataset_order <- c("ecoli", "ovarian_cancer", "ccRCC_proteomics", "quartet_multiomics")
ari_dataset_labels <- c(
  "ecoli" = "E. coli",
  "ovarian_cancer" = "Ovarian\ncancer",
  "ccRCC_proteomics" = "ccRCC\nproteomics",
  "quartet_multiomics" = "Quartet\nmultiomics"
)

ari_method_labels <- c(
  "BC_Cntrl" = "Central before BEC",
  "BC_Fed" = "Federated before BEC",
  "AC_Cntrl" = "Central after BEC",
  "AC_Fed" = "Federated after BEC"
)

# Match the left-to-right dataset order used in Figure 4a.
classification_task_order <- unname(classification_task_labels)

simulated_classification_datasets <- c(
  "Balanced Simulated",
  "Balanced Simulated Data",
  "Mildly Imbalanced Simulated",
  "Strongly Imbalanced Simulated"
)
excluded_classification_datasets <- c("Proteomics Multibatch Data")
average_client_name <- "All"
ari_seeds_per_group <- 10

clustering_feature_label <- "All complete-case features"
classification_feature_label <- "All union features"

point_size <- 1.05

figure_width <- 7.5
figure_height <- 8.5
per_cohort_figure_height <- 10
figure_dpi <- 600

# Load data: the two canonical source tables. Select the matching correction arm
# and its uncorrected baseline; keep the original result CSV unchanged.

ari <- utils::read.delim(ari_path, stringsAsFactors = FALSE, check.names = FALSE)
classification_report <- utils::read.csv(
  classification_path,
  stringsAsFactors = FALSE,
  check.names = FALSE
)

preprocessing <- if (CORRECTION_ARM == "label_aware") "label_aware" else "label_free"
classification_report <- classification_report[
  classification_report$data_preprocessing_name %in% c("uncorrected", preprocessing),
  , drop = FALSE
]
classification_report$data_preprocessing_name[
  classification_report$data_preprocessing_name == preprocessing
] <- "corrected"

# Prepare figure data: condition-target clustering ARI for panel a and real MCC
# tasks for panels b and c. Both keep one score per test cohort and seed for both
# learning methods: the held-out cohort (LOCO) or a cohort's 70:30 test part.
# Duplicate metric identities are rejected by the shared helper.

figure4_data <- prepare_figure4_plot_data(
  ari = ari,
  classification_report = classification_report,
  dataset_order = ari_dataset_order,
  dataset_labels = ari_dataset_labels,
  method_labels = ari_method_labels,
  task_labels = classification_task_labels,
  task_order = classification_task_order,
  scenario_labels = classification_scenario_labels(),
  scenario_order = scenario_order,
  simulated_datasets = simulated_classification_datasets,
  excluded_datasets = excluded_classification_datasets,
  average_client_name = average_client_name,
  seeds_per_group = ari_seeds_per_group
)

# Exact rank tests over seeds: per dataset (panel a), per held-out cohort
# (Figure S7) and per 70:30 test cohort (Figure S8). Panels b and c are overviews.

# Incomplete leave-one-cohort-out results skip panel b and Figure S7; errors of
# panel a or Figure S8 recur in the second attempt and stop the script.
figure4_stats <- tryCatch(
  compute_figure4_statistics(figure4_data, CORRECTION_ARM),
  error = function(error) {
    warning("Skipping panel b and Figure S7: ", conditionMessage(error), call. = FALSE)
    NULL
  }
)
if (is.null(figure4_stats)) {
  figure4_data$leave_one_cohort_out <- figure4_data$leave_one_cohort_out[0, , drop = FALSE]
  figure4_stats <- compute_figure4_statistics(figure4_data, CORRECTION_ARM)
}
has_loo <- nrow(figure4_data$leave_one_cohort_out) > 0
statistics_path <- write_figure4_statistics(
  figure4_stats, paste0(tools::file_path_sans_ext(output_path), "_statistics.tsv")
)
cat("Saved Figure 4, S7 and S8 statistics to", statistics_path, "\n")
cat("Significant contrasts with |mean difference| <", figure4_min_difference, ":\n")
print(figure4_negligible_effects(figure4_stats), row.names = FALSE, digits = 3)
figure4_annotations <- list(ari = figure4_annotation_data(figure4_stats, "4a ARI"))
statistics_caption <- figure4_statistics_caption()

# Build and save Figure 4 twice: plain at the output path, and with significance
# brackets and the test caption at the same path with a `_statistics` suffix.

build_figure4 <- function(annotations, caption) {
  plot <- make_figure4_plot(
    figure4_data = figure4_data,
    dataset_labels = ari_dataset_labels,
    dataset_order = ari_dataset_order,
    task_order = classification_task_order,
    scenario_order = scenario_order,
    scenario_colors = scenario_colors,
    point_size = point_size,
    clustering_feature_label = clustering_feature_label,
    classification_feature_label = classification_feature_label,
    annotations = annotations,
    annotation_caption = caption
  )
  plot$panels[[1]] <- plot$panels[[1]] + ggplot2::labs(
    title = paste(correction_arm_titles[[CORRECTION_ARM]], plot$panels[[1]]$labels$title, sep = "\n")
  )
  plot
}

figure4_extension <- tools::file_ext(output_path)
statistics_output_path <- paste0(
  tools::file_path_sans_ext(output_path), "_statistics.",
  if (nzchar(figure4_extension)) figure4_extension else "png"
)
figure4_variants <- list(
  list(path = output_path, plot = build_figure4(NULL, NULL)),
  list(path = statistics_output_path,
       plot = build_figure4(figure4_annotations, statistics_caption))
)
for (variant in figure4_variants) {
  saved_path <- save_grid_png(
    function() draw_figure4(variant$plot),
    output_path = variant$path,
    width = figure_width,
    # without panel b, a and c keep their size
    height = if (has_loo) figure_height else figure_height * 2 / 3,
    dpi = figure_dpi
  )
  cat("Saved Figure 4 to", saved_path, "\n")
}

# Figures S7 and S8: the panel b and c scores per test cohort, with the per-cohort
# tests. They use the Figure 4 output's directory and format. Returns the brackets
# per task.

save_per_cohort_figure <- function(data, figure, panel, description) {
  tasks <- observed_levels(data$Task, classification_task_order)
  annotations <- stats::setNames(lapply(tasks, function(task) {
    figure4_annotation_data(figure4_stats, panel, dataset = one_line(task))
  }), tasks)
  plot <- make_per_cohort_plot(
    cohort_data = data,
    task_order = classification_task_order,
    scenario_order = scenario_order,
    scenario_colors = scenario_colors,
    point_size = point_size,
    feature_label = classification_feature_label,
    annotations = annotations,
    annotation_caption = statistics_caption
  )
  plot$panels[[1]] <- plot$panels[[1]] + ggplot2::labs(
    title = paste(
      paste(correction_arm_titles[[CORRECTION_ARM]], description, sep = " · "),
      plot$panels[[1]]$labels$title,
      sep = "\n"
    )
  )
  extension <- tools::file_ext(output_path)
  path <- file.path(
    dirname(output_path),
    paste0(sub("^figure4", figure, output_prefix), CORRECTION_ARM, ".",
           if (nzchar(extension)) extension else "png")
  )
  saved_path <- save_grid_png(
    function() draw_figure4(plot),
    output_path = path,
    width = figure_width,
    height = per_cohort_figure_height,
    dpi = figure_dpi
  )
  cat("Saved Figure", sub("^figure", "", figure), "to", saved_path, "\n")
  annotations
}

if (has_loo) {
  figureS7_annotations <- save_per_cohort_figure(
    figure4_data$leave_one_cohort_out, "figureS7", "S7 LOCO",
    "RF leave-one-cohort-out per held-out cohort"
  )
}
figureS8_annotations <- save_per_cohort_figure(
  figure4_data$train_test_split, "figureS8", "S8 train/test",
  "RF 70:30 train-test split per test cohort"
)
