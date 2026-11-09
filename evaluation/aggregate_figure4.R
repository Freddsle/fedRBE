# Aggregate Figure 4
#
# Build Figure 4 from the clustering and classification tables. K-means uses all
# complete-case features and RF uses all union features.
# Run from the repository root:
#   CORRECTION_ARM=label_aware pixi run Rscript evaluation/aggregate_figure4.R
# Set CORRECTION_ARM=label_aware (default) or CORRECTION_ARM=no_label.
# Each arm writes figures/figure4_<arm>.png without statistics and
# figures/figure4_<arm>_statistics.png with significance brackets in panels a and c.
# FIGURE4_ARI_TABLE, FIGURE4_RF_REPORT and FIGURE4_OUTPUT override the source and
# output paths; an output path ending in .svg writes an SVG.
# The script also writes the test table (<output>_statistics.tsv) and Figure S7
# (leave-one-cohort-out per held-out cohort with brackets, figureS7_<arm>) next to
# the Figure 4 output.

source(file.path("evaluation_utils", "figure_aggregate_utils.R"))
source(file.path("evaluation_utils", "figure4_statistics.R"))
require_figure4_packages()

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
figureS7_height <- 10
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
# tasks for panels b and c. The 70:30 panel uses one pooled `All` score per seed
# for both learning methods. Leave-one-cohort-out keeps each held-out cohort score.
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

# Exact rank tests over seeds: per dataset
# (panels a and c) and per held-out cohort (Figure S7). Panel b is an overview.

figure4_stats <- compute_figure4_statistics(figure4_data, CORRECTION_ARM)
statistics_path <- write_figure4_statistics(
  figure4_stats, paste0(tools::file_path_sans_ext(output_path), "_statistics.tsv")
)
cat("Saved Figure 4 and S7 statistics to", statistics_path, "\n")
cat("Significant contrasts with |mean difference| <", figure4_min_difference, ":\n")
print(figure4_negligible_effects(figure4_stats), row.names = FALSE, digits = 3)
figure4_annotations <- list(
  ari = figure4_annotation_data(figure4_stats, "4a ARI"),
  split = figure4_annotation_data(figure4_stats, "4c train/test")
)
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
    height = figure_height,
    dpi = figure_dpi
  )
  cat("Saved Figure 4 to", saved_path, "\n")
}

# Figure S7: the panel b scores per held-out cohort, with the per-cohort tests.
# It uses the Figure 4 output's directory and format.

figureS7_tasks <- observed_levels(figure4_data$leave_one_cohort_out$Task, classification_task_order)
figureS7_annotations <- stats::setNames(lapply(figureS7_tasks, function(task) {
  figure4_annotation_data(figure4_stats, "S7 LOCO", dataset = one_line(task))
}), figureS7_tasks)
figureS7_plot <- make_figureS7_plot(
  loo_data = figure4_data$leave_one_cohort_out,
  task_order = classification_task_order,
  scenario_order = scenario_order,
  scenario_colors = scenario_colors,
  point_size = point_size,
  feature_label = classification_feature_label,
  annotations = figureS7_annotations,
  annotation_caption = statistics_caption
)
figureS7_plot$panels[[1]] <- figureS7_plot$panels[[1]] + ggplot2::labs(
  title = paste(
    paste(correction_arm_titles[[CORRECTION_ARM]], "RF leave-one-cohort-out per held-out cohort",
          sep = " · "),
    figureS7_plot$panels[[1]]$labels$title,
    sep = "\n"
  )
)
figureS7_extension <- tools::file_ext(output_path)
figureS7_prefix <- sub("^figure4", "figureS7", output_prefix)
figureS7_path <- file.path(
  dirname(output_path),
  paste0(figureS7_prefix, CORRECTION_ARM, ".",
         if (nzchar(figureS7_extension)) figureS7_extension else "png")
)
saved_path <- save_grid_png(
  function() draw_figure4(figureS7_plot),
  output_path = figureS7_path,
  width = figure_width,
  height = figureS7_height,
  dpi = figure_dpi
)
cat("Saved Figure S7 to", saved_path, "\n")
