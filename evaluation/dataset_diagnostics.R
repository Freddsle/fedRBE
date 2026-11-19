# Per-dataset diagnostics of the label-aware correction. Run from the repository
# root:
#   pixi run Rscript evaluation/dataset_diagnostics.R <dataset>
# <dataset> is ecoli, ovarian_cancer, ccRCC_studies, quartet_multiomics or simulated.
#
# The real datasets write the diagnostic panels of their central limma output:
# evaluation_data/ecoli/plots/data_plot_Rcorrected.png,
# evaluation_data/ovarian_cancer/before/plots/norm_vs_tumor_plot_Rcorrected_UNION.png,
# evaluation_data/ccRCC_studies/plots/data_plot_Rcorrected.png and
# evaluation_data/quartet_multiomics/plots/<Modality>_limma_after_diagnostics.png.
# simulated writes the PCA plots of each scenario before and after fedRBE to
# evaluation/eval_simulation/pca_plots/pca_<mode>.png and prints the differences
# between the central R and the federated (SMPC) corrections.

source(file.path("evaluation_utils", "figure_aggregate_utils.R"))
source_plotting_utils()

arm <- "label_aware"

# Central limma output in its own feature order, with the metadata sample order.
read_central <- function(path, metadata) {
  read_matrix_table(path)[, rownames(metadata)]
}

diagnostics <- list(
  ecoli = function() {
    metadata <- load_ecoli_data(arm)$metadata
    central_corrected <- read_central(
      file.path("evaluation_data", "ecoli", "after", arm, "intensities_log_Rcorrected_UNION.tsv"),
      metadata
    )
    layout <- plots_multiple(central_corrected, metadata, "Bacterial dataset, R corrected")
    ggsave(file.path("evaluation_data", "ecoli", "plots", "data_plot_Rcorrected.png"),
           plot = layout, width = 12, height = 15)
  },

  ovarian_cancer = function() {
    metadata <- load_ovarian_data(arm)$metadata
    central_corrected <- read_central(
      file.path("evaluation_data", "ovarian_cancer", "after", arm, "central_corrected_UNION.tsv"),
      metadata
    )
    set.seed(1)  # Same feature subsample on every run.
    plot_res <- plots_normtumor(central_corrected[sample(1:nrow(central_corrected), 10000), ],
                                metadata %>% mutate(Sample_geo_ID = file), " ", "norm_vs_tumor")
    layout <- (plot_res[[1]] + plot_res[[2]]) / (plot_res[[3]])
    ggsave(file.path("evaluation_data", "ovarian_cancer", "before", "plots",
                     "norm_vs_tumor_plot_Rcorrected_UNION.png"),
           plot = layout, width = 14, height = 12)
  },

  ccRCC_studies = function() {
    metadata <- load_ccrcc_data(arm)$metadata
    central_corrected <- read_central(
      file.path("evaluation_data", "ccRCC_studies", "after", arm, "intensities_log_Rcorrected_UNION.tsv"),
      metadata
    )
    layout <- plots_multiple(central_corrected, metadata, "ccRCC proteomics, R corrected")
    ggsave(file.path("evaluation_data", "ccRCC_studies", "plots", "data_plot_Rcorrected.png"),
           plot = layout, width = 12, height = 15)
  },

  quartet_multiomics = function() {
    for (modality in c("Transcriptomics", "Proteomics", "Metabolomics")) {
      metadata <- load_quartet_modality_data(modality, arm)$metadata
      central_corrected <- read_central(
        file.path("evaluation_data", "quartet_multiomics", "after", arm, modality,
                  "intensities_log_Rcorrected_UNION.tsv"),
        metadata
      )
      plot_diagnostics(central_corrected, metadata, modality, "after",
                       file.path("evaluation_data", "quartet_multiomics", "plots"))
    }
  },

  simulated = function() {
    data_path <- file.path("evaluation_data", "simulated")
    plots_dir <- file.path("evaluation", "eval_simulation", "pca_plots")
    dir.create(plots_dir, recursive = TRUE, showWarnings = FALSE)
    for (mode in c("balanced", "mild_imbalanced", "strong_imbalanced")) {
      before_correction <- NULL
      for (lab in c("lab1", "lab2", "lab3")) {
        lab_data <- read.csv(file.path(data_path, mode, "before", lab, "intensities.tsv"),
                             sep = "\t", header = TRUE, check.names = FALSE)
        if (is.null(before_correction)) {
          before_correction <- lab_data
        } else {
          before_correction <- merge(before_correction, lab_data, by = "rowname", all = TRUE)
        }
      }
      before_correction <- before_correction %>% column_to_rownames("rowname")
      central_corrected <- read.csv(file.path(data_path, mode, "after", "intensities_R_corrected.tsv"),
                                    sep = "\t", header = TRUE, row.names = 1, check.names = FALSE)
      fed_corrected <- read.csv(file.path(data_path, mode, "after", "FedApp_corrected_data_smpc.tsv"),
                                sep = "\t", header = TRUE, row.names = 1, check.names = FALSE)

      metadata <- read.csv(file.path(data_path, mode, "all_metadata.tsv"),
                           sep = "\t", header = TRUE, row.names = 1)
      rownames(metadata) <- metadata$file
      metadata$condition <- as.factor(metadata$condition)
      metadata$lab <- as.factor(metadata$lab)

      # order the columns in intensity dataframes
      before_correction <- before_correction[, rownames(metadata)]
      central_corrected <- central_corrected[rownames(before_correction), rownames(metadata)]
      fed_corrected <- fed_corrected[rownames(before_correction), rownames(metadata)]

      plot_pca_uncorrected <- pca_plot(
        before_correction, metadata,
        title = paste("Before correction"),
        quantitative_col_name = "file",
        col_col = "condition", shape_col = "lab", show_legend = TRUE, cbPalette = c("#a70a66", "#2a609d"),
        point_size = 1
      ) +
        geom_point(size = 1.5, alpha = 0.7) +
        guides(
          colour = guide_legend(override.aes = list(size = 3)),
          shape = guide_legend(override.aes = list(size = 3))
        )
      plot_pca_corrected <- pca_plot(
        fed_corrected, metadata,
        title = paste("After FedRBE correction"),
        quantitative_col_name = "file",
        col_col = "condition", shape_col = "lab", show_legend = FALSE, cbPalette = c("#a70a66", "#2a609d"),
        point_size = 1
      ) +
        geom_point(size = 2, alpha = 0.8)

      layout <- plot_pca_uncorrected /
                plot_pca_corrected
      layout <- layout + plot_annotation(paste0("PCA plots, simulated ", mode))
      ggsave(file.path(plots_dir, paste0("pca_", mode, ".png")),
             plot = layout, width = 5, height = 7, dpi = 600)

      print(paste("Mode:", mode))
      differences <- calculated_differences(central_corrected, fed_corrected)
      cat("Min difference:", differences[2], "\n")
      cat("Mean difference:", differences[3], "\n")
      cat("Max difference:", differences[1], "\n")
      cat("\n")
    }
  }
)

dataset <- match.arg(commandArgs(trailingOnly = TRUE)[1], names(diagnostics))
diagnostics[[dataset]]()
