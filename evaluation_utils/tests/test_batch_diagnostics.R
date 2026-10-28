# Run from the repository root: pixi run Rscript evaluation_utils/tests/test_batch_diagnostics.R
# Check the dataset loaders and diagnostic models against the prepared correction inputs.
source("evaluation_utils/figure_aggregate_utils.R")
source("evaluation_utils/evaluation_funcs.R")

same_partition <- function(left, right) {
  identical(outer(as.character(left), as.character(left), "=="),
            outer(as.character(right), as.character(right), "=="))
}

tracked <- system2("git", c("ls-files", "--"), stdout = TRUE)
read_design <- function(path) {
  stopifnot(path %in% tracked)
  utils::read.delim(path, check.names = FALSE)
}
median_term <- function(data, metadata, form, term) {
  fractions <- fixed_effect_variance_partition(data, metadata, form)
  median(fractions$value[fractions$variable == term], na.rm = TRUE)
}

for (arm in c("label_aware", "no_label")) {
  Sys.setenv(CORRECTION_ARM = arm)
  stopifnot(correction_arm() == arm)
  joint <- load_quartet_data(arm)
  stopifnot(nrow(joint$metadata) == 48L, nlevels(joint$metadata$batch) == 4L,
            same_partition(joint$metadata$batch, joint$metadata$batch_code))

  for (modality in c("Transcriptomics", "Proteomics", "Metabolomics")) {
    item <- prepare_plot_matrices(load_quartet_modality_data(modality, arm))
    metadata <- item$metadata
    design <- do.call(rbind, lapply(unique(metadata$client), function(client) {
      read_design(file.path("evaluation_data/quartet_multiomics/before",
                            modality, client, "design.tsv"))
    }))
    stopifnot(!anyDuplicated(design$file), setequal(design$file, metadata$file),
              identical(as.character(metadata$batch),
                        design$batch[match(metadata$file, design$file)]),
              identical(dim(table(metadata$batch, metadata$condition, metadata$rep)),
                        c(4L, 4L, 3L)),
              all(table(metadata$batch, metadata$condition, metadata$rep) == 1L),
              same_partition(metadata$batch, metadata$batch_code))
    joint_order <- match(joint$metadata$file, metadata$pseudo_sample)
    stopifnot(!anyNA(joint_order), !anyDuplicated(joint_order),
              same_partition(joint$metadata$batch, metadata$batch[joint_order]),
              identical(as.character(joint$metadata$condition),
                        as.character(metadata$condition[joint_order])))
    old_median <- median_term(item$plot_matrices$uncorrected, metadata,
                              ~ condition + lab, "lab")
    batch_median <- median_term(item$plot_matrices$uncorrected, metadata,
                                ~ condition + batch, "batch")
    corrected <- median_term(item$plot_matrices$fed_corrected, metadata,
                             ~ condition + batch, "batch")
    if (modality == "Metabolomics") {
      stopifnot(abs(batch_median - old_median) < 1e-10)
    } else {
      stopifnot(batch_median > old_median + 1e-10)
    }
    stopifnot(is.finite(corrected), corrected < 1e-10)
    cat(arm, modality, "before lab/batch medians", old_median, batch_median,
        "; after batch median", corrected, "\n")
  }

  ccrcc <- load_ccrcc_data(arm)
  expected <- do.call(rbind, lapply(c("PDC000127", "PXD030344", "PXD042844"),
                                    function(site) {
    design <- read_design(file.path("evaluation_data/ccRCC_studies/before",
                                    site, "design.tsv"))
    stopifnot(all(design$Tumor + design$Normal == 1))
    data.frame(file = design$Sample,
               condition = ifelse(design$Tumor == 1, "Tumor", "Normal"), lab = site)
  }))
  stopifnot(identical(ccrcc$metadata$file, expected$file),
            identical(ccrcc$metadata$condition, expected$condition),
            identical(ccrcc$metadata$lab, expected$lab),
            identical(rownames(ccrcc$metadata), expected$file),
            identical(colnames(ccrcc$uncorrected), expected$file),
            identical(colnames(ccrcc$fed_corrected), expected$file))
  cat(arm, "ccRCC sample/condition/study alignment matches tracked input designs\n")
  rm(joint, ccrcc)
  invisible(gc())
}
cat("Batch diagnostic regression checks passed for both correction arms.\n")
