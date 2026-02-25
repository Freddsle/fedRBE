library(tidyverse)

script_dir <- tryCatch(
  normalizePath(dirname(rstudioapi::getActiveDocumentContext()$path)),
  error = function(e) {
    file_arg <- grep("--file=", commandArgs(FALSE), value = TRUE)
    if (length(file_arg) > 0) normalizePath(dirname(sub("--file=", "", file_arg))) else getwd()
  }
)

source(file.path(script_dir, "../../evaluation_utils/plots_eda.R"))

path_to_data <- paste0(script_dir, "/")

args <- commandArgs(trailingOnly = TRUE)
n_runs <- if (length(args) >= 1) as.integer(args[1]) else 15

# If any intermediate intensities file is missing, run the simulation script first
simulation_needed <- any(sapply(
    c("balanced", "mild_imbalanced", "strong_imbalanced"),
    function(mode) any(sapply(1:n_runs, function(j)
        !file.exists(paste0(path_to_data, mode, "/before/intermediate/", j, "_intensities_data.tsv"))
    ))
))
if (simulation_needed) {
    message("Some intermediate intensities files are missing - running 01_data_simulation.R ...")
    system2("Rscript", args = c(file.path(script_dir, "01_data_simulation.R"), n_runs))
}

for(mode in c(
    "balanced",
    "mild_imbalanced", "strong_imbalanced"
)){
    # if folder does not exist, create it
    if(!dir.exists(paste0(path_to_data, mode, "/after/runs"))){
        dir.create(paste0(path_to_data, mode, "/after/runs"), recursive = T)
    }

    print(paste0("Processing mode: ", mode))

    metadata <- read.csv(paste0(path_to_data, mode, "/all_metadata.tsv"), sep = "\t") %>%
        as.data.frame()

    for(j in 1:n_runs){
        intensities <- read.csv(paste0(path_to_data, mode, "/before/intermediate/", j, "_intensities_data.tsv"), sep = "\t") %>%
            as.data.frame() %>%
            column_to_rownames("rowname")

        metadata <- metadata %>%
          mutate(condition = as.factor(condition), lab = as.factor(lab))

        design <- model.matrix(~ condition, metadata)
        colnames(design) <- c("Intercept", "condition")

        intensities_corrected <- limma::removeBatchEffect(
                intensities[,metadata$file],
                metadata$lab,
                design = design) %>%
            as.data.frame()

        # write to file
        write.table(intensities_corrected %>% rownames_to_column("rowname"),
                    paste0(path_to_data, mode, "/after/runs/", j, "_R_corrected.tsv"),
                    sep = "\t", row.names = F)

        print(paste0("\t\tSaved corrected intensities for mode: ", mode))
    }
}