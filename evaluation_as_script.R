### VARIABLES TO SET
design_concatted_all_vec <- c("test_data/raw_files_first_Imalanced/bath_info_all.tsv", "test_data/raw_files_first_Imalanced/bath_info_all.tsv", "test_data/raw_files_first_Balanced/bath_info_all.tsv", "test_data/raw_files_first_Balanced/bath_info_all.tsv")
central_cured_vec <- c("/results/Imalanced/central_cured.csv", "/results/Imalanced_nocov/central_cured.csv", "/results/Balanced/central_cured.csv", "/results/Balanced_nocov/central_cured.csv")
folder_fed_results_vec <- c(paste0(getwd(), "/results/Imalanced"), paste0(getwd(), "/results/Imalanced_nocov"), paste0(getwd(), "/results/Balanced", ""), paste0(getwd(), "/results/Balanced_nocov", ""))
plot_folder_name_vec <- c("Imalanced", "Imalanced_nocov", "Balanced", "Balanced_nocov")
cohorts_vector <- list(c('lab_A', 'lab_B', 'lab_C', 'lab_D', 'lab_E'), c('lab_A', 'lab_B', 'lab_C', 'lab_D', 'lab_E'), c('lab_A', 'lab_B', 'lab_C', 'lab_D', 'lab_E'), c('lab_A', 'lab_B', 'lab_C', 'lab_D', 'lab_E'))
plotting <- FALSE #TODO: plotting is not yet taken from the ipynb

### IMPORTS
if (!requireNamespace("magrittr", quietly = TRUE)) {
  install.packages("magrittr")
}
if (!requireNamespace("dplyr", quietly = TRUE)) {
  install.packages("dplyr")
}
if (!requireNamespace("tibble", quietly = TRUE)) {
  install.packages("tibble")
}
if (!requireNamespace("ragg", quietly = TRUE)) {
  install.packages("ragg")
}
if (!requireNamespace("tidyverse", quietly = TRUE)) {
  install.packages("tidyverse")
}

suppressPackageStartupMessages(library(magrittr))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(tibble))
suppressPackageStartupMessages(library(tidyverse))

### FUNCTIONS
main <- function() {
    stopifnot(
        length(design_concatted_all_vec) == length(central_cured_vec),
        length(central_cured_vec) == length(folder_fed_results_vec),
        length(folder_fed_results_vec) == length(plot_folder_name_vec),
        length(plot_folder_name_vec) == length(cohorts_vector)
    )
    for (idx in seq_along(design_concatted_all_vec)) {
        run_comparison(design_concatted_all_vec[idx], central_cured_vec[idx], 
            folder_fed_results_vec[idx], plot_folder_name_vec[idx], cohorts_vector[[idx]])
    }
}

run_comparison <- function(design_concat_all_file, central_cured_file, 
    folder_fed_results, plot_folder, cohorts) {
    print(paste0("CHECKING ", central_cured_file))
    print("=================================================================")
    path <- paste0(getwd(), "/", design_concat_all_file)
    batch_info_ref <- read.csv(path, check.names = FALSE, sep="\t") %>%
    column_to_rownames('rowname') %>%
    mutate(lab = factor(lab), condition = factor(condition))

    #dim(batch_info_ref)
    #head(batch_info_ref)



    path <- paste0(getwd(), "/", central_cured_file)
    cured_central <- read.csv(path, sep='\t', row.names = 1,  check.names = FALSE)
    #dim(cured_central)
    #head(cured_central)


    cured_federated <- NULL

    for(c in cohorts) {
        file_path <- paste0(folder_fed_results, "/", c, "/intensities_corrected.csv")
        temp_df <- read.csv(file_path, sep=',', row.names = 1,  check.names = FALSE)

        # Combine the dataframes
        if (is.null(cured_federated)) {
            cured_federated <- temp_df
        } else {
            cured_federated <- cbind(cured_federated, temp_df)
        }
    }
    
    cured_federated <- cured_federated[rownames(cured_central), batch_info_ref$file]
    print("Shape of cured federated results:")
    print(dim(cured_federated))
    print("Shape of cured central results:")
    print(dim(cured_central))

    ### Plotting
    # plot_name_prefix <- paste0("plots/", "FED_after_correction")
    # subname <- "A_B_conditions"
    # number <- "03"
    # plot_three_plots(cured_federated, batch_info_ref, plot_name_prefix, subname, number)

    # identical?
    print("identical: ")
    print(identical(round(cured_central, 6), round(cured_federated, 6)))

    # Check Row-by-Row and Column-by-Column Equality
    print("Row-by-Row and Column-by-Column Equality")
    print(all.equal(cured_central, cured_federated))

    # Calculate the mean of the absolute differences, removing NA's
    difference <- cured_central - cured_federated
    abs_difference <- abs(difference)
    mean_abs_difference <- mean(apply(abs_difference, c(1, 2), mean, na.rm = TRUE), na.rm = TRUE)
    max_abs_difference <- max(apply(abs_difference, c(1, 2), mean, na.rm = TRUE), na.rm = TRUE)
    print("mean_abs_difference")
    print(mean_abs_difference)
    print("max_abs_difference")
    print(max_abs_difference)

    # plotting
    if (plotting) {
        # TODO: this is still not functional, either use the ipynb or fix this
        pg_matrix <- NULL
        for (name in labs_list) {
        file_name_prefix <- paste0('/home/yuliya/repos/other/removeBatch/test_data/raw_files_first_', MODE, '/', name)

        if(is.null(pg_matrix)){
            pg_matrix <- read.csv(paste0(file_name_prefix, '_protein_groups_matrix.tsv'), check.names = FALSE, sep="\t") 
        } else {
            pg_matrix <- inner_join(pg_matrix, 
                            read.csv(paste0(file_name_prefix, '_protein_groups_matrix.tsv'), check.names = FALSE, sep="\t"),
                            by = "rowname")
        }
        }

        pg_matrix <- pg_matrix %>% column_to_rownames('rowname')
        pg_matrix <- log2(pg_matrix + 1)

        temp_df <- read.csv( paste0("results/", MODE, '/lab_A', "_intensities_corrected.tsv"), sep='\t', row.names = 1,  check.names = FALSE)
        pg_matrix <- pg_matrix[rownames(temp_df),  batch_info_ref$file]

        dim(pg_matrix)
        plot_name_prefix <- paste0("plots/", plot_folder, "/BEFORE_correction")
        subname <- "A_B_conditions"
        number <- "02"
        plot_three_plots(pg_matrix, batch_info_ref, plot_name_prefix, subname, number)

        plot_name_prefix <- paste0("plots/", plot_folder, "/R_after_correction")
        subname <- "A_B_conditions"
        number <- "02"
        plot_three_plots(cured_central, batch_info_ref, plot_name_prefix, subname, number)

        plot_name_prefix <- paste0("plots/", plot_folder, "/FED_after_correction")
        subname <- "A_B_conditions"
        number <- "03"
        plot_three_plots(cured_federated, batch_info_ref, plot_name_prefix, subname, number)
    }
    

}

## plotting functions used in run_comparison
pca_plot <- function(df, batch_info, title, path) {
  pca <- prcomp(t(na.omit(df)))
  # Plot PCA
  pca_df <-
    pca$x %>%
    as.data.frame() %>%
    rownames_to_column("file") %>% 
    left_join(batch_info,  by = "file") 
  # add % of explained variance
  var_expl <- pca$sdev^2 / sum(pca$sdev^2)
  names(var_expl) <- paste0("PC", 1:length(var_expl))
  # Add the label for the specific point
  pca_plot <- pca_df %>%
    ggplot(aes(PC1, PC2)) +
    geom_point(aes(col=condition, shape=lab), size=2) +
    theme_classic() +
    labs(title = title,
         x = glue::glue("PC1 [{round(var_expl['PC1']*100, 2)}%]"),
         y = glue::glue("PC2 [{round(var_expl['PC2']*100, 2)}%]"))

  ggsave(path, pca_plot)
}

# boxplot
boxplot_pg <- function(protein_matrix, title, path) {
  # Reshape data into long format
  long_data <- tidyr::gather(protein_matrix, 
                             key = "file", value = "Intensity")
  # Log tranformed scale
  boxplot <- ggplot(long_data, aes(x = file, y = Intensity)) + 
    geom_boxplot() +
    stat_summary(fun = mean, geom = "point", shape = 4, size = 1.5, color = "red") +
    theme_bw() +
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1, size = 5)) +
    labs(title = title) 

  ggsave(path, boxplot, width = 6, height = 6)
}

heatmap_plot <- function(pg_matrix, batch_info, name, plot_name_prefix){
    cor_matrix <- cor(na.omit(pg_matrix), use = "pairwise.complete.obs")
    pheatmap::pheatmap(cor_matrix, 
                        annotation_col = select(batch_info, c(condition, lab)),
                        treeheight_row = 0, treeheight_col = 0, 
                        fontsize_row = 5, fontsize_col = 5,
                        width = 7, height = 7,
                        main = paste0(name, ' heatmap'),
                        filename = plot_name_prefix)
}

plot_three_plots <- function(pg_matrix, batch_info, plot_name_prefix, subname, number){

        batch_info <- batch_info %>%
                mutate(file = case_when(
                lab %in% c('lab_A', 'lab_E') ~ str_split(file, "_") %>% 
                map_chr(~ if (length(.x) == 4) paste(.x[1], .x[2], .x[4], sep = "_") else paste(.x[1], .x[2], sep = "_")),
                
                lab == 'lab_C' ~ str_split(file, "_") %>% 
                map_chr(~ paste(.x[5], .x[6], sep = "_")),
                
                lab == 'lab_D' ~ str_split(file, "_") %>% 
                map_chr(~ paste(.x[6], .x[8], sep = "_")),
                
                TRUE ~ file
        ))

        rownames(batch_info) <- batch_info$file
        colnames(pg_matrix) <- batch_info$file


        boxplot_pg(pg_matrix, 
                paste0(subname, ' boxplot'), 
                paste0(plot_name_prefix, "_", subname, "_", number, "_boxplot.png")
        )

        pca_plot(pg_matrix,
                batch_info, 
                paste0(subname, ' pca'), 
                paste0(plot_name_prefix, "_", subname, "_", number, "_pca.png")
        )

        heatmap_plot(pg_matrix, 
                batch_info, 
                subname, 
                paste0(plot_name_prefix, "_", subname, "_", number, "_heatmap.png")
        )

}

### Call of main
main()