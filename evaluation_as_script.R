### VARIABLES TO SET
design_concatted_all_vec <- c("test_data/raw_files_first_Imalanced/bath_info_all.tsv")
central_cured_vec <- c("/results/Imalanced/central_cured.csv")
folder_fed_results_vec <- c(paste0(getwd(), "/results/Imalanced"))

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
suppressPackageStartupMessages(library(magrittr))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(tibble))

### FUNCTIONS
main <- function() {
    #TODO: add assert, vectors of same length
    for (idx in seq_along(design_concatted_all_vec)) {
    run_comparison(design_concatted_all_vec[idx], central_cured_vec[idx], folder_fed_results_vec[idx])
    }
}

run_comparison <- function(design_concat_all_file, central_cured_file, folder_fed_results) {
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




    cohorts <- c('lab_A', 'lab_B', 'lab_C', 'lab_D', 'lab_E')

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
    print("identical:")
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
}

### Call of main
main()