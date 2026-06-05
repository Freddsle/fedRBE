"""
This script analyses the difference between centralized results and
federated results. It compares the results of the experiments for the
differences of each individual value.
It saves the result in csv format and prints them.
It checks for:
- Check for same shape, same columns, same rows (after sorting columns and rows)
- Check for maximal and mean value difference over all corresponding values
The analysis can be extended by adding more experiments to the
experiment_results list. This can be found under the ### EXPERIMENT RESULTS
comment.
"""
import os
from evaluation_utils import utils_analyse as utils


### EXPERIMENT RESULTS
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation_data")
experiment_results = list()
    # Add all results to this list as utils.ExperimentResult objects
    # basically specifying the corrected data files of the central and federated results

# SIMULATED
experiment_results.append(utils.ExperimentResult(
    name="Simulated Balanced",
    federated_result_file=os.path.join(base_dir, "simulated", "balanced", "after", "FedApp_corrected_data.tsv"),
    central_result_file=os.path.join(base_dir, "simulated", "balanced", "after", "intensities_R_corrected.tsv"),
))
experiment_results.append(utils.ExperimentResult(
    name="Simulated Balanced (SMPC)",
    federated_result_file=os.path.join(base_dir, "simulated", "balanced", "after", "FedApp_corrected_data_smpc.tsv"),
    central_result_file=os.path.join(base_dir, "simulated", "balanced", "after", "intensities_R_corrected.tsv"),
))

experiment_results.append(utils.ExperimentResult(
    name="Simulated Mildly Imbalanced",
    federated_result_file=os.path.join(base_dir, "simulated", "mild_imbalanced", "after", "FedApp_corrected_data.tsv"),
    central_result_file=os.path.join(base_dir, "simulated", "mild_imbalanced", "after", "intensities_R_corrected.tsv"),
))
experiment_results.append(utils.ExperimentResult(
    name="Simulated Mildly Imbalanced (SMPC)",
    federated_result_file=os.path.join(base_dir, "simulated", "mild_imbalanced", "after", "FedApp_corrected_data_smpc.tsv"),
    central_result_file=os.path.join(base_dir, "simulated", "mild_imbalanced", "after", "intensities_R_corrected.tsv"),
))

experiment_results.append(utils.ExperimentResult(
    name="Simulated Strongly Imbalanced",
    federated_result_file=os.path.join(base_dir, "simulated", "strong_imbalanced", "after", "FedApp_corrected_data.tsv"),
    central_result_file=os.path.join(base_dir, "simulated", "strong_imbalanced", "after", "intensities_R_corrected.tsv"),
))
experiment_results.append(utils.ExperimentResult(
    name="Simulated Strongly Imbalanced (SMPC)",
    federated_result_file=os.path.join(base_dir, "simulated", "strong_imbalanced", "after", "FedApp_corrected_data_smpc.tsv"),
    central_result_file=os.path.join(base_dir, "simulated", "strong_imbalanced", "after", "intensities_R_corrected.tsv"),
))

# PROTEOMICS
experiment_results.append(utils.ExperimentResult(
    name="E. coli",
    federated_result_file=os.path.join(base_dir, "ecoli", "after", "FedApp_corrected_data.tsv"),
    central_result_file=os.path.join(base_dir, "ecoli", "after", "intensities_log_Rcorrected_UNION.tsv"),
))
experiment_results.append(utils.ExperimentResult(
    name="E. coli (SMPC)",
    federated_result_file=os.path.join(base_dir, "ecoli", "after", "FedApp_corrected_data_smpc.tsv"),
    central_result_file=os.path.join(base_dir, "ecoli", "after", "intensities_log_Rcorrected_UNION.tsv"),
))

experiment_results.append(utils.ExperimentResult(
    name="Quartet",
    federated_result_file=os.path.join(base_dir, "quartet", "after", "FedApp_corrected_data.tsv"),
    central_result_file=os.path.join(base_dir, "quartet", "after", "intensities_log_Rcorrected_UNION.tsv"),
))
experiment_results.append(utils.ExperimentResult(
    name="Quartet (SMPC)",
    federated_result_file=os.path.join(base_dir, "quartet", "after", "FedApp_corrected_data_smpc.tsv"),
    central_result_file=os.path.join(base_dir, "quartet", "after", "intensities_log_Rcorrected_UNION.tsv"),
))

# MULTIOMICS
for modality in ["Transcriptomics", "Proteomics", "Metabolomics"]:
    for name_suffix, result_filename in [
        ("", "FedApp_corrected_data.tsv"),
        (" (SMPC)", "FedApp_corrected_data_smpc.tsv"),
    ]:
        experiment_results.append(utils.ExperimentResult(
            name=f"Multiomics {modality}{name_suffix}",
            federated_result_file=os.path.join(
                base_dir,
                "multiomics",
                "after",
                modality,
                result_filename,
            ),
            central_result_file=os.path.join(
                base_dir,
                "multiomics",
                "after",
                modality,
                "intensities_log_Rcorrected_UNION.tsv",
            ),
        ))

# MICROARRAY
experiment_results.append(utils.ExperimentResult(
    name="Ovarian cancer",
    federated_result_file=os.path.join(base_dir, "ovarian_cancer", "after", "FedApp_corrected_data.tsv"),
    central_result_file=os.path.join(base_dir, "ovarian_cancer", "after", "central_corrected_UNION.tsv")
))
experiment_results.append(utils.ExperimentResult(
    name="Ovarian cancer (SMPC)",
    federated_result_file=os.path.join(base_dir, "ovarian_cancer", "after", "FedApp_corrected_data_smpc.tsv"),
    central_result_file=os.path.join(base_dir, "ovarian_cancer", "after", "central_corrected_UNION.tsv")
))

# ccRCC PROTEOMICS
experiment_results.append(utils.ExperimentResult(
    name="ccRCC E. coli",
    federated_result_file=os.path.join(base_dir, "ccRCC_studies", "after", "FedApp_corrected_data.tsv"),
    central_result_file=os.path.join(base_dir, "ccRCC_studies", "after", "intensities_log_Rcorrected_UNION.tsv"),
))
experiment_results.append(utils.ExperimentResult(
    name="ccRCC E. coli (SMPC)",
    federated_result_file=os.path.join(base_dir, "ccRCC_studies", "after", "FedApp_corrected_data_smpc.tsv"),
    central_result_file=os.path.join(base_dir, "ccRCC_studies", "after", "intensities_log_Rcorrected_UNION.tsv"),
))


### MAIN, just runs compare_experiments on all experiments and prints the results
result_df = utils.compare_experiments(experiment_results)
print("Final results:")
if result_df is None:
    print("ERROR: No comparison could be made.")
    exit(1)
print(result_df)
# get the failed experiments
failed_experiments = result_df[result_df[utils.RESULT_DF_COLUMNS[3]] > utils.FAILURE_THRESHOLD]["Experiment"].tolist()
print(f"Failed experiments:\n{failed_experiments}")
print("saving results to evaluation_data/fed_vc_cent_results.tsv")
result_df.to_csv(os.path.join(base_dir, "fed_vc_cent_results.tsv"), sep="\t", index=False)
