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
    federated_result_file=os.path.join(base_dir, "ecoli", "after", "label_aware", "FedApp_corrected_data.tsv"),
    central_result_file=os.path.join(base_dir, "ecoli", "after", "label_aware", "intensities_log_Rcorrected_UNION.tsv"),
))
experiment_results.append(utils.ExperimentResult(
    name="E. coli (SMPC)",
    federated_result_file=os.path.join(base_dir, "ecoli", "after", "label_aware", "FedApp_corrected_data_smpc.tsv"),
    central_result_file=os.path.join(base_dir, "ecoli", "after", "label_aware", "intensities_log_Rcorrected_UNION.tsv"),
))

# QUARTET MULTIOMICS
for modality in ["Transcriptomics", "Proteomics", "Metabolomics"]:
    for name_suffix, result_filename in [
        ("", "FedApp_corrected_data.tsv"),
        (" (SMPC)", "FedApp_corrected_data_smpc.tsv"),
    ]:
        experiment_results.append(utils.ExperimentResult(
            name=f"Quartet Multiomics {modality}{name_suffix}",
            federated_result_file=os.path.join(
                base_dir,
                "quartet_multiomics",
                "after",
                "label_aware",
                modality,
                result_filename,
            ),
            central_result_file=os.path.join(
                base_dir,
                "quartet_multiomics",
                "after",
                "label_aware",
                modality,
                "intensities_log_Rcorrected_UNION.tsv",
            ),
        ))

# MICROARRAY
experiment_results.append(utils.ExperimentResult(
    name="Ovarian cancer",
    federated_result_file=os.path.join(base_dir, "ovarian_cancer", "after", "label_aware", "FedApp_corrected_data.tsv"),
    central_result_file=os.path.join(base_dir, "ovarian_cancer", "after", "label_aware", "central_corrected_UNION.tsv")
))
experiment_results.append(utils.ExperimentResult(
    name="Ovarian cancer (SMPC)",
    federated_result_file=os.path.join(base_dir, "ovarian_cancer", "after", "label_aware", "FedApp_corrected_data_smpc.tsv"),
    central_result_file=os.path.join(base_dir, "ovarian_cancer", "after", "label_aware", "central_corrected_UNION.tsv")
))

# ccRCC PROTEOMICS
experiment_results.append(utils.ExperimentResult(
    name="ccRCC proteomics",
    federated_result_file=os.path.join(base_dir, "ccRCC_studies", "after", "label_aware", "FedApp_corrected_data.tsv"),
    central_result_file=os.path.join(base_dir, "ccRCC_studies", "after", "label_aware", "intensities_log_Rcorrected_UNION.tsv"),
))
experiment_results.append(utils.ExperimentResult(
    name="ccRCC proteomics (SMPC)",
    federated_result_file=os.path.join(base_dir, "ccRCC_studies", "after", "label_aware", "FedApp_corrected_data_smpc.tsv"),
    central_result_file=os.path.join(base_dir, "ccRCC_studies", "after", "label_aware", "intensities_log_Rcorrected_UNION.tsv"),
))


# Existing real and simulated comparisons preserve biological labels.
arm_by_experiment = {experiment.name: "label_aware" for experiment in experiment_results}
label_aware_path = os.path.join("after", "label_aware", "")
no_label_path = os.path.join("after", "no_label", "")
for experiment in experiment_results.copy():
    if label_aware_path not in experiment.central_result_file:
        continue  # Simulated outputs have no label-free counterpart.
    no_label = utils.ExperimentResult(
        name=experiment.name + " (no_label)",
        federated_result_file=experiment.federated_result_file.replace(
            label_aware_path, no_label_path),
        central_result_file=experiment.central_result_file.replace(
            label_aware_path, no_label_path),
    )
    experiment_results.append(no_label)
    arm_by_experiment[no_label.name] = "no_label"


### MAIN, just runs compare_experiments on all experiments and prints the results
if __name__ == "__main__":
    result_df = utils.compare_experiments(experiment_results)
    print("Final results:")
    if result_df is None:
        print("ERROR: No comparison could be made.")
        raise SystemExit(1)
    result_df.insert(1, "correction_arm", result_df["Experiment"].map(arm_by_experiment))
    print(result_df)
    mean_difference_column = "Mean difference"
    max_difference_column = "Max difference"
    simulated_rows = result_df[result_df["Experiment"].str.contains("Simulated")]
    other_rows = result_df[~result_df["Experiment"].str.contains("Simulated")]
    print("\nSimulated experiments:")
    print(simulated_rows)
    print("Mean absolute element wise difference (Simulated):", simulated_rows[mean_difference_column].mean())
    print("Max absolute element wise difference (Simulated):", simulated_rows[max_difference_column].max())
    print("\nOther experiments:")
    print(other_rows)
    print("Mean absolute element wise difference (Other):", other_rows[mean_difference_column].mean())
    print("Max absolute element wise difference (Other):", other_rows[max_difference_column].max())
    # get the failed experiments
    failed_experiments = result_df[result_df[utils.RESULT_DF_COLUMNS[3]] > utils.FAILURE_THRESHOLD]["Experiment"].tolist()
    print(f"Failed experiments:\n{failed_experiments}")
    print("saving results to evaluation_data/fed_vc_cent_results.tsv")
    result_df.to_csv(os.path.join(base_dir, "fed_vc_cent_results.tsv"), sep="\t", index=False)
