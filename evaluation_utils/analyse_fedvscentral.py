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
import utils_analyse as utils


### EXPERIMENT RESULTS
base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evaluation_data")
experiment_results = list()
    # Add all results to this list as utils.ExperimentResult objects
    # basically specifying the corrected data files of the central and federated results

# MICROBIOME
# experiment_results.append(utils.ExperimentResult(
#     name="Microbiome",
#     federated_result_file=os.path.join(base_dir, "microbiome", "after", "FedApp_corrected_data.tsv"),
#     central_result_file=os.path.join(base_dir, "microbiome", "after", "normalized_logmin_counts_5centers_Rcorrected.tsv"),
# ))

# PROTEOMICS
experiment_results.append(utils.ExperimentResult(
    name="Proteomics",
    federated_result_file=os.path.join(base_dir, "proteomics", "after", "FedApp_corrected_data.tsv"),
    central_result_file=os.path.join(base_dir, "proteomics", "after", "intensities_log_Rcorrected_UNION.tsv"),
    complete_analysis=True,
))

# MICROARRAY
# experiment_results.append(utils.ExperimentResult(
#     name="Microarray",
#     federated_result_file=os.path.join(base_dir, "microarray", "after", "FedApp_corrected_data.tsv"),
#     central_result_file=os.path.join(base_dir, "microarray", "after", "central_corrected_UNION.tsv")
# ))

# SIMULATED
# experiment_results.append(utils.ExperimentResult(
#     name="Simulated Balanced",
#     federated_result_file=os.path.join(base_dir, "simulated", "balanced", "after", "FedApp_corrected_data.tsv"),
#     central_result_file=os.path.join(base_dir, "simulated", "balanced", "after", "intensities_R_corrected.tsv"),
# ))

# experiment_results.append(utils.ExperimentResult(
#     name="Simulated Mildly Imbalanced",
#     federated_result_file=os.path.join(base_dir, "simulated", "mild_imbalanced", "after", "FedApp_corrected_data.tsv"),
#     central_result_file=os.path.join(base_dir, "simulated", "mild_imbalanced", "after", "intensities_R_corrected.tsv"),
# ))

# experiment_results.append(utils.ExperimentResult(
#     name="Simulated Strongly Imbalanced",
#     federated_result_file=os.path.join(base_dir, "simulated", "strong_imbalanced", "after", "FedApp_corrected_data.tsv"),
#     central_result_file=os.path.join(base_dir, "simulated", "strong_imbalanced", "after", "intensities_R_corrected.tsv"),
# ))


### MAIN, just runs compare_experiments on all experiments and prints the results
result_df = utils.compare_experiments(experiment_results)
print("Final results:")
print(result_df)
if result_df is not None:
    print("saving results to evaluation_data/fed_vc_cent_results.tsv")
    result_df.to_csv(os.path.join(base_dir, "fed_vc_cent_results.tsv"), sep="\t", index=False)
