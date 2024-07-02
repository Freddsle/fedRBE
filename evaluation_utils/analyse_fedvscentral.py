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
import pandas as pd
import os
import numpy as np
from typing import List, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt

@dataclass
class ExperimentResult():
    name: str
    central_result_file: str
    federated_result_file: str
    complete_analysis: bool = False


def compare_experiments(experiment_results: List[ExperimentResult]) -> Union[pd.DataFrame, None]:
    """
    Compare the results of the experiments.
    Compares the dataframe represented by the central_result_file and the federated_result_file.
    Checks that they
    - have the same shape
    - have the same columns
    - have the same rows
    - Checks the maximum and mean difference of the values element-wise
    Args:
        experiment_results: List of ExperimentResult objects
    Returns:
        result_df: DataFrame containing the maximal and mean difference of the experiments
                   columns are ["Experiment", "Maximal difference", "Mean difference"]
                   returns None if no comparison could be made
    """
    result_df = None
    for exp in experiment_results:
        print(f"_________________________Comparing {exp.name}_________________________")
        central_df = pd.read_csv(exp.central_result_file, sep="\t", index_col=0)
        federated_df = pd.read_csv(exp.federated_result_file, sep="\t", index_col=0)
        print(f"shape of CENTRAL: {central_df.shape}")
        print(f"shape of FEDERATED: {federated_df.shape}")
        central_df = central_df.sort_index(axis=0).sort_index(axis=1)
        federated_df = federated_df.sort_index(axis=0).sort_index(axis=1)
        ### compare columns and index ###
        failed = False
        if not central_df.columns.equals(federated_df.columns):
            print(f"Columns do not match for {exp.central_result_file} and {exp.federated_result_file}")
            union_cols = central_df.columns.union(federated_df.columns)
            intercept_cols = central_df.columns.intersection(federated_df.columns)
            print(f"Union-Intercept of columns: {union_cols.difference(intercept_cols)}")
            failed = True
        if not central_df.index.equals(federated_df.index):
            print(f"Rows do not match for {exp.central_result_file} and {exp.federated_result_file}")
            union_rows = central_df.index.union(federated_df.index)
            intercept_rows = central_df.index.intersection(federated_df.index)
            print(f"Union-Intercept of rows: {union_rows.difference(intercept_rows)}")
            failed = True
        if failed:
            print(f"_________________________FAILED {exp.name}_________________________")
            continue


        ### Compare value by value ###
        # we extarct all differences and perform basic statiscs on them
        if exp.complete_analysis:
            # columnwise comparison
            differences = list()
            nan_count = 0
            fed_nan_count = 0
            central_nan_count = 0
            result_df = None
            col2diffsum = dict()
            for col in central_df.columns:
                for value1, value2 in zip(central_df[col].values, federated_df[col].values):
                    nan_count, central_nan_count, fed_nan_count = \
                        _compare_vals(value1, value2, nan_count, central_nan_count, fed_nan_count, differences, col2diffsum, col)
            print("Columnwise comparison")
            _analyse_differences(differences, nan_count, central_nan_count, fed_nan_count, result_df, exp, col2diffsum)


            # rowwise comparison
            differences = list()
            nan_count = 0
            fed_nan_count = 0
            central_nan_count = 0
            result_df = None
            row2diffsum = dict()
            for row in central_df.index:
                for value1, value2 in zip(central_df.loc[row].values, federated_df.loc[row].values):
                    nan_count, central_nan_count, fed_nan_count = \
                        _compare_vals(value1, value2, nan_count, central_nan_count, fed_nan_count, differences, row2diffsum, row)
            print("Rowwise comparison")
            _analyse_differences(differences, nan_count, central_nan_count, fed_nan_count, result_df, exp, row2diffsum)

        # total comparison
        differences = list()
        nan_count = 0
        fed_nan_count = 0
        central_nan_count = 0
        result_df = None # reset, we only show this in the end
        for value1, value2 in zip(central_df.values.flatten(), federated_df.values.flatten()):
            nan_count, central_nan_count, fed_nan_count = \
                _compare_vals(value1, value2, nan_count, central_nan_count, fed_nan_count, differences)
        print("Total comparison")
        _analyse_differences(differences, nan_count, central_nan_count, fed_nan_count, result_df, exp)
    return result_df

def _analyse_differences(differences: List[float], nan_count: int, central_nan_count: int, fed_nan_count: int,
                         result_df: Union[pd.DataFrame, None], exp: ExperimentResult, feature2diffsum: Union[dict, None] = None,
                         plot: bool = False):
    print(f"Maximal difference: {np.max(differences)}")
    print(f"Mean difference: {np.mean(differences)}")
    print(f"Number of NaN values: {nan_count} (Central: {central_nan_count}, Federated: {fed_nan_count})")
    if result_df is None:
        result_df = pd.DataFrame({
            "Experiment": [exp.name],
            "Maximal difference": [np.max(differences)],
            "Mean difference": [np.mean(differences)],
        })
    else:
        result_df.loc[len(result_df)] = [exp.name, np.max(differences), np.mean(differences)]

    # show the top 10 differences from feature2diffsum
    if feature2diffsum is not None:
        print("Top 10 differences")
        for k, v in sorted(feature2diffsum.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"Feature: {k}, Difference: {v}")

    # show the differences as a plot, y-axis are differences, x-axis are the index of the differences
    if plot:
        plt.hist(differences, bins=100)
        plt.title(f"Differences of {exp.name}")
        plt.xlabel("Element of data")
        plt.ylabel("Difference")
        plt.show()

def _compare_vals(value1, value2, nan_count, central_nan_count, fed_nan_count, differences,
                  feature2diffsum=None, feature=None):
    if np.isnan(value1) or np.isnan(value2):
        nan_count += 1
        if np.isnan(value1) and np.isnan(value2):
            # if both are NaN there is no difference
            # we don't consider these differences at all as they would
            # change the mean values
            return nan_count, central_nan_count, fed_nan_count
        elif np.isnan(value1):
            central_nan_count += 1
            print(f"Central result contains NaN: {value2}, federated value: {value2}")
        elif np.isnan(value2):
            fed_nan_count += 1
            print(f"Federated result contains NaN: {value1}, central value: {value1}")
    if value1 != value2:
        diff = np.abs(value1 - value2)
        differences.append(diff)
        if feature2diffsum is not None and feature is not None:
            feature2diffsum[feature] = diff
    return nan_count, central_nan_count, fed_nan_count

### EXPERIMENT RESULTS
base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evaluation_data")
experiment_results = list()

## Microbiome
# logmin_counts_5centers_corrected
# experiment_results.append(ExperimentResult(
#     name="Microbiome",
#     central_result_file=os.path.join(base_dir, "microbiome", "after", "normalized_logmin_counts_5centers_corrected_CENTRAL.tsv"),
#     federated_result_file=os.path.join(base_dir, "microbiome", "after", "normalized_logmin_counts_5centers_corrected_FEDERATED.tsv"),
# ))
# # logmin_counts_5centers_corrected_smpc
# experiment_results.append(ExperimentResult(
#     name="Microbiome_smpc",
#     central_result_file=os.path.join(base_dir, "microbiome", "after", "normalized_logmin_counts_5centers_corrected_CENTRAL.tsv"),
#     federated_result_file=os.path.join(base_dir, "microbiome", "after", "normalized_logmin_counts_5centers_corrected_smpc_FEDERATED.tsv"),
# ))

# ## Proteomics
# # balanced
# experiment_results.append(ExperimentResult(
#     name="Proteomics_balanced",
#     central_result_file=os.path.join(base_dir, "proteomics", "after", "balanced", "central_intensities_log_filtered_corrected.tsv"),
#     federated_result_file=os.path.join(base_dir, "proteomics", "after", "balanced", "federated_intensities_log_filtered_corrected.tsv"),
# ))
# # balanced_smpc
# experiment_results.append(ExperimentResult(
#     name="Proteomics_balanced_smpc",
#     central_result_file=os.path.join(base_dir, "proteomics", "after", "balanced", "central_intensities_log_filtered_corrected.tsv"),
#     federated_result_file=os.path.join(base_dir, "proteomics", "after", "balanced", "federated_intensities_log_filtered_corrected_smpc.tsv"),
# ))
# # imbalanced
# experiment_results.append(ExperimentResult(
#     name="Proteomics_imbalanced",
#     central_result_file=os.path.join(base_dir, "proteomics", "after", "imbalanced", "central_intensities_log_filtered_corrected.tsv"),
#     federated_result_file=os.path.join(base_dir, "proteomics", "after", "imbalanced", "federated_intensities_log_filtered_corrected.tsv"),
# ))
# # imbalanced_smpc
# experiment_results.append(ExperimentResult(
#     name="Proteomics_imbalanced_smpc",
#     central_result_file=os.path.join(base_dir, "proteomics", "after", "imbalanced", "central_intensities_log_filtered_corrected.tsv"),
#     federated_result_file=os.path.join(base_dir, "proteomics", "after", "imbalanced", "federated_intensities_log_filtered_corrected_smpc.tsv"),
# ))

# ## Microarray
# # default
# experiment_results.append(ExperimentResult(
#     name="Microarray",
#     central_result_file=os.path.join(base_dir, "microarray", "after", "central_corrected.tsv"),
#     federated_result_file=os.path.join(base_dir, "microarray", "after", "federated_corrected.csv"),
# ))
# # smpc
# experiment_results.append(ExperimentResult(
#     name="Microarray_smpc",
#     central_result_file=os.path.join(base_dir, "microarray", "after", "central_corrected.tsv"),
#     federated_result_file=os.path.join(base_dir, "microarray", "after", "federated_corrected_smpc.csv"),
# ))

### NEW APP USING ALL FEATURES
experiment_results.append(ExperimentResult(
    name="Microarray_UNION",
    central_result_file=os.path.join(base_dir, "microarray", "after", "central_corrected_UNION.tsv"),
    federated_result_file=os.path.join(base_dir, "microarray", "after", "federated_corrected_UNION.csv"),
    complete_analysis=True,
))

### MAIN, just runs compare_experiments on all experiments and prints the results
result_df = compare_experiments(experiment_results)
print("Final results:")
print(result_df)
if result_df is not None:
    print("saving results to evaluation_data/fed_vc_cent_results.tsv")
    result_df.to_csv(os.path.join(base_dir, "fed_vc_cent_results.tsv"), sep="\t", index=False)
