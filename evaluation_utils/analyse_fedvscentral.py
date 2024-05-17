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

@dataclass
class ExperimentResult():
    name: str
    central_result_file: str
    federated_result_file: str


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
        # compare columns and index
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
        # Compare value by value
        # we extarct all differences and perform basic statiscs on them
        differences = list()

        for value1, value2 in zip(central_df.values.flatten(), federated_df.values.flatten()):
            if np.isnan(value1) or np.isnan(value2):
                if np.isnan(value1) and np.isnan(value2):
                    # if both are NaN there is no difference
                    # we don't consider these differences at all as they would
                    # change the mean values
                    continue
                elif np.isnan(value1):
                    print(f"Central result contains NaN: {value2}, federated value: {value2}")
                elif np.isnan(value2):
                    print(f"Federated result contains NaN: {value1}, central value: {value1}")
            if value1 != value2:
                differences.append(np.abs(value1 - value2))
        print(f"Maximal difference: {np.max(differences)}")
        print(f"Mean difference: {np.mean(differences)}")
        if result_df is None:
            result_df = pd.DataFrame({
                "Experiment": [exp.name],
                "Maximal difference": [np.max(differences)],
                "Mean difference": [np.mean(differences)],
            })
        else:
            result_df.loc[len(result_df)] = [exp.name, np.max(differences), np.mean(differences)]
    return result_df

### EXPERIMENT RESULTS
base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evaluation_data")
experiment_results = list()

## Microbiome
# logmin_counts_5centers_corrected
experiment_results.append(ExperimentResult(
    name="Microbiome",
    central_result_file=os.path.join(base_dir, "microbiome", "after", "normalized_logmin_counts_5centers_corrected_CENTRAL.tsv"),
    federated_result_file=os.path.join(base_dir, "microbiome", "after", "normalized_logmin_counts_5centers_corrected_FEDERATED.tsv"),
))
# logmin_counts_5centers_corrected_smpc
experiment_results.append(ExperimentResult(
    name="Microbiome_smpc",
    central_result_file=os.path.join(base_dir, "microbiome", "after", "normalized_logmin_counts_5centers_corrected_CENTRAL.tsv"),
    federated_result_file=os.path.join(base_dir, "microbiome", "after", "normalized_logmin_counts_5centers_corrected_smpc_FEDERATED.tsv"),
))

## Proteomics
# balanced
experiment_results.append(ExperimentResult(
    name="Proteomics_balanced",
    central_result_file=os.path.join(base_dir, "proteomics", "after", "balanced", "central_intensities_log_filtered_corrected.tsv"),
    federated_result_file=os.path.join(base_dir, "proteomics", "after", "balanced", "federated_intensities_log_filtered_corrected.tsv"),
))
# balanced_smpc
experiment_results.append(ExperimentResult(
    name="Proteomics_balanced_smpc",
    central_result_file=os.path.join(base_dir, "proteomics", "after", "balanced", "central_intensities_log_filtered_corrected.tsv"),
    federated_result_file=os.path.join(base_dir, "proteomics", "after", "balanced", "federated_intensities_log_filtered_corrected_smpc.tsv"),
))
# imbalanced
experiment_results.append(ExperimentResult(
    name="Proteomics_imbalanced",
    central_result_file=os.path.join(base_dir, "proteomics", "after", "imbalanced", "central_intensities_log_filtered_corrected.tsv"),
    federated_result_file=os.path.join(base_dir, "proteomics", "after", "imbalanced", "federated_intensities_log_filtered_corrected.tsv"),
))
# imbalanced_smpc
experiment_results.append(ExperimentResult(
    name="Proteomics_imbalanced_smpc",
    central_result_file=os.path.join(base_dir, "proteomics", "after", "imbalanced", "central_intensities_log_filtered_corrected.tsv"),
    federated_result_file=os.path.join(base_dir, "proteomics", "after", "imbalanced", "federated_intensities_log_filtered_corrected_smpc.tsv"),
))

## Microarray
# default
experiment_results.append(ExperimentResult(
    name="Microarray",
    central_result_file=os.path.join(base_dir, "microarray", "after", "central_corrected.tsv"),
    federated_result_file=os.path.join(base_dir, "microarray", "after", "federated_corrected.csv"),
))
# smpc
experiment_results.append(ExperimentResult(
    name="Microarray_smpc",
    central_result_file=os.path.join(base_dir, "microarray", "after", "central_corrected.tsv"),
    federated_result_file=os.path.join(base_dir, "microarray", "after", "federated_corrected_smpc.csv"),
))

### MAIN, just runs compare_experiments on all experiments and prints the results
result_df = compare_experiments(experiment_results)
print("Final results:")
print(result_df)
if result_df is not None:
    print("saving results to evaluation_data/fed_vc_cent_results.tsv")
    result_df.to_csv(os.path.join(base_dir, "fed_vc_cent_results.tsv"), sep="\t", index=False)