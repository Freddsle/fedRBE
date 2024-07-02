import pandas as pd
import os
import numpy as np
from typing import List, Union, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
from dataclasses import dataclass

base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evaluation_data")

@dataclass
class ExperimentResult():
    name: str
    central_result_file: str
    federated_result_file: str
    complete_analysis: bool = False


def main():
    exp = ExperimentResult(
        name="Microarray_UNION",
        central_result_file=os.path.join(base_dir, "microarray", "after", "central_corrected_UNION.tsv"),
        federated_result_file=os.path.join(base_dir, "microarray", "after", "federated_corrected_UNION.csv"),
        complete_analysis=True,
    )

    # Get features etc
    df_cent = pd.read_csv(exp.central_result_file, sep="\t", index_col=0)
    df_fed = pd.read_csv(exp.federated_result_file, sep="\t", index_col=0)

    old_features, old_samples = get_old_features_and_samples(pd.read_csv(os.path.join(base_dir, "microarray", "after", "central_corrected.tsv"), sep="\t", index_col=0))
    intersect_features, total_features, union_samples = get_features_and_samples(
        os.path.join(base_dir, "microarray", "before"), "GSE", "expr_for_correction_UNION.tsv")
    non_intersect_features = total_features - intersect_features

    # ensure that intersect_features are the same as old_features
    if not intersect_features == old_features:
        print("intersect features are not the same as old features")
        print(intersect_features.difference(old_features))
        print(old_features.difference(intersect_features))
        exit(1)

    # get differences
    # check all features
    print(f"__________________________ALL_________________________")
    diff_df = compare_dfs(df_cent, "central", df_fed, "federated")
    analyse_diff_df(diff_df)
    # check intersect features
    print(f"__________________________INTERSECT_________________________")
    diff_df = compare_dfs(df_cent.loc[list(intersect_features)], "central", df_fed.loc[list(intersect_features)], "federated")
    analyse_diff_df(diff_df)
    # check non intersect features
    print(f"__________________________NON INTERSECT_________________________")
    diff_df = compare_dfs(df_cent.loc[list(non_intersect_features)], "central", df_fed.loc[list(non_intersect_features)], "federated")
    analyse_diff_df(diff_df)
    # check intersect features and old_samples
    print(f"__________________________INTERSECT AND OLD SAMPLES_________________________")
    diff_df = compare_dfs(df_cent.loc[list(intersect_features), list(old_samples)], "central", df_fed.loc[list(intersect_features), list(old_samples)], "federated")
    analyse_diff_df(diff_df)
    # check non intersect features and old_samples
    print(f"__________________________NON INTERSECT AND OLD SAMPLES_________________________")
    diff_df = compare_dfs(df_cent.loc[list(non_intersect_features), list(old_samples)], "central", df_fed.loc[list(non_intersect_features), list(old_samples)], "federated")
    analyse_diff_df(diff_df)


def get_old_features_and_samples(df_cent_old: pd.DataFrame):
    """
    Loads the old results and gets the features and samples
    """
    # get the intersect features and rows from the old intersect app results
    old_features = set(df_cent_old.index)
    old_samples = set(df_cent_old.columns)
    return old_features, old_samples

def get_features_and_samples(input_folder, client_folder_prefix, client_filename):
    """
    Iterates all clients and finds the intersection and union of all features
    available to all clients.
    Also retrieves the union of all samples.
    Args:
        Input_folder: str
            The folder containing the clients
        client_folder_prefix: str
            The prefix of the client folders, used to identify which folder in
            the input_folder belongs to a client
        client_filename: str
            The filename of the datafile in the client folder
    Returns:
        intersect_features: set
            The intersect features of all clients
        union_features: set
            The union features of all clients
        union_samples: set
            The union samples of all clients
    """
    total_features = set()
    intersect_features = set()
    union_samples = set()

    for clientfolder in os.listdir(input_folder):
        if clientfolder.startswith(client_folder_prefix):
            data_file = os.path.join(input_folder, clientfolder, client_filename)
            df = pd.read_csv(data_file, sep="\t", index_col=0)
            # clean of rows with only NaNs
            df = df.dropna(how="all")
            if len(intersect_features) == 0:
                intersect_features = set(df.index)
                total_features = set(df.index)
            else:
                intersect_features = intersect_features.intersection(set(df.index))
                total_features = total_features.union(set(df.index))
            union_samples = union_samples.union(set(df.columns))
    return intersect_features, total_features, union_samples


def analyse_diff_df(diff_df: pd.DataFrame, n=2):
    """
    Analyses the df, prints out:
        Element wise:
            - mean
            - max
        Highest n indexes:
            - mean
            - max
        Highest n columns:
            - mean
            - max
    Args:
        diff_df: pd.DataFrame
            The dataframe to analyse
    Returns:
        None
    """
    # element wise comparison
    print(f"Element wise comparison")
    print(f"Mean: {diff_df.mean().mean()}")
    print(f"Max: {diff_df.max().max()}")

    # highest 10 indexes
    print(f"Highest {n} indexes")
    print(f"Mean: {diff_df.mean(axis=1).nlargest(n)}")
    print(f"Max: {diff_df.max(axis=1).nlargest(n)}")

    # highest 10 columns
    print(f"Highest {n} columns")
    print(f"Mean: {diff_df.mean(axis=0).nlargest(n)}")
    print(f"Max: {diff_df.max(axis=0).nlargest(n)}")


def compare_dfs(df1, df1_name, df2, df2_name) -> pd.DataFrame:
    """
    Compare the two dataframes df1 and df2.
    Checks that:
        - rows are identical (order does not matter)
        - columns are identical (order does not matter)
        - finds the differences between the two dataframes elementwise
    Args:
        df1: pd.DataFrame
            The first dataframe to compare
        df1_name: str
            The name of the first dataframe
        df2: pd.DataFrame
            The second dataframe to compare
        df2_name: str
            The name of the second dataframe
    Returns:
        None if error, otherwise a dtaframe of the same shape as the input dataframes
        containing the differences between the two dataframes
    """
    print(f"{df1_name} shape: {df1.shape}")
    print(f"{df2_name} shape: {df2.shape}")

    ### Order the dataframes ###
    df1 = df1.sort_index(axis=0).sort_index(axis=1)
    df2 = df2.sort_index(axis=0).sort_index(axis=1)

    ### Compare columns and rows ###
    failed = False
    if not df1.columns.equals(df2.columns):
        print(f"Columns do not match for {df1_name} and {df2_name}")
        union_cols = df1.columns.union(df2.columns)
        intercept_cols = df1.columns.intersection(df2.columns)
        print(f"Union-Intercept of columns: {union_cols.difference(intercept_cols)}")
        failed = True
    if not df1.index.equals(df2.index):
        print(f"Rows do not match for {df1_name} and {df2_name}")
        union_rows = df1.index.union(df2.index)
        intercept_rows = df1.index.intersection(df2.index)
        print(f"Union-Intercept of rows: {union_rows.difference(intercept_rows)}")
        failed = True
    if failed:
        print(f"_________________________FAILED {df1_name} and {df2_name}_________________________")
        raise ValueError("Columns or rows do not match")

    ### Compare value by value ###
    # Ensure that the dataframes are sorted in index and columns
    df1 = df1.sort_index(axis=0).sort_index(axis=1)
    df2 = df2.sort_index(axis=0).sort_index(axis=1)

    # Get the differences
    df_diff = df1.combine(df2, custom_diff_columns)

    return df_diff

def custom_diff_columns(a, b):
    """
    Calculate the difference between two columns.
    Raises an error if only one val in the column is NaN, while the other is not.
    If both values in the columns are NaN, the difference is 0 as they are equal.
    Args:
        a: pd.Series
            The first column to compare
        b: pd.Series
            The second column to compare
    Returns:
        The absolute difference between the two columns
    """
    return a.combine(b, custom_diff_values)

def custom_diff_values(a, b):
    """
    Calculate the difference between two values.
    Raises an error if only one value is NaN, while the other is not.
    If both values are NaN, the difference is 0 as they are equal.
    Args:
        a: float
            The first value to compare
        b: float
            The second value to compare
    Returns:
        The absolute difference between the two values
    """
    if pd.isna(a) and pd.isna(b):
        return 0
    elif pd.isna(a) ^ pd.isna(b):  # XOR operation
        raise ValueError("Only one value in the dfs to compare is NaN")
    else:
        return abs(a - b)


if __name__ == "__main__":
    main()


# def compare_experiment(exp: ExperimentResult, intersect_features: Set[str]):
#     print(f"_________________________Comparing {exp.name}_________________________")
#     central_df = pd.read_csv(exp.central_result_file, sep="\t", index_col=0)
#     federated_df = pd.read_csv(exp.federated_result_file, sep="\t", index_col=0)
#     print(f"shape of CENTRAL: {central_df.shape}")
#     print(f"shape of FEDERATED: {federated_df.shape}")
#     central_df = central_df.sort_index(axis=0).sort_index(axis=1)
#     federated_df = federated_df.sort_index(axis=0).sort_index(axis=1)
#     ### compare columns and index ###
#     failed = False
#     if not central_df.columns.equals(federated_df.columns):
#         print(f"Columns do not match for {exp.central_result_file} and {exp.federated_result_file}")
#         union_cols = central_df.columns.union(federated_df.columns)
#         intercept_cols = central_df.columns.intersection(federated_df.columns)
#         print(f"Union-Intercept of columns: {union_cols.difference(intercept_cols)}")
#         failed = True
#     if not central_df.index.equals(federated_df.index):
#         print(f"Rows do not match for {exp.central_result_file} and {exp.federated_result_file}")
#         union_rows = central_df.index.union(federated_df.index)
#         intercept_rows = central_df.index.intersection(federated_df.index)
#         print(f"Union-Intercept of rows: {union_rows.difference(intercept_rows)}")
#         failed = True
#     if failed:
#         print(f"_________________________FAILED {exp.name}_________________________")
#         return None


#     ### Compare value by value ###
#     # we extarct all differences and perform basic statiscs on them
#     if exp.complete_analysis:
#         # columnwise comparison
#         differences = list()
#         nan_count = 0
#         fed_nan_count = 0
#         central_nan_count = 0
#         result_df = None
#         col2diffsum = dict()
#         for col in central_df.columns:
#             for value1, value2 in zip(central_df[col].values, federated_df[col].values):
#                 nan_count, central_nan_count, fed_nan_count = \
#                     _compare_vals(value1, value2, nan_count, central_nan_count, fed_nan_count, differences, col2diffsum, col)
#         print("Columnwise comparison")
#         _analyse_differences(differences, nan_count, central_nan_count, fed_nan_count, result_df, exp)


#         # rowwise comparison
#         differences = list()
#         nan_count = 0
#         fed_nan_count = 0
#         central_nan_count = 0
#         result_df = None
#         row2diffsum = dict()
#         for row in central_df.index:
#             for value1, value2 in zip(central_df.loc[row].values, federated_df.loc[row].values):
#                 nan_count, central_nan_count, fed_nan_count = \
#                     _compare_vals(value1, value2, nan_count, central_nan_count, fed_nan_count, differences, row2diffsum, row)
#         print("Rowwise comparison")
#         _analyse_differences(differences, nan_count, central_nan_count, fed_nan_count, result_df, exp)
#         print("rowwise differences of all features")
#         print([f"{k}: {v}" for k, v in sorted(row2diffsum.items(), key=lambda item: item[1], reverse=True)][:10])
#         print("rowwise differences of only intersect features")
#         print([f"{k}: {v}" for k, v in sorted(row2diffsum.items(), key=lambda item: item[1], reverse=True) if k in intersect_features][:10])
#         print("rowwise differences of only non-intersect features")
#         print([f"{k}: {v}" for k, v in sorted(row2diffsum.items(), key=lambda item: item[1], reverse=True) if k not in intersect_features][:10])

#     # total comparison
#     differences = list()
#     nan_count = 0
#     fed_nan_count = 0
#     central_nan_count = 0
#     total_count = 0
#     result_df = None # reset, we only show this in the end
#     for value1, value2 in zip(central_df.values.flatten(), federated_df.values.flatten()):
#         total_count += 1
#         nan_count, central_nan_count, fed_nan_count = \
#             _compare_vals(value1, value2, nan_count, central_nan_count, fed_nan_count, differences)
#     print("Total comparison")
#     print(f"Total count: {total_count}")
#     _analyse_differences(differences, nan_count, central_nan_count, fed_nan_count, result_df, exp)
#     return result_df

# def _analyse_differences(differences: List[float], nan_count: int, central_nan_count: int, fed_nan_count: int,
#                          result_df: Union[pd.DataFrame, None], exp: ExperimentResult,
#                          plot: bool = False):
#     print(f"Maximal difference: {np.max(differences)}")
#     print(f"Mean difference: {np.mean(differences)}")
#     print(f"Number of NaN values: {nan_count} (Central: {central_nan_count}, Federated: {fed_nan_count})")
#     if result_df is None:
#         result_df = pd.DataFrame({
#             "Experiment": [exp.name],
#             "Maximal difference": [np.max(differences)],
#             "Mean difference": [np.mean(differences)],
#         })
#     else:
#         result_df.loc[len(result_df)] = [exp.name, np.max(differences), np.mean(differences)]

#     # show the differences as a plot, y-axis are differences, x-axis are the index of the differences
#     if plot:
#         plt.hist(differences, bins=100)
#         plt.title(f"Differences of {exp.name}")
#         plt.xlabel("Element of data")
#         plt.ylabel("Difference")
#         plt.show()



# ### MAIN
# if __name__ == "__main__":
#     exp = ExperimentResult(
#         name="Microarray_UNION",
#         central_result_file=os.path.join(base_dir, "microarray", "after", "central_corrected_UNION.tsv"),
#         federated_result_file=os.path.join(base_dir, "microarray", "after", "federated_corrected_UNION.csv"),
#         complete_analysis=True,
#     )

#     # get the intercept features
#     total_features = set()
#     intersect_features = set()
#     input_folder = os.path.join(base_dir, "microarray", "before")
#     client2features = dict()
#     for clientfolder in os.listdir(input_folder):
#         if clientfolder.startswith("GSE"):
#             data_file = os.path.join(input_folder, clientfolder, "expr_for_correction_UNION.tsv")
#             df = pd.read_csv(data_file, sep="\t", index_col=0)
#             # clean of rows with only NaNs
#             df = df.dropna(how="all")
#             client2features[clientfolder] = set(df.index)
#             if len(intersect_features) == 0:
#                 intersect_features = set(df.index)
#                 total_features = set(df.index)
#             else:
#                 intersect_features = intersect_features.intersection(set(df.index))
#                 total_features = total_features.union(set(df.index))

#     # # find features that are only in ONE client
#     # only_one_client_features = None
#     # for client, features in client2features.items():
#     #     if only_one_client_features is None:
#     #         only_one_client_features = features
#     #     else:
#     #         only_one_client_features = only_one_client_features.symmetric_difference(features)

#     # # analyse the experiment
#     # if total_features == intersect_features:
#     #     print("All features are the same!!!")
#     #     exit(1)
#     # print(f"there are {len(only_one_client_features)} features that are only in one client")
#     # result_df = compare_experiment(exp, intersect_features=intersect_features)


#     # # try comparing with the old results
#     # print(f"_________________________Comparing {exp.name} WITH OLD RESULT_________________________")
#     # central_df = pd.read_csv(os.path.join(base_dir, "microarray", "after", "central_corrected.tsv"), sep="\t", index_col=0)
#     # federated_df = pd.read_csv(exp.federated_result_file, sep="\t", index_col=0)
#     # print(f"shape of CENTRAL: {central_df.shape}")
#     # print(f"shape of FEDERATED: {federated_df.shape}")
#     # central_df = central_df.sort_index(axis=0).sort_index(axis=1)
#     # federated_df = federated_df.sort_index(axis=0).sort_index(axis=1)

#     # # identify common rows and columns
#     # common_rows = central_df.index.intersection(federated_df.index)
#     # common_cols = central_df.columns.intersection(federated_df.columns)
#     # central_df = central_df.loc[common_rows, common_cols]
#     # federated_df = federated_df.loc[common_rows, common_cols]

#     # central_df = central_df.sort_index(axis=0).sort_index(axis=1)
#     # federated_df = federated_df.sort_index(axis=0).sort_index(axis=1)

#     # # compare
#     # differences = list()
#     # nan_count = 0
#     # fed_nan_count = 0
#     # central_nan_count = 0
#     # result_df = None
#     # for value1, value2 in zip(central_df.values.flatten(), federated_df.values.flatten()):
#     #     nan_count, central_nan_count, fed_nan_count = \
#     #         _compare_vals(value1, value2, nan_count, central_nan_count, fed_nan_count, differences)
#     # print("Total comparison")
#     # print(f"Total count: {central_df.size}")
#     # _analyse_differences(differences, nan_count, central_nan_count, fed_nan_count, result_df, exp)

#     # # compare the common_rows and common_cols with the new result
#     # print(f"_________________________Comparing {exp.name} WITH NEW RESULT - ONLY OLD RESULTS ROWS AND COLS_________________________")
#     # central_df = pd.read_csv(exp.central_result_file, sep="\t", index_col=0)
#     # federated_df = pd.read_csv(exp.federated_result_file, sep="\t", index_col=0)
#     # central_df = central_df.loc[common_rows, common_cols]
#     # federated_df = federated_df.loc[common_rows, common_cols]
#     # central_df = central_df.sort_index(axis=0).sort_index(axis=1)
#     # federated_df = federated_df.sort_index(axis=0).sort_index(axis=1)
#     # print(f"shape of CENTRAL: {central_df.shape}")
#     # print(f"shape of FEDERATED: {federated_df.shape}")
#     # central_df = central_df.sort_index(axis=0).sort_index(axis=1)
#     # federated_df = federated_df.sort_index(axis=0).sort_index(axis=1)

#     #  # compare
#     # differences = list()
#     # nan_count = 0
#     # fed_nan_count = 0
#     # central_nan_count = 0
#     # result_df = None
#     # for value1, value2 in zip(central_df.values.flatten(), federated_df.values.flatten()):
#     #     nan_count, central_nan_count, fed_nan_count = \
#     #         _compare_vals(value1, value2, nan_count, central_nan_count, fed_nan_count, differences)
#     # print("Total comparison")
#     # print(f"Total count: {central_df.size}")
#     # _analyse_differences(differences, nan_count, central_nan_count, fed_nan_count, result_df, exp)

#     # print(f"len of common_rows: {len(common_rows)}")
#     # print(f"len of intersect_features: {len(intersect_features)}")
#     # print(f"{len(common_rows.intersection(intersect_features))} features are in intersect_features and in common_rows")

#     # manual code of comparing
#     central_df = pd.read_csv(exp.central_result_file, sep="\t", index_col=0)
#     federated_df = pd.read_csv(exp.federated_result_file, sep="\t", index_col=0)

#     # get central cols and rows
#     central_cols = central_df.columns
#     central_rows = central_df.index

#     differences = dict()
#     for col in central_cols:
#         for row in central_rows:
#             if np.isnan(central_df.loc[row, col]) and np.isnan(federated_df.loc[row, col]):
#                 continue
#             diff = np.abs(central_df.loc[row, col] - federated_df.loc[row, col])
#             differences[(row, col)] = diff
