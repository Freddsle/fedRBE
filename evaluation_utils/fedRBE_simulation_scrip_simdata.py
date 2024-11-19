#!/usr/bin/env python3

### Imports ###
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import argparse
from typing import List, Tuple, Union, Dict
from batchcorrection.classes.client import Client
from batchcorrection.classes.coordinator_utils import select_common_features_variables, \
    compute_beta, create_beta_mask

import numpy as np
import pandas as pd
import time

### Helper Functions ###
### Just run this, these functions are needed in various places ###

### Define the client class ###
class ClientWrapper:
    """
    Holds all information necessary for the simulation run to work.
    Defines the input data of the client, it's name and if it should be
    considered as the coordinator
    """
    def __init__(self, id: str, input_folder: str, coordinator: bool = False):
        self.id = id
        self.input_folder = input_folder
        self.is_coordinator = coordinator
        self.client_class: Client # initiated later
        self.data_corrected: pd.DataFrame # initiated later

def _check_consistency_clientwrappers(clientWrappers: List[ClientWrapper]) -> None:
    """
    Checks for a list of clients if they were created correctly
    Raises a ValueError in case of inconsistencies
    Checks:
        1. If exactly one coordinator exists
    """
    coord = False
    for clientWrapper in clientWrappers:
        if coord and clientWrapper.is_coordinator:
            raise ValueError("More than one coordinator was defined, please check "+\
                            "the code defining the clients")
        if clientWrapper.is_coordinator:
            coord = True
    if not coord:
        raise ValueError("No client instance is a coordinator, please designate "+\
                        "any client as a coordinator")

def _compare_central_federated_dfs(name:str,
                                   central_df: pd.DataFrame,
                                   federated_df: pd.DataFrame,
                                   intersection_features: List[str]) -> None:
    """
    Compares two dataframes for equality. First checks that index and columns
    are the same, then analyses the element wise differences.
    See the analyse_diff_df function for more details on the difference analysis.
    If both dataframes contain a NaN value at the same position, this is considered
    as equal (0 as difference).
    Args:
        name: Name used for printing
        central_df: The central dataframe
        federated_df: The federated dataframe
        intersection_features: The features that are common to both dataframes
    """
    central_df = central_df.sort_index(axis=0).sort_index(axis=1)
    federated_df = federated_df.sort_index(axis=0).sort_index(axis=1)
    print(f"_________________________Analysing: {name}_________________________")
    ### compare columns and index ###
    failed = False
    if not central_df.columns.equals(federated_df.columns):
        print(f"Columns do not match for central_df and federated_df")
        union_cols = central_df.columns.union(federated_df.columns)
        intercept_cols = central_df.columns.intersection(federated_df.columns)
        print(f"Union-Intercept of columns: {union_cols.difference(intercept_cols)}")
        failed = True
    if not central_df.index.equals(federated_df.index):
        print(f"Rows do not match for central_df and federated_df")
        union_rows = central_df.index.union(federated_df.index)
        intercept_rows = central_df.index.intersection(federated_df.index)
        print(f"Union-Intercept of rows: {union_rows.difference(intercept_rows)}")
        failed = True
    if failed:
        print(f"_________________________FAILED: {name}_________________________")

    df_diff = (central_df - federated_df).abs()
    print(f"Max difference: {df_diff.max().max()}")
    print(f"Mean difference: {df_diff.mean().mean()}")
    print(f"Max diff at position: {df_diff.idxmax().idxmax()}")

    df_diff_intersect = df_diff.loc[intersection_features]
    print(f"Max difference in intersect: {df_diff_intersect.max().max()}")
    print(f"Mean difference in intersect: {df_diff_intersect.mean().mean()}")
    print(f"Max diff at position in intersect: {df_diff_intersect.idxmax().idxmax()}")

def _concat_federated_results(clientWrappers: List[ClientWrapper],
                              samples_in_columns=True) -> Tuple[pd.DataFrame, List[str]]:
    """
    Concatenates the results of the federated clients into one dataframe
    Also checks which features are common to all clients
    and returns them
    Args:
        clientWrappers: List of ClientWrapper instances, containing the data
            in the data_corrected attribute
        samples_in_columns: If True, the samples are in the columns, if False
            they are in the rows. For expression files this is true.
            This decides how to aggregate the dataframes
    Returns:
        merged_df: The merged dataframe containing the data of all clients
        intersection_features: The features that are common to all clients
    """
    merged_df = None
    intersection_features = set()
    for clientWrapper in clientWrappers:
        # get the data in the correct format
        if not hasattr(clientWrapper, "data_corrected") or \
            clientWrapper.data_corrected is None:
            raise ValueError("No data was found in the clientWrappers")
        corrected_data = clientWrapper.data_corrected
        if not samples_in_columns:
            corrected_data = corrected_data.T

        cleaned_corrected_features = set(corrected_data.dropna().index)
        # initialize the merged_df
        if merged_df is None:
            merged_df = corrected_data
            intersection_features = cleaned_corrected_features
            continue

        # merge the data
        merged_df = pd.concat([merged_df, corrected_data], axis=1)
        intersection_features = intersection_features.intersection(cleaned_corrected_features)

    # final check
    if merged_df is None:
        raise ValueError("No data was found in the clientWrappers")
    # reverse the Transpose if necessary
    if not samples_in_columns:
        merged_df = merged_df.T
    return merged_df, list(intersection_features)

# ====================================================================================================
# ====================================================================================================

### This part defines the data used. A ClientWrapper class is used to      ###
### describe all cohorts. If other data should be tested, this part should ###
### be changed                                                             ###
### Define the different clients ###
    # we use a helper class for each client, see the helper function
    # code block or the later definitions here for more info

################### Simulation Data ###################

def main():
    parser = argparse.ArgumentParser(description='Run simulation with specified mode.')
    parser.add_argument('mode', type=str, help='Mode to use in simulation (e.g., "balanced", "strong_imbalanced")')
    args = parser.parse_args()

    mode = args.mode

    # First define the basefolder where all files are located
    base_dir = os.path.join("..", "..")
    # Go back to the git repo's root dir
    base_dir = os.path.join(base_dir, "evaluation_data", "simulated", mode, "before")


    print("Starting simulation")
    print(f"Mode: {mode}")
    print("Base directory: ", base_dir)
    # List of cohort names
    cohort_names = [
        'lab1',  # Client 1 (Coordinator)
        'lab2',  # Client 2
        'lab3',  # Client 3 # reference client
    ]

    # Initialize clientWrappers list
    clientWrappers: List[ClientWrapper] = []

    # Iterate over cohort names and create ClientWrapper instances
    for i, cohortname in enumerate(cohort_names):
        clientWrappers.append(ClientWrapper(
            id=cohortname,
            input_folder=os.path.join(base_dir, cohortname),
            coordinator=(i == 0)  # Set the first client as coordinator
        ))

    # Double check that we only have one coordinator
        _check_consistency_clientwrappers(clientWrappers)

    # ====================================================================================================
    # ====================================================================================================
    # ====================================================================================================
    # Analysis of the data
    # ====================================================================================================

    # measure time for all clients
    time_tracker = {}

    ###                                  INFO                                  ###
    ### The following code blocks run the simulation. They are divided into    ###
    ### multiple logical blocks to ease the use                                ###

    ### SIMULATION: all: initial ###
    ### Initial reading of the input folder

    send_feature_variable_batch_info = list()
    for clientWrapper in clientWrappers:
        # define the client class
        cohort_name = clientWrapper.id
        client = Client()
        client.config_based_init(clientname = cohort_name,
                                input_folder = clientWrapper.input_folder,
                                use_hashing = False)
        clientWrapper.client_class = client
        assert isinstance(client.hash2feature, dict)
        assert isinstance(client.hash2variable, dict)
        batch_feature_presence_info: Dict[str, List[str]] = client.get_batch_feature_presence_info()
        send_feature_variable_batch_info.append((cohort_name,
                                            list(client.hash2variable.keys()),
                                            client.position,
                                            batch_feature_presence_info))
        
        
    ### SIMULATION: Coordinator: global_feature_selection ###
    ### Aggregate the features and variables

    # obtain and safe common genes and indices of design matrix
    # wait for each client to send the list of genes they have
    # also memo the feature presence matrix and feature_to_cohorts

    broadcast_features_variables = tuple()
    for clientWrapper in clientWrappers:
        if clientWrapper.is_coordinator:

            time_tracker["Coordinator"] = time.time()

            global_feature_names, global_variables, feature_presence_matrix, cohorts_order = \
                select_common_features_variables(
                    feature_variable_batch_info=send_feature_variable_batch_info,
                    min_clients=1,
                    default_order=cohort_names
                )

            # memo the feature presence matrix and feature_to_cohorts
            broadcast_features_variables = global_feature_names, global_variables, cohorts_order
            print(f"Got this final cohort order: {cohorts_order}")
            end_time = time.time()
            time_tracker["Coordinator"] = end_time - time_tracker["Coordinator"]

            
    ### SIMULATION: All: validate ###
    ### Expand data to fullfill the global format. Also performs consistency checks
    cols = []
    for clientWrapper in clientWrappers:

        time_tracker[clientWrapper.id] = time.time()

        global_feauture_names_hashed, global_variables_hashed, cohorts_order = \
            broadcast_features_variables
        client = clientWrapper.client_class
        client.validate_inputs(global_variables_hashed)
        client.set_data(global_feauture_names_hashed)

        err = client.create_design(cohorts_order)
        cols.append(client.design.columns)
            # we choose the last batch of the last cohort as reference batch
        if err:
            raise ValueError(err)

        end_time = time.time()
        time_tracker[clientWrapper.id] = end_time - time_tracker[clientWrapper.id]
    for col in cols:
        print(col)
        assert cols[0].equals(col)


    ### Simulatuion: Coordinator: create design mask based on feature presence matrix ###
    ### Create the mask for the design matrix based on the feature presence matrix
    ### that will be used for the beta computation
    for clientWrapper in clientWrappers:
        if clientWrapper.is_coordinator:
            start_time = time.time()
            client = clientWrapper.client_class

            n=len(client.feature_names)
            k=client.design.shape[1]

            global_mask = create_beta_mask(feature_presence_matrix, n, k)
            # memo the global mask

            end_time = time.time()
            time_tracker["Coordinator"] += end_time - start_time


    ### SIMULATION: All: prepare for compute_XtX_XtY ###

    for clientWrapper in clientWrappers:
        start_time = time.time()

        client = clientWrapper.client_class
        client.sample_names = client.design.index.values

        # Error check if the design index and the data index are the same
        # we check by comparing the sorted indexes
        client._check_consistency_designfile()

        # Extract only relevant (the global) features and samples
        client.data = client.data.loc[client.feature_names, client.sample_names]
        client.n_samples = len(client.sample_names)

        end_time = time.time()
        time_tracker[clientWrapper.id] += end_time - start_time


    ### SIMULATION: All: compute_XtX_XtY ###
    ### Compute XtX and XtY and share it
    send_XtX_XtY_list: List[List[np.ndarray]] = list()
    for clientWrapper in clientWrappers:
        start_time = time.time()

        client = clientWrapper.client_class

        # compute XtX and XtY
        XtX, XtY, err = client.compute_XtX_XtY()
        if err != "":
            raise ValueError(err)

        # send XtX and XtY
        send_XtX_XtY_list.append([XtX, XtY])

        end_time = time.time()
        time_tracker[clientWrapper.id] += end_time - start_time


    ### SIMULATION: Coordinator: compute_beta
    ### Compute the beta values and broadcast them to the others
    broadcast_betas = None # np.ndarray of shape num_features x design_columns

    for clientWrapper in clientWrappers:
        if clientWrapper.is_coordinator:

            start_time = time.time()

            client = clientWrapper.client_class
            beta = compute_beta(XtX_XtY_list=send_XtX_XtY_list,
                                n=len(client.feature_names),
                                k=client.design.shape[1],
                                global_mask=global_mask)

            # send beta to clients so they can correct their data
            broadcast_betas = beta

            end_time = time.time()
            time_tracker["Coordinator"] += end_time - start_time

    ### SIMULATION: All: include_correction
    ### Corrects the individual data
    for clientWrapper in clientWrappers:

        start_time = time.time()

        client = clientWrapper.client_class

        # remove the batch effects in own data and safe the results
        client.remove_batch_effects(beta)
        print(f"DEBUG: Shape of corrected data: {client.data_corrected.shape}")

        end_time = time.time()
        time_tracker[clientWrapper.id] += end_time - start_time

        # As this is a simulation we don't save the corrected data to csv, instead
        # we save it as a variable to the clientwrapper
        clientWrapper.data_corrected = client.data_corrected
        # client.data_corrected.to_csv(os.path.join(os.getcwd(), "mnt", "output", "only_batch_corrected_data.csv"),
        #                                 sep=self.load("separator"))
        # client.data_corrected_and_raw.to_csv(os.path.join(os.getcwd(), "mnt", "output", "all_data.csv"),
        #                              sep=self.load("separator"))
        # with open(os.path.join(os.getcwd(), "mnt", "output", "report.txt"), "w") as f:
        #     f.write(client.report)

    # print the time tracker for the coordinator
    print(f"Time tracker for coordinator, ms: {round(time_tracker['Coordinator']*1000, 2)}")

    # print the time tracker for the clients
    for clientWrapper in clientWrappers:
        print(f"Time tracker for {clientWrapper.id}, ms: {round(time_tracker[clientWrapper.id]*1000, 2)}")

    ###                                  INFO                                  ###
    ###                            SIMULATION IS DONE                          ###
    ### The simulation is done. The corrected data is saved in the             ###
    ### clientWrapper instances. Now we analyse the data by comparing to the   ###
    ### calculated centralized corrected data.                                 ###

    federated_df, intersect_features = _concat_federated_results(clientWrappers, samples_in_columns=True)

    # Simulation data
    federated_df.to_csv(os.path.join("..", "..", "evaluation_data", "simulated", mode, "after", "FedSim_corrected_data_v2.tsv"), sep="\t")
    print(f"DEBUG: Shape of federated data: {federated_df.shape}")
    print(f"Results saved to: {os.path.join('..', '..', 'evaluation_data', 'simulated', mode, 'after', 'FedSim_corrected_data_v2.tsv')}")


if __name__ == "__main__":
    print("Starting simulation")
    main()
