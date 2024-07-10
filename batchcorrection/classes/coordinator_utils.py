"""
Contains multiple functions used by the coordinator to perform different tasks
relevant for batch effect correction.
"""
import numpy as np
from numpy import linalg
from typing import List, Tuple, Union

def select_common_features_variables(
    lists_of_features_and_variables: List[Tuple[List[str], List[str]]]) -> \
        Tuple[List[str], Union[List[str], None]]:
    """
    Extracts the union of features and the intersection of variables from the
    lists of features and variables provided by the clients.
    Args:
        lists_of_features_and_variables: A list of tuples containing the features
        and variables available on each client. Each tuple represents one client,
        the feature_names are the first element of the tuple and the variables
        are the second element of the tuple. The featurenames should be hashes,
        but this is not checked here.
    Returns:
        global_feature_names, global_variables
            global_feature_names: A list of all the features that are available
                on all clients. The list is sorted and contains no duplicates.
            global_variables: A list of all the variables that are available on
                all clients. The list is sorted and contains no duplicates.
                If no variables are available, this is None.
    """
    # generate a sorted list of the genes that are available on each client
    global_feature_names = set()
    global_variables = set()
    for tup in lists_of_features_and_variables:
        local_feature_name = tup[0]
        local_variable_list = tup[1]
        if len(global_feature_names) == 0:
            global_feature_names = set(local_feature_name)
        else:
            global_feature_names = global_feature_names.union(set(local_feature_name))
        if local_variable_list:
            if len(global_variables) == 0:
                global_variables = set(local_variable_list)
            else:
                global_variables.intersection(set(local_variable_list))
    global_feature_names = sorted(list(global_feature_names))
    print("[global_feature_selection] all_features were combined")

    # Send feature_names and variables to all
    if not global_variables:
        global_variables = None
    else:
        global_variables = sorted(list(global_variables))
    return list(global_feature_names), global_variables

def compute_beta(XtX_XtY_list: List[List[np.ndarray]], n: int, k: int) -> np.ndarray:
    """
    Gets a list of a List containing the XtX and XtY matrices from each client
    and calculates the linear model from them, returning the beta vector.
    Args:
        XtX_XtY_list: A list of tuples containing the XtX and XtY matrices from
            each client. The first element of the tuple is the XtX matrix and the
            second element is the XtY matrix.
            XtX should be of shape n x k x k and XtY should be of shape n x k.
        smpc: A boolean indicating if then computation was done using SMPC
            if yes, the data is already aggregated
        n: The expected number of features.
        k: The expected number of columns of the design matrix. Generally, is
            len(['intercept'] + [covariates] + [len(clients)-1])
            XtX should be of shape n x k x k and XtY should be of shape n x k.
    Returns:
        beta: The beta vector calculated from the XtX and XtY matrices.
            Shape of beta is n x k.
    """
    if len(XtX_XtY_list) == 0:
        raise ValueError("No data received from clients")

    XtX_glob = np.zeros((n, k, k))
    XtY_glob = np.zeros((n, k))
    beta = np.zeros((n, k))

    for XtX, XtY in XtX_XtY_list:
        if XtX.shape[0] != n or XtX.shape[1] != k or XtY.shape[0] != n or XtY.shape[1] != k:
            raise ValueError(f"Shape of received XtX or XtY does not match the expected shape: {XtX.shape} {XtY.shape}")
        XtX_glob += XtX
        XtY_glob += XtY

    inverse_count = 0 #TODO: rmv
    # calculate the betas
    # formula is beta = (XtX)^-1 * XtY
    # if XtX is singular, we need to use the pseudo inverse
    for i in range(0, n):
        # if the determant is 0 the inverse cannot be formed so we need
        # to use the pseudo inverse instead
        if linalg.det(XtX_glob[i, :, :]) == 0:
            inv_XtX = linalg.pinv(XtX_glob[i, :, :])
            print(f"INFO: XtX_glob[{i}] is singular") #TODO: rmv
            print(f"INFO: XtX_glob[{i}]:\n{XtX_glob[i, :, :]}") #TODO: rmv
            inverse_count += 1 #TODO: rmv
        else:
            inv_XtX = linalg.inv(XtX_glob[i, :, :])
        beta[i, :] = inv_XtX @ XtY_glob[i, :]

    print(f"INFO: Shape of beta: {beta.shape}")
    print(f"INFO: Number of pseudo inverses: {inverse_count}") #TODO: rmv
    return beta