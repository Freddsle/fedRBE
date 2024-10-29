"""
Contains multiple functions used by the coordinator to perform different tasks
relevant for batch effect correction.
"""
import numpy as np
from numpy import linalg
from typing import List, Tuple, Union

def create_feature_presence_matrix(
        lists_of_features_and_variables: List[Tuple[str, List[str], List[str]]],
        global_feature_names: List[str]
        ) -> Tuple[np.ndarray, List[str]]:
    """
    Creates a matrix that indicates the presence of features in the different
    clients. The matrix has the shape (num_features, num_clients) and contains
    1 if the feature is present in the client and 0 otherwise. Also returns
    a dictionary indicating which clients contain each feature.

    Args:
        lists_of_features_and_variables: A list of tuples containing the cohort name, features
        and variables available on each client. Each tuple represents one client,
        the feature_names is the first element of the tuple and the variables
        is the second element of the tuple. The featurenames should be hashes,
        but this is not checked here.
        global_feature_names: A list of all the features that are available on all clients.

    Returns:
        matrix: A matrix of shape (num_features, num_clients) that indicates
            the presence of features in the different clients. The matrix contains
            1 if the feature is present in the client and 0 otherwise.
        cohorts_order: A list containing the names of the clients.
    """
    feature_index = {feature: idx for idx, feature in enumerate(global_feature_names)}
    num_features = len(global_feature_names)
    num_cohorts = len(lists_of_features_and_variables)

    # Initialize the matrix
    matrix = np.zeros((num_features, num_cohorts), dtype=int)
    cohorts_order = []

    # Populate the matrix and the dictionary
    for cohort_idx, (cohort_name, features, _) in enumerate(lists_of_features_and_variables):
        for feature in features:
            if feature in feature_index:
                matrix[feature_index[feature], cohort_idx] = 1
        cohorts_order.append(cohort_name)

    return matrix, cohorts_order

def select_common_features_variables(
    lists_of_features_and_variables: List[Tuple[str, List[str], List[str]]],
    min_clients=2) -> \
        Tuple[List[str], Union[List[str], None], np.ndarray, List[str]]:
    """
    Extracts the union of features and the intersection of variables from the
    lists of features and variables provided by the clients.
    outdated : For the union, only features that are available on at least two clients
    are selected.
    Args:
        lists_of_features_and_variables: A list of tuples containing the features
        and variables available on each client. Each tuple represents one client,
        the feature_names is the first element of the tuple and the variables
        is the second element of the tuple. The featurenames should be hashes,
        but this is not checked here.
    Returns:
        global_feature_names, global_variables
            global_feature_names: A list of all the features that are available
                on all clients. The list is sorted and contains no duplicates.
            global_variables: A list of all the variables that are available on
                all clients. The list is sorted and contains no duplicates.
                If no variables are available, this is None.
            feature_presence_matrix: A matrix of shape (num_features, num_clients) that indicates
                the presence of features in the different clients. The matrix contains
                1 if the feature is present in the client and 0 otherwise.
            cohorts_order: A list of names of the clients to which the columns of the matrix correspond.
    """
    feature_count = {}
    global_variables = None

    for _, features, variables in lists_of_features_and_variables:
        # Count features
        for feature in features:
            if feature in feature_count:
                feature_count[feature] += 1
            else:
                feature_count[feature] = 1

        # Intersect variables
        if variables:
            if global_variables is None:
                global_variables = set(variables)
            else:
                global_variables.intersection_update(variables)

    # Select features present in at least two clients
    global_feature_names = sorted([feature for feature, count in feature_count.items() if count >= min_clients])

    # Sort variables
    if global_variables:
        global_variables = sorted(global_variables)
    else:
        global_variables = None

    # Create the feature presence matrix - for gloal mask
    feature_presence_matrix, cohorts_order = create_feature_presence_matrix(lists_of_features_and_variables, global_feature_names)

    return global_feature_names, global_variables, feature_presence_matrix, cohorts_order

def reorder_matrix(feature_matrix: np.ndarray,
                   all_client_names: List[str],
                   cohorts_order: List[str]) -> np.ndarray:
    """
    Reorders the columns of a feature matrix according to a specified order of
    clients.
    Args:
        feature_matrix: A matrix of shape (num_features, num_clients) containing
            the features of the clients.
        all_client_names: A list containing the names of the clients in
            the desired order (that will be used for the design matrix).
        cohorts_order: A list of all the client names.
    Returns:
        reordered_matrix: The feature matrix with the columns reordered according
            to the specified order of clients.
    """
    # Create a mapping of column indices
    index_mapping = {name: idx for idx, name in enumerate(cohorts_order)}
    ordered_indices = [index_mapping[cohort] for cohort in all_client_names]

    # Reorder the columns
    reordered_matrix = feature_matrix[:, ordered_indices]

    return reordered_matrix

def create_beta_mask(feature_presence_matrix: np.ndarray, n: int, k: int) -> np.ndarray:
    """
    Creates a mask that indicates which features are present in the clients
    and which are not. The mask has the shape (num_features, num_clients) and
    contains 1 if the feature is absent in the client and 1 otherwise.
    Args:
        feature_presence_matrix: A matrix of shape (num_features, num_clients) that indicates
            the presence of features in the different clients. The matrix contains
            1 if the feature is present in the client and 0 otherwise.
        n: The number of features.
        k: The number of clients.
    Returns:
        global_mask: A matrix of shape (num_features, num_clients) that indicates
            which features are present in the clients. The matrix contains 1
            if the feature is absent in the client and 1 otherwise.
    """

    # Initialize the mask with zeros
    global_mask = np.zeros((n, k))
    # Get the number of columns in the feature_presence_matrix
    num_cols = feature_presence_matrix.shape[1]

    # Iterate over each row in the feature_presence_matrix
    for i in range(n):
        row = feature_presence_matrix[i]

        # check if we need to do anything - if any batch is missing?
        zero_count = np.sum(row == 0)
        if zero_count > feature_presence_matrix.shape[1] - 1:
            raise ValueError("The number of zeros in the row is greater than the number of columns")
        if zero_count == 0:
            continue

        # presence of the first and last beach
        last_batch_present = row[-1] == 1
        # transform presence to mask, 0 means present, 1 is absent
        transformed_row = np.where(row[:-1] == 0, 1, 0)

        if not last_batch_present:
            # If the last column is 0 (the reference batch is not present),
            # process the row as described
            # move a 0 from the last present batch to the first batch
            if 0 in transformed_row:
                last_zero_index = np.where(transformed_row == 0)[0][-1]
                transformed_row[last_zero_index] = 1
            global_mask[i, -num_cols+1:] = transformed_row

        else:
            # Check if there is a last_present_index and it is not zero
            if 0 in transformed_row:
                first_absent_index = np.where(transformed_row == 1)[0][0]
                last_present_indices = np.where(transformed_row == 0)[0]
                if len(last_present_indices) > 0 and last_present_indices[-1] != 0:
                    last_present_index = last_present_indices[-1]

                    if last_present_index > first_absent_index:
                        # in transformed_row interchange values between the first absent and the last present.
                        # Swap the values
                        transformed_row[first_absent_index], transformed_row[last_present_index] = \
                            transformed_row[last_present_index], transformed_row[first_absent_index]
            global_mask[i, -num_cols+1:] = transformed_row

    # Convert to boolean mask
    global_mask = global_mask > 0
    return global_mask

def compute_beta(XtX_XtY_list: List[List[np.ndarray]],
                 n: int, k: int,
                 global_mask: np.ndarray
                 ) -> np.ndarray:
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
        global_mask: A matrix of shape (num_features, num_clients) that indicates
            which features are present in the clients. The matrix contains 1
            if the feature is absent in the client and 1 otherwise.
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
        # due to serialization, the matrices are received as lists
        XtX = np.array(XtX)
        XtY = np.array(XtY)
        if XtX.shape[0] != n or XtX.shape[1] != k or XtY.shape[0] != n or XtY.shape[1] != k:
            raise ValueError(f"Shape of received XtX or XtY does not match the expected shape: {XtX.shape} {XtY.shape}")
        XtX_glob += XtX
        XtY_glob += XtY

    inverse_count = 0 #TODO: rmv
    # calculate the betas
    # formula is beta = (XtX)^-1 * XtY
    # if XtX is singular, we need to use the pseudo inverse
    for i in range(0, n):
        # using the mask to remove the columns and rows that are not present
        mask = global_mask[i, :]
        submatrix = XtX_glob[i, :, :][np.ix_(~mask, ~mask)]

        if linalg.det(submatrix) == 0:
            inverse_count += 1 #TODO: rmv
            print(f"INFO: XtX_glob[{i}] is singular")

        invXtX = linalg.inv(submatrix)
        beta[i, ~mask] = invXtX @ XtY_glob[i, ~mask]

        # # TODO: rmv
        # # if the determant is 0 the inverse cannot be formed so we need
        # # to use the pseudo inverse instead
        # if linalg.det(XtX_glob[i, :, :]) == 0:
        #     inv_XtX = linalg.pinv(XtX_glob[i, :, :])
        #     print(f"INFO: XtX_glob[{i}] is singular") #TODO: rmv
        #     print(f"INFO: XtX_glob[{i}]:\n{XtX_glob[i, :, :]}") #TODO: rmv
        #     inverse_count += 1 #TODO: rmv
        # else:
        #     inv_XtX = linalg.inv(XtX_glob[i, :, :])
        # beta[i, :] = inv_XtX @ XtY_glob[i, :]

    print(f"INFO: Shape of beta: {beta.shape}")
    print(f"INFO: Number of pseudo inverses: {inverse_count}") #TODO: rmv
    return beta