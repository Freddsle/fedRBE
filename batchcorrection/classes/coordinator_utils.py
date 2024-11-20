"""
Contains multiple functions used by the coordinator to perform different tasks
relevant for batch effect correction.
"""
import numpy as np
from numpy import linalg
from typing import List, Tuple, Union, Dict

def create_feature_presence_matrix(
        feature_batch_info: List[Tuple[str, Union[int, None], Dict[str, List[str]]]],
        global_feature_names: List[str],
        default_order: List[str]
        ) -> Tuple[np.ndarray, List[str]]:
    """
    Creates a matrix that indicates the presence of features in the different
    clients. The matrix has the shape (num_features, num_batches) and contains
    1 if the feature is present in the client and 0 otherwise. Also returns
    a dictionary indicating which clients contain each feature.
    Finally, fixes the order of the clients.

    Args:
        feature_variable_batch_info: A list of tuples containing four items:
            the clientname, the list of variables (hashed),
            the position of the client relevant for ordering,
            and a dictionary containing as keys the batch names
            (format is "<client_name>|<batch_name>") and as values the list of
            features(hashed) available in that batch.
        global_feature_names: A list of all the features that are available on all clients.
        default_order: A list of names of the clients in which to order the clients

    Returns:
        matrix: A matrix of shape (num_features, num_batches) that indicates
            the presence of features in the different clients. The matrix contains
            1 if the feature is present in the client and 0 otherwise.
        cohorts_order: A list containing the names of the clients.
    """
    feature2index = {feature: idx for idx, feature in enumerate(global_feature_names)}
    num_features = len(global_feature_names)
    all_cohorts: List[List[str]] = list()
        # each element is a list of strings representing the batches of one client
    for _, _, batch_feature_presence_info in feature_batch_info:
        # The keys of the dictionary are the batch names of the whole client
        all_cohorts.append(list(batch_feature_presence_info.keys()))
    # we need to set the order of the clients either as sent by the clients
    # or as found in the given default_order
    if any([not isinstance(position, int) for _, position, _ in feature_batch_info]):
        # if any position is None, we use the default order
        print(f"INFO: Using the default client order: {default_order}")
        client_order = default_order
    else:
        # if all positions are integers, we use the order of the positions
        client_order = [cohort_name for cohort_name, _, _ in sorted(feature_batch_info, key=lambda x: x[1])] # type: ignore
            # pylance complains as it doesn't understand that we ensured we only have ints for the position by the if clause
        print(f"INFO: Using given specific client order: {client_order}")
    # we need to sort the cohorts in the order of the client_order
    # we do this in an inefficient O(n^2) way, but the number of batches
    # is so small it doesn't matter and this is way more readable than any fancier
    # sorting algorithm
    cohorts_order: List[str] = []
    for client_name in client_order:
        for cohort_list in all_cohorts:
            # all cohorts of one client, extract which client
            referencing_client_name = cohort_list[0].split("|")[0]
            if client_name == referencing_client_name:
                # in this case the cohort_list are the batches of the client
                # we are considering right now from the client_order
                cohorts_order.extend(sorted(cohort_list))
                break

    print(f"INFO: Cohorts order: {cohorts_order}")

    # Initialize the matrix
    num_cohorts = len(cohorts_order)
    matrix = np.zeros((num_features, num_cohorts), dtype=int)

    # we populate the matrix and the dictionary
    for cohort_idx, batch_name in enumerate(cohorts_order):
        # batch_name is a string like "<client_label>|<batch_label>"
        if len(batch_name.split()) > 2:
            # sanity check of the input
            raise ValueError(f"Batch name incorrectly formatted: {batch_name}")

        # get the corresponding features, we just iterate over all the
        # batch information objects and check if the batch_name is in the
        # batch_feature_presence_info dictionary
        batch_features = []
        for _, _, batch_feature_presence_info in feature_batch_info:
            if batch_name in batch_feature_presence_info:
                batch_features = batch_feature_presence_info[batch_name]
                break
        if len(batch_features) == 0:
            raise ValueError(f"No features found for batch {batch_name}")
        for feature in batch_features:
            if feature in feature2index:
                # cohort_idx has this feature
                matrix[feature2index[feature], cohort_idx] = 1
            # the 0 in the else case is taken care of as the matrix was
            # initialized with np.zeros

    return matrix, cohorts_order

def select_common_features_variables(
    feature_batch_info: List[Tuple[str, Union[int, None], Dict[str, List[str]]]],
    default_order: List[str],
    min_clients=3) -> \
        Tuple[List[str], np.ndarray, List[str]]:
    """
    Extracts the union of features and the intersection of variables from the
    lists of features and variables provided by the clients.
    outdated : For the union, only features that are available on at least two clients
    are selected.
    Args:
        feature_batch_info: A list of tuples containing items:
            the clientname,
            the position of the client relevant for ordering,
            and a dictionary containing as keys the batch names
            (format is "<client_name>|<batch_name>") and as values the list of
            features(hashed) available in that batch.
        default_order: A list of names of the clients in which to order the clients
        min_clients: The minimum number of clients that should have a feature for the
            feature to be included in the global feature list.
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
            cohorts_order: A list of names of the clients in which to order the clients
    """
    feature_count = {}


    for _,  _, batch_feature_presence_info in feature_batch_info:
        # Count features
        for features in batch_feature_presence_info.values():
            for feature in features:
                if feature in feature_count:
                    feature_count[feature] += 1
                else:
                    feature_count[feature] = 1

    # Select features present in at least min_clients clients, also sorting them
    global_feature_names = sorted([feature for feature, count in feature_count.items() if count >= min_clients])

    # Create the feature presence matrix - for global mask
    feature_presence_matrix, cohorts_order = create_feature_presence_matrix(feature_batch_info, global_feature_names, default_order) # type: ignore

    return global_feature_names, feature_presence_matrix, cohorts_order

def create_beta_mask(feature_presence_matrix: np.ndarray, n: int, k: int) -> np.ndarray:
    """
    Creates a mask that indicates which features are present in the clients
    and which are not. The mask has the shape (num_features, num_batches).
    Args:
        feature_presence_matrix: A matrix of shape (num_features, num_batches) that indicates
            the presence of features in the different clients. The matrix contains
            1 if the feature is present in the client and 0 otherwise.
        n: The number of features.
        k: The number of columns in self.design.
    Returns:
        global_mask: A matrix of shape (num_features, num_batches) that indicates
            which features should be ignored in the linear model.
            The inverse of this mask should be used on XtX and XtY.
    """

    # Initialize the mask with zeros
    global_mask = np.zeros((n, k))
    # Get the number of columns in the feature_presence_matrix
    num_batch_cols = feature_presence_matrix.shape[1]

    # Iterate over each row in the feature_presence_matrix
    for feature_idx in range(n):
        row = feature_presence_matrix[feature_idx]
            # row is a list of 0s and 1s, 0 means the feature is present in the batch
            # and 1 means the feature is absent in the batch

        # check if we need to do anything - if any batch is missing?
        zero_count = np.sum(row == 0)
        if zero_count > feature_presence_matrix.shape[1] - 1:
            raise ValueError("The number of zeros in the row is greater than the number of columns")
        if zero_count == 0:
            # each batch has data, nothing to mask
            continue

        # at least one batch is missing
        # presence of the first and last beach
        last_batch_present = row[-1] == 1
        # transform presence to mask, 0 means present, 1 is absent
        # basically flip fromn 0 to 1 and vice versa
        # furthermore, we eliminated the last batch from the mask as
        # it is also not in design, we just need the info if it is present
        transformed_row = np.where(row[:-1] == 0, 1, 0)

        if not last_batch_present:
            # If the last batch is not present, this means for this feature
            # the reference batch does not have any data.
            # In this case we set the global mask to 1 for the last batch
            # that is still present, effectively using it
            # as the reference batch as with the mask we remove it from the
            # regression model training
            if 0 in transformed_row:
                last_present_index = np.where(transformed_row == 0)[0][-1]
                transformed_row[last_present_index] = 1
            global_mask[feature_idx, -num_batch_cols+1:] = transformed_row

        else:
            # Check if the feature exists at least in two batches
            if 0 in transformed_row:
                first_absent_index = np.where(transformed_row == 1)[0][0]
                present_indices = np.where(transformed_row == 0)[0]
                if len(present_indices) > 0 and present_indices[-1] != 0:
                    last_present_index = present_indices[-1]
                    if last_present_index > first_absent_index:
                        # in transformed_row interchange values between the first absent and the last present.
                        # Swap the values
                        transformed_row[first_absent_index], transformed_row[last_present_index] = \
                            transformed_row[last_present_index], transformed_row[first_absent_index]
            global_mask[feature_idx, -num_batch_cols+1:] = transformed_row

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

    # calculate the betas
    # formula is beta = (XtX)^-1 * XtY
    # if XtX is singular, we need to use the pseudo inverse
    for i in range(0, n):
        # using the mask to remove the columns and rows that are not present
        mask = global_mask[i, :]
        submatrix = XtX_glob[i, :, :][np.ix_(~mask, ~mask)]
            # submatrix is of dimension (k, K)
            # we only take the rows and columns where the mask is set to
            # False (~ -> not mask)

        if linalg.det(submatrix) == 0:
            raise ValueError(
                "ERROR: Cannot calculate the linear models as the design matrix containing batch information " +\
                "and covariates is singular, please check for colinearity between covariates/covariates and batches")

        invXtX = linalg.inv(submatrix)
        beta[i, ~mask] = invXtX @ XtY_glob[i, ~mask]

    return beta