"""
The whole federated batch effect correction protocol as one function.
run_fedrbe only needs an object fulfilling the ProtocolFedLearning protocol
(is_coordinator, id, clients, send_data_to_coordinator, gather_data,
broadcast_data, await_data). FeatureCloud's AppState fulfills it, so states.py
simply calls run_fedrbe. A local simulation (run_simulation_native of
evaluation_classification_after_correction/fc_fed_forest_simple_app/
fc_fedlearnsim_helper) can run the same function without docker, see
evaluation_utils/fedRBE_simulation_scrip_simdata.py.
"""
import os

import numpy as np

try:
    # inside the app container, the app folder is the working directory
    from classes.client import Client
    from classes.coordinator_utils import select_common_features_variables, \
        compute_beta, create_beta_mask, batch_coefficients
    from perf import NullRecorder
except ImportError:
    # imported from the repository, e.g. by tests or local simulations
    from batchcorrection.classes.client import Client
    from batchcorrection.classes.coordinator_utils import select_common_features_variables, \
        compute_beta, create_beta_mask, batch_coefficients
    from batchcorrection.perf import NullRecorder

# Exponent used for the fixed point representation in SMPC, we use a bigger
# exponent than FeatureCloud's default
SMPC_EXPONENT = 12


def smpc_error_bound(smpc: bool, num_clients: int) -> float:
    """
    Maximum absolute error per aggregated XtX entry caused by the fixed point
    representation of SMPC, zero without SMPC.
    """
    if not smpc:
        return 0.0
    return num_clients * 10.0 ** -SMPC_EXPONENT


def run_fedrbe(protocol, inputfolder: str, outputfolder: str, perf=None) -> None:
    """
    Runs fedRBE for one client. The coordinator is also a client.
    Args:
        protocol: Used for sending and receiving data, see the module docstring
        inputfolder: Folder containing the config file and the data of this client
        outputfolder: Folder the corrected data and the report are written to
        perf: Optional perf.PerfRecorder, marks the spans of work in the
            client and the aggregator role. The coordinator's client_await
            spans contain its aggregator spans, as it awaits the aggregated
            result while it aggregates
    Raises:
        RuntimeError: if the clients' inputs are inconsistent
    """
    perf = perf or NullRecorder()
    ### initial
    perf.start("whole_protocol", "whole_protocol")
    perf.start("client_compute", "commonVariables")
    # defining the client
    cohort_name = protocol.id
    client = Client()
    client.config_based_init(clientname = cohort_name, input_folder = inputfolder)
    smpc = client.smpc
    perf.info["smpc"] = smpc
    # size of the client's own data, used to relate the memory usage to it
    perf.info["n_samples"] = client.n_samples
    perf.info["n_features"] = len(client.feature_names)
    # send list of protein names (genes) to coordinator
    # we use the hashed values of the feature names and variables
    assert isinstance(client.hash2feature, dict)
    assert isinstance(client.hash2variable, dict)

    # exchange the batch_labels and covariates
    perf.end("client_compute", "commonVariables")
    perf.start("client_send", "commonVariables")
    protocol.send_data_to_coordinator((client.batch_labels,
                                       list(client.hash2variable.keys()), client.smpc,
                                       client.feature_block_size),
                                      send_to_self=True,
                                      use_smpc=False)
    perf.end("client_send", "commonVariables")
    perf.start("client_await", "commonVariables")
    if protocol.is_coordinator:
        # union the batch_labels and intersect the covariates
        perf.start("agg_gather", "commonVariables")
        list_labels_variables = protocol.gather_data(is_json=False)
        perf.end("agg_gather", "commonVariables")
        perf.start("agg_compute", "commonVariables")
        global_variables_hashed = None  # will become a list, not a set
        global_batch_labels = list()
        for labels, variables, client_smpc, feature_block_size in list_labels_variables:
            if client_smpc != client.smpc:
                raise RuntimeError("All clients must use the same boolean 'smpc' setting")
            if feature_block_size != client.feature_block_size:
                raise RuntimeError("All clients must use the same 'feature_block_size' setting")
            # intersect the variables while preserving deterministic order
            # (the order of the first client's list, which comes from the
            # config file). Using a set here would cause non-deterministic
            # iteration order across Docker containers with different
            # PYTHONHASHSEED values, leading to misaligned design matrix
            # columns and wrong batch correction for features with NAs.
            if global_variables_hashed is None:
                global_variables_hashed = list(variables)
            else:
                var_set = set(variables)
                global_variables_hashed = [v for v in global_variables_hashed
                                           if v in var_set]
            # extend the batch_labels
            global_batch_labels.extend(labels)
        if global_variables_hashed is None:
            global_variables_hashed = []
        # ensure the batch_labels are unique
        if len(global_batch_labels) != len(set(global_batch_labels)):
            raise RuntimeError("Batch labels are not unique")
        # send around the number of batches and the variables
        perf.end("agg_compute", "commonVariables")
        perf.start("agg_broadcast", "commonVariables")
        protocol.broadcast_data((global_variables_hashed, len(global_batch_labels)),
                                send_to_self=True,
                                memo="commonVariables")
        perf.end("agg_broadcast", "commonVariables")
    # we receive the common variables and the number of batches
    global_variables_hashed, num_batches = protocol.await_data(n=1, is_json=False, memo="commonVariables")
    perf.end("client_await", "commonVariables")
    perf.start("client_compute", "commonGenes")
    # now we can calculate the min_samples_per_feature
    min_samples = max(num_batches+len(global_variables_hashed)+1, client.min_samples)
    # leave values out of the fit whose leverage within this client is high,
    # before anything about the features is sent
    client.exclude_high_leverage([client.hash2variable[hashed] for hashed in global_variables_hashed],
                                 min_samples)

    batch_feature_presence = client.get_batch_feature_presence_info(min_samples=min_samples)
    empty_batches = [batch for batch, features in batch_feature_presence.items() if not features]
    if empty_batches:
        raise RuntimeError(
            f"No feature has at least {min_samples} finite values in batches {empty_batches}. "
            f"For privacy, each batch needs max(min_samples={client.min_samples}, "
            f"{num_batches} batches + {len(global_variables_hashed)} covariates + 1) finite "
            "values per feature. Add samples to these batches, or merge or remove them."
        )

    perf.end("client_compute", "commonGenes")
    perf.start("client_send", "commonGenes")
    protocol.send_data_to_coordinator((cohort_name,
                                       client.position,
                                       client.reference_batch,
                                       batch_feature_presence),
                                      send_to_self=True,
                                      use_smpc=False)
    perf.end("client_send", "commonGenes")
    perf.start("client_await", "commonGenes")

    ### global_feature_selection
    feature_presence_matrix = None
    if protocol.is_coordinator:
        # wait for each client to send the list of genes they have
        print("[global_feature_selection] Gathering features from all clients")
        perf.start("agg_gather", "commonGenes")
        feature_information = protocol.gather_data(is_json=False)
        perf.end("agg_gather", "commonGenes")
        perf.start("agg_compute", "commonGenes")
        # synchronize the covariate and batch information of all clients
        # so that each client can decide which features it can use
        # the clients need the info about covariates and batches to know
        # the shape of the design matrix
        # and therefore to know how many samples each feature should have out of privacy reasons
        print("[global_feature_selection] Gathered data from all clients")
        global_feature_names, feature_presence_matrix, cohorts_order = \
              select_common_features_variables(feature_information,
                                               default_order=list(protocol.clients))
        perf.end("agg_compute", "commonGenes")
        perf.start("agg_broadcast", "commonGenes")
        protocol.broadcast_data((global_feature_names, cohorts_order),
                                send_to_self=True, memo="commonGenes")
        perf.end("agg_broadcast", "commonGenes")

    ### validate
    # obtain and safe common genes and indices of design matrix
    print("[validate] waiting for common features and covariates")
    global_feauture_names_hashed, cohorts_order = protocol.await_data(n=1, is_json=False, memo="commonGenes")
    perf.end("client_await", "commonGenes")
    perf.start("client_compute", "XtX_block")

    client.validate_inputs(global_variables_hashed)
    print("[validate] Inputs have been validated")
    client.set_data(global_feauture_names_hashed)
    perf.info["n_features_used"] = len(client.feature_names)
    print("[validate] Data has been set to contain all global features")
    # get all client names to generate design matrix
    err = client.create_design(cohorts_order)
    if err:
        raise RuntimeError(err)
    print("[validate] design has been created")

    ### compute_XtX_XtY
    client.sample_names = client.design.index.values

    # Error check if the design index and the data index are the same
    # we check by comparing the sorted indexes
    client._check_consistency_designfile()

    # Extract only relevant (the global) features and samples
    client.data = client.data.loc[client.feature_names, client.sample_names]
    client.n_samples = len(client.sample_names)

    # compute XtX and XtY
    XtX, XtY, err = client.compute_XtX_XtY(smpc_error_bound(smpc, len(protocol.clients)))
    if err:
        raise RuntimeError(f"Error in compute_XtX_XtY: {err}")

    if smpc:
        # The controller sums int64 fixed-point values; keep the clients' sum representable.
        limit = np.iinfo(np.int64).max / 10 ** SMPC_EXPONENT / len(protocol.clients)
        largest = max(np.abs(XtX).max(), np.abs(XtY).max())
        if largest >= limit:
            raise RuntimeError(
                f"XtX/XtY values reach {largest:.3g}, above the SMPC limit of {limit:.3g} for "
                f"{len(protocol.clients)} clients. Use log-scale data "
                '(normalizationMethod: "log2(x+1)"), rescale large covariates or set smpc: false.'
            )

    # send XtX and XtY
    print("[compute_XtX_XtY] Computation done, sending data to coordinator")
    print(f"[compute_XtX_XtY] XtX of shape {XtX.shape}, X of shape {client.design.shape}, XtY of shape {XtY.shape}")
    ### compute_beta
    n = len(client.feature_names)
    k = client.design.shape[1]
    n_batch_columns = num_batches - 1
    perf.end("client_compute", "XtX_block")
    if protocol.is_coordinator:
        # calculate the global mask used to eliminate linearly dependant features
        perf.start("agg_compute", "XtX_block")
        global_mask = create_beta_mask(feature_presence_matrix, n, k)
        beta = np.empty((n, n_batch_columns))
        perf.end("agg_compute", "XtX_block")

    block_size = client.feature_block_size or n
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        statistics_memo = f"XtX_XtY_{start}"
        done_memo = f"block_done_{start}"
        block = start // block_size
        print(f"[compute_beta] features {start}:{stop}")
        perf.start("client_send", "XtX_block", block)
        protocol.send_data_to_coordinator([XtX[start:stop], XtY[start:stop]],
                                         send_to_self=True, use_smpc=smpc,
                                         memo=statistics_memo)
        perf.end("client_send", "XtX_block", block)
        perf.start("client_await", "XtX_block", block)
        if protocol.is_coordinator:
            perf.start("agg_gather", "XtX_block", block)
            XtX_XtY_list = protocol.gather_data(use_smpc=smpc, memo=statistics_memo)
            perf.end("agg_gather", "XtX_block", block)
            perf.start("agg_compute", "XtX_block", block)
            block_beta = compute_beta(
                XtX_XtY_list, n=stop - start, k=k,
                global_mask=global_mask[start:stop],
                xtx_error=smpc_error_bound(smpc, len(protocol.clients)),
                n_batch_columns=n_batch_columns)
            del XtX_XtY_list
            beta[start:stop] = batch_coefficients(block_beta, n_batch_columns)
            perf.end("agg_compute", "XtX_block", block)
            perf.start("agg_broadcast", "XtX_block", block)
            protocol.broadcast_data(None, send_to_self=True, memo=done_memo)
            perf.end("agg_broadcast", "XtX_block", block)

        # Wait before queuing another SMPC block.
        protocol.await_data(n=1, is_json=False, memo=done_memo)
        perf.end("client_await", "XtX_block", block)

    perf.start("client_await", "beta")
    if protocol.is_coordinator:
        # Share only batch coefficients, after all blocks succeed.
        perf.start("agg_broadcast", "beta")
        protocol.broadcast_data(beta, send_to_self=True, memo="beta")
        perf.end("agg_broadcast", "beta")

    ### include_correction
    beta = protocol.await_data(n=1, is_json=False, memo="beta")
    perf.end("client_await", "beta")
    perf.start("client_compute", "final")
    # remove the batch effects in own data and safe the results
    client.remove_batch_effects(beta)
    print(f"DEBUG: Shape of corrected data: {client.data_corrected.shape}")
    client.data_corrected.to_csv(os.path.join(outputfolder, "only_batch_corrected_data.csv"),
                                 sep=client.separator)
    client.full_corrected_data.to_csv(os.path.join(outputfolder, "full_corrected_data.csv"),
                                      sep=client.separator)
    with open(os.path.join(outputfolder, "report.txt"), "w") as f:
        f.write(client.report)
    perf.end("client_compute", "final")
    perf.end("whole_protocol", "whole_protocol")
