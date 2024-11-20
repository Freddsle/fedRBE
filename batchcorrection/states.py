import os

import numpy as np

from typing import List, Dict
from FeatureCloud.app.engine.app import AppState, app_state, Role, LogLevel

from classes.client import Client
from classes.coordinator_utils import select_common_features_variables, \
    compute_beta, create_beta_mask

# FeatureCloud requires that apps define the at least the 'initial' state.
# This state is executed after the app instance is started.
INPUT_FOLDER = os.path.join(os.getcwd(), "mnt", "input")
@app_state('initial')
class InitialState(AppState):

    def register(self):
        self.register_transition('global_feature_selection', Role.COORDINATOR)
        self.register_transition('validate', Role.PARTICIPANT)

    def run(self):
        # defining the client
        cohort_name = self.id
        client = Client()
        client.config_based_init(clientname = cohort_name, input_folder = INPUT_FOLDER)

        self.store(key="smpc", value=client.smpc)
        self.store(key='client', value=client)
        self.store(key="separator", value=client.separator)
        self.configure_smpc(exponent=12) # set the default values, we use a bigger exponent though
        # send list of protein names (genes) to coordinator
        # we use the hashed values of the feature names and variables
        assert isinstance(client.hash2feature, dict)
        assert isinstance(client.hash2variable, dict)

        # exchange the batch_labels and covariates
        self.send_data_to_coordinator((client.batch_labels,
                                       list(client.hash2variable.keys())),
                                    send_to_self=True,
                                    use_smpc=False)
        if self.is_coordinator:
            # union the batch_labels and intersect the covariates
            list_labels_variables = self.gather_data(is_json=False)
            global_variables_hashed = set()
            global_batch_labels = list()
            for labels, variables in list_labels_variables:
                # intersect the variables
                if len(global_variables_hashed) == 0:
                    global_variables_hashed = set(variables)
                else:
                    global_variables_hashed = global_variables_hashed.intersection(set(variables))
                # extend the batch_labels
                global_batch_labels.extend(labels)
            # ensure the batch_labels are unique
            if len(global_batch_labels) != len(set(global_batch_labels)):
                self.log("Batch labels are not unique", LogLevel.FATAL)
            # send around the number of batches and the variables
            self.broadcast_data((global_variables_hashed, len(global_batch_labels)),
                                send_to_self=True,
                                memo="commonVariables")
        # we receive the common variables and the number of batches
        global_variables_hashed, num_batches = self.await_data(n=1, is_json=False, memo="commonVariables")
        self.store(key='global_variables_hashed', value=global_variables_hashed)
        # now we can calculate the min_samples_per_feature
        min_samples = max(num_batches+len(global_variables_hashed)+1, client.min_samples)

        batch_feature_presence = client.get_batch_feature_presence_info(min_samples=min_samples)

        self.send_data_to_coordinator((cohort_name,
                                        client.position,
                                        batch_feature_presence),
                                    send_to_self=True,
                                    use_smpc=False)
        if self.is_coordinator:
            return 'global_feature_selection'
        return 'validate'


@app_state('global_feature_selection')
class globalFeatureSelection(AppState):
    def register(self):
        self.register_transition('validate', Role.COORDINATOR)

    def run(self):
        # wait for each client to send the list of genes they have
        self.log("[global_feature_selection] Gathering features from all clients")
        feature_information = self.gather_data(is_json=False)
        # synchronize the covariate and batch information of all clients
        # so that each client can decide which features it can use
        # the clients need the info about covariates and batches to know
        # the shape of the design matrix
        # and therefore to know how many samples each feature should have out of privacy reasons

        self.log("[global_feature_selection] Gathered data from all clients")
        assert self._app is not None
        global_feature_names, feature_presence_matrix, cohorts_order = \
              select_common_features_variables(feature_information,
                                               default_order=self._app.clients)
        self.broadcast_data((global_feature_names, cohorts_order),
                            send_to_self=True, memo="commonGenes")
        self.store(key='feature_presence_matrix', value=feature_presence_matrix)
        self.log("[global_feature_selection] Data was set to be broadcasted:")
        self.log("[global_feature_selection] transitioning to feature_presence_matrix")
        return 'validate'

@app_state('validate')
class ValidationState(AppState):

    def register(self):
        self.register_transition('compute_XtX_XtY', Role.BOTH)

    def run(self):
        # obtain and safe common genes and indices of design matrix
        self.log("[validate] waiting for common features and covariates")
        global_feauture_names_hashed, cohorts_order = self.await_data(n=1, is_json=False, memo="commonGenes")
        global_variables_hashed = self.load("global_variables_hashed")
        client = self.load('client')

        client.validate_inputs(global_variables_hashed)
        self.log("[validate] Inputs have been validated")
        client.set_data(global_feauture_names_hashed)
        self.log("[validate] Data has been set to contain all global features")
        # get all client names to generate design matrix
        err = client.create_design(cohorts_order)
        if err:
            self.log(err, LogLevel.FATAL)
        self.log("[validate] design has been created")
        self.store(key='client', value=client)
        return 'compute_XtX_XtY'


@app_state('compute_XtX_XtY')
class ComputeState(AppState):

    def register(self):
        self.register_transition('compute_beta', Role.COORDINATOR)
        self.register_transition('include_correction', Role.PARTICIPANT)

    def run(self):
        client = self.load('client')
        client.sample_names = client.design.index.values

        # Error check if the design index and the data index are the same
        # we check by comparing the sorted indexes
        client._check_consistency_designfile()

        # Extract only relevant (the global) features and samples
        client.data = client.data.loc[client.feature_names, client.sample_names]
        client.n_samples = len(client.sample_names)

        # compute XtX and XtY
        XtX, XtY, err = client.compute_XtX_XtY()
        if err:
            print(f"Error in compute_XtX_XtY: {err}")
            self.log(err, LogLevel.FATAL)


        # send XtX and XtY
        self.log("[compute_XtX_XtY] Computation done, sending data to coordinator")
        self.log(f"[compute_XtX_XtY] XtX of shape {XtX.shape}, X of shape {client.design.shape}, XtY of shape {XtY.shape}")
        self.send_data_to_coordinator([XtX, XtY],
                                send_to_self=True,
                                use_smpc=self.load("smpc"))

        if self.is_coordinator:
            return 'compute_beta'
        return 'include_correction'


@app_state('compute_beta')
class ComputeCorrectionState(AppState):

    def register(self):
        self.register_transition('include_correction', Role.COORDINATOR)

    def run(self):
        client = self.load('client')
        feauture_presence_matrix = self.load('feature_presence_matrix')
        # calculate the global mask used to eliminate linearly dependant features
        n = len(client.feature_names)
        k = client.design.shape[1]
        global_mask = create_beta_mask(feauture_presence_matrix, n, k)

        # wait for each client to compute XtX and XtY and collect data
        self.log("[compute_beta] gathering data")
        XtX_XtY_list = self.gather_data(use_smpc=self.load("smpc"))
        self.log("[compute_beta] Got XtX_XtY_list from gather_data")
        beta = compute_beta(XtX_XtY_list,
                            n=n,
                            k=k,
                            global_mask=global_mask)

        # send beta to clients so they can correct their data
        self.log("[compute_beta] broadcasting betas")
        self.broadcast_data(beta,
                            send_to_self=True,
                            memo="beta")
        return 'include_correction'


@app_state('include_correction')
class IncludeCorrectionState(AppState):

    def register(self):
        self.register_transition('terminal')

    def run(self):
        # wait for the coordinator to calcualte beta
        beta = self.await_data(n=1, is_json=False, memo="beta")
        client = self.load('client')

        # remove the batch effects in own data and safe the results
        client.remove_batch_effects(beta)
        print(f"DEBUG: Shape of corrected data: {client.data_corrected.shape}")
        client.data_corrected.to_csv(os.path.join(os.getcwd(), "mnt", "output", "only_batch_corrected_data.csv"),
                                     sep=self.load("separator"))
        # client.data_corrected_and_raw.to_csv(os.path.join(os.getcwd(), "mnt", "output", "all_data.csv"),
        #                              sep=self.load("separator"))
        with open(os.path.join(os.getcwd(), "mnt", "output", "report.txt"), "w") as f:
            f.write(client.report)
        return 'terminal'
