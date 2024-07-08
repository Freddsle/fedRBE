import os

import numpy as np

from FeatureCloud.app.engine.app import AppState, app_state, Role, LogLevel

from classes.client import Client
from classes.coordinator_utils import select_common_features_variables, \
    compute_beta

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
        self.configure_smpc() # set the default values
        # send list of protein names (genes) to coordinator
        # we use the hashed values of the feature names and variables
        self.send_data_to_coordinator((list(client.hash2feature.keys()), list(client.hash2variable.keys())),
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
        lists_of_features_and_variables = self.gather_data(is_json=False)
        self.log("[global_feature_selection] Gathered data from all clients")
        global_feature_names, global_variables = select_common_features_variables(lists_of_features_and_variables)
        self.broadcast_data((global_feature_names, global_variables),
                            send_to_self=True, memo="commonGenes")
        self.log("[global_feature_selection] Data was set to be broadcasted:")
        self.log("[global_feature_selection] transitioning to validate")
        return 'validate'

@app_state('validate')
class ValidationState(AppState):

    def register(self):
        self.register_transition('compute_XtX_XtY', Role.BOTH)

    def run(self):
        # obtain and safe common genes and indices of design matrix
        self.log("[validate] waiting for common features and covariates")
        global_feauture_names_hashed, global_variables_hashed = self.await_data(n=1, is_json=False, memo="commonGenes")
        client = self.load('client')

        client.validate_inputs(global_variables_hashed)
        self.log("[validate] Inputs have been validated")
        client.set_data(global_feauture_names_hashed)
        self.log("[validate] Data has been set to contain all global features")
        # get all client names to generate design matrix
        all_client_names = self.clients
        err = client.create_design(all_client_names[:-1])
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
        if not np.array_equal(sorted(client.sample_names), sorted(client.data.columns.values)):
            self.log("The sample names in the design file and the data file do not match")
            des_idx = set(client.sample_names)
            data_idx = set(client.data.index.values)
            union_indexes = des_idx.union(data_idx)
            intercept_indexes = des_idx.intersection(data_idx)
            self.log(f"The following indexes are in the union of both files (union): {union_indexes}")
            self.log(f"The following indexes are in both files (intercept): {intercept_indexes}")
            self.log(f"The following indexes are only in one of the files (union-intercept): {union_indexes.difference(intercept_indexes)}")
            self.log("aborting...", LogLevel.FATAL)

        # sort data by sample names and proteins
        client.data = client.data.loc[client.feature_names, client.sample_names]
        client.n_samples = len(client.sample_names)

        # compute XtX and XtY
        XtX, XtY, err = client.compute_XtX_XtY()
        if err != None:
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
        # wait for each client to compute XtX and XtY and collect data
        self.log("[compute_beta] gathering data")
        XtX_XtY_list = self.gather_data(use_smpc=self.load("smpc"))
        self.log("[compute_beta] Got XtX_XtY_list from gather_data")
        client = self.load('client')
        beta = compute_beta(XtX_XtY_list, n=len(client.feature_names), k=client.design.shape[1])

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
