import os
import bios

import numpy as np

from scipy import linalg
from FeatureCloud.app.engine.app import AppState, app_state, Role, LogLevel

from classes.client import Client

# FeatureCloud requires that apps define the at least the 'initial' state.
# This state is executed after the app instance is started.
INPUT_FOLDER = os.path.join(os.getcwd(), "mnt", "input")
@app_state('initial')
class InitialState(AppState):

    def register(self):
        self.register_transition('common_features', Role.COORDINATOR)
        self.register_transition('validate', Role.PARTICIPANT)

    def run(self):
        # read in the config
        try:
            config = bios.read(os.path.join(os.getcwd(), "mnt", "input", "config.yml"))
        except Exception as e1:
            try:
                config = bios.read(os.path.join(os.getcwd(), "mnt", "input", "config.yaml"))
            except Exception as e2:
                self.log(f"Could not read the config file, tried config.yml: {e1} and config.yaml: {e2}", LogLevel.FATAL)
        self.log(f"Got the following config:\n{config}")
        if "flimmaBatchCorrection" not in config:
            self.log("Incorrect format of your config file, the key flimmaBatchCorrection must be in your config file", LogLevel.FATAL)
        config = config["flimmaBatchCorrection"]
        # read min_samples
        if "min_samples" not in config:
            minSamples = 0
            self.log("min_samples was not given so it was set to 0", LogLevel.DEBUG) # debug is used like info
        else:
            minSamples = config["min_samples"]
        # read covariates
        if "covariates" not in config:
            covariates = None
        else:
            covariates = config["covariates"]
        # read smpc
        smpc = True
        if "smpc" in config:
            smpc = config["smpc"]
        # read design file and design_seperator
        design_file_path = None
        if "design_filename" in config:
            design_file_path = os.path.join(INPUT_FOLDER, config["design_filename"])
        if "design_separator" not in config:
            design_separator = "\t"
        else:
            design_separator = config["design_separator"]
        # read expression file or data file
        if "data_filename" not in config:
            self.log("No data_filename was given in the config, cannot continue", LogLevel.FATAL)
        datafile_path = os.path.join(INPUT_FOLDER, config["data_filename"])
        # read the seperator
        if "separator" not in config:
            self.log("No separator was given in the config, cannot continue", LogLevel.FATAL)
        # read the normalization method
        if "normalizationMethod" not in config:
            normalizationMethod = None
        else:
            normalizationMethod = config["normalizationMethod"]
        # read whether given file is an expression file
        if "expression_file_flag" not in config:
            expr_file_flag = False
        else:
            expr_file_flag = config["expression_file_flag"]
        # checking for the index_col
        if "index_col" not in config:
            index_col = None
        else:
            index_col = config["index_col"]
        if not index_col:
            self.log("No index_col was given in the config, automatically creating an index.")

        # defining the client
        cohort_name = self.id
        client = Client(
            cohort_name=cohort_name,
            datafile_path=datafile_path,
            expr_file_flag=expr_file_flag,
            design_file_path=design_file_path,
            separator=config["separator"],
            design_separator=design_separator,
            index_col=index_col,
            covariates=covariates,
            normalizationMethod=normalizationMethod,
        ) # initializing the client includes loading of the data

        self.store(key="smpc", value=smpc)
        self.store(key='client', value=client)
        self.store(key='minSamples', value=minSamples)
        self.store(key='covariates', value=covariates)
        self.store(key="separator", value=config["separator"])
        self.configure_smpc()
        # send list of protein names (genes) to coordinator
        # we use the hashed values of the feature names and variables
        self.send_data_to_coordinator((list(client.hash2feature.keys()), list(client.hash2variable.keys())),
                                    send_to_self=True,
                                    use_smpc=False)
        if self.is_coordinator:
            return 'common_features'
        return 'validate'


@app_state('common_features')
class CommonGenesState(AppState):

    def register(self):
        self.register_transition('validate', Role.COORDINATOR)

    def run(self):
        # wait for each client to send the list of genes they have
        self.log("[common_features] Gathering features from all clients")
        lists_of_features_and_variables = self.gather_data(is_json=False)
        self.log("[common_features] Gathered data from all clients")
        # generate a sorted list of the genes that are available on each client
        global_feature_names = set()
        global_variables = set()
        for tup in lists_of_features_and_variables:
            local_feature_name = tup[0]
            local_variable_list = tup[1]
            if len(global_feature_names) == 0:
                global_feature_names = set(local_feature_name)
            else:
                global_feature_names = global_feature_names.intersection(set(local_feature_name))
            if local_variable_list:
                if len(global_variables) == 0:
                    global_variables = set(local_variable_list)
                else:
                    global_variables.intersection(set( local_variable_list))
        global_feature_names = sorted(list(global_feature_names))
        self.log("[common_features] common_features were found")

        # Send feature_names and variables to all
        if not global_variables:
            global_variables = None
        self.broadcast_data((global_feature_names, global_variables),
                            send_to_self=True, memo="commonGenes")
        self.log("[common_features] Data was set to be broadcasted:")
        self.log("[common_features] transitioning to validate")
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

        client.validate_inputs(global_feauture_names_hashed, global_variables_hashed)
        self.log("[validate] Inputs have been validated")
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
        XtX, XtY, err = client.compute_XtX_XtY(self.load("minSamples"))
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
        k = client.design.shape[1]
        n = len(client.feature_names)
        XtX_glob = np.zeros((n, k, k))
        XtY_glob = np.zeros((n, k))
        stdev_unscaled = np.zeros((n, k))
        if not self.load("smpc"):
            XtX_list = list()
            XtY_list = list()
            for ele in XtX_XtY_list:
                XtX_list.append(ele[0])
                XtY_list.append(ele[1])

            # set up matrices for global XtX and XtY
            for i in range(0, len(self.clients)):
                XtX_glob += XtX_list[i]
                XtY_glob += XtY_list[i]
        else:
            # smpc case, already aggregated
            XtX_XtY_list = XtX_XtY_list[0]
            XtX_glob += XtX_XtY_list[0]
            XtY_glob += XtX_XtY_list[1]

        # calcualte beta and std. dev.
        beta = np.zeros((n, k))
        for i in range(0, n):
            # if the determant is 0 the inverse cannot be formed so we need
            # to use the pseudo inverse instead
            if linalg.det(XtX_glob[i, :, :]) == 0:
                invXtX = linalg.pinv(XtX_glob[i, :, :])
            else:
                invXtX = linalg.inv(XtX_glob[i, :, :])
            beta[i, :] = invXtX @ XtY_glob[i, :]
            stdev_unscaled[i, :] = np.sqrt(np.diag(invXtX))

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
        client.data_corrected.to_csv(os.path.join(os.getcwd(), "mnt", "output", "only_batch_corrected_data.csv"),
                                     sep=self.load("separator"))
        client.data_corrected_and_raw.to_csv(os.path.join(os.getcwd(), "mnt", "output", "all_data.csv"),
                                     sep=self.load("separator"))
        with open(os.path.join(os.getcwd(), "mnt", "output", "report.txt"), "w") as f:
            f.write(client.report)
        return 'terminal'
