from typing import List, Tuple, Dict
import pandas as pd
import numpy as np
import hashlib
import os
import bios

class Client:
    def __init__(self):
        """
        Please use config_based_init to intialize the class
        """
        self.client_name = ""
        self.smpc = False
        self.covariates = None
        self.min_samples = 0
        self.XtX = None
        self.XtY = None
        self.data = None
        self.rawdata = None
        self.design = None
        self.data_corrected = None
        self.report = None
        self.feature_names = None
        self.variables = []
        self.variables_in_data = False
        self.expr_file_flag = False
        self.num_excluded_numeric = 0
        self.num_excluded_federation = 0
        self.extra_global_features = None
        self.hash2feature = None
        self.feature2hash = None
        self.hash2variable = None
        self.position = None
        self.batch_labels = []

    def hash_names(self,
                   names: List[str]) -> Tuple[dict, dict]:
        """
        Hashes the names for privacy reasons.
        """
        hash2name = {}
        name2hash = {}
        for name in names:
            name_hash = hashlib.sha3_256(name.encode()).hexdigest()
            hash2name[name_hash] = name
            name2hash[name] = name_hash
        return name2hash, hash2name

    def hash_covariates(self,
                        covariates: List[str]) -> dict:
        """
        Hashes the covariates for privacy reasons.
        """
        hash2variable = {}
        for variable in covariates:
            variable_hash = hashlib.sha3_256(variable.encode()).hexdigest()
            hash2variable[variable_hash] = variable
        return hash2variable


    def config_based_init(self,
                          clientname: str,
                          config_filename: str="",
                          input_folder: str = os.path.join("mnt", "input"),
                          use_hashing: bool=True):
        """
        Initializes the client based on a given config file. File must be named
        config.yml or config.yaml, see the documentation of the
        batchcorrection app for more information of the config contents.
        Calls the __init__ method, which also loads the data and design matrix.
        Raises an RuntimeError if something goes wrong.
        """
        # read in the config
        if config_filename:
            config = bios.read(os.path.join(input_folder, config_filename))
        else:
            # automatic reading of the config file
            try:
                config = bios.read(os.path.join(input_folder, "config.yml"))
            except Exception as e1:
                try:
                    config = bios.read(os.path.join(input_folder, "config.yaml"))
                except Exception as e2:
                    raise RuntimeError(f"Could not read the config file, tried config.yml: {e1} and config.yaml: {e2}")
        print(f"Got the following config:\n{config}")
        if "flimmaBatchCorrection" not in config:
            raise RuntimeError("Incorrect format of your config file, the key flimmaBatchCorrection must be in your config file")

        config = config["flimmaBatchCorrection"]
        # read min_samples
        if "min_samples" not in config:
            min_samples = 0
            print("min_samples was not given so it was set to 0")
        else:
            min_samples = config["min_samples"]
            try:
                min_samples = int(min_samples)
            except:
                raise ValueError("min_samples must be an integer")
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
            design_file_path = os.path.join(input_folder, config["design_filename"])
        if "design_separator" not in config:
            design_separator = "\t"
        else:
            design_separator = config["design_separator"]
        # read expression file or data file
        if "data_filename" not in config:
            raise RuntimeError("No data_filename was given in the config, cannot continue")
        datafile_path = os.path.join(input_folder, config["data_filename"])
        # read the seperator
        if "separator" not in config:
            raise RuntimeError("No separator was given in the config, cannot continue")
        separator = config["separator"]
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

        position = None
        if "position" in config:
            position = config["position"]
            if position and not isinstance(position, int):
                raise ValueError("Position must be an integer")

        reference_batch = None
        if "reference_batch" in config:
            reference_batch = config["reference_batch"]

        self.batch_col = None
        if "batch_col" in config:
            self.batch_col = config["batch_col"]
            if self.batch_col and not isinstance(self.batch_col, str):
                raise ValueError("Batch column must be a string")
            if self.batch_col and not design_file_path:
                raise ValueError("Batch column was given but no design file was given")

        # set variables
        self.client_name = clientname
        self.smpc = smpc
        self.variables = covariates
        self.min_samples = min_samples
        self.separator = separator
        self.position = position
        self.reference_batch = reference_batch

        # load the data
        try:
            self.open_dataset(datafile_path, expr_file_flag, design_file_path, separator, design_separator, index_col)
        except:
            raise ValueError(f"Client {self.client_name}: Error loading dataset.")
        self.normalize(normalizationMethod)
        assert isinstance(self.data, pd.DataFrame)
        assert self.feature_names is not None
        assert self.variables is not None
        # Hashing logic
        if use_hashing:
            # we hash the feature names and variables to hide covariates and features
            # that this client has but other clients do not have
            if not self.feature_names:
                raise ValueError(f"Client {self.client_name}: Error loading feature names.")
            if not self.variables:
                raise ValueError(f"Client {self.client_name}: Error loading variables.")

            self.feature2hash, self.hash2feature = self.hash_names(self.feature_names)
            self.data.rename(index=self.feature2hash, inplace=True)

            self.feature_names = [self.feature2hash[feature] for feature in self.feature_names]

            if covariates:
                self.hash2variable = self.hash_covariates(covariates)
        else:
            self.hash2feature = {name: name for name in self.feature_names}
            self.feature2hash = {name: name for name in self.feature_names}
            self.hash2variable = {name: name for name in covariates} if covariates else {}


    ######### open dataset #########
    def open_dataset(self, datafile_path, expr_file_flag, design_file_path=None,
                    separator="\t", design_separator="\t", index_col=None):
        """
        Opens the dataset and loads the data and design matrix.
        The data is converted in the format features x samples.
        Furthermore, we only consider numerical values.
        All other data is removed from batch correction and added later.
        Furthermore, variables are extracted and added to the design matrix.
        A report of which features were batch corrected is given in the end.
        Args:
            datafile_path: Path to the file containing the data to be corrected
            expr_file_flag: Flag indicating whether the given file is an expression file
                An expression file has features as rows and samples as columns
                If False, the file is considered a normal csv file with samples as rows
                and features as columns
            design_file_path: Path to the design matrix file
            separator: Separator used in the data file
            design_separator: Separator used in the design file
            index_col: Name of the column that should be used as the index of the data_file
                If None, the first column is used as the index
        """
        print(f"Opening dataset {datafile_path}")
        self.expr_file_flag = expr_file_flag
        # Manage the design file
        self.batch_labels = [self.client_name]
        if design_file_path:
            relevant_cols = []
            if self.batch_col:
                relevant_cols.append(self.batch_col)
            if self.variables:
                relevant_cols.extend(self.variables)
            self.design = pd.read_csv(design_file_path, sep=design_separator, index_col=0)[relevant_cols]
            if self.batch_col:
                # extract which batches exist
                self.batch_labels = [f"{self.client_name}|{batchname}" for batchname in self.design[self.batch_col].unique()]

        # first we open the data file
        # expression file flags should have the feature names as the first
        # column per default
        if expr_file_flag:
            if index_col is None:
                # In expression files, the defualt is the first column containing the feature names
                # default pandas behaviour would however be to create an index col
                index_col = 0
            self.rawdata = pd.read_csv(datafile_path, sep=separator, index_col=index_col)
            print(f"Shape of rawdata(expr_file): {self.rawdata.shape}")
        # for normak csv, the default is not having samples so we autogenerate
        # sample integers
        else:
            self.rawdata = pd.read_csv(datafile_path, sep=separator, index_col=index_col)
            print(f"Shape of rawdata(default csv): {self.rawdata.shape}")
        self.data = self.rawdata

        self.data = self.data.dropna(axis=1, how='all')
        self.data = self.data.dropna(axis=0, how='all')

        self.variables_in_data = False
        # handling of default csv files
        if not expr_file_flag:
            # First we remove the covariates from the training data
            # if no design matrix is given
            if self.variables and not design_file_path:
                self.data = self.data.drop(columns=self.variables)
                self.variables_in_data = True

            # data cleanup
            # we ensure that we only have numerical values
            if self.data is None:
                raise ValueError(f"Client {self.client_name}: Error loading data.")
            self.data = self.data.select_dtypes(include=np.number)
            self.num_excluded_numeric = len(self.rawdata.columns) - len(self.data.columns)
            # finally we transpose as feature x sample data is expected
            self.data = self.data.T

        # handling of expression files
        else:
            # for expression files we just have to remove the rows that have non-numeric values
            # we also need to potentially extract covariates
            tmp = self.data.T
            if self.variables and not design_file_path:
                self.variables_in_data = True
                tmp = tmp.drop(columns=self.variables)
            tmp = tmp.select_dtypes(include=np.number)
            self.num_excluded_numeric = len(self.data.columns) - len(tmp.columns)
            self.data = tmp.T

        self.feature_names = list(self.data.index.values)
        self.sample_names = list(self.data.columns.values)
        # we need to ensure that the sample_names are the same as the design matrix index
        if not self.design is None:
            if not self.sample_names == list(self.design.index):
                raise ValueError(f"Client {self.client_name}: Sample names in data and design matrix do not match or are not in the same order")
        self.n_samples = len(self.sample_names)
        print(f"finished loading data, shape of data: {self.data.shape}, num_features: {len(self.feature_names)}, num_samples: {self.n_samples}")

    def normalize(self, normalizationMethod):
        assert isinstance(self.data, pd.DataFrame)
        if not normalizationMethod or normalizationMethod == "":
            # do nothing
            return
        if normalizationMethod == "log2(x+1)":
            self.data = np.log2(self.data + 1)
        else:
            print(f"Normalization method {normalizationMethod} not recognized, no normalization applied")
            return

    def get_batch_feature_presence_info(self, min_samples: int) -> Dict[str, List[str]]:
        """
        Returns the information about the batches and which features(given as
        hashed feature names) are available on this client for each batch.
        Also transforms the data, removing features that should not be
        corrected due to privacy concerns.

        Args:
            min_samples: Minimum number of samples that a feature must
                have per client to be considered for the batch effect correction

        Returns:
            batch_feature_presence_info: Dict with the batch names as keys and
                a list of hashed feature names as values
            features_ignored_privacy: List of hashed feature names that were
                ignored for privacy reasons
        """
        assert isinstance(self.data, pd.DataFrame)
        assert isinstance(self.feature_names, list)
        assert isinstance(self.hash2feature, dict)
        assert isinstance(self.design, pd.DataFrame)

        print(f"INFO: Each feature must at least have {min_samples} samples in a batch to be considered for batch effect correction. Otherwise, privacy cannot be guaranteed.")

        # initialize the dict so we can always append
        batch_feature_presence_info = {}
        for batch in self.batch_labels:
            batch_feature_presence_info[batch] = list()

        # first we need to ensure for each batch that at least 2 samples
        # are available, otherwise the specific entry of XtY from the
        # combination of the batch column and y leaks the specific non nan value
        # of the feature
        # we only need to do this if we have multiple batches, otherwise the
        # next step does a more strict check anyways
        if self.batch_col:
            for batch_label in self.batch_labels:
                pure_batch_name = batch_label.split("|")[1]
                batch_indices = self.design[self.design[self.batch_col] == pure_batch_name].index.tolist()
                batch_data = self.data.loc[:, batch_indices]
                non_nan_samples = batch_data.notnull().sum(axis=1)
                # find all features that have less than 2 samples
                ignore_index = non_nan_samples[non_nan_samples == 1].index
                if len(ignore_index) > 0:
                    print(f"INFO: Ignoring the following {len(ignore_index)} features for batch {batch_label} due to privacy reasons: {[self.hash2feature.get(hashed, hashed) for hashed in ignore_index]}")

                # set all batch_indices columns and ingore_index rows to NaN
                self.data.loc[ignore_index, batch_indices] = np.nan


        # Second we check how many samples (non nan!) are available for each feature
        # for the whole client
        non_nan_samples = self.data.notnull().sum(axis=1)
        ignore_index = non_nan_samples[non_nan_samples < min_samples].index
        if len(ignore_index) > 0:
            print(f"INFO: Ignoring the following {len(ignore_index)} features for client {self.client_name} due to privacy reasons: {[self.hash2feature.get(hashed, hashed) for hashed in ignore_index]}")
        # now we set ALL columns of the ignore_index rows to NaN
        self.data.loc[ignore_index, :] = np.nan

        # check for each batch each feature if it is available in the data
        for batch_label in self.batch_labels:
            if self.batch_col:
                pure_batch_name = batch_label.split("|")[1]
                # go from "client|batch" to "batch"
                # we have multiple batches for this client
                # we need to get the corresponding samples
                batch_indices = self.design[self.design[self.batch_col] == pure_batch_name].index.tolist()
                batch_data = self.data.loc[:, batch_indices]
                    # samples are the index in self.design but columns in self.data
            else:
                # all samples are from the same batch, just use all data
                batch_data = self.data

            for feature in self.feature_names:
                if feature in batch_data.index and not batch_data.loc[feature, :].isnull().all():
                    # feature is available in this batch and not all samples are NaN
                    batch_feature_presence_info[batch_label].append(feature)
        return batch_feature_presence_info

    def validate_inputs(self, global_variables_hashed):
        """
        Checks if protein_names match global protein group names.
        Important to ensure that client's protein names are in the global protein names. But it's not important that all global protein names are in the client's protein names.
        """
        self.validate_variables(global_variables_hashed)
        print(f"INFO: Client {self.client_name}: Inputs validated.")

    def validate_variables(self, global_variables_hashed):
        """
        Checks if the covariates in the local data differ from the global
        intersection of variables. Only uses global variables
        """
        assert isinstance(self.hash2variable, dict)
        assert isinstance(self.data, pd.DataFrame)
        # handle global/local variables being None
        if global_variables_hashed is None and self.variables is not None:
            print("WARNING: No common global covariates were selected, but local covariates were selected. The local covariates will be ignored.")
            self.variables = None
            return
        elif global_variables_hashed is not None and self.variables is None:
            raise Exception("Globally covariates were selected that cannot be represented as this client does not have these covariates")
        if global_variables_hashed is None and self.variables is None:
            return
        global_variables = [self.hash2variable.get(hashed, hashed) for hashed in global_variables_hashed]

        # handle global/local variables having different covariates
        assert self.variables is not None
        extra_global_variables = set(global_variables).difference(set(self.variables))
        extra_local_variables = set(self.variables).difference(set(global_variables))
        if len(extra_global_variables) > 0:
            raise Exception("Globally covariates were selected that cannot be represented as this client does not have these covariates")
        if len(extra_local_variables) > 0:
            print(f"WARNING: Client {self.client_name}: {len(extra_local_variables)} extra covariates that are not available on all clients will NOT be considered in the correction.")
            # we continue but ignore these features
        self.variables = global_variables
        # now we need to check if these variables are in the design matrix/in the data
        if self.design is not None:
            for var in self.variables:
                if var not in self.design.columns:
                    raise Exception(f"Variable {var} was not found in the design matrix")
        else:
            for var in self.variables:
                assert self.data is not None
                if var not in self.data.index.tolist():
                    raise Exception(f"Variable {var} was not found in the data")


    def set_data(self, global_hashed_features: List[str]):
        """
        Checks the global_hashed_features that should be used. Fills self.data
        with 0 values for features from the global_hashed_features list that this
        client does not have.
        Ensures that self.data has the features in the order of global_hashed_features.
        Args:
            global_hashed_features: List of hashed features that should be used
                for the batch effect correction. The order of the features is
                important.
        Returns:
            None, just sets the self.data matrix
        """
        # we get which features are available only locally and which only globally
        # then we fill the data matrix with 0 values for the global features
        # that are not available locally
        # then we correct the order of the features in the data matrix
        # first we ensure that the features are in the loaded data matrix
        assert isinstance(self.data, pd.DataFrame)

        # Get features only we have and features only global has
        if not self.feature_names:
            raise Exception("Feature names are not loaded yet, cannot set data")

        self.extra_local_features = set(self.feature_names).difference(set(global_hashed_features))
        print(f"Number of features available in this client: {len(self.feature_names)}")
        print(f"Number of features given globally: {len(global_hashed_features)}")
        print(f"Number of features only available on this client: {len(self.extra_local_features)}")
        self.extra_global_features = set(global_hashed_features).difference(set(self.feature_names))

        print(f"Number of features available in other clients but not this client: {len(list(self.extra_global_features))}")

        # remove the local features that cannot be corrected
        self.data = self.data.drop(index=list(self.extra_local_features))

        # for all extra global features we add NaN values
        # reminder: data is features x samples

        print(f"Adding {len(self.extra_global_features)} extra global features")
        # Reindex the Series to include all indices in the extra_global_features list
        self.data = self.data.reindex(self.data.index.union(list(self.extra_global_features)))
        self.data.loc[list(self.extra_global_features)] = np.nan

        # now we apply the order of the global features to the data matrix
        if set(global_hashed_features) != set(self.data.index):
            raise Exception("INTERNAL ERROR: something went wrong adding all features from all clients, data matrix index != global features list")
        self.data = self.data.reindex(global_hashed_features)
        self.feature_names = global_hashed_features

    def create_design(self, cohorts: List[str]):
        """
        Creates the design matrix. Loads the covariates from the data or the
        design file depending on the config to the design matrix,
        adds an intercept column and adds the batch columns.
        The created design matrix has samples as rows and the following columns,
        considering k batches are given in cohorts:
        intercept, covariates, batch_1, ..., batch_k-1
        Intercept is always 1, covariates are the covariate values, batch_i
        has value 0 if the sample belongs to the batch and value 0 otherwise.
        The reference batch has the value -1 in ALL batch columns.
        Args:
            cohorts: The cohort (batch) names. The order given in cohorts
                is considered as the batch order. Format is ["client|batch"]
        Returns:
            None, sets self.design
        """
        assert isinstance(self.rawdata, pd.DataFrame)
        # first add intercept colum
        if self.design is None:
            self.design = pd.DataFrame({'intercept': np.ones(len(self.sample_names))},
                                        index=self.sample_names)
        else:
            self.design['intercept'] = np.ones(len(self.sample_names))

        design_cohorts = cohorts[:-1]
        # check if we have multiple batches for this client
        cohorts_splitlist = [cohort.split("|") for cohort in cohorts]
        design_cohorts_splitlist = cohorts_splitlist[:-1]
        reference_batch_splitlist = cohorts_splitlist[-1]

        if len([split_batch[0] for split_batch in cohorts_splitlist if self.client_name == split_batch[0]]) == 1:
            print(f"INFO: Client {self.client_name} has only one batch")
            # we only have one batch for this client, check if it is the reference batch
            if self.client_name == cohorts[-1].split("|")[0]:
                print(f"INFO: Client {self.client_name} is the reference batch")
                # we are the reference batch, we just set all batches to -1
                for cohort in design_cohorts:
                    self.design[cohort] = -1
            else:
                # we are not the reference batch, we set our batch to 1 and all others to 0
                print(f"INFO: Client {self.client_name} is not the reference batch")
                for cohort in design_cohorts:
                    if self.client_name == cohort.split("|")[0]:
                        self.design[cohort] = 1
                    else:
                        self.design[cohort] = 0
        else:
            # we have multiple batches for this client, we need to decide per
            # sample
            if not self.batch_col:
                print(f"ERROR: Client {self.client_name} has multiple batches but no batch column was given")
                print(f"The following batches were found for this client: {cohorts}")
                raise ValueError("Batch column was not given but multiple batches for this client were found")
            for cohort_idx, cohorts_split in enumerate(design_cohorts_splitlist):
                # we iterate all non reference batches
                if self.client_name != cohorts_split[0]:
                    # another client, set to 0
                    self.design[design_cohorts[cohort_idx]] = 0
                else:
                    # 0 initialize this whole batch column
                    self.design[design_cohorts[cohort_idx]] = 0
                    # set the samples of this batch to 1
                    sample_indices = self.design[self.design[self.batch_col] == cohorts_split[1]].index
                    self.design.loc[sample_indices, design_cohorts[cohort_idx]] = 1
            # now we fix the reference batch
            reference_batch_indices = self.design[self.design[self.batch_col] == cohorts_splitlist[-1][1]].index
            self.design.loc[reference_batch_indices, design_cohorts] = -1

        if self.batch_col:
            # we remove the batch column from the design matrix, not needed anymore
            self.design = self.design.drop(columns=[self.batch_col])

        # Rearange columns - intersept column,  covariates columns, all cohorts columns
        if self.variables:
            # first we ensure that the variables are in the loaded design matrix
            if self.variables_in_data:
                self.design.join(self.rawdata.T[self.variables])
            else:
                if not all(column_name in self.design.columns for column_name in self.variables):
                    return f"ERROR: the given variables {self.variables} were not found in the given design matrix file."
            self.design = self.design.loc[:, ['intercept'] + self.variables + design_cohorts]
                # IMPORTANT: this order of intercept, variables, cohorts is
                # relevant for the batch correction later as well as for the
                # computation of betas with sample averaging, be careful
                # with changing the order
        else:
            self.design = self.design.loc[:, ['intercept'] + design_cohorts]
        # for privacy reasons, we should protect the covariates added to the
        # design matrix. The design matrix itself is completely reproducable for
        # only one cohort. In case covariates are added, we ensure that
        # there are enough samples (more samples than columns in the design
        # matrix so that XtX=A cannot be solved for X given A
        if self.design.shape[0] <= self.design.shape[1]:
            return f"Privacy Error: There are not enough samples to provide sufficient " +\
                f"privacy, please provide more samples than 1 + #cohorts + #covariantes " +\
                f"({self.design.shape[1]} in this case)"
        return None

    ####### limma: linear regression #########
    def compute_XtX_XtY(self) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Computes the XtX and XtY matrices for the linear regression.
        Returns:
            Returns a tuple of (XtX, XtY, Error) where
            - XtX is the XtX matrix of shape k x k
            - XtY is the XtY vector of shape k
            - Error is a string with an error message or None if no error occured
        """
        assert isinstance(self.design, pd.DataFrame)
        assert isinstance(self.data, pd.DataFrame)

        X = self.design.values
        Y = self.data.values  # Y -> features x samples
        n = Y.shape[0]  # [0] = rows = features
        k = self.design.shape[1]  # [1] = columns = samples

        XtX = np.zeros((n, k, k))
            # for each feature, we calculate the XtX matrix individually
            # X is of shape k x len([intercept, variables, cohorts]), so XtX
            # is of shape k x k
        XtY = np.zeros((n, k))
            # for each feature, we calculate the XtY vector individually
            # X is of shape k x len([intercept, variables, cohorts]), y is of
            # len of samples

        # linear models for each row (for each feature)
        for feature_idx in range(n):
            y = Y[feature_idx, :] # y is all values of one specific feature
            non_nan_idxs = np.argwhere(np.isfinite(y)).reshape(-1)
                # gives the indices of the non-NaN values in y as an 1d array

            if len(non_nan_idxs) > 0:
                x = X[non_nan_idxs, :]
                y = y[non_nan_idxs]
                XtX[feature_idx, :, :] = x.T @ x
                XtY[feature_idx, :] = x.T @ y

        return XtX, XtY, ""

    def remove_batch_effects(self, betas: np.ndarray) -> None:
        """
        The algorithm is from limma::removeBatchEffect()
        Uses the linear model represented by the betas to predict the batch
        effect on the features given the covariates/batch information of the
        samples. That prediction is than substracted from the real data,
        effectively removing the batch effect
        Args:
            betas: an np.array of shape (#features x (1+#covariates+#batches-1))
        Returns:
            None, sets self.data_corrected and adds to self.report
        """
        assert self.data is not None
        assert self.design is not None
        print("INFO: Removing the batch effects with the trained betas")
        self.data_corrected = np.where(self.data == 'NA', np.nan, self.data)
        # Create a mask to only consider the betas concerning batches
        mask = np.ones(betas.shape[1], dtype=bool)
            # Used to reduce beta to just the beta coefficients representing batches
            # mask is set to be true for betas concerning clients and
            # false for betas concerning variables and intercept
            # this is done so that we predict only the batch effect and then
            # remove that batch effect, the covariates effect and the intercept
            # is wanted and should therefore not be removed
        if self.variables:
            mask[[self.design.columns.get_loc(col) for col in ['intercept', *self.variables]]] = False
        else:
            mask[[self.design.columns.get_loc(col) for col in ['intercept']]] = False
        betas_reduced = betas[:, mask]
            # only keep the betas concerning the batches
            # shape of betas is #features x [intercept + #variables + #cohorts-1]
            # shape of betas_reduces is #features x #cohorts-1

        # Calculate the prediction of the data only based on the betas concerning
        # batches (calculate the batch effect)
        if self.variables:
            batch_effect = betas_reduced @ self.design.drop(columns=['intercept', *self.variables]).T
        else:
            batch_effect = betas_reduced @ self.design.drop(columns=['intercept']).T

        # Substract the calculated batch effect (for NaN values we don't need to substract, we keep them as NaN)
        assert isinstance(self.data, pd.DataFrame)
        self.data_corrected = np.where(np.isnan(self.data_corrected), self.data_corrected, self.data_corrected - batch_effect)
        # Add the column_names/index information from the original DataFrame
        self.data_corrected = pd.DataFrame(self.data_corrected, index=self.data.index, columns=self.data.columns)
        # now we drop the extra global features that were added and
        # that we don't actually have data for
        if self.extra_global_features is not None:
            self.data_corrected = self.data_corrected.drop(index=list(self.extra_global_features))
        # finally replace the hashed feature names with the real feature names
        self.data_corrected.rename(index=self.hash2feature, inplace=True)
        np.set_printoptions(threshold=np.iinfo(np.int64).max)
            # we should use sys.maxsize, but that might behave a bit weirdly in docker
            # so we just use the maximum int size which should hopefully be enough
        self.report = ""
        self.report += f"Client {self.client_name}:\n"
        self.report += f"The following betas were used for batch correction:\n{betas}\n"
        self.report += f"The corresponding design matrix was:\n{self.design}\n"
        np.set_printoptions(threshold=1000)
        print(f"INFO: Shape of corrected data after correction: {self.data_corrected.shape}")

    ### Helpers ###
    def _check_consistency_designfile(self) -> None:
        """
        Used to checks whether the design files row names (samples) are consistent
        with the given data (self.data)
        Raises an ValueError if they are not the same
        """
        assert isinstance(self.data, pd.DataFrame)
        design_samples = self.sample_names
        samples = self.data.columns.values
        if not np.array_equal(sorted(design_samples), sorted(samples)):
            print("The sample names in the design file and the data file do not match")
            des_idx = set(design_samples)
            data_idx = set(samples)
            union_indexes = des_idx.union(data_idx)
            intercept_indexes = des_idx.intersection(data_idx)
            print(f"The following indexes are in the union of both files (union): {union_indexes}")
            print(f"The following indexes are in both files (intercept): {intercept_indexes}")
            print(f"The following indexes are only in one of the files (union-intercept): {union_indexes.difference(intercept_indexes)}")
            raise ValueError("aborting...")
