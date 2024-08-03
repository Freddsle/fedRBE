from typing import List, Union
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
        self.cohort_name = None
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

    def hash_names(self, 
                   names: List[str]) -> dict:
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

        # set variables
        self.cohort_name = clientname
        self.smpc = smpc
        self.variables = covariates
        self.min_samples = min_samples
        self.separator = separator

        # load the data
        try:
            self.open_dataset(datafile_path, expr_file_flag, design_file_path, separator, design_separator, index_col)
        except:
            raise ValueError(f"Client {self.cohort_name}: Error loading dataset.")
        self.normalize(normalizationMethod)

        # Hashing logic
        if use_hashing:
            # we hash the feature names and variables to hide covariates and features
            # that this client has but other clients do not have
            if not self.feature_names:
                raise ValueError(f"Client {self.cohort_name}: Error loading feature names.")
            if not self.variables:
                raise ValueError(f"Client {self.cohort_name}: Error loading variables.")
            
            self.feature2hash, self.hash2feature = self.hash_names(self.feature_names)
            self.data.rename(index=self.feature2hash, inplace=True)
            print("shape of data after renaming: ", self.data.shape)
            
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
        """
        print(f"Opening dataset {datafile_path}")
        self.expr_file_flag = expr_file_flag
        if design_file_path:
            self.design = pd.read_csv(design_file_path, sep=design_separator, index_col=0)[self.variables]
        # first we open the data file
        # expression file flags should have the feature names as the first
        # column per default
        if expr_file_flag:
            if index_col is None:
                # In expression files, the defualt is the fiurst column containing the feature names
                # default would however be to create an index col
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
                raise ValueError(f"Client {self.cohort_name}: Error loading data.")
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
        self.n_samples = len(self.sample_names)
        print(f"finished loading data, shape of data: {self.data.shape}, num_features: {len(self.feature_names)}, num_samples: {self.n_samples}")

    def normalize(self, normalizationMethod):
        if normalizationMethod == "log2(x+1)":
            self.data = np.log2(self.data + 1)

    def validate_inputs(self, global_variables_hashed):
        """
        Checks if protein_names match global protein group names.
        Important to ensure that client's protein names are in the global protein names. But it's not important that all global protein names are in the client's protein names.
        """
        #self.validate_feature_names(global_features_hashed)
        self.validate_variables(global_variables_hashed)
        print(f"Client {self.cohort_name}: Inputs validated.")

    def validate_variables(self, global_variables_hashed):
        """
        Checks if the covariates in the local data differ from the global
        intersection of variables. Only uses global variables
        """
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
            print(f"WARNING: Client {self.cohort_name}: {len(extra_local_variables)} extra variables that are not available on all clients will NOT be corrected.")
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
        if self.data is None:
            raise Exception("Data matrix is not loaded yet, cannot set data")

        # Get features only we have and features only global has
        if not self.feature_names:
            raise Exception("Feature names are not loaded yet, cannot set data")

        extra_local_features = set(self.feature_names).difference(set(global_hashed_features))
        print(f"feature names: {len(self.feature_names)}")
        print(f"global features: {len(global_hashed_features)}")
        print(f"Extra local features: {len(extra_local_features)}")
        self.extra_global_features = set(global_hashed_features).difference(set(self.feature_names))
        # print(f"Extra global features: {list(self.extra_global_features)[:10]}")
        print(f"Extra global features: {len(list(self.extra_global_features))}")


        # we ignore the only us features for now
        # if len(extra_local_features) > 0:
        #     print(f"Client {self.cohort_name}: {len(extra_local_features)} features are not available on all clients and will NOT be corrected")

        # for all extra global features we add a columns of zeros
        # reminder: data is features x samples

        print(f"Adding {len(self.extra_global_features)} extra global features")
        # Reindex the Series to include all indices in the extra_global_features list
        self.data = self.data.reindex(self.data.index.union(list(self.extra_global_features)))
        self.data.loc[list(self.extra_global_features)] = np.nan

        # now we apply the order of the global features to the data matrix
        print(f"Got {len(global_hashed_features)} global features and {len(self.data.index)} features in the data matrix")
        if set(global_hashed_features) != set(self.data.index):
            raise Exception("INTERNAL ERROR: something went wrong adding all features from all clients, data matrix index != global features list")
        print(f"Before reindexing got this data: {self.data.shape}")
        self.data = self.data.reindex(global_hashed_features)
        print(f"After reindexing got this data: {self.data.shape}")
        self.feature_names = global_hashed_features

    def validate_feature_names(self, global_features_hashed):
        """
        DEPRECATED
        Compares the features gotten from the global aggregation vs the one
        the client has. It only uses the global features.
        """
        global_features = [self.hash2feature.get(hashed, hashed) for hashed in global_features_hashed]
        assert self.feature_names is not None
        extra_local_features = set(self.feature_names).difference(set(global_features))
        self.num_excluded_federation = 0
        extra_global_features = set(global_features).difference(set(self.feature_names))
        if len(extra_global_features) > 0:
            print(f"Found these additional global features: {extra_global_features}")
            print("my features are: ", self.feature_names)
            print("global features are: ", global_features)
            print(f"len of my features: {len(self.feature_names)}, len of global features: {len(global_features)}")
            raise Exception("ERROR: Some features found globally were not found locally, this is likely an error with the app")
        if len(extra_local_features) > 0:
            self.num_excluded_federation = len(extra_local_features)
            print(f"Client {self.cohort_name}: These extra features that are not available on all clients will NOT be corrected {extra_local_features}")

        # reorder genes and only look at the global_features
        self.feature_names = sorted(global_features)
        if self.data is None:
            raise Exception("Data matrix is not loaded yet, cannot set data")
        self.data = self.data.loc[self.feature_names, :]
        # set feature_names and variables to sets again in case this was changed here
        self.feature_names = list(self.feature_names)
        self.variables = list(self.variables) if self.variables else None

    def create_design(self, cohorts):
        """add covariates to model cohort effects."""
        # first add intercept colum
        if self.design is None:
            self.design = pd.DataFrame({'intercept': np.ones(len(self.sample_names))},
                                        index=self.sample_names)
        else:
            self.design['intercept'] = np.ones(len(self.sample_names))


        # Design contains #clients-1 entries, if this client is the one that
        # is excluded it sets -1 to all cohorts, otherwise we set 0 for other
        # cohorts and 1 for this one
        if self.cohort_name not in cohorts:
            for cohort in cohorts:
                self.design[cohort] = -1
        else:
            for cohort in cohorts:
                if self.cohort_name == cohort:
                    self.design[cohort] = 1
                else:
                    self.design[cohort] = 0

        # if covariates is not None - rearrange columns - intersept column,  covariates columns, all cohorts columns
        if self.variables:
            # first we ensure that the variables are in the loaded design matrix
            if self.variables_in_data:
                assert self.data is not None
                self.design.join(self.rawdata.T[self.variables])
            else:
                if not all(column_name in self.design.columns for column_name in self.variables):
                    return f"ERROR: the given variables {self.variables} were not found in the given design matrix file."
            self.design = self.design.loc[:, ['intercept'] + self.variables + cohorts]
                # IMPORTANT: this order of intercept, variables, cohorts is
                # relevant for the batch correction later as well as for the
                # computation of betas with sample averaging, be careful
                # with changing the order
        # for privacy reasons, we should protect the covariates added to the
        # design matrix. The design matrix itself is completely reproducable for
        # only one cohort. In case covariates are added, we ensure that
        # there are enough samples (more samples than columns in the design
        # matrix so that XtX=A cannot be solved for X given A
        if self.design.shape[0] <= self.design.shape[1]:
            return f"Privacy Error: There are not enough samples to provide sufficient " +\
                f"privacy, please more samples than #cohorts + #covariantes " +\
                f"({self.design.shape[1]} in this case)"

        pd.set_option('display.max_columns', None)
        print(f"design was finally created: {self.design}")
        print(f"shape of design: {self.design.shape}")
        pd.reset_option('display.max_columns')
        return None

    ####### limma: linear regression #########
    def compute_XtX_XtY(self):
        assert self.design is not None
        assert self.data is not None

        X = self.design.values
        Y = self.data.values  # Y -> features x samples
        n = Y.shape[0]  # [0] = rows = features
        k = self.design.shape[1]  # [1] = columns = samples
        
        self.XtX = np.zeros((n, k, k))
            # for each feature, we calculate the XtX matrix individually
            # X is of shape k x len([intercept, variables, cohorts]), so XtX
            # is of shape k x k
        self.XtY = np.zeros((n, k))
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
                self.XtX[feature_idx, :, :] = x.T @ x
                self.XtY[feature_idx, :] = x.T @ y

                # privacy check, ensure that y holds enough values
                # Even when just one value is present, it is unknown which sample
                # exactly is choosen
                # counts_y = np.sum((y != 0) & (~np.isnan(y)))
                # if counts_y > 0 and counts_y < minSamples:
                #     return None, None, f"Privacy Error: your expression data must not contain a " +\
                #         f"protein with less than min_sample ({minSamples}) value(s) " +\
                #         f"that are neither 0 nor NaN."
                if self.min_samples != 0:
                    x_boolean = np.where(x != 0, 1, 0)
                    y_boolean = np.where(y != 0, 1, 0)
                    XtY_boolean = x_boolean.T @ y_boolean
                    if not np.all((XtY_boolean >= self.min_samples) | (XtY_boolean == 0)):
                        return None, None, "Privacy error, less than minSamples would be represented in a value that you would share. The training was stopped"

            # #TODO: rmv:
            # if 0 == len(non_nan_idxs):
            #     print(f"In XtX calc, feature {self.data.index[feature_idx]} has NO non-NaN values")
            #     print(f"This results in the following XtX matrix: \n{self.XtX[feature_idx, :, :]}")
            #     print(f"This results in the following XtY vector: \n{self.XtY[feature_idx, :]}")
            #     print(f"shape of x: {x.shape}, shape of y: {y.shape}")
            #     print(f"XtX shape: {self.XtX[feature_idx, :, :].shape}, XtY shape: {self.XtY[feature_idx, :].shape}")
            #     print(f"k is {k}, n is {n}")
            #     if not np.all(self.XtX[feature_idx, :, :] == 0):
            #         raise ValueError("XtX is not all zeros, although no data was present")
            #     if not np.all(self.XtY[feature_idx, :] == 0):
            #         raise ValueError("XtY is not all zeros, although no data was present")
            # if np.all(y == 0):
            #     print(f"Feature {feature_idx} has only zeros in Y")
            #     print(f"This results in the following XtX matrix: \n{self.XtX[feature_idx, :, :]}")
            #     print(f"This results in the following XtY vector: \n{self.XtY[feature_idx, :]}")
            #     print(f"shape of x: {x.shape}, shape of y: {y.shape}")
            #     if not np.all(self.XtX[feature_idx, :, :] == 0):
            #         raise ValueError("XtX is not all zeros, although no data was present")
            #     if not np.all(self.XtY[feature_idx, :] == 0):
            #         raise ValueError("XtY is not all zeros, although no data was present")
            # if np.all(np.isnan(self.XtX[feature_idx, :, :])):
            #     print(f"Feature {feature_idx} has only NaN in XtX")
            #     if not np.all(self.XtX[feature_idx, :, :] == 0):
            #         raise ValueError("XtX is not all zeros, although no data was present")
            #     if not np.all(self.XtY[feature_idx, :] == 0):
            #         raise ValueError("XtY is not all zeros, although no data was present")
            # if np.all(np.isnan(self.XtY[feature_idx, :])):
            #     print(f"Feature {feature_idx} has only NaN in XtY")
            #     if not np.all(self.XtX[feature_idx, :, :] == 0):
            #         raise ValueError("XtX is not all zeros, although no data was present")
            #     if not np.all(self.XtY[feature_idx, :] == 0):
            #         raise ValueError("XtY is not all zeros, although no data was present")
            if len(non_nan_idxs) == 0:
                if not np.all(self.XtX[feature_idx, :, :] == 0):
                    raise ValueError("XtX is not all zeros, although no data was present")
                if not np.all(self.XtY[feature_idx, :] == 0):
                    raise ValueError("XtY is not all zeros, although no data was present")
        print(f"final vectors to be sent: XtX shape: {self.XtX.shape}, XtY shape: {self.XtY.shape}")
        return self.XtX, self.XtY, None

    ####### limma: removeBatchEffects #########

    def remove_batch_effects(self, beta):
        """remove batch effects from intensities using server beta coefficients"""
        #  pg_matrix - np.dot(beta, batch.T)
        assert self.data is not None
        assert self.design is not None
        print("start remove_batch_effects")
        print(f"Shape of data: {self.data.shape}")
        print("Shape of beta: ", beta.shape)
        self.data_corrected = np.where(self.data == 'NA', np.nan, self.data)
        mask = np.ones(beta.shape[1], dtype=bool)
            # mask has true for betas concerning clients and false for betas concerning variables and intercept
        if self.variables:
            mask[[self.design.columns.get_loc(col) for col in ['intercept', *self.variables]]] = False
        else:
            mask[[self.design.columns.get_loc(col) for col in ['intercept']]] = False
        beta_reduced = beta[:, mask]
            # only keep the betas concerning the clients
            # shape of beta is #features x [intercept + variables + cohorts-1]
        if self.variables:
            dot_product = beta_reduced @ self.design.drop(columns=['intercept', *self.variables]).T
            # only keep betas concerning this client
            # the client that set -1 to all clients uses all betas
        else:
            dot_product = beta_reduced @ self.design.drop(columns=['intercept']).T

        print(f"Beta_reduced contains {np.sum(np.isnan(beta_reduced))} Nan values")
        self.data_corrected = np.where(np.isnan(self.data_corrected), self.data_corrected, self.data_corrected - dot_product)
        self.data_corrected = pd.DataFrame(self.data_corrected, index=self.data.index, columns=self.data.columns)
        print(f"Shape of corrected data after correction: {self.data_corrected.shape}")
        # now we drop the extra global features that were added and
        # that we don't actually have data for
        if self.extra_global_features is not None:
            self.data_corrected = self.data_corrected.drop(index=list(self.extra_global_features))
        # finally replace the hashed feature names with the real feature names
        print(f"index is {self.data_corrected.index}")
        print(f"Amount of index found in hash2feature: {len([hashed for hashed in self.data_corrected.index if hashed in self.hash2feature])}/{len(self.data_corrected.index)}")
        self.data_corrected.rename(index=self.hash2feature, inplace=True)
        print(f"After renaming got this data_corrected: {self.data_corrected}")
        # if self.expr_file_flag:
        #     # add the removed columns (non numerical ones)
        #     additional_columns = self.rawdata.T.columns.difference(self.data_corrected.T.columns)
        #     print(f"Additional columns expr file: {additional_columns}")
        #     self.data_corrected_and_raw = self.data_corrected.T.join(self.rawdata.T[additional_columns]).T
        # else:
        #     # add the removed columns (non numerical ones)
        #     self.data_corrected = self.data_corrected.T
        #     additional_columns = self.rawdata.columns.difference(self.data_corrected.columns)
        #     self.data_corrected_and_raw = self.data_corrected.join(self.rawdata[additional_columns])
        # generate a report of additional features that were not batch corrected
        np.set_printoptions(threshold=np.inf)
        self.report = ""
        self.report += f"Client {self.cohort_name}:\n"
        # if len(additional_columns) > 0 or self.num_excluded_federation > 0 or self.num_excluded_numeric > 0:
        #     self.report += f"{len(additional_columns)} features were not batch corrected in total. Of these:\n"
        #     self.report += f"{self.num_excluded_numeric} features were excluded for not being numeric data.\n"
        #     self.report += f"The following features were not batch corrected: {additional_columns}\n"
        self.report += f"The following betas were used for batch correction:\n{beta}\n"
        self.report += f"The corresponding design matrix was:\n{self.design}\n"
        np.set_printoptions(threshold=1000)
        print(f"remove batch final corrected data shape: {self.data_corrected.shape}")

    ### Helpers ###
    def _check_consistency_designfile(self) -> None:
        """
        Used to checks whether the design files row names (samples) are consistent
        with the given data
        Raises an ValueError if they are not the same
        """
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
