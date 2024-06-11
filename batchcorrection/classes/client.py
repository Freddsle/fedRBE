from typing import List
import pandas as pd
import numpy as np
import hashlib

class Client:
    def __init__(self,
                cohort_name,
                datafile_path=None,
                expr_file_flag=False,
                design_file_path=None,
                separator="\t",
                design_separator="\t",
                index_col=None,
                covariates=None,
                normalizationMethod=None,
            ):
        """
        Initialize the client and load the datafile and design file already.
        Furthermore, normalization of the input data is done.
        """
        self.cohort_name = cohort_name
        self.data = None
        self.design = None
        self.feature_names = None
        self.sample_names = None
        self.expr_file_flag = expr_file_flag
        self.data_variables = None
        self.extra_global_features = None
        self.variables = covariates

        # for correction
        self.data_corrected = None

        # load dataset
        try:
            self.open_dataset(datafile_path, expr_file_flag, design_file_path, separator, design_separator, index_col)
        except:
            raise ValueError(f"Client {self.cohort_name}: Error loading dataset.")

        self.normalize(normalizationMethod)
        self.XtX = None
        self.XtY = None
        self.SSE = None
        self.cov_coef = None
        self.fitted_logcounts = None
        self.mu = None
        # we hash the feature names and variables to hide covariates and features
        # that this client has but other clients do not have
        self.hash2feature = dict()
        self.feature2hash = dict()
        self.hash2variable = dict()
        if self.feature_names is None:
            raise ValueError(f"Client {self.cohort_name}: Error loading feature names.")
        for feature in self.feature_names:
            feature_hash = hashlib.sha3_256(feature.encode()).hexdigest()
            self.hash2feature[feature_hash] = feature
            self.feature2hash[feature] = feature_hash
        if self.data is None:
            raise ValueError(f"Client {self.cohort_name}: Error loading data.")
        # we use only the hashes of features and only in the end replace again
        # with the real featurenames
        self.data.rename(index=self.feature2hash, inplace=True)

        if covariates:
            for variable in covariates:
                self.hash2variable[hashlib.sha3_256(variable.encode()).hexdigest()] = variable

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
        self.expr_file_flag = expr_file_flag
        if design_file_path:
            design_file = pd.read_csv(design_file_path, sep=design_separator, index_col=0)
            self.design = design_file[self.variables]
        # first we open the data file
        if not index_col:
            # expression file flags should have the feature names as the first
            # column per default
            if expr_file_flag:
                self.rawdata = pd.read_csv(datafile_path, sep=separator, index_col=0)
            # for normak csv, the default is not having samples so we autogenerate
            # sample integers
            else:
                self.rawdata = pd.read_csv(datafile_path, sep=separator)
        else:
            self.rawdata = pd.read_csv(datafile_path, sep=separator, index_col=index_col)
        self.data = self.rawdata

        self.variables_in_data = False
        if not expr_file_flag:
            # First we remove the covariates if no design matrix is given
            if self.variables and not design_file_path:
                self.data = self.data.drop(columns=self.variables)
                self.variables_in_data = True

            # we ensure that we only have numerical values
            self.data = self.data.select_dtypes(include=np.number)
            self.num_excluded_numeric = len(self.rawdata.columns) - len(self.data.columns)
            # finally we transpose as feature x sample data is expected
            self.data = self.data.T


        else:
            # for expression files we just have to remove the rows that have non-numeric values
            # we also need to potentially extract covariates
            tmp = self.rawdata.T
            if self.variables and not design_file_path:
                self.variables_in_data = True
                tmp = tmp.drop(columns=self.variables)
            tmp = tmp.select_dtypes(include=np.number)
            self.num_excluded_numeric = len(self.rawdata.columns) - len(tmp.columns)
            self.data = tmp.T

        self.feature_names = list(self.data.index.values)
        self.sample_names = list(self.data.columns.values)
        self.n_samples = len(self.sample_names)

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
            print(f"WARNING: Client {self.cohort_name}: These extra variables that are not available on all clients will NOT be corrected {extra_local_variables}")
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
        self.extra_global_features = set(global_hashed_features).difference(set(self.feature_names))


        # we ignore the only us features for now
        if len(extra_local_features) > 0:
            print(f"Client {self.cohort_name}: These features are not available on all clients and will NOT be corrected {extra_local_features}")

        # for all extra global features we add a columns of zeros
        # reminder: data is features x samples
        for feature in self.extra_global_features:
            self.data.loc[feature] = 0

        # now we apply the order of the global features to the data matrix
        if set(global_hashed_features) != set(self.data.index):
            raise Exception("INTERNAL ERROR: something went wrong adding all features from all clients, data matrix index != global features list")
        self.data = self.data.reindex(global_hashed_features)
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
        assert self.sample_names
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

        print(f"design was finally created: {self.design}")
        return None

    ####### limma: linear regression #########
    def compute_XtX_XtY(self, minSamples):
        assert self.design is not None
        assert self.data is not None
        X = self.design.values
        Y = self.data.values  # Y - intensities (proteins x samples)
        n = Y.shape[0]  # genes
        k = self.design.shape[1]  # variables
        self.XtX = np.zeros((n, k, k))
        self.XtY = np.zeros((n, k))

        # linear models for each row
        for i in range(n):
            y = Y[i, :]
            # check NA in Y
            ndxs = np.argwhere(np.isfinite(y)).reshape(-1)
            if len(ndxs) != len(y):
                # remove NA from Y and from x
                x = X[ndxs, :]
                y = y[ndxs]
            else:
                x = X
            # privacy check, ensure that y holds enough values
            # Even when just one value is present, it is unknown which sample
            # exactly is choosen
            # counts_y = np.sum((y != 0) & (~np.isnan(y)))
            # if counts_y > 0 and counts_y < minSamples:
            #     return None, None, f"Privacy Error: your expression data must not contain a " +\
            #         f"protein with less than min_sample ({minSamples}) value(s) " +\
            #         f"that are neither 0 nor NaN."
            if minSamples != 0:
                x_boolean = np.where(x != 0, 1, 0)
                y_boolean = np.where(y != 0, 1, 0)
                XtY_boolean = x_boolean.T @ y_boolean
                if not np.all((XtY_boolean >= minSamples) | (XtY_boolean == 0)):
                    return None, None, "Privacy error, less than minSamples would be represented in a value that you would share. The training was stopped"
            self.XtX[i, :, :] = x.T @ x
            self.XtY[i, :] = x.T @ y
        return self.XtX, self.XtY, None

    ####### limma: removeBatchEffects #########

    def remove_batch_effects(self, beta):
        """remove batch effects from intensities using server beta coefficients"""
        #  pg_matrix - np.dot(beta, batch.T)
        assert self.data is not None
        assert self.design is not None
        self.data_corrected = np.where(self.data == 'NA', np.nan, self.data)
        mask = np.ones(beta.shape[1], dtype=bool)
        if self.variables:
            mask[[self.design.columns.get_loc(col) for col in ['intercept', *self.variables]]] = False
        else:
            mask[[self.design.columns.get_loc(col) for col in ['intercept']]] = False
        beta_reduced = beta[:, mask]
        if self.variables:
            dot_product = beta_reduced @ self.design.drop(columns=['intercept', *self.variables]).T
        else:
            dot_product = beta_reduced @ self.design.drop(columns=['intercept']).T

        self.data_corrected = np.where(np.isnan(self.data_corrected), self.data_corrected, self.data_corrected - dot_product)
        self.data_corrected = pd.DataFrame(self.data_corrected, index=self.data.index, columns=self.data.columns)
        # now we drop the extra global features that were added and
        # that we don't actually have data for
        if self.extra_global_features is not None:
            self.data_corrected = self.data_corrected.drop(index=list(self.extra_global_features))
        # finally replace the hashed feature names with the real feature names
        self.data_corrected.rename(index=self.hash2feature, inplace=True)
        if self.expr_file_flag:
            # add the removed columns (non numerical ones)
            additional_columns = self.rawdata.T.columns.difference(self.data_corrected.T.columns)
            self.data_corrected_and_raw = self.data_corrected.T.join(self.rawdata.T[additional_columns]).T
        else:
            # add the removed columns (non numerical ones)
            self.data_corrected = self.data_corrected.T
            additional_columns = self.rawdata.columns.difference(self.data_corrected.columns)
            self.data_corrected_and_raw = self.data_corrected.join(self.rawdata[additional_columns])
        # generate a report of additional features that were not batch corrected
        np.set_printoptions(threshold=np.inf)
        self.report = ""
        self.report += f"Client {self.cohort_name}:\n"
        if len(additional_columns) > 0 or self.num_excluded_federation > 0 or self.num_excluded_numeric > 0:
            self.report += f"{len(additional_columns)} features were not batch corrected in total. Of these:\n"
            self.report += f"{self.num_excluded_numeric} features were excluded for not being numeric data.\n"
            self.report += f"The following features were not batch corrected: {additional_columns}\n"
        self.report += f"The following betas were used for batch correction:\n{beta}\n"
        self.report += f"The corresponding design matrix was:\n{self.design}\n"
        np.set_printoptions(threshold=1000)

