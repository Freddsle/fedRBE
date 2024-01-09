import pandas as pd
import numpy as np

from scipy import linalg

import logging


logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S"
)

EXPERIMENT_TYPE = "TMT"
SAMPLE_TYPE = "sample_type"
TMT_PLEX = "TMT-plex"
REF_SAMPLE = "ref"   


class Client:

    def __init__(self, 
                cohort_name,
                intensities_file_path,
                design_file_path=None,
                experiment_type=EXPERIMENT_TYPE,
            ):

        self.experiment_type = experiment_type
        self.tmt_names = None if self.experiment_type == EXPERIMENT_TYPE else None
        self.cohort_name = cohort_name
        self.intensities = None
        self.design = None
        self.prot_names = None
        self.sample_names = None

        self.variables = None

        # for correction
        self.intensities_corrected = None


        if not self.open_dataset(intensities_file_path, design_file_path):
            raise Exception("Failed to open dataset")

        self.XtX = None
        self.XtY = None
        self.SSE = None
        self.cov_coef = None
        self.fitted_logcounts = None
        self.mu = None
    

    ######### open dataset #########
    def open_dataset(self, intensities_file_path, design_file_path=None):
        """
        For LFQ-data:
        Reads data and design matrices and ensures that sample names are the same.
        Log2(x + 1) transforms intensities.

        """
        self.read_files(intensities_file_path, design_file_path)
        if not self.process_files():
            return False
        # self.check_and_reorder_samples()
        logging.info(
            f"Client {self.cohort_name}: Loaded {len(self.sample_names)} samples and {len(self.prot_names)} proteins."
        )
        return True

    def read_files(self, intensities_file_path, design_file_path=None):
        """Read files using pandas and store the information in the class attributes."""
        self.intensities = pd.read_csv(intensities_file_path, sep="\t", index_col=0)
        print(design_file_path)
        if design_file_path:
            self.design = pd.read_csv(design_file_path, sep="\t", index_col=0)
        self.prot_names = self.intensities.index.values
        self.sample_names = list(self.intensities.columns.values)
        self.n_samples = len(self.sample_names)


    def process_files(self):
        """Process the loaded data based on experiment type."""
        # if self.experiment_type == EXPERIMENT_TYPE:
        #     if not self.process_tmt_files():
        #         return False

        # intensities log2 transformed
        self.intensities = np.log2(self.intensities + 1)
        logging.info(f"Client {self.cohort_name}: Log2(x+1) transformed intensities.")
        return True


    def validate_inputs(self, stored_features, variables):
        """
        Checks if protein_names match global protein group names.
        Important to ensure that client's protein names are in the global protein names. But it's not important that all global protein names are in the client's protein names.
        """
        # store variables for filtering
        self.variables = variables
        self.validate_protein_names(stored_features)
        # self.validate_variables(variables)
        logging.info(
            f"Client {self.cohort_name}: Validated {len(self.sample_names)} samples and {len(self.prot_names)} proteins."
        )

    def validate_protein_names(self, stored_features):
        """
        Ensure that gene names are the same and are in the same order.
        """
        # @Jens do we need all of this?
        # stored_features is the same as client.prot_names, see states.py
        # checking for duplicates could be done by the coordinator instead maybe?
        # and the rest could then be removed I think
        # we could maybe just 
        # 1. move the testing for duplicates to the coordinator
        # 2. reorder the genes here
        print(f"self.prot_names: {self.prot_names}")
        print(f"Given features: {stored_features}")
        global_prots = set(stored_features)
        self_prots = set(self.prot_names).intersection(global_prots)
        #self_prots = set(self.prot_names)
        
        if len(self_prots) != len(set(self_prots)):
            logging.info("Client %s:\tDuplicate protein names found." % self.cohort_name)

        if self_prots != global_prots:
            extra_prots = self_prots.difference(global_prots)
            if len(extra_prots) > 0:
                logging.info(
                    "Client %s:\t%s protein groups absent in other datasets are dropped:"
                    % (self.cohort_name, len(extra_prots))
                )
            else:
                extra_prots = global_prots.difference(self_prots)
                logging.info("Client %s:\t%s protein groups not found." % (self.cohort_name, len(extra_prots)))

        if not self_prots.issubset(global_prots):
            extra_prots = self_prots.difference(global_prots)
            logging.info("Client %s:\t%s protein groups not found in Global proteins. Data load failed." % (self.cohort_name, len(extra_prots)))
            raise ValueError(
                f"Client {self.cohort_name}: Some proteins are missing in the global protein list: {extra_prots}"
            )    
        # reorder genes
        self.prot_names = sorted(self_prots)
        self.intensities = self.intensities.loc[self.prot_names, :] 
    
    def create_design(self, cohorts, minSamples):
        """add covariates to model cohort effects."""

        # first add intercept colum
        assert self.sample_names
        if self.design is None:
            self.design = pd.DataFrame({'intercept': np.ones(len(self.sample_names))},
                                        index=self.sample_names)
        else:
            self.design['intercept'] = np.ones(len(self.sample_names))


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
            if not all(column_name in self.design.columns for column_name in self.variables):
                return f"ERROR: the given variables {self.variables} were not found in the given design matrix file."
            self.design = self.design.loc[:, ['intercept'] + self.variables + cohorts]
            logging.info(f"Client {self.cohort_name}: Design matrix created.")
            logging.info(f"Client {self.cohort_name}: Design matrix columns: {self.design.columns.values}")

        # for privacy reasons, we should protect the covariates added to the 
        # design matrix. The design matrix itself is completely reproducable for
        # only one cohort. In case covariates are added, we ensure that
        # there are enough samples (more samples than columns in the design 
        # matrix so that XtX=A cannot be solved for X given A
        # This way, for every added covariate, we get the information of the
        # combination of the covariate with all other covariates and the number
        # of 1s in the the covariate (via the multiplication with the intercept)
        # However, each covariate adds #samples new unknown values, so as long 
        # as #samples > #covariates, the exact values of samples cannot be 
        # extracted, only the count. 
        if self.design.shape[0] <= self.design.shape[1]:
            return f"Privacy Error: There are more enough samples to provide sufficient " +\
                f"privacy, please more samples than #cohorts + #covariantes " +\
                f"({self.design.shape[1]} in this case)"
        # Furthermore, as we later use each column in the calculation of XtY, 
        # we ensure that each column has at least minSamples values that are neither NaN
        # nor 0. Otherwise, we might have very little values of y represented in
        # a XtY value. 
        # TODO: keep this in or not?
        # counts = self.design.apply(lambda col: (col[(col != 0) & col.notna()].count()))
        # if counts.apply(lambda count: count > 0 and count < minSamples).all():
        #     return f"Privacy Error: All column in the design matrix must " +\
        #            f"contain more than {minSamples} values for privacy reasons."

    ####### limma: linear regression #########
    def compute_XtX_XtY(self, minSamples):
        X = self.design.values
        Y = self.intensities.values  # Y - intensities (proteins x samples)
        n = Y.shape[0]  # genes
        k = self.design.shape[1]  # variables
        self.XtX = np.zeros((n, k, k))
        self.XtY = np.zeros((n, k))

        # linear models for each row
        for i in range(n):  #
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

            self.XtX[i, :, :] = x.T @ x
            self.XtY[i, :] = x.T @ y
        return self.XtX, self.XtY, None

    ####### limma: removeBatchEffects #########

    def remove_batch_effects(self, beta):
        """remove batch effects from intensities using server beta coefficients"""
        #  pg_matrix - np.dot(beta, batch.T)
        self.intensities_corrected = np.where(self.intensities == 'NA', np.nan, self.intensities)

        print("beta shape is {beta.shape}")
        mask = np.ones(beta.shape[1], dtype=bool)
        if self.variables:
            mask[[self.design.columns.get_loc(col) for col in ['intercept', *self.variables]]] = False
        else:
            mask[[self.design.columns.get_loc(col) for col in ['intercept']]] = False
        print(f"mask is: {mask}")
        beta_reduced = beta[:, mask]
        print(f"shape of beta_reduces is {beta_reduced.shape}")
        if self.variables:
            dot_product = beta_reduced @ self.design.drop(columns=['intercept', *self.variables]).T
        else:
            dot_product = beta_reduced @ self.design.drop(columns=['intercept']).T

        self.intensities_corrected = np.where(np.isnan(self.intensities_corrected), self.intensities_corrected, self.intensities_corrected - dot_product)
        self.intensities_corrected = pd.DataFrame(self.intensities_corrected, index=self.intensities.index, columns=self.intensities.columns)
        
        logging.info("Client %s:\tBatch effects removed." % self.cohort_name)