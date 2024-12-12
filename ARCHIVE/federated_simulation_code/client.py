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
                experiment_type=EXPERIMENT_TYPE
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
        logging.info(
            f"Client {self.cohort_name}: Loaded {len(self.sample_names)} samples and {len(self.prot_names)} proteins."
        )
        return True

    def read_files(self, intensities_file_path, design_file_path=None):
        """Read files using pandas and store the information in the class attributes."""
        self.intensities = pd.read_csv(intensities_file_path, sep="\t", index_col=0)
        if design_file_path:
            self.design = pd.read_csv(design_file_path, sep="\t", index_col=0)
        self.validate_useful_genes()
        self.sample_names = list(self.intensities.columns.values)
        self.n_samples = len(self.sample_names)


    def validate_useful_genes(self):
        """
        Check if there are genes where only one sample is given. These need to
        be removed as they would be observable during later steps.
        """
        n_col = self.intensities.shape[1]
        num_nans = self.intensities.isna().sum(axis=1)
        n = 2 # number of existing values at least needed to include the protein

        select_rows = self.intensities[(n_col - num_nans) >= n]
        self.prot_names = select_rows.index.values
        
        return

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
    
    def create_design(self, cohorts):
        """add covariates to model cohort effects."""

        # first add intersept colum
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

        # if covariates is not None - rearrange columns - intersept column, all cohorts columns, covariates columns
        self.design = self.design.loc[:, ['intercept'] + self.variables + cohorts]
        logging.info(f"Client {self.cohort_name}: Design matrix created.")
        logging.info(f"Client {self.cohort_name}: Design matrix columns: {self.design.columns.values}")

    ####### limma: linear regression #########
    def compute_XtX_XtY(self):
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

            self.XtX[i, :, :] = x.T @ x
            self.XtY[i, :] = x.T @ y
        return self.XtX, self.XtY

    ####### limma: removeBatchEffects #########

    def remove_batch_effects(self, beta):
        """remove batch effects from intensities using server beta coefficients"""
        #  pg_matrix - np.dot(beta, batch.T)
        self.intensities_corrected = np.where(self.intensities == 'NA', np.nan, self.intensities)
        
        positions = [self.design.columns.get_loc(col) for col in ['intercept', *self.variables]]
        beta_reduced = np.delete(beta, positions, axis=1)
        dot_product = beta_reduced @ self.design.drop(columns=['intercept', *self.variables]).T
        
        # correct values that are non nan, ignore nans
        self.intensities_corrected = np.where(np.isnan(self.intensities_corrected), self.intensities_corrected, self.intensities_corrected - dot_product)
        self.intensities_corrected = pd.DataFrame(self.intensities_corrected, index=self.intensities.index, columns=self.intensities.columns)
        
        logging.info("Client %s:\tBatch effects removed." % self.cohort_name)
