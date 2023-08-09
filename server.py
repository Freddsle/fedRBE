import pandas as pd
import numpy as np

from scipy import linalg

import logging


logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S"
)


def cov2cor(cov_coef):
    cor = np.diag(cov_coef) ** -0.5 * cov_coef
    cor = cor.T * np.diag(cov_coef) ** -0.5
    np.fill_diagonal(cor, 1)
    return cor


class Server:
    def __init__(self, covariates):
        self.covariates = sorted(covariates)
        self.variables = self.covariates
        self.client_names = []
        self.n_samples_per_cli = []
        # self.n_tmt_per_cli = []
        self.stored_features = []

        # attributes for fedLmFit
        self.XtX_glob = None
        self.Xty_glob = None
        self.beta = None
        self.cov_coef = None
        self.stdev_unscaled = None
        self.var = None
        self.sigma = None
        self.df_residual = None
        self.df_total = None
        self.Amean = None
        self.results = None
        self.table = None


    def join_client(self, client):
        """
        Collects names of genes and variables, and the number of samples from client.
        """
        if client.cohort_name in self.client_names:
            logging.error(f"Choose client name other than {self.client_names}")
            logging.error(f"Failed to join client {client.cohort_name}")
            return False

        if len(self.client_names) == 0:
            self.stored_features = sorted(set(client.prot_names))
        else:
            # keep all features
            #self.stored_features = sorted(list(set(self.stored_features + client.prot_names)))
            # keep only shared features - intersection of features from all clients
            self.stored_features = sorted(list(set(self.stored_features) & set(client.prot_names)))

        self.n_samples_per_cli.append(client.n_samples)
        # self.n_tmt_per_cli.append(client.n_tmt)
        self.client_names.append(client.cohort_name)
        logging.info(f"Server: joined client  {client.cohort_name}")
        # self.prots_na_table = self.prots_na_table.loc[self.stored_features, :]
        return True

    ###### fedLmFit ######## 
    def compute_beta_and_beta_stdev(self, XtX_list, XtY_list):
        """Calcualtes global beta and variance of beta"""
        k = len(self.variables)
        n = len(self.stored_features)
        self.XtX_glob = np.zeros((n, k, k))
        self.XtY_glob = np.zeros((n, k))
        self.stdev_unscaled = np.zeros((n, k))

        logging.info(f"Server: computing global beta and beta stdev, k = {k}, n = {n}")

        for i in range(0, len(self.client_names)):
            self.XtX_glob += XtX_list[i]
            self.XtY_glob += XtY_list[i]
            
        self.beta = np.zeros((n, k))
        self.rank = np.ones(n) * k

        for i in range(0, n):
            invXtX = linalg.inv(self.XtX_glob[i, :, :])
            
            self.beta[i, :] = invXtX @ self.XtY_glob[i, :]
            self.stdev_unscaled[i, :] = np.sqrt(np.diag(invXtX))  # standart err for b coefficients
    