import os

import numpy as np

from scipy import linalg
from FeatureCloud.app.engine.app import AppState, app_state, Role

from classes.client import Client


# CONFIG
use_smpc = True
gene_threshold = 2

# FeatureCloud requires that apps define the at least the 'initial' state.
# This state is executed after the app instance is started.
@app_state('initial')
class InitialState(AppState):

    def register(self):
        self.register_transition('common_genes', Role.COORDINATOR)
        self.register_transition('validate', Role.PARTICIPANT)
    
    def run(self):
        # defining the client
        cohort_name = self.id
        intensity_file_path = os.path.join(os.getcwd(), "mnt", "input", "protein_groups_matrix.tsv")
        experiment_type = 'DIA'
        client = Client(
            cohort_name,
            intensity_file_path,
            experiment_type,
            gene_threshold
        ) # initializing the client includes loading and preprocessing of the data
        self.store(key='client', value=client)

        # send list of protein names (genes) to coordinator
        self.send_data_to_coordinator(client.prot_names,
                                    send_to_self=True,
                                    use_smpc=use_smpc)

        if self.is_coordinator:
            return 'common_genes'
        return 'validate'


@app_state('common_genes')
class CommonGenesState(AppState):

    def register(self):
        self.register_transition('validate', Role.COORDINATOR)
        
    def run(self):
        # wait for each client to send the list of genes they have
        print(1)
        lists_of_genes = self.gather_data(is_json=use_smpc)
        print(2)
        # generate a sorted list of the genes that are available on each client
        prot_names = list()
        print(3)
        for l in lists_of_genes:
            if len(prot_names) == 0:
                prot_names = l
            else:
                prot_names = sorted(list(set(prot_names) & set(l)))
        print(4)

        # create index for design matrix
        variables = ['intercept'] + self.clients[:-1]
        print(5)
        # Send prot_names and variables to all 
        self.broadcast_data([prot_names, variables],
                            send_to_self=True,
                            use_smpc=use_smpc)
        print(6)
        return 'validate'


@app_state('validate')
class ValidationState(AppState):

    def register(self):
        self.register_transition('compute_XtX_XtY', Role.BOTH)

    def run(self):
        # obtain and safe common genes and indices of design matrix
        prot_names, variables = self.await_data(n=1, is_json=use_smpc)
        client = self.load('client')
        client.variables = variables
        client.prot_names = prot_names

        client.validate_inputs(client.prot_names, client.variables)

        # get all client names to generate design matrix
        all_client_names = self.clients
        client.create_design(all_client_names[:-1])
        # filter the intinsities to only use the columns that are given on each client
        client.intensities = client.intensities.loc[client.prot_names, :]
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
        # sort intensities by sample names and proteins
        client.intensities = client.intensities.loc[client.prot_names, client.sample_names]
        client.n_samples = len(client.sample_names)
        
        # compute XtX and XtY
        XtX, XtY = client.compute_XtX_XtY()

        # TODO: check rank of XtY
        # stop if number of rows (or rows - 1, which would be more safe) is equal than rank of matrix

        # send XtX and XtY
        self.send_data_to_coordinator([XtX, XtY],
                                send_to_self=True,
                                use_smpc=use_smpc)

        if self.is_coordinator:
            return 'compute_beta'
        return 'include_correction'
    

@app_state('compute_beta')
class ComputeCorrectionState(AppState):

    def register(self):
        self.register_transition('include_correction', Role.COORDINATOR)

    def run(self):
        # wait for each client to compute XtX and XtY and collect data
        XtX_XtY_list = self.gather_data(is_json=use_smpc)
        XtX_list = list()
        XtY_list = list()
        for XtX, XtY in XtX_XtY_list:
            XtX_list.append(XtX)
            XtY_list.append(XtY)

        # set up matrices for global XtX and XtY
        client = self.load('client')
        k = len(client.variables)
        n = len(client.prot_names)
        XtX_glob = np.zeros((n, k, k))
        XtY_glob = np.zeros((n, k))
        stdev_unscaled = np.zeros((n, k))
        for i in range(0, len(self.clients)):
            XtX_glob += XtX_list[i]
            XtY_glob += XtY_list[i]
        
        # calcualte beta and std. dev.
        beta = np.zeros((n, k))
        for i in range(0, n):
            invXtX = linalg.inv(XtX_glob[i, :, :])
            beta[i, :] = invXtX @ XtY_glob[i, :]
            stdev_unscaled[i, :] = np.sqrt(np.diag(invXtX))

        # send beta to clients so they can correct their data
        self.broadcast_data(beta,
                            send_to_self=True,
                            use_smpc=use_smpc)
    
        return 'include_correction'


@app_state('include_correction')
class IncludeCorrectionState(AppState):

    def register(self):
        self.register_transition('terminal')

    def run(self):
        # wait for the coordinator to calcualte beta
        beta = self.await_data(n=1, is_json=use_smpc)
        client = self.load('client')

        # remove the batch effects in own data and safe the results
        client.remove_batch_effects(beta[:,1:])
        output_file = os.path.join(os.getcwd(), "mnt", "output", "intensities_corrected.csv")
        client.intensities_corrected.to_csv(output_file)
        
        return 'terminal'
