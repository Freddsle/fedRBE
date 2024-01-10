import os
import bios
import time #TODO: rmv

import numpy as np

from scipy import linalg
from FeatureCloud.app.engine.app import AppState, app_state, Role, LogLevel

from classes.client import Client

# FeatureCloud requires that apps define the at least the 'initial' state.
# This state is executed after the app instance is started.
@app_state('initial')
class InitialState(AppState):

    def register(self):
        self.register_transition('common_genes', Role.COORDINATOR)
        self.register_transition('validate', Role.PARTICIPANT)
    
    def run(self):
        # read in the config
        config = bios.read(os.path.join(os.getcwd(), "mnt", "input", "config.yaml"))
        config = config["flimmaBatchCorrection"]
        minSamples = config["min_samples"]
        covariates = config["covariates"]
        design_file_path = os.path.join(os.getcwd(), "mnt", "input", config["annotation_filename"])
        # defining the client
        cohort_name = self.id
        intensity_file_path = os.path.join(os.getcwd(), "mnt", "input", config["expression_filename"])
        experiment_type = 'DIA' #TODO: also support other types?
        client = Client(
            cohort_name,
            intensity_file_path,
            design_file_path,
            experiment_type,
            covariates,
        ) # initializing the client includes loading and preprocessing of the data
        self.store(key='client', value=client)
        self.store(key='minSamples', value=minSamples)
        self.store(key='covariates', value=covariates)
        self.configure_smpc()
        # send list of protein names (genes) to coordinator
        print("[initial] Sending the following prot_names to the coordinator")
        print(client.prot_names)
        self.send_data_to_coordinator(client.prot_names,
                                    send_to_self=True,
                                    use_smpc=False)

        if self.is_coordinator:
            return 'common_genes'
        return 'validate'


@app_state('common_genes')
class CommonGenesState(AppState):

    def register(self):
        self.register_transition('validate', Role.COORDINATOR)
        
    def run(self):
        # wait for each client to send the list of genes they have
        print("[common_genes] Gathering genes from all clients")
        lists_of_genes = self.gather_data(is_json=False)
            # SMPC will not work as strings can't be averaged
        print("[common_genes] Gathered data from all clients")
        # generate a sorted list of the genes that are available on each client
        prot_names = list()
        for l in lists_of_genes:
            if len(prot_names) == 0:
                prot_names = l
            else:
                prot_names = sorted(list(set(prot_names) & set(l)))
        print("[common_genes] Common_genes were found")

        # create index for design matrix
        variables = self.load("covariates")
        print("[common_genes] index of design matrix created")
        # Send prot_names and variables to all 
        self.broadcast_data((prot_names, variables),
                            send_to_self=True, memo="commonGenes")
        print("[common_genes] Data was set to be broadcasted:")
        print(prot_names)
        print(variables)
        print("[common_genes] transitioning to validate, data was broadcastet")
        return 'validate'


@app_state('validate')
class ValidationState(AppState):

    def register(self):
        self.register_transition('compute_XtX_XtY', Role.BOTH)

    def run(self):
        # obtain and safe common genes and indices of design matrix
        print("[validate] {} waiting for data".format(self.id)) #TODO: rmv
        prot_names, variables = self.await_data(n=1, is_json=False, memo="commonGenes")
        print(f"[validate] Got prot_names={prot_names}, variabes={variables}")
        client = self.load('client')
        client.variables = variables
        client.prot_names = prot_names

        client.validate_inputs(client.prot_names, client.variables)
        print("[validate] {} Inputs have been validated".format(self.id))
        # get all client names to generate design matrix
        all_client_names = self.clients
        err = client.create_design(all_client_names[:-1], self.load("minSamples"))
        if err:
            self.log(err, LogLevel.FATAL)
        print("[validate] {} design has been created".format(self.id))
        # filter the intinsities to only use the columns that are given on each client
        client.intensities = client.intensities.loc[client.prot_names, :]
        print("[validate] {} intensities has been created".format(self.id))
        self.store(key='client', value=client)
        print("[validate] {} changing states".format(self.id))
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
        XtX, XtY, err = client.compute_XtX_XtY(self.load("minSamples"))
        if err != None:
            self.log(err, LogLevel.FATAL)


        # send XtX and XtY
        print("[compute_XtX_XtY] Computation done, sending data to coordinator")
        print(f"[compute_XtX_XtY] XtX of shape {XtX.shape}, X of shape {client.design.shape}, XtY of shape {XtY.shape}")
        self.send_data_to_coordinator([XtX, XtY],
                                send_to_self=True,
                                use_smpc=False)

        if self.is_coordinator:
            return 'compute_beta'
        return 'include_correction'
    

@app_state('compute_beta')
class ComputeCorrectionState(AppState):

    def register(self):
        self.register_transition('include_correction', Role.COORDINATOR)

    def run(self):
        # wait for each client to compute XtX and XtY and collect data
        print("[compute_beta] gathering data")
        XtX_XtY_list = self.gather_data(use_smpc=False)
        print("[compute_beta] Got XtX_XtY_list from gather_data")
        XtX_list = list()
        XtY_list = list()
        for ele in XtX_XtY_list:
            # ele is list[XtX, XtY], see send_data of compute_XtX_XtY
            print(f"XtXList, shape of element: {ele[0].shape}")
            print(f"XtYList, shape of element: {ele[1].shape}")
            XtX_list.append(ele[0])
            XtY_list.append(ele[1])

        # set up matrices for global XtX and XtY
        client = self.load('client')
        k = client.design.shape[1]
        print(f"k is: {k}")
        n = len(client.prot_names)
        print(f"n is {n}")
        XtX_glob = np.zeros((n, k, k))
        XtY_glob = np.zeros((n, k))
        stdev_unscaled = np.zeros((n, k))
        for i in range(0, len(self.clients)):
            XtX_glob += XtX_list[i]
            XtY_glob += XtY_list[i]
        
        # calcualte beta and std. dev.
        beta = np.zeros((n, k))
        for i in range(0, n):
            #TODO: this might throw an error if XtX_glob[i, :, :] is a singular
            # matrix
            print(f"Creating beta with XtX = {XtX_glob[i, :, :]}")
            invXtX = linalg.inv(XtX_glob[i, :, :])
            beta[i, :] = invXtX @ XtY_glob[i, :]
            stdev_unscaled[i, :] = np.sqrt(np.diag(invXtX))

        print(f"[compute_beta] betas calculated: beta.shape is {beta.shape}")
        print(f"[compute_beta] stdev_unscaled.shape is {stdev_unscaled.shape}")
        # send beta to clients so they can correct their data
        print("[compute_beta] broadcasting betas")
        self.broadcast_data(beta,
                            send_to_self=True, memo="beta")
        
        return 'include_correction'


@app_state('include_correction')
class IncludeCorrectionState(AppState):

    def register(self):
        self.register_transition('terminal')

    def run(self):
        # wait for the coordinator to calcualte beta
        beta = self.await_data(n=1, is_json=False, memo="beta")
        print(f"[include_correction] beta gotten is of shape {beta.shape}")
        client = self.load('client')

        # remove the batch effects in own data and safe the results
        client.remove_batch_effects(beta)
        output_file = os.path.join(os.getcwd(), "mnt", "output", "intensities_corrected.csv")
        client.intensities_corrected.to_csv(output_file)
        
        return 'terminal'
