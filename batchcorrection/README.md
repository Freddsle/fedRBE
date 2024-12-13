# Federated limma remove batch effect - Featurecloud
## Description
Federated implementation of limma remove batch effect. Each client is assumed to 
represent one batch, multiple batches per client are NOT supported. 
Normalization can be applied, multiple input formats are supported, 
check the [config](#config) for more information.
The app is provided for free for all use, including commercial use. The app is open source 
and the source code can be reviewed on [github](https://github.com/Freddsle/removeBatch/tree/main/batchcorrection).

## Usage
### Prerequisites
To use fedRBE, the [prerequisites of FeatureCloud need to be fullfilled](https://featurecloud.ai/developers)
In short:
1. Docker needs to be installed
1. The `featurecloud` pip package must be installed: `pip install featurecloud`
1. the FeatureCloud controller needs to be started: `featurecloud controller start`

Furthermore, the app image itself needs to be downloaded. It is provided in the
docker registry of featurecloud, but built on the linux/amd64 platform.
Especially if you're using a Macbook with any of the M-series chips or any other device
not compatible with linux/amd64, please build the image locally:
```
docker build . -t featurecloud.ai/bcorrect:latest
```
Otherwise, you can simply pull it:
``` 
featurecloud app download featurecloud.ai/bcorrect
# alternatively:
docker pull featurecloud.ai/bcorrect:latest
```

### Running the provided sample data
After taking care of the the prerequisites, you can run the 
[provided bash script](https://github.com/Freddsle/removeBatch/blob/main/evaluation_utils/run_sample_experiment.py).
Simply clone the repository and start the script:
```
git clone git@github.com:Freddsle/removeBatch.git
python3 ./evaluation_utils/run_sample_experiment.py
```
The given repository contains the app but furthermore includes all the experiments
done with the app.

### Simulation of a federated workflow
In case fedRBE should be used to test how it would work, with the different
datasets all on the same machine, the testbed of FeatureCloud can be used.

Then, you can use the [testbed of FeatureCloud to run the app](https://featurecloud.ai/development/test)

### Running a federated workflow
To run a federated workflow, simply follow the steps of [creating/joining a project](https://featurecloud.ai/projects)
in FeatureCloud. A project should be joined by at least 3 different clients.


## Input
- a datafile in csv format. Either samples x features or features x samples (expression file). See [config](#config) for more info.
- `config.yml` as specified in [config](#config)
- (Optional) the design matrix with given covariates. Covariates can also be given in the datafile, see [config](#config). 

## Output
The following output is given in each client:
- `only_batch_corrected_data.csv`: This contains only the features that could be batch corrected. 
- `report.txt`: This contains textual information about which features were excluded from the batch effect corection.
Furthermore, it holds the `beta` values calculated for the batch effect correction as well as the internally used design matrix.

The CSV format of output files uses the same `seperator` as given in the config files `seperator`.



## Config
Use the config file to customize. Just upload it together with your training data as `config.yml`
```
flimmaBatchCorrection:
  data_filename: "lab_A_protein_groups_matrix.tsv" 
    # the file containing the expression data/data to be batch corrected. 
  design_filename: lab_A_design.tsv  # the file containing covariates
                                     # it is read in the following way:
                                     # pd.read_csv(design_file_path, sep=seperator, index_col=0)
                                     # should therefore be in the format samples x covariates
                                     # with the first column being the sample indices
  expression_file_flag: True # If true, the datafile is expected to have the samples as columns
                             # and the features as rows. If false, the datafile is expected to have
                             # the samples as rows and the features as columns.
                             # format: boolean
  index_col: "sample" # if expression_file_flag is true, the index_col is the column name of the
                      # features in the data file. 
                      # If expression_file_flag is false, the index_col
                      # is the column name of the samples in the data file.
                      # if no index_col is given, the index is taken from the 0th column for
                      # expression files and generated automatically for samples x features datafiles
                      # format: str or int, int is interpreted as the column index (starting from 0)
  covariates: ['Pyr'] # covariates in linear model. In case a design matrix is given
                      # they are expected in the design matrix, otherwise they are expected 
                      # to be features in the given data (in the data_filename file).
                      # format: list of strings
  separator: "\t" # the separator used in the annotation or data file
  design_separator: "\t" # the separator used in the design file
  normalizationMethod: "log2(x+1)" # the method used for normalization, supported:
                                   # "log2(x+1)"
                                   # if None is given, doesn't the app doesn't normalize
  smpc: True     # whether to use secure multi party computation to securely
                 # aggregate information sent from clients to the coordinator
                 # for more information see https://featurecloud.ai/assets/developer_documentation/privacy_preserving_techniques.html#smpc-secure-multiparty-computation
  min_samples: 5 # The minimum number of samples per feature that are required
                 # to be present and non-missing on the whole client.
                 # if for a feature less than min_samples samples are present,
                 # the client will not send any information about that feature
                 # format: int
                 # Please note that the actual used min_samples might be different
                 # as for privacy reasons min_samples = max(min_samples, len(design.columns)+1)
                 # This is to ensure that a sent Xty matrix always has more samples
                 # than features so that neither X not y can be reconstructed from the Xty matrix
  position: 1    # if a number x is given, the order of the clients will be
                 # be determined by sorting after this number x. 
                 # Example:
                 # Client1 -> position : 0, Client2 -> position: 5, Client3 -> position: 2
                 # Order -> Client1, Client3, Client2
                 # The last client in that order is always used as the reference
                 # batch, so in this case Client2
                 # if empty/None, the order is random, making the batch correction
                 # run non deterministic
  reference_batch: "" # if a string is given, the specified batch is used as
                      # the reference batch for the batch correction
                      # if True, this client is used as the reference batch
                      # if True and multiple batches exist for this client,
                      # the program will halt
                      # if False or an empty string, the last client in the order
                      # determined by the position parameter or the
                      # coordinator is used as the reference batch
                      # if the position parameter and the reference_batch parameter
                      # result in different orders, the program halts
