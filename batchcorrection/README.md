# Federated limma remove batch effect - Featurecloud
## Description
Federated implementation of limma remove batch effect. Each client is assumed to 
represent one batch. Normalization can be applied, multiple input formats are 
supported, check the [config](#config) for more information.

## Input
- a datafile in csv format. Either samples x features or features x samples (expression file). See config for more info.
- `config.yml` as specified in [Config](#config)
- (Optional) the design matrix with given covariates. Covariates can also be given in the datafile, see [Config](#config). 

## Output
The following output is given in each client and only concerns each clients individual data. 
The CSV format of output files uses the same `seperator` as given in the config files `seperator`.
The following output is produced:

- `only_batch_corrected_data.csv`: This contains only the features that could be batch corrected. Only features that are available in ALL clients and are numeric can be batch corrected. 
- `all_data.csv`: This contains all batch corrected features as well as all other features that could not be batch corrected. CSV format uses the same `seperator` as given in `seperator`.
- `report.txt`: This contains textual information about which features were excluded from the batch effect corection. Furthermore, it holds the `beta` values calculated for the batch effect correction as well as the internally used design matrix.


## Config
Use the config file to customize. Just upload it together with your training data as `config.yml`
```
flimmaBatchCorrection:
  data_filename: "lab_A_protein_groups_matrix.tsv" 
    # the file containing the expression data. Must be a 2d matrix where
    # samples are given as columns and features as rows
  design_filename: lab_A_design.tsv # the file containing covariates
                                     # it is read in the following way:
                                     # pd.read_csv(design_file_path, sep=seperator, index_col=0)
  expression_file_flag: True # If true, the datafile is expected to have the samples as columns
                             # and the features as rows. If false, the datafile is expected to have
                             # the samples as rows and the features as columns.
                             # format: boolean
  index_col: "sample" # if expression_file_flag is true, the index_col is the column name of the
                      # features in the data file. If expression_file_flag is false, the index_col
                      # is the column name of the samples in the data file.
                      # if no index_col is given, the index is taken from the 0th column for
                      # expression files and generated autopmatically for standard samplesxfeatures datafiles
                      # format: str or int, int is interpreted as the column number

  covariates: ['Pyr'] # covariates in linear model. In case a design matrix is given
                      # they are expected in the design matrix, otherwise they are expected 
                      # to be features in the given data.
                      # format: list of strings
  separator: "\t" # the separator used in the annotation or data file
  design_separator: "\t" # the separator used in the design file
  normalizationMethod: "log2(x+1)" # the method used for normalization, supported:
                                   # "log2(x+1)"
  # The following options are for privacy, we recommend one of these tiers:
  # minimum privacy:
  #   smpc = False
  #   min_samples = 0
  # medium privacy, medium slowdown
  #   smpc = True
  #   min_samples = 0
  # maximum privacy, medium to high slowdown
  #   smpc = True
  #   min_samples = 5
  # generally minimum or medium privacy is enough, only when covariates are
  # potentially known privacy might become an issue
  # With min samples > 0, the program might halt depending on the given data
  smpc: True
  min_samples: 0 # In case a covariat is known to an attacker, this ensures that
                 # in each calculation, the vector of all samples of one protein
                 # and of one covariat contain at least min_samples non Zero 
                 # non NaN samples, so that results are fuzzy enough for
                 # attackers to not quess samples. 
                 # format: int




