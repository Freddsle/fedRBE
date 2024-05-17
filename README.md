# Federated Limma Batch Effect Removal
This repository serves two purposes
1. Contains a federated version of limma batch effect removal in the subfolder 
`batchcorrection`. The implementation uses the 
[FeatureCloud platform](http://dx.doi.org/10.2196/42621)
1. Contains the analyses done in [TODO: reference paper] of the federated version
of limma batch effect removal

# Analysis of federated limma remove batch effect(limma RBE)
## evaluation_utils
The scripts in this folder generally serve in getting batch corrected data and 
analysing this data.
1. [get_federated_corrected_data.py](evaluation_utils/get_federated_corrected_data.py)
This script runs the federated batch effect removal on the data provided in
`evaluation_data`. Can be extended to use more data, see the description in th
file. Uses the [featurecloud_api_extension.py](evaluation_utils/featurecloud_api_extension.py)
script. If extended, the [analyse_fedvscentral.py](evaluation_utils/analyse_fedvscentral.py)
script should also be extended to also analyse results.
1. [featurecloud_api_extension.py](evaluation_utils/featurecloud_api_extension.py)
This script can be used to to run a featurecloud app with multiple folders simulating
multiple clients. See the file for more information.
1. [analyse_fedvscentral.py](evaluation_utils/analyse_fedvscentral.py):
This script checks the differences between federated batch effect corrected data
and centralized batch effect corrected data. Tests to see that the results
have the same index, columns and gives the mean and maximum difference element 
wise. Uses the results produced by [get_federated_corrected_data.py](evaluation_utils/get_federated_corrected_data.py)
Please note that some of the data exceeds GitHubs file size limit, so they
are provided in zip format. If this script throws a FileNotFound error, you
probably just need to unzip the corresponding files/generate them using
[get_federated_corrected_data.py](evaluation_utils/get_federated_corrected_data.py)
1. [filtering.R](evaluation_utils/filtering.R)
TODO: description
1. [plots_eda.R](evaluation_utils/plots_eda.R)
TODO: description
1. [upset_plot.R](evaluation_utils/upset_plot.py)
TODO: description

TODO: describe the evaluation folder?
TODO: describe the federated simulation folder?

# Implementation of federated limma RBE
The documentation for the implementation can be found in the corresponding 
subfolders README at [batchcorrection/README.md](batchcorrection/README.md)
and in the [paper](TODO: ref the paper) 
