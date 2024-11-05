# Purpose of this repository
This repository serves two purposes
1. Contains a federated version of the limma batch effect removal algorithm,
hereby called fedRBE, in the subfolder `batchcorrection`. 
The implementation uses the [FeatureCloud platform](http://dx.doi.org/10.2196/42621)
1. Contains the code to reproduce the analyses of fedRBE done in 
this [paper](TODO:link).

# The fedRBE FeatureCloud app
As the app is also published in the [FeatureCloud store](https://featurecloud.ai/app-store), it can
be accessed from there directly. Otherwise, it can be built locally by 
the following commands:
```
cd batchcorrection
docker build . -t <tag-of-the-app>
```
The app can then be used using the [Featurecloud testbed](https://featurecloud.ai/development/test), 
please read the [FeatureCloud documentation](https://featurecloud.ai/assets/developer_documentation/index.html) 
for more information.

# Reproduction of the paper/example usage

## Getting the centrally (limma::RemoveBatchEffect()) corrected data
#TODO

## Getting the federated (fedRBE) corrected data
We provide a utility script to reproduce the results given in the [paper](TODO:link)
The centrally normalized and batch corrected data is provided directly, if a
full reproduction is wanted, please ensure that the data provided to the
federated batch efffect correction in the folders in the `before` folder of
each data set has underwent the same normalization than in central.

- [get_federated_corrected_data.py](evaluation_utils/get_federated_corrected_data.py)
This script runs the federated batch effect removal on the data provided in
`evaluation_data`. Can be extended to use more data, see the description in the
file. Uses the [featurecloud_api_extension.py](evaluation_utils/featurecloud_api_extension.py)
script. If extended, the [analyse_fedvscentral.py](evaluation_utils/analyse_fedvscentral.py)
script should also be extended to also analyse results. Before running this script,
please build the batchcorrection app via
```
cd batchcorrection
docker build . -t bcorrect
```

## Comparing limma::RemoveBatchEffect() to fedRBE
This can be done using the provided script:
- [analyse_fedvscentral.py](evaluation_utils/analyse_fedvscentral.py):
This script checks the differences between federated batch effect corrected data
and centralized batch effect corrected data provided in the repo. 
Tests to see that the results have the same index, columns and gives the 
mean and maximum difference element wise. 
Uses the results produced by [get_federated_corrected_data.py](evaluation_utils/get_federated_corrected_data.py)
Please note that some of the data exceeds GitHubs file size limit, so they
are provided in zip format. If this script throws a FileNotFound error, you
probably just need to unzip the corresponding files/generate them using.

# Additional utils
A few other files can provide additional use cases:
1. [featurecloud_api_extension.py](evaluation_utils/featurecloud_api_extension.py)
This script can be used to to run a featurecloud app with multiple folders simulating
multiple clients. See the file for more information. This is used by the `get_federated_corrected_data.py` script.
[get_federated_corrected_data.py](evaluation_utils/get_federated_corrected_data.py)
1. [filtering.R](evaluation_utils/filtering.R)
TODO: description
1. [plots_eda.R](evaluation_utils/plots_eda.R)
TODO: description
1. [upset_plot.R](evaluation_utils/upset_plot.py)
TODO: description

# Implementation of federated limma RBE
The documentation for the implementation can be found in the corresponding 
subfolders README at [batchcorrection/README.md](batchcorrection/README.md)
and in the [paper](TODO: ref the paper) 

