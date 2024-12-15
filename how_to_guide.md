# Getting Started with Federated Limma Remove Batch Effect (fedRBE)

This guide aims to help beginners quickly set up and run the federated limma remove batch effect (fedRBE) app. 
For more detailed technical information, advanced configuration options, and specific implementation details, 
please refer to the [main README file](./batchcorrection/README.md).

## Introduction
`fedRBE` allows you to remove batch effects from data in a federated manner, 
meaning multiple parties can collaborate on data analysis without sharing raw data.

*What is a batch effect?*  
Batch effects occur when technical differences between datasets (e.g. different labs, conditions, 
or times of data collection) introduce bias. `fedRBE` helps correct these biases so that 
comparisons between datasets are more meaningful.

For a more formal description and details, see the [Description section of the README](./batchcorrection/README.md#description).

## Requirements and Setup
Before running `fedRBE`, ensure you meet the basic prerequisites of the FeatureCloud platform:
1. **Docker Installed**  
2. **FeatureCloud CLI Installed**:  
   ```bash
   pip install featurecloud
   featurecloud controller start
   ```
   
For more details, including how to build or pull the `fedRBE` image, 
please see the [Prerequisites in the README](./batchcorrection/README.md#prerequisites).

## Running a Quick Test Locally
If you want to try `fedRBE` locally before deploying in a fully federated setting, 
a sample dataset and a test script are provided. To run it:
```bash
git clone git@github.com:Freddsle/removeBatch.git
cd removeBatch/evaluation_utils
python3 run_sample_experiment.py
```
This demonstration shows how the app works on sample data. For more context and troubleshooting, 
refer to the [Running the provided sample data section in the README](./batchcorrection/README.md#running-the-provided-sample-data).

## Simulation of a Federated Workflow (All on One Machine)
You can simulate a federated environment on a single machine using the FeatureCloud testbed.  
For instructions, see the [Federated workflow simulation](./batchcorrection/README.md#simulation-of-a-federated-workflow) in the README.

## Running a Real Federated Workflow
To run `fedRBE` in a real federated scenario:
1. **Project Creation & Tokens**: Have a coordinator create a project and invite participants via FeatureCloud.
2. **Participants Join**: Each participant joins the project using their tokens and local data.
3. **Data Placement**: Each participant places their data and `config.yml` in the appropriate folder.
4. **Run the App**: Start the workflow through the FeatureCloud platform.

For more step-by-step details, consult the [Running a federated workflow](./batchcorrection/README.md#running-a-federated-workflow) section of the README.

## Understanding Input Files
You need:
- **Expression Data File (CSV/TSV)**: Contains samples and features.  
- **Design File (Optional)**: Contains covariates.
- **`config.yml` File**: Customizes how `fedRBE` processes data.  
  See the [Config section in the README](./batchcorrection/README.md#config) for a full explanation of available options.

**Important Notes for Beginners**:
- Make sure sample names match between the expression and design files.
- Pay attention to whether your data is in `samples x features` or `features x samples` format. See `config.yml` and the `expression_file_flag` parameter.

## Normalization and Batch Correction
`fedRBE` supports normalization methods (e.g. `"log2(x+1)"`). If unsure, start with the defaults.  
Check the [Config section](./batchcorrection/README.md#config) for details on normalization and other parameters.

## Outputs and Results
After running `fedRBE`, you will find:
- `only_batch_corrected_data.csv`: The batch-corrected features.
- `report.txt`: A summary of excluded features, beta values (correction coefficients), and the design matrix used.

For more in-depth explanation, see [Output section in the README](./batchcorrection/README.md#output).

## Additional Resources
- [FeatureCloud Developer Guide](https://featurecloud.ai/developers)
- [Privacy-preserving techniques in FeatureCloud](https://featurecloud.ai/assets/developer_documentation/privacy_preserving_techniques.html)

If you encounter difficulties or need more detailed guidance, return to the 
[main README](./batchcorrection/README.md) for advanced instructions.