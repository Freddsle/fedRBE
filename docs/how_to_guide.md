# Getting Started with Federated Limma Remove Batch Effect (fedRBE)

This guide is designed for beginners who want a quick and easy way to start using `fedRBE` and test its functionality. 
For more technical details and advanced usage and specific implementation details, please refer to the [main README file](https://freddsle.github.io/fedRBE/batchcorrection/).

`fedRBE` allows you to remove batch effects from data in a federated manner, ensuring data privacy.

For a more formal description and details, see the fedRBE's preprint on [ArXiv](https://doi.org/10.48550/arXiv.2412.05894).


## List of Contents

- [Getting Started with Federated Limma Remove Batch Effect (fedRBE)](#getting-started-with-federated-limma-remove-batch-effect-fedrbe)
  - [List of Contents](#list-of-contents)
  - [Key Concepts](#key-concepts)
  - [Minimal Requirements and Setup](#minimal-requirements-and-setup)
  - [Understanding the Workflow](#understanding-the-workflow)
  - [File Preparation](#file-preparation)
  - [Step-by-Step Scenario](#step-by-step-scenario)
  - [Results and Output:](#results-and-output)
  - [Single-Machine Simulations using the Provided Sample Data](#single-machine-simulations-using-the-provided-sample-data)
  - [Troubleshooting Tips](#troubleshooting-tips)
    - [Choosing the Correct Data Orientation](#choosing-the-correct-data-orientation)
    - [Incorporating Covariates](#incorporating-covariates)
    - [Selecting a Reference Batch](#selecting-a-reference-batch)
  - [Glossary \& Further Resources](#glossary--further-resources)


## Key Concepts

- **Federated learning**: Federation allows multiple participants (clients) to collaborate on data analysis without directly sharing their raw data. This ensures privacy while still enabling collaborative results.
  
- **Batch Effects**: When data come from different sources (e.g., different labs, time points, or technologies), they can contain systematic non-biological differences (batch effects) that make direct comparisons misleading. 

## Minimal Requirements and Setup

**Prerequisites** (see [README](https://freddsle.github.io/fedRBE/batchcorrection/#prerequisites) for details):
1. **Docker** installed (check [Docker website](https://www.docker.com/) for installation instructions).
2. **FeatureCloud CLI** installed and running:
   ```bash
   pip install featurecloud
   featurecloud controller start
   ```
3. **App Image** (either build locally or pull):
   ```bash
   # Pull (linux/amd64):
   featurecloud app download featurecloud.ai/bcorrect
   
   # Or build (for non-amd64 architectures - e.g., ARM (Mac M1)):
   docker build . -t featurecloud.ai/bcorrect:latest
   ```

## Understanding the Workflow

Below is a simplified workflow of how to use `fedRBE`:
1. **Coordinator creates a FeatureCloud project and distributes tokens** to at least 3 participants.
2. **Each Participant (Client)** prepares their data and a `config.yml` file.
3. **All Clients join the project** using FeatureCloud and run the app locally.
4. **fedRBE aligns and corrects batch effects** without sharing raw data.
5. **Results are produced locally at each client**, ensuring privacy.

## File Preparation

You need two main inputs:
1. **Expression Data File** (CSV/TSV)
2. **`config.yml`** for custom settings
3. and **Optional Design File** with covariates (if needed).

<p align="center">
   <img src="../figures/how_to1.png" alt="Required files figure" width="70%">
</p>

**Minimal Example Directory Structure**:
```text
client_folder/
├─ config.yml
├─ expression_data.csv
├─ design.csv
```

If you want to simulate a federated workflow on a single machine, you can use the provided sample data and test script. In this case, you need to create at least three folders, each with the sample data and a `config.yml` file (for example, `clientA`, `clientB`, `clientC` folders).

**Example `config.yml` snippet**:
```yaml
flimmaBatchCorrection:
  data_filename: "expression_data_client1.csv"
  expression_file_flag: False # True if data is in samples x features format
  index_col: "GeneIDs"  # Column name to use as index
  covariates: ["Pyr"]   # Covariates column name to include in the design matrix
  separator: ","  # Separator used in the data file
  design_separator: "," # Separator used in the design file
  normalizationMethod: "log2(x+1)"  # Normalization method or log transformation
  smpc: True  # Recommended to set to True
  min_samples: 2  # Minimum number of samples to include a feature
  position: 1   # position of the client (first, second, third, etc.)
  reference_batch: ""  # if True, this client is used as the reference batch
```

For more details on the `config.yml` parameters, see the [main README](https://freddsle.github.io/fedRBE/batchcorrection/#config).

## Step-by-Step Scenario

**Scenario**: Three clients (A, B, and C) collaborate on a federated analysis. Video tutorial: [link](https://featurecloud.ai/researchers).

1. **Coordinator Actions**:  
   - The coordinator logs into the FeatureCloud platform and **creates a new project**.
   - Add the fedRBE app into the workflow and *finalize the project*.
   - The coordinator **creates tokens** and sends them to Clients A, B, and C.

   <p align="center">
   <img src="../figures/how_to2.png" alt="Coordinator step 1" width="70%">
   </p>
   
2. **Client Setup**:
   - **Client A, B, C**: Place `expression_data_client.csv` and `config.yml` in a local folder.
   - Adjust `config.yml` parameters as needed (e.g., change `data_filename` to match the correct file name).
   
3. **Joining the Project**:
   - Each client uses the FeatureCloud to login and join the project using the provided token.
   - After joining, each client uploads their data and config file to the FeatureCloud GUI client as a one .zip file (without any folder structure inside). It will not be sent to the coordinator or other clients, but makes it available for the Docker container with the app.

   <p align="center">
   <img src="../figures/how_to3.png" alt="Required files figure" width="70%">
   </p>
   
4. **Running fedRBE**:
   - After all clients join, the coordinator starts the project.
   - The app runs locally at each client, securely combining results.
   
## Results and Output:

After completion, each client finds:
   - `only_batch_corrected_data.csv`: The batch-corrected expression data.
   - `report.txt`: Details on excluded features, beta values, and the used design matrix.
   - logs: Detailed logs of the process.
   

## Single-Machine Simulations using the Provided Sample Data

If you’d like to test everything on one machine, you can run the provided sample data and test script. This simulates multiple clients locally, so you can see the federated workflow in action without needing multiple machines.

**Steps:**
1. Ensure prerequisites are met (Docker, `featurecloud` package, configured FeatureCloud controller and the app image).
2. Clone the repository:
   ```bash
   git clone git@github.com:Freddsle/fedRBE.git
   ```
3. Run the provided sample experiment:
   ```bash
   python3 ./evaluation_utils/run_sample_experiment.py
   ```
   
This will start a local simulation of multiple clients and show you how the batch correction is applied in practice. More details can be found in the [main README](https://freddsle.github.io/fedRBE/batchcorrection/#running-the-provided-sample-data).


## Troubleshooting Tips

- **Missing Files**: If you see "file not found," ensure that `config.yml` and data files are in the same directory.
- **Incorrect Format**: Check if `expression_file_flag` and `index_col` are set correctly based on your data orientation.
- **No Output Produced**: Review `report.txt` and logs. 

### Choosing the Correct Data Orientation

Depending on how your data is structured, you must correctly set `expression_file_flag` in your `config.yml`:

- **If your file is features (rows) x samples (columns)**:  
  `expression_file_flag: True` and `index_col: <feature_id_column>`

- **If your file is samples (rows) x features (columns)**:  
  `expression_file_flag: False` and `index_col: <sample_id_column>`


### Incorporating Covariates

If you have additional covariates (e.g., age, treatment type) that might influence your data, you can include them either directly in the `design_filename` file or list them in your `config.yml` under `covariates`. If no separate design file is provided, these covariates must exist as features in the main data file.

**Example:**
```yaml
covariates: ["Age", "Treatment"]
```

### Selecting a Reference Batch

`fedRBE` needs a reference batch to align the other batches against. By default, if no `reference_batch` is set, it uses the last client in the positional order defined by the `position` parameter. If all parameters are unset, it may choose a batch at random, resulting in non-deterministic runs.

**Example:**
```yaml
position: 2
reference_batch: ""
```

## Glossary & Further Resources

- **FeatureCloud**: A platform enabling federated analyses. [FeatureCloud docs](https://featurecloud.ai/)
- **limma**: A popular R package for differential expression analysis. `RemoveBatchEffect` is a function from limma.

For more advanced configurations and detailed explanations, see the 
[main README](https://freddsle.github.io/fedRBE/batchcorrection/#config) and the [ArXiv preprint](https://doi.org/10.48550/arXiv.2412.05894).

If you encounter difficulties, please:
- Check the logs for error messages.
- Revisit the [main README](https://freddsle.github.io/fedRBE/batchcorrection/).
- Reach out to the support by creating an issue on the [GitHub repository](https://github.com/Freddsle/fedRBE)