# Reproduce the Federated Limma Remove Batch Effect (fedRBE) Preprint

This guide provides step-by-step instructions to reproduce the analyses and results from the [fedRBE preprint](https://arxiv.org/abs/2412.05894). It leverages the utility scripts and data provided in this repository to demonstrate both centralized and federated batch effect correction using limma and fedRBE, respectively.

## Table of Contents

- [Reproduce the Federated Limma Remove Batch Effect (fedRBE) Preprint](#reproduce-the-federated-limma-remove-batch-effect-fedrbe-preprint)
  - [Table of Contents](#table-of-contents)
  - [Prerequisites and setup](#prerequisites-and-setup)
  - [Repository Structure](#repository-structure)
  - [Running the Analysis](#running-the-analysis)
    - [1. Running a Sample Federated Experiment](#1-running-a-sample-federated-experiment)
    - [2. Obtaining Federated Corrected Data](#2-obtaining-federated-corrected-data)
    - [3. Obtaining Centrally Corrected Data](#3-obtaining-centrally-corrected-data)
    - [4. Comparing Federated and Central Corrections](#4-comparing-federated-and-central-corrections)
  - [Utility Scripts Overview](#utility-scripts-overview)
  - [Troubleshooting](#troubleshooting)
  - [Additional Resources](#additional-resources)
  - [Contact Information](#contact-information)

---

## Prerequisites and setup

Before you begin, ensure you have the following installed and configured:

1. **Docker**: Essential for containerizing applications. [Install Docker](https://www.docker.com/get-started).
2. **Git**: For cloning the repository.
3. **Python 3.8+**.
4. **FeatureCloud CLI**: Get and conhfigure the FeatureCloud CLI using our [installation guide](./batchcorrection/README.md#prerequisites).
5. **R**: Necessary for running R scripts (`filtering.R` and `plots_eda.R`). [Install R](https://www.r-project.org/).
6. Install the required packages. Run the following commands:

   ```bash
   pip install -r requirements.txt
   ```

   ```R
   install.packages(c("limma", "ggplot2", "pheatmap", "UpSetR"))
   ```

## Repository Structure

Understanding the repository layout helps in navigating the files and scripts.

```
fedRBE/
├── README.md                                   # General repository overview
├── batchcorrection/                            # fedRBe FeatureCloud app
├── evaluation_data/                            # Data used for evaluation
│   ├── microarray/                             # Microarray datasets
        ├── before/                             # Uncorrected data with structure needed to run the app
        ├── after/                              # Corrected data
│   │   └── 01_Preprocessing_and_RBE.ipynb      # Data preparation notebook with centralized removeBatchEffect run
│   ├── microbiome_v2/                          # Microbiome datasets with similar structure as microarray
│   ├── proteomics/                             # Proteomics datasets
│   ├── proteomics_multibatch/                  # Multi-batch proteomics datasets (several ba)
│   └── simulated/                              # Simulated datasets
├── evaluation_utils/                           # Utility scripts for evaluations
│       ├── analyse_fedvscentral.py
│       ├── debugging_analyse_experiments.py
│       ├── evaluation_funcs.R
│       ├── featurecloud_api_extension.py
│       ├── fedRBE_simulation_scrip_simdata.py
│       ├── filtering.R
│       ├── get_federated_corrected_data.py
│       ├── plots_eda.R
│       ├── run_sample_experiment.py
│       ├── simulation_func.R
│       ├── upset_plot.py
│       └── utils_analyse.py
├── evaluation/                                 # Main evaluation scripts
│   ├── eval_simulation/                        # Evaluations on simulated data
│   ├── evaluation_microarray.ipynb
│   ├── evaluation_microbiome.ipynb
│   ├── evaluation_proteomics.ipynb
│   ├── evaluation_simulated_3runs.ipynb
│   └── evaluation_simulated.ipynb
└── [other directories/files]
```

## Running the Analysis

This section guides you through running both federated and centralized batch effect corrections and comparing their results.

### 1. Running a Sample Federated Experiment

To simulate a federated workflow on a single machine using provided sample data:

```bash
python3 ./evaluation_utils/run_sample_experiment.py
```

What This Does:
- Simulates multiple clients locally.
- Each client performs batch effect correction using fedRBE.
- Results are aggregated securely without sharing raw data.


### 2. Obtaining Federated Corrected Data

Use the provided utility script to perform federated batch effect correction on your datasets.

```bash
python3 ./evaluation_utils/get_federated_corrected_data.py
```

Steps Performed by the Script:

1. Sets Up Multiple Clients: Simulates clients based on the datasets in `evaluation_data/[dataset]/before/`.
2. Runs fedRBE on Each Client: Applies federated batch effect correction using FeatureCloud testing environment.
3. Aggregates Results: Combines corrected data securely.

Output:

- Corrected Data: Saved in `evaluation_data/after/federated/`.
- Report Files: Detailed logs and correction reports.

Customization:

- To extend to more datasets, add additional [datasets] in `evaluation_data/[dataset]/before/` following the existing structure.

### 3. Obtaining Centrally Corrected Data

Perform centralized batch effect correction using limma's `removeBatchEffect` for comparison.

1. **Navigate to the R Scripts Directory**:

   ```bash
   cd evaluation_utils
   ```

2. **Run the Centralized Correction Script**:

   ```bash
   Rscript filtering.R
   ```

Output:

- Corrected Data: Saved in `evaluation_data/after/`.

### 4. Comparing Federated and Central Corrections

Use the provided script to analyze and compare the results of federated and centralized batch effect corrections.

```bash
python3 ./evaluation_utils/analyse_fedvscentral.py
```

What This Does:

- Loads Corrected Datasets: Imports both federated and centralized corrected data.
- Performs Comparisons:
  - Checks for consistency in indices and columns.
  - Calculates mean and maximum element-wise differences.
- Generates Analysis Reports: Highlights the similarities and differences between the two correction methods.

Output:

- Comparison Metrics: Printed in the console and saved as `comparison_report.txt` in the `evaluation_data/[dataset]/after/` directory.
- Visualizations: Generated plots showcasing the comparison results.

## Utility Scripts Overview

This repository includes several utility scripts to facilitate data processing, analysis, and visualization placed in `evaluation_utils/`.

- `get_federated_corrected_data.py`: Automates the federated batch effect correction process using fedRBE.

    Functionality:

    - Initializes a federated project with multiple clients.
    - Assigns datasets to each client.
    - Executes the fedRBE app to perform batch effect correction.
    - Collects and stores the corrected data.

- `analyse_fedvscentral.py`: Compares the results of federated and centralized batch effect corrections.

    Functionality:

    - Loads federated and centralized corrected datasets.
    - Validates consistency in data structure.
    - Computes statistical differences (mean, max) between the two methods.
    - Generates reports to illustrate findings.

- `featurecloud_api_extension.py`: Extends FeatureCloud API functionalities to support custom workflows and simulations. 

    Functionality:

    - Provides additional API endpoints for managing federated experiments.
    - Supports custom simulations and workflows.
    - This script is primarily used by other utility scripts to manage federated experiments programmatically.

- `filtering.R`: Includes neccesary filters for data preprocessing before centralized batch effect correction using limma's `removeBatchEffect`.
    
    Functionality:

    - Filters and preprocesses data to prepare for batch effect correction.
    - Functions include data normalization and filtering.
    - Functions from this script is used by other utility scripts to preprocess data before running centralized batch effect correction.
        

- `plots_eda.R`: Includes neccesary functions to generates plots to visualize data distributions and corrections.

    Functionality:

    - Creates plots such as boxplots, PCA plots, UMAPs, and heatmaps.
    - Visualizes the impact of batch effect corrections.
    - Functions from this script is used by other utility scripts to generate exploratory data analysis (EDA) plots for evaluation purposes.

- `upset_plot.py`: Generates UpSet plots to visualize intersections and overlaps in datasets or features.

    Functionality:

    - Creates UpSet plots to compare feature overlaps.
    - Functions from this script is used by other utility scripts to visualize feature overlaps in datasets.


## Troubleshooting

Encountering issues? Below are common problems and their solutions:

- **Docker Build Failures**:
  - **Solution**: Ensure Docker is installed and running.
  
- **FeatureCloud Controller Not Starting**:
  - **Solution**: Verify that no other services are occupying the required ports. Check logs for error messages. Check if you have logged in to FeatureCloud.ai.
  
- **Script Execution Errors**:
  - **Solution**: Ensure all prerequisites are installed. Check file paths and permissions.
  
- **Missing Data Files**:
  - **Solution**: Confirm that all required data files are present in the `evaluation_data/[dataset]/before/` directory.
  
- **Inconsistent Results**:
  - **Solution**: Ensure that the same configuration parameters and filtering rules are used for both federated and centralized corrections.

For unresolved issues, consider reaching out via the [GitHub Issues](https://github.com/Freddsle/removeBatch/issues) page.

## Additional Resources

- **FeatureCloud Documentation**: [https://featurecloud.ai/assets/developer_documentation/index.html](https://featurecloud.ai/assets/developer_documentation/index.html)
- **limma Package Documentation**: [https://bioconductor.org/packages/release/bioc/html/limma.html](https://bioconductor.org/packages/release/bioc/html/limma.html)
- **ArXiv Preprint**: [https://arxiv.org/abs/2412.05894](https://arxiv.org/abs/2412.05894)
- **GitHub Repository**: [https://github.com/Freddsle/removeBatch](https://github.com/Freddsle/removeBatch)
- **Federated Learning Overview**: [https://en.wikipedia.org/wiki/Federated_learning](https://en.wikipedia.org/wiki/Federated_learning)

## Contact Information

For questions, issues, or support, please:

- **Open an Issue**: [GitHub Issues](https://github.com/Freddsle/removeBatch/issues)