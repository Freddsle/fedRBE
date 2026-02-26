
<table>
  <tr>
    <td><a href="https://freddsle.github.io/fedRBE/docs/how_to_guide.html"><img src="https://img.shields.io/badge/HowTo_Guide-Click_Here!-007EC6?style=for-the-badge" alt="HowTo Guide"></a></td>
    <td><a href="https://freddsle.github.io/fedRBE/"><img src="https://img.shields.io/badge/Documentation-Click_Here!-007EC6?style=for-the-badge" alt="Documentation"></a></td>
    <td><a href="https://github.com/Freddsle/fedRBE/"><img src="https://img.shields.io/badge/GitHub-Click_Here!-007EC6?style=for-the-badge" alt="GitHub"></a></td>
    <td><a href="https://featurecloud.ai/app/fedrbe"><img src="https://img.shields.io/badge/FeatureCloud_App-Click_Here!-007EC6?style=for-the-badge" alt="FeatureCloud App"></a></td>
  </tr>
</table>

# Reproduce the fedRBE Preprint <!-- omit in toc -->

This guide provides step-by-step instructions to reproduce the analyses and results from the [fedRBE preprint](https://arxiv.org/abs/2412.05894). It leverages the utility scripts and data provided in this repository to demonstrate both centralized and federated batch effect correction using limma and fedRBE, respectively.

## Table of Contents <!-- omit in toc -->

- [Prerequisites and setup](#prerequisites-and-setup)
  - [Install required packages](#install-required-packages)
    - [Option 1: Using mamba with `environment.yml`](#option-1-using-mamba-with-environmentyml)
    - [Option 2: Using `pip` and `R` requirements files](#option-2-using-pip-and-r-requirements-files)
- [Repository structure](#repository-structure)
- [Running the analysis](#running-the-analysis)
  - [1. Running a sample federated experiment](#1-running-a-sample-federated-experiment)
  - [2. Obtaining federated corrected data](#2-obtaining-federated-corrected-data)
  - [3. Obtaining centrally corrected data](#3-obtaining-centrally-corrected-data)
  - [4. Comparing federated and central corrections](#4-comparing-federated-and-central-corrections)
  - [5. Produce tables and figures](#5-produce-tables-and-figures)
- [Utility scripts overview](#utility-scripts-overview)
- [Troubleshooting](#troubleshooting)
- [Additional resources](#additional-resources)
- [Contact information](#contact-information)

---

## Prerequisites and setup

Before you begin, ensure you have the following installed and configured:

1. **Docker**: Essential for containerizing applications. [Install Docker](https://www.docker.com/get-started).
2. **Git**: For cloning the repository. [Install Git](https://git-scm.com/downloads).
3. **Git LFS**: This repository uses [Git Large File Storage](https://git-lfs.com/) for large data files. Install it before cloning to ensure all data files are retrieved correctly.
4. **Python 3.8+**: Required for running Python scripts.
5. **R** : Necessary for running R scripts. [Install R](https://www.r-project.org/).

### Install required packages

You can install the necessary packages using **Conda**/**Mamba** or manually via `pip` and `R` requirements files.

We suggest using Conda/Mamba for a consistent environment setup.

#### Option 1: Using mamba with `environment.yml`

This is the recommended method as it sets up both Python and R dependencies in a single environment.

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/Freddsle/fedRBE
   cd fedRBE
   ```

2. **Create and Activate the Mamba/Conda Environment:**

   ```bash
   mamba env create -f environment.yml
   mamba activate fedRBE
   ```

#### Option 2: Using `pip` and `R` requirements files

If you prefer not to use Mamba, you can install Python and R dependencies separately.

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/Freddsle/fedRBE
   cd fedRBE
   ```

2. **Set Up Python Environment:**

   - **Create a Virtual Environment:**

     ```bash
     python3 -m venv fedrbe_env
     source fedrbe_env/bin/activate  
      # On Windows: fedrbe_env\Scripts\activate
      # fedrbe_env/bin also contains other activate scripts e.g. for fish users
     ```

   - **Upgrade `pip`:**

     ```bash
     pip install --upgrade pip
     ```

   - **Install Python Dependencies:**

     ```bash
     pip install -r requirements.txt
     ```

3. **Set Up R Environment:**

   - **Install R Packages:**

     Open R and run the following commands:

     ```R
     install.packages("remotes")  # If not already installed
     remotes::install_deps("requirements_r.txt", repos = "http://cran.rstudio.com/")
     ```

     Alternatively, you can use the `requirements_r.txt` with a script:

     ```bash
     Rscript install_packages.R
     ```

     Where `install_packages.R` contains:

     ```R
     packages <- readLines("requirements_r.txt")
     install.packages(packages, repos = "http://cran.rstudio.com/")
     ```

     > **Note:** Ensure you have an active internet connection for installing R packages.

---

## Repository structure

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
├── analyse_fedvscentral.py                     # Compares federated and centralized batch effect corrections.
├── generate_fedrbe_corrected_datasets.py       # A script performing fedRBE on all datasets and save the results.
├── run_sample_experiment.py                    # A script performing fedRBE on one dataset only
├── evaluation_utils/                           # Utility scripts for evaluations
│       ├── evaluation_funcs.R
│       ├── featurecloud_api_extension.py
│       ├── fedRBE_simulation_scrip_simdata.py
│       ├── filtering.R
│       ├── plots_eda.R
│       ├── simulation_func.R
│       ├── upset_plot.py
│       └── utils_analyse.py
├── evaluation/                                 # Main evaluation scripts to produce results and figures
│   ├── eval_simulation/                        # Evaluations on simulated data
│   ├── evaluation_microarray.ipynb             # Evaluation of microarray datasets
│   ├── evaluation_microbiome.ipynb
│   ├── evaluation_proteomics.ipynb
└── [other directories/files]
```

## Running the analysis

This section guides you through running both federated and centralized batch effect corrections and comparing their results.

### 1. Running a sample federated experiment

To simulate a federated workflow on a single machine using provided sample data:

```bash
python3 ./run_sample_experiment.py
```

What this does:
- Simulates multiple clients locally.
- Each client performs batch effect correction using fedRBE.
- Results are aggregated securely without sharing raw data.


### 2. Obtaining federated corrected data

Use the provided utility script to perform federated batch effect correction on your datasets.

```bash
python3 ./generate_fedrbe_corrected_datasets.py
```

Steps Performed by the Script:

1. Sets up multiple clients: Simulates clients based on the datasets in `evaluation_data/[dataset]/before/`.
2. Runs fedRBE on each client: Applies federated batch effect correction using FeatureCloud testing environment.
3. Aggregates results: Combines corrected data securely.

Output:

- Corrected data: saved in `evaluation_data/[dataset]/after/individual_results/`. Furthermore stores the merged corrected data of all clients directly in `evaluation_data/[dataset]/after/` as `FedApp_corrected_data.tsv` or `FedApp_corrected_data.tsv` if SMPC was used.
- Report files: In `evaluation_data/[dataset]/after/individual_results/`, detailed logs and correction reports can be found.

_Note: The script may take some time to complete, depending on the dataset size and the number of clients. It usually takes a few hours upto a day.
_Note 2: To process this dataset one need >16GB RAM. To skip the correction on microarray datasets, comment the corresponding lines in the script (generate_fedrbe_corrected_datasets.py, search for `experiments.append(microarray_experiment)` to find the relevant 4 lines of code)._
_Note 3: This was already performed and the fedRBE corrected data is stored in the repository.
You can skip this if you just want to look at the results._

Customization:
- If you want to run the correction not on all datasets, comment the corresponding lines in the script (you can just search for the comment `## ADD EXPERIMENTS, CHANGE HERE TO INCLUDE/EXCLUDE EXPERIMENTS` in `generate_fedrbe_corrected_datasets.py`).
- To extend to more datasets, add additional [datasets] in `evaluation_data/[dataset]/before/` following the existing structure.

### 3. Obtaining centrally corrected data

Perform centralized batch effect correction using limma's `removeBatchEffect` for comparison.

1. Navigate to the dataset directory:

   ```bash
   cd evaluation_data/[dataset_name]
   ```

2. Run the data preprocessing and centralized correction script inside ipynb. 

  The code is located in `*central_RBE.ipynb` Jupyter notebooks in the `evaluation_data/[dataset]/` directory.

  Output:

  - Corrected data: saved in `evaluation_data/[dataset]/after/` for each dataset.

_Note: The preprocessing steps and centralized correction are already implemented in the provided notebooks. It is possible to skip this step completely and use the provided corrected data._

### 4. Comparing federated and central corrections

Use the provided script to analyze and compare the results of federated and centralized batch effect corrections.

```bash
python3 ./analyse_fedvscentral.py
```

What This Does:

- Loads Corrected Datasets: Imports both federated and centralized corrected data.
- Performs Comparisons:
  - Checks for consistency in indices and columns.
  - Calculates mean and maximum element-wise differences.
- Generates Analysis Reports: Highlights the similarities and differences between the two correction methods.

Output:

- Comparison Metrics: Printed in the console and saved as `fed_vc_cent_results.tsv` in the `evaluation_data/ directory.

### 5. Produce tables and figures

To reproduce the tables and figures from the preprint, run the provided Jupyter notebooks in the `evaluation/` directory.

## Utility scripts overview

This repository includes several utility scripts to facilitate data processing, analysis, and visualization. The main script `generate_fedrbe_corrected_datasets.py` is located at the repository root; the remaining helpers are placed in `evaluation_utils/`.

- `generate_fedrbe_corrected_datasets.py` (repo root): Automates the federated batch effect correction process using fedRBE.

    Functionality:

    - Initializes a federated project with multiple clients.
    - Assigns datasets to each client.
    - Executes the fedRBE app to perform batch effect correction.
    - Collects and stores the corrected data.
    - This is done for all datasets.

- `analyse_fedvscentral.py` (repo root): Compares the results of federated and centralized batch effect corrections.

    Functionality:

    - Loads federated and centralized corrected datasets.
    - Validates consistency in data structure.
    - Computes statistical differences (mean, max) between the two methods.
    - Generates reports to illustrate findings.

- `featurecloud_api_extension.py`: Extends FeatureCloud API functionalities to support custom workflows and simulations. 

    Functionality:
    
    - Helper script used by `generate_fedrbe_corrected_datasets.py`.
    - Forms an API wrapper around the FeatureCloud testbed, allowing to run simulated federated learning
    with utility features such as automatic restarts and automatic result extraction.

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
  
- **Errors with Test runs**: 
  - **Solution**: Ensure the is no leftover running Docker containers. Restart Docker / System if necessary. 

- **Script Execution Errors**:
  - **Solution**: Ensure all prerequisites are installed. Check file paths and permissions.
  
- **Missing Data Files**:
  - **Solution**: Confirm that all required data files are present in the `evaluation_data/[dataset]/before/` directory.
  
- **Inconsistent Results**:
  - **Solution**: Ensure that the same configuration parameters and filtering rules are used for both federated and centralized corrections.

For unresolved issues, consider reaching out via the [GitHub Issues](https://github.com/Freddsle/fedRBE/issues) page.

## Additional resources

- **FeatureCloud Documentation**: [https://featurecloud.ai/assets/developer_documentation/index.html](https://featurecloud.ai/assets/developer_documentation/index.html)
- **limma Package Documentation**: [https://bioconductor.org/packages/release/bioc/html/limma.html](https://bioconductor.org/packages/release/bioc/html/limma.html)
- **ArXiv Preprint**: [https://arxiv.org/abs/2412.05894](https://arxiv.org/abs/2412.05894)
- **GitHub Repository**: [https://github.com/Freddsle/fedRBE](https://github.com/Freddsle/fedRBE)
- **Federated Learning Overview**: [https://en.wikipedia.org/wiki/Federated_learning](https://en.wikipedia.org/wiki/Federated_learning)

## Contact information

For questions, issues, or support, please:

- **Open an Issue**: [GitHub Issues](https://github.com/Freddsle/fedRBE/issues)
