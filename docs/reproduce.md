
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
- [Repository structure](#repository-structure)
- [Running the analysis](#running-the-analysis)
  - [1. Obtaining federated corrected data](#1-obtaining-federated-corrected-data)
  - [2. Obtaining centrally corrected data](#2-obtaining-centrally-corrected-data)
  - [3. Comparing federated and central corrections](#3-comparing-federated-and-central-corrections)
  - [4. Produce tables and figures](#4-produce-tables-and-figures)
  - [5. Reproduce the classification analysis](#5-reproduce-the-classification-analysis-comparing-fedrbe-corrected-to-non-corrected-data)
- [Utility scripts overview](#utility-scripts-overview)
- [Troubleshooting](#troubleshooting)
- [Additional resources](#additional-resources)
- [Contact information](#contact-information)

---

## Prerequisites and setup
1. **Docker**: [Installation Instructions](https://www.docker.com/get-started)
1. **Git LFS** (required for large files used in this workflow)
1. **Python 3.8+**: [Installation Instructions](https://www.python.org/) with dependencies from `requirements.txt`.
1. **R 4.0+** with dependencies from `requirements_r.txt`
1. **System resources**:
  - ≥ 16 GB RAM (The script uses very close to 16 GB, so be careful with other programs running!)
  - ≥ 20 GB free disk space

### Setup steps

1. **Set up Git LFS:**

   Install git lfs following the [git lfs documentation](https://git-lfs.com/).
   
   Make sure you setup git lfs:
   ```bash
   git lfs install
   ```
   
   If you already cloned the [repository](https://github.com/Freddsle/fedRBE), you need to now
   download the large files from the large file storage:
   ```bash
   git lfs pull 
   ```

   If you have not yet cloned the repository, `git clone` will automatically download LFS files if `git lfs install` has been run before.

2. **Clone the repository:**

   If you didn't clone the repository yet please do so:
   ```bash
   git clone https://github.com/Freddsle/fedRBE.git
   cd fedRBE
   ```

3. **Set up Python environment:**

   We recommend using a virtual environment:
   ```bash
   python3 -m venv fedrbe_env
   source fedrbe_env/bin/activate  # on Windows: fedrbe_env\Scripts\activate
   pip install -r requirements.txt
   ```

4. **Set up R environment:**

   `limma` and `variancepartition` are Bioconductor packages and must be installed separately from the CRAN packages:

   ```bash
   # Install CRAN packages
   Rscript -e 'pkgs <- readLines("requirements_r.txt"); pkgs <- pkgs[!grepl("^#", pkgs) & nzchar(trimws(pkgs))]; install.packages(pkgs[!pkgs %in% c("limma", "variancepartition")], repos = "http://cran.rstudio.com/")'
   # Install Bioconductor packages
   Rscript -e 'if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager"); BiocManager::install(c("limma", "variancepartition"))'
   ```

---

## Running the analysis

This section guides you through running both federated and centralized batch effect corrections and comparing their results.

### 1. Obtaining federated corrected data

Use the provided utility script to perform federated batch effect correction on your datasets.

```bash
python3 ./generate_fedrbe_corrected_datasets.py
```

Steps Performed by the Script:

1. Sets up multiple clients: Simulates clients based on the datasets in `evaluation_data/[dataset]/before/`.
2. Runs fedRBE on each client: Applies federated batch effect correction using FeatureCloud testing environment.
3. Aggregates results: Combines corrected data.

Output:

- Corrected data: saved in `evaluation_data/[dataset]/after/individual_results/`. Furthermore stores the merged corrected data of all clients directly in `evaluation_data/[dataset]/after/` as `FedApp_corrected_data.tsv` or `FedApp_corrected_data_smpc.tsv` if SMPC was used.
- Report files: In `evaluation_data/[dataset]/after/individual_results/`, detailed logs and correction reports can be found.

_Note: The script may take some time to complete, depending on the dataset size and the number of clients. It usually takes a few hours upto a day.
_Note 2: To process this dataset one need >16GB RAM. To skip the correction on microarray datasets, comment the corresponding lines in the script (generate_fedrbe_corrected_datasets.py, search for `experiments.append(microarray_experiment)` to find the relevant 4 lines of code)._
_Note 3: This was already performed and the fedRBE corrected data is stored in the repository.
You can skip this if you just want to look at the results._

Customization:
- If you want to run the correction not on all datasets, comment the corresponding lines in the script (you can just search for the comment `## ADD EXPERIMENTS, CHANGE HERE TO INCLUDE/EXCLUDE EXPERIMENTS` in `generate_fedrbe_corrected_datasets.py`).
- To extend to more datasets, add additional [datasets] in `evaluation_data/[dataset]/before/` following the existing structure.

### 2. Obtaining centrally corrected data

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

### 3. Comparing federated and central corrections

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

### 4. Produce tables and figures

To reproduce the tables and figures from the preprint, run the provided Jupyter notebooks in the `evaluation/` directory.

### 5. Reproduce the classification analysis comparing fedRBE corrected to non corrected data

This is split into three python scripts. Make sure the required python packages from `requirements.txt` are installed!

First run the classification experiments. This takes multiple hours, so the repo already contains the relevant results if you want to skip this. 

The classification experiments are split into the two different experiment types:
1. **train_test_split**: Each client reserves 20% of the data as test data, trains on the other 80% and reports metrics when predicting on the test data. This takes upto an hour. To run it, simply run
```bash
python3 evaluation_classification_after_correction/run_classification_train_test_split.py
```
1. **leave_one_cohort_out**: The classification model is trained on all except one client. Then the model is used to predict on all of the data of the left out client and the client reports the metrics. Therefore, *n_clients* models are trained. For this reason, this takes multiple hours. To run it, simply run:
```bash
python3 evaluation_classification_after_correction/run_classification_leave_one_cohort_out.py
```

The experiment results are saved in `evaluation_classification_after_correction/results`.

To finally visualize the experiments with plot, simply run the corresponding analyze script:
```bash
python3 evaluation_classification_after_correction/analyse_classification_metric_report.py
```
The resulting plots can be found in `evaluation_classification_after_correction/plots`.

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
│   ├── microbiome/                             # Microbiome datasets with similar structure as microarray
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

---


## Utility scripts overview

This repository includes several utility scripts to facilitate data processing, analysis, and visualization. All main scripts are located at the repository root; the remaining helpers are placed in `evaluation_utils/`.

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

- `filtering.R`: Includes necessary filters for data preprocessing before centralized batch effect correction using limma's `removeBatchEffect`.
    
    Functionality:

    - Filters and preprocesses data to prepare for batch effect correction.
    - Functions include data normalization and filtering.
    - Functions from this script is used by other utility scripts to preprocess data before running centralized batch effect correction.
        

- `plots_eda.R`: Includes necessary functions to generates plots to visualize data distributions and corrections.

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
