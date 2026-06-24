
<table>
  <tr>
    <td><a href="https://freddsle.github.io/fedRBE/docs/how_to_guide.html"><img src="https://img.shields.io/badge/HowTo_Guide-Click_Here!-007EC6?style=for-the-badge" alt="HowTo Guide"></a></td>
    <td><a href="https://freddsle.github.io/fedRBE/"><img src="https://img.shields.io/badge/Documentation-Click_Here!-007EC6?style=for-the-badge" alt="Documentation"></a></td>
    <td><a href="https://github.com/Freddsle/fedRBE/"><img src="https://img.shields.io/badge/GitHub-Click_Here!-007EC6?style=for-the-badge" alt="GitHub"></a></td>
    <td><a href="https://featurecloud.ai/app/fedrbe"><img src="https://img.shields.io/badge/FeatureCloud_App-Click_Here!-007EC6?style=for-the-badge" alt="FeatureCloud App"></a></td>
  </tr>
</table>

# Reproduce the fedRBE Preprint <!-- omit in toc -->

This guide explains how to reproduce the analyses from the [fedRBE preprint](https://arxiv.org/abs/2412.05894), including centralized limma correction, federated fedRBE correction, and downstream evaluation.

## Table of Contents <!-- omit in toc -->

- [Prerequisites and setup](#prerequisites-and-setup)
  - [Setup steps](#setup-steps)
- [Running the analysis](#running-the-analysis)
  - [1. Obtaining federated corrected data](#1-obtaining-federated-corrected-data)
  - [2. Obtaining centrally corrected data](#2-obtaining-centrally-corrected-data)
  - [3. Comparing federated and central corrections](#3-comparing-federated-and-central-corrections)
  - [4. Produce tables and figures](#4-produce-tables-and-figures)
  - [5. Reproduce the classification analysis comparing fedRBE-corrected to uncorrected data](#5-reproduce-the-classification-analysis-comparing-fedrbe-corrected-to-uncorrected-data)
    - [Prerequisites](#prerequisites)
    - [Reproduction steps](#reproduction-steps)
  - [6. Reproduce the clustering analysis comparing fedRBE-corrected to uncorrected data](#6-reproduce-the-clustering-analysis-comparing-fedrbe-corrected-to-uncorrected-data)
    - [Prerequisites](#prerequisites-1)
    - [Real datasets](#real-datasets)
- [Repository structure](#repository-structure)
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
   
   Initialize Git LFS:
   ```bash
   git lfs install
   ```
   
   If the [repository](https://github.com/Freddsle/fedRBE) is already cloned, pull the large files:
   ```bash
   git lfs pull 
   ```

   If you have not yet cloned the repository, `git clone` will automatically download LFS files if `git lfs install` has been run before.

2. **Clone the repository:**

   If the repository is not cloned yet:
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

   `requirements_r.txt` lists CRAN, Bioconductor, and GitHub dependencies in
   separate sections. Install them with:

   ```bash
   Rscript -e 'install.packages(c("data.table","devtools","ggpubr","gridExtra","ggsci","ggtext","glue","IRkernel","invgamma","jsonlite","knitr","nipals","patchwork","pheatmap","reshape2","scales","tidyverse","umap","viridis","WGCNA"), repos="https://cloud.r-project.org")'
   Rscript -e 'if (!requireNamespace("BiocManager", quietly=TRUE)) install.packages("BiocManager", repos="https://cloud.r-project.org"); BiocManager::install(c("affy","GEOquery","hgu133acdf","hgu133plus2cdf","limma","variancePartition"))'
   Rscript -e 'if (!requireNamespace("devtools", quietly=TRUE)) install.packages("devtools", repos="https://cloud.r-project.org"); devtools::install_github("mwgrassgreen/RobNorm")'
   ```

---

## Running the analysis

This section guides you through running both federated and centralized batch effect corrections and comparing their results.

Recommended order:

1. Set up dependencies and fetch LFS data.
2. Ensure the per-dataset `before/` inputs exist. When regenerating inputs from raw or source data, run the dataset-specific preparation notebooks listed in [2. Obtaining centrally corrected data](#2-obtaining-centrally-corrected-data) before launching the federated correction script.
3. Run the federated correction script, or use the corrected outputs already committed in `evaluation_data/*/after/`.
4. Run or reuse the central correction notebooks.
5. Compare federated vs central corrections, then run the figure/table, classification, and clustering analyses.

### 1. Obtaining federated corrected data

Use the provided utility script to perform the configured federated batch
effect correction experiments. Edit the experiment list in the script to
enable or disable any of the datasets; currently all datasets are enabled by default.

```bash
python3 ./generate_fedrbe_corrected_datasets.py
```

Steps Performed by the Script:

1. Sets up multiple clients: Simulates clients based on the datasets in `evaluation_data/[dataset]/before/`.
2. Runs fedRBE on each client: Applies federated batch effect correction using FeatureCloud testing environment.
3. Aggregates results: Combines corrected data and writes outputs to `evaluation_data/[dataset]/after/` for each dataset.

Output:

- Corrected data: saved in `evaluation_data/[dataset]/after/individual_results/`. The merged client data is also written directly to `evaluation_data/[dataset]/after/` as `FedApp_corrected_data.tsv`, or `FedApp_corrected_data_smpc.tsv` when SMPC is used.
- Report files: In `evaluation_data/[dataset]/after/individual_results/`, detailed logs and correction reports can be found.

Notes:
- Runtime depends on dataset size and client count; the full run can take several hours to a day.
- Ovarian cancer correction can require at least 16 GB RAM. To skip it, comment the `experiments.append(ovarian_cancer_experiment)` block in `generate_fedrbe_corrected_datasets.py`.
- FedRBE-corrected outputs are already stored in the repository, so this step can be skipped when only inspecting results.

Customization:
- To run a subset of datasets, edit the experiment list under `## ADD EXPERIMENTS, CHANGE HERE TO INCLUDE/EXCLUDE EXPERIMENTS` in `generate_fedrbe_corrected_datasets.py`.
- To add datasets, create `evaluation_data/[dataset]/before/` entries following the existing structure.
- Quartet multiomics requires the client folders written by `evaluation_data/quartet_multiomics/02_prepare_RBE_inputs.ipynb`; run that notebook before the federated correction script if those folders were regenerated or removed.

### 2. Obtaining centrally corrected data

Perform centralized batch effect correction using limma's `removeBatchEffect` for comparison.

Run the dataset preparation and central-correction notebooks from their own directories, or set the notebook working directory to the listed folder so relative paths resolve correctly.

| Dataset | Run order |
|---------|-----------|
| Simulated | `evaluation_data/simulated/01_data_prep_and_central_RBE.ipynb`; run `evaluation_data/simulated/00_data_simulation.ipynb` first only to regenerate all simulation runs |
| E. coli | `evaluation_data/ecoli/01_data_prep_and_central_RBE.ipynb` |
| Ovarian cancer | `evaluation_data/ovarian_cancer/00_harmonize_meta_load_data.ipynb`, `evaluation_data/ovarian_cancer/01_check_datasets_intersection.ipynb`, then `evaluation_data/ovarian_cancer/02_central_RBE.ipynb` |
| ccRCC proteomics | `python evaluation_data/ccRCC_studies/prepare_ccRCC_data.py`, then `evaluation_data/ccRCC_studies/01_central_RBE.ipynb` |
| Quartet multiomics | `evaluation_data/quartet_multiomics/02_prepare_RBE_inputs.ipynb`, then `evaluation_data/quartet_multiomics/03_central_RBE.ipynb`; `evaluation_data/quartet_multiomics/01_preprocess_eda.ipynb` is EDA-only and optional |

Output:

- Corrected data: saved in `evaluation_data/[dataset]/after/` for each dataset.
- Dataset-specific details and outputs are documented in the corresponding `evaluation_data/<dataset>/README.md` files.

_Note: The preprocessing and centralized correction notebooks have already been run for the committed outputs. You can skip this step when using the provided corrected data._

For simulated data, committed run-1 inputs support a quick check. The full 30-run evaluation requires generated `before/intermediate/` and `after/runs/` files.

### 3. Comparing federated and central corrections

Use the provided script to analyze and compare the results of federated and centralized batch effect corrections.

```bash
python3 ./analyse_fedvscentral.py
```

What this does:

- Loads corrected datasets: imports both federated and centralized corrected data.
- Performs comparisons:
  - Checks for consistency in indices and columns.
  - Calculates mean and maximum element-wise differences.
- Generates reports: summarizes similarities and differences between the two correction methods.

Output:

- Comparison Metrics: Printed in the console and saved as `fed_vc_cent_results.tsv` in the `evaluation_data/` directory.

### 4. Produce tables and figures

To reproduce the tables and figures from the preprint, run the provided Jupyter notebooks in the `evaluation/` directory:

1. `evaluation/evaluation_simulated.ipynb`
2. `evaluation/evaluation_simulated_30runs.ipynb`
3. `evaluation/evaluation_ecoli.ipynb`
4. `evaluation/evaluation_ovarian_cancer.ipynb`
5. `evaluation/evaluation_ccRCC.ipynb`
6. `evaluation/evaluation_quartet_multiomics.ipynb`

These notebooks expect the corrected data from Steps 1 and 2. `evaluation_simulated_30runs.ipynb` requires the full generated simulated `after/runs/` files. Figures are written under `evaluation/plots/` and related evaluation output folders.

### 5. Reproduce the classification analysis comparing fedRBE-corrected to uncorrected data

#### Prerequisites
- You need to have the Python 3 environment set up with the required dependencies as described in the [Prerequisites and setup](#prerequisites-and-setup) section.

#### Reproduction steps
This part uses three Python scripts. Install the packages from `requirements.txt` first.

First run the classification experiments. This can take several hours; the repository already contains the generated results.

The classification experiments are split into the two different experiment types:
1. **train_test_split**: Each client reserves 20% of the data as test data, trains on the other 80%, and reports test-set metrics. This takes up to an hour.
```bash
python3 evaluation_classification_after_correction/run_classification_train_test_split.py
```
1. **leave_one_cohort_out**: The model is trained on all clients except one, then evaluated on the held-out client. This trains *n_clients* models and can take several hours.
```bash
python3 evaluation_classification_after_correction/run_classification_leave_one_cohort_out.py
```

The experiment results are saved in `evaluation_classification_after_correction/results`.

To visualize the experiments with plots, run the corresponding analysis script:
```bash
python3 evaluation_classification_after_correction/analyse_classification_metric_report.py
```
The resulting plots can be found in `evaluation_classification_after_correction/plots`.
You can also use the helper shell scripts:
- `run_all_classification_analysis.sh`: runs all three scripts sequentially; this can take several hours.
- `run_all_classification_analysis_tmux.sh`: runs the same script in a tmux session so the terminal can disconnect safely. Logs are stored as `classification_analysis_{TIMESTAMP}.log`.

### 6. Reproduce the clustering analysis comparing fedRBE-corrected to uncorrected data

To reproduce the clustering results from the preprint, run the real-dataset clustering notebooks under `evaluation_clusterization_after_correction/`.

#### Prerequisites
- You need to have both the `R` and `Python3` environments set up with the required dependencies as described in the [Prerequisites and setup](#prerequisites-and-setup) section.
- To rerun federated k-means, build the Docker image for the federated k-means app. Central-only k-means does not require Docker.
```bash
cd evaluation_clusterization_after_correction/federated_kmeans_upd/
./build.sh
```

#### Real datasets

Run the notebooks in `evaluation_clusterization_after_correction/real_datasets/`:

1. `00_build_kmeans_matrices.ipynb` — required only for Quartet multiomics.
2. `01_data_preparation.ipynb`
3. `02_central_kmeans.ipynb`
4. `03_federated_runs.ipynb` — optional; required only when regenerating federated k-means outputs.
5. `04_analysis_metrics_plots.ipynb`
6. `05_multiple_runs.ipynb` — optional repeated seeded federated runs.

See `evaluation_clusterization_after_correction/real_datasets/README.md` for detailed options and output paths.

---

## Repository structure

Understanding the repository layout helps in navigating the files and scripts.

```
fedRBE/
├── README.md                                   # General repository overview
├── batchcorrection/                            # fedRBE FeatureCloud app
├── evaluation_data/                            # Data used for evaluation
│   ├── ccRCC_studies/                          # ccRCC proteomics datasets
│   ├── ecoli/                                  # E. coli dataset
│   ├── ovarian_cancer/                         # Ovarian cancer datasets
│   │   ├── before/                             # Uncorrected data with structure needed to run the app
│   │   ├── after/                              # Corrected data
│   │   ├── 00_harmonize_meta_load_data.ipynb   # Data harmonization notebook
│   │   ├── 01_check_datasets_intersection.ipynb
│   │   └── 02_central_RBE.ipynb                # Centralized removeBatchEffect run
│   ├── quartet_multiomics/                     # Quartet multiomics datasets
│   └── simulated/                              # Simulated datasets
├── analyse_fedvscentral.py                     # Compares federated and centralized batch effect corrections.
├── generate_fedrbe_corrected_datasets.py       # Runs fedRBE on all configured datasets and saves results
├── run_sample_experiment.py                    # Runs fedRBE on the sample dataset
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
│   ├── evaluation_ccRCC.ipynb
│   ├── evaluation_ecoli.ipynb
│   ├── evaluation_ovarian_cancer.ipynb         # Evaluation of ovarian_cancer datasets
│   ├── evaluation_quartet_multiomics.ipynb
│   ├── evaluation_simulated.ipynb
│   ├── evaluation_simulated_30runs.ipynb
├── evaluation_classification_after_correction/ # Classification comparison scripts and outputs
├── evaluation_clusterization_after_correction/ # K-means comparison notebooks and FeatureCloud app
└── [other directories/files]
```

---


## Utility scripts overview

Main scripts live at the repository root; shared helpers live in `evaluation_utils/`.

| File | Purpose |
|------|---------|
| `generate_fedrbe_corrected_datasets.py` | Runs configured fedRBE experiments, assigns client inputs, executes the app, and stores corrected outputs. |
| `analyse_fedvscentral.py` | Loads federated and central corrected datasets, checks shape/index consistency, and writes mean/max difference metrics. |
| `evaluation_utils/featurecloud_api_extension.py` | Wraps the FeatureCloud testbed with automatic restart and result-extraction helpers used by the fedRBE correction script. |
| `evaluation_utils/filtering.R` | Provides filtering and preprocessing functions used before centralized limma correction. |
| `evaluation_utils/plots_eda.R` | Provides boxplot, PCA, UMAP, heatmap, and correction-diagnostic plotting helpers. |
| `evaluation_utils/upset_plot.py` | Creates UpSet plots for feature-overlap comparisons. |


## Troubleshooting

Common checks:

- **Docker build failures**: Confirm Docker is installed and running.
- **FeatureCloud controller not starting**: Check port availability, logs, and FeatureCloud login status.
- **Test-run errors**: Stop leftover Docker containers, then restart Docker if needed.
- **Script errors**: Verify prerequisites, file paths, and permissions.
- **Missing data files**: Confirm required inputs exist under `evaluation_data/[dataset]/before/`.
- **Inconsistent results**: Use the same configuration parameters and filtering rules for federated and centralized corrections.

For unresolved issues, open a [GitHub issue](https://github.com/Freddsle/fedRBE/issues).

## Additional resources

- **FeatureCloud Documentation**: [https://featurecloud.ai/assets/developer_documentation/index.html](https://featurecloud.ai/assets/developer_documentation/index.html)
- **limma Package Documentation**: [https://bioconductor.org/packages/release/bioc/html/limma.html](https://bioconductor.org/packages/release/bioc/html/limma.html)
- **ArXiv Preprint**: [https://arxiv.org/abs/2412.05894](https://arxiv.org/abs/2412.05894)
- **GitHub Repository**: [https://github.com/Freddsle/fedRBE](https://github.com/Freddsle/fedRBE)
- **Federated Learning Overview**: [https://en.wikipedia.org/wiki/Federated_learning](https://en.wikipedia.org/wiki/Federated_learning)

## Contact information

For questions, issues, or support, open a [GitHub issue](https://github.com/Freddsle/fedRBE/issues).
