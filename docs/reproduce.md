
<table>
  <tr>
    <td><a href="https://freddsle.github.io/fedRBE/docs/how_to_guide.html"><img src="https://img.shields.io/badge/HowTo_Guide-Click_Here!-007EC6?style=for-the-badge" alt="HowTo Guide"></a></td>
    <td><a href="https://freddsle.github.io/fedRBE/"><img src="https://img.shields.io/badge/Documentation-Click_Here!-007EC6?style=for-the-badge" alt="Documentation"></a></td>
    <td><a href="https://github.com/Freddsle/fedRBE/"><img src="https://img.shields.io/badge/GitHub-Click_Here!-007EC6?style=for-the-badge" alt="GitHub"></a></td>
    <td><a href="https://featurecloud.ai/app/fedrbe"><img src="https://img.shields.io/badge/FeatureCloud_App-Click_Here!-007EC6?style=for-the-badge" alt="FeatureCloud App"></a></td>
  </tr>
</table>

# Reproduce the fedRBE Preprint <!-- omit in toc -->

This guide explains how to reproduce centralized limma correction, federated
fedRBE correction, and downstream evaluation from the
[fedRBE preprint](https://arxiv.org/abs/2412.05894). You can inspect saved results
or regenerate the analysis stages you need.

## Table of Contents <!-- omit in toc -->

- [Prerequisites and setup](#prerequisites-and-setup)
  - [Setup steps](#setup-steps)
- [Running the analysis](#running-the-analysis)
  - [1. Preparing inputs and obtaining centrally corrected data](#1-preparing-inputs-and-obtaining-centrally-corrected-data)
  - [2. Obtaining federated corrected data](#2-obtaining-federated-corrected-data)
  - [3. Comparing federated and central corrections](#3-comparing-federated-and-central-corrections)
  - [4. Produce tables and figures](#4-produce-tables-and-figures)
  - [5. Reproduce the classification analysis comparing fedRBE-corrected to uncorrected data](#5-reproduce-the-classification-analysis-comparing-fedrbe-corrected-to-uncorrected-data)
  - [6. Reproduce the clustering analysis comparing fedRBE-corrected to uncorrected data](#6-reproduce-the-clustering-analysis-comparing-fedrbe-corrected-to-uncorrected-data)
    - [Prerequisites](#prerequisites)
    - [Reproducing the clustering analysis](#reproducing-the-clustering-analysis)
- [Repository structure](#repository-structure)
- [Utility scripts overview](#utility-scripts-overview)
- [Troubleshooting](#troubleshooting)
- [Additional resources](#additional-resources)
- [Contact information](#contact-information)

---

## Prerequisites and setup

1. **Docker**: [Installation Instructions](https://www.docker.com/get-started)
1. **Git LFS** (required for large files used in this workflow)
1. **Either** [pixi](https://pixi.sh) (recommended, Linux x86-64), Conda or Mamba for `environment.yml`, **or** separate Python 3.8+ and R 4.0+ installations.
1. **System resources**:
   - ≥ 16 GB RAM; the correction script can use nearly all of it, so leave capacity for other processes
   - ≥ 20 GB free disk space

### Setup steps

1. **Set up Git LFS locally:**

   Install [Git LFS](https://git-lfs.com/), then initialize it:
   ```bash
   git lfs install
   ```

2. **Get the repository and large files:**

   For a fresh clone:
   ```bash
   git clone https://github.com/Freddsle/fedRBE.git
   cd fedRBE
   git lfs pull
   ```

   For an existing clone:
   ```bash
   cd fedRBE
   git pull
   git lfs pull
   ```

3. **Set up Python and R dependencies:**

   Choose one of the following setup paths.

   **Option A: pixi environment (recommended)**

   `pixi.toml` and `pixi.lock` define one locked environment with Python 3.12, R 4.5,
   the CRAN and Bioconductor packages, Jupyter with the `python3` and `ir` kernels, and
   the FeatureCloud Python package:
   ```bash
   pixi install --locked
   pixi run setup-affymetrix   # once, only for raw Affymetrix CEL preprocessing
   ```

   `setup-affymetrix` downloads the Affymetrix CDF data packages and rebuilds
   `preprocessCore` without threading, which `affy::rma()` needs. Run notebooks with
   `pixi run jupyter lab` and scripts with `pixi run python ...` or
   `pixi run Rscript ...`, or activate the environment with `pixi shell`.

   **Option B: Conda or Mamba environment**

   The Conda environment installs Python 3.11, R, the CRAN and Bioconductor
   packages used by the notebooks, and the FeatureCloud Python package:
   ```bash
   conda env create -f environment.yml
   conda activate fedRBE
   ```

   Mamba can use the same `environment.yml`. The file pins
   `conda-forge::openblas=0.3.3` to avoid `pthread_create()` failures in
   `affy::rma()` on conda environments.

   If that OpenBLAS build cannot be resolved, activate the environment and use
   the `preprocessCore` source-install command from Option C.

   **Option C: Separate Python and R installations**

   Set up Python dependencies:
   ```bash
   python3 -m venv fedrbe_env
   source fedrbe_env/bin/activate  # on Windows: fedrbe_env\Scripts\activate
   pip install -r requirements.txt
   ```

   `requirements_r.txt` lists CRAN and Bioconductor dependencies in separate
   sections. Install them with:

   ```bash
   Rscript -e 'install.packages(c("data.table","ggpubr","gridExtra","ggsci","ggtext","glue","IRkernel","jsonlite","knitr","nipals","patchwork","pheatmap","reshape2","scales","tidyverse","umap","viridis"), repos="https://cloud.r-project.org")'
   Rscript -e 'if (!requireNamespace("BiocManager", quietly=TRUE)) install.packages("BiocManager", repos="https://cloud.r-project.org"); BiocManager::install(c("AnnotationDbi","affy","GEOquery","GO.db","hgu133acdf","hgu133plus2cdf","impute","limma","variancePartition"), ask=FALSE, update=FALSE)'
   Rscript -e 'BiocManager::install("preprocessCore", configure.args=c(preprocessCore="--disable-threading"), type="source", force=TRUE, ask=FALSE, update=FALSE)'
   Rscript -e 'install.packages("WGCNA", repos="https://cloud.r-project.org")'
   ```

---

## Running the analysis

After fetching Git LFS files and installing dependencies, choose where to start:

- **Inspect existing results:** reuse committed inputs, corrected matrices, and
  result tables. Skip a computation only when the inputs needed by the next
  analysis are available locally.
- **Regenerate results:** run the required stages in order. Preparation and
  correction can be expensive; select the datasets you need before starting.

Recommended order:

1. Complete [setup](#prerequisites-and-setup), including Git LFS downloads.
2. [Prepare or reuse inputs and central corrections](#1-preparing-inputs-and-obtaining-centrally-corrected-data).
3. [Run or reuse federated corrections](#2-obtaining-federated-corrected-data).
4. [Compare central and federated outputs](#3-comparing-federated-and-central-corrections), then
   [produce tables and figures](#4-produce-tables-and-figures).
5. Run the [classification](#5-reproduce-the-classification-analysis-comparing-fedrbe-corrected-to-uncorrected-data)
   and [clustering](#6-reproduce-the-clustering-analysis-comparing-fedrbe-corrected-to-uncorrected-data)
   analyses, or inspect their available saved results.

Paths below are relative to the repository root. Run shell commands there unless
another working directory is specified. Run notebook kernels from the directory
containing the notebook so their relative paths resolve correctly.

Full simulated 30-run recalculation and federated clustering need the generated
inputs described below.

### 1. Preparing inputs and obtaining centrally corrected data

Prepare per-dataset `before/` inputs in the order below. Run preparation scripts
and notebooks from their dataset directory.

| Dataset | Run order |
|---------|-----------|
| Simulated | `evaluation_data/simulated/01_data_prep_and_central_RBE.ipynb`|
| E. coli | `evaluation_data/ecoli/01_data_prep_and_central_RBE.ipynb` |
| Ovarian cancer | `evaluation_data/ovarian_cancer/00_harmonize_meta_load_data.ipynb` (run time ~30 minutes), then `evaluation_data/ovarian_cancer/02_central_RBE.ipynb` |
| ccRCC proteomics | `python prepare_ccRCC_data.py`, then `evaluation_data/ccRCC_studies/01_central_RBE.ipynb` |
| Quartet multiomics | `evaluation_data/quartet_multiomics/01_preprocess_eda.ipynb`, `evaluation_data/quartet_multiomics/02_prepare_RBE_inputs.ipynb`, then `evaluation_data/quartet_multiomics/03_central_RBE.ipynb` |

The real-data notebooks prepare inputs only. Each writes one filtered, uncorrected
central input: `before/central_input.tsv` (`before/<Modality>/` for Quartet;
`central_input.zip` for ovarian). Run central R/limma from the root:

```bash
pixi run python -m evaluation_utils.run_correction_experiments correct
```

This writes label-aware outputs under `after/label_aware/` for all four real datasets.
Use `--datasets` for a subset and `--arms label_aware no_label` for both variants.
`--arms no_label` writes under `after/no_label/`
(`after/<arm>/<Modality>/` for Quartet). Simulation correction runs in its notebook.
For isolated central experiments, add `--output-dir <run-directory>`. Both
correction arms are then written under that directory without changing canonical
results. The dataset evaluation notebooks in `evaluation/` write the
after-correction diagnostic plots.

`--arms no_label` needs cohort/batch metadata but no biological or evaluation
labels. Biological design columns are required only for `label_aware`.

For ccRCC, the starting inputs are the tracked `report_filtered.tsv` and
`design.tsv` files under `evaluation_data/ccRCC_studies/before/<study>/` for
PDC000127, PXD030344, and PXD042844. `prepare_ccRCC_data.py` aligns their sample
order, encodes `Condition`, and builds the union matrix. This route starts from
prepared study inputs. Source acquisition and QC provenance are documented in the
[ccRCC preparation archive](../evaluation_data/ccRCC_studies/archive/README.md).

For simulated data, committed run-1 inputs support single-run correction. For all
30 runs, first run `evaluation_data/simulated/00_data_simulation.ipynb` (1–2 hours)
to generate `before/intermediate/` and `after/runs/`, then rerun
`01_data_prep_and_central_RBE.ipynb`.

Output:

- Prepared matrices, metadata, and client inputs: `evaluation_data/[dataset]/before/`.
- Real-data centrally corrected matrices: `evaluation_data/[dataset]/after/<arm>/`.
- Dataset-specific prerequisites, filenames, and outputs: each `evaluation_data/<dataset>/README.md`.

Skip this step when using the committed inputs and central corrections.

### 2. Obtaining federated corrected data

This stage runs the fedRBE app on the prepared client inputs under
`evaluation_data/[dataset]/before/` and combines their corrected outputs.
Use the Docker and FeatureCloud environment installed during setup.

All datasets are enabled by default. Before starting, select the experiments:

- In `generate_fedrbe_corrected_datasets.py`, comment or uncomment complete
  `(experiment, result_path)` entries under `ADD EXPERIMENTS` to select runs.
  Result paths are relative to `evaluation_data/`.
- To add datasets, create `evaluation_data/[dataset]/before/` entries following the existing structure.
- Quartet multiomics requires the client folders written by `evaluation_data/quartet_multiomics/02_prepare_RBE_inputs.ipynb`; run that notebook before the federated correction script if those folders were regenerated or removed.

If your local image is missing or predates the current code, build it from the
repository root:

```bash
pixi run featurecloud app build ./batchcorrection featurecloud.ai/bcorrect latest
```

Run fedRBE from the repository root:

```bash
pixi run python ./generate_fedrbe_corrected_datasets.py
```

Both real-data correction arms run with and without SMPC by default.
No-covariate variants retain the label-aware filtering threshold and other
correction settings; preparation and evaluation labels remain available.
No-covariate results use `after/no_label/`; label-aware results use
`after/label_aware/`. Quartet uses `after/<arm>/<Modality>/`.
For Quartet, select modalities in `multiomics_modalities` near the definitions.

The script sets up clients, executes fedRBE in the FeatureCloud test environment,
and aggregates results. Real-data outputs under
`evaluation_data/[dataset]/after/<arm>/` (`<arm>/<Modality>/` for Quartet) are:

- `individual_results/`: client corrections, reports, and logs.
- `FedApp_corrected_data.tsv`, or `FedApp_corrected_data_smpc.tsv` with SMPC:
  merged client data.
- `datainfo.json`: classification description of `individual_results/`, linked to
  the original client designs.

In each result folder, the SMPC run follows the non-SMPC run, so
`individual_results/` and `datainfo.json` hold the SMPC client results.
Classification uses these per-client results, and k-means uses the SMPC matrices.

Notes:

- Runtime depends on dataset size and client count; the full run can take several hours to a day.
- Ovarian cancer correction can require at least 16 GB RAM. To skip it, comment out the ovarian entries in `generate_fedrbe_corrected_datasets.py`.
- Committed fedRBE outputs can be reused without rerunning correction.

For simulated data, the script checks that `before/lab*/` inputs still match run 1 before writing the single-run `FedApp_corrected_data*.tsv` files used by `analyse_fedvscentral.py`.

### 3. Comparing federated and central corrections

Once both correction outputs from Steps 1 and 2 are available, compare them from
the repository root. Committed outputs can be used without rerunning correction:

```bash
python3 ./analyse_fedvscentral.py
```

The script loads central and federated matrices, checks their sample/feature
alignment, and calculates mean and maximum element-wise differences. It prints
the comparison and saves the metrics to `evaluation_data/fed_vc_cent_results.tsv`.
The `correction_arm` column identifies `label_aware` and `no_label` for real
outputs. Simulated outputs use `label_aware` because correction preserves their
A/B biological condition; no label-free simulated counterpart is included.

### 4. Produce tables and figures

These notebooks produce the dataset-specific evaluation figures and tables.
Open the repository-relative paths below, then run each notebook with `evaluation/`
as its working directory:

1. `evaluation/evaluation_simulated.ipynb`
2. `evaluation/evaluation_simulated_30runs.ipynb` (may take several hours to run if all 30 runs are generated)
3. `evaluation/evaluation_ecoli.ipynb`
4. `evaluation/evaluation_ovarian_cancer.ipynb`
5. `evaluation/evaluation_ccRCC.ipynb`
6. `evaluation/evaluation_quartet_multiomics.ipynb`

These notebooks expect the corrected data from Steps 1 and 2. `evaluation_simulated_30runs.ipynb` can use the committed precomputed 30-run LMPV summary cache for the final variance-partitioning plots; full recalculation still requires the generated simulated `before/intermediate/` and `after/runs/` files. Figures are written under `evaluation/plots/` and related evaluation output folders.

### 5. Reproduce the classification analysis comparing fedRBE-corrected to uncorrected data

Classification compares predictive performance before and after correction.
To regenerate results, run both experiment types below from the repository root;
the full evaluation can take several hours. To inspect committed results, skip
the experiment runs and use the plotting command below.

**Train/test split:** Use stratified 70:30 training/testing splits, pooled for
centralized learning and per client for federated learning. This takes up to an hour.

```bash
python3 evaluation_classification_after_correction/run_classification_train_test_split.py
```
**Leave one cohort out:** Train on all clients except one and evaluate on the
held-out client. This trains *n_clients* models and can take several hours.

```bash
python3 evaluation_classification_after_correction/run_classification_leave_one_cohort_out.py
```

Both experiment types save metrics under
`evaluation_classification_after_correction/results`.

Both launchers run ten seeds (42–51) and reuse existing result rows. Use
`--datasets` and `--seeds` to select runs, `--results-file` to choose the CSV, and
`--force-run` to recompute. 70:30 scores include MCC and support-weighted F1.

Both launchers accept `--correction-arm label_aware` (default) or
`--correction-arm no_label` to select the real-data `after/<arm>/datainfo.json`.
Use the same seeds and results CSV for both arms. Results are labelled
`uncorrected`, `corrected` (label-aware), or `no_label`, so the existing analyzer
can compare them. Simulated data keeps its ordinary `after/` input.
All four real datasets use these launchers. Quartet joint matrices and RF client
files are prepared automatically from its modality outputs and original metadata.

For custom inputs, use `--config rf_datasets.yaml` instead of `--correction-arm`,
with entries such as:

```yaml
ecoli:
  classification:
    before: evaluation_data/ecoli/before/datainfo.json
    corrected: evaluation_data/ecoli/after/label_aware/datainfo.json
```

Paths are absolute or repository-relative. Optional `preprocessing_name` labels
corrected results (default: `corrected`). Each correction result folder has its own
descriptor, such as `evaluation_data/ecoli/after/no_label/datainfo.json`. For example:

```bash
python3 evaluation_classification_after_correction/run_classification_train_test_split.py \
  --config rf_datasets.yaml --datasets ecoli --seeds 42 \
  --results-file evaluation_classification_after_correction/results/custom_metrics.csv
```

Plot the saved metrics from the repository root:

```bash
python3 evaluation_classification_after_correction/analyse_classification_metric_report.py
```
Plots go to `evaluation_classification_after_correction/plots`. Optional helpers:

- `run_all_classification_analysis.sh`: runs all three scripts sequentially; this can take several hours.
- `run_all_classification_analysis_tmux.sh`: runs the same script in a tmux session so the terminal can disconnect safely. Logs are stored as `classification_analysis_{TIMESTAMP}.log`.

RandomForest implementation: `evaluation_classification_after_correction/fc_fed_forest_simple_app`.

### 6. Reproduce the clustering analysis comparing fedRBE-corrected to uncorrected data

Compare central and federated k-means before and after batch correction using
the real-dataset notebooks under `evaluation_clusterization_after_correction/`.

#### Prerequisites

- Complete the [R and Python setup](#prerequisites-and-setup).
- Have uncorrected matrices, matching metadata, and the central or federated
  correction outputs selected for analysis available, either regenerated or
  fetched from the repository.
- To rerun federated k-means, build its Docker image. Start this command block
  from the repository root. Central-only k-means does not require Docker.

```bash
cd evaluation_clusterization_after_correction/federated_kmeans_upd/
./build.sh
```

#### Reproducing the clustering analysis

Run notebook kernels from `evaluation_clusterization_after_correction/real_datasets/`
and follow this order:

1. `02_central_kmeans.ipynb`
2. `03_federated_runs.ipynb` — optional; required only when regenerating federated k-means outputs.
3. `04_analysis_metrics_plots.ipynb`
4. `05_multiple_runs.ipynb` — repeated-seed central and federated evaluation.

Notebooks 02, 03, and 05 prepare inputs automatically for all four datasets,
including Quartet joint matrices, from correction outputs and original metadata.
For central-only analysis, run 02 and 04. Federated analysis also needs 03 unless
matching outputs are available locally; FeatureCloud result ZIPs are not committed.
Notebook 00 optionally inspects the same Quartet preparation function.

For repeated-seed evaluation, run 05 directly. Set `CORRECTION_ARM` to
`"label_aware"` (default) or `"no_label"` in notebooks 02–05. Both arms use
the same preparation, evaluation labels, algorithms, and metric tables. Outputs
stay in the notebook directory for `label_aware` and its `no_label/` subdirectory
for `no_label`; keep settings consistent between notebooks 02–04. Compare the
ordinary `metrics_ari_multirun.tsv` tables using matching datasets and seeds.
A custom `DATASET_CONFIG` uses its explicit paths; keep `CORRECTION_ARM` at its
default and choose a separate `OUTPUT_ROOT` for each custom run.

Notebook 05 defaults to ten seeds (1, 11, …, 91) and `n_init=1`.

See the [clustering instructions](../evaluation_clusterization_after_correction/real_datasets/README.md)
for options and output paths.

FedKMeans implementation: `evaluation_clusterization_after_correction/federated_kmeans_upd`.

---

## Repository structure

```
fedRBE/
├── README.md                                   # General repository overview
├── batchcorrection/                            # fedRBE FeatureCloud app
├── evaluation_data/                            # Data used for evaluation
│   ├── ccRCC_studies/                          # ccRCC proteomics datasets
│   ├── ecoli/                                  # E. coli dataset
│   ├── ovarian_cancer/                         # Ovarian cancer datasets
│   │   ├── before/                             # Uncorrected data with structure needed to run the app
│   │   ├── after/                              # label_aware/ and no_label/ corrected data
│   │   ├── 00_harmonize_meta_load_data.ipynb   # Data harmonization notebook
│   │   ├── 01_check_datasets_intersection.ipynb
│   │   └── 02_central_RBE.ipynb                # Prepare central correction inputs
│   ├── quartet_multiomics/                     # Quartet multiomics datasets
│   └── simulated/                              # Simulated datasets
├── analyse_fedvscentral.py                     # Compares federated and centralized batch effect corrections.
├── generate_fedrbe_corrected_datasets.py       # Runs fedRBE on all configured datasets and saves results
├── run_sample_experiment.py                    # Runs fedRBE on the sample dataset
├── pixi.toml, pixi.lock                        # Locked Python and R environment for all evaluation
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
| `evaluation_utils/run_correction_experiments.py` | Runs central limma correction with selected covariates into canonical or isolated experiment outputs. |
| `evaluation_utils/central_correction.R` | Provides the shared central limma correction step for real datasets. |
| `evaluation_utils/plots_eda.R` | Provides boxplot, PCA, UMAP, heatmap, and correction-diagnostic plotting helpers. |
| `evaluation_utils/upset_plot.py` | Creates UpSet plots for feature-overlap comparisons. |


## Troubleshooting

- **Docker build failures**: Confirm Docker is installed and running.
- **FeatureCloud controller not starting**: Check port availability, logs, and FeatureCloud login status.
- **Test-run errors**: Stop leftover Docker containers, then restart Docker if needed.
- **Script errors**: Verify prerequisites, file paths, and permissions.
- **Missing data files**: Confirm required inputs exist under `evaluation_data/[dataset]/before/`.
- **Inconsistent results**: Use the same configuration parameters and filtering rules for federated and centralized corrections.

## Additional resources

- **FeatureCloud Documentation**: [https://featurecloud.ai/assets/developer_documentation/index.html](https://featurecloud.ai/assets/developer_documentation/index.html)
- **limma Package Documentation**: [https://bioconductor.org/packages/release/bioc/html/limma.html](https://bioconductor.org/packages/release/bioc/html/limma.html)
- **ArXiv Preprint**: [https://arxiv.org/abs/2412.05894](https://arxiv.org/abs/2412.05894)
- **GitHub Repository**: [https://github.com/Freddsle/fedRBE](https://github.com/Freddsle/fedRBE)
- **Federated Learning Overview**: [https://en.wikipedia.org/wiki/Federated_learning](https://en.wikipedia.org/wiki/Federated_learning)

## Contact information

For questions, issues, or support, open a [GitHub issue](https://github.com/Freddsle/fedRBE/issues).
