
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
  - [4. Optional dataset diagnostics](#4-optional-dataset-diagnostics)
  - [5. Reproduce the classification analysis comparing fedRBE-corrected to uncorrected data](#5-reproduce-the-classification-analysis-comparing-fedrbe-corrected-to-uncorrected-data)
  - [6. Reproduce the clustering analysis comparing fedRBE-corrected to uncorrected data](#6-reproduce-the-clustering-analysis-comparing-fedrbe-corrected-to-uncorrected-data)
  - [7. Generate Figures 3, 4, S5, S7 and S8 for each correction arm](#7-generate-figures-3-4-s5-s7-and-s8-for-each-correction-arm)
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
   Rscript -e 'install.packages(c("coin","data.table","gridExtra","ggtext","glue","IRkernel","jsonlite","knitr","patchwork","reshape2","scales","tidyverse"), repos="https://cloud.r-project.org")'
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
4. [Compare central and federated outputs](#3-comparing-federated-and-central-corrections).
5. Run the [classification](#5-reproduce-the-classification-analysis-comparing-fedrbe-corrected-to-uncorrected-data)
   and [clustering](#6-reproduce-the-clustering-analysis-comparing-fedrbe-corrected-to-uncorrected-data)
   analyses, or inspect their available saved results.
6. [Generate Figures 3, 4, S5, S7 and S8](#7-generate-figures-3-4-s5-s7-and-s8-for-each-correction-arm).

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
| Ovarian cancer | `evaluation_data/ovarian_cancer/00_harmonize_meta_load_data.ipynb` (run time ~30 minutes), then `evaluation_data/ovarian_cancer/02_prepare_correction_inputs.ipynb` |
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
results. `evaluation/dataset_diagnostics.R` writes the after-correction
diagnostic plots.

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
`01_data_prep_and_central_RBE.ipynb`. For the separate 90-dataset robustness campaign,
see the [simulation README](../evaluation_data/simulated/README.md#robustness-simulations).

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

For a quick check without Docker, run the same experiments in one process with
the app's protocol code and compare each merged result to its central result:

```bash
pixi run python evaluation_utils/fedRBE_simulation_scrip_simdata.py --experiments [--only <text>]
```

This skips the SMPC experiments and writes no results. Both routes write the
experiment's `config.yml` into the tracked client folders for the duration of the
run and restore it afterwards, so run only one of them at a time.

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
- `FedApp_batch_coefficients.csv`, or `FedApp_batch_coefficients_smpc.csv`: the
  batch coefficients written by the coordinator. Rows are the hashed feature
  names (sha3-256), columns the batch columns of the design matrix; the
  reference batch has no column.
- `datainfo.json`: classification description of `individual_results/`, linked to
  the original client designs.

In each result folder, the SMPC run follows the non-SMPC run, so
`individual_results/` and `datainfo.json` hold the SMPC client results.
Classification uses these per-client results, and k-means uses the SMPC matrices.

Each experiment publishes outputs only on success. Rejection leaves its output
directory unchanged. A failed campaign exits nonzero and can leave a mixture of
fresh and retained files. Check failed output paths and
`evaluation_data/<dataset>/logs/` before comparison, learning or plotting.

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

### 4. Optional dataset diagnostics

These commands write the dataset-specific diagnostics of the label-aware correction.
Generate Figures 3 and 4 in Step 7. Run from the repository root after the
correction outputs from Steps 1 and 2 are available:

```bash
for dataset in simulated ecoli ovarian_cancer ccRCC_studies quartet_multiomics; do
  pixi run Rscript evaluation/dataset_diagnostics.R "$dataset"
done
```

The real datasets write their central-correction panels:
`evaluation_data/{ecoli,ccRCC_studies}/plots/data_plot_Rcorrected.png`,
`evaluation_data/ovarian_cancer/before/plots/norm_vs_tumor_plot_Rcorrected_UNION.png`
and `evaluation_data/quartet_multiomics/plots/<Modality>_limma_after_diagnostics.png`.
`simulated` writes the single-run PCA plots to `evaluation/eval_simulation/pca_plots/`
and prints the federated-versus-R differences.

For the optional 30-run analysis, run from the repository root:

```bash
pixi run Rscript evaluation/evaluation_simulated_30runs.R
```

Cached plotting uses the committed
`evaluation/eval_simulation/precomputed/simulated_30runs_lmpv_summary.tsv.gz`
and reads the saved `evaluation/eval_simulation/<mode>/differences_fedRBE_R.tsv`
tables. Variance-partitioning plots are written to
`evaluation/eval_simulation/lmpv_plots/`.

For each scenario with all 30 `<run>_FedSim_corrected.tsv` and
`<run>_R_corrected.tsv` files under
`evaluation_data/simulated/<mode>/after/runs/`, the script automatically
recalculates its difference table before plotting, using that scenario's
`all_metadata.tsv`. Full recalculation can take several hours.

Without the summary cache, missing before-correction LMPV tables are computed
from `evaluation_data/simulated/<mode>/before/intermediate/<run>_intensities_data.tsv`
and `all_metadata.tsv`. Building the cache then requires both before- and
after-correction per-run LMPV tables under
`evaluation/eval_simulation/<mode>/lmvp/` for all 30 runs of all three scenarios;
the script does not generate the after-correction LMPV tables.

### 5. Reproduce the classification analysis comparing fedRBE-corrected to uncorrected data

Classification compares predictive performance before and after correction.
To regenerate results, run both experiment types below from the repository root;
the full evaluation can take several hours. To inspect committed results, skip
the experiment runs and use the plotting command below. Results are labelled
`uncorrected`, `label_free` (correction without covariates) or `label_aware`
(the label as fedRBE covariate).

**Train/test (70:30).** Each repetition splits every cohort's uncorrected fedRBE
input 70:30, stratified by the label (and by the batch for clients holding
several batches; Quartet modalities share one split by pseudo-sample). fedRBE
then runs on the training samples only, in the native simulation of the app's
protocol (`batchcorrection/fedrbe.py`, plain aggregation without SMPC).
Federated (one client per cohort) and centralized (pooled) random forests are
trained on all corrected training samples. The test samples are corrected with
the coordinator's batch coefficients of that fedRBE run, which needs no labels,
and predicted. Values fedRBE did not correct, because a feature is too sparse
in a batch or present in fewer than three clients, stay uncorrected in both
training and test samples, so every arm uses the same features. The
uncorrected baseline uses the same splits. Each repetition reports MCC and
support-weighted F1 pooled over all test samples (`All`) and per test cohort,
for both learning types.

```bash
pixi run python evaluation_classification_after_correction/run_classification_train_test_split.py
```

This runs every dataset (`balanced`, `mild_imbalanced`, `strong_imbalanced`,
`ecoli`, `ovarian_cancer`, `ccRCC_proteomics`, `quartet_multiomics`), both
correction arms and ten repetitions (seeds 42–51, each seed is one split).
Select runs with `--datasets`, `--arms label_free label_aware` and `--seeds`;
use `--results-file` to choose the CSV, `--work-dir` for the temporary files
(default: system temp) and `--force-run` to recompute. Complete result rows are
reused; after code or input changes, use a fresh `--results-file` or
`--force-run`. The fedRBE configurations are those of
`generate_fedrbe_corrected_datasets.py` (label-free: `without_covariates`).
Every repetition's split and batch coefficients are saved next to the results
file, by default under
`evaluation_classification_after_correction/results/train_test_splits/<dataset>/seed_<seed>/`.
Ovarian cancer needs about 4 GB RAM.

**Leave-one-cohort-out.** Each cohort is held out once. fedRBE runs on the other
(training) cohorts only, in the same native simulation, with `min_clients: 2` so
that datasets with three cohorts keep two training clients. Federated and
centralized random forests are trained on the corrected training cohorts. Besides
the batch coefficients, the coordinator saves the training level: per feature, the
mean of the corrected training data (the intercept, plus the covariate effects at
the training data's covariate mean). Each batch of the held-out cohort is shifted
per feature so that its mean equals this level, with
`batchcorrection.classes.client.align_to_training_level`, the function behind
`batchcorrection/align_new_batch.py`. Features with fewer than `min_samples` values
in a held-out batch stay uncorrected. The forests then predict the held-out cohort;
MCC and support-weighted F1 are reported for that cohort and both learning types.

The held-out cohort's labels are not used in either arm, so its alignment is
label-free: if its label mix differs from the training data's, part of that
difference is removed as batch effect. The arms differ only in the correction of
the training cohorts.

```bash
pixi run python evaluation_classification_after_correction/run_classification_leave_one_cohort_out.py
```

This runs every dataset of the train/test evaluation, both correction arms and ten
random forest seeds (42–51). The correction does not depend on the seed, so it runs
once per held-out cohort. The options are those of the train/test script
(`--datasets`, `--arms`, `--seeds`, `--results-file`, `--work-dir`, `--force-run`).
The coordinator's batch coefficients, training level and `training_level_info.yml`
of every held-out cohort and arm are saved next to the results file, by default
under
`evaluation_classification_after_correction/results/loco_corrections/<dataset>/<held-out cohort>/<arm>/`.

Both experiment types save metrics to
`evaluation_classification_after_correction/results/classification_metric_report.csv`.

Plot the saved metrics from the repository root:

```bash
python3 evaluation_classification_after_correction/analyse_classification_metric_report.py
```
Plots go to `evaluation_classification_after_correction/plots/{label_aware,label_free}/`:
MCC and support-weighted F1 boxplots with dots for each validation method in the
results. Train/test and LOCO plots show each test cohort's score per seed.
Use `--results-file` and `--plots-dir` to override paths. Plotting requires R.

Optional helpers (default settings):

- `run_all_classification_analysis.sh`: runs all three scripts sequentially; this can take several hours.
- `run_all_classification_analysis_tmux.sh`: runs the same script in a tmux session so the terminal can disconnect safely. Logs are stored as `classification_analysis_{TIMESTAMP}.log`.

RandomForest implementation: `evaluation_classification_after_correction/fc_fed_forest_simple_app`.

### 6. Reproduce the clustering analysis comparing fedRBE-corrected to uncorrected data

Use `evaluation_clusterization_after_correction/real_datasets/multiple_runs.py`
for the ten-seed comparison, with one start per seed
for central, local federated and coordinator k-means. It prepares all four datasets,
including Quartet joint matrices, recomputes central fits and reuses saved
federated assignments unless `--rerun-federated` is given. Both central
and federated correction outputs and their original metadata must be available.

K-means uses all complete-case features. From the repository root, reuse saved
results for both arms:

```bash
pixi run python evaluation_clusterization_after_correction/real_datasets/multiple_runs.py
```

`--arms` selects correction arms (`label_aware`, `no_label`). Cached reuse reads the requested seed/k columns
without checking inputs or settings. Contingency and fixed-partition analyses
need only assignment TSVs. For new federated fits, build the app with
`evaluation_clusterization_after_correction/federated_kmeans_upd/build.sh` from
its directory, then add `--rerun-federated`. For changed cohorts or experiments,
set a separate `--root`. Individual FeatureCloud tests time out after 1,800
seconds; cached reuse needs no Docker.

The [clustering instructions](../evaluation_clusterization_after_correction/real_datasets/README.md)
are the canonical reference for compact assignment/metric tables, saved
assignment reuse, local output retention and robustness analyses. Figure 4 uses
these results for each correction arm.

#### Shared evaluation metrics

[`evaluation_utils/metrics.py`](../evaluation_utils/metrics.py) provides
`classification_scores(truth, labels)` for MCC and support-weighted F1, and
`clustering_scores(truth, labels)` for ARI and arithmetic AMI/NMI. Inputs must share
sample order. F1/MCC require matching class meanings; partition scores allow
arbitrary cluster names. `classification_scores_from_confusion(matrix)` scores
counts whose true rows and predicted columns share class order.
Add metrics to the appropriate helper; experiment writers select which to save.
Scikit-learn returns MCC 0 for constant-class predictions; zero does not establish
random guessing.

### 7. Generate Figures 3, 4, S5, S7 and S8 for each correction arm

After batch correction, run from the repository root. Figure 4 also requires the
classification and clustering results from Steps 5 and 6.

```bash
for arm in label_aware no_label; do
  for script in aggregate_figure3 aggregate_figure4 \
      quartet_separated_modality_pca_fixed_effect_variance; do
    CORRECTION_ARM="$arm" pixi run Rscript "evaluation/$script.R"
  done
done
```

Outputs under `figures/`: `figure3_<arm>.png`, `figure4_<arm>.png`,
`figure4_<arm>_statistics.png`, `figureS5_<arm>.png`, `figureS7_<arm>.png` and
`figureS8_<arm>.png`.
Figure 4 is drawn twice: `figure4_<arm>.png` without statistics and
`figure4_<arm>_statistics.png` with significance brackets and the test caption.
Figures 3 and S5 also write matching `_variance.tsv` files with median variance
fractions and feature counts. Figure 4 writes a matching `_statistics.tsv`
covering Figures 4, S7 and S8.

#### Scenario comparisons in Figures 4, S7 and S8

Figures 4b (leave-one-cohort-out) and 4c (70:30) show one MCC per test cohort
and seed and are overviews; their tests are per test cohort in Figures S7 and
S8. Both correction arms use 10 runs per scenario (Figure 4a: ARI per dataset;
Figure S7: MCC per held-out cohort; Figure S8: MCC per 70:30 test cohort).
These show significance brackets in `figure4_<arm>_statistics.png`,
`figureS7_<arm>.png` and `figureS8_<arm>.png`.

- C1/C2: before versus after correction within each learner; exact two-sided
  Wilcoxon signed-rank tests paired by seed, with Pratt's method for zeros.
- C3/C4: centralized versus federated learning, separately before and after
  correction; exact two-sided unpaired Mann–Whitney U tests.

Tables report raw p-values and Holm-adjusted p-values for the four contrasts
within each dataset, held-out cohort or 70:30 test cohort, and correction arm.
`*` marks adjusted p < 0.05; `n.s.` otherwise.

Mean differences (`mean_b - mean_a`) have 95% percentile bootstrap intervals
(10,000 resamples, seed 1), using paired resampling for C1/C2 and independent
resampling for C3/C4. The `relevant` flag marks |mean difference| ≥0.05.

In the 70:30 and LOCO panels, correction is fitted without the test samples;
held-out LOCO cohorts are aligned to the training level without their labels.
K-means uses the corrections of all samples; there, label-aware correction uses
all labels.

K-means uses all complete-case features and RF uses all union features. ARI inputs
are `metrics_ari_multirun.tsv` (label-aware) and `no_label/metrics_ari_multirun.tsv`
under `evaluation_clusterization_after_correction/real_datasets/`; the RF report is
`evaluation_classification_after_correction/results/classification_metric_report.csv`.
Override paths with `FIGURE4_ARI_TABLE`, `FIGURE4_RF_REPORT` and `FIGURE4_OUTPUT`.

Set `FIGURE4_OUTPUT` to a path ending in `.svg` to export Figure 4 as SVG.
PNG is the default; a path without an extension gets `.png`. The annotated
variant uses the same path with a `_statistics` suffix. For example:

```bash
CORRECTION_ARM=label_aware FIGURE4_OUTPUT=figures/figure4_label_aware.svg \
  pixi run Rscript evaluation/aggregate_figure4.R
```

Please run the following to generate the performance related supplementary figures:

```bash
pixi run Rscript evaluation/aggregate_performance.R
```

The script reads every `*_performance_clients.csv` that
`generate_fedrbe_corrected_datasets.py` writes next to the corrected data in Step 2
(one row per client and run, from the Docker runs over the `internet` channel with
the default query interval of 5 s).

Outputs: `figures/performance_runtime.png` (per dataset: total runtime, client
computation and waiting time of all participants, with and without SMPC, and the
coordinator's runtime split into client computation, aggregation and waiting) and
`figures/performance_memory.png` (container and app process peak memory against the
participant's data size). `PERFORMANCE_DATA_DIR` overrides the searched folder
(default `evaluation_data`), and `PERFORMANCE_RUNTIME_OUTPUT` and
`PERFORMANCE_MEMORY_OUTPUT` override the output paths; a path ending in `.svg`
writes an SVG.

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
│   │   └── 02_prepare_correction_inputs.ipynb # Prepare central and federated correction inputs
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
│   ├── aggregate_figure3.R                     # Figure 3
│   ├── aggregate_figure4.R                     # Figure 4
│   ├── dataset_diagnostics.R                   # Central-correction panels and simulated PCA
│   ├── evaluation_simulated_30runs.R           # Simulated 30-run differences and variance partitioning
│   ├── quartet_separated_modality_pca_fixed_effect_variance.R  # Figure S5
├── evaluation_classification_after_correction/ # Classification comparison scripts and outputs
├── evaluation_clusterization_after_correction/ # K-means comparison scripts and FeatureCloud app
└── [other directories/files]
```

---


## Utility scripts overview

Main scripts live at the repository root; shared helpers live in `evaluation_utils/`.

| File | Purpose |
|------|---------|
| `generate_fedrbe_corrected_datasets.py` | Runs configured fedRBE experiments, assigns client inputs, executes the app, and stores corrected outputs. |
| `analyse_fedvscentral.py` | Loads federated and central corrected datasets, checks shape/index consistency, and writes mean/max difference metrics. |
| `evaluation_utils/featurecloud_api_extension.py` | Wraps the FeatureCloud testbed with automatic restart for the fedRBE correction script. |
| `evaluation_utils/fedRBE_simulation_scrip_simdata.py` | Runs the app's protocol (`batchcorrection/fedrbe.py`) locally in threads without Docker or SMPC: `<input_dir> <output_file>` merges the corrected data of the client subfolders (simulated notebook 01 and `robustness_simulation.R`); `--experiments [--only <text>]` reruns the non-SMPC experiments of `generate_fedrbe_corrected_datasets.py` and compares them to the central results. |
| `evaluation_utils/filtering.R` | Provides filtering and preprocessing functions used before centralized limma correction. |
| `evaluation_utils/run_correction_experiments.py` | Runs central limma correction with selected covariates into canonical or isolated experiment outputs. |
| `evaluation_utils/central_correction.R` | Provides the shared central limma correction step for real datasets. |
| `evaluation_utils/plots_eda.R` | Provides PCA, boxplots, density plots, and correction diagnostics. |
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
