# ccRCC Dataset — Run Instructions

Three ccRCC proteomics studies integrated as a single fedRBE evaluation dataset
(`ccRCC_proteomics`): PDC000127, PXD030344, PXD042844.

Each study is a batch; tumor versus normal tissue is the biological condition
to preserve. The workflow prepares a common input matrix, runs central and
federated correction, then compares their results. Steps 2 and 3 both depend on
Step 1, but neither correction depends on the other. Step 4 compares the
corrections; Step 5 evaluates a selected correction result through clustering.

## Directory layout after setup

```
evaluation_data/ccRCC_studies/
├── before/
│   ├── PDC000127/          # site data (design.tsv, report_filtered.tsv, config.yml)
│   ├── PXD030344/          # site data
│   ├── PXD042844/          # site data
│   ├── central_intensities_log_UNION.tsv  # uncorrected union from Step 1
│   └── central_input.tsv                 # filtered, uncorrected central input from Step 1
├── after/
│   ├── label_aware/       # correction preserving Condition
│   │   ├── intensities_log_Rcorrected_UNION.tsv  # created by Step 2
│   │   ├── FedApp_corrected_data.tsv             # created by Step 3
│   │   └── FedApp_corrected_data_smpc.tsv        # created by Step 3
│   └── no_label/          # same filenames without biological correction covariates
├── archive/                # source-input provenance
├── data/                   # local raw inputs and intermediate reports; ignored
├── plots/                  # before- and after-correction diagnostic plots
├── prepare_ccRCC_data.py   # Step 1 script
└── 01_central_RBE.ipynb    # Step 1 filtering and central-input preparation
```

## Prepared inputs

The workflow starts from tracked `before/<site>/report_filtered.tsv`
and `design.tsv` files for the three studies above. The
[preparation archive](archive/README.md) documents source-input provenance. Its
notebooks require unavailable helper APIs and a source workbook, and do not
specify every export step, so they are not a runnable preparation pipeline in this
checkout. Raw inputs and intermediate reports are local under `data/`.

To reproduce correction from the prepared inputs, start at Step 1. To inspect
existing correction results, start at Step 4 or 5 after checking that the files
listed under Steps 1–3 exist. Rerunning preparation or correction overwrites its
outputs; it is not necessary just to view existing results.

## Step 1 — Data preparation

**What:** Add numeric `Condition` (`Tumor=1`, `Normal=0`) to each site's design
and align its rows to the corresponding report columns. Join the three
`report_filtered.tsv` matrices by feature using an outer join: features absent
from one study remain as missing values rather than being discarded everywhere.

**Run from `evaluation_data/ccRCC_studies/`:**

```bash
python prepare_ccRCC_data.py
```

**Outputs:**

- `before/PDC000127/design.tsv`, `before/PXD030344/design.tsv`, and
  `before/PXD042844/design.tsv` — updated in place with the aligned, encoded design.
- `before/central_intensities_log_UNION.tsv` — combined uncorrected matrix,
  overwritten if it already exists.

Then execute [01_central_RBE.ipynb](01_central_RBE.ipynb) with an R kernel from
`evaluation_data/ccRCC_studies/`. It writes `before/central_input.tsv` and the
diagnostic plot `plots/data_plot.png`.

## Step 2 — Central (R-based) batch effect correction

**Run after Step 1, from the repository root:**

```bash
pixi run python -m evaluation_utils.run_correction_experiments correct --datasets ccRCC_proteomics
```

**Output:** `after/label_aware/intensities_log_Rcorrected_UNION.tsv`.
`pixi run Rscript evaluation/dataset_diagnostics.R ccRCC_studies`, run from the
repository root, then writes the after-correction diagnostic plot
`plots/data_plot_Rcorrected.png`.

Other correction modes: [reproduction guide](../../docs/reproduce.md).

## Step 3 — Federated batch effect correction (FeatureCloud)

**What:** Run fedRBE with one client per study and `Condition` as the biological
covariate. The driver includes runs with and without secure multiparty
computation (SMPC).

**Prerequisites:** Step 1, Docker, a working FeatureCloud controller setup, and
the image `featurecloud.ai/bcorrect:latest`.

**Run from the repository root:** The driver runs every enabled experiment. To
select only ccRCC, retain its `(Experiment, output path)` entries in
`generate_fedrbe_corrected_datasets.py` and comment out the other rows. The four
ccRCC rows cover both correction arms with and without SMPC.

```bash
python generate_fedrbe_corrected_datasets.py
```

**Outputs:**

- `after/label_aware/FedApp_corrected_data.tsv`
- `after/label_aware/FedApp_corrected_data_smpc.tsv`
- `after/no_label/FedApp_corrected_data.tsv` and `after/no_label/FedApp_corrected_data_smpc.tsv` for the no-covariate variants
- `after/label_aware/individual_results/PDC000127/`, `.../PXD030344/`, `.../PXD042844/`
- `after/label_aware/datainfo.json` and `after/no_label/datainfo.json`:
  classification descriptions of each arm's per-client results

## Step 4 — Visual / statistical evaluation

**What:** Inspect the centrally corrected matrix with PCA, sample boxplots and
densities, and compare uncorrected and federated matrices with PCA and variance
partitioning (Figure 3c). Metadata comes from the tracked `before/<site>/design.tsv`
files.

**Run after Steps 2 and 3,** from the repository root:

```bash
pixi run Rscript evaluation/dataset_diagnostics.R ccRCC_studies
```

This writes `plots/data_plot_Rcorrected.png`. Figure 3c, which assesses remaining
study effects alongside the biological condition signal, is rendered with the
[figure commands](../../docs/reproduce.md#7-generate-figures-3-4-and-s5-for-each-correction-arm).

## Step 5 — Clustering evaluation (k-means)

**What:** Evaluate whether central and federated k-means recover the biological
condition (tumor or normal) before and after central and federated correction.
Adjusted Rand index (ARI) and adjusted mutual information (AMI) measure agreement
with the condition labels over ten seeds.

**Run:** After Steps 1–3, from the repository root:

```bash
pixi run python evaluation_clusterization_after_correction/real_datasets/multiple_runs.py \
  --datasets ccRCC_proteomics
```

Inputs are prepared automatically. Central fits are recomputed and saved federated
assignments are reused; new federated fits (`--rerun-federated`) require Docker
and the clustering app described in the
[clustering instructions](../../evaluation_clusterization_after_correction/real_datasets/README.md).

## Dependencies

| Step | Language / Tool |
|------|----------------|
| 1    | Repository environment from `environment.yml` or separate Python/R setup, pandas, tidyverse |
| 2    | Repository evaluation environment, Python launcher and R limma |
| 3    | Repository environment from `environment.yml` or separate Python install from `requirements.txt`, FeatureCloud controller, Docker |
| 4    | Repository environment from `environment.yml` or separate R install from `requirements_r.txt`, limma, tidyverse, patchwork, variancePartition |
| 5    | Repository environment from `environment.yml` or separate Python install from `requirements.txt`, scikit-learn |
