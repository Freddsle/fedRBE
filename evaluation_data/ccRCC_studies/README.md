# ccRCC Dataset — Run Instructions

Three ccRCC proteomics studies integrated as a single fedRBE evaluation dataset
(`ccRCC_proteomics`): PDC000127, PXD030344, PXD042844.

---

## Directory layout after setup

```
evaluation_data/ccRCC_studies/
├── before/
│   ├── PDC000127/          # site data (design.tsv, report_filtered.tsv, config.yml)
│   ├── PXD030344/          # site data
│   ├── PXD042844/          # site data
│   └── central_intensities_log_UNION.tsv  # created by Step 1
├── after/
│   ├── intensities_log_Rcorrected_UNION.tsv  # created by Step 2
│   ├── FedApp_corrected_data.tsv             # created by Step 3
│   └── FedApp_corrected_data_smpc.tsv        # created by Step 3
├── data/
│   ├── ccRCC_metadata.csv  # combined sample metadata (Sample, Condition, Dataset)
│   └── ...
├── prepare_ccRCC_data.py   # Step 1 script
└── 01_central_RBE.ipynb    # Step 2 notebook
```

---

## Step 1 — Data preparation

**What:** Adds a `Condition` column (`"Tumor"` / `"Normal"`) to each site's
`design.tsv` and builds the outer-join union matrix across all three studies.

**Run from the repository root:**

```bash
python evaluation_data/ccRCC_studies/prepare_ccRCC_data.py
```

**Outputs:**
- `before/PDC000127/design.tsv` — updated with `Condition` column
- `before/PXD030344/design.tsv` — updated with `Condition` column
- `before/PXD042844/design.tsv` — updated with `Condition` column
- `before/central_intensities_log_UNION.tsv` — outer-joined union matrix

---

## Step 2 — Central (R-based) batch effect correction

**What:** Runs `limma::removeBatchEffect` centrally on the union matrix,
treating `Dataset` as batch and `Condition` (Tumor/Normal) as covariate.
Must be run **after Step 1**.

**Run:** Open and execute all cells of:

```
evaluation_data/ccRCC_studies/01_central_RBE.ipynb
```

Open the notebook from its own directory (or set the working directory to
`evaluation_data/ccRCC_studies/`) so that relative paths resolve correctly.

**Output:**
- `after/intensities_log_Rcorrected_UNION.tsv`

---

## Step 3 — Federated batch effect correction (FeatureCloud)

**What:** Runs the federated limma RBE app via FeatureCloud for the
`ccRCC_proteomics` experiment (and SMPC variant).
Must be run **after Step 1**. Requires the FeatureCloud controller to be
running and the Docker image `featurecloud.ai/bcorrect:latest` to be available.

**Run from the repository root:**

```bash
python generate_fedrbe_corrected_datasets.py
```

To run **only** the ccRCC experiments (skip others), temporarily comment out all
`experiments.append(...)` lines except the two ccRCC lines in
`generate_fedrbe_corrected_datasets.py`.

**Outputs:**
- `after/FedApp_corrected_data.tsv`
- `after/FedApp_corrected_data_smpc.tsv`
- `after/individual_results/PDC000127/`, `.../PXD030344/`, `.../PXD042844/`

---

## Step 4 — Visual / statistical evaluation

**What:** PCA, violin plots, and variance partitioning comparing uncorrected vs
corrected matrices. Must be run **after Steps 2 and 3**.

**Run:** Open and execute all cells of:

```
evaluation/evaluation_ccRCC.ipynb
```

Open from the `evaluation/` directory (or set the working directory there).

---

## Step 5 — Clustering evaluation (k-means)

**What:** Evaluates how well k-means clustering recovers biological signal
(Condition) vs batch signal (Dataset) before and after correction.
Must be run **after Steps 1, 2, and 3** in order.

**Run the notebooks in sequence from `evaluation_clusterization_after_correction/real_datasets/`:**

1. `evaluation_clusterization_after_correction/real_datasets/01_data_preparation.ipynb` — loads and aligns matrices, writes to
   `real_datasets/ccRCC_proteomics/prepared/`
2. `evaluation_clusterization_after_correction/real_datasets/02_central_kmeans.ipynb` — runs centralized k-means on prepared data
3. `evaluation_clusterization_after_correction/real_datasets/03_federated_runs.ipynb` — runs federated k-means via FeatureCloud
4. `evaluation_clusterization_after_correction/real_datasets/04_analysis_metrics_plots.ipynb` — computes ARI, MCC, F1, and plots

---

## Dependencies

| Step | Language / Tool |
|------|----------------|
| 1    | Python ≥ 3.8, pandas |
| 2    | R ≥ 4.0, limma, tidyverse |
| 3    | Python, FeatureCloud controller, Docker |
| 4    | R ≥ 4.0, limma, tidyverse, patchwork, variancePartition |
| 5    | Python, scikit-learn (see `requirements.txt`) |
