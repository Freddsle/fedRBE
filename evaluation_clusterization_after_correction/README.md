# Clustering Evaluation After Batch Correction

K-means clustering evaluation (central vs. federated) on real datasets
(`ecoli`, `ovarian_cancer`, `quartet`, `ccRCC_proteomics`, and
`multiomics`). It measures the effect of fedRBE batch correction on
clustering quality using ARI, MCC, and accuracy.

All data is read directly from `evaluation_data/` — no copies are made.
Generated `prepared/`, `inputs/`, `kmeans_res/`, and `metrics/` directories
under `real_datasets/` are not tracked. The notebooks, scripts, app source,
and dataset manifests required to regenerate them are tracked.

## Notebooks (run in order)

| Step | Notebook | Purpose |
|------|----------|---------|
| 1 | `01_data_preparation.ipynb` | Load data from `evaluation_data/`, filter, align, save prepared matrices |
| 2 | `02_central_kmeans.ipynb` | Run centralized k-means, compute metrics |
| 3 | `03_federated_runs.ipynb` | *(Optional)* Run or aggregate federated k-means via FeatureCloud |
| 4 | `04_analysis_metrics_plots.ipynb` | Combine results, generate ARI bar charts, PCA plots, summary |

## Quick start

```bash
# Run notebooks 01-02-04 for central-only evaluation (no Docker needed):
cd evaluation_clusterization_after_correction/
jupyter execute 01_data_preparation.ipynb
jupyter execute 02_central_kmeans.ipynb
jupyter execute 04_analysis_metrics_plots.ipynb
```

## Shared utilities (in `evaluation_utils/`)

| File | Contents |
|------|----------|
| `datasets.yaml` | Dataset path configs (single source of truth) |
| `real_datasets_utils.py` | Data loading, alignment, k-means, metrics, LFS fallback |
| `featurecloud_kmeans_utils.py` | Docker, FeatureCloud test execution, zip extraction |

## Default dataset set

The notebooks are currently configured to run:

- `ecoli`
- `ovarian_cancer`
- `quartet`
- `ccRCC_proteomics`

Multiomics uses the dedicated `real_datasets/06_multiomics_kmeans.py`
workflow after its joint matrices have been built.

## Adding a new dataset

1. Add a new entry to `evaluation_utils/datasets.yaml`.
2. Add the dataset name to the `DATASETS` list in each notebook's configuration cell.
