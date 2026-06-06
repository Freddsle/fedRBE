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

The notebooks 01-04 are configured to run all five real datasets:

- `ecoli`
- `ovarian_cancer`
- `quartet`
- `ccRCC_proteomics`
- `multiomics` — joint k-means across all three modalities
  (Transcriptomics + Proteomics + Metabolomics, each block row-zscored and
  divided by `sqrt(n_features)` so no single modality dominates the
  Euclidean distance). The joint matrices are already k-means-ready, so the
  `multiomics` entry in `datasets.yaml` sets `pre_scaled: true` (and
  `n_init: 50`), which makes both the central k-means and the FeatureCloud
  per-client config skip a second StandardScaler pass.

### Multiomics-specific prep

Multiomics needs one extra step *before* notebook 01 — building the joint
matrices and per-client splits from the per-modality matrices:

```bash
cd evaluation_clusterization_after_correction/real_datasets/multiomics
jupyter execute --kernel_name=ir 00_build_kmeans_matrices.ipynb
```

This writes `all_modalities_{before,corrected,fedsim}_kmeans_matrix.tsv`
under `evaluation_data/multiomics/after/` and per-client
`design.tsv`/`intensities.tsv` under `evaluation_data/multiomics/before/`.
After that, notebooks 01-04 treat multiomics like any other dataset.

The number of clients (3 by default, 4 with `INCLUDE_CLIENT_04 = True`) is
controlled by `evaluation_data/multiomics/fedrbe_multiomics_utils.py`; it
flows through both the per-modality fedRBE pipeline and the joint k-means
matrices.

## Adding a new dataset

1. Add a new entry to `evaluation_utils/datasets.yaml`.
   - Set `pre_scaled: true` (and a higher `n_init`) only for matrices that
     are already k-means-ready, e.g. row-zscored joint blocks.
2. Add the dataset name to the `DATASETS` list in each notebook's configuration cell.
