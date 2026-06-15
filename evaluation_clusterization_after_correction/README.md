# Clustering Evaluation After Batch Correction

K-means clustering evaluation (central vs. federated) on real datasets
(`ecoli`, `ovarian_cancer`, `ccRCC_proteomics`, and
`Quartet - Multiomics`). It measures the effect of fedRBE batch correction on
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
- `ccRCC_proteomics`
- `Quartet - Multiomics` — joint k-means across all three modalities
  (Transcriptomics + Proteomics + Metabolomics rows stacked vertically).
  Both central k-means and the FeatureCloud `fc_kmeans` app apply their
  standard per-feature scaling on the joint matrix, the same way they do for
  every other dataset; the only multiomics-specific tweak in `datasets.yaml`
  is `n_init: 50` (a robustness override for the small 48-sample matrix).

### Multiomics-specific prep

Multiomics needs one extra step *before* notebook 01 — building the joint
matrices and per-client splits from the per-modality matrices:

```bash
cd evaluation_clusterization_after_correction/real_datasets
jupyter execute --kernel_name=ir 00_build_kmeans_matrices.ipynb
```

This writes `all_modalities_before_kmeans_matrix.tsv`,
`all_modalities_corrected_kmeans_matrix.tsv`, and either
`all_modalities_fedapp_kmeans_matrix.tsv` or
`all_modalities_fedsim_kmeans_matrix.tsv` under
`evaluation_data/quartet_multiomics/after/` (shared with
`evaluation/evaluation_quartet_multiomics.ipynb`) and per-client
`design.tsv`/`intensities.tsv` under
`evaluation_clusterization_after_correction/real_datasets/quartet_multiomics/before/`
(k-means-only inputs).
After that, notebooks 01-04 treat multiomics like any other dataset.

The number of clients (3 by default, 4 with `INCLUDE_CLIENT_04 = True`) is
controlled by `evaluation_data/quartet_multiomics/fedrbe_multiomics_utils.py`; it
flows through both the per-modality fedRBE pipeline and the joint k-means
matrices.

## Adding a new dataset

1. Add a new entry to `evaluation_utils/datasets.yaml`.
   - Override `n_init` only if k-means convergence on that dataset benefits
     from extra restarts (e.g. small joint matrices).
2. Add the dataset name to the `DATASETS` list in each notebook's configuration cell.
