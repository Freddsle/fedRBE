# Clustering Evaluation After Batch Correction

This directory evaluates how batch correction changes sample clustering in
E. coli, ovarian cancer, ccRCC proteomics, and Quartet multiomics. It runs k-means
on uncorrected and corrected matrices, then compares the resulting clusters with
known biological conditions and laboratory/client labels using adjusted Rand
index (ARI). Looking at both targets helps distinguish biological grouping from
grouping driven by the site where a sample was measured.

The workflow supports central k-means and federated k-means through FeatureCloud.
Batch correction and clustering are separate stages: these notebooks read
correction results prepared under `evaluation_data/`; they do not run batch
correction themselves. Choose `CORRECTED_SOURCE` in notebooks 02 and 03;
input preparation runs automatically. Notebook 05 compares both correction sources.

Original matrices stay under `evaluation_data/`. The notebooks write prepared
analysis matrices, app inputs, cluster assignments, and metric tables under
`real_datasets/`. These generated directories are untracked; the notebooks,
scripts, app source, and dataset manifests needed to rebuild them are tracked.

## Real-dataset notebooks (run in order)

| Step | Notebook | Purpose |
|------|----------|---------|
| Optional | `real_datasets/00_build_kmeans_matrices.ipynb` | Inspect shared Quartet joint-matrix preparation |
| 2 | `real_datasets/02_central_kmeans.ipynb` | Run centralized k-means and compute metrics |
| 3 | `real_datasets/03_federated_runs.ipynb` | Optional: run or aggregate federated k-means via FeatureCloud |
| 4 | `real_datasets/04_analysis_metrics_plots.ipynb` | Combine results, generate ARI bar charts, PCA plots, and summary |

## Quick start

For a first run, the central-only route prepares the data, runs local k-means,
and displays the comparison plots. It requires no Docker. Install the environment
and prepare the corrected matrices using the
[detailed prerequisites](real_datasets/README.md#prerequisites) first. All four
datasets, including Quartet joint matrices, are prepared automatically.

Start this command block from the repository root:

```bash
# Run notebooks 02 and 04 for central-only evaluation (no Docker needed):
cd evaluation_clusterization_after_correction/real_datasets/
jupyter execute 02_central_kmeans.ipynb
jupyter execute 04_analysis_metrics_plots.ipynb
```

To compare central and federated k-means, run notebook 03 between 02 and 04.
It can launch new FeatureCloud runs or collect results from an earlier local run.
The [detailed instructions](real_datasets/README.md) explain both routes, the
configuration cells, and where to find each output. For a subset of datasets,
set the same `DATASETS` list in every notebook you run. Notebook 04 also includes
existing federated assignment tables when present, even if you skip 03; keep
only results that match the current prepared inputs and configuration.

## Shared utilities (in `evaluation_utils/`)

| File | Contents |
|------|----------|
| `datasets.yaml` | Dataset path configs (single source of truth) |
| `real_datasets_utils.py` | Data loading, alignment, k-means, metrics, LFS fallback |
| `featurecloud_kmeans_utils.py` | Docker, FeatureCloud test execution, zip extraction |

## Default dataset set

Notebooks 02–05 are configured to evaluate all four datasets:

- `ecoli`
- `ovarian_cancer`
- `ccRCC_proteomics`
- `quartet_multiomics` (Quartet Multiomics) — joint k-means across all three modalities
  (Transcriptomics + Proteomics + Metabolomics rows stacked vertically).

Central k-means and the FeatureCloud `fc_kmeans` app apply their standard
per-feature scaling to every dataset, including the joint matrix. Central k-means
uses the same default initialization count unless `datasets.yaml` overrides `n_init`.

### Multiomics preparation

Quartet starts with separate transcriptomics, proteomics, and metabolomics
matrices. The ordinary evaluators use the shared `build_quartet_matrices()` helper
to align them by pseudo-sample ID, stack their features, and retain the original
labels and missingness. Notebook 00 optionally inspects that same Python helper.
The joint uncorrected matrix is written under
`evaluation_data/quartet_multiomics/before/`, corrected joint matrices under
`evaluation_data/quartet_multiomics/after/<arm>/`, and metadata at the dataset root.

The columns are pseudo-matched joint samples, not a claim that every modality
measured the same physical aliquot. The shared `INCLUDE_CLIENT_04` toggle controls
the client layout: three clients by default, or four when enabled. See the
[detailed preparation description](real_datasets/README.md#optional-quartet-preparation-inspection-00_build_kmeans_matricesipynb)
for paths, matching keys, and evaluation targets.

## Adding a new dataset

1. Add a new entry to `evaluation_utils/datasets.yaml`.
   - Override `n_init` only if k-means convergence on that dataset benefits
     from extra restarts (e.g. small joint matrices).
2. Add the dataset name to the `DATASETS` list in each notebook's configuration cell.
