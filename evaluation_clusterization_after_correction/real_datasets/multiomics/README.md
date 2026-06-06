# Quartet multiomics k-means evaluation

Multiomics is one of the datasets driven by the standard real-dataset flow
(`../01_data_preparation.ipynb` … `../04_analysis_metrics_plots.ipynb`, plus
optionally `../05_multiple_runs.ipynb`). The only multiomics-specific step is
the R notebook in this directory, `00_build_kmeans_matrices.ipynb`, which
prepares the joint k-means matrices and the per-client splits that the
standard flow consumes.

## Pre-step: build the joint matrices and per-client splits

```bash
cd evaluation_clusterization_after_correction/real_datasets/multiomics
jupyter execute --kernel_name=ir 00_build_kmeans_matrices.ipynb
```

This produces:

- `evaluation_data/multiomics/after/all_modalities_{before,corrected,fedsim}_kmeans_matrix.tsv`
  — joint matrices used as `before_matrix` / `corrected_central` /
  `corrected_federated` in `evaluation_utils/datasets.yaml`.
- `evaluation_data/multiomics/before/<client>/{design.tsv, intensities.tsv}`
  — per-client splits used when 03 builds FeatureCloud k-means inputs.

## Why this is k-means-ready

Each per-modality block (Transcriptomics, Proteomics, Metabolomics) is
row-zscored and divided by `sqrt(n_features)` before vertical concatenation,
so no single modality dominates the joint Euclidean distance. The
`evaluation_utils/datasets.yaml` entry for `multiomics` therefore sets
`pre_scaled: true`, which makes both `run_central_kmeans()` and the
FeatureCloud per-client `config_kmeans.yml` skip a second scaling pass.

K-means runs on the **joint matrix containing all three modalities together**
(Transcriptomics + Proteomics + Metabolomics rows stacked, samples as columns).

## Targets

Must match the active client list in
`evaluation_data/multiomics/fedrbe_multiomics_utils.py`, controlled by the
`INCLUDE_CLIENT_04` toggle — default `False`, i.e. 3 clients / 48 joint samples.

- `condition`: donor labels `D5/D6/F7/M8` (`D6` is the limma reference donor), evaluated with `k = 4`.
- `client`: client labels (default 3-client config: `client_01_L01`,
  `client_02_L02`, `client_03_L05_L04`; with `INCLUDE_CLIENT_04 = True` also
  `client_04_L03_L14`), evaluated with `k = len(CLIENT_NAMES)`. This is the
  technical structure that batch correction is meant to remove. The metadata
  column reusing the real-dataset schema is named `lab` for compatibility with
  the other real-dataset scripts.

Joint samples are matched across modalities via the `pseudo_sample` keys built
in `02_prepare_RBE_inputs.ipynb` (`{client}_{donor}_{rep}_{i}`); see that
notebook's README for details.

## Outputs from the standard flow

Under `../<output_root>/multiomics/`:

- `prepared/before_matrix.tsv`, `prepared/corrected_matrix.tsv`, `prepared/metadata.tsv`
- `kmeans_res/runs/1_metadata_cntrl_kmeans_res.tsv` (Before, Cor cluster labels)
- `kmeans_res/runs/{seed}_metadata_fed_kmeans_res.tsv` (federated, optional)
- `metrics/metrics_ari.tsv` — appended into `../metrics_ari.tsv` by 04.
