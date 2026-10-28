# Clustering Evaluation After Batch Correction

Evaluate central and FeatureCloud federated k-means before and after correction
for E. coli, ovarian cancer, ccRCC proteomics and Quartet multiomics. Correction
outputs and original inputs live under `evaluation_data/`; this component runs
clustering and evaluates biological condition labels using ARI and AMI.

Use [real_datasets/multiple_runs.py](real_datasets/multiple_runs.py)
for the canonical repeated-seed comparison: ten seeds and ten starts for central,
local federated and coordinator k-means. It reads both correction backends and
supports the shared variance panel or all complete-case features. The
[real-dataset instructions](real_datasets/README.md) cover preparation, saved
assignment reuse, RF panel exports, sensitivity analyses and Figure 4 rendering.

From the repository root, after the
[environment and correction setup](../docs/reproduce.md):

```bash
pixi run python evaluation_clusterization_after_correction/real_datasets/multiple_runs.py
```

Reuse reads the requested assignment columns without checking inputs or
settings. Saved-result analyses read assignment TSVs directly. For new federated
runs, build `federated_kmeans_upd` and add `--rerun-federated`. `--arms` and
`--spaces` select correction arms and feature spaces; for a new cohort or
experiment, set a separate `--root`.

Retained results consist of compact per-dataset assignments, aggregate metrics
and selected analysis summaries. App inputs, result archives, extracted client
files and per-seed exports are local generated artifacts. See the
[retention layout](real_datasets/README.md#retained-results-and-local-outputs).

Shared implementation lives in `evaluation_utils/`: `datasets.yaml` defines input
paths, `real_datasets_utils.py` prepares aligned matrices, central fits and
assignment tables, `featurecloud_kmeans_utils.py` runs FeatureCloud tests, and
`feature_panel.py` prepares the shared panel. Add new datasets to the registry and
select them with `--datasets`.
