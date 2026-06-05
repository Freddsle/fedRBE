# Federated K-Means FeatureCloud App

FeatureCloud app used by the clustering evaluation under
`evaluation_clusterization_after_correction/real_datasets/`.

The app:

- intersects feature names across clients;
- filters features by the configured missing-value threshold;
- computes federated means and standard deviations;
- runs k-means for the configured range of `k` values; and
- writes client clustering, centroid, silhouette, scaling, and log files.

## Build

From the repository root:

```bash
featurecloud app build \
  evaluation_clusterization_after_correction/federated_kmeans_upd \
  fc_kmeans_upd
```

Or build directly with Docker:

```bash
docker build \
  -t fc_kmeans_upd \
  evaluation_clusterization_after_correction/federated_kmeans_upd
```

## Run

`real_datasets/03_federated_runs.ipynb` creates the per-client
`intensities.tsv`, `design.tsv`, and `config_kmeans.yml` files and launches the
FeatureCloud tests. `real_datasets/05_multiple_runs.ipynb` uses the same app
for repeated seeded runs.

Each client directory must contain a configuration file named
`config_kmeans.yml` or `config_kmeans.yaml`. The notebooks generate the
complete configuration, including input/output names, scaling options,
cluster range, initialization counts, iteration limit, and random seed.

Generated FeatureCloud test archives, logs, and extracted results are ignored
by the parent repository.
