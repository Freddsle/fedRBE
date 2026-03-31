# Clustering Evaluation After Batch Correction

K-means clustering evaluation (central vs. federated) on real datasets
(proteomics, microarray, microbiome). Data is read from `evaluation_data/`.

## Build the federated k-means app

```bash
cd /home/yuliya-cosybio/repos/cosybio/fedRBE
featurecloud app build ./evaluation_clusterization_after_correction/federated_kmeans_upd fc_kmeans_upd
```

## Real datasets evaluation

See [`real_datasets/README.md`](real_datasets/README.md) for full usage.