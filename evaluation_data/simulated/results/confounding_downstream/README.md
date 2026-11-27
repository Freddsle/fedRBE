# Downstream utility under confounding

Four scenarios × ten datasets × five learner seeds. Random forests (RF) use
training-only correction and MCC. K-means uses joint correction and ARI.
[Study protocol](../../README.md#confounding-downstream-evaluation)

## Saved outputs

Unsuffixed files contain leave-one-lab-out RF and all-sample k-means.
[Metrics](metrics.tsv.gz) · [Summary](summary.tsv) · [Paired summary](paired_summary.tsv)

[Predictions and assignments](predictions.zip) are stored in Git LFS.
[Extraction instructions](../README.md#sample-level-predictions)

### Secondary within-lab 70:30 RF

These files use the `_within_lab_70_30` suffix.
[Metrics](metrics_within_lab_70_30.tsv.gz) · [Summary](summary_within_lab_70_30.tsv) · [Paired summary](paired_summary_within_lab_70_30.tsv)

## Reproduce

Run from the repository root with pixi and use a fresh output directory.
The full RF run takes about a day with one worker.

```bash
pixi run --frozen python evaluation_data/simulated/downstream_simulation.py \
  --rf-protocols loco within_lab_70_30 \
  --runs 1,2,3,4,5,6,7,8,9,10 --learner-seeds 101,211,321,431,541 \
  --reference evaluation_data/simulated/results/confounding_downstream/reference_metrics.tsv.gz \
  --out /tmp/fedrbe_confounding_rf
```

For k-means, replace `--rf-protocols loco within_lab_70_30` with
`--correction joint --tasks kmeans` and use a separate output directory.
