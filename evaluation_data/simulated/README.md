# Controlled simulation study

Synthetic log2-scale data test biological recovery, feature retention, and
downstream learning. The main designs use 600 samples and 6,000 features. Label-aware
correction includes condition as a covariate. [FedProt design](https://doi.org/10.1038/s43588-025-00832-7)

Run from the repository root with the locked [pixi environment](../../README.md#installation)
and use fresh output directories. [Saved results](results/README.md)

<a id="robustness-simulations"></a>

## Biological recovery under confounding and nonlinear effects

Nine three-client scenarios × ten runs cover class imbalance, lab–condition
confounding, and intensity-dependent batch effects. [Simulation driver](robustness_simulation.R)

```bash
pixi run --frozen Rscript evaluation_data/simulated/robustness_simulation.R \
  --out /tmp/fedrbe_robustness
```

Use `--runs 1` for one dataset per scenario. [Recovery summary](robustness/summary.tsv) ·
[fedRBE–limma comparisons](robustness/equivalence.tsv)

### Correction and metrics

Compare fedRBE with limma using matched features and observed cells at a tolerance
of `1e-8`. Score recovery against batch-free truth using feature-centred RMSE, contrast
slopes, and differential-expression precision and recall. Contrast slopes regress the
estimated contrasts on their true values through the origin.

<a id="sample-size-scaling"></a>

## Sample-size scaling of the fedRBE–limma agreement

Compare both correction arms at 600, 3,000, 12,000, and 50,004 samples, with ten
runs per size. Allow about 60 GB RAM and 25 GB temporary space per concurrent
50,004-sample run. [Summary](sample_size_scaling/summary.tsv) · [Run-level differences](sample_size_scaling/differences.tsv)

```bash
pixi run --frozen Rscript evaluation_data/simulated/sample_size_scaling.R \
  --out /tmp/fedrbe_scaling
```

Use `--samples 600 --runs 1` for a quick check.

For FeatureCloud, the runner uses prebuilt run-1 client inputs in
`sample_size_scaling/n<SIZE>_run001/before/`. The expression matrices are
gzip-compressed TSV files tracked with Git LFS. Select sizes in the
`experiments` list in `generate_fedrbe_corrected_datasets.py`, then run the
script.

```bash
pixi run --frozen python generate_fedrbe_corrected_datasets.py
```

The runner uses the prepared clients for both correction arms with and without
SMPC and writes outputs to `sample_size_scaling/n<SIZE>_run001/after/`.
To regenerate a missing input bundle, pass its size to
[`prepare_sample_size_inputs.R`](prepare_sample_size_inputs.R).

The run-1 central limma matrices are stored as `intensities_R_corrected.tsv.gz`
under `after/label_aware/` and `after/no_label/` for each size. Both corrections
use the saved client inputs.

## Five-client missingness and feature overlap

Ten roots cross full and partial panels with 0%, 10%, 30%, and 50% nominal
abundance-dependent missingness. Each client has balanced classes, and panels and
missingness levels share draws within each root.
[Results and simulation command](results/missingness_overlap/README.md)

## Contrast slopes by missing fraction

Compare masked and corrected contrasts with complete truth by missing fraction,
pooling the 10%, 30%, and 50% levels within each root and panel.
[Saved slopes](results/missing_fraction_slopes.tsv)

```bash
pixi run --frozen Rscript evaluation_data/simulated/missing_fraction_slopes.R \
  /tmp/fedrbe_missing_fraction_slopes.tsv
```

<a id="confounding-downstream-evaluation"></a>

## Downstream utility under confounding

Random forests score MCC on leave-one-lab-out and stratified within-lab 70:30
splits. The driver fits fedRBE for federated learners and limma for central learners
on training samples. Held-out labs align to the corrected
training mean, while test samples from training labs use their fitted batch
coefficients. Label-aware fits use training labels as a covariate.

K-means uses joint correction on all samples and scores adjusted Rand index.
Joint label-aware correction uses all condition labels.
[Results and rerun commands](results/confounding_downstream/README.md)

### Downstream utility with missingness and partial overlap

Random forests use label-free correction on the same two split types. Keep
requires six observations in each of at least three training labs. Drop uses the
keep features that are complete in training. Impute requires six observations in
every training lab and fills missing cells with the sample's lowest observed value.
[Saved learner results](results/missingness_overlap/README.md#files)

First generate the root inputs with the [simulation command](results/missingness_overlap/README.md#reproduce).
For d-ComBat on drop and impute, supply `distributedCombat.R`, `neuroCombat.R`,
and `neuroCombat_helpers.R` from commit `45b5a25`. [Distributed-ComBat source](https://github.com/andy1764/Distributed-ComBat)

```bash
pixi run --frozen python evaluation_data/simulated/downstream_simulation.py \
  --analysis missingness_overlap --rf-protocols loco within_lab_70_30 \
  --saved-inputs /tmp/fedrbe_missingness/runs --dcombat /path/to/Distributed-ComBat \
  --runs 1,2,3,4,5,6,7,8,9,10 --learner-seeds 101,211,321,431,541 \
  --reference evaluation_data/simulated/results/missingness_overlap/informative_missingness.tsv \
  --out /tmp/fedrbe_missingness_rf
```

For joint correction, add `--correction joint` and use a separate output directory.

<a id="variance-partitioning"></a>

## Variance partitioning

On shared features and cells, the interaction model separates lab, condition,
lab × condition, and residual variance. [Implementation](variance_partition.R)

<a id="notebook-datasets"></a>

## Notebook and application examples

The notebooks generate and prepare complete-data examples for the balanced,
mildly imbalanced, and strongly imbalanced scenarios.
[Simulation](00_data_simulation.ipynb) · [Preparation and central correction](01_data_prep_and_central_RBE.ipynb)

The diagnostic scripts write single-run PCA plots and 30-run comparisons under
`evaluation/eval_simulation/`.

```bash
pixi run --frozen Rscript evaluation/dataset_diagnostics.R simulated
pixi run --frozen Rscript evaluation/evaluation_simulated_30runs.R
```
