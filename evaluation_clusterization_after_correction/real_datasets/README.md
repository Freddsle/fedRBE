# Real-Dataset Clustering Evaluation

[`multiple_runs.py`](multiple_runs.py) is the canonical repeated-seed comparison
of central and federated k-means before and after correction. Datasets are
`ecoli`, `ovarian_cancer`, `ccRCC_proteomics` and `quartet_multiomics`. Both
correction arms, `label_aware` and `no_label`, use ten seeds (1, 11, …, 91) and
ten starts for central, local federated and coordinator k-means. The target is
biological condition; k is its number of classes.

## Prerequisites

Follow the [environment and correction guide](../../docs/reproduce.md). Inputs,
metadata and both correction backends must exist at the paths in
[`datasets.yaml`](../../evaluation_utils/datasets.yaml); fetch Git LFS data when
needed. `multiple_runs.py` prepares Quartet joint matrices automatically from
modality inputs. Joint columns are pseudo-matched samples, not necessarily
measurements from the same physical aliquot. The dataset preparation's `INCLUDE_CLIENT_04` setting
controls its three-client default or four-client layout.

Fresh federated runs require Docker and the app image. From the repository root:

```bash
cd evaluation_clusterization_after_correction/federated_kmeans_upd
./build.sh
```

Cached assignment reuse and central fits need no Docker.

## Run the repeated-seed comparison

From the repository root:

```bash
pixi run python evaluation_clusterization_after_correction/real_datasets/multiple_runs.py
```

| Option | Meaning |
| --- | --- |
| `--arms` | `label_aware` and/or `no_label` (default: both) |
| `--spaces` | `all` complete-case features and/or the shared `panel` (default: both) |
| `--rerun-federated` | Launch new FeatureCloud tests instead of reusing saved assignments |
| `--datasets`, `--seeds` | Restrict the run; other rows of the metrics table are kept |
| `--config` | Custom registry with explicit correction paths; use with `--arms label_aware` |
| `--root` | Output base (default: this directory) |

Results go to `--root` for `all` and to `<root>/panel5000/` for `panel`, with
`no_label/` appended for that arm. Central fits are recomputed in every run.
Saved federated assignments are reused without checking them against the current
matrices or settings; after changing either, use `--rerun-federated`, with a
separate `--root` to keep the canonical results. Assignment tables reject other
samples or metadata. Each FeatureCloud test times out after 1,800 seconds.

## Retained results and local outputs

There are four canonical output roots: `.` and `no_label/` for all features,
`panel5000/` and `panel5000/no_label/` for the shared panel. Each retains:

| File within an output root | Use |
| --- | --- |
| `<dataset>/assignments.tsv` | One wide table per dataset with `file`, `condition`, `lab` and `<method>_seed_NNN_kK` prediction columns |
| `metrics_ari_multirun.tsv` | One row per dataset/method/seed: ARI, arithmetic AMI and inertia |

The four roots contain 16 assignment tables in total. Methods are `BC_Cntrl`,
`AC_Cntrl`, `BC_Fed` and `AC_Fed`; for example, `BC_Cntrl_seed_001_k2`.
Metadata is stored once per dataset table. Exact assignments support contingency,
co-assignment and fixed-partition uncertainty analyses; aggregate ARI alone cannot
reproduce those analyses.

Central inertia uses population-SD scaling; federated pooled inertia is recomputed
from assignments with sample-SD scaling.

Keep generated `inputs/`, `fc_kmeans_res/`, per-seed `kmeans_res/` exports and
bulk analysis tables local. These can be regenerated or retained in ignored
`real_pipeline_runs/<run>/` records.
Selected summaries remain under `panel5000/analysis/`.
The four canonical Figure 4 PNGs remain under the repository's `figures/`.

## Shared variance panel

After preparing both correction arms and Quartet joint inputs, run from the
repository root:

```bash
pixi run python -m evaluation_utils.feature_panel
```

This selects up to 5,000 complete-case features per dataset or Quartet modality
by pooled within-batch variance of uncorrected log-scale data, before scaling.
Batches are clients, or client × batch for Quartet. Ties use feature names;
ordered selections must agree across uncorrected, central and federated states
in both arms.

Lists are written to `panel5000/panel_top5000_<dataset>.tsv` under this directory.
Add `--prepare-rf` to export panel-only cohorts to
`evaluation_data/<dataset>/panel5000/` and write `panel5000/rf_datasets.yaml`.
Use the registry with `--config` and a separate `--results-file`; see the
[RF reproduction instructions](../../docs/reproduce.md).

## Sensitivity analyses and descriptive tables

[`figure4_analysis.py`](figure4_analysis.py) accepts `--output-dir` for every
subcommand; dataset analyses also accept `--datasets`, `--arms` and `--seeds`.
Defaults can be expensive: all datasets, both arms and seeds 1, 11, …, 91.

| Subcommand | Analysis and principal controls |
| --- | --- |
| `sensitivity` | Feature-size/ranking/scaling grid and paired ARI/AMI changes; `--sizes`, `--rankings pooled equal total`, `--scalings zscore centered` |
| `starts` | Individual starts, selected fits and partition frequencies; `--n-inits`, `--spaces`, `--states`; defaults to ovarian and Quartet, with Quartet before correction only |
| `subsample` | Client × class subsampling with fresh clustering; `--repeats`, `--fraction`; correction and full-data panel remain fixed |
| `arrival-order` | Offline client-order reconstruction; `--max-orders` bounds a partial check |
| `contingency` | Study × class × cluster counts and class co-assignment from compact assignments |
| `intervals` | Fixed-partition jackknife ARI and paired within-study bootstrap ARI/AMI; `--repeats`, `--alpha` |
| `class-r2` | Feature-wise class R² by state/feature space, counting excluded constant features |
| `rf-cohorts` | Held-out cohort MCC/F1 and seed summaries; `--reports` supplies CSVs |

`contingency` and `intervals` need only compact assignment tables, with
`--run-root` set to this directory or `panel5000/`; `no_label/` is appended for
that arm. They read saved metadata without requiring correction matrices.
Other matrix analyses require prepared inputs, including Quartet joint files; their
`panel` space uses the saved panel lists. `rf-cohorts` reads its reports directly.

For a bounded matrix-analysis check from the repository root:

```bash
pixi run python \
  evaluation_clusterization_after_correction/real_datasets/figure4_analysis.py \
  sensitivity --datasets ecoli --arms label_aware --seeds 1 \
  --sizes all --rankings pooled --scalings zscore \
  --output-dir real_pipeline_runs/clustering_smoke/sensitivity
```

Within-batch rankings use the uncorrected matrix in every state; total-variance
rankings are state-specific. Feature limits apply per Quartet layer, before
scaling. Subsampling intervals condition on correction; fixed-partition intervals
also condition on assignments. Offline arrival-order analysis uses installed
scikit-learn and does not recover actual FeatureCloud message arrival order.

## Figures

[`evaluation/aggregate_figure4.R`](../../evaluation/aggregate_figure4.R)
reads the aggregate clustering tables and RF reports. `FIGURE4_FEATURE_SPACE=panel`
(default) selects shared-panel inputs and `figures/figure4_<arm>.png`; `all`
selects ordinary metrics/reports and `figures/figure4_all_features_<arm>.png`.
`CORRECTION_ARM` chooses the arm. `FIGURE4_ARI_TABLE`, `FIGURE4_RF_REPORT` and
`FIGURE4_OUTPUT` override paths relative to the repository root or absolutely;
keep overrides consistent with the feature-space label. See the
[figure-rendering commands](../../docs/reproduce.md#7-generate-figures-3-4-and-s5-for-each-correction-arm).
