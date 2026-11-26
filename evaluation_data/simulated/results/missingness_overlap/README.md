# Missingness severity and partial feature overlap

Ten roots × two feature panels × four nominal missingness levels (0%, 10%, 30%,
50%) across five balanced clients. [Simulation protocol](../../README.md#five-client-missingness-and-feature-overlap)

## Files

[Recovery summary](informative_summary.tsv) · [Feature eligibility](informative_eligibility.tsv.gz) · [Observed missingness](informative_missingness.tsv) · [fedRBE–limma checks](informative_status.tsv)

Random forests use label-free correction on keep, drop, and impute routes.
The d-ComBat comparison covers drop and impute.

| RF summary | Training-only correction | Joint correction |
| --- | --- | --- |
| Leave-one-lab-out | [Summary](informative_training_only_summary.tsv) | [Summary](informative_dcombat_summary.tsv) |
| Within-lab 70:30 | [Summary](informative_training_only_summary_within_lab_70_30.tsv) | [Summary](informative_dcombat_summary_within_lab_70_30.tsv) |

[Predictions](predictions.zip) are stored in Git LFS. [Extraction instructions](../README.md#sample-level-predictions)

## Reproduce

Run from the repository root with pixi and use a fresh output directory.

```bash
pixi run --frozen Rscript evaluation_data/simulated/missingness_overlap.R \
  --roots 1,2,3,4,5,6,7,8,9,10 --out /tmp/fedrbe_missingness
```

Pass `/tmp/fedrbe_missingness/runs` to `--saved-inputs` in the learner command.
Use `--dcombat DIR` to include d-ComBat and `--correction joint` to fit on all samples.
[Learner protocol](../../README.md#downstream-utility-with-missingness-and-partial-overlap)

## Correction scores on shared measurements

Score the six-observation no-label fit on the features and cells that also
pass the seven-observation label-aware filter.
[Reference scores](informative_reference_scores.tsv) · [Common-cell slopes](informative_common_cell_slopes.tsv) · [Label-aware diagnostics](informative_label_aware_diagnostics.tsv) · [Agreement checks](informative_label_aware_agreement.tsv)

Use the matching learner exports and the Distributed-ComBat sources specified
in the learner protocol.

```bash
pixi run --frozen Rscript evaluation_data/simulated/five_client_diagnostics.R \
  /tmp/fedrbe_missingness/runs /path/to/learner_exports \
  /path/to/Distributed-ComBat /tmp/fedrbe_missingness_diagnostics
```
