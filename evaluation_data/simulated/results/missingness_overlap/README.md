# Missingness severity and partial feature overlap

This block of the [simulation study](../../README.md) crosses ten roots × two feature
panels × four nominal missingness levels (0, 10, 30, 50%) in five balanced clients.
The design has 80 derived datasets and 160
native fedRBE–limma comparisons over both correction arms. Settings share truth
and random draws within each root. The
[protocol](../../README.md#five-client-missingness-and-feature-overlap)
and [driver](../../missingness_overlap.R) define construction and scoring.

All clients have 60 A and 60 B samples, including 36 latent-B samples. The A/B
signal consists of 50 A and 50 B on/off features, with +1.45 in about 80% of
own-class cells and an off-class downward shift of one median background
within-feature SD. The other 2,150 former A/B features have no class effect;
150 latent features retain their −1.25 subgroup effect. Detection follows a
logit-linear curve with nominal lab missing-fraction targets of 0.6 to 1.4 times
the level. Realized missingness within designed panels is recorded below.

## Files

| File | Content |
| --- | --- |
| [informative_metrics.tsv.gz](informative_metrics.tsv.gz), [informative_summary.tsv](informative_summary.tsv) | 42,720 root metrics and 4,272 summaries: retention, recovery against masked and complete truth, RMS_lab, `eta2_obs` summaries and fedRBE–limma agreement |
| [informative_eligibility.tsv.gz](informative_eligibility.tsv.gz) | Eligibility of every feature in both arms, with its `eta2_obs` (960,000 rows) |
| [informative_learner_eligibility.tsv](informative_learner_eligibility.tsv) | Features per role on the keep (`rf_features`), drop (`kmeans_features`) and impute (`impute_features`) routes |
| [informative_missingness.tsv](informative_missingness.tsv), [informative_status.tsv](informative_status.tsv) | Achieved lab missing shares; outcomes of the 160 native comparisons |

Roots are the independent replicates.

Native correction requires at least three clients with six observations per feature
in the no-label arm or seven in the label-aware arm.

## Reproduce

From the repository root, in the locked pixi environment (R 4.5.3, limma 3.66.0):

```bash
pixi run Rscript evaluation_data/simulated/missingness_overlap.R \
  --roots 1,2,3,4,5,6,7,8,9,10 \
  --out /tmp/fedrbe_missingness_informative
```

Use a fresh output directory and one process per directory. Each run saves root
objects under `runs/`.
