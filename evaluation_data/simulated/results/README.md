# Controlled simulation study: results

The study evaluates biological recovery, missing-data eligibility and downstream
utility using a common synthetic generator and matched no-label/label-aware
correction arms. Saved evidence is organized by scientific question. The
[protocol](../README.md) gives the shared design and each block's reproduction
command; tables and figures can be inspected without running the experiments.

| Experimental block | Design and saved evidence |
| --- | --- |
| [Biological recovery](../README.md#robustness-simulations) | Nine three-client scenarios × ten runs; [summary](../robustness/summary.tsv), [180 native correction comparisons](../robustness/equivalence.tsv) |
| [Missingness severity and partial overlap](missingness_overlap/README.md) | Ten roots × two panels × four nominal missingness levels in five balanced clients with class-informative on/off features; 80 derived datasets, 160 native comparisons, retention, biological recovery, and LOCO and within-lab 70:30 RF on keep, drop and impute routes, with d-ComBat as a second correction on the drop and impute routes |
| [Downstream utility under confounding](confounding_downstream/README.md) | Four scenarios × ten realizations × five learner seeds; LOCO RF/all-sample k-means and separately reported within-lab 70:30 RF |

The [sample-size scaling check](../README.md#sample-size-scaling) extends the balanced
fedRBE–limma comparison to four sample counts. The
[summary](../sample_size_scaling/summary.tsv) covers 40 datasets, including the ten
600-sample datasets from the recovery block.

[missing_fraction_slopes.tsv](missing_fraction_slopes.tsv) splits the masked-truth and
corrected contrast slopes by the missing fraction of each feature for the full and
partially overlapping five-client panels ([protocol](../README.md#contrast-slopes-by-missing-fraction)).

## Sample-level predictions

Each learner folder stores its per-dataset prediction tables in one Git LFS archive,
`predictions.zip`. No reproduction step reads them. To inspect them, extract an archive
into its folder from the repository root; Git ignores the extracted `predictions/` folder:

```bash
pixi run python -m zipfile -e evaluation_data/simulated/results/missingness_overlap/predictions.zip \
  evaluation_data/simulated/results/missingness_overlap/
```

For corrected inputs, the saved learner tables use fedRBE or d-ComBat outputs for
both learners. The current driver uses limma or ComBat for the central learner.
