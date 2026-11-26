# Controlled simulation study: results

Find the figures, tables, and rebuild command in the [report index](reported/README.md).
For experiment design and reruns, see the [simulation protocol](../README.md).

| Analysis | Saved results |
| --- | --- |
| Three-client biological recovery | [Summary](../robustness/summary.tsv), [fedRBE–limma agreement](../robustness/equivalence.tsv) |
| Five-client missingness and overlap | [Feature retention, recovery, and random forests](missingness_overlap/README.md) |
| Downstream learning under confounding | [Random forests and k-means](confounding_downstream/README.md) |
| Sample-size scaling | [fedRBE–limma summary across four sample sizes](../sample_size_scaling/summary.tsv) |
| Contrast slopes by missing fraction | [Masked-truth and corrected slopes for both five-client panels](missing_fraction_slopes.tsv) |

## Sample-level predictions

Each learner folder has a Git LFS archive, `predictions.zip`. To inspect it,
extract it from the repository root. Git ignores the extracted `predictions/` directory.

```bash
pixi run python -m zipfile -e evaluation_data/simulated/results/missingness_overlap/predictions.zip \
  evaluation_data/simulated/results/missingness_overlap/
```

Archived predictions use fedRBE or d-ComBat for both learners on corrected inputs.
Reruns use limma or ComBat for the central learner.
