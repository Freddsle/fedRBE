# Controlled simulation study

The study uses synthetic log2-scale data based on FedProt [1] to measure batch
removal and biological signal recovery. Run the commands from the repository root
in the [pixi environment](../../README.md#installation).

## Notebook datasets

The notebooks use `balanced`, `mild_imbalanced` and `strong_imbalanced` inputs
without missing values.

- `00_data_simulation.ipynb`: simulation setup.
- `01_data_prep_and_central_RBE.ipynb`: preprocessing and central correction.
- Each scenario's `before/datainfo.json` and `after/datainfo.json`: classification
  input definitions for the uncorrected and corrected data.
- `pixi run Rscript evaluation/dataset_diagnostics.R simulated`, run from the
  repository root: PCA plots of each scenario before and after fedRBE in
  `evaluation/eval_simulation/pca_plots/` and federated-versus-R differences.
- `pixi run Rscript evaluation/evaluation_simulated_30runs.R`: 30-run
  differences and variance-partitioning plots in `evaluation/eval_simulation/`.

Each `before/` contains metadata, raw, log2 and filtered-log2 TSVs, plus per-lab
FeatureCloud folders with data and covariate designs. Central limma RBE uses lab
as batch and condition as a covariate.

The RF launchers read the scenario descriptors, align per-lab matrices and design
labels by sample ID, and create sample-by-feature `tmp_data.csv` inputs
automatically. The prediction target is `condition`; corrected descriptors read
`after/individual_results/<lab>/full_corrected_data.csv` and reuse the matching
`before/<lab>/design.tsv` labels.

<a id="robustness-simulations"></a>

## Biological recovery under confounding and nonlinear effects

The nine scenarios yield 90 datasets and 180 fedRBE–limma comparisons across both
correction arms. The two intensity-offset variants reuse the balanced roots.
[Simulation driver](robustness_simulation.R)

Prevalences in the table refer to the latent subgroup within condition B in each lab.

| Scenario | Design |
| --- | --- |
| `balanced` | 100 samples per lab and condition; prevalence 0.6/0.6/0.6 |
| `mild_imbalanced` | Mild lab/condition imbalance; prevalence 0.4/0.5/0.66 |
| `strong_imbalanced` | Strong lab/condition imbalance; prevalence 0.2/0.5/0.7 |
| `intensity_dependent` | Balanced run with raw-scale offsets in labs 2 and 3 |
| `confounded_p50`, `confounded_p30`, `confounded_p15`, `confounded_p05` | 150/300/150 samples per lab; condition-B fractions p, 1 - p and p for p = 0.5, 0.3, 0.15 and 0.05; prevalence 0.6/0.6/0.6 |
| `intensity_lower` | Balanced run with the same raw-scale offsets at half strength |

`eta2_condition` measures the between-lab fraction of condition-indicator variance.
Condition counts are rounded to whole samples.

### Run

```bash
# Focused numerical checks
pixi run Rscript evaluation_utils/tests/test_robustness_simulation.R
# One run per scenario (nine datasets)
pixi run Rscript evaluation_data/simulated/robustness_simulation.R \
  --runs 1 --out /tmp/robustness_check
# Full recovery block in a fresh output directory
pixi run Rscript evaluation_data/simulated/robustness_simulation.R \
  --out /tmp/robustness_full
```

Use `--runs 1,5` or `--scenarios balanced,intensity_dependent` to select a subset. The default
output directory is `evaluation_data/simulated/robustness/`. Use one process per
directory. Completed runs are skipped. Code or parameter changes require a fresh `--out`.

### Model

The RobNorm-based generator [2] creates 1,125 A and 1,125 B condition features
(shift 1.45 in 80% of signal cells), 150 latent features (shift −1.25 within a
subgroup of B) and 3,600 null features.

The simplified ComBat-inspired model [3] adds a lab-wide offset and lab-scaled
Gaussian noise. Noise SDs are 0.5 times inverse-gamma proposals with permuted shapes
3/1/5 and a shared scale drawn from 0.5/1/2. Lab parameters must differ by at least
20%. Offsets and noise are redrawn until technical PC1 explains ≥20% of variance.

- **Intensity.** Apply `log2(2^y + s * o)` to batch-affected `y`. Offset `o` is
  zero for lab 1 and the 10th and 25th percentiles of batch-free raw intensities
  for labs 2 and 3 [4]. Use `s = 0.5` for `intensity_lower` and `s = 1` for
  `intensity_dependent`.
- **Confounding.** With a homogeneous condition effect and per-lab offsets, the
  no-label retention benchmark is `1 - eta2_condition`.
- **Seeds.** Base cohorts use `20261016 + 1000 * scenario index + run`. The three
  sample-imbalance designs use indices 1–3 and the four confounding designs use 6–9
  in table order. Both offset scenarios reuse the balanced run.

### Correction and metrics

fedRBE and `limma::removeBatchEffect` use lab as batch and preserve NAs. A feature
needs at least `labs + covariates + 1` observations in each lab. The methods must retain
the same features and NA mask and agree within `1e-8`. Recovery metrics use fedRBE
output. In the tables, `with_condition` is label-aware and `without_condition` is
no-label.

- `rmse_to_truth`: feature-centred RMSE to batch-free truth. Its components are
  `residual_lab_rms` (feature-by-lab residual means) and `within_lab_rmse` (remaining
  error); their squares sum to `rmse_to_truth^2`.
- `attenuation_condition`, `attenuation_condition_a_de`, `attenuation_condition_b_de` and
  `attenuation_latent`: retention of the condition and latent-within-B contrasts on signal
  features, `sum(estimate * truth) / sum(truth^2)`, where both terms are contrasts.
  A value of 1 means unit gain, 0 no aligned signal, and negative values reversal.
- `shift_condition`, `shift_latent`: mean estimated minus truth contrast over null features.
- `gain_condition_<lab>` and `gain_latent_<lab>` (before stage): the same slopes within
  each lab.
- `<slope or shift>_bin1` to `_bin3`: the contrast slopes and shifts within fixed
  within-role abundance tertiles of the batch-free truth (median truth over each role's
  unshifted samples: condition B for A features, A for B features, non-latent B for
  latent features, all samples for null features), at the `before` and `after` stages
  of the balanced run and its two offset scenarios.
- `de_condition_precision`, `de_condition_recall`, `de_condition_fp`, `de_condition_fn`:
  condition-feature DE recovery against null features (limma, BH q < 0.05).
  Before/after/truth use `~condition`; `joint` uses `~condition + lab` on uncorrected input.
  Latent features are excluded; precision is unavailable when no features are called.
- All 6,000 features are present in every lab in these scenarios.
- `input` rows record drawn `offset_<lab>` and `noise_sd_<lab>` before intensity
  perturbations, plus `eta2_condition` and `eta2_latent_in_b`
  (lab-explained variance of the latent indicator within B).

`before`, `after` and `truth` share retained features and observed cells.

| Additional stage | Scoring |
| --- | --- |
| `baseline_after` | For the two intensity-offset scenarios: balanced-run correction fitted on full eligible data, scored on the scenario's retained rows and observed cells |
| `paired_after` | `after` minus `baseline_after` |
| `observed_truth` | Masked vs complete truth contrasts; unit slopes and zero shifts without missingness |
| `after_vs_complete` | Corrected vs complete truth contrasts |

`variancePartition::fitExtractVarPartModel` fits `~condition + lab` to the retained
features at `before`, `after` and `truth` in both arms for `balanced`, `mild_imbalanced`,
`strong_imbalanced`, `intensity_dependent` and `intensity_lower`. The output includes
per-feature fractions and median condition and lab fractions for each run, stage and arm.

### Outputs

- `runs/<scenario>_runNNN.tsv`: scalar metrics.
- `runs/<scenario>_runNNN_variance.tsv.gz`: per-feature variance fractions, complete data
  only, for the five scenarios listed above. Values are rounded to six significant
  digits; scalar medians use the unrounded fitted fractions.
- `summary.tsv`: mean, median, finite/non-finite counts (`n_runs`, `n_missing`),
  Monte Carlo standard error (`mcse` = sample SD / sqrt(n)) and approximate 95%
  Student-t mean intervals (`ci_low`, `ci_high`). MCSE and intervals require at least two
  finite runs; paired intervals summarize run-level differences.
  `mean_abs_dev_1` is mean absolute deviation from unit gain for contrast slopes
  and per-lab gains, excluding paired differences.
- `equivalence.tsv`: local fedRBE versus central checks.

<a id="sample-size-scaling"></a>

## Sample-size scaling of the fedRBE–limma agreement

[sample_size_scaling.R](sample_size_scaling.R) repeats the recovery block's comparison
of fedRBE with central `limma::removeBatchEffect` at larger sample counts. It generates
the `balanced` scenario above with 6,000 features at 600, 3,000, 12,000 and 50,004
samples, ten runs each, and compares the two corrections in both arms: 40 datasets and
80 comparisons. Generator, central correction, local fedRBE run and difference metric
are those of the recovery block. The app protocol runs locally with plaintext
aggregation. Central limma uses the original numeric values, while fedRBE reads
inputs serialized to 15 significant digits. The comparison includes this input rounding.

Saved results from the full campaign are available as the
[summary by sample count and arm](sample_size_scaling/summary.tsv) and
[per-dataset differences](sample_size_scaling/differences.tsv).

Allow about 1 GB of memory for a 600-sample dataset. A concurrent 50,004-sample
dataset needs an estimated 60 GB of memory and 25 GB of temporary space.

From the repository root in the [pixi environment](../../README.md#installation):

```bash
# One 600-sample dataset
pixi run Rscript evaluation_data/simulated/sample_size_scaling.R \
  --samples 600 --runs 1 --out /tmp/sample_size_check
# Full campaign
pixi run Rscript evaluation_data/simulated/sample_size_scaling.R
```

Options are `--samples 600,3000`, `--runs 1,5`, `--out DIR` and `--workers N`. The
default output directory is `evaluation_data/simulated/sample_size_scaling_work/`
and is ignored by Git. Each output directory holds checkpoints and summaries of all
its completed datasets. Use one invocation at a time and the same numerical environment
when resuming. Completed datasets are skipped. If parameters or code change, use a
fresh output directory. With `--workers N`, up to N datasets run side by side, each
with the memory and temporary space given above. Temporary files use R's temporary
directory, which the `TMPDIR` environment variable sets.

A balanced design with three labs and two conditions needs a multiple of six samples,
hence 50,004. The 600-sample runs are the balanced runs of the recovery block. Larger
sample counts use the seeds `20261016 + 1000 + run + 100 * k`, with `k` = 1, 2 and 3.

Outputs:

- `differences.tsv`: per dataset and arm, the number of compared values and the minimum,
  mean and maximum absolute difference between central and local fedRBE correction.
- `summary.tsv`: per sample count and arm, the mean of the dataset means and the
  extremes over the datasets.

The `runs/` directory holds the per-dataset checkpoints.

## Five-client missingness and feature overlap

Each client has 60 A and 60 B samples, including 36 latent-B samples. Fifty on/off
features per class have a +1.45 shift in about 80% of own-class cells and a downward
shift of one median background within-feature SD in the other class. The 150 latent
features retain a −1.25 subgroup effect. The other 2,150 original A/B features have no
class effect. The study chose the feature count using roots 1 and 2 and the shift using
three separate roots.

Each root combines full and partial panels with 0%, 10%, 30% and 50% nominal
missingness. In the partial panel, 10/20/20/20/30% of features appear in 1–5 clients,
respectively, at every level [5]. Abundance offsets target a feature-mean SD ten times
the median background within-feature SD. Detection is logit-linear in abundance [6],
with a slope of 0.5 per median within-feature SD. The five lab targets are
0.6/0.8/1.0/1.2/1.4 times the nominal level. Masks share draws across panels and are
nested across levels. Intercepts are fitted before the on/off changes.
[Simulation driver](missingness_overlap.R)

The 80 datasets yield 160 fedRBE–limma comparisons across both correction arms.
The ten roots are the independent replicates. [Results and eligibility records](results/missingness_overlap/README.md)

```bash
pixi run Rscript evaluation_utils/tests/test_missingness_overlap.R
pixi run Rscript evaluation_data/simulated/missingness_overlap.R \
  --roots 1 \
  --out /tmp/missingness_root1_check
```

For all 80 datasets, use `--roots 1,2,3,4,5,6,7,8,9,10` and a fresh `--out`.
Use one process per directory. Completed roots are skipped. Use a fresh directory
after code or parameter changes.

`summary.tsv` reports retention, recovery slopes, RMSE and lab residual RMS overall
and by presence in three, four or five clients. `status.tsv` records comparison
outcomes. `learner_eligibility.tsv` counts features by role and missing-value route.

`eligibility.tsv.gz` records each fitted feature's `eta2_obs`, the share of condition
variance between labs among its fitted cells. For features fitted on the same cells
in both arms, no-label correction retains `1 - eta2_obs` of the label-aware condition
contrast. `eta2_identity` reports the maximum deviation from that relationship.

## Contrast slopes by missing fraction

This descriptive analysis compares masked truth and fedRBE-corrected contrasts with
complete truth in both correction arms. It pools the 10%, 30% and 50% levels within
each of the ten five-client roots, separately for full and partially overlapping panels.

Missing fractions use cells in labs assigned the feature. Slopes are reported for
right-closed groups of 0–5%, 5–10%, 10–20%, 20–30%, 30–50% and 50–100%, plus an
`all` group. [Scoring](#correction-and-metrics)

```bash
pixi run Rscript evaluation_data/simulated/missing_fraction_slopes.R
```

An optional argument sets the output path. The default is [results/missing_fraction_slopes.tsv](results/missing_fraction_slopes.tsv).

<a id="variance-partitioning"></a>

## Variance partitioning

[variance_partition.R](variance_partition.R) describes each feature's variance by
lab, condition, lab × condition and residual, using the interaction model of the
variancePartition vignette [7], `~ (1 | lab) + (1 | condition) + (1 | lab:condition)`.
Each feature is fitted by maximum likelihood with lme4 and variancePartition's lme4
settings, and `variancePartition::calcVarPart()` returns its fractions.
`fitExtractVarPartModel()` (variancePartition 1.40.2) is not used: it refits a template
model, which with lme4 2.0.6 can stop away from the optimum.

`matched_states(native, truth)` matches both arms' inputs and corrections on shared
features and cells, `fit_fractions(x, meta)` fits each feature from nine starting values
and keeps the highest likelihood, and `analyse_dataset(states, meta, role)` returns
per-feature fractions and a summary.

Focused checks are part of `test_robustness_simulation.R`, including fits that must
reach the optimum of an exhaustive grid where a template refit or a single start stops
early.

## References

1. Burankova et al. (2025). [FedProt](https://doi.org/10.1038/s43588-025-00832-7).
2. Wang et al. (2021). [RobNorm](https://doi.org/10.1093/bioinformatics/btaa904).
3. Johnson et al. (2007). [ComBat](https://doi.org/10.1093/biostatistics/kxj037).
4. Huber et al. (2002). [Variance stabilization and calibration](https://doi.org/10.1093/bioinformatics/18.suppl_1.S96).
5. Schumann et al. (2025). [BERT](https://doi.org/10.1038/s41467-025-62237-4).
6. Li and Smyth (2023). [Detection probability curves](https://doi.org/10.1093/bioinformatics/btad200).
7. Hoffman and Schadt (2016). [variancePartition](https://doi.org/10.1186/s12859-016-1323-z); [vignette](https://bioconductor.org/packages/3.23/bioc/vignettes/variancePartition/inst/doc/variancePartition.html), "Including interaction terms".
