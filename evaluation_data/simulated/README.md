# Simulated datasets

Three-lab E. coli simulations based on FedProt [1].

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

## Robustness simulations

`robustness_simulation.R` tests batch removal and signal recovery with **10 runs per
scenario**: 90 datasets and 180 central corrections in two arms. Prevalences below
are for the latent subgroup within condition B, by lab.

| Scenario | Design |
| --- | --- |
| `balanced` | 100 samples per lab and condition; prevalence 0.6/0.6/0.6 |
| `mild_imbalanced` | Mild lab/condition imbalance; prevalence 0.4/0.5/0.66 |
| `strong_imbalanced` | Strong lab/condition imbalance; prevalence 0.2/0.5/0.7 |
| `intensity_dependent` | Balanced run with raw-scale offsets in labs 2 and 3 |
| `missingness` | Balanced run with lab-specific missingness and feature overlap |
| `confounded_p50`, `confounded_p30`, `confounded_p15`, `confounded_p05` | 150/300/150 samples per lab; condition-B fractions p, 1 - p and p for p = 0.5, 0.3, 0.15 and 0.05; prevalence 0.6/0.6/0.6 |

`eta2_condition`, the between-lab fraction of condition-indicator variance, is about
0, 0.025 and 0.044 for the first three compositions and 0, 0.16, 0.495 and 0.804 for
the confounded family. Condition counts are rounded to whole samples.

### Run

From the repository root in the [pixi environment](../../README.md#installation):

```bash
# Focused numerical checks
pixi run Rscript evaluation_utils/tests/test_robustness_simulation.R
# One run per scenario (nine datasets)
pixi run Rscript evaluation_data/simulated/robustness_simulation.R \
  --runs 1 --out /tmp/robustness_check
# Full campaign
pixi run Rscript evaluation_data/simulated/robustness_simulation.R
```

Options: `--runs 1,5`, `--scenarios balanced,missingness`, `--out DIR`.
Default output: `evaluation_data/simulated/robustness/`; `summary.tsv`,
`equivalence.tsv` and `provenance.txt` are tracked, while the per-run files,
`summary.pdf` and the log are regenerable and ignored.
Completed runs are skipped; code or parameter changes require a new output directory.

The full campaign took 65 minutes and 2.4 GB RAM on an idle 14-core machine; expect
up to about 3.5 hours under load. Use one process per output directory.

### Model

The RobNorm-based generator [2] produces 600 samples and 6,000 features: 1,125 A
and 1,125 B condition features (shift 1.45 in 80% of signal cells), 150 latent
features (shift -1.25 within a subgroup of B) and 3,600 null features.

The simplified ComBat-inspired model [3] adds a lab-wide offset and lab-scaled
Gaussian noise. Noise SDs are 0.5 times inverse-gamma proposals with permuted shapes
3/1/5 and a shared scale drawn from 0.5/1/2. Lab parameters must differ by at least
20%; offsets and noise are redrawn until technical PC1 explains ≥20% of variance.

- Intensity: `log2(2^y + o)` on batch-affected `y`, where `o` is zero for lab 1 and the
  10th/25th percentiles of batch-free raw intensities for labs 2/3 [4].
- Missingness: `simulateMissingValues` per lab, with alpha 0.1/0.2/0.3 and beta 0.8,
  combines an abundance-dependent (MNAR) threshold and random removal (MCAR within
  lab) [5]. Alpha does not fix the realized missing fraction. Then 10%/20%/70% of
  features, stratified by truth role, are kept in one/two/three labs [6].
- Graded confounding: with a homogeneous condition effect and per-lab offsets,
  the omitted-covariate retention benchmark is `1 - eta2_condition`.
- Seeds: `20261016 + 1000 * scenario index + run`, using the table order. Derived
  scenarios reuse the balanced run; missingness draws use its seed + 500.

### Correction and metrics

Both arms use `limma::removeBatchEffect` with lab as batch. `with_condition` supplies
`~condition`; `without_condition` omits it. Features need at least
`labs + covariates + 1` observations per lab and presence in three labs
(`filter_fedrbe_client`). Missingness excludes 1,800 one-/two-lab features.
Limma fits observed samples and preserves NA cells.

Metrics:

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
- `de_condition_precision`, `de_condition_recall`, `de_condition_fp`, `de_condition_fn`:
  condition-feature DE recovery against null features (limma, BH q < 0.05).
  Before/after/truth use `~condition`; `joint` uses `~condition + lab` on uncorrected input.
  Latent features are excluded; precision is unavailable when no features are called.
- Retained/excluded feature counts and per-lab missingness:
  `panel_excluded_features` counts features in fewer than three labs;
  `eligibility_excluded_features` counts additional observation-threshold losses.
  `missing_fraction_<lab>` is the NA fraction over all features after panel absence;
  `missing_fraction_present_<lab>` is the fraction among features present in the lab.
- `input` rows record drawn `offset_<lab>` and `noise_sd_<lab>` before intensity or
  missingness perturbations, plus `eta2_condition` and `eta2_latent_in_b`
  (lab-explained variance of the latent indicator within B).

`before`, `after` and `truth` share retained features and observed cells.

| Additional stage | Scoring |
| --- | --- |
| `baseline_after` | For intensity and missingness: balanced-run correction fitted on full eligible data, scored on the scenario's retained rows and observed cells |
| `paired_after` | `after` minus `baseline_after` |
| `observed_truth` | Masked vs complete truth contrasts; unit slopes and zero shifts without missingness |
| `after_vs_complete` | Corrected vs complete truth contrasts |

`variancePartition` fits `~condition + lab` on complete-data stages of `balanced`,
`mild_imbalanced`, `strong_imbalanced` and `intensity_dependent`.

Local verification calls client/coordinator functions without SMPC or hashing on
runs 1, 5 and 10 of each scenario in both arms (54 full-campaign checks). Retained
features and missing-value positions must match central correction, with maximum
absolute difference ≤1e-8.

### Outputs

- `runs/<scenario>_runNNN.tsv`: scalar metrics.
- `runs/<scenario>_runNNN_variance.tsv.gz`: per-feature variance fractions, complete data
  only, for the four scenarios listed above.
- `summary.tsv`: mean, median, finite/non-finite counts (`n_runs`, `n_missing`),
  Monte Carlo standard error (`mcse` = sample SD / sqrt(n)) and approximate 95%
  Student-t mean intervals (`ci_low`, `ci_high`). MCSE and intervals require at least two
  finite runs; paired intervals summarize run-level differences.
  `mean_abs_dev_1` is mean absolute deviation from unit gain for contrast slopes
  and per-lab gains, excluding paired differences.
- `equivalence.tsv`: local fedRBE versus central checks.
- `summary.pdf`: run-level recovery, paired changes, observed-truth and confounding plots.
- `provenance.txt`: parameters, seeds, code/lock checksums and `sessionInfo()`.

### References

1. Burankova et al. (2025). [FedProt](https://doi.org/10.1038/s43588-025-00832-7).
2. Wang et al. (2021). [RobNorm](https://doi.org/10.1093/bioinformatics/btaa904).
3. Johnson et al. (2007). [ComBat](https://doi.org/10.1093/biostatistics/kxj037).
4. Huber et al. (2002). [Variance stabilization and calibration](https://doi.org/10.1093/bioinformatics/18.suppl_1.S96).
5. Jin et al. (2021). [Missing-value imputation in label-free proteomics](https://doi.org/10.1038/s41598-021-81279-4).
6. Schumann et al. (2025). [BERT](https://doi.org/10.1038/s41467-025-62237-4).
