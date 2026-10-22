# Simulated datasets

Three-lab E. coli simulations based on FedProt [1].

## Notebook datasets

The notebooks use `balanced`, `mild_imbalanced` and `strong_imbalanced` compositions.
Missingness injection is disabled, so their input matrices contain no NA values.

- `00_data_simulation.ipynb`: simulation setup.
- `01_data_prep_and_central_RBE.ipynb`: preprocessing and central correction.
- `{balanced,mild_imbalanced,strong_imbalanced}/prepare_data.py`: scenario-specific app folder preparation.
- `../../evaluation/evaluation_simulated.ipynb` and
  `../../evaluation/evaluation_simulated_30runs.ipynb`: evaluation plots and metrics.

Each scenario's `before/` contains combined TSVs (metadata, raw, raw-log2 and
filtered-log2) and one FeatureCloud folder per lab with data and a covariate design.
Central limma RBE uses condition as a covariate and lab as batch.

## Robustness simulations

`robustness_simulation.R` tests batch removal and biological-signal recovery in a new
cohort. Each scenario has **10 runs**: 50 datasets and 100 central corrections across
two arms. Prevalences below refer to the latent subgroup within condition B, by lab.

| Scenario | Design |
| --- | --- |
| `balanced` | 100 samples per lab and condition; prevalence 0.6/0.6/0.6 |
| `mild_imbalanced` | Mild lab/condition imbalance; prevalence 0.4/0.5/0.66 |
| `strong_imbalanced` | Strong lab/condition imbalance; prevalence 0.2/0.5/0.7 |
| `intensity_dependent` | Balanced run with raw-scale offsets in labs 2 and 3 |
| `missingness` | Balanced run with lab-specific missingness and feature overlap |

### Run

From the repository root in the [pixi environment](../../README.md#installation):

```bash
# Focused numerical checks
pixi run Rscript evaluation_utils/tests/test_robustness_simulation.R
# One run per scenario (five datasets)
pixi run Rscript evaluation_data/simulated/robustness_simulation.R \
  --runs 1 --out /tmp/robustness_check
# Full campaign (50 datasets)
pixi run Rscript evaluation_data/simulated/robustness_simulation.R
```

Options: `--runs 1,5`, `--scenarios balanced,missingness` and `--out DIR`. Outputs default
to `evaluation_data/simulated/robustness/` (not tracked). Reruns skip completed runs;
changed code or parameters require a new `--out`.

### Model

The adapted RobNorm generator [2] produces 6,000 features and 600 samples.
The simplified ComBat-inspired batch model [3] adds a lab offset and lab-scaled
Gaussian noise without scaling biology. There are 1,125 A and 1,125 B condition features
(shift 1.45 in 80% of signal cells), 150 latent features (shift -1.25 in a subgroup of B),
and 3,600 null features.

- Intensity: `log2(2^y + o)`, where `o` is zero for lab 1 and the 10th/25th percentiles
  of batch-free raw intensities for labs 2/3. This raw-scale additive calibration term
  [4] shifts low log intensities more. The quantiles are stress-test settings, not
  fitted instrument parameters.
- Missingness: `simulateMissingValues` per lab, with alpha 0.1/0.2/0.3 and beta 0.8,
  combines an abundance-dependent (MNAR) threshold and random removal (MCAR within
  lab) [5]. Alpha does not fix the realized missing fraction. Then 10%/20%/70% of
  features, stratified by truth role, are kept in one/two/three labs, introducing
  whole-feature absence as in BERT [6].
- Seeds: `20261016 + 1000 * scenario index + run`. Derived scenarios reuse the balanced
  run; missingness draws use its seed + 500.

### Correction and metrics

Both arms use `limma::removeBatchEffect` with lab as batch: `with_condition` supplies
design `~condition`; `without_condition` omits it. `filter_fedrbe_client` applies
production eligibility: at least `labs + covariates + 1` observations per lab and
presence in at least three labs. Limma fits observed samples and preserves NA cells.

Primary outcomes:

- `rmse_to_truth`: feature-centered RMSE to batch-free truth.
- `attenuation_condition` and `attenuation_latent`: retention of condition and
  latent-within-B contrasts. For relevant signal features, retention is
  `sum(estimate * truth) / sum(truth^2)`, where both terms are contrasts.
  A value of 1 means unit gain, 0 no aligned signal, and negative values reversal.
  This measures average gain across features.
- Retained/excluded feature counts and per-lab missingness: coverage.
  `panel_excluded_features` counts features in fewer than three labs;
  `eligibility_excluded_features` counts additional observation-threshold losses.

Recovery stages (`before`, `after`, `truth`) use identical retained features and
observed cells. `variancePartition` fractions for `~condition + lab` are secondary
diagnostics on complete data.

Intensity and missingness runs are paired with their balanced source run.
`baseline_after` scores the balanced correction, fitted on its full eligible data,
on the added scenario's retained rows and observed cells. `paired_after` is
added-scenario minus baseline; negative RMSE differences mean lower error.

`selection` uses the same formula with observed-cell truth contrasts as estimates
and full-truth contrasts as references, on retained features. This diagnoses selection
of observed values; complete scenarios have unit gain. These slopes do not form an
exact decomposition of signal loss.

Production fedRBE runs locally on runs 1, 5 and 10 of each scenario in both arms
(30 checks). It must match central correction's retained features and NA cells,
with maximum absolute difference at most 1e-8. SMPC and network transport are untested.

### Outputs

- `runs/<scenario>_runNNN.tsv`: scalar metrics.
- `runs/<scenario>_runNNN_variance.tsv.gz`: per-feature variance fractions, complete data only.
- `summary.tsv`: mean, finite/unavailable counts (`n_runs`, `n_missing`), Monte Carlo
  standard error (`mcse` = sample SD / sqrt(n)) and approximate 95% Student-t mean
  interval (`ci_low`, `ci_high`). MCSE and intervals require at least two finite runs;
  paired intervals summarize run-level differences.
- `equivalence.tsv`: local fedRBE versus central checks.
- `summary.pdf`: recovery, paired differences and selection diagnostics.
- `provenance.txt`: parameters, seeds, code/lock checksums and `sessionInfo()`.

### Limitations

- Ten runs are exploratory; inspect Monte Carlo uncertainty before drawing conclusions.
- Confounding and sample composition change together. Complete confounding is not
  identifiable by linear correction.
- One-/two-lab features are excluded; their correction is not evaluated. Observed-cell
  recovery does not establish robustness to MNAR missingness.
- Added scenarios use only balanced data. Missingness and panel absence are combined,
  so their separate effects and interactions with confounding are not identified.
- The offset model covers raw-scale additive effects, not run-order drift or other
  nonlinear effects.
- There is no DE, feature-selection, classification or clustering benchmark.

### References

1. Burankova et al. (2025). [FedProt](https://doi.org/10.1038/s43588-025-00832-7).
2. Wang et al. (2021). [RobNorm](https://doi.org/10.1093/bioinformatics/btaa904).
3. Johnson et al. (2007). [ComBat](https://doi.org/10.1093/biostatistics/kxj037).
4. Huber et al. (2002). [Variance stabilization and calibration](https://doi.org/10.1093/bioinformatics/18.suppl_1.S96).
5. Jin et al. (2021). [Missing-value imputation in label-free proteomics](https://doi.org/10.1038/s41598-021-81279-4).
6. Schumann et al. (2025). [BERT](https://doi.org/10.1038/s41467-025-62237-4).
