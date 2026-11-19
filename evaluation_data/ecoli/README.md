# E. coli dataset

Five E. coli lab datasets from the FedProt project: <https://doi.org/10.48550/arXiv.2407.15220>.

The labs are the batches to correct; the `Glu` and `Pyr` conditions are the
biological signal to preserve. The data originate from DIA-NN outputs, but the
workflow here starts from assembled intensity and metadata tables rather than
downloading or processing raw DIA-NN files.

## Structure

- `before/initial_data/` — source intensity matrix and sample metadata read by
  the preparation notebook.
- `before/central_intensities_log_UNION.tsv` — combined log2 intensity matrix.
- `before/lab_A/` through `before/lab_E/` — per-lab FeatureCloud inputs: intensity
  matrix, covariate design, and configuration. The design encodes `Pyr`, with
  `Glu` as reference.
- `after/label_aware/` and `after/no_label/` — central and federated correction outputs.
- `plots/` — preparation and after-correction diagnostic plots.

The app inputs are already log2-transformed. Disable log transformation during
batch effect correction to avoid transforming them twice.

## Preparation

**When to run:** Use this step to regenerate prepared inputs. If using existing
inputs and correction results, continue with the
[reproduction guide](../../docs/reproduce.md) instead.

**Inputs and prerequisites:** Use the [repository environment](../../environment.yml)
with an R notebook kernel. The notebook reads:

- `before/initial_data/central_batch_info.tsv` — sample, lab, and condition metadata.
- `before/initial_data/central_intensities.tsv` — untransformed intensity matrix.

The `before/lab_*/intermediate/` and `plots/` directories must already
exist; the notebook does not create all required output directories.

**Run:** Execute all cells of
[01_data_prep_and_central_RBE.ipynb](01_data_prep_and_central_RBE.ipynb), with
`evaluation_data/ecoli/` as the working directory.

**What it does:**

1. Align sample columns to metadata, log2-transform intensities, and export the
   combined and per-lab uncorrected matrices. Per-lab exports omit features
   missing from every sample in that lab.
2. For central correction, mask features with fewer than seven observations
   within a lab, then retain features observed in at least three labs. There is
   no separate per-condition filter.

**Outputs:**

- `before/central_intensities_log_UNION.tsv` and
  `before/lab_*/intensities_log_UNION.tsv` — uncorrected log2 matrices.
- `before/lab_*/design.tsv` and `before/lab_*/intermediate/metadata.tsv` — aligned
  per-lab covariates and metadata.
- `before/central_input.tsv` — filtered, uncorrected central input.
- `plots/data_plot.png` — preparation diagnostic plot.

Existing output files are overwritten.

The notebook exports the combined and per-lab uncorrected matrices **before** its
central filtering steps. Regenerating them therefore does not establish that the
central and app inputs have identical feature masks. Check the retained features
when comparing correction results.

## Central correction

After preparation, run from the repository root:

```bash
pixi run python -m evaluation_utils.run_correction_experiments correct --datasets ecoli
```

**Output:** `after/label_aware/intensities_log_Rcorrected_UNION.tsv`.
`pixi run Rscript evaluation/dataset_diagnostics.R ecoli`, run from the repository
root, then writes the after-correction diagnostic plot `plots/data_plot_Rcorrected.png`.
Other correction modes: [reproduction guide](../../docs/reproduce.md).

## Federated correction and downstream evaluation

Follow the [reproduction guide](../../docs/reproduce.md) to run FeatureCloud
correction from the per-lab inputs, or to reuse existing corrected data. The
following files support the downstream analyses:

- [before/datainfo.json](before/datainfo.json) and `after/<arm>/datainfo.json`:
  classification input definitions for the uncorrected and corrected data.

The RF launchers read these descriptors, align each lab's matrix and design by
sample ID, and create sample-by-feature `tmp_data.csv` inputs automatically.
`Pyr` is the prediction target. Corrected descriptors select each lab's
`full_corrected_data.csv` and reuse its `before/lab_*/design.tsv` labels.
