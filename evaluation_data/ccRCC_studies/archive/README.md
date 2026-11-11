# Historical ccRCC preparation

Archived on 16 October 2026. These notebooks preserve source preparation and QC
history, including original cells, saved outputs, and session information.
They were moved unchanged and were not rerun. Saved plots and logs are historical
evidence, not validation against the current environment.

For the current evaluation, follow the [dataset workflow](../README.md), starting
with tracked `before/` inputs, [prepare_ccRCC_data.py](../prepare_ccRCC_data.py),
and [01_central_RBE.ipynb](../01_central_RBE.ipynb).

| Study | Preserved notebooks | Role / original working directory |
| --- | --- | --- |
| PDC000127 | [Preprocess](PDC000127/01_preprocess.ipynb), [QC](PDC000127/02_QC.ipynb) | Current cohort; `data/PDC000127/` |
| PXD030344 | [Preprocess](PXD030344/01_preprocess.ipynb), [QC](PXD030344/02_QC.ipynb) | Current cohort; `data/PXD030344/` |
| PXD042844 | [Preprocess](PXD042844/01_preprocess.ipynb), [QC](PXD042844/02_QC.ipynb) | Current cohort; `data/PXD042844/` |

Original directories are relative to `evaluation_data/ccRCC_studies/`. Raw inputs
and intermediate reports remain there, ignored by Git. Relative `article/` and
report paths in the snapshots refer to those original working directories, not
this archive. The three current cohorts' local `report_filtered.tsv` files were
byte-identical to the corresponding tracked `before/` inputs when archived.

The current-cohort snapshots remain tracked as dataset provenance.

## Limitations before any reuse

- QC notebooks import removed `evaluation_utils/plots/eda_plots.R` and
  `evaluation_utils/filtering/filtering_normalization.R`. They also use obsolete
  APIs (`min_number`, `filter_by_condition`, `boxplot_pg`, and, in label-free
  cohorts, `medianNorm`); changing import paths alone is insufficient.
- PXD042844 expects `article/41467_2023_39981_MOESM4_ESM.xlsx`, which is absent
  locally. Its duplicate-row comment describes quality selection, but the code
  keeps the first row; do not infer a quality-based rule from that comment.
- The snapshots write CSV reports and metadata. They do not record the export
  and encoding step that produced the tracked TSV reports and designs.

Reusing archived preparation would require recovering its inputs and validating
a port to the current helpers.
