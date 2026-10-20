# Ovarian cancer dataset

This dataset combines six GEO studies: `GSE6008`, `GSE26712`, `GSE40595`, `GSE69428`,
`GSE38666`, and `GSE14407`. The current evaluation compares high-grade serous
carcinoma (HGSC) with controls, using 332 retained arrays: 279 HGSC and 53 controls.
Each study is treated as a batch.

## Using the prepared inputs

The prepared matrices can be used without repeating GEO acquisition and raw-data
preprocessing:

- `before/` contains combined metadata in `all_metadata.tsv` and the expression
  matrix, also the central correction input, in `central_input.zip` (containing
  `central_input.tsv`), plus one folder per study. Each study folder provides
  `expr_for_correction_UNION.tsv` and `design.tsv` for the FeatureCloud app;
  `HGSC` is the biological covariate in the design.
- `after/label_aware/central_corrected_UNION.tsv` is the central limma reference. Federated
  results are stored in study-named folders under `after/label_aware/individual_results/`.
- For app runs, **disable log2 transformation**: the expression data have already
  been normalized and log-transformed with RMA.
- [evaluation_ovarian_cancer.ipynb](../../evaluation/evaluation_ovarian_cancer.ipynb)
  contains evaluation plots and metrics for the prepared data and corrections.
  It reads the combined expression ZIP directly if the TSV is not extracted.

For classification, [before/datainfo.json](before/datainfo.json) and each
correction arm's `after/<arm>/datainfo.json` identify the expression matrices,
study folders, designs, and HGSC prediction target. The classification loaders
read these configurations and create their own temporary inputs. The former
standalone CSV-export script is no longer a preparation step; its
[archive note](archive/README.md) explains the replacement and how to recover the
original source.

## Source preprocessing and central correction

The notebooks below document how the prepared inputs were created. Their relative
paths assume `evaluation_data/ovarian_cancer/` as the working directory. Recreating
the inputs requires the raw data, platform annotations, and intermediate files
read by these notebooks; prepared app inputs alone are not sufficient.

| Notebook | Purpose |
| --- | --- |
| [00_harmonize_meta_load_data.ipynb](00_harmonize_meta_load_data.ipynb) | Harmonize GEO metadata, acquire raw `.CEL` files, apply RMA, and collapse probes to GenBank accessions. |
| [01_check_datasets_intersection.ipynb](01_check_datasets_intersection.ipynb) | Inspect the overlap of GenBank accessions across studies. |
| [02_central_RBE.ipynb](02_central_RBE.ipynb) | Select HGSC/control samples from intermediate data, assemble the feature union, and export combined, central and per-study inputs and designs. Zipping `central_input.zip` uses the system `zip` command. |

Raw `.CEL` files come from [GEO](https://www.ncbi.nlm.nih.gov/geo/). RMA performs
normalization and log transformation; `WGCNA::collapseRows(maxRowVariance)` then
selects a representative probe for each GenBank accession. Expression matrices
contain no NA values, although auxiliary metadata may be missing.

After input preparation, run central correction from the repository root:

```bash
pixi run python -m evaluation_utils.run_correction_experiments correct --datasets ovarian_cancer
```

Output: `after/label_aware/central_corrected_UNION.tsv`. The evaluation notebook then writes the
after-correction diagnostic plot `before/plots/norm_vs_tumor_plot_Rcorrected_UNION.png`.

## Cohort metadata and interpretation

The final [cohort table](preprocessing_info/cohort_metadata.tsv) and
[figure](preprocessing_info/cohort_metadata.png) describe the retained arrays.
They were checked on 16 October 2026 against sample designs and clinical sources;
the figure was reproduced exactly. Sources were GEO annotations, GSE14407 paper
Table 1 (PMCID PMC2806370), and full curatedOvarianData GSE26712 (Zenodo 23125873),
joined by exact GSM/OCI identifiers. No metadata were imputed.

Read the clinical fields as follows:

- Age is available for 236 arrays, with source-specific definitions.
- `NA`/`Missing` means unavailable; `Control: N/A` means a tumor endpoint is
  undefined. `Conflict` marks two disputed early/late stage labels.
- Survival codes are DOD (dead of disease), AWD (alive with disease), and NED
  (no evidence of disease).

**Counts represent arrays, not unique patients.** Donor IDs are study-local. The
shared-donor flag covers only GSE14407/GSE38666, with 23 shared IDs; one donor's
recorded ages differ by a year. A false flag does not establish independence.
GSE69428 has nine tumor/control pairs, and six GSE40595 control specimen names
also occur in GSE26712.

The retained metadata audit artifacts are the final cohort PNG/TSV above.

## Source-input dimensions

These dimensions precede the final HGSC/control selection and therefore do not
describe the 332-array evaluation cohort above.

| Dataset | Rows | Columns |
| --- | ---: | ---: |
| GSE6008 | 21,128 | 103 |
| GSE26712 | 21,128 | 195 |
| GSE40595 | 21,128 | 37 |
| GSE69428 | 21,128 | 19 |
| GSE38666 | 21,128 | 30 |
| GSE14407 | 21,128 | 24 |
