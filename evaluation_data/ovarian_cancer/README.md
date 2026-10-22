# Ovarian cancer dataset

The selected evaluation combines five GEO studies: `GSE6008`, `GSE26712`,
`GSE40595`, `GSE69428`, and `GSE38666`. It compares high-grade serous carcinoma
(HGSC) with controls, using 308 arrays: 267 HGSC and 41 controls. Each selected
study is one batch and one client. GSE14407 is excluded because all 24 of its raw
CEL measurements also occur in GSE38666; all 30 previously selected GSE38666
arrays are retained. The existing HGSC/control eligibility criteria are unchanged.

| Selected study | HGSC | Controls | Arrays |
| --- | ---: | ---: | ---: |
| GSE6008 | 23 | 4 | 27 |
| GSE26712 | 185 | 10 | 195 |
| GSE40595 | 31 | 6 | 37 |
| GSE69428 | 10 | 9 | 19 |
| GSE38666 | 18 | 12 | 30 |
| Total | 267 | 41 | 308 |

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
read these configurations and create their own temporary inputs. The descriptors
and the ovarian correction experiment select the same five cohorts. K-means
preparation discovers the five `before/GSE*` folders containing `design.tsv`;
the excluded GSE14407 dataset folder is absent. The former
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
| [02_prepare_correction_inputs.ipynb](02_prepare_correction_inputs.ipynb) | Select HGSC/control samples from intermediate data, assemble the feature union, and export combined, central and per-study inputs and designs. Zipping `central_input.zip` uses the system `zip` command. |

Raw `.CEL` files come from [GEO](https://www.ncbi.nlm.nih.gov/geo/). RMA performs
normalization and log transformation; `WGCNA::collapseRows(maxRowVariance)` then
selects a representative probe for each GenBank accession. Intermediate per-study
expression matrices contain no NA values; prepared union inputs contain missing
values for platform-specific features. Auxiliary
clinical metadata may also be missing.

After input preparation, run central correction from the repository root:

```bash
pixi run python -m evaluation_utils.run_correction_experiments correct \
  --datasets ovarian_cancer --arms label_aware no_label
```

Outputs: `after/<arm>/central_corrected_UNION.tsv` for both arms. The evaluation notebook
uses label-aware correction and writes the
after-correction diagnostic plot `before/plots/norm_vs_tumor_plot_Rcorrected_UNION.png`.

## Refreshing downstream results

After changing the selected cohorts, regenerate both correction arms before
rerunning downstream analysis. Use fresh RF result files and new federated k-means
assignment folders; old results can retain removed cohorts even when input
selection changes. Rerun the uncorrected baseline as well as both corrected
states, including all held-out cohorts and the train/test evaluation.

The [reproduction guide](../../docs/reproduce.md) documents the ordinary RF
launchers (`--datasets ovarian_cancer`, seeds 42–51) and clustering notebook 05
(`DATASETS = ["ovarian_cancer"]`, seeds 1, 11, ..., 91). Enable
`RERUN_FEDERATED = True` when regenerating clustering assignments. Select each
correction arm and retain separate outputs. Both RF methods use fedRBE outputs;
central k-means uses the central limma reference.

## Duplicate-measurement exclusion

The [raw CEL pair mapping](preprocessing_info/duplicate_cel_pairs.tsv) records
24 excluded GSE14407 accessions and their retained GSE38666 counterparts, with
source URLs, decompressed byte counts, SHA-256 hashes, and direct byte-comparison
results. The [audit record](preprocessing_info/duplicate_cel_audit.json) describes
the check completed on 21 October 2026: all 24 pairs are byte-for-byte identical
after gzip decompression, representing 24 distinct CEL measurements among 48
GEO files. The gzip wrappers can differ, so compressed-file hashes are not the
identity criterion.

This establishes duplicate measurements, without requiring independent clinical
identity verification. It includes GSM360047/GSM947309, whose reported identifiers
are `588`/`5883`; the earlier 23 matching identifiers omitted this pair. The
selection retains the complete eligible 30-array GSE38666 subset and does not depend on
classification performance. The excluded accession remains part of the source
inventory, and its provenance and clinical annotations are preserved. Its
expression matrices, app configuration and preparation files are not included.

To verify a pair, download its two source URLs, decompress each gzip stream,
compare the complete CEL bytes, and check the common SHA-256 and byte count
against the mapping. Check all 24 rows before reusing this exclusion in a new
input build.

## Cohort metadata and interpretation

The selected [cohort table](preprocessing_info/cohort_metadata.tsv),
[count summary](preprocessing_info/selected_cohort_summary.tsv), and
[figure](preprocessing_info/cohort_metadata.png) describe the five-study,
308-array selection. Regenerate them from the repository root with the existing
Python environment containing matplotlib:

```bash
python evaluation_data/ovarian_cancer/preprocessing_info/build_cohort_metadata.py
```

The builder checks selected sample IDs and HGSC labels against the active
descriptor and per-study designs. It reads the preserved 332-array
[source clinical inventory](preprocessing_info/source_cohort_metadata.tsv) and
the raw CEL pair mapping. The source clinical inventory is a curated snapshot
checked on 16 October 2026 against GEO annotations, GSE14407 paper Table 1
(PMCID PMC2806370), and full curatedOvarianData GSE26712 (Zenodo 23125873), joined
by GSM/OCI identifiers. Its original generating scripts were not retained.
The selected outputs preserve the available annotations without imputation;
`age_source`, `grade_source`, and `stage_source_discrepancy` identify annotation
sources and disagreements. `excluded_duplicate_sample` links a retained array
to its excluded raw-identical counterpart; `NA` means no counterpart in this
mapping.

Read the clinical fields as follows:

- Age is available for 212 selected arrays, with source-specific definitions.
- `NA`/`Missing` means unavailable; `Control: N/A` means a tumor endpoint is
  undefined. The historical source inventory also records disputed stage labels.
- Survival codes are DOD (dead of disease), AWD (alive with disease), and NED
  (no evidence of disease).

**Counts represent arrays, not unique patients.** Donor IDs remain study-local;
the raw CEL audit only establishes the documented duplicate-measurement pairs.
GSE69428 has nine tumor/control pairs, and six GSE40595 control specimen names
also occur in GSE26712. The exclusion does not establish independence of every
remaining biological specimen.

## Source-input dimensions

These source dimensions precede HGSC/control selection and duplicate exclusion;
they do not describe the 308-array evaluation cohort. The retained
[source-intake summary](preprocessing_info/metadata_summary.tsv) includes all six
acquired accessions, including excluded GSE14407. It is a source inventory,
not the selected cohort summary. Rows below are GenBank-accession features in
`before/<study>/intermediate/expr_GBacc_MaxVar.tsv`; the GSE14407 dimensions are
historical measurements recorded before its dataset files were removed.

| Dataset | Rows | Columns |
| --- | ---: | ---: |
| GSE6008 | 21,128 | 103 |
| GSE26712 | 21,128 | 195 |
| GSE40595 | 51,276 | 38 |
| GSE69428 | 51,276 | 20 |
| GSE38666 | 51,276 | 30 |
| GSE14407 (excluded) | 51,276 | 24 |
