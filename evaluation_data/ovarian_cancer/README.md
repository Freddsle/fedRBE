# Ovarian cancer dataset

Six GEO studies provide 331 arrays: 278 high-grade serous carcinoma (HGSC) and
53 nonmalignant controls. Each study is one batch and one client; `HGSC` is the
classification target and label-aware correction covariate.

| Study | Control tissue | HGSC | Controls | Arrays |
| --- | --- | ---: | ---: | ---: |
| GSE6008 | normal ovary, bulk tissue | 23 | 4 | 27 |
| GSE26712 | ovarian surface epithelium, brushing | 185 | 10 | 195 |
| GSE40595 | ovarian surface epithelium, microdissected | 31 | 6 | 37 |
| GSE69428 | oviduct epithelium, LCM | 10 | 9 | 19 |
| GSE38666 | ovarian surface epithelium, brushing | 18 | 12 | 30 |
| GSE10971 | fallopian tube epithelium, LCM | 11 | 12 | 23 |
| Total | | 278 | 53 | 331 |

## Prepared inputs

The prepared matrices can be used without repeating GEO acquisition or raw-data
preprocessing:

- `before/all_metadata.tsv` — combined sample metadata.
- `before/central_input.zip` — combined expression matrix (`central_input.tsv`).
- `before/<study>/` — per-study `expr_for_correction_UNION.tsv`, `design.tsv`,
  and `config.yml` for the FeatureCloud app.
- `after/<arm>/` — central `central_corrected_UNION.tsv`, federated
  `FedApp_corrected_data.tsv` and `FedApp_corrected_data_smpc.tsv`, and per-study
  outputs under `individual_results/`.

**Disable log2 transformation in the app:** inputs are already RMA-normalized
and log-transformed. The classification loaders read `before/datainfo.json`
and `after/<arm>/datainfo.json`; k-means discovers `before/GSE*` folders containing
`design.tsv`.

## Preparation

Use an R kernel for notebooks 00 and 02, and Python for notebook 01, with
`evaluation_data/ovarian_cancer/` as the working directory. Recreating inputs
requires raw CEL files, platform annotations, and the intermediate matrices read
by the notebooks; use the repository evaluation environment.

| Notebook | Purpose |
| --- | --- |
| [00_harmonize_meta_load_data.ipynb](00_harmonize_meta_load_data.ipynb) | Acquire GEO metadata and CEL files, apply RMA, and collapse probes to GenBank accessions. |
| [01_check_datasets_intersection.ipynb](01_check_datasets_intersection.ipynb) | Inspect feature overlap across studies. |
| [02_prepare_correction_inputs.ipynb](02_prepare_correction_inputs.ipynb) | Select HGSC/control samples, assemble the feature union, and export matrices, metadata and designs. Requires the system `zip` command. |

Notebook 00 reuses downloaded RAW archives and skips studies with an existing
`intermediate/expr_GBacc_MaxVar.tsv`; set `force_rebuild <- TRUE` to rebuild them.
RMA normalizes and log-transforms the data; `WGCNA::collapseRows(maxRowVariance)`
selects a representative probe per GenBank accession. The union retains missing
values for platform-specific features.

For GSE10971, notebook 00 selects 12 FTEn controls (BRCA status unavailable) and
11 HGSC arrays, one per patient, before RMA. It excludes FTEb carrier controls
and retains ovarian-site arrays when both tumour sites are available.

## Correction and evaluation

After preparation, run central correction from the repository root:

```bash
pixi run python -m evaluation_utils.run_correction_experiments correct \
  --datasets ovarian_cancer --arms label_aware no_label
```

Outputs are `after/<arm>/central_corrected_UNION.tsv`. Follow the
[reproduction guide](../../docs/reproduce.md) for federated correction,
classification and k-means. Both RF methods use fedRBE outputs; central k-means
uses the limma reference. The
[evaluation notebook](../../evaluation/evaluation_ovarian_cancer.ipynb) displays
plots and metrics and reads the combined expression ZIP directly.

Changing inputs requires regenerating both correction arms and the uncorrected
and corrected learning results. Use fresh RF outputs and enable
`RERUN_FEDERATED = True` in clustering notebook 05 to regenerate assignments.
Keep each correction arm's outputs separate.

## Metadata

The [cohort table](preprocessing_info/cohort_metadata.tsv),
[count summary](preprocessing_info/selected_cohort_summary.tsv), and
[figure](preprocessing_info/cohort_metadata.png) describe the selected arrays.
Regenerate them from the repository root with Python and matplotlib:

```bash
python evaluation_data/ovarian_cancer/preprocessing_info/build_cohort_metadata.py
```

The builder combines the [source clinical inventory](preprocessing_info/source_cohort_metadata.tsv)
with harmonized study metadata and checks sample IDs and HGSC labels against
the descriptors and designs. Annotations are not imputed; `age_source`,
`grade_source` and `stage_source_discrepancy` record provenance and disagreements.
`NA`/`Missing` means unavailable; `Control: N/A` means a tumour endpoint is
undefined. `tissue` and `brca_status` retain the available source annotations.

Counts represent arrays, not necessarily independent patients. Donor IDs are
study-local; GSE69428 includes paired tumour/control samples, and six GSE40595
control specimen names also occur in GSE26712.

## Intermediate matrices

Rows are GenBank-accession features in
`before/<study>/intermediate/expr_GBacc_MaxVar.tsv`. Samples are selected at
different preprocessing stages; these dimensions differ from the final cohort
counts above. The [intake summary](preprocessing_info/metadata_summary.tsv)
records the harmonized source metadata.

| Study | Features | Arrays |
| --- | ---: | ---: |
| GSE6008 | 21,128 | 103 |
| GSE26712 | 21,128 | 195 |
| GSE40595 | 51,276 | 38 |
| GSE69428 | 51,276 | 20 |
| GSE38666 | 51,276 | 30 |
| GSE10971 | 51,276 | 23 |
