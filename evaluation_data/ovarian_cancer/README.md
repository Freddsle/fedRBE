# Ovarian cancer dataset

Six GEO datasets: `GSE6008`, `GSE26712`, `GSE40595`, `GSE69428`, `GSE38666`, and `GSE14407`.

Preprocessing:
- Raw `.CEL` files come from <https://www.ncbi.nlm.nih.gov/geo/>.
- Expression data are normalized and log-transformed with RMA.
- Rows are collapsed to GenBank accession with `WGCNA::collapseRows(maxRowVariance)`.

The data do not contain NA values.

Central run:  
- Runs limma RBE with `Status` (`normal` / `tumor`) as covariate and dataset as batch.
- Auxiliary metadata fields may contain NA values; the central correction uses `Status` and dataset.

# Structure

`before/` contains both combined and app-ready inputs:
- Combined TSV files for all datasets (`all_metadata` and `all_expression`).
- One folder per dataset for the FeatureCloud app. Each folder contains data and a design file with covariates.

For app runs, log2 transformation must be disabled.

Main files:
- `00_harmonize_meta_load_data.ipynb`: metadata harmonization and raw data loading.
- `01_check_datasets_intersection.ipynb`: feature-intersection checks.
- `02_central_RBE.ipynb`: central correction.
- `prepare_data.py`: app-ready folder preparation.
- `../../evaluation/evaluation_ovarian_cancer.ipynb`: evaluation plots and metrics.


Info (rows x cols):
```
[1] "GSE6008"
[1] 21128   103
[1] "GSE26712"
[1] 21128   195
[1] "GSE40595"
[1] 21128    37
[1] "GSE69428"
[1] 21128    19
[1] "GSE38666"
[1] 21128    30
[1] "GSE14407"
[1] 21128    24
```
