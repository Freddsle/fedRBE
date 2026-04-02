# Proteomics multi-batch dataset (Quartet, PXD045065)

Quartet reference materials proteomics data (balanced design) for evaluating batch-effect correction across multiple labs with internal batches.

- **Dataset:** ProteomeXchange [PXD045065](http://proteomecentral.proteomexchange.org)
- **Processed data:** Figshare [10.6084/m9.figshare.29567333.v2](https://doi.org/10.6084/m9.figshare.29567333.v2)
- **Paper:** Chen et al., Nature Communications (2025), https://www.nature.com/articles/s41467-025-64718-y

## Data overview

Protein-level MaxLFQ intensities (log2-transformed) from the balanced Quartet design.

| Center | Batches        | # batches | Samples per batch |
|--------|----------------|-----------|-------------------|
| APT    | DDA_APT, DIA_APT | 2       | 12                |
| FDU    | DDA_FDU, DIA_FDU | 2       | 12                |
| NVG    | DDA_NVG          | 1       | 12                |
| BGI    | DIA_BGI          | 1       | 12                |

- 4 centers (labs), 6 batches total, 72 samples.
- 4 biological conditions (sample types): D5, D6, F7, M8 — 3 replicates each per batch.

## Preprocessing (01_data_prep_RBE.ipynb)

1. **Load:** Read `data_paper/expdata_log_noNA.csv` and `data_paper/meta.csv`. Replace sentinel values (-1e6) with NA. Remove all-NA rows.
2. **Per-center filtering:** For each center, remove features with fewer than 2 non-NA values per center and per condition class (`filter_per_center`).
3. **Additional filtering:** Remove features with fewer than 7 non-NA values per center; remove features present in fewer than 3 centers.
4. **Median normalization:** Per center, shift each sample's intensities so that sample medians equal the center's global median.

## Central correction

- `limma::removeBatchEffect` on the union of all centers.
- Batch factor: 6 batches (DDA_APT, DIA_APT, DDA_FDU, DIA_FDU, DDA_NVG, DIA_BGI).
- Covariates: D6, F7, M8 (D5 as reference).
- NA values are preserved (not imputed).

## Covariate encoding

Design matrix with D5 as reference level:

| Column | Value |
|--------|-------|
| D6     | 1 if sample is D6, else 0 |
| F7     | 1 if sample is F7, else 0 |
| M8     | 1 if sample is M8, else 0 |

## Folder structure

```
proteomics_multibatch/
├── data_paper/                    # Source CSV files
│   ├── expdata_log_noNA.csv       #   Protein intensities (features × samples)
│   └── meta.csv                   #   Sample annotations (run_id, mode, lab, batch, sample, ...)
├── before/
│   ├── 00_initial_data/           # Unfiltered central data
│   │   ├── central_batch_info.tsv
│   │   ├── central_intensities.tsv
│   │   └── central_intensities_log_UNION.tsv
│   ├── APT/                       # One folder per center
│   │   ├── intensities_log_UNION.tsv    # Filtered log2 intensities
│   │   ├── design.tsv                   # file, D6, F7, M8, batch, condition
│   │   └── config.yml                   # batch_col: batch (2 internal batches)
│   ├── FDU/                       # Same structure; batch_col: batch
│   ├── NVG/                       # Same structure; batch_col: null (1 batch)
│   └── BGI/                       # Same structure; batch_col: null (1 batch)
├── after/
│   ├── intensities_log_Rcorrected_UNION.tsv   # Centrally corrected
│   ├── FedApp_corrected_data.tsv              # Federated corrected
│   └── FedApp_corrected_data_smpc.tsv         # Federated corrected (SMPC)
├── plots/
├── 01_data_prep_RBE.ipynb         # Data preparation notebook
└── README.md
```

## FedRBE app configuration

Each center's `config.yml` specifies:
- `batch_col: batch` for centers with multiple internal batches (APT, FDU).
- `batch_col: null` for single-batch centers (NVG, BGI).
- Covariates: D6, F7, M8.
- Data/design filenames and separators.

## Citation

Chen, Q. All corrected/uncorrected data matrices at precursor, peptide, protein levels. https://doi.org/10.6084/m9.figshare.29567333.v2 (2025).