# E. coli multi-batch dataset (Quartet, PXD045065)

Quartet reference materials ecoli data (balanced design) for evaluating batch-effect correction across multiple labs with internal batches.

- **Dataset:** ProteomeXchange [PXD045065](http://proteomecentral.proteomexchange.org)
- **Processed data:** Figshare [10.6084/m9.figshare.29567333.v2](https://doi.org/10.6084/m9.figshare.29567333.v2)
- **Paper:** Chen et al., Nature Communications (2025), https://www.nature.com/articles/s41467-025-64718-y

## Data overview

Protein-level MaxLFQ intensities (log2-transformed) from the balanced Quartet design.

| Center | Batches          | # batches | Samples per batch |
|--------|------------------|-----------|-------------------|
| APT    | DDA_APT, DIA_APT | 2         | 12                |
| FDU    | DDA_FDU, DIA_FDU | 2         | 12                |
| NVG    | DDA_NVG          | 1         | 12                |
| BGI    | DIA_BGI          | 1         | 12                |

- 4 centers (labs), 6 batches total, 72 samples.
- 4 biological conditions (sample types): D5, D6, F7, M8 — 3 replicates each per batch.

## Notebooks

| Notebook | Language | Purpose |
|----------|----------|---------|
| `01_data_prep_RBE.ipynb` | R | Data loading, filtering, normalization, central batch correction, saving per-center inputs |
| `02_run_fedrbe.ipynb` | Python | Run FedRBE via FeatureCloud Docker (non-SMPC and SMPC), diagnostic comparison |

## Preprocessing and central correction (01_data_prep_RBE.ipynb)

### 1. Load raw data

Read `data_paper/expdata_log_noNA.csv` (protein intensities) and `data_paper/meta.csv` (sample annotations). Replace sentinel values (`-1e6`) with `NA`. Remove all-NA rows.

### 2. Per-center union

For each center, drop protein rows that are entirely NA within that center, then merge all centers by feature union (outer join). Features absent from a center get `NA`.

### 3. Filtering

- **Per-batch:** Within each batch, if a feature has only 1 non-NA sample, set those values to `NA` (`min_samples=2, drop_row=FALSE`).
- **Per-center:** Within each center, if a feature has fewer than 10 non-NA values, set all that center's values for the feature to `NA` (`min_samples=10, drop_row=FALSE`).
- Remove any features that became all-NA globally after filtering.
- **Cross-center:** Remove features present in fewer than 3 centers after filtering.

### 4. Median normalization

Per center, shift each sample's intensities so that the sample median equals the center's global median of medians. `NA` values are preserved.

### 5. Central batch correction

`limma::removeBatchEffect` on the normalized union matrix:

- **Batch factor:** 6 batches — DDA_APT, DIA_APT, DDA_FDU, DIA_FDU, DDA_NVG, DIA_BGI (reference = DIA_BGI, the first level in the factor).
- **Covariates:** D6, F7, M8 (with D5 as factor reference, i.e., intercept).
- **NA handling:** preserved; limma fits per-feature models excluding NA samples, then sets missing betas to 0.

### 6. Save outputs

- Per-center input files to `before/{APT,FDU,NVG,BGI}/` (intensities, design matrix, config).
- Initial unfiltered data to `before/00_initial_data/`.
- Centrally corrected result to `after/intensities_log_Rcorrected_UNION.tsv`.
- Diagnostic plots to `plots/`.

## Running FedRBE (02_run_fedrbe.ipynb)

1. Defines the experiment for 4 centers with per-center `config.yml` overrides.
2. Runs FedRBE via the FeatureCloud controller in two modes:
   - **Non-SMPC** → `after/FedApp_corrected_data.tsv`
   - **SMPC** → `after/FedApp_corrected_data_smpc.tsv`
3. Extracts per-client results to `after/individual_results/{APT,FDU,NVG,BGI}/`.
4. Runs a diagnostic simulation comparing central R correction, local numpy simulation, and actual Docker FedRBE output.

## Covariate encoding

Design matrix with D5 as reference level:

| Column | Value |
|--------|-------|
| D6     | 1 if sample is D6, else 0 |
| F7     | 1 if sample is F7, else 0 |
| M8     | 1 if sample is M8, else 0 |

## Folder structure

```
quartet/
├── data_paper/                        # Source CSV files from the original publication
│   ├── expdata_log_noNA.csv           #   Protein intensities (features × samples, -1e6 = NA)
│   └── meta.csv                       #   Sample annotations (run_id, mode, lab, batch, sample, ...)
├── before/
│   ├── 00_initial_data/               # Unfiltered data (before any filtering)
│   │   ├── central_batch_info.tsv
│   │   ├── central_intensities.tsv
│   │   └── central_intensities_log_UNION.tsv
│   ├── APT/                           # Per-center input for FedRBE
│   │   ├── intensities_log_UNION.tsv  #   Filtered + normalized log2 intensities
│   │   ├── design.tsv                 #   Sample design (file, D6, F7, M8, batch, condition)
│   │   └── config.yml                 #   FedRBE config (batch_col, covariates, position, ...)
│   ├── FDU/                           # Same structure
│   ├── NVG/                           # Same structure
│   └── BGI/                           # Same structure (reference_batch: true)
├── after/
│   ├── intensities_log_Rcorrected_UNION.tsv   # Central limma correction
│   ├── FedApp_corrected_data.tsv              # FedRBE non-SMPC result
│   ├── FedApp_corrected_data_smpc.tsv         # FedRBE SMPC result
│   └── individual_results/                    # Per-center FedRBE outputs
│       ├── APT/
│       ├── FDU/
│       ├── NVG/
│       └── BGI/
├── plots/
│   ├── data_plot.png                  # Before correction
│   └── data_plot_Rcorrected.png       # After central correction
├── 01_data_prep_RBE.ipynb             # R: data prep + central correction
├── 02_run_fedrbe.ipynb                # Python: FedRBE execution + diagnostics
└── README.md
```

## FedRBE app configuration

Each center's `config.yml` specifies:
- `batch_col: batch` — all centers use a `batch` column in their design file. Single-batch centers (NVG, BGI) have one batch name for all their samples.
- `covariates: [D6, F7, M8]`
- `reference_batch: true` only for BGI (the last center by position).
- `position: 0..3` — determines client ordering and which client is coordinator.

## Citation

Chen, Q. All corrected/uncorrected data matrices at precursor, peptide, protein levels. https://doi.org/10.6084/m9.figshare.29567333.v2 (2025).
