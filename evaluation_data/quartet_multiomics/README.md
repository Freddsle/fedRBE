# Quartet multiomics batch-effect dataset

Figshare dataset for the Quartet batch-effect correction study:

- **Dataset:** [10.6084/m9.figshare.22188349.v1](https://doi.org/10.6084/m9.figshare.22188349.v1), CC BY 4.0
- **Batch-effect paper:** Yu et al., Genome Biology 24, 201 (2023), https://doi.org/10.1186/s13059-023-03047-z
- **Quartet Data Portal paper:** Yang et al., Genome Biology 24, 245 (2023), https://doi.org/10.1186/s13059-023-03091-9

The data use Quartet reference materials from four matched B-lymphoblastoid
cell lines: **D5**, **D6**, **F7**, and **M8** (Yang et al. 2023). Measurements
across transcriptomics, proteomics, and metabolomics allow correction to be
compared within each modality and clustering to be evaluated across modalities.
Following the source release, `D6` is the common reference donor and the limma
intercept in notebook 03; `D5/F7/M8` are the study groups. Preserve downloaded
`figshare_data/` inputs so the preparation can be traced back to the source.

## Main matrices

The preparation starts from the three full matrices below. Rows are features
and columns are measurement libraries; each batch contains all four donors with
three replicates each. Notebook 01 selects the libraries used in the federation,
so the full source dimensions below differ from the prepared sample counts.

| Omics | Full matrix | Features | Samples | Scale |
|-------|-------------|----------|---------|-------|
| Transcriptomics | `Transcriptomics_fulldataset_log2FPKM_r26907c180.csv` | 26,907 genes | 180 | log2(FPKM + 0.01) |
| Proteomics | `Proteomics_fulldataset_log2FOT_r3489c180.csv` | 3,489 proteins | 180 | log2(FOT + 0.01) |
| Metabolomics | `Metabolomics_fulldataset_log2expr_r71c180.csv` | 71 metabolites | 180 | log2(expr + 1) |

Each full matrix has **15 batches x 12 libraries** per omics type.

## Optional source exports (not required by the pipeline)

The Figshare release also provides balanced and confounded subsets, each with
45 study samples. This preparation selects batches from the full matrices
instead, so these optional exports and their scenario metadata are not required
and are ignored by Git:

| Omics | Balanced subset | Confounded subset |
|-------|-----------------|-------------------|
| Transcriptomics | `Transcriptomics_Balanced_log2FPKM_r26907c45.csv` | `Transcriptomics_Confounded_log2FPKM_r26907c45.csv` |
| Proteomics | `Proteomics_Balanced_log2FOT_r3489c45.csv` | `Proteomics_Confounded_log2FOT_r3489c45.csv` |
| Metabolomics | `Metabolomics_Balanced_log2expr_r71c45.csv` | `Metabolomics_Confounded_log2expr_r71c45.csv` |

- **Balanced:** one D5/F7/M8 replicate from each of 15 batches.
- **Confounded:** five batches assigned to each D5/F7/M8 group, with all three replicates retained.
- Scenario metadata also lists 45 D6 reference samples per omics/scenario, but D6 columns are not present in the subset matrices.

## Metadata

| File | Purpose |
|------|---------|
| `meta_full_dataset_3omics.csv` | Full metadata: 540 rows, 180 libraries per omics type. Key columns: `library`, `sample`, `batch`, `lab`, `platform`, `protocol`, `datatype`, `rep`, `date`. |
| `meta_balancedconfounded_dataset_3omics.xlsx` | Scenario metadata: 540 rows, with study/reference role and scenario labels. |

- Use `sample` / `Sample group` as biological condition and `batch` as the batch factor.
- `lab`, `platform`, and `protocol` are useful EDA covariates or possible site/covariate candidates.
- Proteomics subset matrix columns use periods in instrument tokens such as `QE.HFX`, `QE.Plus`, and `QE.HF`, while metadata uses hyphens. The notebook normalizes this before joining.

## Notebooks (run in order)

Use an R kernel for notebooks 01–03, with their imported packages installed. Run from the repository root; the dataset directory also works
once `figshare_data/` exists. Notebook 01 needs network access only for missing
downloads; the FeatureCloud requirements are listed below.

1. `01_preprocess_eda.ipynb` — downloads the required Figshare files when missing, loads the full matrices, applies lab/client selection, writes `filtered_data/`, and saves before-correction PCA / boxplot / density plots to `plots/`. Selected via `DATASET_ID` and `DATASET_IDS`.
2. `02_prepare_RBE_inputs.ipynb` — builds the three- or four-client federation
   described below. It writes the central matrix and client inputs for each
   modality, together with metadata and `before/fedrbe_client_groups.tsv`.
3. `03_central_RBE.ipynb` — checks prepared inputs, saves before-correction
   diagnostics, and copies each modality's central matrix to
   `before/<Modality>/central_input.tsv`. Then run central correction from the
   repository root:

   ```bash
   pixi run python -m evaluation_utils.run_correction_experiments correct --datasets quartet_multiomics
   ```

   Output: `after/label_aware/<Modality>/intensities_log_Rcorrected_UNION.tsv`.
   Sample metadata is read from `before/<Modality>/metadata.tsv`.
4. Federated correction — run `generate_fedrbe_corrected_datasets.py` from the repository root as described in the [reproduction guide](../../docs/reproduce.md#2-obtaining-federated-corrected-data). It reads the client folders prepared by notebook 02 and writes `after/label_aware/<Modality>/FedApp_corrected_data.tsv` and `FedApp_corrected_data_smpc.tsv`.
5. Downstream evaluation — use the ordinary RF launchers and k-means script
   described in the [reproduction guide](../../docs/reproduce.md#5-reproduce-the-classification-analysis-comparing-fedrbe-corrected-to-uncorrected-data).
   They prepare joint matrices and RF client files automatically for the selected
   correction arm from the modality outputs and original metadata.
   `05_create_merged_omics_folder.py --correction-arm <arm>` exposes the same RF
   exporter independently when joint matrices already exist.

`pixi run Rscript evaluation/dataset_diagnostics.R quartet_multiomics`, run from the
repository root, saves the after-correction counterparts of notebook 03's diagnostics
to `plots/`. The separate script below exports the combined per-modality diagnostic
figure.

The [S5 script](../../evaluation/quartet_separated_modality_pca_fixed_effect_variance.R)
runs after federated correction, using the default layout of 48 profiles and four
batches across three clients per modality. Set `CORRECTION_ARM=label_aware`
(default) or `no_label`. Its sequential model is `~ condition + batch` (donor first);
PCA batch codes are modality-specific. See the
[run command and outputs](../../docs/reproduce.md#7-generate-figures-3-4-s5-and-s7-for-each-correction-arm).

## Directory orientation

The notebook sequence separates downloaded inputs, selected libraries, and
correction inputs/results. Use this overview to locate a stage; the detailed
output layout below lists the files generated by later steps.

```text
quartet_multiomics/
├── figshare_data/       # Source matrices and metadata; preserve the downloads
├── filtered_data/      # Selected libraries and aligned metadata from notebook 01
├── before/             # Shared central and client correction inputs from notebook 02
├── after/              # label_aware/ and no_label/ correction arms, then modality
├── plots/              # Exploratory and correction diagnostic plots
├── 01_preprocess_eda.ipynb
├── 02_prepare_RBE_inputs.ipynb
└── 03_central_RBE.ipynb
```

## Client federation (notebook 02)

Figshare lab IDs (`L01`..`L15`) are enumerated **independently within each
modality**: the same ID can denote different institutions across modalities.
The federation is constructed so that each client has data in all three
modalities. It is a grouping for the evaluation, rather than evidence that each
client represents one institution providing matched measurements across omics.

- `L01`, `L02`, and `L05` occur in all three modalities and form the three
  default client groups.
- Optional synthetic client 4 combines `L03` (Metab+RNA) and `L14` (Protein), the
  unique Protein-only lab with two batches that pair with L03.
- Batches are selected so each client contributes the same number of libraries
  in every modality: 12 or 24. The table below records the exact batch membership.

### `INCLUDE_CLIENT_04` toggle

The optional fourth client, `client_04_L03_L14`, is **disabled by default**.
The default run therefore uses three clients. Keep `INCLUDE_CLIENT_04`
synchronized across the federated launcher and R notebooks so preparation,
correction, and expected sample counts refer to the same layout:

- `INCLUDE_MULTIOMICS_CLIENT_04` in `generate_fedrbe_corrected_datasets.py`.
- `INCLUDE_CLIENT_04` in `01_preprocess_eda.ipynb`, `02_prepare_RBE_inputs.ipynb` and `03_central_RBE.ipynb`.

Defaults (`INCLUDE_CLIENT_04 = False`):

- **3 clients per modality, 48 libraries across 4 batches per modality.**
- No client-4 folder, metadata, design, config, or corrected output is written.
  Its six batches (listed below) remain in the full R `ALL_*` registries.
- Central limma and FedRBE use the alphabetically last batch of `client_03_L05_L04`
  as reference: `R_ILM_L5_B2` (RNA), `FDU_QE-HFX_4` (Protein), `U_L5_01` (Metab).

Set `INCLUDE_MULTIOMICS_CLIENT_04 = True` in the launcher and `INCLUDE_CLIENT_04 <- TRUE` in all three R notebooks to enable the full **4-client / 72-library / 6-batch** federation. The reference batch then becomes the last batch of `client_04_L03_L14` (`R_BGI_L3_B1` / `TMO_QE-HFX_1` / `U_L3_02`).

### Per-client batch contributions (full federation)

| client | RNA batches | Protein batches | Metabolomics batches |
|---|---|---|---|
| `client_01_L01` | `P_ILM_L1_B1` | `ABS_QTOF6600_1` | `U_L1_01` |
| `client_02_L02` | `P_ILM_L2_B1` | `APT_QE-HFX_1` | `U_L2_01` |
| `client_03_L05_L04` | `P_ILM_L5_B1`, `R_ILM_L5_B2` | `FDU_Lumos_1`, `FDU_QE-HFX_4` | `U_L5_01`, `T_L4_01` |
| `client_04_L03_L14` *(optional)* | `P_BGI_L3_B1`, `R_BGI_L3_B1` | `TMO_Exploris480_1`, `TMO_QE-HFX_1` | `U_L3_01`, `U_L3_02` |

When a lab has more batches than the chosen layout needs, the following
selection rules keep the preparation reproducible and explain why particular
batches are retained:

- L01 / L02 RNA: keep the poly(A)-mRNA batch (`P_*`) and drop the rRNA-depleted batch (`R_*`); poly(A)-mRNA matches the layer measured by proteomics/metabolomics.
- L05 Proteomics: keep `FDU_Lumos_1` and `FDU_QE-HFX_4`; drop `FDU_QE-HFX_1`. Among the two QE-HFX runs, `_4` has a much lower fraction of cells at the `−6.644` imputation floor (8.4% vs 16.7%). The value `−6.644 = log2(0.01)` is the most frequent value in the source FOT matrix (~19% of all cells) and corresponds to the `log2(FOT + 0.01)` floor used by the figshare release for non-detected proteins.
- L05 Metabolomics: add `T_L4_01` (alphabetical-first L04 batch) so the client reaches 24 libraries, matching its RNA / Protein size. The original `lab="L04"` label is preserved in metadata.
- L03 + L14 (Proteomics): L14 is the unique Protein-only lab with exactly two batches.

The selected batches and their client assignments are declared in R. If source
data or the selection changes, also update the client names and reference batches
listed for Quartet in `generate_fedrbe_corrected_datasets.py`:

- R: identical `ALL_SELECTED_BATCHES`, `ALL_CLIENT_LAB_MAPS`, `ALL_CLIENT_NAMES` lists declared at the top of `01_preprocess_eda.ipynb`, with the matching `INCLUDE_CLIENT_04` toggle. `02_prepare_RBE_inputs.ipynb` and `03_central_RBE.ipynb` carry the same toggle so the validation constants (`N_LIBS_PER_MODALITY`, `N_BATCHES_PER_MODALITY`, `CLIENT_NAMES`) stay in sync.

## Joint sample identifiers

The three modalities do not share library identifiers, so notebook 02 cannot
join them directly by the original column names. Instead, it constructs
synthetic joint sample identifiers within each client, donor, and replicate
group for the all-modality k-means matrices:

```
pseudo_sample = "{client}_{donor}_{rep}_{i}"
```

`i = 1..k` indexes libraries within each `(client, donor, rep)` cell, sorted
alphabetically by source `batch`. Cell sizes match across modalities: one library
for 12-library clients, two for 24-library clients. Index-wise pairing yields
**48 joint samples per modality** by default, or **72** with `INCLUDE_CLIENT_04 = True`.
This pairing aligns the evaluation groups by a deterministic rule; it does not
establish that the paired libraries were measurements of the same physical
sample aliquot.

## Federated batch correction

FedRBE (`flimmaBatchCorrection`) implements federated `limma::removeBatchEffect`
**within each modality**. Each client uses its own libraries and exchanges
aggregate `X^T X` / `X^T Y` statistics; raw measurements stay local. Correction
uses the original Quartet `batch` as `batch_col` and includes `D5/F7/M8` as donor
covariates, with D6 as reference, to retain biological differences. Notebook 02
records the reference batch in `before/fedrbe_client_groups.tsv`.

### Shared inputs and feature filtering

Notebook 02 prepares the central and federated inputs from the same filtered
matrix. For Proteomics, the minimum raw value (`-6.644`) is treated as the
zero/missing-equivalent value. If a feature's third quartile within a client is
at this floor, all its values for that client are masked to `NA`. The app's
usability rule is then applied, retaining features usable in at least three
clients. Deriving both inputs from this matrix ensures that differences between
correction methods are not caused by different feature sets.

Metabolomics is already `log2(expr + 1)`. Notebook 02 applies **no feature filter**:
all metabolites from `filtered_data/Metabolomics_filtered.tsv` (71 by default),
including zeros, pass unchanged to both corrections. Adding `T_L4_01` to
`client_03_L05_L04` balances modality sample counts independently of
`INCLUDE_CLIENT_04`; metadata preserves `lab="L04"`.

### Client size and execution

The privacy floor requires each client to have more samples than design
coefficients: `n_samples > 1 + |COVARIATES| + (n_total_batches - 1)`.
This means at least 8 libraries for the default three-client/four-batch layout
and at least 10 for four clients/six batches. The smallest client has 12, so
neither layout requires merging clients to satisfy this constraint.

**Run:** `generate_fedrbe_corrected_datasets.py` runs the FeatureCloud app (`featurecloud.ai/bcorrect:latest`) for each modality without and with SMPC. It requires the FeatureCloud Python package, Docker Python SDK, and a running Docker daemon with the local image. Outputs: `after/label_aware/<Modality>/FedApp_corrected_data.tsv` and `FedApp_corrected_data_smpc.tsv`.

## Outputs after a clean run

Steps 2–5 from the run order above produce:

```
all_modalities_metadata.tsv               ← shared evaluation preparation
before/                                     ← written by notebook 02
  datainfo.json
  fedrbe_client_groups.tsv
  preparation_summary.tsv
  <Modality>/
    central_intensities_log_UNION.tsv
    central_input.tsv                       ← notebook 03
    metadata.tsv
    <client_xx_LXX[_LXX]>/
      intensities_log_UNION.tsv
      design.tsv
      config.yml
  all_modalities_before_kmeans_matrix.tsv   ← shared evaluation preparation
after/                                      ← correction launchers + shared evaluation preparation
  label_aware/
    all_modalities_corrected_kmeans_matrix.tsv  ← shared evaluation preparation
    all_modalities_fedapp_kmeans_matrix.tsv     ← shared evaluation preparation
    <Modality>/
      intensities_log_Rcorrected_UNION.tsv      ← central launcher
      FedApp_corrected_data.tsv                 ← federated launcher
      FedApp_corrected_data_smpc.tsv            ← federated launcher (SMPC)
      individual_results/<client>/
        only_batch_corrected_data.csv
        full_corrected_data.csv
        report.txt
  no_label/
    <Modality>/                              ← same per-modality filenames, no covariates
merged_omics/                                 ← 05_create_merged_omics_folder.py
  before/
    <client_xx_LXX[_LXX]>/
      merged_data.tsv
  after/label_aware/
    <client_xx_LXX[_LXX]>/
      merged_data.tsv
```

Validation: central limma places the FedRBE reference batch last to match
`removeBatchEffect()`'s `contr.sum` coding. All modalities then match the app to
machine precision; the default inputs contain no missing values.

## Additional optional metabolomics file

`Metabolomics_fullset_expr_r984c204_randomfloor.csv` is a separate non-log export
(984 rows × 204 samples). Only 180 columns match `meta_full_dataset_3omics.csv`;
24 extra `T_L1_*_01-03` and `T_L4_*_13-15` columns lack metadata. Do not use it as
the default correction input without recovering their metadata.
