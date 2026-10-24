# Real-Dataset Clustering Evaluation

Evaluates k-means clustering quality (ARI) on real omics datasets before and after
batch correction, for both central and federated k-means.

The notebooks start from prepared expression/protein matrices and their sample
metadata. Batch correction must already have been performed. This workflow
aligns and filters the matrices for clustering, runs the selected clustering
methods, and compares their assignments with known condition and batch labels.

**Datasets:** `ecoli`, `ovarian_cancer`, `ccRCC_proteomics`, and `quartet_multiomics`
(transcriptomics, proteomics, and metabolomics; joint samples are pseudo-matched
as described below).

## Choose an execution route

| Goal | Notebook order | FeatureCloud/Docker needed? |
| --- | --- | --- |
| Evaluate central k-means | 02 → 04 | No |
| Compare central and newly run federated k-means | 02 → 03 → 04 | Yes, for 03 |
| Reuse existing local federated results | 02 → 03 → 04, with `RERUN_FEDERATED = False` | No; matching result files must exist |
| Compare both correction sources across seeds | 05 | Only for new federated runs |

Notebooks 02, 03, and 05 prepare inputs for all four datasets automatically from
correction outputs and original metadata, including Quartet joint matrices.
Notebook 00 provides an optional view of that same preparation function.

Set `CORRECTION_ARM = "label_aware"` (default) or `"no_label"` in notebooks 02–05 to select `evaluation_data/<dataset>/after/<arm>/`.
Preparation, labels, seeds, algorithms, and metric formats are the same for both
arms. The default `OUTPUT_ROOT` is the notebook directory for `label_aware` and
its `no_label/` subdirectory for `no_label`, keeping app inputs and cached results
separate. Keep the arm and output root consistent between notebooks 02–04.
For custom `DATASET_CONFIG` files, use explicit corrected paths, leave
`CORRECTION_ARM` at its default, and set a separate `OUTPUT_ROOT`.

Notebook 04 loads existing federated assignment tables even when you skip 03.
Retain only tables from runs matching the current prepared inputs and configuration;
skipping execution does not establish that saved results are current.

---

## Prerequisites

1. **Python environment** — from the repository root, use Conda:
   ```bash
   conda env create -f environment.yml
   conda activate fedRBE
   ```
   or a separate Python environment:
   ```bash
   python3 -m venv fedrbe_env
   source fedrbe_env/bin/activate
   pip install -r requirements.txt
   ```

2. **Input matrices** — prepare the original matrices, site metadata, and corrected
   TSVs at paths in `evaluation_utils/datasets.yaml`. Notebooks 02 and 03 default to
   central correction; set `CORRECTED_SOURCE` to select another existing source.
   Notebook 05 compares central and federated correction separately.
   If the files are Git LFS pointers, fetch them first:
   ```bash
   git lfs pull
   ```

3. **FeatureCloud (for new runs in notebooks 03 or 05)** — either setup provides the CLI;
   Docker and the `fc_kmeans_upd` image are also required (see Step 3).
   Result archives under `<dataset>/inputs/{before,corrected}/tests/tests/`
   are generated locally and are not committed.

---

## Step-by-Step Instructions

Start each shell command block from the repository root. Run notebook cells in
order with `evaluation_clusterization_after_correction/real_datasets/` as their
working directory, after completing prerequisite steps.
Keep dataset, correction-source, and output settings consistent between notebooks
02 and 03 when comparing their clustering results.

### Optional Quartet preparation inspection (`00_build_kmeans_matrices.ipynb`)

The ordinary evaluators call `build_quartet_matrices()` in
`evaluation_utils/real_datasets_utils.py` when Quartet is selected. The Python
notebook calls that same function to inspect its outputs independently.
It reads per-modality inputs and metadata prepared by
`evaluation_data/quartet_multiomics/02_prepare_RBE_inputs.ipynb` and the selected
arm's central and federated correction outputs. It aligns modalities by
`pseudo_sample` and stacks their feature rows without scaling or dropping NAs.

```bash
cd evaluation_clusterization_after_correction/real_datasets
jupyter execute --kernel_name=python3 00_build_kmeans_matrices.ipynb
```

**Outputs:**

- `evaluation_data/quartet_multiomics/before/all_modalities_before_kmeans_matrix.tsv`
  — joint uncorrected matrix.
- `evaluation_data/quartet_multiomics/after/<arm>/all_modalities_corrected_kmeans_matrix.tsv`
  plus `all_modalities_fedapp_kmeans_matrix.tsv` for FeatureCloud fedRBE outputs
  — joint matrices used as `before_matrix` / `corrected_central` /
  `corrected_federated` in `evaluation_utils/datasets.yaml`. Kept under
  `evaluation_data/` because `evaluation/evaluation_quartet_multiomics.ipynb` also
  reads them.
- `evaluation_data/quartet_multiomics/all_modalities_metadata.tsv`
  — condition and client labels for the joint columns.
The evaluators read joint metadata directly and create per-client app inputs
through their shared preparation code.

The three matrices are stacked vertically: each row is a feature prefixed by its
modality name, and each column is a pseudo-matched sample. This preserves the
origin of each feature while giving clustering a single matrix. No manual
per-feature scaling is applied here; Steps 2 and 3 each apply their own scaling,
as they do for the other datasets.

**Targets** must match the active Quartet client list, controlled by the
`INCLUDE_CLIENT_04` toggle in the Quartet notebooks and launcher — default `False`,
i.e. 3 clients / 48 joint samples.

- `condition`: donor labels `D5/D6/F7/M8` (`D6` is the limma reference donor),
  evaluated with `k = 4`.
- `client`: client labels (default 3-client config: `client_01_L01`,
  `client_02_L02`, `client_03_L05_L04`; with `INCLUDE_CLIENT_04 = True` also
  `client_04_L03_L14`), evaluated with `k = len(CLIENT_NAMES)`. This is the
  technical structure that batch correction is meant to remove. The metadata
  column reusing the real-dataset schema is named `lab` for compatibility with
  the other real-dataset scripts.

Joint samples are matched across modalities via the `pseudo_sample` keys built
in `evaluation_data/quartet_multiomics/02_prepare_RBE_inputs.ipynb`
(`{client}_{donor}_{rep}_{i}`).

---

### Step 2 — Central K-Means (`02_central_kmeans.ipynb`)

Runs scikit-learn k-means on the prepared matrices (k = number of conditions and number of
batches), computes ARI against condition and batch labels.

These are two evaluation targets: condition labels represent biological groups,
while batch labels represent the site/client structure. Inspect both when
comparing uncorrected and corrected data.

1. Open `02_central_kmeans.ipynb` and select `DATASETS`, `CORRECTED_SOURCE`, and
   `OUTPUT_ROOT`. Set `DATASET_CONFIG` to use an alternative dataset registry.
2. Run all cells; current matrices and metadata are prepared automatically.
3. For central-only analysis, continue to Step 4. To add federated clustering,
   complete Step 3 first.

**Outputs per dataset:**

```
<dataset>/prepared/before_matrix.tsv
<dataset>/prepared/corrected_matrix.tsv
<dataset>/prepared/metadata.tsv
<dataset>/kmeans_res/runs/1_metadata_cntrl_kmeans_res.tsv
<dataset>/metrics/metrics_ari.tsv
metrics_ari.tsv                          # combined across all datasets
```

---

### Step 3 — Federated K-Means (`03_federated_runs.ipynb`)

This step prepares a separate input directory for each site and runs the
FeatureCloud `fc_kmeans` app on the uncorrected and corrected variants. It then
combines per-client assignments into tables that Step 4 can compare with central
k-means. Input preparation runs automatically. Choose whether to launch new runs
or reuse existing local results.

#### Option A — Re-run federated k-means with FeatureCloud (default)

1. Install the FeatureCloud CLI and ensure Docker is running (see
   [FeatureCloud docs](https://featurecloud.ai/)).

2. Build the fc_kmeans Docker image before running the notebook:

   **Using the FeatureCloud CLI:**
   ```bash
   featurecloud app build evaluation_clusterization_after_correction/federated_kmeans_upd fc_kmeans_upd
   ```
   **Or directly with Docker:**
   ```bash
   cd evaluation_clusterization_after_correction/federated_kmeans_upd
   docker build -t fc_kmeans_upd .
   ```

3. Open `03_federated_runs.ipynb` and keep `RERUN_FEDERATED = True` in its
   **Configuration** cell.
4. Check the dataset list and adjust `CONTROLLER_HOST`, `TIMEOUT`, or
   `QUERY_INTERVAL` for your local setup if needed.
5. Run all cells.

   With Docker running, the notebook starts and stops the controller for each
   `<dataset>/inputs/{before,corrected}/` variant via `_startup()`;
   no manual `featurecloud controller start` is needed.

   The controller saves result ZIPs to
   `<dataset>/inputs/{before,corrected}/tests/tests/results_test_<id>_client_<id>_<name>.zip`,
   which are automatically extracted and aggregated by the notebook.

#### Option B — Aggregate results from an earlier local run

Use this route when you already have results from a run with the same inputs and
configuration. It rebuilds the combined assignment tables without launching the
app, so Docker is not needed.

1. Retain either the extracted clustering CSVs under
   `<dataset>/fc_kmeans_res/<variant>/1_fed_clustering/`, or the result ZIPs
   there or under `<dataset>/inputs/<variant>/tests/tests/`. Here `<variant>` is
   `before` or `corrected`.
2. Open `03_federated_runs.ipynb` and set `RERUN_FEDERATED = False` in its
   **Configuration** cell.
3. Check the dataset list and run all cells to collect the existing results.

**Outputs per dataset:**

```
<dataset>/inputs/before/<site>/intensities.tsv        # FC input files
<dataset>/inputs/before/<site>/design.tsv
<dataset>/inputs/before/<site>/config_kmeans.yml
<dataset>/inputs/corrected/<site>/...                 # same for corrected
<dataset>/fc_kmeans_res/before/1_fed_clustering/      # intermediate per-client clustering CSVs
<dataset>/fc_kmeans_res/corrected/1_fed_clustering/
<dataset>/kmeans_res/runs/1_metadata_before_fedclusters.tsv
<dataset>/kmeans_res/runs/1_metadata_after_fedclusters.tsv
```

---

### Step 4 — Analysis & Plots (`04_analysis_metrics_plots.ipynb`)

Combines central and federated results, computes ARI for all methods, and produces
ARI bar charts, PCA plots, and a delta-ARI summary table.

1. Complete Step 2, plus Step 3 if you want federated results in the comparison.
2. Open `04_analysis_metrics_plots.ipynb`, check the dataset list, and run all cells.
   Missing federated results are handled, so this also works for central-only runs.

Use the `Target` column to separate condition and batch comparisons, and compare
before/after values within the same method and target. The `N` column records the
number of samples with valid labels used for each ARI calculation.

**Outputs (written to `OUTPUT_ROOT`):**

```
metrics_ari.tsv          # full metrics table (all datasets, methods, targets)
```
Plots are displayed inline in the notebook.

---

### Step 5 — Repeated Seeds (`05_multiple_runs.ipynb`)

Run 05 directly to compare central and federated k-means on uncorrected data and
on their respective corrected matrices. It evaluates condition labels with ten
seeds (1, 11, …, 91) and one initialization per seed.

`RERUN_FEDERATED = False` reuses the saved assignment tables under
`<dataset>/kmeans_res/runs/seed_NNN_metadata_before_multi_fedclusters.tsv` and
`seed_NNN_metadata_corrected_fed_fedclusters.tsv`. The notebook checks sample
alignment and recomputes ARI from these labels; central k-means is recomputed from
the current matrices. Reuse assignments only with matching inputs and settings.
Missing assignments raise an error. Set `RERUN_FEDERATED = True` to regenerate
them with FeatureCloud and Docker.

Select `CORRECTION_ARM` as described above; 05 also reads the `CORRECTION_ARM`
environment variable, defaulting to `label_aware`. Outputs are relative to
`OUTPUT_ROOT`:

- `metrics_ari_multirun.tsv`: individual condition-target scores for every seed.
- `ari_boxplots.png`: Figure 4-style boxplots and seed dots, saved by the shared
  R plotting helper and displayed inline.

Plotting requires the R environment from the
[reproduction guide](../../docs/reproduce.md). FeatureCloud archives and extracted
client CSVs are unnecessary when reusing assignment TSVs.

---

## Output Directory Layout

Steps 2–4 list output paths relative to `OUTPUT_ROOT`; the optional preparation notebook uses
repository-relative paths. After a complete run, the generated files are arranged
as follows. A central-only run does not create the Step 3 directories or federated
assignment tables.

```text
real_datasets/
├── <dataset>/
│   ├── prepared/                              # Steps 2–3: aligned inputs
│   │   ├── before_matrix.tsv
│   │   ├── corrected_matrix.tsv
│   │   └── metadata.tsv
│   ├── inputs/                                # Step 3: app inputs and ZIPs
│   │   ├── before/
│   │   │   ├── <site>/{intensities.tsv,design.tsv,config_kmeans.yml}
│   │   │   └── tests/tests/results_test_<id>_client_*.zip
│   │   └── corrected/                         # same layout as before/
│   ├── fc_kmeans_res/                         # Step 3: extracted client results
│   │   ├── before/1_fed_clustering/
│   │   └── corrected/1_fed_clustering/
│   ├── kmeans_res/runs/
│   │   ├── 1_metadata_cntrl_kmeans_res.tsv     # Step 2
│   │   ├── 1_metadata_before_fedclusters.tsv   # Step 3
│   │   └── 1_metadata_after_fedclusters.tsv    # Step 3
│   └── metrics/metrics_ari.tsv                # dataset-level metrics
└── metrics_ari.tsv                            # combined results
```

The assignment tables retain sample labels for downstream inspection; the
combined `metrics_ari.tsv` is the starting point for comparing datasets and methods.
