# Simulated-Data Clustering Evaluation

Evaluates k-means clustering quality (ARI) on simulated omics datasets across
**30 repetitions** and **3 imbalance scenarios**, comparing uncorrected,
centrally corrected, and federated-corrected data.

**Scenarios:** `balanced`, `mild_imbalanced`, `strong_imbalanced`  
**Targets:** condition (k = 2), batch/lab (k = 3)  
**Methods:** `uncorrected`, `central` (R/limma), `federated` (fedRBE)

---

## Prerequisites

1. **Python environment** — install repo dependencies:
   ```bash
   pip install -r requirements.txt   # from repo root
   ```

2. **Simulated data** — the following files must exist under `evaluation_data/simulated/`:
   ```
   <scenario>/all_metadata.tsv
   <scenario>/before/lab{1,2,3}/intensities.tsv
   <scenario>/after/runs/{1..30}_R_corrected.tsv
   <scenario>/after/runs/{1..30}_FedSim_corrected.tsv
   ```
   If the TSV files are Git LFS pointers, fetch them first:
   ```bash
   git lfs pull
   ```
   Note: `after/runs/*.tsv` are listed in `.gitignore` (too many files); they must
   be generated locally by the simulation scripts in `evaluation_utils/`.

---

## Step-by-Step Instructions

### Step 1 — Run K-Means on All Scenarios & Runs (`01_simulated_kmeans.ipynb`)

Loads the uncorrected and corrected matrices for each scenario and each of the 30
simulation runs, runs k-means (k = 2 for condition, k = 3 for batch), and computes ARI.

1. Open `01_simulated_kmeans.ipynb`.
2. (Optional) Edit the **Configuration** cell:
   - `SCENARIOS` — subset of scenarios to run.
   - `N_RUNS` — number of runs to process (default: 30).
   - `SEED` — random seed for k-means reproducibility (default: 11).
3. Run all cells (`Run All`).

The notebook prints progress every 10 runs per scenario and a summary table at the end.

**Outputs (written to `simulated/outputs/`):**
```
outputs/balanced_ari_results.tsv
outputs/mild_imbalanced_ari_results.tsv
outputs/strong_imbalanced_ari_results.tsv
```

Each TSV has columns: `scenario`, `run`, `method`, `target`, `ARI`, `N`.

> **Note:** The `outputs/` directory is listed in `.gitignore` — results are not committed.

---

### Step 2 — Analysis & Plots (`02_simulated_analysis.ipynb`)

Loads the per-scenario TSVs from Step 1 and produces:
- **Violin plots** of ARI distributions across 30 runs, faceted by target
  (`condition` / `batch`), grouped by scenario and coloured by method.
- **Mean ARI heatmap** — scenario × method for each target.
- **Summary table** — mean ± SD, min, max, and run count per group.

1. Ensure Step 1 has been run (all three `*_ari_results.tsv` files must exist).
2. Open `02_simulated_analysis.ipynb`.
3. Run all cells.

**Outputs (written to `simulated/outputs/`):**
```
outputs/violin_ari_condition.pdf
outputs/violin_ari_batch.pdf
outputs/heatmap_mean_ari.pdf
outputs/ari_summary.tsv
```

Plots are also displayed inline in the notebook.

---

## Output Directory Layout

After both notebooks are run:

```
simulated/
├── outputs/
│   ├── balanced_ari_results.tsv
│   ├── mild_imbalanced_ari_results.tsv
│   ├── strong_imbalanced_ari_results.tsv
│   ├── ari_summary.tsv
│   ├── violin_ari_condition.pdf
│   ├── violin_ari_batch.pdf
│   └── heatmap_mean_ari.pdf
├── 01_simulated_kmeans.ipynb
└── 02_simulated_analysis.ipynb
```
