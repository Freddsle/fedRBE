"""
Prepares the ccRCC proteomics dataset for fedRBE evaluation.

Steps performed:
1. Adds a 'Condition' column ("Tumor" / "Normal") to each site's design.tsv,
   derived from the existing binary Normal/Tumor columns.
2. Builds a union matrix (outer join across all sites) from each site's
   report_filtered.tsv and writes it to before/central_intensities_log_UNION.tsv.

Run from the repository root or from this file's directory:
    python evaluation_data/ccRCC_studies/prepare_ccRCC_data.py

Idempotent: safe to re-run; existing files are overwritten.
"""

from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).parent
BEFORE_DIR = SCRIPT_DIR / "before"
SITES = ["PDC000127", "PXD030344", "PXD042844"]

# ──────────────────────────────────────────────────────────────
# Step 1: Add Condition column to each site's design.tsv
# ──────────────────────────────────────────────────────────────
print("Step 1: Adding 'Condition' column to design.tsv files...")
for site in SITES:
    design_path = BEFORE_DIR / site / "design.tsv"
    df = pd.read_csv(design_path, sep="\t")

    if "Tumor" not in df.columns or "Normal" not in df.columns:
        raise ValueError(
            f"{design_path}: expected 'Tumor' and 'Normal' columns, "
            f"got: {df.columns.tolist()}"
        )
    # Encode Condition as 0/1 integer (Tumor=1, Normal=0) so that the
    # fedRBE app can use it directly as a numeric covariate in the design
    # matrix (same encoding as other datasets, e.g. proteomics Pyr column).
    df["Condition"] = df["Tumor"].astype(int)

    # Reorder design rows to match the column order in report_filtered.tsv.
    # The fedRBE app requires sample_names == list(design.index) (strict order).
    mat_path = BEFORE_DIR / site / "report_filtered.tsv"
    data_cols = pd.read_csv(mat_path, sep="\t", index_col=0, nrows=0).columns.tolist()
    first_col = df.columns[0]  # rowname column (used as index in the app)
    df = df.set_index(first_col).loc[data_cols].reset_index()

    df.to_csv(design_path, sep="\t", index=False)
    print(f"  {site}: wrote {len(df)} rows "
          f"(Tumor/1={int((df['Condition']==1).sum())}, "
          f"Normal/0={int((df['Condition']==0).sum())})"
          f" — reordered to match report_filtered.tsv columns")

# ──────────────────────────────────────────────────────────────
# Step 2: Build union matrix across all three sites
# ──────────────────────────────────────────────────────────────
print("\nStep 2: Building union matrix...")
frames = []
for site in SITES:
    mat_path = BEFORE_DIR / site / "report_filtered.tsv"
    df = pd.read_csv(mat_path, sep="\t", index_col=0)
    frames.append(df)
    print(f"  {site}: {df.shape[0]} features x {df.shape[1]} samples")

union = frames[0].join(frames[1:], how="outer")
union.index.name = "Gene"

out_path = BEFORE_DIR / "central_intensities_log_UNION.tsv"
union.to_csv(out_path, sep="\t")
print(f"\nUnion matrix written to: {out_path}")
print(f"  Shape: {union.shape[0]} features x {union.shape[1]} samples")
print(f"  Missing values: {union.isna().sum().sum()} "
      f"({100 * union.isna().mean().mean():.1f}% of all entries)")
