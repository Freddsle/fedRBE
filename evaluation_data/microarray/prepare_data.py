"""
Prepares microarray data for federated classification.
Converts TSV expression data (features as rows) to CSV format (features as columns) with target variable.
"""
import pandas as pd
from pathlib import Path

# Study folders
studies = ["GSE14407", "GSE26712", "GSE38666", "GSE40595", "GSE6008", "GSE69428"]

for study in studies:
    print(f"Processing {study}...")

    # Paths
    study_path = Path("before") / study
    design_path = study_path / "design.tsv"
    expr_path = study_path / "expr_for_correction_UNION.tsv"

    # Load design (sample IDs and target)
    design = pd.read_csv(design_path, sep="\t")

    # Load expression data (genes as rows, samples as columns)
    expr = pd.read_csv(expr_path, sep="\t", index_col=0)

    # Transpose so samples are rows and genes are columns
    expr_t = expr.T

    # Filter to only include samples in the design file
    expr_t = expr_t.loc[design['file']]

    # Add the target column (HGSC)
    expr_t['HGSC'] = design.set_index('file').loc[expr_t.index, 'HGSC'].values

    # Save as CSV
    output_path = study_path / "data.csv"
    expr_t.to_csv(output_path, index=True)

    print(f"  Created {output_path} with {len(expr_t)} samples and {len(expr_t.columns)-1} features")

print("\nProcessing corrected data...")
# Process corrected data files (they are TSV with genes as rows)
# Order is: 0=GSE6008, 1=GSE14407, 2=GSE26712, 3=GSE38666, 4=GSE40595, 5=GSE69428
corrected_files = [
    ("after/individual_results/only_batch_corrected_data_0.csv", "GSE6008"),
    ("after/individual_results/only_batch_corrected_data_1.csv", "GSE14407"),
    ("after/individual_results/only_batch_corrected_data_2.csv", "GSE26712"),
    ("after/individual_results/only_batch_corrected_data_3.csv", "GSE38666"),
    ("after/individual_results/only_batch_corrected_data_4.csv", "GSE40595"),
    ("after/individual_results/only_batch_corrected_data_5.csv", "GSE69428"),
]

for corrected_file, study in corrected_files:
    corrected_path = Path(corrected_file)
    if corrected_path.exists():
        print(f"Processing corrected data for {study}...")

        # Load corrected data (TSV with genes as rows)
        corrected = pd.read_csv(corrected_path, sep="\t", index_col=0)

        # Transpose so samples are rows
        corrected_t = corrected.T

        # Load design to get target
        design_path = Path("before") / study / "design.tsv"
        design = pd.read_csv(design_path, sep="\t")

        # Add the target column
        corrected_t['HGSC'] = design.set_index('file').loc[corrected_t.index, 'HGSC'].values

        # Save as data.csv in individual_results folder
        output_dir = Path("after/individual_results") / study
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "data.csv"
        corrected_t.to_csv(output_path, index=True)
        print(f"  Created {output_path} with {len(corrected_t)} samples")

print("\nDone!")
