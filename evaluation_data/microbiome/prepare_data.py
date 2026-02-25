"""
Prepare microbiome data for federated random forest classification.
Converts TSV format (features as rows) to CSV format (features as columns).
"""
import pandas as pd
from pathlib import Path

# Define the centers/labs
centers = ["China1", "China3", "China5", "France1", "Germany1", "Germany2"]

print("Processing uncorrected data...")
# Process each center's before data
for center in centers:
    print(f"Processing {center}...")

    # Read the intensity data (TSV with genes as rows, samples as columns)
    intensities_path = f"before/{center}/UQnorm_log_counts_for_corr.tsv"
    intensities = pd.read_csv(intensities_path, sep='\t', index_col=0)

    # Transpose so samples are rows and features are columns
    intensities_t = intensities.T

    # Read design file to get CRC target
    design_path = f"before/{center}/design.tsv"
    design = pd.read_csv(design_path, sep='\t')

    # Match samples and add CRC column
    intensities_t['CRC'] = intensities_t.index.map(design.set_index('sample')['CRC'])

    # Save as CSV
    output_dir = Path(f"before/{center}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "data.csv"
    intensities_t.to_csv(output_path, index=False)
    print(f"  Created {output_path} with {len(intensities_t)} samples and {len(intensities_t.columns)-1} features")

print("\nProcessing corrected data...")
# Process corrected data files (they are TSV with genes as rows)
# Order mapping based on file inspection
corrected_files = [
    ("after/individual_results/only_batch_corrected_data_0.csv", "China1"),
    ("after/individual_results/only_batch_corrected_data_1.csv", "China3"),
    ("after/individual_results/only_batch_corrected_data_2.csv", "China5"),
    ("after/individual_results/only_batch_corrected_data_3.csv", "France1"),
    ("after/individual_results/only_batch_corrected_data_4.csv", "Germany1"),
    ("after/individual_results/only_batch_corrected_data_5.csv", "Germany2"),
]

for corrected_file, center in corrected_files:
    print(f"Processing corrected data for {center}...")

    # Read corrected data (TSV format with genes as rows)
    corrected = pd.read_csv(corrected_file, sep='\t', index_col=0)

    # Transpose so samples are rows and features are columns
    corrected_t = corrected.T

    # Read design file to get CRC target
    design_path = f"before/{center}/design.tsv"
    design = pd.read_csv(design_path, sep='\t')

    # Match samples and add CRC column
    corrected_t['CRC'] = corrected_t.index.map(design.set_index('sample')['CRC'])

    # Save as CSV
    output_dir = Path(f"after/individual_results/{center}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "data.csv"
    corrected_t.to_csv(output_path, index=False)
    print(f"  Created {output_path} with {len(corrected_t)} samples")

print("\nDone!")
