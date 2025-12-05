import pandas as pd
from pathlib import Path

# Process each lab
for lab in ['lab1', 'lab2', 'lab3']:
    # Before (uncorrected) data
    base_path = Path(f'/home/jk/featurecloudALL/apps/fedRBE/evaluation_data/simulated/mild_imbalanced/before/{lab}')

    # Read intensities (features are rows, samples are columns)
    intensities = pd.read_csv(base_path / 'intensities.tsv', sep='\t', index_col=0)

    # Read design file
    design = pd.read_csv(base_path / 'design.tsv', sep='\t', index_col=0)

    # Transpose intensities so samples are rows
    data = intensities.T

    # Add condition column from design
    data['condition'] = design['condition']

    # Save as CSV
    data.to_csv(base_path / 'data.csv', index=True)
    print(f"Created {base_path / 'data.csv'}")

# After (corrected) data - from individual results
after_path = Path('/home/jk/featurecloudALL/apps/fedRBE/evaluation_data/simulated/mild_imbalanced/after/individual_results')

for idx, lab in enumerate(['lab1', 'lab2', 'lab3']):
    # Read corrected data - it's tab-separated with features as rows
    corrected_data = pd.read_csv(after_path / f'only_batch_corrected_data_{idx}.csv',
                                   sep='\t', index_col=0)

    # Transpose so samples are rows (like the uncorrected data)
    corrected_data = corrected_data.T

    # Read original design to get condition labels
    design_path = Path(f'/home/jk/featurecloudALL/apps/fedRBE/evaluation_data/simulated/mild_imbalanced/before/{lab}')
    design = pd.read_csv(design_path / 'design.tsv', sep='\t', index_col=0)

    # Add condition column from design - match by index
    corrected_data['condition'] = design.loc[corrected_data.index, 'condition']

    # Create output directory
    output_dir = after_path / lab
    output_dir.mkdir(exist_ok=True)

    # Save data
    corrected_data.to_csv(output_dir / 'data.csv', index=True)

    # Copy config
    import shutil
    shutil.copy(design_path / 'config_forest.yaml', output_dir / 'config_forest.yaml')

    print(f"Created {output_dir / 'data.csv'}")
