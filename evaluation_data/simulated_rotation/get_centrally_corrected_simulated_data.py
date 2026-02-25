# Small helper script that:
# 1. Calls the R script for central RBE correction (if needed)
# 2. Copies the relevant corrected data files into the DATASETS folder.
import subprocess
from pathlib import Path
import pandas as pd

BASE_FOLDER = Path(__file__).parent
DATASETS = ["strong_imbalanced", "mild_imbalanced", "balanced"]
SIMULATION_NUMBER = 1

# 1. Run the R script to produce centrally corrected data
subprocess.run(
    ["Rscript", str(BASE_FOLDER / "01_data_prep_and_central_RBE.R"), str(SIMULATION_NUMBER)],
    check=True,
    cwd=BASE_FOLDER,
)

# 2. Distribute corrected data into per-lab folders
# Source: {dataset}/after/runs/{SIMULATION_NUMBER}_R_corrected.tsv
# Destination: {dataset}/after/{lab}/intensities_R_corrected.tsv
for dataset in DATASETS:
    corrected_file = BASE_FOLDER / dataset / "after" / "runs" / f"{SIMULATION_NUMBER}_R_corrected.tsv"
    corrected = pd.read_csv(corrected_file, sep="\t", index_col="rowname")
    # We really just want to copy the corrected file to the dataset's after folder
    destination_file = BASE_FOLDER / dataset / "after" / "intensities_R_corrected.tsv"
    corrected.to_csv(destination_file, sep="\t")

