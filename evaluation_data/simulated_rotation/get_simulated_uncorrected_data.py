# Small helper script that:
# 1. Calls the R script for data generation (if needed)
# 2. Creates the lab1,...labn folders and fills them with data based on the chosen simulation number
import subprocess
from pathlib import Path
import pandas as pd

BASE_FOLDER = Path(__file__).parent
DATASETS = ["strong_imbalanced", "mild_imbalanced", "balanced"]
METAFILE = "all_metadata.tsv"
DATAFILE_SUFFIX = "_intensities_data.tsv"
SIMULATION_NUMBER = 1

# 1. Call the R script to generate data if needed
subprocess.run(["Rscript", "01_data_simulation.R", str(SIMULATION_NUMBER)], check=True)

# Load the metadata to determine:
# - which labs are involved
# - lab membership of samples
# - conditions of samples (A/B)
# Example of the metadata file:
# "file"	"condition"	"lab"
# "s.1"	"A"	"lab3"
# Example of the design file given for a lab:
# "file"	"condition"	"lab"	"A"
# "s.13"	"A"	"lab1"	0
# Data file format (1_intensities_data.tsv) for first simulation
# column rowname contains the features, other columns are samples
for dataset in DATASETS:
    # Get labs for this dataset
    metafile = BASE_FOLDER / dataset / METAFILE
    metadata = pd.read_csv(metafile, sep="\t", index_col="file")
    labs = metadata["lab"].unique()
    for lab in labs:
        # Get samples for this lab and dataset
        samples = metadata[metadata["lab"] == lab].index
            # s.1, s.2, ... s.n (n depends on the dataset and lab)

        lab_folder = BASE_FOLDER / dataset / "before" / lab
        lab_folder.mkdir(parents=True, exist_ok=True)

        # add the A column to the design file for this lab
        # sometimes we need a binary label, that's why we add that extra col
        # that contains the same info as condition but in 0/1 format (A=0, B=1)
        design = metadata.copy()
        design["A"] = (design["condition"] == "B").astype(int)

        # Now add the actual data for this lab and dataset
        datafile = BASE_FOLDER / dataset / "before" / "intermediate" / f"{SIMULATION_NUMBER}{DATAFILE_SUFFIX}"
        data = pd.read_csv(datafile, sep="\t", index_col="rowname")

        # save the files for this lab and dataset
        data[data.columns.intersection(samples)].to_csv(lab_folder / "intensities.tsv", sep="\t")
        design.loc[samples].to_csv(lab_folder / "design.tsv", sep="\t")
        print(f"Prepared data for {dataset} - {lab} with {len(samples)} samples and {data.shape[0]} features.")