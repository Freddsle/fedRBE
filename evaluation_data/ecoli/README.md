# E. coli dataset

Five E. coli lab datasets from the FedProt project: <https://doi.org/10.48550/arXiv.2407.15220>.

Preprocessing:
- Raw data comes from DIA-NN outputs.
- Filtering keeps rows with at least two non-NA values per center and two per covariate class.
- Intensities are log2-transformed.

The matrices contain NA values; PCA plots omit rows with NA.

Central run:  
- Uses the union matrix.
- Runs limma RBE with condition as covariate and lab as batch.
- Keeps NA values.

# Structure

`before/` contains both combined and app-ready inputs:
- Combined TSV files for all labs (`metadata` and filtered log2 intensities).
- One folder per lab for the FeatureCloud app. Each folder contains data and a design file with covariates.

Use log2-transformed app inputs and disable log transformation during batch effect correction.

Main files:
- `01_data_prep_and_central_RBE.ipynb`: preprocessing and central correction.
- `prepare_data.py`: app-ready folder preparation.
- `../../evaluation/evaluation_ecoli.ipynb`: evaluation plots and metrics.
