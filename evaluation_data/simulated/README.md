# Simulated datasets

Three E. coli simulation scenarios with three labs each, based on the FedProt project: <https://doi.org/10.48550/arXiv.2407.15220>.

Scenarios: `balanced`, `mild_imbalanced`, and `strong_imbalanced`.

The generated matrices contain NA values. PCA plots replace NA with 0.

Central run:  
- Runs limma RBE with condition as covariate and lab as batch.
- Uses matrices without NA values.

# Structure

`before/` contains both combined and app-ready inputs:
- Combined TSV files for all labs (`metadata`, raw, raw-log2, and filtered-log2).
- One folder per lab for the FeatureCloud app. Each folder contains data and a design file with covariates.

Main files:
- `00_data_simulation.ipynb`: simulation setup.
- `01_data_prep_and_central_RBE.ipynb`: preprocessing and central correction.
- `{balanced,mild_imbalanced,strong_imbalanced}/prepare_data.py`: scenario-specific app folder preparation.
- `../../evaluation/evaluation_simulated.ipynb` and `../../evaluation/evaluation_simulated_30runs.ipynb`: evaluation plots and metrics.
