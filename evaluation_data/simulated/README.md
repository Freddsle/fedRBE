# Simulated datasets

Three datasets, 5 labs, from FedProt project (https://doi.org/10.48550/arXiv.2407.15220).

Simulated proteomics data.

Contains NA - for PCA plots - NA were replaced with 0.

Central run:  
- limmaRBE with condition as covariates and lab as batches.
- with NA.

# Structure

Three datasets, different balance of samples per lab.

In /before folder there are two versions on the dataset.  
- one file for all - all labs in one tsv (metadata, raw, raw-log2, and filtered-log2).
- the save but folder per lab - for App. Contains additionally design file - covariates info in sutable for App format.
