# Proteomics dataset

5 labs, from FedProt project [https://doi.org/10.48550/arXiv.2407.15220].

Preprocessing:
- raw data from DIA-NN outputs.
- filtering - keep only rows with 2 not-NA values per center AND 2 per covariate class.
- log2-transformed.

Contains NA - for PCA plots - omit rows with NA.

Central run:  
- union
- limmaRBE with condition as covariates and lab as batches.
- with NA.

# Structure

In /before folder there are two versions on the dataset.  
- one file for all - all labs in one tsv (metadata, and filtered-log2).
- the save but folder per lab - for App. Contains additionally design file - covariates info in sutable for App format.

For App log2-transformed data should be used - log transform during BEC disabled.