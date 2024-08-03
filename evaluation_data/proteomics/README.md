# Proteomics dataset

5 labs, from FedProt project (**NOT PUBLISHED** yet).

Preprocessing:
- raw data from DIA-NN outputs were log2-transformed.
- filtering - keep only rows with 2 not-NA values per center AND 2 per covariate class.

Contains NA - for PCA plots - omit rows with NA.

Central run:  
- limmaRBE with condition as covariates and lab as batches.
- with NA.

# Structure

In /before folder there are two versions on the dataset.  
- one file for all - all labs in one tsv (metadata, raw, raw-log2, and filtered-log2).
- the save but folder per lab - for App. Contains additionally design file - covariates info in sutable for App format.

For App non-filtered-not-log2 data should be used.