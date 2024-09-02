# Microbiome data

Data from 5 studies:
- PRJEB27928 (GERMANY)
- PRJEB6070  (FRANCE)
- PRJNA429097 (CHINA)
- PRJEB10878 (CHINA)
- PRJNA731589 (CHINA)


# Preprocessing:
1. Filter - remove NA rows.
2. Filter - keep rows with min 2 not NA values per study.
3. Filter - keep only samples with min 2 not-NA values per sample (column).
4. Keeps samples only with upper quantile > 0.
<!-- 5. Normalization to total sequencing coverage (?). -->
6. Upper-quantile normalization - taking into account only non-NA values.
   

Central run:  
- limmaRBE with classes as covariates and study_accession as batches.
- 0 as 0.

For plots - data contains zero values. Now need to do imputation.

# Structure

In /before folder there are two versions on the dataset.  
- one file for all - all studies in one tsv (metadata, normalized counts and raw counts).
- the save but folder per study - for App. Contains additionally design file - covariates info in sutable for App format.

For the App better to use normalized-log-transformed data, but with log2 inside App transformation disabled.