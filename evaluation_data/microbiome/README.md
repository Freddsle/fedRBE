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
4. Upper-quantile normalization - taking into account only non-NA values.
   

Central run:  
- limmaRBE with classes as covariates and study_accession as batches.
- 0 as 0.