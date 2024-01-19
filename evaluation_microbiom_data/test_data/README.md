# Description
Here microbiome data used in FemAI can be found as well as some small 
processing tools.

# Raw Data
The data is publically available, all RAW data can be found [here](https://entrepot.recherche.data.gouv.fr/dataset.xhtml?persistentId=doi:10.57745/7IVO3E)

1. `metadata_2250_CRC_cohort_20231114.tsv`
   This file contains all metadata, the sample columns values are the column
   names used in `species_signal_2250_CRC_cohort_20231115.tsv`
2. `species_signal_2250_CRC_cohort_20231115.tsv` contains the samples as columns
   and the MSPs (taxa information) as rows

# Scripts for extraction
1. The script `generateMSPcsv.py` is available and can be extended, it merges 
   the MSP info and the columns `study_accession` and `country` into a 
   `mergedMSP.tsv` file. It can easily be modified to include other columns in
   the merge

# Extracted data
1. `mergedMSP.tsv` contains
   - All MSP expression data
   - country
   - study accession (batch)