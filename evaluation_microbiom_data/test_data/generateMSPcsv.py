# Small script that merges MSP data with Metadata and saves a merged csv file
import pandas as pd


### VARIABLES
metaDataFile = 'metadata_2250_CRC_cohort_20231114.tsv'
mspSpeciesFile = 'species_signal_2250_CRC_cohort_20231115.tsv'
metaInfoColumns = ["sample", "study_accession", "country"]
    # the columns to be merged into the final tsv output, must be a column name
    # of the metaDataFile

### LOGIC
dfMeta = pd.read_csv(metaDataFile, sep="\t")
dfMSP = pd.read_csv(mspSpeciesFile, sep="\t")
dfMSP = dfMSP.transpose().reset_index()
dfMSP.columns = dfMSP.iloc[0]
dfMSP = dfMSP.iloc[1:]

# check that the samples column is unique
#print(dfMeta["sample"].is_unique)

finaldf = pd.merge(dfMeta[metaInfoColumns], dfMSP, left_on="sample", right_on="msp_id")
finaldf.to_csv("mergedMSP.tsv", sep='\t', index=False)
