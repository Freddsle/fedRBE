"""
Helper script to create relevant folder structures for federated learning on the multi-omics data
Requires the use of the previous scripts to create the merged data files.

Expected layout (relative to this file):
    all_modalities_metadata.tsv
    <before|after>/all_modalities_before_kmeans_matrix.tsv


Produces:
    merged_omics/
        <before|after>/
            <cohort>/
                merged_data.tsv      - features x samples (all omics + meta columns)
            datainfo.json
"""

from __future__ import annotations

from pathlib import Path
import json

import pandas as pd

def create_merged_omics_folder():
    script_folder = Path(__file__).parent
    datafiles = [
        script_folder / "before" / "all_modalities_before_kmeans_matrix.tsv",
        script_folder / "after" / "all_modalities_fedapp_kmeans_matrix.tsv",
    ]
    clients = [
        "client_01_L01",
        "client_02_L02",
        "client_03_L05_L04",
    ]
    prediction_target = "condition"
    merged_omics_folder = script_folder / "merged_omics"
    merged_omics_folder.mkdir(parents=True, exist_ok=True)
    index_name = "sample_id"

    # Read the meta file
    df_meta = pd.read_csv(script_folder / "all_modalities_metadata.tsv", sep="\t", index_col="pseudo_sample")
    # df_meta already is in sample x feature format

    for datafile in datafiles:
        print(f"\n{'='*60}")
        print(f"Processing file: {datafile}")
        print(f"{'='*60}")
        datafile_folder = datafile.parent.name  # before or after

        # read the merged data file
        df = pd.read_csv(datafile, sep="\t", index_col="rowname")
        # features x samples format, need to rotate to samples x features
        # current index are the feature names, current columns are the sample names
        df = df.transpose()

        for client in clients:
            # write the data of the client
            print(f"Processing client: {client}")
            # e.g merged_omics / before / client_01_L01 / merged_data.tsv
            client_folder = merged_omics_folder / datafile_folder / client
            client_folder.mkdir(parents=True, exist_ok=True)

            # use all samples whose index contains the client name
            # e.g. idx is client_02_L02_D6_3_1, client is client_02_L02
            client_samples = [sample for sample in df_meta.index if client in sample]
            df_client = df.loc[client_samples]

            # merge in the prediction target column from the meta file
            df_client[prediction_target] = df_meta.loc[client_samples, prediction_target]
            df_client.index.name = index_name
            df_client.to_csv(client_folder / "merged_data.tsv", sep="\t", index=True)

        # write the datainfo of all clients
        data_info = {
            "data_name": "Quartet Multiomics",
            "covariates": [],
            "prediction_targets": [prediction_target],
            "datafile": {
                "filename": "merged_data.tsv",
                "separator": "\t",
                "rotation": "samples x features",
                "samplename_column": index_name,
            },
            "cohorts": [{
                "name": client,
                "folder": client,
                "designfile": None,
            } for client in clients],
        }

        with open(merged_omics_folder / datafile_folder / "datainfo.json", "w") as f:
            json.dump(data_info, f, indent=4)

if __name__ == "__main__":
    create_merged_omics_folder()
