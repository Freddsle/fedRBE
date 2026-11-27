"""
Helper script to create relevant folder structures for federated learning on the multi-omics data
Requires the use of the previous scripts to create the merged data files.

Expected layout (relative to this file):
    all_modalities_metadata.tsv
    before/all_modalities_before_kmeans_matrix.tsv
    after/<correction_arm>/all_modalities_fedapp_kmeans_matrix.tsv


Produces:
    merged_omics/
        <before|after/<correction_arm>>/
            <cohort>/
                merged_data.tsv      - samples x features (all omics + target)
            datainfo.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json

import pandas as pd

def create_merged_omics_folder(datafiles=None, output_folder=None, metadata_file=None,
                              *, correction_arm="label_aware"):
    """Export supplied matrices or one canonical correction arm to RF descriptors."""
    if correction_arm not in ("label_aware", "no_label"):
        raise ValueError("correction_arm must be label_aware or no_label")
    script_folder = Path(__file__).parent
    if datafiles is None:
        datafiles = {
            "before": script_folder / "before" / "all_modalities_before_kmeans_matrix.tsv",
            f"after/{correction_arm}": (script_folder / "after" / correction_arm /
                                       "all_modalities_fedapp_kmeans_matrix.tsv"),
        }
    prediction_target = "condition"
    merged_omics_folder = Path(output_folder) if output_folder else script_folder / "merged_omics"
    merged_omics_folder.mkdir(parents=True, exist_ok=True)
    index_name = "sample_id"

    # Read the meta file
    metadata_file = metadata_file or script_folder / "all_modalities_metadata.tsv"
    df_meta = pd.read_csv(metadata_file, sep="\t", index_col="pseudo_sample")
    clients = df_meta["client"].drop_duplicates().tolist()
    # df_meta already is in sample x feature format

    for datafile_folder, datafile in datafiles.items():
        print(f"\n{'='*60}")
        print(f"Processing file: {datafile}")
        print(f"{'='*60}")

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

            # Use the same explicit client mapping for both correction arms.
            client_samples = df_meta.index[df_meta.client.eq(client)]
            df_client = df.loc[client_samples].copy()

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correction-arm", choices=("label_aware", "no_label"),
                        default="label_aware", help="Canonical correction folder to export")
    create_merged_omics_folder(correction_arm=parser.parse_args().correction_arm)
