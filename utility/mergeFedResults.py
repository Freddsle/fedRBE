import os
import pandas as pd

resultsFolder = "results"
dataset_folders = ["fedprot", "microbiome"]
index_cols = [0, None]
federated_folder = "federated"
concat_data = pd.DataFrame()


# Get the path of the dataset folders
this_file = os.path.abspath(__file__)
parent_folder = os.path.dirname(os.path.dirname(this_file))
results = os.path.join(parent_folder, resultsFolder)

# iterate over datasets
for dataset_folder, index_col in zip(dataset_folders, index_cols):
    dataset_folder_path = os.path.join(results, dataset_folder)
    # iterate over the different conditions given here
    for specific_dataset in os.listdir(dataset_folder_path):
        specific_dataset_path = os.path.join(dataset_folder_path, specific_dataset)
        specific_dataset_federated_path = os.path.join(dataset_folder_path, specific_dataset, federated_folder)
        # now we iterate all folders, check that there is only one file and merge the files
        for lab_folder in os.listdir(specific_dataset_federated_path):
            lab_folder_path = os.path.join(specific_dataset_federated_path, lab_folder)
            # check that there is only one file
            files = os.listdir(lab_folder_path)
            assert len(files) == 1
            file = files[0]
            df = pd.read_csv(os.path.join(lab_folder_path, file), sep="\t", index_col=index_col)
            # check cols
            if not concat_data.empty and concat_data.columns.equals(df.columns):
                # check that the columns are the same
                print("ERROR, could not merge the data")
                exit()
            # merge into the df
            concat_data = pd.concat([concat_data, df], axis=1)
        # save the concatted data to the corresponding federated folder
        concat_data.to_csv(os.path.join(specific_dataset_federated_path, "merged.tsv"), sep="\t")
