'''
This file is used to run the federated batch correction experiments.
It uses featurecloud_api_extension.py to run the experiments.
The Experiment class is used to define experiments that are then run.
To change the experiments done, define a new Experiment object
and add it to the experiments list.
See the Experiment class and the examples below for more information.
To change the postprocessing done, change the logic under the
###POSTPROCESSING comment.
The postprocessing assumes the use of the federated limma RBE app given in
the folder /batchcorrection of this repo.
'''
import os
import json
import zipfile
from typing import List
import pandas as pd
import time
from copy import deepcopy

import featurecloud_api_extension as util



### SETTINGS
## GENERALT SETTINGS
# The directory that contains ALL data needed in ANY experiment.
data_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'evaluation_data'))
# The base configuration file used in ALL experiments, all keys can be overwritten by the experiments.
base_config = {
    "flimmaBatchCorrection": {
        "data_filename": "data.tsv",
        "design_filename": "design.tsv",
        "expression_file_flag": True,
        "index_col": 0,
        "batch_col": None,
        "covariates": [],
        "separator": "\t",
        "design_separator": "\t",
        "normalizationMethod": None,
        "smpc": False,
        "min_samples": 0,
        "position": None,
    },
}

# The name of the docker image that contains the application to be tested
app_image_name = "bcorrect:latest"
#TODO: change this again

### SETTING THE EXPERIMENTS
experiments: List[util.Experiment] = list()
    # Here aree all the experiments defined, including the location of
    # the data, the configuration file changes and the app image name.
result_file_names: List[str] = list()
    # List of the names for the batch corrected result files generated
    # by the experiments
    # this list should have the same length as the experiments
    # and is iterated with the experiments together, so experiments[i]
    # will output to result_file_names[i]

# Small helper to add a position argument to the config files of an experiment
# based on the order in clients
def add_position_to_config(exp: util.Experiment):
    if not exp.config_file_changes:
        raise ValueError("ERROR: cannot add the position argument if exp.config_file_changes is not defined")
    if len(exp.config_file_changes) != len(exp.clients):
        raise RuntimeError("Cannot add a position, not all clients have config_file_changes")
    for idx, _ in enumerate(exp.clients):
        tmp = exp.config_file_changes[idx]
        tmp["flimmaBatchCorrection"]["position"] = idx
        exp.config_file_changes[idx] = deepcopy(tmp)

def set_smpc_true(exp: util.Experiment):
    if not exp.config_file_changes:
        raise ValueError("ERROR: cannot add the position argument if exp.config_file_changes is not defined")
    if len(exp.config_file_changes) != len(exp.clients):
        raise RuntimeError("Cannot add a position, not all clients have config_file_changes")
    for idx, _ in enumerate(exp.clients):
        tmp = exp.config_file_changes[idx]
        tmp["flimmaBatchCorrection"]["smpc"] = True
        exp.config_file_changes[idx] = deepcopy(tmp)


## SIMULATED
base_simulated_config_file_changes = {
    "flimmaBatchCorrection": {
        "data_filename": "intensities.tsv",
        "design_filename": "design.tsv",
        "covariates": ["A"],
        "index_col": "rowname"
    }
}
simulated_balanced_experiment = util.Experiment(
    name="Simulated Balanced",
    fc_data_dir=data_dir,
    clients=[
        os.path.join(data_dir, "simulated", "balanced", "before", "lab1"),
        os.path.join(data_dir, "simulated", "balanced", "before", "lab2"),
        os.path.join(data_dir, "simulated", "balanced", "before", "lab3"),
    ],
    app_image_name=app_image_name,
    config_files=[deepcopy(base_config) for _ in range(3)],
    config_file_changes=[deepcopy(base_simulated_config_file_changes) for _ in range(3)],
)
simulated_balanced_experiment_smpc = deepcopy(simulated_balanced_experiment)
set_smpc_true(simulated_balanced_experiment_smpc)
add_position_to_config(simulated_balanced_experiment_smpc)
add_position_to_config(simulated_balanced_experiment)
simulated_mildly_imbalanced_experiment = util.Experiment(
        name="Simulated Mildly Imbalanced",
        fc_data_dir=data_dir,
        clients=[os.path.join(data_dir, "simulated", "mild_imbalanced", "before", "lab1"),
                 os.path.join(data_dir, "simulated", "mild_imbalanced", "before", "lab2"),
                 os.path.join(data_dir, "simulated", "mild_imbalanced", "before", "lab3"),
        ],
        app_image_name=app_image_name,
        config_files=[deepcopy(base_config) for _ in range(3)],
        config_file_changes=[deepcopy(base_simulated_config_file_changes) for _ in range(3)],
)
simulated_mildly_imbalanced_experiment_smpc = deepcopy(simulated_mildly_imbalanced_experiment)
set_smpc_true(simulated_mildly_imbalanced_experiment_smpc)
add_position_to_config(simulated_mildly_imbalanced_experiment_smpc)
add_position_to_config(simulated_mildly_imbalanced_experiment)
simulated_strongly_imbalanced_experiment = util.Experiment(
        name="Simulated Strongly Imbalanced",
        fc_data_dir=data_dir,
        clients=[os.path.join(data_dir, "simulated", "strong_imbalanced", "before", "lab1"),
                 os.path.join(data_dir, "simulated", "strong_imbalanced", "before", "lab2"),
                 os.path.join(data_dir, "simulated", "strong_imbalanced", "before", "lab3"),
        ],
        app_image_name=app_image_name,
        config_files=[deepcopy(base_config) for _ in range(3)],
        config_file_changes=[deepcopy(base_simulated_config_file_changes) for _ in range(3)],
)
simulated_strongly_imbalanced_experiment_smpc = deepcopy(simulated_strongly_imbalanced_experiment)
set_smpc_true(simulated_strongly_imbalanced_experiment_smpc)
add_position_to_config(simulated_strongly_imbalanced_experiment_smpc)
add_position_to_config(simulated_strongly_imbalanced_experiment)

## SIMULATED ROTATION
simulated_rotation_balanced_experiment = util.Experiment(
    name="Simulated Rotation Balanced",
    fc_data_dir=data_dir,
    clients=[
        os.path.join(data_dir, "simulated_rotation", "balanced", "before", "lab1"),
        os.path.join(data_dir, "simulated_rotation", "balanced", "before", "lab2"),
        os.path.join(data_dir, "simulated_rotation", "balanced", "before", "lab3"),
    ],
    app_image_name=app_image_name,
    config_files=[deepcopy(base_config) for _ in range(3)],
    config_file_changes=[deepcopy(base_simulated_config_file_changes) for _ in range(3)],
)
simulated_rotation_balanced_experiment_smpc = deepcopy(simulated_rotation_balanced_experiment)
set_smpc_true(simulated_rotation_balanced_experiment_smpc)
add_position_to_config(simulated_rotation_balanced_experiment_smpc)
add_position_to_config(simulated_rotation_balanced_experiment)

simulated_rotation_mildly_imbalanced_experiment = util.Experiment(
        name="Simulated Rotation Mildly Imbalanced",
        fc_data_dir=data_dir,
        clients=[
            os.path.join(data_dir, "simulated_rotation", "mild_imbalanced", "before", "lab1"),
            os.path.join(data_dir, "simulated_rotation", "mild_imbalanced", "before", "lab2"),
            os.path.join(data_dir, "simulated_rotation", "mild_imbalanced", "before", "lab3"),
        ],
        app_image_name=app_image_name,
        config_files=[deepcopy(base_config) for _ in range(3)],
        config_file_changes=[deepcopy(base_simulated_config_file_changes) for _ in range(3)],
)
simulated_rotation_mildly_imbalanced_experiment_smpc = deepcopy(simulated_rotation_mildly_imbalanced_experiment)
set_smpc_true(simulated_rotation_mildly_imbalanced_experiment_smpc)
add_position_to_config(simulated_rotation_mildly_imbalanced_experiment_smpc)
add_position_to_config(simulated_rotation_mildly_imbalanced_experiment)

simulated_rotation_strongly_imbalanced_experiment = util.Experiment(
        name="Simulated Rotation Strongly Imbalanced",
        fc_data_dir=data_dir,
        clients=[
            os.path.join(data_dir, "simulated_rotation", "strong_imbalanced", "before", "lab1"),
            os.path.join(data_dir, "simulated_rotation", "strong_imbalanced", "before", "lab2"),
            os.path.join(data_dir, "simulated_rotation", "strong_imbalanced", "before", "lab3"),
        ],
        app_image_name=app_image_name,
        config_files=[deepcopy(base_config) for _ in range(3)],
        config_file_changes=[deepcopy(base_simulated_config_file_changes) for _ in range(3)],
)
simulated_rotation_strongly_imbalanced_experiment_smpc = deepcopy(simulated_rotation_strongly_imbalanced_experiment)
set_smpc_true(simulated_rotation_strongly_imbalanced_experiment_smpc)
add_position_to_config(simulated_rotation_strongly_imbalanced_experiment_smpc)
add_position_to_config(simulated_rotation_strongly_imbalanced_experiment)

## MICROBIOME
microbiomev2_config_file_changes = {
    "flimmaBatchCorrection": {
        "data_filename": "UQnorm_log_counts_for_corr.tsv",
        "design_filename": "design.tsv",
        "covariates": ["CRC"]
    }
}
microbiomev2_experiment = util.Experiment(
    name="Microbiome v2",
    fc_data_dir=data_dir,
    clients=[
        os.path.join(data_dir, "microbiome", "before", "China1"),
        os.path.join(data_dir, "microbiome", "before", "China3"),
        os.path.join(data_dir, "microbiome", "before", "China5"),
        os.path.join(data_dir, "microbiome", "before", "France1"),
        os.path.join(data_dir, "microbiome", "before", "Germany1"),
        os.path.join(data_dir, "microbiome", "before", "Germany2"),
    ],
    app_image_name=app_image_name,
    config_files=[deepcopy(base_config) for _ in range(6)],
    config_file_changes=[deepcopy(microbiomev2_config_file_changes) for _ in range(6)],
)
microbiomev2_experiment_smpc = deepcopy(microbiomev2_experiment)
set_smpc_true(microbiomev2_experiment_smpc)
add_position_to_config(microbiomev2_experiment_smpc)
add_position_to_config(microbiomev2_experiment)

## PROTEOMICS
proteomics_config_file_changes_base = {
    "flimmaBatchCorrection": {
        "data_filename": "intensities_log_UNION.tsv",
        "design_filename": "design.tsv",
        "covariates": ["Pyr"],
        "index_col": "rowname",
        "min_samples": 2
    }
}
proteomics_experiment = util.Experiment(
    name="Proteomics",
    fc_data_dir=data_dir,
    clients=[
        os.path.join(data_dir, "proteomics", "before", "lab_A"),
        os.path.join(data_dir, "proteomics", "before", "lab_B"),
        os.path.join(data_dir, "proteomics", "before", "lab_C"),
        os.path.join(data_dir, "proteomics", "before", "lab_D"),
        os.path.join(data_dir, "proteomics", "before", "lab_E"),
    ],
    app_image_name=app_image_name,
    config_files=[deepcopy(base_config) for _ in range(5)],
    config_file_changes=[deepcopy(proteomics_config_file_changes_base) for _ in range(5)],
)
proteomics_experiment_smpc = deepcopy(proteomics_experiment)
set_smpc_true(proteomics_experiment_smpc)
add_position_to_config(proteomics_experiment_smpc)
add_position_to_config(proteomics_experiment)

## PROTEOMICS MULTI_BATCH
proteomics_config_file_changes_multibatch = deepcopy(proteomics_config_file_changes_base)
proteomics_config_file_changes_multibatch["flimmaBatchCorrection"]["batch_col"] = "batch"
proteomics_config_file_changes_multibatch["flimmaBatchCorrection"]["min_samples"] = 0  # To avoid privacy error

proteomics_multibatch_experiment = util.Experiment(
    name="Proteomics Multi Batch",
    fc_data_dir=data_dir,
    clients=[
        os.path.join(data_dir, "proteomics_multibatch", "before", "center1"),
        os.path.join(data_dir, "proteomics_multibatch", "before", "center2"),
        os.path.join(data_dir, "proteomics_multibatch", "before", "center3"),
    ],
    app_image_name=app_image_name,
    config_files=[deepcopy(base_config) for _ in range(3)],
    config_file_changes=[deepcopy(proteomics_config_file_changes_multibatch) for _ in range(3)],
)
proteomics_multibatch_experiment_smpc = deepcopy(proteomics_multibatch_experiment)
set_smpc_true(proteomics_multibatch_experiment_smpc)
add_position_to_config(proteomics_multibatch_experiment_smpc)
add_position_to_config(proteomics_multibatch_experiment)

## MICROARRAY
base_microarray_config_file_changes = {
    "flimmaBatchCorrection": {
        "data_filename": "expr_for_correction_UNION.tsv",
        "design_filename": "design.tsv",
        "covariates": ["HGSC"],
        "index_col": "Gene",
        "min_samples": 2
    }
}
microarray_experiment = util.Experiment(
    name="Microarray",
    fc_data_dir=data_dir,
    clients=[
        os.path.join(data_dir, "microarray", "before", "GSE6008"),
        os.path.join(data_dir, "microarray", "before", "GSE14407"),
        os.path.join(data_dir, "microarray", "before", "GSE26712"),
        os.path.join(data_dir, "microarray", "before", "GSE38666"),
        os.path.join(data_dir, "microarray", "before", "GSE40595"),
        os.path.join(data_dir, "microarray", "before", "GSE69428"),
    ],
    app_image_name=app_image_name,
    config_files=[deepcopy(base_config) for _ in range(6)],
    config_file_changes=[deepcopy(base_microarray_config_file_changes) for _ in range(6)],
)
microarray_experiment_smpc = deepcopy(microarray_experiment)
set_smpc_true(microarray_experiment_smpc)
add_position_to_config(microarray_experiment_smpc)
add_position_to_config(microarray_experiment)

## ADD EXPERIMENTS, CHANGE HERE TO INCLUDE/EXCLUDE EXPERIMENTS
# Simulated
experiments.append(simulated_balanced_experiment)
result_file_names.append(os.path.join(data_dir, "simulated", "balanced", "after", "FedApp_corrected_data.tsv"))
# TODO: activate the experiments again
# experiments.append(simulated_balanced_experiment_smpc)
# result_file_names.append(os.path.join(data_dir, "simulated", "balanced", "after", "FedApp_corrected_data_smpc.tsv"))

# experiments.append(simulated_mildly_imbalanced_experiment)
# result_file_names.append(os.path.join(data_dir, "simulated", "mild_imbalanced", "after", "FedApp_corrected_data.tsv"))
# experiments.append(simulated_mildly_imbalanced_experiment_smpc)
# result_file_names.append(os.path.join(data_dir, "simulated", "mild_imbalanced", "after", "FedApp_corrected_data_smpc.tsv"))

# experiments.append(simulated_strongly_imbalanced_experiment)
# result_file_names.append(os.path.join(data_dir, "simulated", "strong_imbalanced", "after", "FedApp_corrected_data.tsv"))
# experiments.append(simulated_strongly_imbalanced_experiment_smpc)
# result_file_names.append(os.path.join(data_dir, "simulated", "strong_imbalanced", "after", "FedApp_corrected_data_smpc.tsv"))

# # Simulated rotation
# experiments.append(simulated_rotation_balanced_experiment)
# result_file_names.append(os.path.join(data_dir, "simulated_rotation", "balanced", "after", "FedApp_corrected_data.tsv"))
# experiments.append(simulated_rotation_balanced_experiment_smpc)
# result_file_names.append(os.path.join(data_dir, "simulated_rotation", "balanced", "after", "FedApp_corrected_data_smpc.tsv"))

# experiments.append(simulated_rotation_mildly_imbalanced_experiment)
# result_file_names.append(os.path.join(data_dir, "simulated_rotation", "mild_imbalanced", "after", "FedApp_corrected_data.tsv"))
# experiments.append(simulated_rotation_mildly_imbalanced_experiment_smpc)
# result_file_names.append(os.path.join(data_dir, "simulated_rotation", "mild_imbalanced", "after", "FedApp_corrected_data_smpc.tsv"))

# experiments.append(simulated_rotation_strongly_imbalanced_experiment)
# result_file_names.append(os.path.join(data_dir, "simulated_rotation", "strong_imbalanced", "after", "FedApp_corrected_data.tsv"))
# experiments.append(simulated_rotation_strongly_imbalanced_experiment_smpc)
# result_file_names.append(os.path.join(data_dir, "simulated_rotation", "strong_imbalanced", "after", "FedApp_corrected_data_smpc.tsv"))

# ## Microbiome v2
# experiments.append(microbiomev2_experiment)
# result_file_names.append(os.path.join(data_dir, "microbiome", "after", "FedApp_corrected_data.tsv"))
# experiments.append(microbiomev2_experiment_smpc)
# result_file_names.append(os.path.join(data_dir, "microbiome", "after", "FedApp_corrected_data_smpc.tsv"))

# ## Proteomics
# experiments.append(proteomics_experiment)
# result_file_names.append(os.path.join(data_dir, "proteomics", "after", "FedApp_corrected_data.tsv"))
# experiments.append(proteomics_experiment_smpc)
# result_file_names.append(os.path.join(data_dir, "proteomics", "after", "FedApp_corrected_data_smpc.tsv"))

# ## Proteomics Multi Batch
# experiments.append(proteomics_multibatch_experiment)
# result_file_names.append(os.path.join(data_dir, "proteomics_multibatch", "after", "FedApp_corrected_data.tsv"))
# experiments.append(proteomics_multibatch_experiment_smpc)
# result_file_names.append(os.path.join(data_dir, "proteomics_multibatch", "after", "FedApp_corrected_data_smpc.tsv"))

# ## Microarray
# experiments.append(microarray_experiment)
# result_file_names.append(os.path.join(data_dir, "microarray", "after", "FedApp_corrected_data.tsv"))
# experiments.append(microarray_experiment_smpc)
# result_file_names.append(os.path.join(data_dir, "microarray", "after", "FedApp_corrected_data_smpc.tsv"))

### ACTUAL PROGRAM, NO NEED TO CHANGE IF A DIFFERENT EXPERIMENT WANTS TO BE RUN

if len(experiments) != len(result_file_names):
    raise RuntimeError("Number of experiments and result file names do not match, please fix this!")

### MAIN
# Starts the FeatureCloud controler and runs the experiments
# The results are then postprocessed and saved in the result files.
# This postprocessing simply takes the individual results (assumes the default
# output file name used by the batch correction app (federated limma RBE)
# of this repo) and concats them.

# Run the experiments
for exp, result_filename in zip(experiments, result_file_names):
    ### RUN THE FEATURECLOUD EXPERIMENT AND EXTARCT INDIVIDUAL RESULTS
    print(f"Starting experiment:\n{exp}")
    try:
        result_files_zipped, _, _ = exp.run_test()
    except Exception as e:
        print(f"Experiment could not be started or aborted too many times! Error: \n{e}")
        print("_______________¡FAILED_EXPERIMENT!_______________")
        continue
    print(f"Experiment finished successfully! Result files: {result_files_zipped}")

    ### Postprocessing
    time.sleep(10)
    print("Starting postprocessing...")
    result_folder = os.path.dirname(result_filename)
    print("Results will be saved in: ", result_folder)
    individual_results_dir = os.path.join(result_folder, "individual_results")
    if not os.path.exists(individual_results_dir):
        os.makedirs(individual_results_dir)

    # Derive client names from the experiment's client paths
    client_names = [os.path.basename(client_path) for client_path in exp.clients]

    # Extract all output files from each client's zip into individual_results/<clientname>/
    for idx, (zipfilename, client_name) in enumerate(zip(result_files_zipped, client_names)):
        while True:
            if os.path.exists(os.path.join(data_dir, "tests", "tests", zipfilename)):
                break
            print(f"Waiting for file {zipfilename} to be available...")
            time.sleep(5)
        client_result_dir = os.path.join(individual_results_dir, client_name)
        if not os.path.exists(client_result_dir):
            os.makedirs(client_result_dir)
        with zipfile.ZipFile(os.path.join(data_dir, "tests", "tests", zipfilename), 'r') as zip_ref:
            zip_names = zip_ref.namelist()
            for fname in ["only_batch_corrected_data.csv", "full_corrected_data.csv", "report.txt"]:
                if fname in zip_names:
                    # extract() preserves the filename at the root of client_result_dir
                    zip_ref.extract(fname, client_result_dir)
                # required file so we throw an error
                elif fname == "full_corrected_data.csv":
                    raise RuntimeError(
                        f"'{fname}' not found in result zip for client '{client_name}'. "
                        f"Did the batch effect correction app change its output filenames? "
                        f"Files present in zip: {zip_names}"
                    )
                #
                else:
                    print(f"WARNING: '{fname}' not found in zip for client '{client_name}', skipping.")

    # Update the datainfo.json in the result folder so each cohort points to
    # individual_results/<clientname>/full_corrected_data.csv
    datainfo_path = os.path.join(result_folder, "datainfo.json")
    if os.path.exists(datainfo_path):
        with open(datainfo_path, "r") as f:
            datainfo = json.load(f)
        for cohort in datainfo.get("cohorts", []):
            cohort["folder"] = os.path.join("individual_results", cohort["name"])
            cohort["datafile"] = "full_corrected_data.csv"
        # full_corrected_data.csv is written by the app using client.separator,
        # which comes from the experiment's config. Derive it from the first client's
        # merged config (base + changes) so it is never hardcoded here.
        client0_config = deepcopy(exp.config_files[0])
        if exp.config_file_changes:
            for key, val in exp.config_file_changes[0].get("flimmaBatchCorrection", {}).items():
                client0_config["flimmaBatchCorrection"][key] = val
        output_separator = client0_config["flimmaBatchCorrection"].get("separator", "\t")
        datainfo["csv_separator"] = output_separator
        with open(datainfo_path, "w") as f:
            json.dump(datainfo, f, indent=2)
        print(f"Updated datainfo.json at {datainfo_path}")
    else:
        print(f"WARNING: No datainfo.json found at {datainfo_path}, skipping datainfo update.")

    ### CONCAT THE RESULTS AND PRODUCE FINAL MERGED RESULT
    print("Postprocessing finished! Concatenating results...")
    final_df = None
    for client_name in client_names:
        result_file = os.path.join(individual_results_dir, client_name, "only_batch_corrected_data.csv")
        if not os.path.exists(result_file):
            print(f"WARNING: Result file not found for client '{client_name}': {result_file}")
            continue
        client_df = pd.read_csv(result_file, sep="\t", index_col=0)
        if final_df is None:
            final_df = client_df
        else:
            final_df = pd.concat([final_df, client_df], axis=1)
    # Save the final df
    if final_df is not None:
        final_df.to_csv(os.path.join(result_filename), sep="\t")
    else:
        raise RuntimeError("No data found in the test results!")

    print("_______________¡SUCCESSFUL_EXPERIMENT!_______________")
