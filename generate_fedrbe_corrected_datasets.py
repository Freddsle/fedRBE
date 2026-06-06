'''
This file is used to run the federated batch correction experiments.
It uses featurecloud_api_extension.py to run the experiments.
The Experiment class is used to define experiments that are then run.
To add a new dataset, define a new Experiment object
and add it to the experiments list.
To skip a certain dataset, simply comment out the addition of the corresponding
Experiment object to the experiments list below this comment:
## ADD EXPERIMENTS, CHANGE HERE TO INCLUDE/EXCLUDE EXPERIMENTS
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

from evaluation_utils import featurecloud_api_extension as util



### SETTINGS
## GENERALT SETTINGS
# The directory that contains ALL data needed in ANY experiment.
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'evaluation_data'))
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
app_image_name = "featurecloud.ai/bcorrect:latest"

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
    fc_data_dir=os.path.join(data_dir, "simulated", "balanced"),
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
        fc_data_dir=os.path.join(data_dir, "simulated", "mild_imbalanced"),
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
        fc_data_dir=os.path.join(data_dir, "simulated", "strong_imbalanced"),
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

## PROTEOMICS
ecoli_config_file_changes_base = {
    "flimmaBatchCorrection": {
        "data_filename": "intensities_log_UNION.tsv",
        "design_filename": "design.tsv",
        "covariates": ["Pyr"],
        "index_col": "rowname",
        "min_samples": 2
    }
}
ecoli_experiment = util.Experiment(
    name="E. coli",
    fc_data_dir=os.path.join(data_dir, "ecoli"),
    clients=[
        os.path.join(data_dir, "ecoli", "before", "lab_A"),
        os.path.join(data_dir, "ecoli", "before", "lab_B"),
        os.path.join(data_dir, "ecoli", "before", "lab_C"),
        os.path.join(data_dir, "ecoli", "before", "lab_D"),
        os.path.join(data_dir, "ecoli", "before", "lab_E"),
    ],
    app_image_name=app_image_name,
    config_files=[deepcopy(base_config) for _ in range(5)],
    config_file_changes=[deepcopy(ecoli_config_file_changes_base) for _ in range(5)],
)
ecoli_experiment_smpc = deepcopy(ecoli_experiment)
set_smpc_true(ecoli_experiment_smpc)
add_position_to_config(ecoli_experiment_smpc)
add_position_to_config(ecoli_experiment)

## PROTEOMICS MULTI_BATCH (Quartet, PXD045065)
# 4 centers (labs): APT and FDU have 2 internal batches each (DDA + DIA),
# NVG and BGI have 1 batch each (single batch name for all samples).
# All centers use batch_col: batch.
ecoli_config_file_changes_quartet = deepcopy(ecoli_config_file_changes_base)
ecoli_config_file_changes_quartet["flimmaBatchCorrection"]["covariates"] = ["D6", "F7", "M8"]
ecoli_config_file_changes_quartet["flimmaBatchCorrection"]["batch_col"] = "batch"
ecoli_config_file_changes_quartet["flimmaBatchCorrection"]["min_samples"] = 0  # To avoid privacy error

quartet_centers = ["APT", "FDU", "NVG", "BGI"]
quartet_experiment = util.Experiment(
    name="Quartet",
    fc_data_dir=os.path.join(data_dir, "quartet"),
    clients=[
        os.path.join(data_dir, "quartet", "before", center)
        for center in quartet_centers
    ],
    app_image_name=app_image_name,
    config_files=[deepcopy(base_config) for _ in range(4)],
    config_file_changes=[deepcopy(ecoli_config_file_changes_quartet) for _ in range(4)],
)
quartet_experiment_smpc = deepcopy(quartet_experiment)
set_smpc_true(quartet_experiment_smpc)
add_position_to_config(quartet_experiment_smpc)
add_position_to_config(quartet_experiment)

## MICROARRAY
ovarian_cancer_config_file_changes = {
    "flimmaBatchCorrection": {
        "data_filename": "expr_for_correction_UNION.tsv",
        "design_filename": "design.tsv",
        "covariates": ["HGSC"],
        "index_col": "Gene",
        "min_samples": 2
    }
}
ovarian_cancer_experiment = util.Experiment(
    name="Ovarian cancer",
    fc_data_dir=os.path.join(data_dir, "ovarian_cancer"),
    clients=[
        os.path.join(data_dir, "ovarian_cancer", "before", "GSE6008"),
        os.path.join(data_dir, "ovarian_cancer", "before", "GSE14407"),
        os.path.join(data_dir, "ovarian_cancer", "before", "GSE26712"),
        os.path.join(data_dir, "ovarian_cancer", "before", "GSE38666"),
        os.path.join(data_dir, "ovarian_cancer", "before", "GSE40595"),
        os.path.join(data_dir, "ovarian_cancer", "before", "GSE69428"),
    ],
    app_image_name=app_image_name,
    config_files=[deepcopy(base_config) for _ in range(6)],
    config_file_changes=[deepcopy(ovarian_cancer_config_file_changes) for _ in range(6)],
)
ovarian_cancer_experiment_smpc = deepcopy(ovarian_cancer_experiment)
set_smpc_true(ovarian_cancer_experiment_smpc)
add_position_to_config(ovarian_cancer_experiment_smpc)
add_position_to_config(ovarian_cancer_experiment)

## ccRCC PROTEOMICS (3 studies: PDC000127, PXD030344, PXD042844)
# Condition column derived from binary Normal/Tumor columns (see prepare_ccRCC_data.py).
# Data matrices use 'Gene' as the row index column.
ccRCC_config_file_changes_base = {
    "flimmaBatchCorrection": {
        "data_filename": "report_filtered.tsv",
        "design_filename": "design.tsv",
        "covariates": ["Condition"],
        "index_col": "Gene",
        "min_samples": 2
    }
}
ccRCC_ecoli_experiment = util.Experiment(
    name="ccRCC",
    fc_data_dir=os.path.join(data_dir, "ccRCC_studies"),
    clients=[
        os.path.join(data_dir, "ccRCC_studies", "before", "PDC000127"),
        os.path.join(data_dir, "ccRCC_studies", "before", "PXD030344"),
        os.path.join(data_dir, "ccRCC_studies", "before", "PXD042844"),
    ],
    app_image_name=app_image_name,
    config_files=[deepcopy(base_config) for _ in range(3)],
    config_file_changes=[deepcopy(ccRCC_config_file_changes_base) for _ in range(3)],
)
ccRCC_ecoli_experiment_smpc = deepcopy(ccRCC_ecoli_experiment)
set_smpc_true(ccRCC_ecoli_experiment_smpc)
add_position_to_config(ccRCC_ecoli_experiment_smpc)
add_position_to_config(ccRCC_ecoli_experiment)

## MULTIOMICS (Quartet full Transcriptomics, Proteomics, Metabolomics)
# Toggle for the synthetic client_04_L03_L14 (L03 + L14 fold-in). Driven from
# the same single source (`fedrbe_multiomics_utils.INCLUDE_CLIENT_04`) used by
# notebooks 01--04 of `evaluation_data/multiomics/`, so the entire pipeline
# switches together. Default False -- only the three real cross-modality
# clients run.
import sys as _sys
_sys.path.insert(0, os.path.join(data_dir, "multiomics"))
from fedrbe_multiomics_utils import (  # noqa: E402
    CLIENT_NAMES as _MULTIOMICS_ACTIVE_CLIENTS,
    INCLUDE_CLIENT_04 as _INCLUDE_CLIENT_04,
)

multiomics_modalities = [
    "Transcriptomics",  "Proteomics",  "Metabolomics"]
# Full federation reference batches (last batch of the last client when each
# client is the federation tail). Pruned below to whichever client is last.
_ALL_MULTIOMICS_CLIENTS = [
    "client_01_L01",
    "client_02_L02",
    "client_03_L05_L04",
    "client_04_L03_L14",
]
_MULTIOMICS_REFERENCE_BATCH_BY_LAST_CLIENT = {
    "client_03_L05_L04": {
        "Transcriptomics": "R_ILM_L5_B2",
        "Proteomics": "FDU_QE-HFX_4",
        "Metabolomics": "U_L5_01",
    },
    "client_04_L03_L14": {
        "Transcriptomics": "R_BGI_L3_B1",
        "Proteomics": "TMO_QE-HFX_1",
        "Metabolomics": "U_L3_02",
    },
}
multiomics_clients = list(_MULTIOMICS_ACTIVE_CLIENTS)
multiomics_reference_batches = _MULTIOMICS_REFERENCE_BATCH_BY_LAST_CLIENT[
    multiomics_clients[-1]
]
multiomics_config_file_changes_base = {
    "flimmaBatchCorrection": {
        "data_filename": "intensities_log_UNION.tsv",
        "design_filename": "design.tsv",
        "covariates": ["D5", "F7", "M8"],
        "index_col": "rowname",
        "min_samples": 0,
        "batch_col": "batch",
        "smpc": False,
    }
}


def build_multiomics_config_file_changes(modality: str) -> List[dict]:
    config_file_changes = []
    for idx, _ in enumerate(multiomics_clients):
        changes = deepcopy(multiomics_config_file_changes_base)
        changes["flimmaBatchCorrection"]["position"] = idx
        changes["flimmaBatchCorrection"]["reference_batch"] = (
            multiomics_reference_batches[modality]
            if idx == len(multiomics_clients) - 1
            else False
        )
        config_file_changes.append(changes)
    return config_file_changes


multiomics_experiments = {
    modality: util.Experiment(
        name=f"Multiomics {modality}",
        fc_data_dir=os.path.join(data_dir, "multiomics"),
        clients=[
            os.path.join(data_dir, "multiomics", "before", modality, client)
            for client in multiomics_clients
        ],
        app_image_name=app_image_name,
        config_files=[deepcopy(base_config) for _ in multiomics_clients],
        config_file_changes=build_multiomics_config_file_changes(modality),
    )
    for modality in multiomics_modalities
}
multiomics_experiments_smpc = deepcopy(multiomics_experiments)
for multiomics_modality, multiomics_experiment_smpc in multiomics_experiments_smpc.items():
    multiomics_experiment_smpc.name = f"Multiomics {multiomics_modality} (SMPC)"
    set_smpc_true(multiomics_experiment_smpc)

## ADD EXPERIMENTS, CHANGE HERE TO INCLUDE/EXCLUDE EXPERIMENTS
# # Simulated
# experiments.append(simulated_balanced_experiment)
# result_file_names.append(os.path.join(data_dir, "simulated", "balanced", "after", "FedApp_corrected_data.tsv"))
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

# ## E. coli
# experiments.append(ecoli_experiment)
# result_file_names.append(os.path.join(data_dir, "ecoli", "after", "FedApp_corrected_data.tsv"))
# experiments.append(ecoli_experiment_smpc)
# result_file_names.append(os.path.join(data_dir, "ecoli", "after", "FedApp_corrected_data_smpc.tsv"))

# ## Quartet
# experiments.append(quartet_experiment)
# result_file_names.append(os.path.join(data_dir, "quartet", "after", "FedApp_corrected_data.tsv"))
# experiments.append(quartet_experiment_smpc)
# result_file_names.append(os.path.join(data_dir, "quartet", "after", "FedApp_corrected_data_smpc.tsv"))

# ## Ovarian cancer
# experiments.append(ovarian_cancer_experiment)
# result_file_names.append(os.path.join(data_dir, "ovarian_cancer", "after", "FedApp_corrected_data.tsv"))
# experiments.append(ovarian_cancer_experiment_smpc)
# result_file_names.append(os.path.join(data_dir, "ovarian_cancer", "after", "FedApp_corrected_data_smpc.tsv"))

# ## ccRCC Proteomics
# experiments.append(ccRCC_ecoli_experiment)
# result_file_names.append(os.path.join(data_dir, "ccRCC_studies", "after", "FedApp_corrected_data.tsv"))
# experiments.append(ccRCC_ecoli_experiment_smpc)
# result_file_names.append(os.path.join(data_dir, "ccRCC_studies", "after", "FedApp_corrected_data_smpc.tsv"))

## Multiomics
for multiomics_modality in multiomics_modalities:
    experiments.append(multiomics_experiments[multiomics_modality])
    result_file_names.append(
        os.path.join(
            data_dir,
            "multiomics",
            "after",
            multiomics_modality,
            "FedApp_corrected_data.tsv",
        )
    )
    experiments.append(multiomics_experiments_smpc[multiomics_modality])
    result_file_names.append(
        os.path.join(
            data_dir,
            "multiomics",
            "after",
            multiomics_modality,
            "FedApp_corrected_data_smpc.tsv",
        )
    )

### ACTUAL PROGRAM, NO NEED TO CHANGE IF A DIFFERENT EXPERIMENT WANTS TO BE RUN

if len(experiments) != len(result_file_names):
    raise RuntimeError("Number of experiments and result file names do not match, please fix this!")

### MAIN
# Starts the FeatureCloud controler and runs the experiments
# The results are then postprocessed and saved in the result files.
# This postprocessing simply takes the individual results (assumes the default
# output file name used by the batch correction app (federated limma RBE)
# of this repo) and concats them.

# The FeatureCloud controller keeps the data directory it was started with.
# Restart it when an experiment switches to a different data directory.
active_fc_data_dir = None

# Run the experiments
for exp, result_filename in zip(experiments, result_file_names):
    ### RUN THE FEATURECLOUD EXPERIMENT AND EXTARCT INDIVIDUAL RESULTS
    print(f"Starting experiment:\n{exp}")
    try:
        if exp.fc_data_dir != active_fc_data_dir:
            exp._startup()
            active_fc_data_dir = exp.fc_data_dir
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
    for idx, (zip_path, client_name) in enumerate(zip(result_files_zipped, client_names)):
        for _ in range(60):
            if os.path.exists(zip_path):
                break
            print(f"Waiting for file {zip_path} to be available...")
            time.sleep(5)
        else:
            raise RuntimeError(f"Zip file not found after timeout: {zip_path}")
        client_result_dir = os.path.join(individual_results_dir, client_name)
        if not os.path.exists(client_result_dir):
            os.makedirs(client_result_dir)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
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
        # full_corrected_data.csv is written by the app using client.separator,
        # which comes from the experiment's config. Derive it from the first client's
        # merged config (base + changes) so it is never hardcoded here.
        client0_config = deepcopy(exp.config_files[0])
        if exp.config_file_changes:
            for key, val in exp.config_file_changes[0].get("flimmaBatchCorrection", {}).items():
                client0_config["flimmaBatchCorrection"][key] = val
        output_separator = client0_config["flimmaBatchCorrection"].get("separator", "\t")
        datainfo["datafile"]["filename"] = "full_corrected_data.csv"
        datainfo["datafile"]["separator"] = output_separator
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
