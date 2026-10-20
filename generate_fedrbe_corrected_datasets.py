'''
This file is used to run the federated batch correction experiments.
It uses featurecloud_api_extension.py to run the experiments.
The Experiment class is used to define experiments that are then run.
To add a new dataset, define a new Experiment object
and add it to the experiments list.
To skip an experiment, comment out its (Experiment, output path) entry below:
## ADD EXPERIMENTS, CHANGE HERE TO INCLUDE/EXCLUDE EXPERIMENTS
See the Experiment class and the examples below for more information.
To change the postprocessing done, change the logic under the
###POSTPROCESSING comment.
The postprocessing assumes the use of the federated limma RBE app given in
the folder /batchcorrection of this repo.
No-covariate variants run with and without SMPC alongside the other experiments.
Results use after/label_aware/ and after/no_label/ folders (before Quartet modality names).
'''
import os
import sys
from pathlib import Path
import json
import zipfile
from typing import List
import pandas as pd
import time
from copy import deepcopy

from evaluation_utils import featurecloud_api_extension as util
from evaluation_utils.real_datasets_utils import (
    read_correction_matrix, validate_corrected_matrix,
)

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


SIMULATED_SINGLE_RUN = 1


def _path_is_under(path: str, parent: str) -> bool:
    path_abs = os.path.abspath(path)
    parent_abs = os.path.abspath(parent)
    return os.path.commonpath([path_abs, parent_abs]) == parent_abs


def ensure_simulated_before_inputs_are_single_run(run_number: int = SIMULATED_SINGLE_RUN):
    """Abort if simulated single-run fedRBE inputs no longer point to run 1."""
    simulated_dir = os.path.join(data_dir, "simulated")
    checked_modes = set()
    for exp, _ in experiments:
        if not _path_is_under(exp.fc_data_dir, simulated_dir):
            continue

        mode = os.path.basename(exp.fc_data_dir)
        if mode in checked_modes:
            continue
        checked_modes.add(mode)

        mode_dir = exp.fc_data_dir
        metadata_path = os.path.join(mode_dir, "all_metadata.tsv")
        intensities_path = os.path.join(
            mode_dir,
            "before",
            "intermediate",
            f"{run_number}_intensities_data.tsv",
        )
        if not os.path.exists(metadata_path):
            raise RuntimeError(f"Missing simulated metadata file: {metadata_path}")
        if not os.path.exists(intensities_path):
            raise RuntimeError(
                f"Missing simulated run-{run_number} input file: {intensities_path}. "
                "Run evaluation_data/simulated/00_data_simulation.ipynb or restore the committed data."
            )

        metadata = pd.read_csv(metadata_path, sep="\t", index_col=0)
        intensities = pd.read_csv(intensities_path, sep="\t", index_col="rowname")

        for client_dir in exp.clients:
            lab_name = os.path.basename(client_dir)
            metadata_lab = metadata.loc[metadata["lab"] == lab_name]
            sample_columns = metadata_lab["file"].tolist()
            missing_columns = sorted(set(sample_columns).difference(intensities.columns))
            if missing_columns:
                raise RuntimeError(
                    f"Missing samples in {intensities_path} for {mode}/{lab_name}: {missing_columns}"
                )

            client_intensities_path = os.path.join(client_dir, "intensities.tsv")
            if not os.path.exists(client_intensities_path):
                raise RuntimeError(f"Missing simulated client input file: {client_intensities_path}")

            current_intensities = pd.read_csv(client_intensities_path, sep="\t", index_col="rowname")
            if not current_intensities.equals(intensities.loc[:, sample_columns]):
                raise RuntimeError(
                    f"Simulated input {client_intensities_path} is not run {run_number}. "
                    "Run the final 'Run test central correction' cell in "
                    "evaluation_data/simulated/01_data_prep_and_central_RBE.ipynb "
                    "or restore the committed simulated before/ files before running this script."
                )


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
ccRCC_proteomics_experiment = util.Experiment(
    name="ccRCC proteomics",
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
ccRCC_proteomics_experiment_smpc = deepcopy(ccRCC_proteomics_experiment)
set_smpc_true(ccRCC_proteomics_experiment_smpc)
add_position_to_config(ccRCC_proteomics_experiment_smpc)
add_position_to_config(ccRCC_proteomics_experiment)

## Quartet MULTIOMICS (Quartet full Transcriptomics, Proteomics, Metabolomics)
# Toggle for the synthetic client_04_L03_L14 (L03 + L14 fold-in). It must match
# INCLUDE_CLIENT_04 in notebooks 01-03 of `evaluation_data/quartet_multiomics/`.
# Default False -- only the three real cross-modality clients run.
INCLUDE_MULTIOMICS_CLIENT_04 = False

multiomics_modalities = [
    "Transcriptomics",  "Proteomics",  "Metabolomics"]
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
multiomics_clients = ["client_01_L01", "client_02_L02", "client_03_L05_L04"]
if INCLUDE_MULTIOMICS_CLIENT_04:
    multiomics_clients.append("client_04_L03_L14")
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
        name=f"Quartet Multiomics {modality}",
        fc_data_dir=os.path.join(data_dir, "quartet_multiomics"),
        clients=[
            os.path.join(data_dir, "quartet_multiomics", "before", modality, client)
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
    multiomics_experiment_smpc.name = f"Quartet Multiomics {multiomics_modality} (SMPC)"
    set_smpc_true(multiomics_experiment_smpc)


def without_covariates(baseline: util.Experiment) -> util.Experiment:
    """Clone a selected experiment, retaining its label-aware privacy threshold."""
    experiment = deepcopy(baseline)
    configs = experiment.config_files
    for config, changes in zip(configs, experiment.config_file_changes or [{}] * len(configs)):
        config["flimmaBatchCorrection"].update(changes.get("flimmaBatchCorrection", {}))
    settings = [config["flimmaBatchCorrection"] for config in configs]
    common_covariates = set.intersection(*(set(c["covariates"]) for c in settings))
    batches = 0
    for client, config in zip(experiment.clients, settings):
        if config.get("batch_col"):
            design = pd.read_csv(Path(client) / config["design_filename"],
                                 sep=config.get("design_separator", "\t"), index_col=0)
            batches += design[config["batch_col"]].astype(str).nunique()
        else:
            batches += 1
    for config in settings:
        # Removing covariates must not loosen the app's feature privacy filter.
        config["min_samples"] = max(batches + len(common_covariates) + 1,
                                    config.get("min_samples", 0))
        config["covariates"] = []
    experiment.config_file_changes = None
    experiment.name += " (no covariates)"
    return experiment


## ADD EXPERIMENTS, CHANGE HERE TO INCLUDE/EXCLUDE EXPERIMENTS
# Each entry pairs an experiment with its output path relative to evaluation_data/.
# Comment/uncomment a complete entry to disable/enable a run.
experiments = [
    # Simulated
    (simulated_balanced_experiment, "simulated/balanced/after/FedApp_corrected_data.tsv"),
    (simulated_balanced_experiment_smpc, "simulated/balanced/after/FedApp_corrected_data_smpc.tsv"),
    (simulated_mildly_imbalanced_experiment, "simulated/mild_imbalanced/after/FedApp_corrected_data.tsv"),
    (simulated_mildly_imbalanced_experiment_smpc, "simulated/mild_imbalanced/after/FedApp_corrected_data_smpc.tsv"),
    (simulated_strongly_imbalanced_experiment, "simulated/strong_imbalanced/after/FedApp_corrected_data.tsv"),
    (simulated_strongly_imbalanced_experiment_smpc, "simulated/strong_imbalanced/after/FedApp_corrected_data_smpc.tsv"),

    # E. coli
    (ecoli_experiment, "ecoli/after/label_aware/FedApp_corrected_data.tsv"),
    (ecoli_experiment_smpc, "ecoli/after/label_aware/FedApp_corrected_data_smpc.tsv"),
    (without_covariates(ecoli_experiment), "ecoli/after/no_label/FedApp_corrected_data.tsv"),
    (without_covariates(ecoli_experiment_smpc), "ecoli/after/no_label/FedApp_corrected_data_smpc.tsv"),

    # Ovarian cancer
    (ovarian_cancer_experiment, "ovarian_cancer/after/label_aware/FedApp_corrected_data.tsv"),
    (ovarian_cancer_experiment_smpc, "ovarian_cancer/after/label_aware/FedApp_corrected_data_smpc.tsv"),
    (without_covariates(ovarian_cancer_experiment), "ovarian_cancer/after/no_label/FedApp_corrected_data.tsv"),
    (without_covariates(ovarian_cancer_experiment_smpc), "ovarian_cancer/after/no_label/FedApp_corrected_data_smpc.tsv"),

    # ccRCC Proteomics
    (ccRCC_proteomics_experiment, "ccRCC_studies/after/label_aware/FedApp_corrected_data.tsv"),
    (ccRCC_proteomics_experiment_smpc, "ccRCC_studies/after/label_aware/FedApp_corrected_data_smpc.tsv"),
    (without_covariates(ccRCC_proteomics_experiment), "ccRCC_studies/after/no_label/FedApp_corrected_data.tsv"),
    (without_covariates(ccRCC_proteomics_experiment_smpc), "ccRCC_studies/after/no_label/FedApp_corrected_data_smpc.tsv"),
]

# Comment modalities in multiomics_modalities above to select individual omics.
for modality in multiomics_modalities:
    experiments.extend([
        (multiomics_experiments[modality], f"quartet_multiomics/after/label_aware/{modality}/FedApp_corrected_data.tsv"),
        (multiomics_experiments_smpc[modality], f"quartet_multiomics/after/label_aware/{modality}/FedApp_corrected_data_smpc.tsv"),
        (without_covariates(multiomics_experiments[modality]), f"quartet_multiomics/after/no_label/{modality}/FedApp_corrected_data.tsv"),
        (without_covariates(multiomics_experiments_smpc[modality]), f"quartet_multiomics/after/no_label/{modality}/FedApp_corrected_data_smpc.tsv"),
    ])


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit("Select experiments in the experiment list; this script takes no arguments.")

    ### ACTUAL PROGRAM, NO NEED TO CHANGE IF A DIFFERENT EXPERIMENT WANTS TO BE RUN

    ensure_simulated_before_inputs_are_single_run()

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
    for exp, result_filename in experiments:
        result_filename = os.path.join(data_dir, result_filename)
        ### RUN THE FEATURECLOUD EXPERIMENT AND EXTARCT INDIVIDUAL RESULTS
        print(f"Starting experiment:\n{exp}")
        config_paths = [Path(client) / "config.yml" for client in exp.clients]
        original_configs = {path: path.read_bytes() if path.exists() else None for path in config_paths}
        try:
            if exp.fc_data_dir != active_fc_data_dir:
                exp._startup()
                active_fc_data_dir = exp.fc_data_dir
            result_files_zipped, _, _ = exp.run_test()
        except Exception as e:
            print(f"Experiment could not be started or aborted too many times! Error: \n{e}")
            print("_______________¡FAILED_EXPERIMENT!_______________")
            continue
        finally:
            for path, content in original_configs.items():
                if content is None:
                    path.unlink(missing_ok=True)
                else:
                    path.write_bytes(content)
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

        if len(result_files_zipped) != len(exp.clients):
            raise ValueError("Correction requires one result ZIP per client")

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
                with zip_ref.open("only_batch_corrected_data.csv") as handle:
                    samples = pd.read_csv(handle, sep="\t", index_col=0, nrows=0).columns
                details = exp.config_files[idx]["flimmaBatchCorrection"]
                expected = pd.read_csv(Path(exp.clients[idx]) / details["data_filename"],
                                       sep=details.get("separator", "\t"), index_col=0, nrows=0).columns
                if not samples.equals(expected):
                    raise ValueError(f"Incomplete correction samples for {client_name}")
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

        # Write datainfo.json in the result folder from the clients' input descriptor,
        # so each cohort points to individual_results/<clientname>/full_corrected_data.csv
        # and keeps its original design file in every correction arm.
        source_path = Path(exp.clients[0]).parent / "datainfo.json"
        with open(source_path, "r") as f:
            datainfo = json.load(f)
        for cohort in datainfo["cohorts"]:
            design = source_path.parent / cohort["folder"] / cohort["designfile"]["filename"]
            cohort["folder"] = os.path.join("individual_results", cohort["name"])
            cohort["designfile"]["filename"] = os.path.relpath(
                design, os.path.join(result_folder, cohort["folder"]))
        # full_corrected_data.csv is written by the app using client.separator,
        # which comes from the experiment's config. Derive it from the first client's
        # merged config (base + changes) so it is never hardcoded here.
        client0_config = deepcopy(exp.config_files[0])
        if exp.config_file_changes:
            for key, val in exp.config_file_changes[0].get("flimmaBatchCorrection", {}).items():
                client0_config["flimmaBatchCorrection"][key] = val
        output_separator = client0_config["flimmaBatchCorrection"].get("separator", "\t")
        datainfo["datafile"] = {
            "filename": "full_corrected_data.csv", "separator": output_separator,
            "rotation": "features x samples", "samplename_column": None,
            "featurename_column": None,
        }
        datainfo_path = os.path.join(result_folder, "datainfo.json")
        with open(datainfo_path, "w") as f:
            json.dump(datainfo, f, indent=2)
        print(f"Wrote datainfo.json at {datainfo_path}")

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
            output = Path(result_filename)
            arm_folder = next((parent for parent in output.parents if parent.name == "no_label"), None)
            reference = (arm_folder.with_name("label_aware") / output.relative_to(arm_folder)
                         if arm_folder is not None else None)
            if reference is not None and reference.exists():
                final_df.index = final_df.index.astype(str)
                final_df.columns = final_df.columns.astype(str)
                final_df = validate_corrected_matrix(final_df, read_correction_matrix(reference))
            final_df.to_csv(os.path.join(result_filename), sep="\t")
        else:
            raise RuntimeError("No data found in the test results!")

        print("_______________¡SUCCESSFUL_EXPERIMENT!_______________")
