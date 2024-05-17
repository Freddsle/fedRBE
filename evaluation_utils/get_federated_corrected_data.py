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
        "covariates": [],
        "separator": "\t",
        "design_separator": "\t",
        "normalizationMethod": None,
        "smpc": False,
        "min_samples": 0,
    },
}

# The name of the docker image that contains the application to be tested  
app_image_name = "bcorrect"

### SETTING THE EXPERIMENTS
experiments: List[util.Experiment] = list()
result_file_names: List[str] = list()

## MICROBIOME
microbiome_config_file_changes_logmin = \
    {"flimmaBatchCorrection.data_filename": "msp_counts_norm_logmin_5C.tsv", 
     "flimmaBatchCorrection.design_filename": "design_5C.tsv", 
     "flimmaBatchCorrection.covariates": ["CRC"]}
microbiome_experiment_logmin = util.Experiment(
        clients=[os.path.join(data_dir, "microbiome", "before", "PRJEB10878"),
                os.path.join(data_dir, "microbiome", "before", "PRJEB27928"),
                os.path.join(data_dir, "microbiome", "before", "PRJEB6070"),
                os.path.join(data_dir, "microbiome", "before", "PRJNA429097"),
                os.path.join(data_dir, "microbiome", "before", "PRJNA731589"),
        ],
        app_image_name=app_image_name,
        config_files=[deepcopy(base_config)]*5,
        config_file_changes=[deepcopy(microbiome_config_file_changes_logmin)]*5,
)

microbiome_config_file_changes_logmin_smpc = \
    {"flimmaBatchCorrection.data_filename": "msp_counts_norm_logmin_5C.tsv", 
     "flimmaBatchCorrection.design_filename": "design_5C.tsv", 
     "flimmaBatchCorrection.covariates": ["CRC"],
     "flimmaBatchCorrection.smpc": True}
microbiome_experiment_logmin_smpc = util.Experiment(
        clients=[os.path.join(data_dir, "microbiome", "before", "PRJEB10878"),
                os.path.join(data_dir, "microbiome", "before", "PRJEB27928"),
                os.path.join(data_dir, "microbiome", "before", "PRJEB6070"),
                os.path.join(data_dir, "microbiome", "before", "PRJNA429097"),
                os.path.join(data_dir, "microbiome", "before", "PRJNA731589"),
        ],
        app_image_name=app_image_name,
        config_files=[deepcopy(base_config)]*5,
        config_file_changes=[deepcopy(microbiome_config_file_changes_logmin_smpc)]*5,
)

## PROTEOMICS
proteomics_config_file_changes_base = \
    {"flimmaBatchCorrection.data_filename": "intensities_log_filtered.tsv", 
     "flimmaBatchCorrection.design_filename": "design.tsv", 
     "flimmaBatchCorrection.covariates": ["Pyr"]}

proteomics_experiment_balanced = util.Experiment(
        clients=[os.path.join(data_dir, "proteomics", "before", "balanced", "lab_A"),
                 os.path.join(data_dir, "proteomics", "before", "balanced", "lab_B"),
                 os.path.join(data_dir, "proteomics", "before", "balanced", "lab_C"),
                 os.path.join(data_dir, "proteomics", "before", "balanced", "lab_D"),
                 os.path.join(data_dir, "proteomics", "before", "balanced", "lab_E"),
        ],
        app_image_name=app_image_name,
        config_files=[deepcopy(base_config)]*5,
        config_file_changes=[deepcopy(proteomics_config_file_changes_base)]*5,
)

proteomics_config_file_changes_smpc = \
    {"flimmaBatchCorrection.data_filename": "intensities_log_filtered.tsv", 
     "flimmaBatchCorrection.design_filename": "design.tsv", 
     "flimmaBatchCorrection.covariates": ["Pyr"],
     "flimmaBatchCorrection.smpc": True}

proteomics_experiment_balanced_smpc = util.Experiment(
        clients=[os.path.join(data_dir, "proteomics", "before", "balanced", "lab_A"),
                 os.path.join(data_dir, "proteomics", "before", "balanced", "lab_B"),
                 os.path.join(data_dir, "proteomics", "before", "balanced", "lab_C"),
                 os.path.join(data_dir, "proteomics", "before", "balanced", "lab_D"),
                 os.path.join(data_dir, "proteomics", "before", "balanced", "lab_E"),
        ],
        app_image_name=app_image_name,
        config_files=[deepcopy(base_config)]*5,
        config_file_changes=[deepcopy(proteomics_config_file_changes_smpc)]*5,
)

proteomics_experiment_imbalanced = util.Experiment(
        clients=[os.path.join(data_dir, "proteomics", "before", "imbalanced", "lab_A"),
                 os.path.join(data_dir, "proteomics", "before", "imbalanced", "lab_B"),
                 os.path.join(data_dir, "proteomics", "before", "imbalanced", "lab_C"),
                 os.path.join(data_dir, "proteomics", "before", "imbalanced", "lab_D"),
                 os.path.join(data_dir, "proteomics", "before", "imbalanced", "lab_E"),
        ],
        app_image_name=app_image_name,
        config_files=[deepcopy(base_config)]*5,
        config_file_changes=[deepcopy(proteomics_config_file_changes_base)]*5,
)

proteomics_experiment_imbalanced_smpc = util.Experiment(
        clients=[os.path.join(data_dir, "proteomics", "before", "imbalanced", "lab_A"),
                 os.path.join(data_dir, "proteomics", "before", "imbalanced", "lab_B"),
                 os.path.join(data_dir, "proteomics", "before", "imbalanced", "lab_C"),
                 os.path.join(data_dir, "proteomics", "before", "imbalanced", "lab_D"),
                 os.path.join(data_dir, "proteomics", "before", "imbalanced", "lab_E"),
        ],
        app_image_name=app_image_name,
        config_files=[deepcopy(base_config)]*5,
        config_file_changes=[deepcopy(proteomics_config_file_changes_smpc)]*5,
)



## MICROARRAY
base_microarray_config_file_changes = \
    {"flimmaBatchCorrection.data_filename": "expr_for_correction.tsv", 
     "flimmaBatchCorrection.design_filename": "design.tsv", 
     "flimmaBatchCorrection.covariates": ["HGSC"]}
microarray_experiment = util.Experiment(
        clients=[os.path.join(data_dir, "microarray", "before", "GSE6008"),
                 os.path.join(data_dir, "microarray", "before", "GSE14407"),
                 os.path.join(data_dir, "microarray", "before", "GSE26712"),
                 os.path.join(data_dir, "microarray", "before", "GSE38666"),
                 os.path.join(data_dir, "microarray", "before", "GSE40595"),
                 os.path.join(data_dir, "microarray", "before", "GSE69428"),
        ],
        app_image_name=app_image_name,
        config_files=[deepcopy(base_config)]*6,
        config_file_changes=[base_microarray_config_file_changes]*6,
)

base_microarray_config_file_changes_smpc = \
    {"flimmaBatchCorrection.data_filename": "expr_for_correction.tsv", 
     "flimmaBatchCorrection.design_filename": "design.tsv", 
     "flimmaBatchCorrection.covariates": ["HGSC"],
     "flimmaBatchCorrection.smpc": True}
microarray_experiment_smpc = util.Experiment(
        clients=[os.path.join(data_dir, "microarray", "before", "GSE6008"),
                 os.path.join(data_dir, "microarray", "before", "GSE14407"),
                 os.path.join(data_dir, "microarray", "before", "GSE26712"),
                 os.path.join(data_dir, "microarray", "before", "GSE38666"),
                 os.path.join(data_dir, "microarray", "before", "GSE40595"),
                 os.path.join(data_dir, "microarray", "before", "GSE69428"),
        ],
        app_image_name=app_image_name,
        config_files=[deepcopy(base_config)]*6,
        config_file_changes=[deepcopy(base_microarray_config_file_changes_smpc)]*6,
)

### ADD EXPERIMENTS, CHANGE HERE TO INCLUDE/EXCLUDE EXPERIMENTS
## Microbiome
# logmin
# experiments.append(microbiome_experiment_logmin) 
# result_file_names.append(os.path.join(data_dir, "microbiome", "after", "normalized_logmin_counts_5centers_corrected_FEDERATED.tsv"))
# # logmin+smpc
# experiments.append(microbiome_experiment_logmin_smpc)
# result_file_names.append(os.path.join(data_dir, "microbiome", "after", "normalized_logmin_counts_5centers_corrected_smpc_FEDERATED.tsv"))

# ## Proteomics
# # balanced
# experiments.append(proteomics_experiment_balanced)
# result_file_names.append(os.path.join(data_dir, "proteomics", "after", "balanced", "federated_intensities_log_filtered_corrected.tsv"))
# # balanced+smpc
# experiments.append(proteomics_experiment_balanced_smpc)
# result_file_names.append(os.path.join(data_dir, "proteomics", "after", "balanced", "federated_intensities_log_filtered_corrected_smpc.tsv"))
# # imbalanced
# experiments.append(proteomics_experiment_imbalanced)
# result_file_names.append(os.path.join(data_dir, "proteomics", "after", "imbalanced", "federated_intensities_log_filtered_corrected.tsv"))
# # imbalanced+smpc
# experiments.append(proteomics_experiment_imbalanced_smpc)
# result_file_names.append(os.path.join(data_dir, "proteomics", "after", "imbalanced", "federated_intensities_log_filtered_corrected_smpc.tsv"))

# ## Microarray
# # default
# experiments.append(microarray_experiment)
# result_file_names.append(os.path.join(data_dir, "microarray", "after", "federated_corrected.csv"))
# smpc 
experiments.append(microarray_experiment_smpc)
result_file_names.append(os.path.join(data_dir, "microarray", "after", "federated_corrected_smpc.csv"))

if len(experiments) != len(result_file_names):
    raise RuntimeError("Number of experiments and result file names do not match, please fix this!")

### MAIN
# Starts the FeatureCloud controler and runs the experiments
# The results are then postprocessed and saved in the result files.
# This postprocessing simply takes the individual results (assumes the default 
# output file name used by the batch correction app (federated limma RBE)
# of this repo) and concats them.

#Start the featurecloud controller
try:
    util.startup(data_dir)
except Exception as e:
    raise RuntimeError(f"Experiment could not be started! Error: \n{e}")

# Run the experiments
for exp, result_filename in zip(experiments, result_file_names):
    ### RUN THE FEATURECLOUD EXPERIMENT AND EXTARCT INDIVIDUAL RESULTS
    print(f"Starting experiment:\n{exp}") 
    try:   
        result_files_zipped = util.run_test(exp, data_dir)
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
    if not os.path.exists(os.path.join(result_folder, "individual_results")):
        os.makedirs(os.path.join(result_folder, "individual_results"))
    # Extract individual results
    for idx, zipfilename in enumerate(result_files_zipped):
        while True:
            if os.path.exists(os.path.join(data_dir, "tests", "tests", zipfilename)):
                break
            print(f"Waiting for file {zipfilename} to be available...")
            time.sleep(5)
        with zipfile.ZipFile(os.path.join(data_dir, "tests", "tests", zipfilename), 'r') as zip_ref:
            zip_ref.extract(f"only_batch_corrected_data.csv", os.path.join(result_folder, "individual_results"))
            # rename the just extracted file so it doesn't get overwritten
            os.rename(os.path.join(result_folder, "individual_results", "only_batch_corrected_data.csv"), os.path.join(result_folder, "individual_results", f"only_batch_corrected_data_{idx}.csv"))
    
    ### CONCAT THE RESULTS AND PRODUCE FINAL MERGED RESULT
    result_folder = os.path.dirname(result_filename)
    idx = 0
    final_df = None
    while True:
        if not os.path.exists(os.path.join(result_folder, "individual_results", f"only_batch_corrected_data_{idx}.csv")):
            # we have concated all files
            break
        if final_df is None:
            final_df = pd.read_csv(os.path.join(
                result_folder, "individual_results", f"only_batch_corrected_data_{idx}.csv"), sep="\t", index_col=0)
        else:
            # in this case we just concat
            final_df = pd.concat([final_df, pd.read_csv(os.path.join(
                result_folder, "individual_results", f"only_batch_corrected_data_{idx}.csv"), sep="\t", index_col=0)], 
                axis=1)
        idx += 1
    # Save the final df
    if final_df is not None:
        final_df.to_csv(os.path.join(result_filename), sep="\t")
    else:
        raise RuntimeError("No data found in the test results!")

    print("_______________¡SUCCESSFUL_EXPERIMENT!_______________")