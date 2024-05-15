### TODO: description
if __name__ != "__main__":
    raise RuntimeError("This script should not be imported, it should be run!")
import os
from typing import List
import lib_runFederatedTests as util



### SETTINGS
## GENERALT SETTINGS
# The directory that contains ALL data needed in ANY experiment.
data_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'evaluation_data'))
print(data_dir)
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

## SETTING THE EXPERIMENTS
# SETTING OF BASE VARIABLES
experiments: List[util.Experiment] = list()

# MICROBIOME
# baseline
base_config_file_changes = \
    {"flimmaBatchCorrection.data_filename": "msp_counts_norm_logmin_5C.tsv", 
     "flimmaBatchCorrection.design_filename": "design_5C.tsv", 
     "flimmaBatchCorrection.covariates": ["CRC"]}
base_microbiome_experiment = util.Experiment(
        clients=[os.path.join(data_dir, "microbiome", "before", "PRJEB10878"),
                os.path.join(data_dir, "microbiome", "before", "PRJEB27928"),
                os.path.join(data_dir, "microbiome", "before", "PRJEB6070"),
                os.path.join(data_dir, "microbiome", "before", "PRJNA429097"),
                os.path.join(data_dir, "microbiome", "before", "PRJNA731589"),
        ],
        app_image_name=app_image_name,
        config_files=[base_config]*5,
        config_file_changes=[base_config_file_changes]*5,
        result_folder=os.path.join(data_dir, "microbiome", "after")
)

# specifics
experiments.append(base_microbiome_experiment) # baseline is enough

### MAIN
# Start the featurecloud controller
try:
    util.startup(data_dir)
except Exception as e:
    raise RuntimeError(f"Experiment could not be started! Error: \n{e}")

# Run the experiments
for exp in experiments:
    print(f"Starting experiment:\n{exp}") 
    try:   
        result_files_zipped = util.run_test(exp, data_dir)
    except Exception as e:
        raise RuntimeError(f"Experiment could not be started or aborted too many times! Error: \n{e}")
    
    ### Postprocessing
    print(f"Experiment finished successfully! Result files: {result_files_zipped}")
    raise NotImplementedError("Postprocessing not implemented yet!")

