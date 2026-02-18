from typing import List, Optional
from pathlib import Path
import copy
import json

from fc_fed_forest_simple_app.fedForestSimple import main
from fc_fed_forest_simple_app.fc_fedlearnsim_helper.run_app_simulation import run_simulation_native

import pandas as pd
import pickle
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef, f1_score

SCRIPT_FOLDER = Path(__file__).parent
RESULTS_FOLDER = SCRIPT_FOLDER.parent / "results"
RESULTS_FOLDER.mkdir(exist_ok=True)
DEFAULT_RESULTFILE = RESULTS_FOLDER / "classification_metric_report.csv"

BASE_FOREST_CONFIG = {
    'simple_forest': {
        'bootstrap': True,
        'csv_seperator': ',',
        'data_filename': 'data.csv',
        'features_as_columns': True,
        'max_depth': 10,
        'max_features': 'sqrt',
        'max_samples': 0.75,
        'min_samples_leaf': 1,
        'min_samples_split': 2,
        'num_estimators_total': 100,
        'predicted_feature_name': 'label',
        'random_state': 42,
        'sample_col': 0,
        'train_test_ratio': 1.0,
    }
}


def _build_forest_config(datainfo_file: Optional[Path],
                          seed: int,
                          train_test_ratio: float) -> dict:
    """
    Builds a forest config dict from the base config, overriding values derived
    from an optional datainfo JSON file.

    Fields read from datainfo.json:
      - covariate          → predicted_feature_name
      - rotation           → features_as_columns  ("samples x features" → True)
      - csv_separator      → csv_seperator  (optional, falls back to ',')
      - cohorts[0].datafile → data_filename
    """
    config = copy.deepcopy(BASE_FOREST_CONFIG)
    sf = config['simple_forest']
    if datainfo_file is not None and datainfo_file.exists():
        with open(datainfo_file, 'r') as f:
            datainfo = json.load(f)
        if 'covariate' in datainfo:
            sf['predicted_feature_name'] = datainfo['covariate']
        if 'rotation' in datainfo:
            sf['features_as_columns'] = 'samples x features' in datainfo['rotation']
        if 'csv_separator' in datainfo:
            sf['csv_seperator'] = datainfo['csv_separator']
        cohorts = datainfo.get('cohorts', [])
        if cohorts and 'datafile' in cohorts[0]:
            sf['data_filename'] = cohorts[0]['datafile']
    sf['random_state'] = seed
    sf['train_test_ratio'] = train_test_ratio
    return config


def _write_forest_config(client_folder: Path,
                          datainfo_file: Optional[Path],
                          seed: int,
                          train_test_ratio: float) -> None:
    """Writes (or overwrites) config_forest.yaml in the given client folder."""
    config = _build_forest_config(datainfo_file, seed, train_test_ratio)
    config_forest_path = client_folder / 'config_forest.yaml'
    with open(config_forest_path, 'w') as f:
        yaml.safe_dump(config, f)

def create_append_row_resultfile(resultfile: Path,
                                 data_name:str,
                                 data_preprocessing_name: str,
                                 metric_name: str,
                                 metric_value: float,
                                 predicted_client_name: str,
                                 cross_validation_method: str,
                                 seed: int) -> None:
    """ Creates or appends a row to the given resultfile with the given metric name and value """
    data = {
        'data_name': data_name,
        'data_preprocessing_name': data_preprocessing_name,
        'metric_name': metric_name,
        'metric_value': metric_value,
        'predicted_client_name': predicted_client_name,
        'cross_validation_method': cross_validation_method,
        'seed': seed
    }
    if resultfile.exists():
        df_existing = pd.read_csv(resultfile)
        df_new = pd.DataFrame([data])
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(resultfile, index=False)
    else:
        df = pd.DataFrame([data])
        df.to_csv(resultfile, index=False)

class ClassificationExperimentLeaveOneCohortOut:
    """
    A helper class that runs a leave one cohort out style cross validation on the given
    input folders, each containing a config.yml and data file.
    It collects all results into a given common output folder <cohort_left_out>_<client_name>_<name>
    It then prints out the classification metrics for the left out cohort using the model saved
    in the coordinator's output folder.
    """

    data_name: str # The name to report that's associated with this experiment
    preprocessing_name: str # A suffix to append to the name
    input_folders: List[Path] # The client folders containing the config.yml and data file
    output_base_folder: Path # where to write the final metrics (and predictions).
                            # Per simulation run a subfolder will be
    predicted_column: str # The name of the column containing the predicted labels
    datainfo_file: Optional[Path] # Path to the datainfo.json describing the dataset
    resultfile: Path # The path to the result file to write the classification metrics to

    def __init__(self, data_name: str, preprocessing_name: str, input_folders: List[str], output_base_folder: str,
                 predicted_column: str,
                 datainfo_file: Optional[str] = None,
                 resultfile: Path = DEFAULT_RESULTFILE):
        self.data_name = data_name
        self.preprocessing_name = preprocessing_name
        self.input_folders = [Path(f) for f in input_folders]
        self.output_base_folder = Path(output_base_folder)
        self.predicted_column = predicted_column
        self.datainfo_file = Path(datainfo_file) if datainfo_file is not None else None
        # Ensure results are always written to the results folder
        if resultfile == SCRIPT_FOLDER.parent / "classification_metric_report.csv":
            self.resultfile = RESULTS_FOLDER / "classification_metric_report.csv"
        else:
            self.resultfile = resultfile

    def run_experiment(self, seed: int):
        num_characters = 60 + len(self.data_name) + len(self.preprocessing_name) + 5
        print("="*num_characters)
        print("="*30 + f" {self.data_name} ({self.preprocessing_name}) " + "="*30)
        print("="*num_characters)
        average_mcc = 0.0
        average_f1 = 0.0
        n_clients = len(self.input_folders)
        for test_client_folder in self.input_folders:
            run_input_folders = [f for f in self.input_folders if f != test_client_folder]
            test_client_name = test_client_folder.name
            output_folders = [f"{self.output_base_folder}/{test_client_folder.name}_{f.name}_{self.data_name}" for f in run_input_folders]
            for out_folder in output_folders:
                out_path = Path(out_folder)
                out_path.mkdir(parents=True, exist_ok=True)
            # overwrite the train_test_ratio of the config_forest.yaml files to 1.0 (use all data for training)
            for client_folder in run_input_folders:
                _write_forest_config(client_folder, self.datainfo_file, seed, train_test_ratio=1.0)
            run_simulation_native(clientpaths=[str(f) for f in run_input_folders],
                                  outputfolders=output_folders,
                                  generic_dir=None,
                                  fed_learning_main_function=main)

            # collect the resulting model
            coordinator_output_folder = output_folders[0]  # Assuming the first output folder is the coordinator's
            model_path = Path(coordinator_output_folder) / 'global_model.pkl'
            if not model_path.exists():
                print(f"Model not found in {model_path}. Skipping evaluation for {test_client_name}.")
                continue
            # Load the model
            with open(model_path, 'rb') as f:
                global_forest: RandomForestClassifier = pickle.load(f)
            with open(Path(coordinator_output_folder) / 'model_info.yaml', 'r') as f:
                model_info = yaml.safe_load(f)

            # Load the test data
            test_data_path = test_client_folder / 'data.csv'
            if not test_data_path.exists():
                print(f"Test data not found in {test_data_path}. Skipping evaluation for {test_client_name}.")
                continue

            test_data = pd.read_csv(test_data_path)
            used_features = model_info['used_features']

            # transpose if needed
            if not any(col in test_data.columns for col in used_features):
                test_data = test_data.transpose()

            if not all(feature in test_data.columns for feature in used_features + [self.predicted_column]):
                print(f"Test data in {test_data_path} does not contain all required features. Adding NaNs for missing features.")
                missing_features = set(used_features + [self.predicted_column]) - set(test_data.columns)
                #print(f"Features missing: {missing_features}")
                # Create all missing columns at once
                missing_df = pd.DataFrame({feature: float('nan') for feature in missing_features},
                                        index=test_data.index)
                test_data = pd.concat([test_data, missing_df], axis=1)

            true_labels = test_data[self.predicted_column].values
            predictions = global_forest.predict(test_data[used_features])

            mcc = float(matthews_corrcoef(true_labels, predictions)) # type: ignore
            f1 = float(f1_score(true_labels, predictions, average='weighted')) # type: ignore
                # linting ignored as just bc of differences between pandas arrays and numpy arrays
                # no actual interpreter issues
            average_mcc += mcc / n_clients
            average_f1 += f1 / n_clients
            print(f"Results for test cohort '{test_client_name}' using model trained on all other cohorts:")
            print(f"  MCC: {mcc:.4f}")
            print(f"  F1 Score: {f1:.4f}")
            create_append_row_resultfile(
                resultfile=self.resultfile,
                data_name=f"{self.data_name}",
                data_preprocessing_name=self.preprocessing_name,
                metric_name="MCC",
                metric_value=mcc,
                predicted_client_name=test_client_name,
                cross_validation_method="Leave-One-Cohort-Out",
                seed=seed
            )
            create_append_row_resultfile(
                resultfile=self.resultfile,
                data_name=f"{self.data_name}",
                data_preprocessing_name=self.preprocessing_name,
                metric_name="F1_score",
                metric_value=f1,
                predicted_client_name=test_client_name,
                cross_validation_method="Leave-One-Cohort-Out",
                seed=seed
            )
        print("-"*(62+len(self.data_name) + len(self.preprocessing_name) + 3))
        print(f"Average MCC across all test cohorts: {average_mcc:.4f}")
        print(f"Average F1 Score across all test cohorts: {average_f1:.4f}")
        print("="*(62+len(self.data_name) + len(self.preprocessing_name) + 3))
        print()

class ClassificationExperimentTrainTestSplit:
    """
    A helper class that runs trains a simple RandomForestClassifier on the given
    input folders, each containing a config.yml and data file.
    A 80/20 train test split is used within each cohort.
    It then reports the MCC and F1 score over the test data.
    It collects all results into the output folders <client_name>_<name>.
    It then prints out the classification metrics for the test data as calculated by
    the random forest classifier.
    """
    data_name: str # The name to report that's associated with this experiment
    preprocessing_name: str # A suffix to append to the name
    input_folders: List[Path] # The client folders containing the config.yml and data file
    output_base_folder: Path # where to write the final metrics (and predictions).
                            # Per simulation run a subfolder will be
    train_test_ratio: float = 0.8 # The train test split ratio to use
    datainfo_file: Optional[Path] # Path to the datainfo.json describing the dataset
    resultfile: Path # The path to the result file to write the classification metrics to

    def __init__(self, data_name: str, preprocessing_name: str,
                 input_folders: List[str], output_base_folder: str,
                  train_test_ratio: float = 0.8,
                  datainfo_file: Optional[str] = None,
                  resultfile: Path = DEFAULT_RESULTFILE):
        self.data_name = data_name
        self.preprocessing_name = preprocessing_name
        self.input_folders = [Path(f) for f in input_folders]
        self.output_base_folder = Path(output_base_folder)
        self.train_test_ratio = train_test_ratio
        self.datainfo_file = Path(datainfo_file) if datainfo_file is not None else None
        # Ensure results are always written to the results folder
        if resultfile == SCRIPT_FOLDER.parent / "classification_metric_report.csv":
            self.resultfile = RESULTS_FOLDER / "classification_metric_report.csv"
        else:
            self.resultfile = resultfile

    def run_experiment(self, seed: int):
        num_characters = 60 + len(self.data_name) + len(self.preprocessing_name) + 5
        print("="*num_characters)
        print("="*30 + f" {self.data_name} ({self.preprocessing_name}) " + "="*30)
        print("="*num_characters)
        output_folders = [f"{self.output_base_folder}/{input_folder.name}_{self.data_name + self.preprocessing_name}" for input_folder in self.input_folders]
        for out_folder in output_folders:
            out_path = Path(out_folder)
            out_path.mkdir(parents=True, exist_ok=True)
        # write config_forest.yaml to each client folder
        for client_folder in self.input_folders:
            _write_forest_config(client_folder, self.datainfo_file, seed, train_test_ratio=self.train_test_ratio)
        run_simulation_native(clientpaths=[str(f) for f in self.input_folders],
                                outputfolders=output_folders,
                                generic_dir=None,
                                fed_learning_main_function=main)

        # read the metrics files and print the results
        for input_folder, output_folder in zip(self.input_folders, output_folders):
            test_client_name = input_folder.name
            global_metrics_path = Path(output_folder) / 'global_metrics.csv'
            local_metrics_path = Path(output_folder) / 'local_metrics.csv'
            # just read both, it checks if the file exists
            self.read_metrics(global_metrics_path, "All", seed)
            self.read_metrics(local_metrics_path, test_client_name, seed)

    def read_metrics(self,
                     metrics_path: Path,
                     client_name: str,
                     seed: int) -> None:
        """
        Reads the given metrics file, appends results to the resultfile and prints them.
        """
        if metrics_path.exists():
            metrics_df = pd.read_csv(metrics_path)
            for _, row in metrics_df.iterrows():
                metric_suffix = row['class']
                metric_name = row['metric']
                metric_value = row['score']
                print(f"{client_name}: {metric_name} ({metric_suffix}): {metric_value:.4f}")
                create_append_row_resultfile(
                    resultfile=self.resultfile,
                    data_name=f"{self.data_name}",
                    data_preprocessing_name=self.preprocessing_name,
                    metric_name=f"{metric_name}",
                    metric_value=metric_value,
                    predicted_client_name=client_name,
                    cross_validation_method="Train-Test-Split",
                    seed=seed
                )
