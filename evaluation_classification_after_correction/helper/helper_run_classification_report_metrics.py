"""
Random forest training (fc_fed_forest_simple_app in the native simulation),
prediction and the result file shared by the classification evaluations
(train_test_correction.py and loco_correction.py).
"""
from typing import List, Union
from pathlib import Path
import copy
import math
import shutil
import sys

import pandas as pd
import pickle
import yaml
from sklearn.ensemble import RandomForestClassifier

SCRIPT_FOLDER = Path(__file__).parent
REPO_ROOT = SCRIPT_FOLDER.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from fc_fed_forest_simple_app.fedForestSimple import main
from fc_fed_forest_simple_app.fc_fedlearnsim_helper.run_app_simulation import run_simulation_native

RESULTS_FOLDER = SCRIPT_FOLDER.parent / "results"
RESULTS_FOLDER.mkdir(exist_ok=True)
LEARNING_TYPE_FEDERATED = "federated"
LEARNING_TYPE_CENTRALIZED = "centralized"
LEARNING_TYPES = [LEARNING_TYPE_FEDERATED, LEARNING_TYPE_CENTRALIZED]
DEFAULT_SAMPLE_COL = "sample_col"
# Names of the correction arms in the results
PREPROCESSING_UNCORRECTED = "uncorrected"
ARM_LABEL_AWARE = "label_aware"
ARM_LABEL_FREE = "label_free"

BASE_FOREST_CONFIG = {
    'simple_forest': {
        'csv_seperator': ',',
        'data_filename': 'tmp_data.csv',
        'feature_name_col': None,
        'features_as_columns': True,
        'max_depth': 10,
        'max_features': 'sqrt',
        'min_samples_leaf': 1,
        'min_samples_split': 2,
        'num_estimators_total': 100,
        'predicted_feature_name': 'label',
        'random_state': 42,
        'save_predictions': False,
        'sample_col': DEFAULT_SAMPLE_COL,
    }
}


# ---------------------------------------------------------------------------
# Forest config helpers
# ---------------------------------------------------------------------------

def _write_forest_config(client_folder: Path,
                          target: str,
                          seed: int) -> None:
    """
    Write (or overwrite) ``config_forest.yaml`` in *client_folder*.

    The config always references ``tmp_data.csv`` (the forest table of the
    client, see ``FedrbeClassificationExperiment._train_and_score``) with comma
    separator and samples-as-rows orientation.
    """
    config = copy.deepcopy(BASE_FOREST_CONFIG)
    sf = config['simple_forest']
    sf['predicted_feature_name'] = target
    sf['random_state'] = seed
    config_forest_path = client_folder / 'config_forest.yaml'
    with open(config_forest_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f)


def _write_centralized_data_file(input_folder: Path, source_folders: List[Path]) -> None:
    """
    Combine prepared per-cohort ``tmp_data.csv`` files into one centralized
    client folder that can be consumed by the forest app.
    """
    input_folder.mkdir(parents=True, exist_ok=True)
    dfs = [
        pd.read_csv(source_folder / 'tmp_data.csv', index_col=DEFAULT_SAMPLE_COL)
        for source_folder in source_folders
    ]
    pd.concat(dfs, axis=0).to_csv(input_folder / 'tmp_data.csv', sep=',', index_label=DEFAULT_SAMPLE_COL)


def _read_model_and_info(output_folder: Path) -> tuple[RandomForestClassifier, dict]:
    """Load the global forest and model metadata from a simulation output folder."""
    model_path = output_folder / 'global_model.pkl'
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found in {model_path}.")

    with open(model_path, 'rb') as f:
        global_forest: RandomForestClassifier = pickle.load(f)
    with open(output_folder / 'model_info.yaml', 'r', encoding='utf-8') as f:
        model_info = yaml.safe_load(f)
    return global_forest, model_info


def predict_test_data(model: RandomForestClassifier,
                      model_info: dict,
                      test_data: pd.DataFrame,
                      predicted_column: str) -> tuple:
    """
    Predict the samples of *test_data* (samples x features including
    *predicted_column*) with a trained forest. Features of the model missing
    in the test data are added as NaN.
    Returns the true labels and the predictions.
    """
    used_features = model_info['used_features']

    # Transpose if features ended up as rows rather than columns
    if not any(col in test_data.columns for col in used_features):
        test_data = test_data.transpose()

    required_cols = used_features + [predicted_column]
    if predicted_column not in test_data.columns:
        raise ValueError(f"The test data has no label column {predicted_column}")
    if not all(col in test_data.columns for col in required_cols):
        missing = set(required_cols) - set(test_data.columns)
        print(f"The test data misses {len(missing)} of the model's {len(used_features)} "
              "features, adding NaNs for them.")
        missing_df = pd.DataFrame(
            {feat: float('nan') for feat in missing}, index=test_data.index
        )
        test_data = pd.concat([test_data, missing_df], axis=1)

    true_labels = test_data[predicted_column].values
    predictions = model.predict(test_data[used_features])
    return true_labels, predictions


def train_forest(learning_type: str, train_folders: List[Path], train_names: List[str],
                 work_folder: Path, folder_suffix: str, predicted_column: str, seed: int,
                 folders_to_cleanup: List[Path]) -> List[Path]:
    """
    Train the forest app on all samples of the ``tmp_data.csv`` files in
    *train_folders* and return its output folders, the coordinator's first.
    Federated learning runs one client per cohort; centralized learning pools
    the cohorts into one input folder. Folders are created in *work_folder*
    and named ``tmp_classification_evaluation_{input,output}_learningtype|<type>_<folder_suffix>``;
    federated outputs end in ``_traincohort|<cohort>``. All created folders are
    appended to *folders_to_cleanup*.
    """
    name = f"learningtype|{learning_type}_{folder_suffix}"
    if learning_type == LEARNING_TYPE_FEDERATED:
        client_folders = list(train_folders)
        output_folders = [
            work_folder / f"tmp_classification_evaluation_output_{name}_traincohort|{train_name}"
            for train_name in train_names
        ]
    else:
        client_folders = [work_folder / f"tmp_classification_evaluation_input_{name}"]
        output_folders = [work_folder / f"tmp_classification_evaluation_output_{name}"]
        folders_to_cleanup.extend(client_folders)
    folders_to_cleanup.extend(output_folders)
    for out_folder in output_folders:
        if out_folder.exists():
            shutil.rmtree(out_folder)
        out_folder.mkdir(parents=True, exist_ok=True)
    if learning_type != LEARNING_TYPE_FEDERATED:
        _write_centralized_data_file(client_folders[0], train_folders)

    for client_folder in client_folders:
        _write_forest_config(client_folder, predicted_column, seed)

    run_simulation_native(clientpaths=[str(f) for f in client_folders],
                          outputfolders=[str(f) for f in output_folders],
                          generic_dir=None,
                          fed_learning_main_function=main)
    return output_folders


# ---------------------------------------------------------------------------
# Result file class
# ---------------------------------------------------------------------------

def _matching_rows(df: pd.DataFrame, **key) -> pd.Series:
    """Rows whose identifying columns (every column except ``metric_value``) equal *key*."""
    mask = pd.Series(True, index=df.index)
    for column, value in key.items():
        mask &= df[column] == value
    return mask


class ResultFile:
    """
    Manages a CSV result file for classification experiments.

    The file has these columns::

        data_name, data_preprocessing_name, metric_name, metric_value,
        predicted_client_name, cross_validation_method, seed, predicted_target,
        learning_type
    """

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path)

    def check_experiment(self,
                         data_name: str,
                         data_preprocessing_name: str,
                         metric_name: str,
                         predicted_client_name: str,
                         cross_validation_method: str,
                         seed: int,
                         predicted_target: str,
                         learning_type: str) -> float:
        """
        Return the existing metric value if there is already a row whose values match *all*
        provided columns (every column except ``metric_value``).

        Use this to detect whether a given experiment result has been recorded
        before so that re-running can be skipped.
        """
        if not self.path.exists():
            return False
        df = pd.read_csv(self.path)
        if 'learning_type' not in df.columns:
            return False
        mask = _matching_rows(df, data_name=data_name, data_preprocessing_name=data_preprocessing_name,
                              metric_name=metric_name, predicted_client_name=predicted_client_name,
                              cross_validation_method=cross_validation_method, seed=seed,
                              predicted_target=predicted_target, learning_type=learning_type)
        if mask.any():
            value = pd.to_numeric(df.loc[mask, 'metric_value'].iloc[0], errors='coerce')
            if math.isfinite(value):
                return value
        return False

    def upsert_experiment(self,
                          data_name: str,
                          data_preprocessing_name: str,
                          metric_name: str,
                          metric_value: float,
                          predicted_client_name: str,
                          cross_validation_method: str,
                          seed: int,
                          predicted_target: str,
                          learning_type: str) -> None:
        """
        Write a row to the result file.

        If a row with matching identifying columns already exists, its
        ``metric_value`` is updated in place.  Otherwise a new row is appended.
        """
        data = {
            'data_name': data_name,
            'data_preprocessing_name': data_preprocessing_name,
            'metric_name': metric_name,
            'metric_value': metric_value,
            'predicted_client_name': predicted_client_name,
            'cross_validation_method': cross_validation_method,
            'seed': seed,
            'predicted_target': predicted_target,
            'learning_type': learning_type
        }
        if self.path.exists():
            df = pd.read_csv(self.path)
            if 'learning_type' not in df.columns:
                df['learning_type'] = pd.NA
            mask = _matching_rows(df, **{column: value for column, value in data.items()
                                         if column != 'metric_value'})
            if mask.any():
                df.loc[mask, 'metric_value'] = metric_value
            else:
                df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
            df.to_csv(self.path, index=False)
        else:
            pd.DataFrame([data]).to_csv(self.path, index=False)


DEFAULT_RESULTFILE = ResultFile(RESULTS_FOLDER / "classification_metric_report.csv")
