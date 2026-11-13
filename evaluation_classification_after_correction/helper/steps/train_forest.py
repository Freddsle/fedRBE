"""
Training the random forest (fc_fed_forest_simple_app in the native simulation)
on the (corrected) training data: federated with one client per cohort, or
centralized on the pooled cohorts.
"""

from __future__ import annotations

import copy
from pathlib import Path
import pickle
import shutil
from typing import List, Tuple

import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier

from ..common import LEARNING_TYPE_FEDERATED
from .datasets import ClientInput, DatasetSpec
from fc_fed_forest_simple_app.fedForestSimple import main
from fc_fed_forest_simple_app.fc_fedlearnsim_helper.run_app_simulation import run_simulation_native

DEFAULT_SAMPLE_COL = "sample_col"
FOREST_DATA_FILE = "tmp_data.csv"

BASE_FOREST_CONFIG = {
    'simple_forest': {
        'csv_seperator': ',',
        'data_filename': FOREST_DATA_FILE,
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
# Forest input
# ---------------------------------------------------------------------------

def forest_table(spec: DatasetSpec, matrices: List[pd.DataFrame], inputs: List[ClientInput],
                 labels: pd.Series) -> pd.DataFrame:
    """
    Combines the per federation matrices (features x samples) of one cohort to
    a samples (keys) x features table with the label column. Features of
    several modalities are prefixed with <modality>__.
    """
    blocks = []
    for federation, matrix, client in zip(spec.federations, matrices, inputs):
        block = matrix.copy()
        block.index = block.index.astype(str)
        if spec.sample_key:
            positions = client.design.index.get_indexer(block.columns)
            if (positions < 0).any():
                raise ValueError(f"Unknown samples {list(block.columns[positions < 0])}")
            block.columns = client.keys[positions]
        if len(spec.federations) > 1:
            block.index = f"{federation.modality}__" + block.index
        blocks.append(block)
    table = pd.concat(blocks, axis=0, join="outer").T
    table = table.loc[labels.index[labels.index.isin(table.index)]]
    if len(table) != len(labels):
        raise ValueError("Samples were lost while loading the data")
    table[spec.target] = labels.loc[table.index]
    return table


def write_forest_input(table: pd.DataFrame, folder: Path) -> Path:
    """Writes a cohort's forest table as the input folder of a forest client."""
    folder.mkdir(parents=True)
    table.to_csv(folder / FOREST_DATA_FILE, sep=",", index=True, index_label=DEFAULT_SAMPLE_COL)
    return folder


def _write_forest_config(client_folder: Path,
                          target: str,
                          seed: int) -> None:
    """
    Write (or overwrite) ``config_forest.yaml`` in *client_folder*.

    The config always references the forest table of the client (see
    write_forest_input) with comma separator and samples-as-rows orientation.
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
    Combine the per-cohort forest tables into one centralized client folder
    that can be consumed by the forest app.
    """
    input_folder.mkdir(parents=True, exist_ok=True)
    dfs = [
        pd.read_csv(source_folder / FOREST_DATA_FILE, index_col=DEFAULT_SAMPLE_COL)
        for source_folder in source_folders
    ]
    pd.concat(dfs, axis=0).to_csv(input_folder / FOREST_DATA_FILE, sep=',',
                                  index_label=DEFAULT_SAMPLE_COL)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(learning_type: str, train_folders: List[Path], train_names: List[str],
          work_folder: Path, folder_suffix: str, predicted_column: str, seed: int,
          folders_to_cleanup: List[Path]) -> List[Path]:
    """
    Train the forest app on all samples of the forest tables in
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


def read_model(output_folder: Path) -> Tuple[RandomForestClassifier, dict]:
    """Load the global forest and model metadata from a simulation output folder."""
    model_path = output_folder / 'global_model.pkl'
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found in {model_path}.")

    with open(model_path, 'rb') as f:
        global_forest: RandomForestClassifier = pickle.load(f)
    with open(output_folder / 'model_info.yaml', 'r', encoding='utf-8') as f:
        model_info = yaml.safe_load(f)
    return global_forest, model_info
