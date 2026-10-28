from typing import List, Optional, Union
from pathlib import Path
from dataclasses import dataclass
import argparse
import copy
from importlib import import_module
import json
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
from evaluation_utils.metrics import classification_scores
from evaluation_utils.real_datasets_utils import build_quartet_matrices, is_git_lfs_pointer

quartet_exporter = import_module("evaluation_data.quartet_multiomics.05_create_merged_omics_folder")
RESULTS_FOLDER = SCRIPT_FOLDER.parent / "results"
RESULTS_FOLDER.mkdir(exist_ok=True)
TRAIN_TEST_RATIO = 0.7
LEARNING_TYPE_FEDERATED = "federated"
LEARNING_TYPE_CENTRALIZED = "centralized"
LEARNING_TYPES = [LEARNING_TYPE_FEDERATED, LEARNING_TYPE_CENTRALIZED]
DEFAULT_SAMPLE_COL = "sample_col"

BASE_FOREST_CONFIG = {
    'simple_forest': {
        'csv_seperator': ',',
        'data_filename': 'tmp_data.csv',
        'feature_name_col': None,
        'features_as_columns': True,
        'max_depth': 10,
        'max_features': 'sqrt',
        'max_samples': 0.75,
        'min_samples_leaf': 1,
        'min_samples_split': 2,
        'num_estimators_total': 100,
        'predicted_feature_name': 'label',
        'random_state': 42,
        'sample_col': DEFAULT_SAMPLE_COL,
        'train_test_ratio': 1.0,
    }
}


# ---------------------------------------------------------------------------
# DataInfo: represents a parsed datainfo.json
# ---------------------------------------------------------------------------

@dataclass
class FileSpec:
    """Describes one data or design file referenced in datainfo.json."""
    filename: str
    separator: str
    rotation: str                     # 'features x samples' or 'samples x features'
    samplename_column: Optional[str]
    featurename_column: Optional[str]

    @staticmethod
    def from_dict(d: dict) -> 'FileSpec':
        return FileSpec(
            filename=d['filename'],
            separator=d['separator'],
            rotation=d['rotation'],
            samplename_column=d.get('samplename_column'),
            featurename_column=d.get('featurename_column'),
        )


@dataclass
class CohortInfo:
    """Describes one cohort entry from datainfo.json."""
    name: str
    folder: str   # path relative to the datainfo.json file's parent directory
    designfile: Optional[FileSpec] = None
        # path to the design file for this cohort, relative to this cohort folder
        # Used to relate the design file from the before correction data to after correction data


class DataInfo:
    """
    Parses the new-format datainfo.json and provides helpers for loading,
    preparing, and cleaning up per-cohort data files for classification experiments.

    Expected JSON structure::

        {
                    "data_name": "<human readable dataset name>",
          "covariates": ["<column_name>"],
          "prediction_targets": [ "<column_name>" ],
          "datafile": {
            "filename": "...",
            "separator": "\\t",
            "rotation": "features x samples",   # or "samples x features"
            "samplename_column": null,
            "featurename_column": "rowname"
          },
          "designfile": null,   # or same structure as datafile
          "cohorts": [
            {"name": "lab1", "folder": "lab1"},
            ...
          ]
        }
    """

    data_name: str
    covariates: Optional[List[str]]
    prediction_targets: List[str]
    datafile: FileSpec
    cohorts: List[CohortInfo]

    def __init__(self, datainfo_file: Union[str, Path]) -> None:
        self._path = Path(datainfo_file)
        self._base_folder = self._path.parent
        raw = json.loads(self._path.read_text(encoding='utf-8'))

        self.data_name = raw['data_name']

        self.covariates = raw['covariates']
        if not self.covariates or len(self.covariates) == 0:
            self.covariates = None

        self.prediction_targets = raw['prediction_targets']
        if not self.prediction_targets:
            raise ValueError("datainfo.json must specify at least one prediction target column in 'prediction_targets'.")

        self.datafile = FileSpec.from_dict(raw['datafile'])
        self.cohorts = [
            CohortInfo(name=c['name'], folder=c['folder'], designfile=FileSpec.from_dict(c['designfile']) if c.get('designfile') else None)
            for c in raw['cohorts']
        ]

    @property
    def cohort_folders(self) -> List[Path]:
        """Absolute paths to every cohort folder."""
        return [self._base_folder / c.folder for c in self.cohorts]

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_file(spec: FileSpec, folder: Path) -> pd.DataFrame:
        """
        Read the file described by *spec* from *folder* and return it in
        **samples × features** orientation (rows = samples, columns = features).
        """

        read_kwargs: dict = {'sep': spec.separator}
        if spec.rotation == 'features x samples':
            # For features-as-rows files the first column is always the feature identifier.
            # Use the named column when provided, otherwise fall back to positional index 0
            # so that after .T the feature names become proper column names rather than
            # an extra data row with integer column indices.
            # we don't support unnamed features
            read_kwargs['index_col'] = spec.featurename_column if spec.featurename_column else 0
        elif spec.samplename_column is not None:
            # In samples-as-rows files, the sample identifier column is optional.
            # If provided, read it in as index_col so that it becomes the DataFrame index.
            # If not provided, pandas will auto-assign integer indices.
            read_kwargs['index_col'] = spec.samplename_column
        filepath = folder / spec.filename
        if is_git_lfs_pointer(filepath):
            raise ValueError(
                f"File '{filepath.name}' is a Git LFS pointer, not the actual data. "
                f"Did you forget to run `git lfs pull`?"
            )
        df = pd.read_csv(filepath, **read_kwargs)
        if spec.rotation == 'features x samples':
            # rows = features, columns = samples  →  transpose to samples × features
            df = df.T

        return df

    def receive_data(self, cohort: CohortInfo) -> pd.DataFrame:
        """
        Load the data file (and optionally the design files) for one cohort.
        The designfile is exclusively used to get the prediction_targets column.

        The returned DataFrame is always in **samples × features** orientation
        and includes any prediction target columns.
        Covariates are specifically excluded.
        Throws an error if the final df misses any of the prediction target columns after
        loading/merging.
        """
        cohort_folder = self._base_folder / cohort.folder
        data_df = self._load_file(self.datafile, cohort_folder)
        if cohort.designfile:
            design_df = self._load_file(cohort.designfile, cohort_folder)
            design_df = design_df[self.prediction_targets]
            sample_intersection = set(data_df.index).intersection(set(design_df.index))
            unique_samples_design = set(design_df.index) - sample_intersection
            unique_samples_data = set(data_df.index) - sample_intersection
            if len(unique_samples_design) > 0:
                print(f"WARNING: Design file {cohort.designfile.filename} contains additional samples that are not used in the data file")
                print("WARNING: This might be intentional so the script will continue")
            if len(unique_samples_data) > 0:
                raise ValueError(
                    f"Data and design files for cohort '{cohort.name}' have mismatching samples. "
                    f"Unique samples in design file: {len(unique_samples_design)}:\n"
                    f"Unique samples in data file: {len(unique_samples_data)}\n"
                    "Please ensure they have the same sample identifiers in their index or samplename_column."
                )
            # take the prediction targets from design even if provided in the data_df
            data_df = data_df.drop(columns=self.prediction_targets, errors='ignore')
            if not design_df.empty:
                # inner join, will remove any extra samples from design (this is intended)
                data_df = pd.merge(data_df, design_df, left_index=True, right_index=True)

        # Delete any covariates
        if self.covariates:
            columns_to_drop = [col for col in self.covariates if col not in self.prediction_targets]
                # covariates might be prediction targets
            data_df = data_df.drop(columns=columns_to_drop, errors='ignore')

        # Ensure the prediction target columns exist after loading/merging.
        for target in self.prediction_targets:
            if target not in data_df.columns:
                raise ValueError(
                    f"Prediction target '{target}' not found in cohort data at {cohort_folder}. "
                    "Either provide a designfile with this target or include it in the datafile."
                )
        return data_df

    # ------------------------------------------------------------------
    # Cohort data-file preparation / cleanup
    # ------------------------------------------------------------------

    def prepare_cohort_data_files(self, predicted_col: str) -> None:
        for cohort in self.cohorts:
            cohort_folder = self._base_folder / cohort.folder
            df = self.receive_data(cohort)

            # Ensure only the predicted target column remains
            other_targets = set(self.prediction_targets) - {predicted_col}
            df = df.drop(columns=other_targets, errors='ignore')

            df.to_csv(cohort_folder / 'tmp_data.csv', sep=',', index=True, index_label=DEFAULT_SAMPLE_COL)

    def cleanup_cohort_data_files(self,
                                    also_config_forest: bool = False,
                                    output_folders: Optional[List[Path]] = None) -> None:
        """
        Remove the generated ``tmp_data.csv`` from every cohort folder.
        Pass ``also_config_forest=True`` to also remove ``config_forest.yaml``.
        Pass ``output_folders`` to also delete those directories entirely.
        """
        targets = ['tmp_data.csv']
        if also_config_forest:
            targets.append('config_forest.yaml')
        for cohort in self.cohorts:
            cohort_folder = self._base_folder / cohort.folder
            for fname in targets:
                fp = cohort_folder / fname
                if fp.exists():
                    fp.unlink()
        if output_folders:
            for folder in output_folders:
                if folder.exists():
                    shutil.rmtree(folder)


# ---------------------------------------------------------------------------
# Forest config helpers
# ---------------------------------------------------------------------------

def _write_forest_config(client_folder: Path,
                          target: str,
                          seed: int,
                          train_test_ratio: float) -> None:
    """
    Write (or overwrite) ``config_forest.yaml`` in *client_folder*.

    The config always references ``tmp_data.csv`` (written by
    :meth:`DataInfo.prepare_cohort_data_files`) with comma separator and
    samples-as-rows orientation.
    """
    config = copy.deepcopy(BASE_FOREST_CONFIG)
    sf = config['simple_forest']
    sf['predicted_feature_name'] = target
    sf['random_state'] = seed
    sf['train_test_ratio'] = train_test_ratio
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


def _evaluate_model_on_test_data(model: RandomForestClassifier,
                                 model_info: dict,
                                 test_data_path: Path,
                                 predicted_column: str) -> tuple[float, float]:
    """Evaluate a trained forest against a prepared held-out cohort file."""
    if not test_data_path.exists():
        raise FileNotFoundError(f"Test data not found in {test_data_path}.")

    test_data = pd.read_csv(test_data_path)
    used_features = model_info['used_features']

    # Transpose if features ended up as rows rather than columns
    if not any(col in test_data.columns for col in used_features):
        test_data = test_data.transpose()

    required_cols = used_features + [predicted_column]
    if not all(col in test_data.columns for col in required_cols):
        print(f"Test data in {test_data_path} does not contain all required features. "
                f"Adding NaNs for missing features.")
        missing = set(required_cols) - set(test_data.columns)
        missing_df = pd.DataFrame(
            {feat: float('nan') for feat in missing}, index=test_data.index
        )
        test_data = pd.concat([test_data, missing_df], axis=1)

    true_labels = test_data[predicted_column].values
    predictions = model.predict(test_data[used_features])

    scores = classification_scores(true_labels, predictions)
    return scores['MCC'], scores['F1_score']


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


# ---------------------------------------------------------------------------
# Experiment classes
# ---------------------------------------------------------------------------

class _ClassificationExperiment:
    """Shared training, caching and result recording of the RF experiments."""

    cross_validation_method: str
    train_test_ratio: float
    data_name: str
    preprocessing_name: str
    datainfo: DataInfo
    resultfile: ResultFile

    def __init__(self, preprocessing_name: str,
                 datainfo: DataInfo,
                 resultfile: ResultFile = DEFAULT_RESULTFILE) -> None:
        self.data_name = datainfo.data_name
        self.preprocessing_name = preprocessing_name
        self.datainfo = datainfo
        self.resultfile = resultfile

    def run_experiment(self, seed: int, force_run: bool = False) -> None:
        output_folders: List[Path] = []
        try:
            self._run_experiment(seed, force_run, output_folders)
        finally:
            self.datainfo.cleanup_cohort_data_files(
                also_config_forest=True,
                output_folders=output_folders,
            )

    def _print_banner(self) -> int:
        """Print the dataset banner and return its width."""
        num_characters = 60 + len(self.data_name) + len(self.preprocessing_name) + 5
        print("=" * num_characters)
        print("=" * 30 + f" {self.data_name} ({self.preprocessing_name}) " + "=" * 30)
        print("=" * num_characters)
        return num_characters

    def _result_key(self, seed: int, predicted_target: str, learning_type: str,
                    client_name: str, metric_name: str) -> dict:
        """Identify one result row by every column except ``metric_value``."""
        return dict(data_name=self.data_name, data_preprocessing_name=self.preprocessing_name,
                    metric_name=metric_name, predicted_client_name=client_name,
                    cross_validation_method=self.cross_validation_method, seed=seed,
                    predicted_target=predicted_target, learning_type=learning_type)

    def _cached_scores(self, seed: int, predicted_target: str, learning_type: str,
                       client_name: str) -> Optional[List[float]]:
        """Return the recorded MCC and F1 of *client_name*, or None unless both exist."""
        scores = [
            self.resultfile.check_experiment(
                **self._result_key(seed, predicted_target, learning_type, client_name, metric_name))
            for metric_name in ("MCC", "F1_score")
        ]
        return None if any(score is False for score in scores) else scores

    def _train(self, learning_type: str, train_folders: List[Path], train_names: List[str],
               folder_suffix: str, predicted_column: str, seed: int,
               folders_to_cleanup: List[Path]) -> List[Path]:
        """
        Train the forest app on the prepared *train_folders* and return its output
        folders, the coordinator's first. Federated learning runs one client per
        cohort; centralized learning pools the cohorts into one input folder.
        Folders sit beside the cohorts and are named
        ``tmp_classification_evaluation_{input,output}_learningtype|<type>_<folder_suffix>``;
        federated outputs end in ``_traincohort|<cohort>``.
        """
        cohorts_parent = self.datainfo.cohort_folders[0].parent
        name = f"learningtype|{learning_type}_{folder_suffix}"
        if learning_type == LEARNING_TYPE_FEDERATED:
            client_folders = train_folders
            output_folders = [
                cohorts_parent / f"tmp_classification_evaluation_output_{name}_traincohort|{train_name}"
                for train_name in train_names
            ]
        else:
            client_folders = [cohorts_parent / f"tmp_classification_evaluation_input_{name}"]
            output_folders = [cohorts_parent / f"tmp_classification_evaluation_output_{name}"]
            folders_to_cleanup.extend(client_folders)
        folders_to_cleanup.extend(output_folders)
        for out_folder in output_folders:
            if out_folder.exists():
                shutil.rmtree(out_folder)
            out_folder.mkdir(parents=True, exist_ok=True)
        if learning_type != LEARNING_TYPE_FEDERATED:
            _write_centralized_data_file(client_folders[0], train_folders)

        for client_folder in client_folders:
            _write_forest_config(client_folder, predicted_column, seed, self.train_test_ratio)

        run_simulation_native(clientpaths=[str(f) for f in client_folders],
                              outputfolders=[str(f) for f in output_folders],
                              generic_dir=None,
                              fed_learning_main_function=main)
        return output_folders


class ClassificationExperimentLeaveOneCohortOut(_ClassificationExperiment):
    """
    Leave-one-cohort-out cross-validation using federated and centralized random forests.

    For each cohort in *datainfo*:

    * the remaining cohorts are used for training (all data, no
      train/test split);
    * the held-out cohort is used as the test set;
    * MCC and F1 are reported for every held-out cohort.

    Temporary input/output folders sit beside the cohorts.
    """

    cross_validation_method = "Leave-One-Cohort-Out"
    train_test_ratio = 1.0  # train on all samples of the remaining cohorts

    def _run_experiment(self, seed: int, force_run: bool,
                        all_output_folders: List[Path]) -> None:
        num_characters = self._print_banner()

        cohort_folders = self.datainfo.cohort_folders
        cohort_names = [c.name for c in self.datainfo.cohorts]
        n_clients = len(cohort_folders)
        cohorts_parent = cohort_folders[0].parent
        # ensure all cohorts share the same parent
        assert all(c.parent == cohorts_parent for c in cohort_folders)
        average_metrics = {
            learning_type: {
                predicted_column: {'MCC': 0.0, 'F1_score': 0.0}
                for predicted_column in self.datainfo.prediction_targets
            }
            for learning_type in LEARNING_TYPES
        }
        for predicted_column in self.datainfo.prediction_targets:
            self.datainfo.prepare_cohort_data_files(predicted_col=predicted_column)
            for learning_type in LEARNING_TYPES:
                for i, (test_folder, test_cohort_name) in enumerate(zip(cohort_folders, cohort_names)):
                    scores = None if force_run else self._cached_scores(
                        seed, predicted_column, learning_type, test_cohort_name)
                    if scores is not None:
                        print(f"Skipping {learning_type} test cohort '{test_cohort_name}' — "
                                f"results for seed={seed} already exist in {self.resultfile.path}.")
                        mcc, f1 = scores
                    else:
                        train_folders = [f for j, f in enumerate(cohort_folders) if j != i]
                        train_names = [n for j, n in enumerate(cohort_names) if j != i]
                        if not train_folders:
                            raise ValueError("Leave-One-Cohort-Out requires at least two cohorts.")

                        output_folders = self._train(
                            learning_type, train_folders, train_names,
                            f"testcohort|{test_cohort_name}_predictedcol|{predicted_column}",
                            predicted_column, seed, all_output_folders)
                        global_forest, model_info = _read_model_and_info(output_folders[0])
                        mcc, f1 = _evaluate_model_on_test_data(
                            global_forest,
                            model_info,
                            test_folder / 'tmp_data.csv',
                            predicted_column,
                        )

                        print(
                            f"Results for {learning_type} test cohort '{test_cohort_name}' on "
                            f"{predicted_column} using model trained on all other cohorts:"
                        )
                        print(f"  MCC: {mcc:.4f}")
                        print(f"  F1 Score: {f1:.4f}")
                        for metric_name, metric_value in (("MCC", mcc), ("F1_score", f1)):
                            self.resultfile.upsert_experiment(
                                metric_value=metric_value,
                                **self._result_key(seed, predicted_column, learning_type,
                                                   test_cohort_name, metric_name))

                    average_metrics[learning_type][predicted_column]['MCC'] += mcc / n_clients
                    average_metrics[learning_type][predicted_column]['F1_score'] += f1 / n_clients

        print("-" * num_characters)
        for learning_type in LEARNING_TYPES:
            print(f"Average {learning_type} MCC across all test cohorts:")
            for predicted_column, metrics in average_metrics[learning_type].items():
                print(f"  {predicted_column}: {metrics['MCC']:.4f}")
            print(f"Average {learning_type} F1 Score across all test cohorts:")
            for predicted_column, metrics in average_metrics[learning_type].items():
                print(f"  {predicted_column}: {metrics['F1_score']:.4f}")
        print("=" * num_characters)
        print()


class ClassificationExperimentTrainTestSplit(_ClassificationExperiment):
    """
    Train/test evaluation with centralized and federated forest ensembles.

    Centralized learning splits the pooled samples; federated learning splits
    within each cohort. Both use *train_test_ratio* for training and report
    pooled ("All") MCC and support-weighted F1. Federated runs also retain
    per-cohort scores. Temporary input/output folders sit beside the cohorts.
    """

    cross_validation_method = "Train-Test-Split"

    def __init__(self, preprocessing_name: str,
                 datainfo: DataInfo,
                 train_test_ratio: float = TRAIN_TEST_RATIO,
                 resultfile: ResultFile = DEFAULT_RESULTFILE) -> None:
        super().__init__(preprocessing_name, datainfo, resultfile)
        self.train_test_ratio = train_test_ratio

    def _run_experiment(self, seed: int, force_run: bool,
                        folders_to_cleanup: List[Path]) -> None:
        for predicted_column in self.datainfo.prediction_targets:
            learning_types_to_run = []
            for learning_type in LEARNING_TYPES:
                if not force_run:
                    client_names = ["All"]
                    if learning_type == LEARNING_TYPE_FEDERATED:
                        client_names.extend(c.name for c in self.datainfo.cohorts)
                    if all(self._cached_scores(seed, predicted_column, learning_type, client_name)
                           is not None for client_name in client_names):
                        print(f"Skipping {learning_type} '{self.data_name}' ({self.preprocessing_name}) — "
                            f"results for seed={seed} already exist in {self.resultfile.path}.")
                        continue
                learning_types_to_run.append(learning_type)

            if not learning_types_to_run:
                continue

            self._print_banner()
            self.datainfo.prepare_cohort_data_files(predicted_col=predicted_column)

            cohort_folders = self.datainfo.cohort_folders
            cohort_names = [c.name for c in self.datainfo.cohorts]

            for learning_type in learning_types_to_run:
                output_folders = self._train(learning_type, cohort_folders, cohort_names,
                                             f"predictedcol|{predicted_column}",
                                             predicted_column, seed, folders_to_cleanup)
                metrics_files = [(output_folders[0] / 'global_metrics.csv', "All")]
                if learning_type == LEARNING_TYPE_FEDERATED:
                    metrics_files.extend((folder / 'local_metrics.csv', name)
                                         for name, folder in zip(cohort_names, output_folders))
                self.read_update_metrics(
                    metrics_files,
                    seed,
                    predicted_target=predicted_column,
                    learning_type=learning_type,
                )

    def read_update_metrics(self,
                            metrics_files: List[tuple[Path, str]],
                            seed: int,
                            predicted_target: str,
                            learning_type: str) -> None:
        """Validate every required metrics file before updating the result file."""
        metrics_by_client = []
        required_columns = {'class', 'metric', 'score'}
        for metrics_path, client_name in metrics_files:
            metrics_df = pd.read_csv(metrics_path)
            if not required_columns.issubset(metrics_df.columns):
                missing = required_columns - set(metrics_df.columns)
                raise ValueError(f"Missing metric columns in {metrics_path}: {missing}.")
            if len(metrics_df) != 2 or set(metrics_df['metric']) != {'MCC', 'F1_score'}:
                raise ValueError(f"Expected one MCC and one F1_score in {metrics_path}.")
            scores = pd.to_numeric(metrics_df['score'], errors='coerce')
            if not scores.map(math.isfinite).all():
                raise ValueError(f"Expected finite metric scores in {metrics_path}.")
            metrics_df['score'] = scores
            metrics_by_client.append((client_name, metrics_df))

        for client_name, metrics_df in metrics_by_client:
            for _, row in metrics_df.iterrows():
                metric_suffix = row['class']
                metric_name = row['metric']
                metric_value = row['score']
                print(f"{learning_type} {client_name}: {metric_name} ({metric_suffix}): {metric_value:.4f}")
                self.resultfile.upsert_experiment(
                    metric_value=metric_value,
                    **self._result_key(seed, predicted_target, learning_type, client_name, metric_name))


def run_classification_experiments(experiment_type, argv=None) -> None:
    """Run existing RF experiments from canonical defaults or a dataset registry.

    Registry entries use classification.before, classification.corrected and an
    optional classification.preprocessing_name. Paths are absolute or relative
    to the repository root, as in the k-means registry. The correction arm selects
    canonical real-data folders; simulated data retain their existing inputs.
    """
    parser = argparse.ArgumentParser(description=experiment_type.__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--config", type=Path, help="Dataset YAML with classification paths")
    inputs.add_argument("--correction-arm", choices=("label_aware", "no_label"),
                        help="Canonical real-data correction folder (default: label_aware)")
    parser.add_argument("--datasets", nargs="+", help="Dataset IDs from the selected registry")
    parser.add_argument("--results-file", type=Path, default=DEFAULT_RESULTFILE.path)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)))
    parser.add_argument("--force-run", action="store_true")
    args = parser.parse_args(argv)
    if args.config:
        entries = yaml.safe_load(args.config.read_text())
        base = SCRIPT_FOLDER.parent.parent
    else:
        correction_arm = args.correction_arm or "label_aware"
        base = SCRIPT_FOLDER.parent.parent / "evaluation_data"
        folders = {"balanced": "simulated/balanced", "ovarian_cancer": "ovarian_cancer",
                   "ecoli": "ecoli", "ccRCC_proteomics": "ccRCC_studies",
                   "quartet_multiomics": "quartet_multiomics/merged_omics"}
        entries = {}
        for name, folder in folders.items():
            after = "after" if name == "balanced" else f"after/{correction_arm}"
            entries[name] = {"classification": {
                "before": f"{folder}/before/datainfo.json",
                "corrected": f"{folder}/{after}/datainfo.json",
                "preprocessing_name": ("no_label" if name != "balanced"
                                       and correction_arm == "no_label" else "corrected"),
            }}
    if not isinstance(entries, dict):
        parser.error("The dataset registry must be a mapping")
    selected = args.datasets or list(entries)
    unknown = set(selected) - set(entries)
    if unknown:
        parser.error(f"Unknown datasets: {', '.join(sorted(unknown))}")
    experiments, seen = [], set()
    resultfile = ResultFile(args.results_file)
    for name in selected:
        if args.config is None and name == "quartet_multiomics":
            quartet_root = base / "quartet_multiomics"
            matrices = build_quartet_matrices(quartet_root, correction_arm,
                                             corrected_source="federated")
            quartet_exporter.create_merged_omics_folder(
                datafiles={"before": matrices["before"],
                           f"after/{correction_arm}": matrices["federated"]},
                metadata_file=matrices["metadata"], output_folder=quartet_root / "merged_omics",
            )
        classification = entries[name].get("classification")
        if not isinstance(classification, dict):
            parser.error(f"Dataset {name} has no classification paths")
        corrected_name = ("no_label" if "no_label" in Path(classification.get("corrected") or "").parts
                          else "corrected")
        for key, preprocessing in (("before", "uncorrected"),
                                   ("corrected", classification.get("preprocessing_name", corrected_name))):
            if not classification.get(key):
                continue
            descriptor = (base / classification[key]).resolve()
            identity = (descriptor, preprocessing)
            if identity not in seen:
                experiments.append(experiment_type(preprocessing, DataInfo(descriptor),
                                                   resultfile=resultfile))
                seen.add(identity)
    if not experiments:
        parser.error("No classification descriptors selected")
    args.results_file.parent.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        for experiment in experiments:
            experiment.run_experiment(seed=seed, force_run=args.force_run)
