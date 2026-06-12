from typing import List, Optional, Union
from pathlib import Path
from dataclasses import dataclass
import copy
import json
import shutil

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
TRAIN_TEST_RATIO = 0.7
LEARNING_TYPE_FEDERATED = "federated"
LEARNING_TYPE_CENTRALIZED = "centralized"
LEARNING_TYPES = [LEARNING_TYPE_FEDERATED, LEARNING_TYPE_CENTRALIZED]

BASE_FOREST_CONFIG = {
    'simple_forest': {
        'csv_seperator': ',',
        'data_filename': 'tmp_data.csv',
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
        if not self.prediction_targets or len(self.prediction_targets) == 0:
            raise ValueError("datainfo.json must specify at least one prediction target column in 'prediction_targets'.")

        # Load data
        df_raw = raw['datafile']
        self.datafile = FileSpec(
            filename=df_raw['filename'],
            separator=df_raw['separator'],
            rotation=df_raw['rotation'],
            samplename_column=df_raw.get('samplename_column'),
            featurename_column=df_raw.get('featurename_column'),
        )

        self.cohorts = [
            CohortInfo(name=c['name'], folder=c['folder'], designfile=FileSpec.from_dict(c['designfile']) if c.get('designfile') else None)
            for c in raw['cohorts']
        ]

    @property
    def cohort_folders(self) -> List[Path]:
        """Absolute paths to every cohort folder."""
        return [self._base_folder / c.folder for c in self.cohorts]

    def get_cohort_folder(self, cohort_name: str) -> Path:
        """Return the absolute path for the named cohort."""
        for c in self.cohorts:
            if c.name == cohort_name:
                return self._base_folder / c.folder
        raise KeyError(f"Cohort '{cohort_name}' not found in DataInfo.")

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
            read_kwargs['index_col'] = spec.featurename_column if spec.featurename_column else 0
        elif spec.featurename_column:
            read_kwargs['index_col'] = spec.featurename_column
        df = pd.read_csv(folder / spec.filename, **read_kwargs)

        if spec.rotation == 'features x samples':
            # rows = features, columns = samples  →  transpose to samples × features
            df = df.T
        else:
            # rows = samples, columns = features
            if spec.samplename_column:
                df = df.set_index(spec.samplename_column)

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
            if len(unique_samples_design) > 0 or len(unique_samples_data) > 0:
                raise ValueError(
                    f"Data and design files for cohort '{cohort.name}' have mismatching samples. "
                    f"Unique samples in design file: {len(unique_samples_design)}:\n"
                    f"Unique samples in data file: {len(unique_samples_data)}\n"
                    "Please ensure they have the same sample identifiers in their index or samplename_column."
                )
            # take the prediction targets from design even if provided in the data_df
            data_df = data_df.drop(columns=self.prediction_targets, errors='ignore')
            if not design_df.empty:
                data_df = pd.merge(data_df, design_df, left_index=True, right_index=True)

        # Delete any covariates
        if self.covariates:
            columns_to_drop = [col for col in self.covariates if col not in self.prediction_targets]
                # covariates might be prediction targets
            data_df = data_df.drop(columns=columns_to_drop, errors='ignore')

        # Ensure the prediction target columns exist after loading/merging.
        for target in self.prediction_targets or []:
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
        """
        For every cohort, load data via :meth:`receive_data` and write a
        ``tmp_data.csv`` (comma-separated, samples × features + predicted target) into
        the cohort folder so the federated forest app can consume it.
        """
        for cohort in self.cohorts:
            cohort_folder = self._base_folder / cohort.folder
            df = self.receive_data(cohort)
            # Ensure only the predicted target column and no other prediction targets are present
            other_targets = set(self.prediction_targets or []) - {predicted_col}
            df = df.drop(columns=other_targets, errors='ignore')
            df.to_csv(cohort_folder / 'tmp_data.csv', sep=',')

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
    sf['data_filename'] = 'tmp_data.csv'
    sf['features_as_columns'] = True
    sf['csv_seperator'] = ','
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
        pd.read_csv(source_folder / 'tmp_data.csv', index_col=0)
        for source_folder in source_folders
    ]
    pd.concat(dfs, axis=0).to_csv(input_folder / 'tmp_data.csv', sep=',')


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

    mcc = float(matthews_corrcoef(true_labels, predictions))  # type: ignore
    f1 = float(f1_score(true_labels, predictions, average='weighted'))  # type: ignore
    return mcc, f1


# ---------------------------------------------------------------------------
# Result file class
# ---------------------------------------------------------------------------

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
        mask = (
            (df['data_name'] == data_name)
            & (df['data_preprocessing_name'] == data_preprocessing_name)
            & (df['metric_name'] == metric_name)
            & (df['predicted_client_name'] == predicted_client_name)
            & (df['cross_validation_method'] == cross_validation_method)
            & (df['seed'] == seed)
            & (df['predicted_target'] == predicted_target)
            & (df['learning_type'] == learning_type)
        )
        if mask.any():
            return df.loc[mask, 'metric_value'].iloc[0]
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
            mask = (
                (df['data_name'] == data_name)
                & (df['data_preprocessing_name'] == data_preprocessing_name)
                & (df['metric_name'] == metric_name)
                & (df['predicted_client_name'] == predicted_client_name)
                & (df['cross_validation_method'] == cross_validation_method)
                & (df['seed'] == seed)
                & (df['predicted_target'] == predicted_target)
                & (df['learning_type'] == learning_type)
            )
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

class ClassificationExperimentLeaveOneCohortOut:
    """
    Leave-one-cohort-out cross-validation using a federated random forest.

    For each cohort in *datainfo*:

    * the remaining cohorts are used for federated training (all data, no
      train/test split);
    * the held-out cohort is used as the test set;
    * MCC and F1 are reported for every held-out cohort.

    Per-cohort output folders are placed next to the cohort folders::

        <cohorts_parent>/tmp_classification_evaluation_output_<cohort_name>/
    """

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
        num_characters = 60 + len(self.data_name) + len(self.preprocessing_name) + 5
        print("=" * num_characters)
        print("=" * 30 + f" {self.data_name} ({self.preprocessing_name}) " + "=" * 30)
        print("=" * num_characters)

        cohort_folders = self.datainfo.cohort_folders
        cohort_names = [c.name for c in self.datainfo.cohorts]
        if not self.datainfo.prediction_targets or len(self.datainfo.prediction_targets) == 0:
            raise ValueError("DataInfo must specify at least one prediction target column in 'prediction_targets'.")
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
        all_output_folders: List[Path] = []
        for predicted_column in self.datainfo.prediction_targets:
            self.datainfo.prepare_cohort_data_files(predicted_col=predicted_column)
            for learning_type in LEARNING_TYPES:
                for i, (test_folder, test_cohort_name) in enumerate(zip(cohort_folders, cohort_names)):
                    if not force_run:
                        mcc_existing = self.resultfile.check_experiment(
                                data_name=self.data_name,
                                data_preprocessing_name=self.preprocessing_name,
                                metric_name="MCC",
                                predicted_client_name=test_cohort_name,
                                cross_validation_method="Leave-One-Cohort-Out",
                                seed=seed,
                                predicted_target=predicted_column,
                                learning_type=learning_type
                        )
                        f1_existing = self.resultfile.check_experiment(
                                data_name=self.data_name,
                                data_preprocessing_name=self.preprocessing_name,
                                metric_name="F1_score",
                                predicted_client_name=test_cohort_name,
                                cross_validation_method="Leave-One-Cohort-Out",
                                seed=seed,
                                predicted_target=predicted_column,
                                learning_type=learning_type
                        )
                        if mcc_existing is not False and f1_existing is not False:
                            print(f"Skipping {learning_type} test cohort '{test_cohort_name}' — "
                                    f"results for seed={seed} already exist in {self.resultfile.path}.")
                            average_metrics[learning_type][predicted_column]['MCC'] += mcc_existing / n_clients
                            average_metrics[learning_type][predicted_column]['F1_score'] += f1_existing / n_clients
                            continue

                    train_folders = [f for j, f in enumerate(cohort_folders) if j != i]
                    train_names = [n for j, n in enumerate(cohort_names) if j != i]
                    if not train_folders:
                        raise ValueError("Leave-One-Cohort-Out requires at least two cohorts.")

                    if learning_type == LEARNING_TYPE_FEDERATED:
                        output_folders = [
                            cohorts_parent / (
                                f"tmp_classification_evaluation_output_learningtype|{learning_type}_"
                                f"testcohort|{test_cohort_name}_predictedcol|{predicted_column}_traincohort|{name}"
                            )
                            for name in train_names
                        ]
                        all_output_folders.extend(output_folders)
                        for out_folder in output_folders:
                            out_folder.mkdir(parents=True, exist_ok=True)

                        # Use all data for training (train_test_ratio=1.0)
                        for client_folder in train_folders:
                            _write_forest_config(client_folder, predicted_column, seed, train_test_ratio=1.0)

                        run_simulation_native(clientpaths=[str(f) for f in train_folders],
                                                outputfolders=[str(f) for f in output_folders],
                                                generic_dir=None,
                                                fed_learning_main_function=main)
                        coordinator_output_folder = output_folders[0]
                    else:
                        centralized_input_folder = cohorts_parent / (
                            f"tmp_classification_evaluation_input_learningtype|{learning_type}_"
                            f"testcohort|{test_cohort_name}_predictedcol|{predicted_column}"
                        )
                        centralized_output_folder = cohorts_parent / (
                            f"tmp_classification_evaluation_output_learningtype|{learning_type}_"
                            f"testcohort|{test_cohort_name}_predictedcol|{predicted_column}"
                        )
                        all_output_folders.extend([centralized_input_folder, centralized_output_folder])
                        centralized_output_folder.mkdir(parents=True, exist_ok=True)
                        _write_centralized_data_file(centralized_input_folder, train_folders)
                        _write_forest_config(centralized_input_folder, predicted_column, seed, train_test_ratio=1.0)

                        run_simulation_native(clientpaths=[str(centralized_input_folder)],
                                                outputfolders=[str(centralized_output_folder)],
                                                generic_dir=None,
                                                fed_learning_main_function=main)
                        coordinator_output_folder = centralized_output_folder

                    try:
                        global_forest, model_info = _read_model_and_info(coordinator_output_folder)
                        mcc, f1 = _evaluate_model_on_test_data(
                            global_forest,
                            model_info,
                            test_folder / 'tmp_data.csv',
                            predicted_column,
                        )
                    except FileNotFoundError as exc:
                        print(f"{exc} Skipping evaluation for {test_cohort_name}.")
                        continue

                    average_metrics[learning_type][predicted_column]['MCC'] += mcc / n_clients
                    average_metrics[learning_type][predicted_column]['F1_score'] += f1 / n_clients

                    print(
                        f"Results for {learning_type} test cohort '{test_cohort_name}' on "
                        f"{predicted_column} using model trained on all other cohorts:"
                    )
                    print(f"  MCC: {mcc:.4f}")
                    print(f"  F1 Score: {f1:.4f}")
                    self.resultfile.upsert_experiment(
                        data_name=self.data_name,
                        data_preprocessing_name=self.preprocessing_name,
                        metric_name="MCC",
                        metric_value=mcc,
                        predicted_client_name=test_cohort_name,
                        cross_validation_method="Leave-One-Cohort-Out",
                        seed=seed,
                        predicted_target=predicted_column,
                        learning_type=learning_type
                    )
                    self.resultfile.upsert_experiment(
                        data_name=self.data_name,
                        data_preprocessing_name=self.preprocessing_name,
                        metric_name="F1_score",
                        metric_value=f1,
                        predicted_client_name=test_cohort_name,
                        cross_validation_method="Leave-One-Cohort-Out",
                        seed=seed,
                        predicted_target=predicted_column,
                        learning_type=learning_type
                    )

        # Collect all output folders created across all leave-one-out rounds
        self.datainfo.cleanup_cohort_data_files(also_config_forest=True,
                                                output_folders=all_output_folders)

        sep = "-" * (62 + len(self.data_name) + len(self.preprocessing_name) + 3)
        print(sep)
        for learning_type in LEARNING_TYPES:
            print(f"Average {learning_type} MCC across all test cohorts:")
            for predicted_column, metrics in average_metrics[learning_type].items():
                print(f"  {predicted_column}: {metrics['MCC']:.4f}")
            print(f"Average {learning_type} F1 Score across all test cohorts:")
            for predicted_column, metrics in average_metrics[learning_type].items():
                print(f"  {predicted_column}: {metrics['F1_score']:.4f}")
        print("=" * (62 + len(self.data_name) + len(self.preprocessing_name) + 3))
        print()


class ClassificationExperimentTrainTestSplit:
    """
    Per-cohort train/test-split evaluation using a federated random forest.

    Within each cohort, *train_test_ratio* of the data is used for training
    and the remainder for testing.  MCC and F1 are reported for each cohort.

    Per-cohort output folders are placed next to the cohort folders::

        <cohorts_parent>/tmp_classification_evaluation_output_<cohort_name>/
    """

    data_name: str
    preprocessing_name: str
    datainfo: DataInfo
    train_test_ratio: float
    resultfile: ResultFile

    def __init__(self, preprocessing_name: str,
                 datainfo: DataInfo,
                 train_test_ratio: float = TRAIN_TEST_RATIO,
                 resultfile: ResultFile = DEFAULT_RESULTFILE) -> None:
        self.data_name = datainfo.data_name
        self.preprocessing_name = preprocessing_name
        self.datainfo = datainfo
        self.train_test_ratio = train_test_ratio
        self.resultfile = resultfile

    def run_experiment(self, seed: int, force_run: bool = False) -> None:
        for predicted_column in self.datainfo.prediction_targets or []:
            learning_types_to_run = []
            for learning_type in LEARNING_TYPES:
                if not force_run:
                    mcc = self.resultfile.check_experiment(
                        data_name=self.data_name,
                        data_preprocessing_name=self.preprocessing_name,
                        metric_name="MCC",
                        predicted_client_name="All",
                        cross_validation_method="Train-Test-Split",
                        seed=seed,
                        predicted_target=predicted_column,
                        learning_type=learning_type
                    )
                    f1 = self.resultfile.check_experiment(
                        data_name=self.data_name,
                        data_preprocessing_name=self.preprocessing_name,
                        metric_name="F1_score",
                        predicted_client_name="All",
                        cross_validation_method="Train-Test-Split",
                        seed=seed,
                        predicted_target=predicted_column,
                        learning_type=learning_type
                    )
                    if mcc is not False and f1 is not False:
                        print(f"Skipping {learning_type} '{self.data_name}' ({self.preprocessing_name}) — "
                            f"results for seed={seed} already exist in {self.resultfile.path}.")
                        continue
                learning_types_to_run.append(learning_type)

            if not learning_types_to_run:
                continue

            num_characters = 60 + len(self.data_name) + len(self.preprocessing_name) + 5
            print("=" * num_characters)
            print("=" * 30 + f" {self.data_name} ({self.preprocessing_name}) " + "=" * 30)
            print("=" * num_characters)

            self.datainfo.prepare_cohort_data_files(predicted_col=predicted_column)

            cohort_folders = self.datainfo.cohort_folders
            cohort_names = [c.name for c in self.datainfo.cohorts]
            cohorts_parent = cohort_folders[0].parent

            output_folders = [
                str(cohorts_parent / (
                    f"tmp_classification_evaluation_output_learningtype|{LEARNING_TYPE_FEDERATED}_"
                    f"predictedcol|{predicted_column}_traincohort|{name}"
                ))
                for name in cohort_names
            ]
            for out_folder in output_folders:
                Path(out_folder).mkdir(parents=True, exist_ok=True)
            if not self.datainfo.prediction_targets or len(self.datainfo.prediction_targets) == 0:
                raise ValueError("DataInfo must specify at least one prediction target column in 'prediction_targets'.")

            folders_to_cleanup = [Path(f) for f in output_folders]
            for learning_type in learning_types_to_run:
                if learning_type == LEARNING_TYPE_FEDERATED:
                    for client_folder in cohort_folders:
                        _write_forest_config(client_folder, predicted_column, seed,
                                                train_test_ratio=self.train_test_ratio)

                    run_simulation_native(clientpaths=[str(f) for f in cohort_folders],
                                            outputfolders=output_folders,
                                            generic_dir=None,
                                            fed_learning_main_function=main)

                    for cohort_name, output_folder in zip(cohort_names, output_folders):
                        global_metrics_path = Path(output_folder) / 'global_metrics.csv'
                        local_metrics_path = Path(output_folder) / 'local_metrics.csv'
                        self.read_update_metrics(
                            global_metrics_path,
                            "All",
                            seed,
                            predicted_target=predicted_column,
                            learning_type=learning_type,
                        )
                        self.read_update_metrics(
                            local_metrics_path,
                            cohort_name,
                            seed,
                            predicted_target=predicted_column,
                            learning_type=learning_type,
                        )
                else:
                    centralized_input_folder = cohorts_parent / (
                        f"tmp_classification_evaluation_input_learningtype|{learning_type}_"
                        f"predictedcol|{predicted_column}"
                    )
                    centralized_output_folder = cohorts_parent / (
                        f"tmp_classification_evaluation_output_learningtype|{learning_type}_"
                        f"predictedcol|{predicted_column}"
                    )
                    folders_to_cleanup.extend([centralized_input_folder, centralized_output_folder])
                    centralized_output_folder.mkdir(parents=True, exist_ok=True)
                    _write_centralized_data_file(centralized_input_folder, cohort_folders)
                    _write_forest_config(centralized_input_folder, predicted_column, seed,
                                            train_test_ratio=self.train_test_ratio)

                    run_simulation_native(clientpaths=[str(centralized_input_folder)],
                                            outputfolders=[str(centralized_output_folder)],
                                            generic_dir=None,
                                            fed_learning_main_function=main)

                    self.read_update_metrics(
                        centralized_output_folder / 'global_metrics.csv',
                        "All",
                        seed,
                        predicted_target=predicted_column,
                        learning_type=learning_type,
                    )

            self.datainfo.cleanup_cohort_data_files(
                also_config_forest=True,
                output_folders=folders_to_cleanup,
            )

    def read_update_metrics(self,
                            metrics_path: Path,
                            client_name: str,
                            seed: int,
                            predicted_target: str,
                            learning_type: str) -> None:
        """Read *metrics_path*, print results, and append them to the result file."""
        if metrics_path.exists():
            metrics_df = pd.read_csv(metrics_path)
            for _, row in metrics_df.iterrows():
                metric_suffix = row['class']
                metric_name = row['metric']
                metric_value = row['score']
                print(f"{learning_type} {client_name}: {metric_name} ({metric_suffix}): {metric_value:.4f}")
                self.resultfile.upsert_experiment(
                    data_name=self.data_name,
                    data_preprocessing_name=self.preprocessing_name,
                    metric_name=f"{metric_name}",
                    metric_value=metric_value,
                    predicted_client_name=client_name,
                    cross_validation_method="Train-Test-Split",
                    seed=seed,
                    predicted_target=predicted_target,
                    learning_type=learning_type
                )
