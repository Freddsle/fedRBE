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

BASE_FOREST_CONFIG = {
    'simple_forest': {
        'bootstrap': True,
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


@dataclass
class CohortInfo:
    """Describes one cohort entry from datainfo.json."""
    name: str
    folder: str   # path relative to the datainfo.json file's parent directory


class DataInfo:
    """
    Parses the new-format datainfo.json and provides helpers for loading,
    preparing, and cleaning up per-cohort data files for classification experiments.

    Expected JSON structure::

        {
          "covariate": "<column_name>",
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

    covariate: str
    datafile: FileSpec
    designfile: Optional[FileSpec]
    cohorts: List[CohortInfo]

    def __init__(self, datainfo_file: Union[str, Path]) -> None:
        self._path = Path(datainfo_file)
        self._base_folder = self._path.parent
        with open(self._path, 'r') as f:
            raw = json.load(f)

        self.covariate = raw['covariate']

        df_raw = raw['datafile']
        self.datafile = FileSpec(
            filename=df_raw['filename'],
            separator=df_raw['separator'],
            rotation=df_raw['rotation'],
            samplename_column=df_raw.get('samplename_column'),
            featurename_column=df_raw.get('featurename_column'),
        )

        des_raw = raw.get('designfile')
        if des_raw is not None:
            self.designfile = FileSpec(
                filename=des_raw['filename'],
                separator=des_raw['separator'],
                rotation=des_raw['rotation'],
                samplename_column=des_raw.get('samplename_column'),
                featurename_column=des_raw.get('featurename_column'),
            )
        else:
            self.designfile = None

        self.cohorts = [
            CohortInfo(name=c['name'], folder=c['folder'])
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

    def receive_data(self, cohort_folder: Path) -> pd.DataFrame:
        """
        Load the data file (and optionally the design file) for one cohort.

        The returned DataFrame is always in **samples × features** orientation
        and includes the covariate column (merged from the design file when one
        is present, or taken directly from the data file otherwise).
        """
        data_df = self._load_file(self.datafile, cohort_folder)

        if self.designfile is not None:
            design_df = self._load_file(self.designfile, cohort_folder)
            # Merge only the covariate column, aligning on sample index
            data_df = data_df.join(design_df[[self.covariate]], how='inner')
        # else: covariate is expected to already be a column in data_df

        return data_df

    # ------------------------------------------------------------------
    # Cohort data-file preparation / cleanup
    # ------------------------------------------------------------------

    def prepare_cohort_data_files(self) -> None:
        """
        For every cohort, load data via :meth:`receive_data` and write a
        ``tmp_data.csv`` (comma-separated, samples × features + covariate) into
        the cohort folder so the federated forest app can consume it.
        """
        for cohort in self.cohorts:
            cohort_folder = self._base_folder / cohort.folder
            df = self.receive_data(cohort_folder)
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
                          covariate: str,
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
    sf['predicted_feature_name'] = covariate
    sf['data_filename'] = 'tmp_data.csv'
    sf['features_as_columns'] = True
    sf['csv_seperator'] = ','
    sf['random_state'] = seed
    sf['train_test_ratio'] = train_test_ratio
    config_forest_path = client_folder / 'config_forest.yaml'
    with open(config_forest_path, 'w') as f:
        yaml.safe_dump(config, f)


# ---------------------------------------------------------------------------
# Result file class
# ---------------------------------------------------------------------------

class ResultFile:
    """
    Manages a CSV result file for classification experiments.

    The file has these columns::

        data_name, data_preprocessing_name, metric_name, metric_value,
        predicted_client_name, cross_validation_method, seed
    """

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path)

    def check_experiment(self,
                         data_name: str,
                         data_preprocessing_name: str,
                         metric_name: str,
                         predicted_client_name: str,
                         cross_validation_method: str,
                         seed: int) -> bool:
        """
        Return ``True`` if there is already a row whose values match *all*
        provided columns (every column except ``metric_value``).

        Use this to detect whether a given experiment result has been recorded
        before so that re-running can be skipped.
        """
        if not self.path.exists():
            return False
        df = pd.read_csv(self.path)
        mask = (
            (df['data_name'] == data_name)
            & (df['data_preprocessing_name'] == data_preprocessing_name)
            & (df['metric_name'] == metric_name)
            & (df['predicted_client_name'] == predicted_client_name)
            & (df['cross_validation_method'] == cross_validation_method)
            & (df['seed'] == seed)
        )
        return bool(mask.any())

    def upsert_experiment(self,
                          data_name: str,
                          data_preprocessing_name: str,
                          metric_name: str,
                          metric_value: float,
                          predicted_client_name: str,
                          cross_validation_method: str,
                          seed: int) -> None:
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
        }
        if self.path.exists():
            df = pd.read_csv(self.path)
            mask = (
                (df['data_name'] == data_name)
                & (df['data_preprocessing_name'] == data_preprocessing_name)
                & (df['metric_name'] == metric_name)
                & (df['predicted_client_name'] == predicted_client_name)
                & (df['cross_validation_method'] == cross_validation_method)
                & (df['seed'] == seed)
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

    def __init__(self, data_name: str, preprocessing_name: str,
                 datainfo: DataInfo,
                 resultfile: ResultFile = DEFAULT_RESULTFILE) -> None:
        self.data_name = data_name
        self.preprocessing_name = preprocessing_name
        self.datainfo = datainfo
        self.resultfile = resultfile

    def run_experiment(self, seed: int, force_run: bool = False) -> None:
        num_characters = 60 + len(self.data_name) + len(self.preprocessing_name) + 5
        print("=" * num_characters)
        print("=" * 30 + f" {self.data_name} ({self.preprocessing_name}) " + "=" * 30)
        print("=" * num_characters)

        self.datainfo.prepare_cohort_data_files()

        cohort_folders = self.datainfo.cohort_folders
        cohort_names = [c.name for c in self.datainfo.cohorts]
        predicted_column = self.datainfo.covariate
        n_clients = len(cohort_folders)
        cohorts_parent = cohort_folders[0].parent
        # ensure all cohorts share the same parent
        assert all(c.parent == cohorts_parent for c in cohort_folders)

        average_mcc = 0.0
        average_f1 = 0.0

        for i, (test_folder, test_name) in enumerate(zip(cohort_folders, cohort_names)):
            if not force_run and self.resultfile.check_experiment(
                data_name=self.data_name,
                data_preprocessing_name=self.preprocessing_name,
                metric_name="MCC",
                predicted_client_name=test_name,
                cross_validation_method="Leave-One-Cohort-Out",
                seed=seed,
            ):
                print(f"Skipping test cohort '{test_name}' — "
                      f"results for seed={seed} already exist in {self.resultfile.path}.")
                continue

            train_folders = [f for j, f in enumerate(cohort_folders) if j != i]
            train_names = [n for j, n in enumerate(cohort_names) if j != i]

            # Coordinator output is the first element; others are sibling folders
            eval_base = cohorts_parent / f"tmp_classification_evaluation_output_{test_name}"
            output_folders = [str(eval_base)] + [
                str(cohorts_parent / f"tmp_classification_evaluation_output_{test_name}_{name}")
                for name in train_names[1:]
            ]
            for out_folder in output_folders:
                Path(out_folder).mkdir(parents=True, exist_ok=True)

            # Use all data for training (train_test_ratio=1.0)
            for client_folder in train_folders:
                _write_forest_config(client_folder, predicted_column, seed, train_test_ratio=1.0)

            run_simulation_native(clientpaths=[str(f) for f in train_folders],
                                  outputfolders=output_folders,
                                  generic_dir=None,
                                  fed_learning_main_function=main)

            # Collect the resulting model from the coordinator's output folder
            coordinator_output_folder = Path(output_folders[0])
            model_path = coordinator_output_folder / 'global_model.pkl'
            if not model_path.exists():
                print(f"Model not found in {model_path}. Skipping evaluation for {test_name}.")
                continue

            with open(model_path, 'rb') as f:
                global_forest: RandomForestClassifier = pickle.load(f)
            with open(coordinator_output_folder / 'model_info.yaml', 'r') as f:
                model_info = yaml.safe_load(f)

            # Load test data from the prepared tmp_data.csv
            test_data_path = test_folder / 'tmp_data.csv'
            if not test_data_path.exists():
                print(f"Test data not found in {test_data_path}. Skipping evaluation for {test_name}.")
                continue

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
            predictions = global_forest.predict(test_data[used_features])

            mcc = float(matthews_corrcoef(true_labels, predictions))  # type: ignore
            f1 = float(f1_score(true_labels, predictions, average='weighted'))  # type: ignore
            average_mcc += mcc / n_clients
            average_f1 += f1 / n_clients

            print(f"Results for test cohort '{test_name}' using model trained on all other cohorts:")
            print(f"  MCC: {mcc:.4f}")
            print(f"  F1 Score: {f1:.4f}")
            self.resultfile.upsert_experiment(
                data_name=self.data_name,
                data_preprocessing_name=self.preprocessing_name,
                metric_name="MCC",
                metric_value=mcc,
                predicted_client_name=test_name,
                cross_validation_method="Leave-One-Cohort-Out",
                seed=seed,
            )
            self.resultfile.upsert_experiment(
                data_name=self.data_name,
                data_preprocessing_name=self.preprocessing_name,
                metric_name="F1_score",
                metric_value=f1,
                predicted_client_name=test_name,
                cross_validation_method="Leave-One-Cohort-Out",
                seed=seed,
            )

        # Collect all output folders created across all leave-one-out rounds
        all_output_folders: List[Path] = []
        for i, test_name in enumerate(cohort_names):
            train_names = [n for j, n in enumerate(cohort_names) if j != i]
            all_output_folders.append(cohorts_parent / f"tmp_classification_evaluation_output_{test_name}")
            for name in train_names[1:]:
                all_output_folders.append(
                    cohorts_parent / f"tmp_classification_evaluation_output_{test_name}_{name}"
                )
        self.datainfo.cleanup_cohort_data_files(also_config_forest=True,
                                                output_folders=all_output_folders)

        sep = "-" * (62 + len(self.data_name) + len(self.preprocessing_name) + 3)
        print(sep)
        print(f"Average MCC across all test cohorts: {average_mcc:.4f}")
        print(f"Average F1 Score across all test cohorts: {average_f1:.4f}")
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

    def __init__(self, data_name: str, preprocessing_name: str,
                 datainfo: DataInfo,
                 train_test_ratio: float = 0.8,
                 resultfile: ResultFile = DEFAULT_RESULTFILE) -> None:
        self.data_name = data_name
        self.preprocessing_name = preprocessing_name
        self.datainfo = datainfo
        self.train_test_ratio = train_test_ratio
        self.resultfile = resultfile

    def run_experiment(self, seed: int, force_run: bool = False) -> None:
        if not force_run and self.resultfile.check_experiment(
            data_name=self.data_name,
            data_preprocessing_name=self.preprocessing_name,
            metric_name="MCC",
            predicted_client_name="All",
            cross_validation_method="Train-Test-Split",
            seed=seed,
        ):
            print(f"Skipping '{self.data_name}' ({self.preprocessing_name}) — "
                  f"results for seed={seed} already exist in {self.resultfile.path}.")
            return

        num_characters = 60 + len(self.data_name) + len(self.preprocessing_name) + 5
        print("=" * num_characters)
        print("=" * 30 + f" {self.data_name} ({self.preprocessing_name}) " + "=" * 30)
        print("=" * num_characters)

        self.datainfo.prepare_cohort_data_files()

        cohort_folders = self.datainfo.cohort_folders
        cohort_names = [c.name for c in self.datainfo.cohorts]
        cohorts_parent = cohort_folders[0].parent

        output_folders = [
            str(cohorts_parent / f"tmp_classification_evaluation_output_{name}")
            for name in cohort_names
        ]
        for out_folder in output_folders:
            Path(out_folder).mkdir(parents=True, exist_ok=True)

        for client_folder in cohort_folders:
            _write_forest_config(client_folder, self.datainfo.covariate, seed,
                                  train_test_ratio=self.train_test_ratio)

        run_simulation_native(clientpaths=[str(f) for f in cohort_folders],
                              outputfolders=output_folders,
                              generic_dir=None,
                              fed_learning_main_function=main)

        for cohort_name, output_folder in zip(cohort_names, output_folders):
            global_metrics_path = Path(output_folder) / 'global_metrics.csv'
            local_metrics_path = Path(output_folder) / 'local_metrics.csv'
            self.read_metrics(global_metrics_path, "All", seed)
            self.read_metrics(local_metrics_path, cohort_name, seed)

        self.datainfo.cleanup_cohort_data_files(
            also_config_forest=True,
            output_folders=[Path(f) for f in output_folders],
        )

    def read_metrics(self, metrics_path: Path, client_name: str, seed: int) -> None:
        """Read *metrics_path*, print results, and append them to the result file."""
        if metrics_path.exists():
            metrics_df = pd.read_csv(metrics_path)
            for _, row in metrics_df.iterrows():
                metric_suffix = row['class']
                metric_name = row['metric']
                metric_value = row['score']
                print(f"{client_name}: {metric_name} ({metric_suffix}): {metric_value:.4f}")
                self.resultfile.upsert_experiment(
                    data_name=self.data_name,
                    data_preprocessing_name=self.preprocessing_name,
                    metric_name=f"{metric_name}",
                    metric_value=metric_value,
                    predicted_client_name=client_name,
                    cross_validation_method="Train-Test-Split",
                    seed=seed,
                )
