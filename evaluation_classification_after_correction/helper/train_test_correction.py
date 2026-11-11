"""
Repeated 70/30 train/test classification in which the batch effect correction
only sees the training samples.

For every dataset and repetition (seed):

1. Each cohort's uncorrected fedRBE input is split 70/30, stratified by the
   label (and by the batch if a client holds several batches). The split is
   shared by all preprocessing arms and both learning types. Quartet
   modalities are split by pseudo-sample, so every modality uses the same
   samples.
2. Per correction arm, fedRBE (batchcorrection/fedrbe.py, run in the native
   simulation) corrects the training samples. The label_aware arm uses the
   label as covariate, the label_free arm no covariates. The coordinator
   returns the batch coefficients.
3. Federated (one client per cohort) and centralized (pooled cohorts) random
   forests are trained on all (corrected) training samples.
4. The test samples are corrected with the training batch coefficients
   (batchcorrection.classes.client.subtract_batch_effect, no labels needed)
   and predicted. Values fedRBE did not correct in a batch's training samples
   stay uncorrected in training and test samples. MCC and F1 are reported pooled over all cohorts ("All") and,
   for federated learning, per cohort.

The uncorrected arm uses the same splits without steps 2 and 4's correction.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
from dataclasses import dataclass
import io
from pathlib import Path
import shutil
import tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from .helper_run_classification_report_metrics import (
    ARM_LABEL_AWARE, ARM_LABEL_FREE, DEFAULT_RESULTFILE, DEFAULT_SAMPLE_COL, LEARNING_TYPES,
    LEARNING_TYPE_FEDERATED, PREPROCESSING_UNCORRECTED, REPO_ROOT, RESULTS_FOLDER, ResultFile,
    _read_model_and_info, predict_test_data, run_simulation_native, train_forest,
)
from batchcorrection.classes.client import Client, subtract_batch_effect
from batchcorrection.fedrbe import BATCH_COEFFICIENTS_FILE, run_fedrbe
from evaluation_utils.metrics import classification_scores

CROSS_VALIDATION_METHOD = "Train-Test-Split"
TRAIN_RATIO = 0.7
ARMS = (ARM_LABEL_FREE, ARM_LABEL_AWARE)
DEFAULT_SEEDS = list(range(42, 52))  # ten repetitions with different splits
ALL_COHORTS = "All"
METRICS = ("MCC", "F1_score")
# The splits and the coordinator's batch coefficients of every repetition are
# saved in this folder next to the results file
SPLITS_FOLDER_NAME = "train_test_splits"
SPLITS_FOLDER = RESULTS_FOLDER / SPLITS_FOLDER_NAME
EVALUATION_DATA = REPO_ROOT / "evaluation_data"
CORRECTED_DATA_FILE = "only_batch_corrected_data.csv"


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

@dataclass
class Federation:
    """The clients of one fedRBE run and their fedRBE configs per correction arm."""
    modality: Optional[str]          # Quartet modality, None for single-modality datasets
    cohorts: List[str]               # client names, the first is the coordinator
    client_dirs: List[Path]          # folders with the uncorrected fedRBE inputs
    configs: Dict[str, List[dict]]   # arm -> per client flimmaBatchCorrection settings


@dataclass
class DatasetSpec:
    """A dataset of the evaluation, one federation per modality."""
    name: str                        # registry ID, used for the CLI and the splits folder
    data_name: str                   # data_name of the result rows
    target: str                      # label column of the design files
    federations: List[Federation]
    sample_key: Optional[str] = None
        # design column identifying a sample across the federations (modalities);
        # None uses the design index (the sample names)

    @property
    def cohorts(self) -> List[str]:
        return self.federations[0].cohorts


def _merged_configs(experiment) -> List[dict]:
    """The flimmaBatchCorrection settings of every client, config_file_changes applied."""
    configs = []
    changes = experiment.config_file_changes or [{}] * len(experiment.config_files)
    for config, change in zip(experiment.config_files, changes):
        merged = copy.deepcopy(config["flimmaBatchCorrection"])
        merged.update(copy.deepcopy(change.get("flimmaBatchCorrection", {})))
        # the native simulation aggregates in plain
        merged["smpc"] = False
        configs.append(merged)
    return configs


def _federation(experiment, without_covariates, modality: Optional[str] = None) -> Federation:
    client_dirs = [Path(client) for client in experiment.clients]
    return Federation(
        modality=modality,
        cohorts=[client.name for client in client_dirs],
        client_dirs=client_dirs,
        configs={ARM_LABEL_AWARE: _merged_configs(experiment),
                 ARM_LABEL_FREE: _merged_configs(without_covariates(experiment))},
    )


def _descriptor(folder: str) -> Tuple[str, str]:
    """data_name and the (single) prediction target of a before/ datainfo.json."""
    info = yaml.safe_load((EVALUATION_DATA / folder / "before" / "datainfo.json").read_text())
    targets = info["prediction_targets"]
    if len(targets) != 1:
        raise ValueError(f"Expected one prediction target for {folder}, got {targets}")
    return info["data_name"], targets[0]


def dataset_registry() -> Dict[str, DatasetSpec]:
    """The datasets and their fedRBE experiments of generate_fedrbe_corrected_datasets.py."""
    import generate_fedrbe_corrected_datasets as generate  # pylint: disable=import-outside-toplevel

    single = {
        "balanced": ("simulated/balanced", generate.simulated_balanced_experiment),
        "mild_imbalanced": ("simulated/mild_imbalanced",
                            generate.simulated_mildly_imbalanced_experiment),
        "strong_imbalanced": ("simulated/strong_imbalanced",
                              generate.simulated_strongly_imbalanced_experiment),
        "ecoli": ("ecoli", generate.ecoli_experiment),
        "ovarian_cancer": ("ovarian_cancer", generate.ovarian_cancer_experiment),
        "ccRCC_proteomics": ("ccRCC_studies", generate.ccRCC_proteomics_experiment),
    }
    registry = {}
    for name, (folder, experiment) in single.items():
        data_name, target = _descriptor(folder)
        registry[name] = DatasetSpec(name, data_name, target,
                                     [_federation(experiment, generate.without_covariates)])
    data_name, target = _descriptor("quartet_multiomics/merged_omics")
    registry["quartet_multiomics"] = DatasetSpec(
        "quartet_multiomics", data_name, target,
        [_federation(generate.multiomics_experiments[modality], generate.without_covariates,
                     modality)
         for modality in generate.multiomics_modalities],
        sample_key="pseudo_sample",
    )
    return registry


# ---------------------------------------------------------------------------
# Inputs and split
# ---------------------------------------------------------------------------

@dataclass
class ClientInput:
    """The uncorrected fedRBE input of one client, as read from its folder."""
    data: pd.DataFrame               # features x samples
    design: pd.DataFrame             # samples x design columns, same sample order as data
    keys: pd.Index                   # per sample (design order) its key across federations


def read_client_input(client_dir: Path, config: dict, sample_key: Optional[str]) -> ClientInput:
    """Reads a client's data and design file as given in its fedRBE config."""
    if not config.get("expression_file_flag"):
        raise ValueError(f"{client_dir}: only expression files (features x samples) are supported")
    if not config.get("design_filename"):
        raise ValueError(f"{client_dir}: a design file with the labels is required")
    index_col = config.get("index_col")
    data = pd.read_csv(client_dir / config["data_filename"], sep=config["separator"],
                       index_col=0 if index_col is None else index_col)
    design = pd.read_csv(client_dir / config["design_filename"],
                         sep=config.get("design_separator", "\t"), index_col=0)
    if list(data.columns) != list(design.index):
        raise ValueError(f"{client_dir}: the samples of data and design differ or are not in "
                         "the same order")
    keys = pd.Index(design[sample_key] if sample_key else design.index)
    if not keys.is_unique:
        raise ValueError(f"{client_dir}: sample keys are not unique")
    return ClientInput(data, design, keys)


def read_inputs(spec: DatasetSpec) -> List[List[ClientInput]]:
    """Per federation and client its input. All arms of a client share the data files."""
    inputs = []
    for federation in spec.federations:
        configs = federation.configs[ARM_LABEL_FREE]
        inputs.append([read_client_input(client_dir, config, spec.sample_key)
                       for client_dir, config in zip(federation.client_dirs, configs)])
    return inputs


def _strata(spec: DatasetSpec, inputs: List[List[ClientInput]], cohort_idx: int) -> pd.DataFrame:
    """Per sample key of a cohort its label and its batch in every federation."""
    columns = {}
    for fed_idx, (federation, fed_inputs) in enumerate(zip(spec.federations, inputs)):
        client = fed_inputs[cohort_idx]
        if spec.target not in client.design.columns:
            raise ValueError(f"Label {spec.target} missing in the design of "
                             f"{federation.client_dirs[cohort_idx]}")
        frame = pd.DataFrame({spec.target: client.design[spec.target].to_numpy()},
                             index=client.keys)
        batch_col = federation.configs[ARM_LABEL_FREE][cohort_idx].get("batch_col")
        if batch_col:
            frame[f"batch_{fed_idx}"] = client.design[batch_col].astype(str).to_numpy()
        if not columns:
            columns = frame.to_dict("series")
            reference_keys = client.keys
            continue
        if set(client.keys) != set(reference_keys):
            raise ValueError(f"Cohort {spec.cohorts[cohort_idx]}: the samples differ between "
                             "the modalities")
        frame = frame.loc[reference_keys]
        if not frame[spec.target].equals(columns[spec.target]):
            raise ValueError(f"Cohort {spec.cohorts[cohort_idx]}: the labels differ between "
                             "the modalities")
        columns.update({column: frame[column] for column in frame.columns if column != spec.target})
    return pd.DataFrame(columns)


def split_samples(spec: DatasetSpec, inputs: List[List[ClientInput]], seed: int,
                  train_ratio: float = TRAIN_RATIO) -> pd.DataFrame:
    """
    Stratified train/test split of every cohort by the label and, if a client
    holds several batches, by its batch.
    Returns one row per sample key: cohort, key, label and part (train/test).
    """
    if any(federation.cohorts != spec.cohorts for federation in spec.federations):
        raise ValueError(f"{spec.name}: all modalities need the same cohorts in the same order")
    rows = []
    for cohort_idx, cohort in enumerate(spec.cohorts):
        strata = _strata(spec, inputs, cohort_idx)
        stratum = strata.astype(str).agg("|".join, axis=1)
        train_keys, _ = train_test_split(strata.index.to_numpy(), train_size=train_ratio,
                                         random_state=seed, stratify=stratum.to_numpy())
        train_keys = set(train_keys)
        rows.append(pd.DataFrame({
            "cohort": cohort, "sample": strata.index, "label": strata[spec.target].to_numpy(),
            "part": ["train" if key in train_keys else "test" for key in strata.index],
        }))
    return pd.concat(rows, ignore_index=True)


def training_batches(federation: Federation, fed_inputs: List[ClientInput],
                     split: pd.DataFrame) -> Dict[str, Tuple[int, int]]:
    """Per fedRBE batch label ("client" or "client|batch") its client index and #training samples."""
    batches = {}
    for cohort_idx, (cohort, client) in enumerate(zip(federation.cohorts, fed_inputs)):
        train_keys = set(split.loc[(split.cohort == cohort) & (split.part == "train"), "sample"])
        in_train = client.keys.isin(train_keys)
        batch_col = federation.configs[ARM_LABEL_FREE][cohort_idx].get("batch_col")
        if batch_col:
            counts = client.design.loc[in_train, batch_col].astype(str).value_counts()
            batches.update({f"{cohort}|{batch}": (cohort_idx, int(count))
                            for batch, count in counts.items()})
        else:
            batches[cohort] = (cohort_idx, int(in_train.sum()))
    return batches


def check_min_samples(spec: DatasetSpec, inputs: List[List[ClientInput]], split: pd.DataFrame,
                      arms) -> None:
    """
    Every training batch needs at least as many samples as fedRBE requires
    finite values per feature, max(#batches + #covariates + 1, min_samples).
    """
    for federation, fed_inputs in zip(spec.federations, inputs):
        batches = training_batches(federation, fed_inputs, split)
        for arm in arms:
            configs = federation.configs[arm]
            covariates = set.intersection(*(set(config["covariates"]) for config in configs))
            for batch, (cohort_idx, count) in batches.items():
                required = max(len(batches) + len(covariates) + 1,
                               configs[cohort_idx].get("min_samples", 5))
                if count < required:
                    raise ValueError(
                        f"{spec.name} {federation.modality or ''} ({arm}): training batch {batch} "
                        f"has {count} samples, fedRBE needs at least {required}")


def write_split_inputs(spec: DatasetSpec, inputs: List[List[ClientInput]], split: pd.DataFrame,
                       folder: Path) -> Dict[Tuple[int, str], List[Path]]:
    """
    Writes the uncorrected train and test inputs of every client in the format
    of the original files to <folder>/<federation>/<part>/<cohort>/.
    Returns per (federation index, part) the client folders.
    """
    folders = {}
    for fed_idx, (federation, fed_inputs) in enumerate(zip(spec.federations, inputs)):
        configs = federation.configs[ARM_LABEL_FREE]
        for part in ("train", "test"):
            part_folders = []
            for cohort, client, config in zip(federation.cohorts, fed_inputs, configs):
                keys = set(split.loc[(split.cohort == cohort) & (split.part == part), "sample"])
                samples = client.design.index[client.keys.isin(keys)]
                client_folder = folder / (federation.modality or "data") / part / cohort
                client_folder.mkdir(parents=True)
                client.data.loc[:, samples].to_csv(client_folder / config["data_filename"],
                                                   sep=config["separator"])
                client.design.loc[samples].to_csv(client_folder / config["design_filename"],
                                                  sep=config.get("design_separator", "\t"))
                part_folders.append(client_folder)
            folders[(fed_idx, part)] = part_folders
    return folders


def _write_fedrbe_config(folder: Path, config: dict) -> None:
    with open(folder / "config.yml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"flimmaBatchCorrection": config}, f)


# ---------------------------------------------------------------------------
# Correction
# ---------------------------------------------------------------------------

def _run_quietly(function, *args, **kwargs):
    """Runs function with its output captured, the output's end is shown on failure."""
    log = io.StringIO()
    try:
        with contextlib.redirect_stdout(log):
            return function(*args, **kwargs)
    except BaseException:
        print("\n".join(log.getvalue().splitlines()[-30:]))
        raise


def run_correction(federation: Federation, arm: str, train_folders: List[Path],
                   output_folder: Path) -> Tuple[List[Path], pd.DataFrame]:
    """
    Runs fedRBE on the training inputs, the first client is the coordinator.
    Returns the clients' output folders and the coordinator's batch coefficients.
    """
    output_folders = []
    for cohort, client_folder, config in zip(federation.cohorts, train_folders,
                                             federation.configs[arm]):
        _write_fedrbe_config(client_folder, config)
        output_folders.append(output_folder / cohort)
        output_folders[-1].mkdir(parents=True)
    _run_quietly(run_simulation_native, [str(f) for f in train_folders],
                 [str(f) for f in output_folders], None, run_fedrbe,
                 client_ids=list(federation.cohorts))
    coefficients_path = output_folders[0] / BATCH_COEFFICIENTS_FILE
    coefficients = pd.read_csv(coefficients_path, sep=federation.configs[arm][0]["separator"],
                               index_col=0)
    if coefficients.empty or not np.isfinite(coefficients.to_numpy()).all():
        raise ValueError(f"Empty or non-finite batch coefficients in {coefficients_path}")
    for other in output_folders[1:]:
        if (other / BATCH_COEFFICIENTS_FILE).exists():
            raise ValueError(f"Only the coordinator should write {BATCH_COEFFICIENTS_FILE}")
    return output_folders, coefficients


def read_corrected_data(output_folder: Path, separator: str) -> pd.DataFrame:
    """
    A client's corrected training data written by fedRBE, features x samples.
    Feature names are read as text, as they are later matched as text.
    """
    path = output_folder / CORRECTED_DATA_FILE
    return pd.read_csv(path, sep=separator, index_col=0, dtype=str).astype(float)


def load_client(folder: Path, cohort: str, config: dict) -> Client:
    """
    Loads an input folder as the fedRBE app does (numeric data, normalization,
    hashed feature names), without covariates: labels are not needed to
    correct new samples.
    """
    config = {**config, "covariates": []}
    _write_fedrbe_config(folder, config)
    client = Client()
    _run_quietly(client.config_based_init, clientname=cohort, input_folder=str(folder))
    return client


def client_batch_labels(client: Client) -> List[str]:
    """The fedRBE batch label ("client" or "client|batch") of every sample of a loaded client."""
    if client.batch_col:
        return [f"{client.client_name}|{batch}" for batch in client.design[client.batch_col]]
    return [client.client_name] * len(client.sample_names)


def batch_design(labels: List[str], samples, coefficients: pd.DataFrame,
                 training_batches: set) -> pd.DataFrame:
    """
    The batch columns of fedRBE's design matrix for the given samples: 1 in
    the column of the sample's batch, -1 in all columns for the reference
    batch (the training batch without a column).
    """
    columns = list(coefficients.columns)
    references = training_batches - set(columns)
    if not set(columns) <= training_batches or len(references) != 1:
        raise ValueError(f"The batch coefficients' columns {columns} do not match the training "
                         f"batches {sorted(training_batches)}")
    reference = references.pop()
    design = pd.DataFrame(0, index=pd.Index(samples), columns=columns)
    for sample, label in zip(samples, labels):
        if label == reference:
            design.loc[sample, :] = -1
        elif label in columns:
            design.loc[sample, label] = 1
        else:
            raise ValueError(f"Sample {sample} belongs to batch {label}, which is not a "
                             "training batch")
    return design


def correct_new_samples(client: Client, coefficients: pd.DataFrame, training_batches: set,
                        corrected_training: pd.DataFrame,
                        training_sample_batches: pd.Series) -> pd.DataFrame:
    """
    Corrects the samples of a loaded client with the batch coefficients of a
    previous fedRBE run on the client's training samples.
    fedRBE does not correct features without coefficients (present in too few
    clients) nor, for privacy, the features of a batch with too few finite
    values. As in the training data (see training_matrix), a value of a new
    sample is only corrected if its feature was corrected in the training
    samples of the sample's batch, otherwise it stays uncorrected.
    Args:
        client: The loaded new samples, see load_client
        coefficients: The coordinator's batch coefficients of the training run
        training_batches: All batch labels of the training run
        corrected_training: The client's corrected training data of that run,
            features (as text) x samples
        training_sample_batches: The batch label of every training sample
    Returns:
        The new samples, features (as text) x samples
    """
    result = uncorrected_matrix(client)
    result.index = result.index.astype(str)
    data = client.data.loc[client.data.index.isin(coefficients.index)]
    labels = client_batch_labels(client)
    design = batch_design(labels, list(data.columns), coefficients, training_batches)
    corrected = subtract_batch_effect(data, coefficients.loc[data.index].to_numpy(), design)
    corrected = corrected.rename(index=client.hash2feature)
    corrected.index = corrected.index.astype(str)
    for label in set(labels):
        training_samples = training_sample_batches.index[training_sample_batches == label]
        corrected_features = corrected_training.index[
            corrected_training.loc[:, training_samples].notna().any(axis=1)]
        features = corrected.index[corrected.index.isin(corrected_features)]
        samples = [sample for sample, sample_label in zip(data.columns, labels)
                   if sample_label == label]
        result.loc[features, samples] = corrected.loc[features, samples]
    return result


def training_matrix(corrected_training: pd.DataFrame, client: Client) -> pd.DataFrame:
    """
    The client's training data for the forest: fedRBE's corrected values and,
    where fedRBE did not correct a feature (NaN for the batch or left out),
    the uncorrected values. Features (as text) x samples.
    """
    raw = uncorrected_matrix(client)
    raw.index = raw.index.astype(str)
    return corrected_training.combine_first(raw).loc[:, corrected_training.columns]


def sample_batches(cohort: str, client: ClientInput, batch_col: Optional[str],
                   samples) -> pd.Series:
    """The fedRBE batch label of the given samples of a client."""
    if batch_col:
        return pd.Series([f"{cohort}|{batch}"
                          for batch in client.design.loc[samples, batch_col].astype(str)],
                         index=pd.Index(samples))
    return pd.Series(cohort, index=pd.Index(samples))


def uncorrected_matrix(client: Client) -> pd.DataFrame:
    """The data of a loaded client, features (real names) x samples."""
    return client.data.rename(index=client.hash2feature)


# ---------------------------------------------------------------------------
# Random forest inputs and evaluation
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


def _score_rows(predictions: Dict[str, Tuple[np.ndarray, np.ndarray]],
                learning_type: str) -> Dict[str, Dict[str, float]]:
    """Scores per cohort (federated only) and pooled over all cohorts."""
    truth = np.concatenate([true for true, _ in predictions.values()])
    predicted = np.concatenate([pred for _, pred in predictions.values()])
    scores = {ALL_COHORTS: classification_scores(truth, predicted)}
    if learning_type == LEARNING_TYPE_FEDERATED:
        for cohort, (true, pred) in predictions.items():
            scores[cohort] = classification_scores(true, pred)
    return scores


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

class ClassificationExperimentTrainTestSplit:
    """
    Repeated stratified 70/30 train/test evaluation of federated and centralized
    random forests on uncorrected data and on data whose batch effect
    correction was fitted on the training samples only.
    """

    cross_validation_method = CROSS_VALIDATION_METHOD

    def __init__(self, spec: DatasetSpec, arms=ARMS,
                 resultfile: ResultFile = DEFAULT_RESULTFILE,
                 train_ratio: float = TRAIN_RATIO,
                 splits_folder: Optional[Path] = SPLITS_FOLDER,
                 work_folder: Optional[Path] = None) -> None:
        unknown = set(arms) - set(ARMS)
        if unknown:
            raise ValueError(f"Unknown correction arms {sorted(unknown)}")
        self.spec = spec
        self.arms = list(arms)
        self.resultfile = resultfile
        self.train_ratio = train_ratio
        self.splits_folder = splits_folder
        self.work_folder = work_folder
        self._inputs = None

    @property
    def preprocessings(self) -> List[str]:
        return [PREPROCESSING_UNCORRECTED, *self.arms]

    def _result_key(self, seed: int, preprocessing: str, learning_type: str,
                    client_name: str, metric_name: str) -> dict:
        return dict(data_name=self.spec.data_name, data_preprocessing_name=preprocessing,
                    metric_name=metric_name, predicted_client_name=client_name,
                    cross_validation_method=self.cross_validation_method, seed=seed,
                    predicted_target=self.spec.target, learning_type=learning_type)

    def _is_cached(self, seed: int, preprocessing: str) -> bool:
        for learning_type in LEARNING_TYPES:
            clients = [ALL_COHORTS]
            if learning_type == LEARNING_TYPE_FEDERATED:
                clients.extend(self.spec.cohorts)
            for client in clients:
                for metric in METRICS:
                    value = self.resultfile.check_experiment(
                        **self._result_key(seed, preprocessing, learning_type, client, metric))
                    if value is False:
                        return False
        return True

    def run_experiment(self, seed: int, force_run: bool = False) -> None:
        todo = [preprocessing for preprocessing in self.preprocessings
                if force_run or not self._is_cached(seed, preprocessing)]
        if not todo:
            print(f"Skipping {self.spec.data_name} seed={seed}, results exist in "
                  f"{self.resultfile.path}.")
            return
        print("=" * 80)
        print(f"{self.spec.data_name}: seed {seed}, {', '.join(todo)}")
        print("=" * 80)
        if self._inputs is None:
            self._inputs = read_inputs(self.spec)
        inputs = self._inputs
        split = split_samples(self.spec, inputs, seed, self.train_ratio)
        arms = [preprocessing for preprocessing in todo if preprocessing in ARMS]
        check_min_samples(self.spec, inputs, split, arms)
        if self.work_folder is not None:
            self.work_folder.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f"train_test_{self.spec.name}_",
                                         dir=self.work_folder) as tmp:
            work = Path(tmp)
            folders = write_split_inputs(self.spec, inputs, split, work / "split")
            all_scores, all_coefficients = {}, {}
            for preprocessing in todo:
                tables, all_coefficients[preprocessing] = self._prepare(
                    preprocessing, inputs, split, folders, work / preprocessing)
                all_scores[preprocessing] = self._train_and_score(
                    tables, work / preprocessing, seed)
        # Only publish once every preprocessing succeeded
        self._save_split(seed, split, all_coefficients)
        for preprocessing, scores_by_type in all_scores.items():
            for learning_type, scores in scores_by_type.items():
                for client, metrics in scores.items():
                    print(f"{preprocessing} {learning_type} {client}: "
                          f"MCC {metrics['MCC']:.4f}, F1 {metrics['F1_score']:.4f}")
                    for metric in METRICS:
                        self.resultfile.upsert_experiment(
                            metric_value=metrics[metric],
                            **self._result_key(seed, preprocessing, learning_type, client, metric))

    def _save_split(self, seed: int, split: pd.DataFrame,
                    coefficients: Dict[str, Dict[str, pd.DataFrame]]) -> None:
        """
        Saves the split and per arm the coordinator's batch coefficients to
        <splits_folder>/<dataset>/seed_<seed>/. The coefficients' rows are the
        hashed feature names (batchcorrection.classes.client.Client.hash_names).
        """
        if self.splits_folder is None:
            return
        folder = self.splits_folder / self.spec.name / f"seed_{seed}"
        folder.mkdir(parents=True, exist_ok=True)
        split.to_csv(folder / "split.tsv", sep="\t", index=False)
        for preprocessing, frames in coefficients.items():
            if preprocessing not in ARMS:
                continue
            arm_folder = folder / preprocessing
            if arm_folder.exists():
                shutil.rmtree(arm_folder)
            arm_folder.mkdir()
            for name, frame in frames.items():
                frame.to_csv(arm_folder / name)

    def _prepare(self, preprocessing: str, inputs: List[List[ClientInput]], split: pd.DataFrame,
                 folders: Dict[Tuple[int, str], List[Path]], work: Path):
        """
        The forest tables of every cohort per part, corrected for the arms.
        Returns {part: {cohort: table}} and the batch coefficients by file name.
        """
        matrices = {"train": [[] for _ in self.spec.cohorts], "test": [[] for _ in self.spec.cohorts]}
        coefficients_files = {}
        for fed_idx, federation in enumerate(self.spec.federations):
            arm = ARM_LABEL_FREE if preprocessing == PREPROCESSING_UNCORRECTED else preprocessing
            configs = federation.configs[arm]
            train_folders, test_folders = folders[(fed_idx, "train")], folders[(fed_idx, "test")]
            if preprocessing == PREPROCESSING_UNCORRECTED:
                for part, part_folders in (("train", train_folders), ("test", test_folders)):
                    for cohort_idx, (cohort, folder) in enumerate(zip(federation.cohorts,
                                                                      part_folders)):
                        client = load_client(folder, cohort, configs[cohort_idx])
                        matrices[part][cohort_idx].append(uncorrected_matrix(client))
                continue
            name = f"{federation.modality or 'data'}_fedrbe"
            outputs, coefficients = run_correction(federation, arm, train_folders, work / name)
            suffix = f"_{federation.modality}" if federation.modality else ""
            coefficients_files[f"batch_coefficients{suffix}.csv"] = coefficients
            batches = set(training_batches(federation, inputs[fed_idx], split))
            for cohort_idx, (cohort, output, folder) in enumerate(
                    zip(federation.cohorts, outputs, test_folders)):
                corrected_training = read_corrected_data(output, configs[cohort_idx]["separator"])
                matrices["train"][cohort_idx].append(training_matrix(
                    corrected_training, load_client(train_folders[cohort_idx], cohort,
                                                    configs[cohort_idx])))
                client = load_client(folder, cohort, configs[cohort_idx])
                training_sample_batches = sample_batches(
                    cohort, inputs[fed_idx][cohort_idx], configs[cohort_idx].get("batch_col"),
                    corrected_training.columns)
                matrices["test"][cohort_idx].append(correct_new_samples(
                    client, coefficients, batches, corrected_training, training_sample_batches))
        tables = {}
        for part in ("train", "test"):
            tables[part] = {}
            for cohort_idx, cohort in enumerate(self.spec.cohorts):
                labels = split.loc[(split.cohort == cohort) & (split.part == part)]
                labels = labels.set_index("sample")["label"]
                tables[part][cohort] = forest_table(
                    self.spec, matrices[part][cohort_idx],
                    [fed_inputs[cohort_idx] for fed_inputs in inputs], labels)
        return tables, coefficients_files

    def _train_and_score(self, tables, work: Path, seed: int):
        """Trains both learning types on the training tables and scores the test tables."""
        target = self.spec.target
        train_folders = []
        for cohort, table in tables["train"].items():
            folder = work / "forest_input" / cohort
            folder.mkdir(parents=True)
            table.to_csv(folder / "tmp_data.csv", sep=",", index=True,
                         index_label=DEFAULT_SAMPLE_COL)
            train_folders.append(folder)
        scores = {}
        for learning_type in LEARNING_TYPES:
            output_folders = _run_quietly(
                train_forest, learning_type, train_folders, list(tables["train"]), work,
                f"predictedcol|{target}", target, seed, [])
            model, model_info = _read_model_and_info(output_folders[0])
            predictions = {
                cohort: predict_test_data(model, model_info, table, target)
                for cohort, table in tables["test"].items()
            }
            scores[learning_type] = _score_rows(predictions, learning_type)
        return scores


def run_train_test_split_experiments(argv=None) -> None:
    """CLI of the repeated train/test evaluation, see the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", help="Dataset IDs (default: all)")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS),
                        help="Correction arms besides the uncorrected baseline (default: both)")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                        help="One repetition with a different split per seed")
    parser.add_argument("--results-file", type=Path, default=DEFAULT_RESULTFILE.path)
    parser.add_argument("--work-dir", type=Path, default=None,
                        help="Parent folder of the temporary files (default: system temp)")
    parser.add_argument("--force-run", action="store_true")
    args = parser.parse_args(argv)
    registry = dataset_registry()
    selected = args.datasets or list(registry)
    unknown = set(selected) - set(registry)
    if unknown:
        parser.error(f"Unknown datasets: {', '.join(sorted(unknown))}. "
                     f"Available: {', '.join(registry)}")
    args.results_file.parent.mkdir(parents=True, exist_ok=True)
    resultfile = ResultFile(args.results_file)
    for name in selected:
        experiment = ClassificationExperimentTrainTestSplit(
            registry[name], arms=args.arms, resultfile=resultfile,
            splits_folder=args.results_file.parent / SPLITS_FOLDER_NAME,
            work_folder=args.work_dir)
        for seed in args.seeds:
            experiment.run_experiment(seed=seed, force_run=args.force_run)
