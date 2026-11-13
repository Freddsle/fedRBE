"""
Batch effect correction with fedRBE (batchcorrection/fedrbe.py, run in the
native simulation) on the training data, and correcting the test data with its
output without another run:

- correct_training: fedRBE on the training samples of every cohort.
- correct_new_samples: test samples of a training batch (70/30), with the batch
  coefficients of their batch.
- align_held_out: a held-out cohort (LOCO), shifted onto the training level.

Values fedRBE did not correct (features in too few clients, or too sparse in a
batch) stay uncorrected in training and test data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from ..common import run_quietly
from .datasets import ClientInput, Federation
from .prepare import load_client, training_batches, uncorrected_matrix, write_fedrbe_config
from batchcorrection.align_new_batch import read_training_level
from batchcorrection.classes.client import Client, align_to_training_level, subtract_batch_effect
from batchcorrection.fedrbe import BATCH_COEFFICIENTS_FILE, run_fedrbe
from fc_fed_forest_simple_app.fc_fedlearnsim_helper.run_app_simulation import run_simulation_native

CORRECTED_DATA_FILE = "only_batch_corrected_data.csv"


# ---------------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------------

def run_correction(federation: Federation, arm: str, train_folders: List[Path],
                   output_folder: Path) -> Tuple[List[Path], pd.DataFrame]:
    """
    Runs fedRBE on the training inputs, the first client is the coordinator.
    Returns the clients' output folders and the coordinator's batch coefficients.
    """
    output_folders = []
    for cohort, client_folder, config in zip(federation.cohorts, train_folders,
                                             federation.configs[arm]):
        write_fedrbe_config(client_folder, config)
        output_folders.append(output_folder / cohort)
        output_folders[-1].mkdir(parents=True)
    run_quietly(run_simulation_native, [str(f) for f in train_folders],
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


def training_matrix(corrected_training: pd.DataFrame, client: Client) -> pd.DataFrame:
    """
    The client's training data for the forest: fedRBE's corrected values and,
    where fedRBE did not correct a feature (NaN for the batch or left out),
    the uncorrected values. Features (as text) x samples.
    """
    raw = uncorrected_matrix(client)
    raw.index = raw.index.astype(str)
    return corrected_training.combine_first(raw).loc[:, corrected_training.columns]


@dataclass
class TrainingCorrection:
    """The result of fedRBE on the training samples of one federation."""
    coefficients: pd.DataFrame       # the coordinator's batch coefficients
    level: pd.Series                 # the coordinator's training level per hashed feature
    level_info: dict                 # the settings of the run, see align_new_batch
    batches: set                     # all training batch labels
    corrected: List[pd.DataFrame]    # per cohort fedRBE's corrected training data
    matrices: List[pd.DataFrame]     # per cohort the forest's training data, see training_matrix


def correct_training(federation: Federation, fed_inputs: List[ClientInput], arm: str,
                     train_folders: List[Path], split: pd.DataFrame,
                     work: Path) -> TrainingCorrection:
    """Runs fedRBE on the training inputs of every cohort of the federation."""
    configs = federation.configs[arm]
    outputs, coefficients = run_correction(
        federation, arm, train_folders, work / f"{federation.modality or 'data'}_fedrbe")
    level, level_info = read_training_level(str(outputs[0]))
    corrected, matrices = [], []
    for cohort, output, folder, config in zip(federation.cohorts, outputs, train_folders,
                                              configs):
        corrected_training = read_corrected_data(output, config["separator"])
        corrected.append(corrected_training)
        matrices.append(training_matrix(corrected_training,
                                        load_client(folder, cohort, config)))
    return TrainingCorrection(coefficients, level, level_info,
                              set(training_batches(federation, fed_inputs, split)),
                              corrected, matrices)


# ---------------------------------------------------------------------------
# Test samples of the training batches (70/30)
# ---------------------------------------------------------------------------

def client_batch_labels(client: Client) -> List[str]:
    """The fedRBE batch label ("client" or "client|batch") of every sample of a loaded client."""
    if client.batch_col:
        return [f"{client.client_name}|{batch}" for batch in client.design[client.batch_col]]
    return [client.client_name] * len(client.sample_names)


def sample_batches(cohort: str, client: ClientInput, batch_col: Optional[str],
                   samples) -> pd.Series:
    """The fedRBE batch label of the given samples of a client."""
    if batch_col:
        return pd.Series([f"{cohort}|{batch}"
                          for batch in client.design.loc[samples, batch_col].astype(str)],
                         index=pd.Index(samples))
    return pd.Series(cohort, index=pd.Index(samples))


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


# ---------------------------------------------------------------------------
# Held-out cohort (LOCO)
# ---------------------------------------------------------------------------

def align_held_out(client: Client, config: dict, correction: TrainingCorrection) -> pd.DataFrame:
    """
    Shifts each batch of a loaded held-out cohort onto the training level of
    the correction (batchcorrection.classes.client.align_to_training_level, as
    batchcorrection/align_new_batch.py does). No labels are used.
    Returns the aligned cohort, features (as text) x samples.
    """
    normalization = config.get("normalizationMethod") or None
    if normalization != correction.level_info["normalizationMethod"]:
        raise ValueError(f"{client.client_name} uses normalizationMethod {normalization!r}, the "
                         f"training cohorts {correction.level_info['normalizationMethod']!r}")
    aligned, _ = align_to_training_level(client, correction.level)
    aligned.index = aligned.index.astype(str)
    return aligned
