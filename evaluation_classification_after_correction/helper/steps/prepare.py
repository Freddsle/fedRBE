"""
Preparing an evaluation run: which samples are training and test samples
(70/30 split or held-out cohort), and their uncorrected input folders.

A split is a DataFrame with one row per sample key: cohort, sample (the key),
label and part ("train" or "test").
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from ..common import ARM_LABEL_FREE, run_quietly
from .datasets import ClientInput, DatasetSpec, Federation
from batchcorrection.classes.client import Client

TRAIN_RATIO = 0.7
# fedRBE's default of 3 leaves no features for LOCO folds of datasets with
# three cohorts, which have only two training clients
LOCO_MIN_CLIENTS = 2


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


def fold_split(spec: DatasetSpec, inputs: List[List[ClientInput]], held_out: str) -> pd.DataFrame:
    """The LOCO split: test for the samples of the held-out cohort, train for all others."""
    rows = []
    for cohort_idx, cohort in enumerate(spec.cohorts):
        strata = _strata(spec, inputs, cohort_idx)
        rows.append(pd.DataFrame({
            "cohort": cohort, "sample": strata.index, "label": strata[spec.target].to_numpy(),
            "part": "test" if cohort == held_out else "train",
        }))
    return pd.concat(rows, ignore_index=True)


def fold_spec(spec: DatasetSpec, held_out: str) -> DatasetSpec:
    """
    The dataset without the held-out cohort: every federation keeps the other
    cohorts, with copied configs that use min_clients 2. Positions and
    reference_batch stay as they are; fedRBE finds a valid order and reference
    without the held-out cohort, and the reference does not change the
    correction.
    """
    if held_out not in spec.cohorts:
        raise ValueError(f"{spec.name}: unknown cohort {held_out}")
    if len(spec.cohorts) < 3:
        raise ValueError(f"{spec.name}: LOCO needs at least three cohorts, fedRBE needs two "
                         "training clients")
    federations = []
    for federation in spec.federations:
        keep = [idx for idx, cohort in enumerate(federation.cohorts) if cohort != held_out]
        configs = {arm: [{**copy.deepcopy(arm_configs[idx]), "min_clients": LOCO_MIN_CLIENTS}
                         for idx in keep]
                   for arm, arm_configs in federation.configs.items()}
        federations.append(Federation(federation.modality,
                                      [federation.cohorts[idx] for idx in keep],
                                      [federation.client_dirs[idx] for idx in keep], configs))
    return DatasetSpec(spec.name, spec.data_name, spec.target, federations, spec.sample_key)


def cohort_labels(split: pd.DataFrame, cohort: str, part: str) -> pd.Series:
    """The labels of a cohort's samples in one part of the split, indexed by sample key."""
    rows = split.loc[(split.cohort == cohort) & (split.part == part)]
    return rows.set_index("sample")["label"]


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
    of the original files to <folder>/<federation>/<part>/<cohort>/. Only the
    clients with samples in a part are written for that part.
    Returns per (federation index, part) the client folders, in the order of
    the federation's cohorts.
    """
    folders = {}
    for fed_idx, (federation, fed_inputs) in enumerate(zip(spec.federations, inputs)):
        configs = federation.configs[ARM_LABEL_FREE]
        for part in ("train", "test"):
            part_folders = []
            for cohort, client, config in zip(federation.cohorts, fed_inputs, configs):
                keys = set(split.loc[(split.cohort == cohort) & (split.part == part), "sample"])
                samples = client.design.index[client.keys.isin(keys)]
                if samples.empty:
                    continue
                client_folder = folder / (federation.modality or "data") / part / cohort
                client_folder.mkdir(parents=True)
                client.data.loc[:, samples].to_csv(client_folder / config["data_filename"],
                                                   sep=config["separator"])
                client.design.loc[samples].to_csv(client_folder / config["design_filename"],
                                                  sep=config.get("design_separator", "\t"))
                part_folders.append(client_folder)
            folders[(fed_idx, part)] = part_folders
    return folders


def write_fedrbe_config(folder: Path, config: dict) -> None:
    """Writes the fedRBE config.yml of a client folder."""
    with open(folder / "config.yml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"flimmaBatchCorrection": config}, f)


def load_client(folder: Path, cohort: str, config: dict) -> Client:
    """
    Loads an input folder as the fedRBE app does (numeric data, normalization,
    hashed feature names), without covariates: labels are not needed to
    correct new samples.
    """
    config = {**config, "covariates": []}
    write_fedrbe_config(folder, config)
    client = Client()
    run_quietly(client.config_based_init, clientname=cohort, input_folder=str(folder))
    return client


def uncorrected_matrix(client: Client) -> pd.DataFrame:
    """The data of a loaded client, features (real names) x samples."""
    return client.data.rename(index=client.hash2feature)
