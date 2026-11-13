"""
The datasets of the evaluations and the clients' uncorrected fedRBE inputs.

A dataset has one federation (fedRBE run) per modality; the fedRBE configs per
correction arm are those of generate_fedrbe_corrected_datasets.py.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml

from ..common import ARM_LABEL_AWARE, ARM_LABEL_FREE, EVALUATION_DATA


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
