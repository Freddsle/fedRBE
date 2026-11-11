"""
Leave-one-cohort-out (LOCO) classification in which the batch effect correction
only sees the training cohorts.

For every dataset and held-out cohort H:

1. fedRBE (batchcorrection/fedrbe.py, run in the native simulation) corrects
   the training cohorts, all other cohorts, with min_clients 2 so that datasets
   with three cohorts still have enough clients. The label_aware arm uses the
   label as covariate, the label_free arm no covariates. Besides the batch
   coefficients, the coordinator saves the training level: per feature, the
   mean of the corrected training data.
2. Federated (one client per training cohort) and centralized (pooled cohorts)
   random forests are trained on the corrected training cohorts. Values fedRBE
   did not correct stay uncorrected.
3. Each batch of H is shifted per feature so that its mean equals the training
   level (batchcorrection.classes.client.align_to_training_level, also used by
   batchcorrection/align_new_batch.py). H's labels are not used in either arm:
   if H's label mix differs from the training data's, part of that difference is
   removed as batch effect.
4. The forests predict H. MCC and F1 are reported for H, for both learning types.

The correction is deterministic, so it runs once per held-out cohort; only the
forests run once per seed. The uncorrected arm uses the same cohorts without
steps 1 and 3.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import shutil
from typing import Dict, List, Optional

import pandas as pd
import yaml

from .helper_run_classification_report_metrics import (
    ARM_LABEL_FREE, DEFAULT_RESULTFILE, PREPROCESSING_UNCORRECTED, RESULTS_FOLDER, ResultFile,
)
from .train_test_correction import (
    ARMS, DEFAULT_SEEDS, ClientInput, DatasetSpec, Federation, FedrbeClassificationExperiment,
    _strata, check_min_samples, correct_training, dataset_registry, forest_table, load_client,
    uncorrected_matrix, write_split_inputs,
)
from batchcorrection.classes.client import align_to_training_level

CROSS_VALIDATION_METHOD = "Leave-One-Cohort-Out"
# fedRBE's default of 3 leaves no features for datasets with three cohorts
LOCO_MIN_CLIENTS = 2
# The coordinator's output of every fold and arm is saved in this folder next to
# the results file
CORRECTIONS_FOLDER_NAME = "loco_corrections"


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


def fold_split(spec: DatasetSpec, inputs: List[List[ClientInput]], held_out: str) -> pd.DataFrame:
    """
    One row per sample key: cohort, key, label and part, test for the samples of
    the held-out cohort and train for all others.
    """
    rows = []
    for cohort_idx, cohort in enumerate(spec.cohorts):
        strata = _strata(spec, inputs, cohort_idx)
        rows.append(pd.DataFrame({
            "cohort": cohort, "sample": strata.index, "label": strata[spec.target].to_numpy(),
            "part": "test" if cohort == held_out else "train",
        }))
    return pd.concat(rows, ignore_index=True)


def _labels(split: pd.DataFrame, cohort: str) -> pd.Series:
    return split.loc[split.cohort == cohort].set_index("sample")["label"]


class ClassificationExperimentLeaveOneCohortOut(FedrbeClassificationExperiment):
    """
    Leave-one-cohort-out evaluation of federated and centralized random forests
    on uncorrected data and on data whose batch effect correction was fitted on
    the training cohorts only, see the module docstring.
    """

    cross_validation_method = CROSS_VALIDATION_METHOD
    pooled_scores = False

    def __init__(self, spec: DatasetSpec, arms=ARMS,
                 resultfile: ResultFile = DEFAULT_RESULTFILE,
                 corrections_folder: Optional[Path] = RESULTS_FOLDER / CORRECTIONS_FOLDER_NAME,
                 work_folder: Optional[Path] = None) -> None:
        super().__init__(spec, arms, resultfile, work_folder)
        self.corrections_folder = corrections_folder

    def run_experiment(self, seeds: List[int], force_run: bool = False) -> None:
        """Runs every held-out cohort for the given seeds."""
        for held_out in self.spec.cohorts:
            self.run_fold(held_out, seeds, force_run)

    def run_fold(self, held_out: str, seeds: List[int], force_run: bool = False) -> None:
        todo = {seed: [preprocessing for preprocessing in self.preprocessings
                       if force_run or not self._is_cached(seed, preprocessing, [held_out])]
                for seed in seeds}
        todo = {seed: preprocessings for seed, preprocessings in todo.items() if preprocessings}
        if not todo:
            print(f"Skipping {self.spec.data_name} held-out {held_out}, results exist in "
                  f"{self.resultfile.path}.")
            return
        preprocessings = [preprocessing for preprocessing in self.preprocessings
                          if any(preprocessing in needed for needed in todo.values())]
        print("=" * 80)
        print(f"{self.spec.data_name}: held-out {held_out}, {', '.join(preprocessings)}, "
              f"seeds {', '.join(map(str, todo))}")
        print("=" * 80)
        inputs = self._read_inputs()
        held_out_idx = self.spec.cohorts.index(held_out)
        training = fold_spec(self.spec, held_out)
        training_inputs = [[client for idx, client in enumerate(fed_inputs) if idx != held_out_idx]
                           for fed_inputs in inputs]
        split = fold_split(self.spec, inputs, held_out)
        check_min_samples(training, training_inputs, split,
                          [preprocessing for preprocessing in preprocessings if preprocessing in ARMS])
        with self._work_directory(f"loco_{self.spec.name}_") as tmp:
            work = Path(tmp)
            folders = write_split_inputs(self.spec, inputs, split, work / "split")
            tables, corrections = {}, {}
            for preprocessing in preprocessings:
                tables[preprocessing], corrections[preprocessing] = self._prepare(
                    preprocessing, held_out, training, training_inputs, inputs, split, folders,
                    work / preprocessing)
            scores = {seed: {preprocessing: self._train_and_score(
                                 tables[preprocessing], work / preprocessing / f"seed_{seed}", seed)
                             for preprocessing in needed}
                      for seed, needed in todo.items()}
        # Only publish once the whole fold succeeded
        self._save_corrections(held_out, corrections)
        for seed, seed_scores in scores.items():
            self._publish(seed, seed_scores)

    def _prepare(self, preprocessing: str, held_out: str, training: DatasetSpec,
                 training_inputs: List[List[ClientInput]], inputs: List[List[ClientInput]],
                 split: pd.DataFrame, folders, work: Path):
        """
        The forest tables {"train": {cohort: table}, "test": {held_out: table}}
        and, for the arms, the coordinator's output per file name.
        """
        held_out_idx = self.spec.cohorts.index(held_out)
        train_matrices = [[] for _ in training.cohorts]
        test_matrices = []
        files = {}
        for fed_idx, (federation, training_federation) in enumerate(
                zip(self.spec.federations, training.federations)):
            train_folders = folders[(fed_idx, "train")]
            (test_folder,) = folders[(fed_idx, "test")]
            # labels are not used for the held-out cohort, so always label-free
            held_out_config = federation.configs[ARM_LABEL_FREE][held_out_idx]
            held_out_client = load_client(test_folder, held_out, held_out_config)
            if preprocessing == PREPROCESSING_UNCORRECTED:
                configs = training_federation.configs[ARM_LABEL_FREE]
                for cohort_idx, (cohort, folder) in enumerate(zip(training_federation.cohorts,
                                                                  train_folders)):
                    train_matrices[cohort_idx].append(
                        uncorrected_matrix(load_client(folder, cohort, configs[cohort_idx])))
                test_matrices.append(uncorrected_matrix(held_out_client))
                continue
            correction = correct_training(training_federation, training_inputs[fed_idx],
                                          preprocessing, train_folders, split, work)
            for cohort_idx, matrix in enumerate(correction.matrices):
                train_matrices[cohort_idx].append(matrix)
            normalization = held_out_config.get("normalizationMethod") or None
            if normalization != correction.level_info["normalizationMethod"]:
                raise ValueError(f"{held_out} uses normalizationMethod {normalization!r}, the "
                                 "training cohorts "
                                 f"{correction.level_info['normalizationMethod']!r}")
            aligned, _ = align_to_training_level(held_out_client, correction.level)
            aligned.index = aligned.index.astype(str)
            test_matrices.append(aligned)
            suffix = f"_{federation.modality}" if federation.modality else ""
            files[f"batch_coefficients{suffix}.csv"] = correction.coefficients
            files[f"training_level{suffix}.csv"] = correction.level.rename("training_level")
            files[f"training_level_info{suffix}.yml"] = correction.level_info
        tables = {
            "train": {cohort: forest_table(training, train_matrices[cohort_idx],
                                           [fed_inputs[cohort_idx] for fed_inputs in training_inputs],
                                           _labels(split, cohort))
                      for cohort_idx, cohort in enumerate(training.cohorts)},
            "test": {held_out: forest_table(self.spec, test_matrices,
                                            [fed_inputs[held_out_idx] for fed_inputs in inputs],
                                            _labels(split, held_out))},
        }
        return tables, files

    def _save_corrections(self, held_out: str, corrections: Dict[str, dict]) -> None:
        """
        Saves per arm the coordinator's batch coefficients, training level and
        its settings to <corrections_folder>/<dataset>/<held_out>/<arm>/. Rows
        are the hashed feature names (batchcorrection.classes.client.Client.hash_names).
        """
        if self.corrections_folder is None:
            return
        for preprocessing, files in corrections.items():
            if preprocessing not in ARMS:
                continue
            folder = self.corrections_folder / self.spec.name / held_out / preprocessing
            if folder.exists():
                shutil.rmtree(folder)
            folder.mkdir(parents=True)
            for name, content in files.items():
                if isinstance(content, dict):
                    # the tables are saved comma separated, so that
                    # batchcorrection/align_new_batch.py can read the folder
                    content = {**content, "separator": ","}
                    (folder / name).write_text(yaml.safe_dump(content, sort_keys=False))
                else:
                    content.to_csv(folder / name, sep=",")


def run_loco_experiments(argv=None) -> None:
    """CLI of the leave-one-cohort-out evaluation, see the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", help="Dataset IDs (default: all)")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS),
                        help="Correction arms besides the uncorrected baseline (default: both)")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                        help="Random forest seeds; the correction does not depend on them")
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
        experiment = ClassificationExperimentLeaveOneCohortOut(
            registry[name], arms=args.arms, resultfile=resultfile,
            corrections_folder=args.results_file.parent / CORRECTIONS_FOLDER_NAME,
            work_folder=args.work_dir)
        experiment.run_experiment(seeds=args.seeds, force_run=args.force_run)
