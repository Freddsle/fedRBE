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
   For the centralized forest, central limma::removeBatchEffect corrects the
   pooled training cohorts instead, with the same batches and covariates but
   without fedRBE's privacy filters; its training level is defined the same way.
2. Federated (one client per training cohort) and centralized (pooled cohorts)
   random forests are trained on the training cohorts, each corrected by its
   learner's correction. Values a correction did not correct stay uncorrected.
3. Each batch of H is shifted per feature so that its mean equals the training
   level of the learner's correction
   (batchcorrection.classes.client.align_to_training_level, also used by
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
from pathlib import Path
import shutil
from typing import Dict, List, Optional

import pandas as pd
import yaml

from .common import (
    ARM_LABEL_FREE, ARMS, DEFAULT_SEEDS, LEARNING_TYPE_FEDERATED, LEARNING_TYPES,
    PREPROCESSING_UNCORRECTED, RESULTS_FOLDER,
)
from .experiment import SAVED_FILE_PREFIX, FedrbeClassificationExperiment, run_units
from .steps import correct, datasets, evaluate, prepare, train_forest

CROSS_VALIDATION_METHOD = "Leave-One-Cohort-Out"
# The coordinator's output of every fold and arm is saved in this folder next to
# the results file
CORRECTIONS_FOLDER_NAME = "loco_corrections"


class ClassificationExperimentLeaveOneCohortOut(FedrbeClassificationExperiment):
    """
    Leave-one-cohort-out evaluation of federated and centralized random forests
    on uncorrected data and on data whose batch effect correction was fitted on
    the training cohorts only, see the module docstring.
    """

    cross_validation_method = CROSS_VALIDATION_METHOD
    pooled_scores = False

    def __init__(self, spec: datasets.DatasetSpec, arms=ARMS,
                 resultfile: evaluate.ResultFile = evaluate.DEFAULT_RESULTFILE,
                 corrections_folder: Optional[Path] = RESULTS_FOLDER / CORRECTIONS_FOLDER_NAME,
                 work_folder: Optional[Path] = None) -> None:
        super().__init__(spec, arms, resultfile, work_folder)
        self.corrections_folder = corrections_folder

    def units(self, seeds: List[int], force_run: bool = False):
        """The independent units of this dataset for run_units, one per held-out cohort."""
        return [(self.run_fold, (held_out, seeds, force_run)) for held_out in self.spec.cohorts]

    def run_experiment(self, seeds: List[int], force_run: bool = False) -> None:
        """Runs every held-out cohort for the given seeds, one after another."""
        run_units(self.units(seeds, force_run))

    def run_fold(self, held_out: str, seeds: List[int], force_run: bool = False) -> None:
        # per seed the preprocessings without (complete) results, seeds without any are left out
        todo = {seed: missing for seed in seeds
                if (missing := [preprocessing for preprocessing in self.preprocessings
                                if force_run or not self._is_cached(seed, preprocessing,
                                                                    [held_out])])}
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
        training = prepare.fold_spec(self.spec, held_out)
        training_inputs = [[client for idx, client in enumerate(fed_inputs) if idx != held_out_idx]
                           for fed_inputs in inputs]
        split = prepare.fold_split(self.spec, inputs, held_out)
        prepare.check_min_samples(
            training, training_inputs, split,
            [preprocessing for preprocessing in preprocessings if preprocessing in ARMS])
        with self._work_directory(f"loco_{self.spec.name}_") as tmp:
            work = Path(tmp)
            folders = prepare.write_split_inputs(self.spec, inputs, split, work / "split")
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

    def _prepare(self, preprocessing: str, held_out: str, training: datasets.DatasetSpec,
                 training_inputs: List[List[datasets.ClientInput]],
                 inputs: List[List[datasets.ClientInput]], split: pd.DataFrame, folders,
                 work: Path):
        """
        The forest tables per learning type, {learning type: {"train": {cohort:
        table}, "test": {held_out: table}}}, corrected for the arms (fedRBE for
        the federated forest, central limma for the centralized one), and the
        corrections' output per file name.
        """
        held_out_idx = self.spec.cohorts.index(held_out)
        train_matrices = {learning_type: [[] for _ in training.cohorts]
                          for learning_type in LEARNING_TYPES}
        test_matrices = {learning_type: [] for learning_type in LEARNING_TYPES}
        files = {}
        for fed_idx, (federation, training_federation) in enumerate(
                zip(self.spec.federations, training.federations)):
            train_folders = folders[(fed_idx, "train")]
            (test_folder,) = folders[(fed_idx, "test")]
            # labels are not used for the held-out cohort, so always label-free
            held_out_config = federation.configs[ARM_LABEL_FREE][held_out_idx]
            held_out_client = prepare.load_client(test_folder, held_out, held_out_config)
            if preprocessing == PREPROCESSING_UNCORRECTED:
                configs = training_federation.configs[ARM_LABEL_FREE]
                for cohort_idx, (cohort, folder) in enumerate(zip(training_federation.cohorts,
                                                                  train_folders)):
                    matrix = prepare.uncorrected_matrix(
                        prepare.load_client(folder, cohort, configs[cohort_idx]))
                    for learning_type in LEARNING_TYPES:
                        train_matrices[learning_type][cohort_idx].append(matrix)
                for learning_type in LEARNING_TYPES:
                    test_matrices[learning_type].append(
                        prepare.uncorrected_matrix(held_out_client))
                continue
            corrections = self._correct_training(training_federation, training_inputs[fed_idx],
                                                 preprocessing, train_folders, split, work)
            suffix = f"_{federation.modality}" if federation.modality else ""
            for learning_type, correction in corrections.items():
                for cohort_idx, matrix in enumerate(correction.matrices):
                    train_matrices[learning_type][cohort_idx].append(matrix)
                test_matrices[learning_type].append(
                    correct.align_held_out(held_out_client, held_out_config, correction))
                prefix = SAVED_FILE_PREFIX[learning_type]
                files[f"{prefix}batch_coefficients{suffix}.csv"] = correction.coefficients
                files[f"{prefix}training_level{suffix}.csv"] = correction.level.rename(
                    "training_level")
            # the settings of the fedRBE run, see batchcorrection/align_new_batch.py
            files[f"training_level_info{suffix}.yml"] = \
                corrections[LEARNING_TYPE_FEDERATED].level_info
        tables = {learning_type: {
            "train": {cohort: train_forest.forest_table(
                          training, train_matrices[learning_type][cohort_idx],
                          [fed_inputs[cohort_idx] for fed_inputs in training_inputs],
                          prepare.cohort_labels(split, cohort, "train"))
                      for cohort_idx, cohort in enumerate(training.cohorts)},
            "test": {held_out: train_forest.forest_table(
                         self.spec, test_matrices[learning_type],
                         [fed_inputs[held_out_idx] for fed_inputs in inputs],
                         prepare.cohort_labels(split, held_out, "test"))},
        } for learning_type in LEARNING_TYPES}
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
    parser.add_argument("--results-file", type=Path, default=evaluate.DEFAULT_RESULTFILE.path)
    parser.add_argument("--work-dir", type=Path, default=None,
                        help="Parent folder of the temporary files (default: system temp)")
    parser.add_argument("--jobs", type=int, default=1,
                        help="Units (dataset and held-out cohort) run at the same time in separate processes; each "
                             "needs the memory of one unit, ovarian cancer about 4 GB "
                             "(default: 1)")
    parser.add_argument("--force-run", action="store_true")
    args = parser.parse_args(argv)
    registry = datasets.dataset_registry()
    selected = args.datasets or list(registry)
    unknown = set(selected) - set(registry)
    if unknown:
        parser.error(f"Unknown datasets: {', '.join(sorted(unknown))}. "
                     f"Available: {', '.join(registry)}")
    args.results_file.parent.mkdir(parents=True, exist_ok=True)
    resultfile = evaluate.ResultFile(args.results_file)
    units = []
    for name in selected:
        experiment = ClassificationExperimentLeaveOneCohortOut(
            registry[name], arms=args.arms, resultfile=resultfile,
            corrections_folder=args.results_file.parent / CORRECTIONS_FOLDER_NAME,
            work_folder=args.work_dir)
        units.extend(experiment.units(args.seeds, args.force_run))
    run_units(units, args.jobs)
