"""
Repeated 70/30 train/test classification in which the batch effect correction
only sees the training samples.

For every dataset and repetition (seed):

1. Each cohort's uncorrected fedRBE input is split 70/30, stratified by the
   label (and by the batch if a client holds several batches). The split is
   shared by all preprocessing arms and both learning types. Quartet
   modalities are split by pseudo-sample, so every modality uses the same
   samples.
2. Per correction arm, the training samples are corrected for each learner:
   fedRBE (batchcorrection/fedrbe.py, run in the native simulation) for the
   federated forest, central limma::removeBatchEffect on the pooled training
   samples for the centralized forest. The label_aware arm uses the label as
   covariate, the label_free arm no covariates. Both return the batch
   coefficients.
3. Federated (one client per cohort) and centralized (pooled cohorts) random
   forests are trained on all training samples, each corrected by its learner's
   correction.
4. The test samples are corrected with the batch coefficients of the learner's
   correction (batchcorrection.classes.client.subtract_batch_effect, no labels
   needed) and predicted. Values a correction did not correct in a batch's
   training samples stay uncorrected in training and test samples. MCC and F1
   are reported pooled over all cohorts ("All") and per test cohort, for both
   learning types.

The uncorrected arm uses the same splits without steps 2 and 4's correction.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .common import (
    ARM_LABEL_FREE, ARMS, DEFAULT_SEEDS, LEARNING_TYPES, PREPROCESSING_UNCORRECTED, RESULTS_FOLDER,
)
from .experiment import SAVED_FILE_PREFIX, FedrbeClassificationExperiment, run_units
from .steps import correct, datasets, evaluate, prepare, train_forest

CROSS_VALIDATION_METHOD = "Train-Test-Split"
# The splits and the batch coefficients of every repetition (fedRBE's and,
# prefixed central_, central limma's) are saved in this folder next to the
# results file
SPLITS_FOLDER_NAME = "train_test_splits"
SPLITS_FOLDER = RESULTS_FOLDER / SPLITS_FOLDER_NAME


class ClassificationExperimentTrainTestSplit(FedrbeClassificationExperiment):
    """
    Repeated stratified 70/30 train/test evaluation of federated and centralized
    random forests on uncorrected data and on data whose batch effect
    correction was fitted on the training samples only.
    """

    cross_validation_method = CROSS_VALIDATION_METHOD

    def __init__(self, spec: datasets.DatasetSpec, arms=ARMS,
                 resultfile: evaluate.ResultFile = evaluate.DEFAULT_RESULTFILE,
                 train_ratio: float = prepare.TRAIN_RATIO,
                 splits_folder: Optional[Path] = SPLITS_FOLDER,
                 work_folder: Optional[Path] = None) -> None:
        super().__init__(spec, arms, resultfile, work_folder)
        self.train_ratio = train_ratio
        self.splits_folder = splits_folder

    def units(self, seeds: List[int], force_run: bool = False):
        """The independent units of this dataset for run_units, one per seed."""
        return [(self.run_experiment, (seed, force_run)) for seed in seeds]

    def run_experiment(self, seed: int, force_run: bool = False) -> None:
        todo = [preprocessing for preprocessing in self.preprocessings
                if force_run or not self._is_cached(seed, preprocessing, self.spec.cohorts)]
        if not todo:
            print(f"Skipping {self.spec.data_name} seed={seed}, results exist in "
                  f"{self.resultfile.path}.")
            return
        print("=" * 80)
        print(f"{self.spec.data_name}: seed {seed}, {', '.join(todo)}")
        print("=" * 80)
        inputs = self._read_inputs()
        split = prepare.split_samples(self.spec, inputs, seed, self.train_ratio)
        arms = [preprocessing for preprocessing in todo if preprocessing in ARMS]
        prepare.check_min_samples(self.spec, inputs, split, arms)
        with self._work_directory(f"train_test_{self.spec.name}_") as tmp:
            work = Path(tmp)
            folders = prepare.write_split_inputs(self.spec, inputs, split, work / "split")
            all_scores, all_coefficients = {}, {}
            for preprocessing in todo:
                tables, all_coefficients[preprocessing] = self._prepare(
                    preprocessing, inputs, split, folders, work / preprocessing)
                all_scores[preprocessing] = self._train_and_score(
                    tables, work / preprocessing, seed)
        # Only publish once every preprocessing succeeded
        self._save_split(seed, split, all_coefficients)
        self._publish(seed, all_scores)

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

    def _prepare(self, preprocessing: str, inputs: List[List[datasets.ClientInput]],
                 split: pd.DataFrame, folders: Dict[Tuple[int, str], List[Path]], work: Path):
        """
        The forest tables of every cohort per learning type and part, corrected
        for the arms (fedRBE for the federated forest, central limma for the
        centralized one). Returns {learning type: {part: {cohort: table}}} and
        the batch coefficients by file name.
        """
        matrices = {learning_type: {part: [[] for _ in self.spec.cohorts]
                                    for part in ("train", "test")}
                    for learning_type in LEARNING_TYPES}
        coefficients_files = {}
        for fed_idx, federation in enumerate(self.spec.federations):
            arm = ARM_LABEL_FREE if preprocessing == PREPROCESSING_UNCORRECTED else preprocessing
            configs = federation.configs[arm]
            train_folders, test_folders = folders[(fed_idx, "train")], folders[(fed_idx, "test")]
            if preprocessing == PREPROCESSING_UNCORRECTED:
                for part, part_folders in (("train", train_folders), ("test", test_folders)):
                    for cohort_idx, (cohort, folder) in enumerate(zip(federation.cohorts,
                                                                      part_folders)):
                        client = prepare.load_client(folder, cohort, configs[cohort_idx])
                        matrix = prepare.uncorrected_matrix(client)
                        for learning_type in LEARNING_TYPES:
                            matrices[learning_type][part][cohort_idx].append(matrix)
                continue
            corrections = self._correct_training(federation, inputs[fed_idx], arm, train_folders,
                                                 split, work)
            suffix = f"_{federation.modality}" if federation.modality else ""
            for learning_type, correction in corrections.items():
                coefficients_files[f"{SAVED_FILE_PREFIX[learning_type]}batch_coefficients"
                                   f"{suffix}.csv"] = correction.coefficients
                for cohort_idx, (cohort, folder) in enumerate(zip(federation.cohorts,
                                                                  test_folders)):
                    corrected_training = correction.corrected[cohort_idx]
                    matrices[learning_type]["train"][cohort_idx].append(
                        correction.matrices[cohort_idx])
                    client = prepare.load_client(folder, cohort, configs[cohort_idx])
                    training_sample_batches = correct.sample_batches(
                        cohort, inputs[fed_idx][cohort_idx], configs[cohort_idx].get("batch_col"),
                        corrected_training.columns)
                    matrices[learning_type]["test"][cohort_idx].append(correct.correct_new_samples(
                        client, correction.coefficients, correction.batches, corrected_training,
                        training_sample_batches))
        tables = {learning_type: {part: {
            cohort: train_forest.forest_table(
                self.spec, matrices[learning_type][part][cohort_idx],
                [fed_inputs[cohort_idx] for fed_inputs in inputs],
                prepare.cohort_labels(split, cohort, part))
            for cohort_idx, cohort in enumerate(self.spec.cohorts)}
            for part in ("train", "test")}
            for learning_type in LEARNING_TYPES}
        return tables, coefficients_files


def run_train_test_split_experiments(argv=None) -> None:
    """CLI of the repeated train/test evaluation, see the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", help="Dataset IDs (default: all)")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS),
                        help="Correction arms besides the uncorrected baseline (default: both)")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                        help="One repetition with a different split per seed")
    parser.add_argument("--results-file", type=Path, default=evaluate.DEFAULT_RESULTFILE.path)
    parser.add_argument("--work-dir", type=Path, default=None,
                        help="Parent folder of the temporary files (default: system temp)")
    parser.add_argument("--jobs", type=int, default=1,
                        help="Units (dataset and seed) run at the same time in separate processes; each "
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
        experiment = ClassificationExperimentTrainTestSplit(
            registry[name], arms=args.arms, resultfile=resultfile,
            splits_folder=args.results_file.parent / SPLITS_FOLDER_NAME,
            work_folder=args.work_dir)
        units.extend(experiment.units(args.seeds, args.force_run))
    run_units(units, args.jobs)
