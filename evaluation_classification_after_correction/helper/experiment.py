"""
The shared run loop of the random forest evaluations (train_test_split.py and
loco.py): result rows and their caching, training plus scoring the federated
and centralized forests on prepared tables, and running independent units of
an evaluation in parallel processes.
"""

from __future__ import annotations

from concurrent.futures import FIRST_EXCEPTION, ProcessPoolExecutor, wait
import multiprocessing
from pathlib import Path
import tempfile
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .common import (
    ALL_COHORTS, ARMS, LEARNING_TYPE_CENTRALIZED, LEARNING_TYPE_FEDERATED, LEARNING_TYPES,
    METRICS, PREPROCESSING_UNCORRECTED, run_quietly,
)
from .steps import correct, datasets, evaluate, train_forest

# Prefix of the saved files of the central correction, next to fedRBE's
SAVED_FILE_PREFIX = {LEARNING_TYPE_FEDERATED: "", LEARNING_TYPE_CENTRALIZED: "central_"}


class FedrbeClassificationExperiment:
    """
    Shared parts of the random forest evaluations whose batch effect correction
    only sees the training samples: result rows, caching, and training and
    scoring the forests. Subclasses set the cross-validation method and whether
    scores are pooled over the test cohorts.
    """

    cross_validation_method: str
    pooled_scores = True

    def __init__(self, spec: datasets.DatasetSpec, arms=ARMS,
                 resultfile: evaluate.ResultFile = evaluate.DEFAULT_RESULTFILE,
                 work_folder: Optional[Path] = None) -> None:
        unknown = set(arms) - set(ARMS)
        if unknown:
            raise ValueError(f"Unknown correction arms {sorted(unknown)}")
        self.spec = spec
        self.arms = list(arms)
        self.resultfile = resultfile
        self.work_folder = work_folder
        self._inputs = None

    @property
    def preprocessings(self) -> List[str]:
        return [PREPROCESSING_UNCORRECTED, *self.arms]

    def _read_inputs(self) -> List[List[datasets.ClientInput]]:
        if self._inputs is None:
            self._inputs = datasets.read_inputs(self.spec)
        return self._inputs

    def _result_key(self, seed: int, preprocessing: str, learning_type: str,
                    client_name: str, metric_name: str) -> dict:
        return dict(data_name=self.spec.data_name, data_preprocessing_name=preprocessing,
                    metric_name=metric_name, predicted_client_name=client_name,
                    cross_validation_method=self.cross_validation_method, seed=seed,
                    predicted_target=self.spec.target, learning_type=learning_type)

    def _scored_clients(self, test_cohorts: List[str]) -> List[str]:
        """The predicted_client_name of every score row, see evaluate.score_predictions."""
        if not self.pooled_scores:
            return list(test_cohorts)
        return [ALL_COHORTS, *test_cohorts]

    def _is_cached(self, seed: int, preprocessing: str, test_cohorts: List[str]) -> bool:
        for learning_type in LEARNING_TYPES:
            for client in self._scored_clients(test_cohorts):
                for metric in METRICS:
                    value = self.resultfile.check_experiment(
                        **self._result_key(seed, preprocessing, learning_type, client, metric))
                    if value is False:
                        return False
        return True

    def _work_directory(self, prefix: str) -> tempfile.TemporaryDirectory:
        if self.work_folder is not None:
            self.work_folder.mkdir(parents=True, exist_ok=True)
        return tempfile.TemporaryDirectory(prefix=prefix, dir=self.work_folder)

    def _publish(self, seed: int, all_scores) -> None:
        """
        Writes the scores {preprocessing: {learning type: {client: metrics}}} in
        one update of the result file.
        """
        rows = []
        for preprocessing, scores_by_type in all_scores.items():
            for learning_type, scores in scores_by_type.items():
                for client, metrics in scores.items():
                    print(f"{self.spec.data_name} seed {seed} {preprocessing} {learning_type} "
                          f"{client}: MCC {metrics['MCC']:.4f}, F1 {metrics['F1_score']:.4f}")
                    rows.extend(
                        dict(metric_value=metrics[metric],
                             **self._result_key(seed, preprocessing, learning_type, client,
                                                metric))
                        for metric in METRICS)
        self.resultfile.upsert_experiments(rows)

    @staticmethod
    def _correct_training(federation: datasets.Federation,
                          fed_inputs: List[datasets.ClientInput], arm: str,
                          train_folders: List[Path], split: pd.DataFrame,
                          work: Path) -> Dict[str, correct.TrainingCorrection]:
        """
        The correction of the training data per learning type: fedRBE for the
        federated forest, central limma for the centralized one.
        """
        return {
            LEARNING_TYPE_FEDERATED: correct.correct_training(
                federation, fed_inputs, arm, train_folders, split, work),
            LEARNING_TYPE_CENTRALIZED: correct.correct_training_central(
                federation, fed_inputs, arm, train_folders, split, work),
        }

    def _train_and_score(self, tables, work: Path, seed: int):
        """
        Trains each learning type on its training tables and scores its test
        tables: tables is {learning type: {"train"/"test": {cohort: table}}}.
        """
        target = self.spec.target
        scores = {}
        for learning_type in LEARNING_TYPES:
            train_tables = tables[learning_type]["train"]
            train_folders = [
                train_forest.write_forest_input(table, work / "forest_input" / learning_type / cohort)
                for cohort, table in train_tables.items()]
            output_folders = run_quietly(
                train_forest.train, learning_type, train_folders, list(train_tables), work,
                f"predictedcol|{target}", target, seed, [])
            model, model_info = train_forest.read_model(output_folders[0])
            predictions = {
                cohort: evaluate.predict(model, model_info, table, target)
                for cohort, table in tables[learning_type]["test"].items()
            }
            scores[learning_type] = evaluate.score_predictions(predictions, self.pooled_scores)
        return scores


def run_units(units: Sequence[Tuple[Callable, Tuple[Any, ...]]], jobs: int = 1) -> None:
    """
    Runs the independent units of an evaluation, each a (function, args) pair,
    for example an experiment's run for one seed or one held-out cohort.
    With jobs > 1, up to jobs units run at the same time in separate processes:
    run_quietly redirects the process-wide stdout, so threads would interfere.
    Units only share the result file, which serializes its writes. On the first
    failure no further units start; running units finish, then the error is
    raised. Every unit publishes its results only once it fully succeeded.
    """
    if jobs < 1:
        raise ValueError(f"jobs must be at least 1, got {jobs}")
    if jobs == 1:
        for function, args in units:
            function(*args)
        return
    # spawn: safe with threads (BLAS, the simulation) of the parent; children
    # get the parent's sys.path and re-import helper
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=jobs, mp_context=context) as executor:
        futures = [executor.submit(function, *args) for function, args in units]
        _, pending = wait(futures, return_when=FIRST_EXCEPTION)
        for future in pending:
            future.cancel()
        wait(pending)
    for future in futures:
        if future.done() and not future.cancelled() and future.exception() is not None:
            raise future.exception()
