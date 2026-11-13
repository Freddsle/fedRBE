"""
The shared run loop of the random forest evaluations (train_test_split.py and
loco.py): result rows and their caching, and training plus scoring the
federated and centralized forests on prepared tables.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
from typing import List, Optional

from .common import (
    ALL_COHORTS, ARMS, LEARNING_TYPES, METRICS, PREPROCESSING_UNCORRECTED, run_quietly,
)
from .steps import datasets, evaluate, train_forest


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
        """Writes the scores {preprocessing: {learning type: {client: metrics}}}."""
        for preprocessing, scores_by_type in all_scores.items():
            for learning_type, scores in scores_by_type.items():
                for client, metrics in scores.items():
                    print(f"seed {seed} {preprocessing} {learning_type} {client}: "
                          f"MCC {metrics['MCC']:.4f}, F1 {metrics['F1_score']:.4f}")
                    for metric in METRICS:
                        self.resultfile.upsert_experiment(
                            metric_value=metrics[metric],
                            **self._result_key(seed, preprocessing, learning_type, client, metric))

    def _train_and_score(self, tables, work: Path, seed: int):
        """
        Trains both learning types on the training tables {cohort: table} and
        scores the test tables {cohort: table}.
        """
        target = self.spec.target
        train_folders = [train_forest.write_forest_input(table, work / "forest_input" / cohort)
                         for cohort, table in tables["train"].items()]
        scores = {}
        for learning_type in LEARNING_TYPES:
            output_folders = run_quietly(
                train_forest.train, learning_type, train_folders, list(tables["train"]), work,
                f"predictedcol|{target}", target, seed, [])
            model, model_info = train_forest.read_model(output_folders[0])
            predictions = {
                cohort: evaluate.predict(model, model_info, table, target)
                for cohort, table in tables["test"].items()
            }
            scores[learning_type] = evaluate.score_predictions(predictions, self.pooled_scores)
        return scores
