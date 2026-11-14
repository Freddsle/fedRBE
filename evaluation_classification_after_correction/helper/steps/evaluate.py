"""
Evaluating a trained forest on the test data: predictions, MCC and F1 scores,
and the CSV result file the scores are recorded in.
"""

from __future__ import annotations

import contextlib
import fcntl
import math
import os
from pathlib import Path
import stat
import tempfile
from typing import Dict, Iterable, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ..common import ALL_COHORTS, RESULTS_FOLDER
from evaluation_utils.metrics import classification_scores


# ---------------------------------------------------------------------------
# Predictions and scores
# ---------------------------------------------------------------------------

def predict(model: RandomForestClassifier,
            model_info: dict,
            test_data: pd.DataFrame,
            predicted_column: str) -> tuple:
    """
    Predict the samples of *test_data* (samples x features including
    *predicted_column*) with a trained forest. Features of the model missing
    in the test data are added as NaN.
    Returns the true labels and the predictions.
    """
    used_features = model_info['used_features']

    # Transpose if features ended up as rows rather than columns
    if not any(col in test_data.columns for col in used_features):
        test_data = test_data.transpose()

    required_cols = used_features + [predicted_column]
    if predicted_column not in test_data.columns:
        raise ValueError(f"The test data has no label column {predicted_column}")
    if not all(col in test_data.columns for col in required_cols):
        missing = set(required_cols) - set(test_data.columns)
        print(f"The test data misses {len(missing)} of the model's {len(used_features)} "
              "features, adding NaNs for them.")
        missing_df = pd.DataFrame(
            {feat: float('nan') for feat in missing}, index=test_data.index
        )
        test_data = pd.concat([test_data, missing_df], axis=1)

    true_labels = test_data[predicted_column].values
    predictions = model.predict(test_data[used_features])
    return true_labels, predictions


def score_predictions(predictions: Dict[str, Tuple[np.ndarray, np.ndarray]],
                      pooled: bool = True) -> Dict[str, Dict[str, float]]:
    """Scores per test cohort and, with pooled, over all test cohorts ("All")."""
    scores = {}
    if pooled:
        truth = np.concatenate([true for true, _ in predictions.values()])
        predicted = np.concatenate([pred for _, pred in predictions.values()])
        scores[ALL_COHORTS] = classification_scores(truth, predicted)
    for cohort, (true, pred) in predictions.items():
        scores[cohort] = classification_scores(true, pred)
    return scores


# ---------------------------------------------------------------------------
# Result file
# ---------------------------------------------------------------------------

def _matching_rows(df: pd.DataFrame, **key) -> pd.Series:
    """Rows whose identifying columns (every column except ``metric_value``) equal *key*."""
    mask = pd.Series(True, index=df.index)
    for column, value in key.items():
        mask &= df[column] == value
    return mask


class ResultFile:
    """
    Manages a CSV result file for classification experiments.

    The file has these columns::

        data_name, data_preprocessing_name, metric_name, metric_value,
        predicted_client_name, cross_validation_method, seed, predicted_target,
        learning_type

    Several processes can write to the same file (see experiment.run_units):
    writes hold an exclusive lock on <file>.lock and replace the file
    atomically, so readers never see a partly written file.
    """

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path)

    @property
    def lock_path(self) -> Path:
        return self.path.with_name(self.path.name + ".lock")

    @contextlib.contextmanager
    def _locked(self):
        """Holds the exclusive write lock of the file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.lock_path, "a", encoding="utf-8") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _write(self, df: pd.DataFrame) -> None:
        """Replaces the file atomically with df."""
        handle, temporary = tempfile.mkstemp(dir=self.path.parent, prefix=f".{self.path.name}.",
                                             suffix=".tmp")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as f:
                df.to_csv(f, index=False)
            # mkstemp creates the file private (0600), keep the file's permissions
            mode = stat.S_IMODE(os.stat(self.path).st_mode) if self.path.exists() else 0o644
            os.chmod(temporary, mode)
            os.replace(temporary, self.path)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary)
            raise

    def check_experiment(self,
                         data_name: str,
                         data_preprocessing_name: str,
                         metric_name: str,
                         predicted_client_name: str,
                         cross_validation_method: str,
                         seed: int,
                         predicted_target: str,
                         learning_type: str) -> float:
        """
        Return the existing metric value if there is already a row whose values match *all*
        provided columns (every column except ``metric_value``).

        Use this to detect whether a given experiment result has been recorded
        before so that re-running can be skipped.
        """
        if not self.path.exists():
            return False
        df = pd.read_csv(self.path)
        if 'learning_type' not in df.columns:
            return False
        mask = _matching_rows(df, data_name=data_name, data_preprocessing_name=data_preprocessing_name,
                              metric_name=metric_name, predicted_client_name=predicted_client_name,
                              cross_validation_method=cross_validation_method, seed=seed,
                              predicted_target=predicted_target, learning_type=learning_type)
        if mask.any():
            value = pd.to_numeric(df.loc[mask, 'metric_value'].iloc[0], errors='coerce')
            if math.isfinite(value):
                return value
        return False

    def upsert_experiment(self,
                          data_name: str,
                          data_preprocessing_name: str,
                          metric_name: str,
                          metric_value: float,
                          predicted_client_name: str,
                          cross_validation_method: str,
                          seed: int,
                          predicted_target: str,
                          learning_type: str) -> None:
        """Writes one row, see upsert_experiments."""
        self.upsert_experiments([dict(
            data_name=data_name, data_preprocessing_name=data_preprocessing_name,
            metric_name=metric_name, metric_value=metric_value,
            predicted_client_name=predicted_client_name,
            cross_validation_method=cross_validation_method, seed=seed,
            predicted_target=predicted_target, learning_type=learning_type)])

    def upsert_experiments(self, rows: Iterable[dict]) -> None:
        """
        Writes rows (dicts with every column of the file) to the result file in
        one locked, atomic update.

        If a row with matching identifying columns already exists, its
        ``metric_value`` is updated in place.  Otherwise a new row is appended.
        """
        rows = list(rows)
        if not rows:
            return
        with self._locked():
            df = pd.read_csv(self.path) if self.path.exists() else None
            for data in rows:
                if df is None:
                    df = pd.DataFrame([data])
                    continue
                if 'learning_type' not in df.columns:
                    df['learning_type'] = pd.NA
                mask = _matching_rows(df, **{column: value for column, value in data.items()
                                             if column != 'metric_value'})
                if mask.any():
                    df.loc[mask, 'metric_value'] = data['metric_value']
                else:
                    df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
            self._write(df)


DEFAULT_RESULTFILE = ResultFile(RESULTS_FOLDER / "classification_metric_report.csv")
