"""Concurrent writes to the result file and the parallel unit runner."""

import operator
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

COMPONENT = Path(__file__).resolve().parents[1]
with patch.object(sys, "path", [str(COMPONENT), *sys.path]):
    from helper.experiment import run_units
    from helper.steps import evaluate


def result_rows(worker: int, count: int, value: float = 0.5):
    return [dict(data_name="Fixture", data_preprocessing_name="uncorrected", metric_name="MCC",
                 metric_value=value + worker, predicted_client_name=f"cohort_{worker}",
                 cross_validation_method="Train-Test-Split", seed=seed, predicted_target="label",
                 learning_type="federated")
            for seed in range(count)]


class ResultFileTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.results = evaluate.ResultFile(Path(directory.name) / "results" / "report.csv")

    def test_concurrent_writers_keep_every_row(self):
        units = [(self.results.upsert_experiments, (result_rows(worker, 25),))
                 for worker in range(4)]
        # spawned workers re-import helper from the component folder
        with patch.object(sys, "path", [str(COMPONENT), *sys.path]):
            run_units(units, jobs=4)
        rows = pd.read_csv(self.results.path)
        self.assertEqual(len(rows), 100)
        self.assertEqual(rows.groupby("predicted_client_name").size().to_dict(),
                         {f"cohort_{worker}": 25 for worker in range(4)})
        self.assertTrue(self.results.lock_path.exists())
        self.assertEqual([path.name for path in self.results.path.parent.iterdir()
                          if path.name.endswith(".tmp")], [])

    def test_upserts_update_in_place_and_keep_the_permissions(self):
        self.results.upsert_experiments(result_rows(0, 3))
        os.chmod(self.results.path, 0o640)
        self.results.upsert_experiments(result_rows(0, 3, value=0.25) + result_rows(1, 1))
        rows = pd.read_csv(self.results.path)
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows.loc[rows.predicted_client_name == "cohort_0", "metric_value"]
                         .tolist(), [0.25] * 3)
        self.assertEqual(self.results.check_experiment(
            **{key: value for key, value in result_rows(1, 1)[0].items()
               if key != "metric_value"}), 1.5)
        self.assertEqual(stat.S_IMODE(os.stat(self.results.path).st_mode), 0o640)


class RunUnitsTests(unittest.TestCase):
    def test_failures_are_raised(self):
        units = [(operator.truediv, (1, 0)), (operator.add, (1, 2))]
        for jobs in (1, 2):
            with self.subTest(jobs=jobs), self.assertRaises(ZeroDivisionError):
                run_units(units, jobs=jobs)

    def test_jobs_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "jobs"):
            run_units([], jobs=0)


if __name__ == "__main__":
    unittest.main()
