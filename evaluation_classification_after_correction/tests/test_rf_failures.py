"""Reject failed, incomplete and stale random-forest evaluation outputs."""

import contextlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile
from threading import Barrier, Thread
import unittest
from unittest.mock import Mock, patch

import pandas as pd

COMPONENT = Path(__file__).resolve().parents[1]
with patch.object(sys, "path", [str(COMPONENT), *sys.path]):
    from helper import helper_run_classification_report_metrics as helper


class NativeSimulationFailureTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.generic = self.root / "generic"
        self.generic.mkdir()
        (self.generic / "config.txt").write_text("settings")
        self.clientpaths, self.outputfolders = [], []
        for index in range(2):
            client = self.root / f"client{index}"
            output = self.root / f"output{index}"
            client.mkdir()
            output.mkdir()
            self.clientpaths.append(str(client))
            self.outputfolders.append(str(output))

    def run_bounded(self, algorithm):
        outcome = []

        def run():
            try:
                helper.run_simulation_native(
                    self.clientpaths, self.outputfolders, str(self.generic), algorithm)
            except BaseException as error:
                outcome.append(error)

        # Daemon children allow the test process to exit even if a regression hangs.
        runner = Thread(target=run, daemon=True)
        runner.start()
        runner.join(timeout=3)
        self.assertFalse(runner.is_alive(), "Native simulation hung after a client failure")
        for clientpath in self.clientpaths:
            self.assertFalse((Path(clientpath) / "config.txt").exists())
        self.assertTrue((self.generic / "config.txt").exists())
        return outcome[0] if outcome else None

    def test_early_failure_releases_each_communication_wait(self):
        for method, args in (
            ("gather_data", ()), ("await_data", ()),
            ("send_data_to_coordinator", ("second",)),
            ("broadcast_data", ("second",)),
        ):
            with self.subTest(method=method):
                ready = Barrier(2, timeout=2)
                original = ValueError("client input is invalid")

                def algorithm(client, *_):
                    if client.is_coordinator:
                        if method == "send_data_to_coordinator":
                            client.send_data_to_coordinator("first")
                        elif method == "broadcast_data":
                            client.broadcast_data("first")
                            client.await_data()
                        ready.wait()
                        getattr(client, method)(*args)
                    else:
                        ready.wait()
                        raise original

                self.assertIs(self.run_bounded(algorithm), original)

    def test_exchange_completes_and_late_failure_is_reported(self):
        for fail in (False, True):
            with self.subTest(fail=fail):
                results = {}
                ready = Barrier(2, timeout=2)
                original = ValueError("metric write failed")

                def algorithm(client, *_):
                    client.send_data_to_coordinator(client.client_id)
                    if client.is_coordinator:
                        client.broadcast_data(sum(client.gather_data()))
                    results[client.client_id] = client.await_data()
                    ready.wait()
                    if fail and not client.is_coordinator:
                        raise original

                self.assertIs(self.run_bounded(algorithm), original if fail else None)
                self.assertEqual(results, {0: 1, 1: 1})


class RandomForestFailureTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        for name in ("A", "B"):
            folder = self.root / name
            folder.mkdir()
            pd.DataFrame({"signal": [0., 1.], "label": [0, 1]},
                         index=[f"{name}_0", f"{name}_1"]).to_csv(
                folder / "data.csv", index_label="sample_id")
        descriptor = self.root / "datainfo.json"
        descriptor.write_text(json.dumps({
            "data_name": "fixture", "covariates": [], "prediction_targets": ["label"],
            "datafile": {"filename": "data.csv", "separator": ",",
                         "rotation": "samples x features", "samplename_column": "sample_id"},
            "cohorts": [{"name": name, "folder": name} for name in ("A", "B")],
        }))
        self.info = helper.DataInfo(descriptor)

    def metrics(self, path, scores=(0., .75)):
        pd.DataFrame({"class": ["overall", "overall"], "metric": ["MCC", "F1_score"],
                      "score": scores}).to_csv(path, index=False)

    def assert_clean(self):
        self.assertFalse(list(self.root.glob("tmp_classification_evaluation_*")))
        for folder in self.info.cohort_folders:
            self.assertFalse((folder / "tmp_data.csv").exists())
            self.assertFalse((folder / "config_forest.yaml").exists())
            self.assertTrue((folder / "data.csv").is_file())

    def test_missing_or_invalid_metrics_publish_no_rows(self):
        result = Mock()
        experiment = helper.ClassificationExperimentTrainTestSplit(
            "uncorrected", self.info, resultfile=result)
        complete = self.root / "complete.csv"
        self.metrics(complete)
        invalid = self.root / "invalid.csv"
        valid = pd.read_csv(complete)
        frames = [
            valid.iloc[:1], valid.assign(metric="MCC"), valid.drop(columns="class"),
            *[valid.assign(score=[0., value]) for value in (float("nan"), float("inf"))],
        ]
        for frame in frames:
            with self.subTest(frame=frame.to_dict("list")):
                frame.to_csv(invalid, index=False)
                with self.assertRaises(ValueError):
                    experiment.read_update_metrics(
                        [(complete, "All"), (invalid, "B")], 42, "label", "federated")
                result.upsert_experiment.assert_not_called()
        with self.assertRaises(FileNotFoundError):
            experiment.read_update_metrics(
                [(complete, "All"), (self.root / "missing.csv", "B")],
                42, "label", "federated")
        result.upsert_experiment.assert_not_called()

    def test_stale_outputs_cannot_supply_missing_new_results(self):
        for experiment_type in (helper.ClassificationExperimentTrainTestSplit,
                                helper.ClassificationExperimentLeaveOneCohortOut):
            for learning in helper.LEARNING_TYPES:
                with self.subTest(experiment=experiment_type.__name__, learning=learning):
                    loco = experiment_type is helper.ClassificationExperimentLeaveOneCohortOut
                    suffix = "testcohort|A_" if loco else ""
                    suffix += "predictedcol|label"
                    if learning == "federated":
                        suffix += "_traincohort|" + ("B" if loco else "A")
                    folder = self.root / (
                        "tmp_classification_evaluation_output_learningtype|"
                        f"{learning}_{suffix}")
                    folder.mkdir()
                    (folder / "global_model.pkl").write_bytes(b"stale model")
                    self.metrics(folder / "global_metrics.csv")

                    def no_new_outputs(**kwargs):
                        for output in kwargs["outputfolders"]:
                            self.assertEqual(list(Path(output).iterdir()), [])

                    result = Mock()
                    experiment = experiment_type("uncorrected", self.info, resultfile=result)
                    with patch.object(helper, "LEARNING_TYPES", [learning]), \
                            patch.object(helper, "run_simulation_native", side_effect=no_new_outputs), \
                            contextlib.redirect_stdout(io.StringIO()), \
                            self.assertRaises(FileNotFoundError):
                        experiment.run_experiment(43, force_run=True)
                    self.assert_clean()
                    result.upsert_experiment.assert_not_called()

    def test_cache_requires_complete_finite_local_scores(self):
        result = helper.ResultFile(self.root / "metrics.csv")
        experiment = helper.ClassificationExperimentTrainTestSplit(
            "uncorrected", self.info, resultfile=result)
        complete = self.root / "complete.csv"
        self.metrics(complete)
        with contextlib.redirect_stdout(io.StringIO()):
            for learning in helper.LEARNING_TYPES:
                experiment.read_update_metrics([(complete, "All")], 42, "label", learning)

        def write_outputs(**kwargs):
            folders = [Path(folder) for folder in kwargs["outputfolders"]]
            self.assertEqual(len(folders), 2)
            self.metrics(folders[0] / "global_metrics.csv")
            for folder in folders:
                self.metrics(folder / "local_metrics.csv")

        with patch.object(helper, "run_simulation_native", side_effect=write_outputs) as runner, \
                contextlib.redirect_stdout(io.StringIO()):
            experiment.run_experiment(42)
            runner.assert_called_once()
            experiment.run_experiment(42)
            runner.assert_called_once()
            cached = pd.read_csv(result.path)
            self.assertEqual(len(cached), 8)
            cached.loc[(cached.predicted_client_name == "B") & (cached.metric_name == "MCC"),
                       "metric_value"] = float("nan")
            cached.to_csv(result.path, index=False)
            experiment.run_experiment(42)
            self.assertEqual(runner.call_count, 2)
        self.assertTrue(pd.read_csv(result.path).metric_value.map(math.isfinite).all())
        self.assert_clean()


if __name__ == "__main__":
    unittest.main()
