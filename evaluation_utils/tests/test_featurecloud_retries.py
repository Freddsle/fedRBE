"""Exercise retry control flow without starting Docker or a controller."""

import contextlib
import io
import itertools
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from evaluation_utils import featurecloud_api_extension as api
from evaluation_utils import featurecloud_kmeans_utils as kmeans


def test_info(status="finished", coordinator=True):
    instances = [
        {"id": 1, "name": "second", "coordinator": coordinator},
        {"id": 0, "name": "first", "coordinator": False},
    ]
    return pd.DataFrame([{"status": status, "instances": instances}])


class FeatureCloudRetryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.experiment = api.Experiment(
            name="fixture", clients=[], app_image_name="fixture:latest",
            fc_data_dir=str(self.root), timeout=900,
            controller_host=api.DEFAULT_CONTROLLER_HOST, query_interval=5,
        )
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))
        self.enterContext(patch.object(api.docker, "from_env"))
        self.enterContext(patch.object(
            api.fc_controller, "status", return_value=SimpleNamespace(status="running")
        ))
        self.enterContext(patch.object(api.Experiment, "_kill_leftover_container"))
        self.startup = self.enterContext(patch.object(api.Experiment, "_startup"))
        self.sleep = self.enterContext(patch.object(api.time, "sleep"))
        self.stop = self.enterContext(patch.object(api.fc, "stop"))
        self.info = self.enterContext(patch.object(api.fc, "info"))
        self.metadata = [
            api.ExperimentResult("first", "first-hashes", "first-config"),
            api.ExperimentResult("replacement", "new-hashes", "new-config"),
        ]

    def starts(self):
        return self.enterContext(patch.object(
            api.Experiment, "_start_test",
            side_effect=[(101, self.metadata[0]), (202, self.metadata[1])],
        ))

    def assert_replacement(self, result):
        files, coordinator, metadata = result
        self.assertEqual(files, [
            str(self.root / "tests/tests/results_test_202_client_0_first.zip"),
            str(self.root / "tests/tests/results_test_202_client_1_second.zip"),
        ])
        self.assertEqual(coordinator, 1)
        self.assertEqual(metadata, self.metadata[1])
        self.assertEqual(
            [entry.kwargs["test_id"] for entry in self.info.call_args_list],
            [101, 202],
        )

    def test_timeout_returns_replacement_id_archives_and_metadata(self):
        self.starts()
        self.info.side_effect = [test_info("running"), test_info()]
        with patch.object(api.time, "monotonic", side_effect=[0, 901, 902]):
            self.assert_replacement(self.experiment.run_test())
        self.stop.assert_called_once_with(
            controller_host=api.DEFAULT_CONTROLLER_HOST, test_id=101
        )

    def test_info_failure_returns_replacement_without_polling_old_test(self):
        self.starts()
        self.info.side_effect = [ConnectionError("controller unavailable"), test_info()]
        self.assert_replacement(self.experiment.run_test())

    def test_missing_coordinator_returns_replacement(self):
        self.starts()
        self.info.side_effect = [test_info(coordinator=False), test_info()]
        self.assert_replacement(self.experiment.run_test())

    def test_different_failure_types_share_one_retry_budget(self):
        client = self.root / "client"
        client.mkdir()
        self.experiment.clients = [str(client)]
        self.experiment.config_files = [{}]
        self.info.side_effect = [
            ConnectionError("status unavailable"), test_info(coordinator=False),
            test_info("running"), ConnectionError("status still unavailable"),
        ]
        with patch.object(api.Experiment, "_get_input_hashes",
                          side_effect=[OSError("input unavailable")] + [{}] * 5) as hashes, \
                patch.object(api.fc, "start",
                             side_effect=[ConnectionError("launch failed"), 301, 302, 303, 304]) \
                as start, \
                patch.object(api.time, "monotonic", side_effect=itertools.count(0, 901)):
            with self.assertRaisesRegex(RuntimeError, "failed after 5 retries"):
                self.experiment.run_test()
        self.assertEqual(hashes.call_count, 6)
        self.assertEqual(start.call_count, 5)
        self.assertEqual(self.startup.call_count, 5)

    def test_start_failure_retries_and_preserves_absolute_generic_directory(self):
        client = self.root / "client"
        client.mkdir()
        shared = self.root / "shared"
        shared.mkdir()
        self.experiment.clients = [str(client)]
        self.experiment.config_files = [{}]
        self.experiment.generic_dir = str(shared)
        self.info.return_value = test_info()
        with patch.object(api.fc, "start", side_effect=[ConnectionError("offline"), 202]) \
                as start:
            files, _, _ = self.experiment.run_test()
        self.assertEqual(
            [entry.kwargs["generic_dir"] for entry in start.call_args_list],
            ["shared", "shared"],
        )
        self.assertEqual(self.experiment.generic_dir, str(shared))
        self.assertTrue(all("results_test_202_" in path for path in files))

    def test_polling_waits_for_configured_interval(self):
        self.starts()
        self.experiment.query_interval = 3
        self.info.side_effect = [test_info("running"), test_info()]
        self.experiment.run_test()
        self.sleep.assert_called_once_with(3)
        self.startup.assert_not_called()

    def test_terminal_error_is_not_retried(self):
        start = self.starts()
        self.info.return_value = test_info("error")
        with self.assertRaisesRegex(RuntimeError, "Status: error"):
            self.experiment.run_test()
        start.assert_called_once()
        self.startup.assert_not_called()

    def test_kmeans_uses_replacement_and_preserves_missing_coordinator_fallback(self):
        self.starts()
        finished = test_info()
        for instance in finished.iloc[0]["instances"]:
            del instance["coordinator"]
        self.info.side_effect = [ConnectionError("controller unavailable"), finished]
        for name in ("first", "second"):
            client = self.root / name
            client.mkdir()
            (client / "config_kmeans.yml").write_text("fixture: true\n")
        with patch.object(kmeans, "wait_for_paths"), \
                patch.object(kmeans.shutil, "copy2"), \
                patch.object(kmeans, "extract_clustering"), \
                patch.object(kmeans, "aggregate_fed_clusters", return_value=pd.DataFrame()):
            kmeans.run_single_federated_variant(
                dataset_name="fixture", variant_label="before",
                variant_input_dir=self.root, dataset_root=self.root / "output",
                metadata=pd.DataFrame(), client_names=["first", "second"], k_values=[2],
                app_image="fixture:latest", controller_host=api.DEFAULT_CONTROLLER_HOST,
                query_interval=5, timeout=900, keep_extracted=False, aggregate_only=False,
            )
        manifest = json.loads((
            self.root / "output/fc_kmeans_res/before/1_fed_clustering/manifest.json"
        ).read_text())
        self.assertEqual(manifest["test_id"], 202)
        self.assertEqual(manifest["zip_files"], [
            "results_test_202_client_0_first.zip",
            "results_test_202_client_1_second.zip",
        ])
        self.assertEqual(manifest["experiment_meta"]["input_hashes"], "new-hashes")


if __name__ == "__main__":
    unittest.main()
