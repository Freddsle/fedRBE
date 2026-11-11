"""Check single-start fits and generated/omitted federated initialization settings."""

import contextlib
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import yaml

from evaluation_clusterization_after_correction.real_datasets import (
    figure4_analysis as analysis,
    multiple_runs as runner,
)
from evaluation_utils import featurecloud_kmeans_utils
from evaluation_utils.real_datasets_utils import run_central_kmeans, write_kmeans_config

APP_ROOT = Path(__file__).resolve().parents[1] / "federated_kmeans_upd"
sys.path.insert(0, str(APP_ROOT))
from kmeans import config as app_config
from kmeans.Clustering import Clustering
from kmeans.FCFederatedKMeans import FCFederatedKMeans


class InitializationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.matrix = pd.DataFrame(np.random.RandomState(2).normal(size=(3, 30)))
        self.data = StandardScaler().fit_transform(self.matrix.to_numpy().T)
        self.seed = 11
        pools = threadpool_limits(limits=1)
        self.addCleanup(pools.restore_original_limits)

    def test_central_and_analysis_default_fits_match_one_start(self):
        expected = KMeans(2, n_init=1, random_state=self.seed).fit(self.data)
        fit = run_central_kmeans(self.matrix, [2], self.seed)[2]
        np.testing.assert_array_equal(fit["labels"], expected.labels_)
        self.assertAlmostEqual(fit["inertia"], expected.inertia_)
        truth = np.repeat([0, 1], 15)
        row, labels = analysis.fit(self.matrix, truth, self.seed)
        np.testing.assert_array_equal(labels, expected.labels_)
        self.assertAlmostEqual(row["inertia"], expected.inertia_)

    def test_analysis_defaults_and_explicit_restart_counts(self):
        parser = analysis.parser()
        args = parser.parse_args(["subsample", "--output-dir", str(self.root)])
        self.assertEqual(args.n_init, 1)
        args = parser.parse_args(["starts", "--output-dir", str(self.root)])
        self.assertEqual(args.n_inits, [1])
        args = parser.parse_args(["starts", "--output-dir", str(self.root),
                                  "--n-inits", "1", "10"])
        self.assertEqual(args.n_inits, [1, 10])
        args = parser.parse_args(["arrival-order", "--output-dir", str(self.root)])
        self.assertEqual((args.n_init_local, args.n_init_global), (1, 1))

    def test_repeated_runner_emits_one_start_for_every_client(self):
        inputs = self.root / "inputs"
        clients = [inputs / name for name in ("A", "B")]
        for client in clients:
            client.mkdir(parents=True)
        metadata = pd.DataFrame({"file": ["a", "b"]})
        export = self.root / "export.tsv"
        pd.DataFrame({"file": ["a", "b"], "Fed_2clusters": [0, 1]}).to_csv(
            export, sep="\t", index=False)
        with patch.object(featurecloud_kmeans_utils, "run_single_federated_variant",
                          return_value=export):
            labels = runner.run_federated(self.root, "fixture", "before_multi", inputs,
                                          metadata, clients, 2, self.seed)
        self.assertEqual(labels.to_dict(), {"a": 0, "b": 1})
        for client in clients:
            algorithm = yaml.safe_load((client / "config_kmeans.yml").read_text())[
                "fc_kmeans"]["algorithm"]
            self.assertEqual((algorithm["n_init_local"], algorithm["n_init_global"]),
                             (1, 1))
            self.assertEqual(algorithm["seed"], self.seed)

    def test_app_omitted_restart_settings_and_explicit_overrides(self):
        source = self.root / "input"
        target = self.root / "output"
        source.mkdir()
        target.mkdir()
        path = source / "config_kmeans.yml"
        write_kmeans_config(path, [2], seed=self.seed)
        contents = yaml.safe_load(path.read_text())
        algorithm = contents["fc_kmeans"]["algorithm"]
        algorithm.pop("n_init_local")
        algorithm.pop("n_init_global")
        for explicit in (False, True):
            if explicit:
                algorithm.update(n_init_local=3, n_init_global=4)
                (target / path.name).unlink()
            path.write_text(yaml.safe_dump(contents))
            config = app_config.FCConfig()
            with patch.object(app_config, "INPUT_DIR", str(source)), \
                    patch.object(app_config, "OUTPUT_DIR", str(target)), \
                    contextlib.redirect_stdout(io.StringIO()):
                config.parse_configuration()
            expected = (3, 4) if explicit else (1, 1)
            self.assertEqual((config.n_init_local, config.n_init_global), expected)
        app = FCFederatedKMeans()
        self.assertEqual((app.n_init_local, app.n_init_global), (1, 1))

    def test_app_local_and_coordinator_default_fits_match_one_start(self):
        tabdata = SimpleNamespace(data=self.data.T, scaled=self.data.T)
        expected = KMeans(2, n_init=1, random_state=np.random.RandomState(self.seed))
        expected.fit(self.data)
        with contextlib.redirect_stdout(io.StringIO()):
            fit = Clustering.init_local_clustering(tabdata, 2, 2, self.seed)
        np.testing.assert_allclose(fit.centroids[2], expected.cluster_centers_)
        centroids = [expected.cluster_centers_, expected.cluster_centers_ + .2]
        expected_global = KMeans(2, n_init=1,
                                 random_state=np.random.RandomState(self.seed)).fit(
            np.concatenate(centroids))
        fit.aggregate_centroids({2: centroids})
        np.testing.assert_allclose(fit.centroids[2], expected_global.cluster_centers_)


if __name__ == "__main__":
    unittest.main()
