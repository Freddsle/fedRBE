"""Check invalid-input rejection and finite-observation privacy filtering."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import yaml

from batchcorrection.classes.client import Client
from batchcorrection.classes.coordinator_utils import create_feature_presence_matrix


class InputGuardTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.samples = [f"s{i}" for i in range(8)]
        self.expression = pd.DataFrame(
            np.arange(16, dtype=float).reshape(2, 8),
            index=["feature_a", "feature_b"],
            columns=self.samples,
        )
        self.design = pd.DataFrame(
            {"cov": [0, 1] * 4, "batch": ["first"] * 4 + ["second"] * 4},
            index=self.samples,
        )
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))

    def initialize(self, **overrides):
        self.expression.to_csv(self.root / "expression.tsv", sep="\t")
        self.design.to_csv(self.root / "design.tsv", sep="\t")
        config = {
            "data_filename": "expression.tsv",
            "design_filename": "design.tsv",
            "expression_file_flag": True,
            "separator": "\t",
            "covariates": ["cov"],
            "batch_col": "batch",
        }
        config.update(overrides)
        (self.root / "config.yml").write_text(
            yaml.safe_dump({"flimmaBatchCorrection": config}), encoding="utf-8"
        )
        client = Client()
        client.config_based_init(
            clientname="local", input_folder=str(self.root), use_hashing=False
        )
        return client

    def test_non_numeric_covariate_is_rejected_before_regression(self):
        self.design["cov"] = ["control", "treated"] * 4
        client = self.initialize()
        error = client.create_design(["local|first", "local|second"])
        self.assertRegex(error, "Covariate 'cov'.*numeric")

    def test_nonfinite_covariates_are_rejected_before_regression(self):
        for value in (np.nan, np.inf):
            with self.subTest(value=value):
                self.design["cov"] = [value, 1, 0, 1, 0, 1, 0, 1]
                client = self.initialize()
                error = client.create_design(["local|first", "local|second"])
                self.assertRegex(error, "Covariate 'cov'.*finite")

    def test_missing_batch_names_are_rejected(self):
        self.design.loc["s7", "batch"] = None
        with self.assertRaisesRegex(ValueError, "Batch.*missing"):
            self.initialize()

    def test_delimiter_in_batch_name_is_rejected(self):
        self.design["batch"] = ["x"] * 4 + ["x|1"] * 4
        with self.assertRaisesRegex(ValueError, r"Batch.*'\|'"):
            self.initialize()

    def test_spaces_in_batch_names_are_preserved(self):
        self.design["batch"] = ["Lab A  run"] * 4 + ["second"] * 4
        client = self.initialize()
        presence = client.get_batch_feature_presence_info(min_samples=5)
        create_feature_presence_matrix(
            [("local", 0, False, presence)], client.feature_names, ["local"]
        )

    def test_smpc_default_and_explicit_false_are_preserved(self):
        self.assertIs(self.initialize().smpc, True)
        self.assertIs(self.initialize(smpc=False).smpc, False)

    def test_nonboolean_smpc_is_rejected(self):
        for flag in (None, "false", 0):
            with self.subTest(flag=flag):
                with self.assertRaisesRegex(ValueError, "smpc.*boolean"):
                    self.initialize(smpc=flag)


class FiniteObservationTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))

    def client(self, values, batches=None):
        client = Client()
        client.client_name = "local"
        client.data = pd.DataFrame.from_dict(values, orient="index")
        client.feature_names = list(client.data.index)
        client.hash2feature = {}
        client.design = pd.DataFrame(index=client.data.columns)
        if batches is None:
            client.batch_col = None
            client.batch_labels = ["local"]
        else:
            client.batch_col = "batch"
            client.design["batch"] = batches
            client.batch_labels = [f"local|{batch}" for batch in dict.fromkeys(batches)]
        return client

    def test_one_finite_observation_is_suppressed_despite_infinite_values(self):
        client = self.client(
            {"feature": [1, np.inf, -np.inf, np.nan, 2, 3, 4, 5, 6, 7]},
            batches=["first"] * 4 + ["second"] * 6,
        )
        presence = client.get_batch_feature_presence_info(min_samples=5)
        self.assertEqual(presence, {"local|first": [], "local|second": ["feature"]})
        self.assertTrue(client.data.iloc[0, :4].isna().all())

    def test_all_infinite_batch_is_absent(self):
        client = self.client(
            {"feature": [np.inf, -np.inf, np.inf, -np.inf, 2, 3, 4, 5, 6, 7]},
            batches=["first"] * 4 + ["second"] * 6,
        )
        presence = client.get_batch_feature_presence_info(min_samples=5)
        self.assertEqual(presence, {"local|first": [], "local|second": ["feature"]})

    def test_client_threshold_counts_only_finite_observations(self):
        client = self.client({
            "too_few": [1, 2, 3, 4, np.inf, np.inf, -np.inf, -np.inf],
            "enough": [1, 2, 3, 4, 5, np.inf, -np.inf, np.nan],
        })
        presence = client.get_batch_feature_presence_info(min_samples=5)
        self.assertEqual(presence, {"local": ["enough"]})
        self.assertTrue(client.data.loc["too_few"].isna().all())


if __name__ == "__main__":
    unittest.main()
