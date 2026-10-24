"""Regression checks for explicit empty correction covariates with hashing."""

import contextlib
import hashlib
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import yaml

from batchcorrection.classes.client import Client


class CovariateInitializationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.samples = [f"sample_{i}" for i in range(8)]
        self.expression = pd.DataFrame(
            np.arange(16, dtype=float).reshape(2, 8),
            index=["feature_a", "feature_b"],
            columns=self.samples,
        )
        self.expression.to_csv(self.root / "expression.tsv", sep="\t")
        self.design = pd.DataFrame(
            {"target": [0, 1] * 4, "batch": ["first"] * 4 + ["second"] * 4},
            index=self.samples,
        )
        self.design.to_csv(self.root / "design.tsv", sep="\t")

    def initialize(self, covariates):
        config = {
            "covariates": covariates,
            "data_filename": "expression.tsv",
            "design_filename": "design.tsv",
            "batch_col": "batch",
            "expression_file_flag": True,
            "separator": "\t",
        }
        (self.root / "config.yml").write_text(
            yaml.safe_dump({"flimmaBatchCorrection": config}), encoding="utf-8"
        )
        client = Client()
        with contextlib.redirect_stdout(io.StringIO()):
            client.config_based_init(
                clientname="local", input_folder=str(self.root), use_hashing=True
            )
        return client

    def test_empty_covariates_hash_features_and_create_batch_only_design(self):
        client = self.initialize([])
        expected_hashes = {
            name: hashlib.sha3_256(name.encode()).hexdigest()
            for name in self.expression.index
        }
        self.assertEqual(client.hash2variable, {})
        self.assertEqual(client.variables, [])
        self.assertEqual(client.feature2hash, expected_hashes)
        self.assertEqual(client.feature_names, list(expected_hashes.values()))
        self.assertEqual(list(client.design.columns), ["batch"])
        with contextlib.redirect_stdout(io.StringIO()):
            client.validate_inputs([])
            error = client.create_design(["local|first", "local|second"])
        self.assertIsNone(error)
        self.assertEqual(list(client.design.columns), ["intercept", "local|first"])
        self.assertEqual(list(client.design.index), self.samples)
        np.testing.assert_array_equal(client.design["intercept"], np.ones(8))
        np.testing.assert_array_equal(client.design["local|first"], [1] * 4 + [-1] * 4)
        np.testing.assert_array_equal(client.data.to_numpy(), self.expression.to_numpy())


if __name__ == "__main__":
    unittest.main()
