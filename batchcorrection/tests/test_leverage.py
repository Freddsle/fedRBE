"""Check the local high-leverage exclusion of values from the fit."""

import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import yaml

from batchcorrection.classes.client import Client, _leverage

REPO = Path(__file__).resolve().parents[2]


def hat_diagonal(design: np.ndarray) -> np.ndarray:
    """Diagonal of the hat matrix, via the pseudo-inverse like limma's QR fit."""
    return np.einsum("ij,ji->i", design, np.linalg.pinv(design))


def batch_indicators(batches: np.ndarray) -> np.ndarray:
    return (batches[:, None] == pd.unique(batches)[None, :]).astype(float)


def make_client(values, covariate, batches, max_leverage=0.5):
    """A client with an in-memory design, as config_based_init would set it up."""
    client = Client()
    client.client_name = "local"
    samples = [f"s{i}" for i in range(len(covariate))]
    client.data = pd.DataFrame(np.atleast_2d(np.asarray(values, dtype=float)), columns=samples,
                               index=[f"feature{i}" for i in range(np.atleast_2d(values).shape[0])])
    client.feature_names = list(client.data.index)
    client.hash2feature = {name: name for name in client.feature_names}
    client.batch_col = "batch"
    client.design = pd.DataFrame({"batch": batches, "cov": covariate}, index=samples)
    client.batch_labels = [f"local|{batch}" for batch in dict.fromkeys(batches)]
    client.variables = ["cov"]
    client.max_leverage = max_leverage
    return client


class LeverageFormulaTests(unittest.TestCase):
    def test_leverage_matches_the_hat_matrix_of_the_local_fit(self):
        rng = np.random.default_rng(1)
        batches = np.repeat(["a", "b"], 10)
        X = rng.normal(size=(20, 2))
        fitted = rng.random((6, 20)) > 0.25
        groups = [np.flatnonzero(batches == b) for b in ("a", "b")]
        h = _leverage(X, groups, fitted)
        design = np.hstack([batch_indicators(batches), X])
        for f in range(6):
            rows = fitted[f]
            np.testing.assert_allclose(h[f, rows], hat_diagonal(design[rows]), atol=1e-10)
            self.assertTrue((h[f, ~rows] == 0).all())

    def test_local_leverage_bounds_the_leverage_over_all_clients(self):
        rng = np.random.default_rng(2)
        sizes = [12, 15, 9]
        batches = np.concatenate([[f"c{c}|b{i % 2}" for i in range(n)] for c, n in enumerate(sizes)])
        X = rng.normal(size=(sum(sizes), 2))
        X[3] = [6, -5]  # an outlier in the first client
        global_h = hat_diagonal(np.hstack([batch_indicators(batches), X]))
        start = 0
        for n in sizes:
            rows = slice(start, start + n)
            local_batches = batches[rows]
            groups = [np.flatnonzero(local_batches == b) for b in pd.unique(local_batches)]
            local_h = _leverage(X[rows], groups, np.ones((1, n), bool))[0]
            self.assertTrue((local_h >= global_h[rows] - 1e-12).all())
            start += n

    def test_constant_covariate_gives_the_batch_share(self):
        groups = [np.arange(4), np.arange(4, 10)]
        h = _leverage(np.full((10, 1), 3.7), groups, np.ones((1, 10), bool))[0]
        np.testing.assert_allclose(h, [1 / 4] * 4 + [1 / 6] * 6, atol=1e-12)


class ExclusionTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))

    def test_covariate_outlier_is_excluded_and_restored_for_the_correction(self):
        covariate = list(range(7)) + [100] + list(range(8))
        values = np.arange(16.0)
        client = make_client(values, covariate, ["a"] * 8 + ["b"] * 8)
        self.assertEqual(client.exclude_high_leverage(["cov"], 5), 1)
        self.assertTrue(np.isnan(client.data.iloc[0, 7]))
        self.assertEqual(int(client.data.notna().sum().sum()), 15)
        client.restore_excluded_values()
        np.testing.assert_array_equal(client.data.iloc[0].to_numpy(), values)
        self.assertIn("1 values were left out", client.leverage_report)

    def test_balanced_design_keeps_every_value(self):
        client = make_client(np.arange(16.0), [0, 1] * 8, ["a"] * 8 + ["b"] * 8)
        self.assertEqual(client.exclude_high_leverage(["cov"], 5), 0)
        self.assertEqual(int(client.data.notna().sum().sum()), 16)

    def test_single_sample_with_a_binary_level_is_excluded(self):
        # With known covariates, XtY reveals this sample's value exactly (h = 1).
        client = make_client(np.arange(16.0), [0] * 7 + [1] + [0] * 8, ["a"] * 8 + ["b"] * 8)
        self.assertEqual(client.exclude_high_leverage(["cov"], 5), 1)
        self.assertTrue(np.isnan(client.data.iloc[0, 7]))

    def test_max_leverage_one_and_no_covariates_disable_the_check(self):
        covariate = [0] * 7 + [1] + [0] * 8
        for max_leverage, variables in ((1.0, ["cov"]), (0.5, [])):
            with self.subTest(max_leverage=max_leverage, variables=variables):
                client = make_client(np.arange(16.0), covariate, ["a"] * 8 + ["b"] * 8, max_leverage)
                self.assertEqual(client.exclude_high_leverage(variables, 5), 0)
                self.assertEqual(int(client.data.notna().sum().sum()), 16)

    def test_batch_below_min_samples_after_exclusion_is_dropped(self):
        values = [1, 2, 3, 4, 5] + list(range(10))
        covariate = [0, 1, 2, 3, 100] + list(range(10))
        client = make_client(values, covariate, ["small"] * 5 + ["large"] * 10)
        client.exclude_high_leverage(["cov"], 5)
        self.assertTrue(client.data.iloc[0, :5].isna().all())
        self.assertTrue(client.data.iloc[0, 5:].notna().all())
        presence = client.get_batch_feature_presence_info(5)
        self.assertEqual(presence, {"local|small": [], "local|large": ["feature0"]})
        client.restore_excluded_values()
        self.assertTrue(client.data.iloc[0, :5].isna().all())


class ConfigTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        samples = [f"s{i}" for i in range(8)]
        pd.DataFrame(np.ones((2, 8)), index=["a", "b"], columns=samples).to_csv(
            self.root / "expression.tsv", sep="\t")
        pd.DataFrame({"cov": [0, 1] * 4}, index=samples).to_csv(self.root / "design.tsv", sep="\t")
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))

    def initialize(self, **overrides):
        config = {"data_filename": "expression.tsv", "design_filename": "design.tsv",
                  "expression_file_flag": True, "separator": "\t", "covariates": ["cov"]}
        config.update(overrides)
        (self.root / "config.yml").write_text(yaml.safe_dump({"flimmaBatchCorrection": config}))
        client = Client()
        client.config_based_init(clientname="local", input_folder=str(self.root), use_hashing=False)
        return client

    def test_default_and_valid_values(self):
        self.assertEqual(self.initialize().max_leverage, 0.5)
        for value in (0.25, 1, 1.0):
            self.assertEqual(self.initialize(max_leverage=value).max_leverage, float(value))

    def test_invalid_values_are_rejected(self):
        for value in (0, -0.5, 1.5, True, "0.5", None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "max_leverage"):
                self.initialize(max_leverage=value)


class FederatedRunTests(unittest.TestCase):
    """run_fedrbe through the local simulator equals a central fit without the excluded value."""

    def setUp(self):
        sys.path.insert(0, str(REPO / "evaluation_classification_after_correction"))
        from fc_fed_forest_simple_app.fc_fedlearnsim_helper.run_app_simulation import \
            run_simulation_native
        from batchcorrection.fedrbe import run_fedrbe
        self.run_simulation_native, self.run_fedrbe = run_simulation_native, run_fedrbe
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        rng = np.random.default_rng(5)
        self.covariates, self.data = [], []
        for c in range(3):
            covariate = np.linspace(0, 10, 12)
            if c == 1:
                covariate[-1] = 100  # local leverage close to 1
            values = rng.normal(size=(4, 12)) + 2 * c + 0.1 * covariate
            values[0, c] = np.nan
            self.covariates.append(covariate)
            self.data.append(values)

    def run_clients(self, max_leverage):
        names = [f"client{c}" for c in range(3)]
        inputs, outputs = [], []
        for c, name in enumerate(names):
            folder, output = self.root / max_leverage / name, self.root / max_leverage / f"{name}_out"
            folder.mkdir(parents=True)
            output.mkdir()
            samples = [f"{name}_s{i}" for i in range(12)]
            pd.DataFrame(self.data[c], index=[f"f{i}" for i in range(4)], columns=samples).to_csv(
                folder / "expression.tsv", sep="\t")
            pd.DataFrame({"cov": self.covariates[c]}, index=samples).to_csv(folder / "design.tsv", sep="\t")
            config = {"data_filename": "expression.tsv", "design_filename": "design.tsv",
                      "expression_file_flag": True, "separator": "\t", "covariates": ["cov"],
                      "smpc": False, "max_leverage": float(max_leverage)}
            (folder / "config.yml").write_text(yaml.safe_dump({"flimmaBatchCorrection": config}))
            inputs.append(str(folder))
            outputs.append(str(output))
        with contextlib.redirect_stdout(io.StringIO()):
            self.run_simulation_native(inputs, outputs, None, self.run_fedrbe, client_ids=names)
        # the output keeps the global feature order, which follows the hashed names
        return np.hstack([pd.read_csv(os.path.join(output, "only_batch_corrected_data.csv"),
                                      sep="\t", index_col=0).loc[[f"f{i}" for i in range(4)]].to_numpy()
                          for output in outputs])

    def central(self, fit):
        """Fit on the values in fit, correct all values (limma with weights 0)."""
        Y = np.hstack(self.data)
        effect = np.zeros((36, 2))  # effect coding, the last client is the reference
        effect[:12, 0], effect[12:24, 1], effect[24:] = 1, 1, -1
        X = np.column_stack([np.ones(36), np.concatenate(self.covariates), effect])
        corrected = np.empty_like(Y)
        for f in range(4):
            rows = fit[f] & np.isfinite(Y[f])
            beta = np.linalg.lstsq(X[rows], Y[f, rows], rcond=None)[0]
            corrected[f] = Y[f] - effect @ beta[2:]
        return corrected

    def test_excluded_value_is_corrected_but_left_out_of_the_fit(self):
        fit = np.ones((4, 36), bool)
        fit[:, 23] = False  # the outlier of the second client
        result = self.run_clients("0.5")
        np.testing.assert_allclose(result, self.central(fit), atol=1e-10)
        self.assertTrue(np.isfinite(result[:, 23]).all())
        # without the check the outlier enters the fit
        np.testing.assert_allclose(self.run_clients("1"), self.central(np.ones((4, 36), bool)),
                                   atol=1e-10)


if __name__ == "__main__":
    unittest.main()
