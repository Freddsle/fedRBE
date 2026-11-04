"""Check the local high-leverage exclusion of values from the fit."""

import contextlib
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import yaml

from batchcorrection.classes.client import Client, _hat_values

REPO = Path(__file__).resolve().parents[2]


def local_design(batches, covariates) -> np.ndarray:
    """One intercept per batch plus the covariates."""
    return np.column_stack([pd.get_dummies(pd.Series(batches)).to_numpy(dtype=float), covariates])


def make_client(values, covariate, batches, max_leverage=0.5):
    """A client with an in-memory design, as config_based_init would set it up."""
    client = Client()
    client.client_name = "local"
    samples = [f"s{i}" for i in range(len(covariate))]
    values = np.atleast_2d(np.asarray(values, dtype=float))
    client.data = pd.DataFrame(values, columns=samples,
                               index=[f"feature{i}" for i in range(len(values))])
    client.feature_names = list(client.data.index)
    client.hash2feature = {name: name for name in client.feature_names}
    client.batch_col = "batch"
    client.design = pd.DataFrame({"batch": batches, "cov": covariate}, index=samples)
    client.batch_labels = [f"local|{batch}" for batch in dict.fromkeys(batches)]
    client.variables = ["cov"]
    client.max_leverage = max_leverage
    return client


class HatValueTests(unittest.TestCase):
    def test_hat_values_are_the_diagonal_of_the_hat_matrix(self):
        rng = np.random.default_rng(1)
        design = local_design(np.repeat(["a", "b"], 10), rng.normal(size=(20, 2)))
        hat = design @ np.linalg.inv(design.T @ design) @ design.T
        np.testing.assert_allclose(_hat_values(design), np.diag(hat), atol=1e-12)

    def test_aliased_covariate_is_dropped(self):
        # constant within each batch, so the covariate adds nothing to the batch intercepts
        design = local_design(["a"] * 4 + ["b"] * 6, [[3.7]] * 4 + [[1.2]] * 6)
        np.testing.assert_allclose(_hat_values(design), [1 / 4] * 4 + [1 / 6] * 6, atol=1e-12)

    def test_local_leverage_bounds_the_leverage_over_all_clients(self):
        rng = np.random.default_rng(2)
        sizes = [12, 15, 9]
        batches = np.concatenate([[f"c{c}|b{i % 2}" for i in range(n)] for c, n in enumerate(sizes)])
        covariates = rng.normal(size=(sum(sizes), 2))
        covariates[3] = [6, -5]  # an outlier in the first client
        global_h = _hat_values(local_design(batches, covariates))
        start = 0
        for n in sizes:
            rows = slice(start, start + n)
            local_h = _hat_values(local_design(batches[rows], covariates[rows]))
            self.assertTrue((local_h >= global_h[rows] - 1e-12).all())
            start += n


class ExclusionTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))

    def test_covariate_outlier_is_excluded_but_kept_for_the_correction(self):
        covariate = list(range(7)) + [100] + list(range(8))
        client = make_client(np.arange(16.0), covariate, ["a"] * 8 + ["b"] * 8)
        self.assertEqual(client.exclude_high_leverage(["cov"], 5), 1)
        self.assertTrue(np.isnan(client.data.iloc[0, 7]))
        self.assertEqual(int(client.data.notna().sum().sum()), 15)
        self.assertEqual(client.leverage_excluded.iloc[0, 7], 7.0)
        self.assertIn("1 values of 1 features", client.leverage_report)

    def test_balanced_design_keeps_every_value(self):
        client = make_client(np.arange(16.0), [0, 1] * 8, ["a"] * 8 + ["b"] * 8)
        self.assertEqual(client.exclude_high_leverage(["cov"], 5), 0)
        self.assertEqual(int(client.data.notna().sum().sum()), 16)

    def test_single_sample_with_a_binary_level_is_excluded(self):
        # With known covariates, XtY reveals this sample's value exactly (h = 1).
        client = make_client(np.arange(16.0), [0] * 7 + [1] + [0] * 8, ["a"] * 8 + ["b"] * 8)
        self.assertEqual(client.exclude_high_leverage(["cov"], 5), 1)
        self.assertTrue(np.isnan(client.data.iloc[0, 7]))

    def test_pair_with_leverage_exactly_at_the_threshold_is_kept(self):
        # only batch a varies, its two samples with cov = 1 have h = 1/8 + 0.5625/1.5 = 0.5
        batches, covariate = ["a"] * 8 + ["b"] * 8, [1, 1, 0, 0, 0, 0, 0, 0] + [0] * 8
        h = _hat_values(local_design(batches, np.array(covariate, float)))
        self.assertAlmostEqual(h.max(), 0.5, places=12)
        client = make_client(np.arange(16.0), covariate, batches)
        self.assertEqual(client.exclude_high_leverage(["cov"], 5), 0)

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
        # the excluded value of the dropped batch is not corrected either
        self.assertTrue(client.leverage_excluded.isna().all().all())
        presence = client.get_batch_feature_presence_info(5)
        self.assertEqual(presence, {"local|small": [], "local|large": ["feature0"]})


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
    """
    run_fedrbe through the local simulator equals limma with the R mirror of
    the leverage check in central_correction.R.
    Client 1 has a covariate outlier; for feature f1 its batch has exactly
    min_samples = 4 clients + 1 covariate + 1 = 6 values, so excluding the
    outlier masks that batch.
    """
    N_CLIENTS, N_SAMPLES = 4, 12
    OUTLIER = 12 + 11  # last sample of client 1

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
        self.names = [f"client{c}" for c in range(self.N_CLIENTS)]
        self.covariates, self.data = [], []
        for c in range(self.N_CLIENTS):
            covariate = np.linspace(0, 10, self.N_SAMPLES)
            if c == 1:
                covariate[-1] = 100  # local leverage close to 1
            values = rng.normal(size=(4, self.N_SAMPLES)) + 2 * c + 0.1 * covariate
            values[0, c] = np.nan
            if c == 1:
                values[1, :6] = np.nan  # 6 values left, including the outlier
            self.covariates.append(covariate)
            self.data.append(values)
        self.samples = [f"{name}_s{i}" for name in self.names for i in range(self.N_SAMPLES)]

    def run_clients(self, max_leverage):
        inputs, outputs = [], []
        for c, name in enumerate(self.names):
            folder, output = self.root / max_leverage / name, self.root / max_leverage / f"{name}_out"
            folder.mkdir(parents=True)
            output.mkdir()
            samples = self.samples[c * self.N_SAMPLES:(c + 1) * self.N_SAMPLES]
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
            self.run_simulation_native(inputs, outputs, None, self.run_fedrbe, client_ids=self.names)
        # the output keeps the global feature order, which follows the hashed names
        return np.hstack([pd.read_csv(os.path.join(output, "only_batch_corrected_data.csv"),
                                      sep="\t", index_col=0).loc[[f"f{i}" for i in range(4)]].to_numpy()
                          for output in outputs])

    def central_r(self, max_leverage):
        """The central reference through central_correction.R and its leverage mirror."""
        rscript = REPO / ".pixi/envs/default/bin/Rscript"
        if not rscript.is_file():
            rscript = Path(shutil.which("Rscript") or "/unavailable")
        if not rscript.is_file():
            self.skipTest("Rscript is unavailable")
        folder = self.root / f"central_{max_leverage}"
        folder.mkdir()
        pd.DataFrame(np.hstack(self.data), index=[f"f{i}" for i in range(4)],
                     columns=self.samples).rename_axis("rowname").to_csv(folder / "data.tsv", sep="\t")
        pd.DataFrame({"intercept": 1.0, "cov": np.concatenate(self.covariates)},
                     index=pd.Index(self.samples, name="file")).to_csv(folder / "design.tsv", sep="\t")
        clients = np.repeat(self.names, self.N_SAMPLES)
        pd.DataFrame({"file": self.samples, "batch": clients,
                      "batch_order": np.repeat(np.arange(1, self.N_CLIENTS + 1), self.N_SAMPLES),
                      "client": clients, "min_samples": self.N_CLIENTS + 2,
                      "max_leverage": float(max_leverage)}).to_csv(folder / "batch.tsv", sep="\t", index=False)
        completed = subprocess.run(
            [str(rscript), str(REPO / "evaluation_utils/central_correction.R"), str(folder / "data.tsv"),
             str(folder / "design.tsv"), str(folder / "batch.tsv"), str(folder / "out.tsv")],
            capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return pd.read_csv(folder / "out.tsv", sep="\t", index_col=0)[self.samples].to_numpy()

    def test_excluded_value_is_corrected_but_left_out_of_the_fit(self):
        result = self.run_clients("0.5")
        np.testing.assert_allclose(result, self.central_r("0.5"), atol=1e-10)
        self.assertTrue(np.isfinite(result[[0, 2, 3], self.OUTLIER]).all())
        self.assertTrue(np.isnan(result[1, 12:24]).all())  # f1 of client 1 dropped
        # without the check, the outlier enters the fit
        without = self.run_clients("1")
        np.testing.assert_allclose(without, self.central_r("1"), atol=1e-10)
        self.assertGreater(np.nanmax(np.abs(result - without)), 0.01)


if __name__ == "__main__":
    unittest.main()
