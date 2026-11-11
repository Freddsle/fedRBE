"""The training level of a fedRBE run, min_clients, and aligning new batches to the level."""

import contextlib
import hashlib
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import yaml

from batchcorrection import align_new_batch
from batchcorrection.classes.client import Client, align_to_training_level
from batchcorrection.classes.coordinator_utils import compute_beta, create_beta_mask
from batchcorrection.fedrbe import TRAINING_LEVEL_FILE, TRAINING_LEVEL_INFO_FILE, run_fedrbe
from evaluation_classification_after_correction.fc_fed_forest_simple_app.fc_fedlearnsim_helper.run_app_simulation import (
    run_simulation_native,
)


def feature_hash(name):
    return hashlib.sha3_256(name.encode()).hexdigest()


def write_client(folder, values, samples, design, **settings):
    """Writes a fedRBE client folder: data.tsv, design.tsv and config.yml."""
    folder.mkdir(parents=True, exist_ok=True)
    values.index.name = "rowname"
    values.columns = samples
    values.to_csv(folder / "data.tsv", sep="\t")
    design.index = pd.Index(samples, name="file")
    design.to_csv(folder / "design.tsv", sep="\t")
    config = {"data_filename": "data.tsv", "design_filename": "design.tsv",
              "expression_file_flag": True, "index_col": "rowname", "covariates": [],
              "separator": "\t", "smpc": False, "min_samples": 0}
    config.update(settings)
    (folder / "config.yml").write_text(yaml.safe_dump({"flimmaBatchCorrection": config}))


def statistics(design, values):
    """XtX and XtY of one feature per row of values over the finite values."""
    xtx = np.zeros((len(values), design.shape[1], design.shape[1]))
    xty = np.zeros((len(values), design.shape[1]))
    for feature, y in enumerate(values):
        finite = np.isfinite(y)
        xtx[feature] = design[finite].T @ design[finite]
        xty[feature] = design[finite].T @ y[finite]
    return xtx, xty


class TrainingLevelFormulaTests(unittest.TestCase):
    """The level equals the mean of the corrected values, computed from XtX and beta only."""

    def fit(self, with_covariate, missing_batch=False):
        rng = np.random.default_rng(3)
        sizes = [4, 9, 20]  # unequal batch sizes
        rows, values = [], []
        for batch, size in enumerate(sizes):
            label = (rng.random(size) < 0.3 + 0.2 * batch).astype(float)
            effect = [[1, 0], [0, 1], [-1, -1]][batch]
            rows += [[1, *([lab] if with_covariate else []), *effect] for lab in label]
            values += list(3 * batch + 2 * label + rng.normal(size=size))
        design = np.array(rows, dtype=float)
        # two features, the second one is missing in the first batch
        data = np.array([values, values])
        if missing_batch:
            data[1, :sizes[0]] = np.nan
        presence = np.array([[1, 1, 1], [0 if missing_batch else 1, 1, 1]])
        xtx, xty = statistics(design, data)
        k = design.shape[1]
        mask = create_beta_mask(presence, 2, k)
        with contextlib.redirect_stdout(io.StringIO()):
            beta, level = compute_beta([[xtx, xty]], 2, k, mask, n_batch_columns=2,
                                       with_training_level=True)
        corrected = data - beta[:, -2:] @ design[:, -2:].T
        return beta, level, np.nanmean(corrected, axis=1), xty[:, 0] / xtx[:, 0, 0]

    def test_level_is_the_mean_of_the_corrected_values(self):
        for with_covariate in (False, True):
            for missing_batch in (False, True):
                with self.subTest(with_covariate=with_covariate, missing_batch=missing_batch):
                    beta, level, corrected_mean, raw_mean = self.fit(with_covariate,
                                                                     missing_batch)
                    np.testing.assert_allclose(level, corrected_mean, rtol=0, atol=1e-10)
                    # unequal batch sizes: not the raw mean
                    self.assertGreater(abs(level[0] - raw_mean[0]), 0.1)
                    if with_covariate:
                        self.assertGreater(abs(level[0] - beta[0, 0]), 0.1)
                    else:
                        np.testing.assert_allclose(level, beta[:, 0], rtol=0, atol=1e-10)

    def test_without_flag_only_beta_is_returned(self):
        design = np.array([[1, 1], [1, 1], [1, -1], [1, -1]], dtype=float)
        xtx, xty = statistics(design, np.array([[1., 2., 5., 6.]]))
        with contextlib.redirect_stdout(io.StringIO()):
            beta = compute_beta([[xtx, xty]], 1, 2, np.zeros((1, 2), dtype=bool),
                                n_batch_columns=1)
        np.testing.assert_allclose(beta, [[3.5, -2.0]])


class FedrbeRunTests(unittest.TestCase):
    """The coordinator of a fedRBE run writes the level; min_clients is configurable."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.rng = np.random.RandomState(7)
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))

    def make_clients(self, names, **settings):
        """Clients with different sizes; client_b holds two batches and misses feature_b in y."""
        for number, name in enumerate(names):
            batches = ["x", "y"] if name == "client_b" else ["x"]
            size = 6 + 3 * number
            samples = [f"{name}_{batch}_{i}" for batch in batches for i in range(size)]
            condition = (np.arange(len(samples)) % 3 == 0).astype(int)
            shift = np.repeat(np.arange(len(batches)), size) * 1.5 + 2 * number
            values = pd.DataFrame(
                self.rng.normal(size=(2, len(samples))) + condition + shift,
                index=["feature_a", "feature_b"])
            if name == "client_b":
                values.iloc[1, size:] = np.nan
            design = pd.DataFrame({"condition": condition,
                                   "batch": np.repeat(batches, size)})
            write_client(self.root / name, values, samples, design,
                         batch_col="batch" if len(batches) > 1 else None, **settings)
        return names

    def run_fedrbe(self, names, tag=""):
        outputs = [self.root / f"output{tag}" / name for name in names]
        for output in outputs:
            output.mkdir(parents=True)
        run_simulation_native([str(self.root / name) for name in names],
                              [str(output) for output in outputs], None, run_fedrbe,
                              client_ids=list(names))
        return outputs

    def test_level_is_written_by_the_coordinator_and_matches_the_corrected_data(self):
        names = ["client_a", "client_b", "client_c"]
        for covariates in ([], ["condition"]):
            with self.subTest(covariates=covariates):
                self.make_clients(names, covariates=covariates)
                outputs = self.run_fedrbe(names, tag=str(len(covariates)))
                self.assertFalse(any((output / TRAINING_LEVEL_FILE).exists()
                                     for output in outputs[1:]))
                level, info = align_new_batch.read_training_level(str(outputs[0]))
                self.assertEqual(info, {"normalizationMethod": None,
                                        "covariates_used": bool(covariates),
                                        "n_batches": 4, "separator": "\t"})
                corrected = pd.concat(
                    [pd.read_csv(output / "only_batch_corrected_data.csv", sep="\t",
                                 index_col=0) for output in outputs], axis=1)
                expected = corrected.mean(axis=1, skipna=True)
                expected.index = [feature_hash(name) for name in expected.index]
                pd.testing.assert_series_equal(level.sort_index(), expected.sort_index(),
                                               check_names=False, rtol=0, atol=1e-10)

    def test_two_clients_need_min_clients_two(self):
        names = self.make_clients(["client_a", "client_c"])
        with self.assertRaisesRegex(ValueError, "at least 3 clients"):
            self.run_fedrbe(names, tag="default")
        names = self.make_clients(["client_a", "client_c"], min_clients=2)
        outputs = self.run_fedrbe(names, tag="two")
        self.assertTrue((outputs[0] / TRAINING_LEVEL_FILE).exists())

    def test_invalid_min_clients_is_rejected(self):
        for value in (1, True, "3", 2.5):
            with self.subTest(value=value):
                self.make_clients(["client_a"], min_clients=value)
                with self.assertRaisesRegex(ValueError, "min_clients"):
                    Client().config_based_init(clientname="client_a",
                                               input_folder=str(self.root / "client_a"))


class AlignmentTests(unittest.TestCase):
    """align_to_training_level and the align_new_batch CLI."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))
        self.features = ["feature_a", "feature_b", "feature_c"]
        self.level = pd.Series([10.0, 20.0], index=[feature_hash("feature_a"),
                                                    feature_hash("feature_b")])

    def new_batch(self, batches=("x",), min_samples=0, **settings):
        """A new batch with known offsets per batch; feature_c has no training level."""
        samples, values, batch_labels = [], [], []
        for number, batch in enumerate(batches):
            for i in range(6):
                samples.append(f"{batch}_{i}")
                batch_labels.append(batch)
                values.append([13 + 4 * number + (i - 2.5), 25 + i, 1.0 + i])
        data = pd.DataFrame(np.array(values).T, index=self.features)
        design = pd.DataFrame({"batch": batch_labels})
        folder = self.root / "new"
        write_client(folder, data, samples, design, min_samples=min_samples,
                     batch_col="batch" if len(batches) > 1 else None, **settings)
        return folder

    def load(self, folder):
        client = Client()
        client.config_based_init(clientname="new", input_folder=str(folder))
        return client

    def test_each_batch_is_shifted_onto_the_level(self):
        for batches in (("x",), ("x", "y")):
            with self.subTest(batches=batches):
                client = self.load(self.new_batch(batches))
                aligned, counts = align_to_training_level(client, self.level)
                self.assertEqual(list(counts.values()), [2] * len(batches))
                for batch in batches:
                    samples = [s for s in aligned.columns if s.startswith(batch)]
                    means = aligned.loc[["feature_a", "feature_b"], samples].mean(axis=1)
                    np.testing.assert_allclose(means, [10, 20])
                    # the spread within the batch is unchanged
                    np.testing.assert_allclose(aligned.loc["feature_a", samples].diff().dropna(),
                                               1.0)
                # no training level: left as loaded
                np.testing.assert_allclose(aligned.loc["feature_c"],
                                           list(np.arange(6) + 1.0) * len(batches))

    def test_sparse_features_stay_as_loaded(self):
        folder = self.new_batch(min_samples=5)
        data = pd.read_csv(folder / "data.tsv", sep="\t", index_col=0)
        data.iloc[1, :2] = np.nan  # feature_b has 4 < 5 values
        data.to_csv(folder / "data.tsv", sep="\t")
        aligned, counts = align_to_training_level(self.load(folder), self.level)
        self.assertEqual(counts, {"new": 1})
        np.testing.assert_allclose(aligned.loc["feature_b"].dropna(), [27, 28, 29, 30])

    def test_no_shared_feature_fails(self):
        client = self.load(self.new_batch())
        with self.assertRaisesRegex(ValueError, "none of its features"):
            align_to_training_level(client, pd.Series([1.0], index=["other"]))

    def write_run_output(self, normalization=None, covariates_used=True):
        output = self.root / "fedrbe_output"
        output.mkdir(exist_ok=True)
        self.level.rename_axis("feature_hash").rename("training_level").to_csv(
            output / TRAINING_LEVEL_FILE, sep="\t")
        (output / TRAINING_LEVEL_INFO_FILE).write_text(yaml.safe_dump({
            "normalizationMethod": normalization, "covariates_used": covariates_used,
            "n_batches": 3, "separator": "\t"}))
        return output

    def test_cli_writes_aligned_data_and_report_without_labels(self):
        # the covariate is not in the design: it must be ignored, not loaded
        folder = self.new_batch(("x", "y"), covariates=["condition"])
        output = self.root / "aligned"
        align_new_batch.main(["--fedrbe-output", str(self.write_run_output()),
                              "--input", str(folder), "--output", str(output)])
        aligned = pd.read_csv(output / align_new_batch.ALIGNED_DATA_FILE, sep="\t",
                              index_col=0)
        self.assertEqual(list(aligned.index), self.features)
        np.testing.assert_allclose(aligned.loc[["feature_a", "feature_b"]].mean(axis=1),
                                   [10, 20])
        report = (output / align_new_batch.REPORT_FILE).read_text()
        self.assertIn("Batch new|x: 2 features aligned, 1 left as loaded", report)
        self.assertIn("used covariates", report)

    def test_cli_uses_the_normalization_of_the_run(self):
        folder = self.new_batch()
        run_output = self.write_run_output(normalization="log2(x+1)")
        aligned = align_new_batch.align_new_batch(str(run_output), str(folder),
                                                  str(self.root / "a"))
        # feature_c is not aligned, so it shows the normalization of the run
        np.testing.assert_allclose(aligned.loc["feature_c"], np.log2(np.arange(6) + 2.0))
        conflicting = self.new_batch(normalizationMethod=None)
        with self.assertRaisesRegex(ValueError, "normalizationMethod"):
            align_new_batch.align_new_batch(str(run_output), str(conflicting),
                                            str(self.root / "b"))


if __name__ == "__main__":
    unittest.main()
