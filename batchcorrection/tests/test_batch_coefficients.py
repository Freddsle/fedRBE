"""The coordinator's batch coefficients reproduce every client's correction."""

import contextlib
import hashlib
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import yaml

from batchcorrection.classes.client import Client, subtract_batch_effect
from batchcorrection.fedrbe import BATCH_COEFFICIENTS_FILE, run_fedrbe
from evaluation_classification_after_correction.fc_fed_forest_simple_app.fc_fedlearnsim_helper.run_app_simulation import (
    run_simulation_native,
)


class BatchCoefficientTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        rng = np.random.RandomState(5)
        # client_b holds two batches, client_c misses feature_c
        self.clients = {"client_a": ["x"], "client_b": ["x", "y"], "client_c": ["x"]}
        for number, (name, batches) in enumerate(self.clients.items()):
            folder = self.root / name
            folder.mkdir()
            samples = [f"{name}_{batch}_{i}" for batch in batches for i in range(8)]
            condition = np.tile([0, 1], 4 * len(batches))
            batch_shift = np.repeat([b for b, _ in enumerate(batches)], 8) * 1.5
            features = ["feature_a", "feature_b", "feature_c"][:3 if number < 2 else 2]
            values = (rng.normal(size=(len(features), len(samples))) + 2 * number
                      + condition + batch_shift)
            values[0, 1] = np.nan
            pd.DataFrame(values, index=pd.Index(features, name="rowname"),
                         columns=samples).to_csv(folder / "data.tsv", sep="\t")
            pd.DataFrame({"condition": condition, "batch": np.repeat(batches, 8)},
                         index=pd.Index(samples, name="file")).to_csv(
                folder / "design.tsv", sep="\t")

    def configure(self, covariates):
        for name in self.clients:
            config = {"data_filename": "data.tsv", "design_filename": "design.tsv",
                      "expression_file_flag": True, "index_col": "rowname",
                      "covariates": covariates, "separator": "\t", "smpc": False,
                      "min_samples": 0, "batch_col": "batch" if name == "client_b" else None}
            (self.root / name / "config.yml").write_text(
                yaml.safe_dump({"flimmaBatchCorrection": config}))

    def run_fedrbe(self, covariates):
        self.configure(covariates)
        names = list(self.clients)
        outputs = [self.root / f"output_{name}_{len(covariates)}" for name in names]
        for output in outputs:
            output.mkdir()
        with contextlib.redirect_stdout(io.StringIO()):
            run_simulation_native([str(self.root / name) for name in names],
                                  [str(output) for output in outputs], None, run_fedrbe,
                                  client_ids=names)
        return outputs

    def test_coordinator_coefficients_reproduce_every_client(self):
        for covariates in ([], ["condition"]):
            with self.subTest(covariates=covariates):
                outputs = self.run_fedrbe(covariates)
                self.assertTrue((outputs[0] / BATCH_COEFFICIENTS_FILE).exists())
                self.assertFalse(any((output / BATCH_COEFFICIENTS_FILE).exists()
                                     for output in outputs[1:]))
                coefficients = pd.read_csv(outputs[0] / BATCH_COEFFICIENTS_FILE, sep="\t",
                                           index_col=0)
                # four batches, the last one is the reference without a column
                self.assertEqual(list(coefficients.columns),
                                 ["client_a", "client_b|x", "client_b|y"])
                # rows are the hashed names of the corrected features, feature_c
                # is not corrected as client_c misses it
                hashes = {hashlib.sha3_256(name.encode()).hexdigest(): name
                          for name in ("feature_a", "feature_b")}
                self.assertEqual(set(coefficients.index), set(hashes))
                for name, output in zip(self.clients, outputs):
                    client = Client()
                    with contextlib.redirect_stdout(io.StringIO()):
                        client.config_based_init(clientname=name,
                                                 input_folder=str(self.root / name))
                    labels = ([f"{name}|{batch}" for batch in client.design["batch"]]
                              if client.batch_col else [name] * len(client.sample_names))
                    design = pd.DataFrame(
                        [[-1] * 3 if label not in coefficients.columns
                         else [int(label == column) for column in coefficients.columns]
                         for label in labels],
                        index=client.data.columns, columns=coefficients.columns)
                    data = client.data.loc[client.data.index.isin(coefficients.index)]
                    corrected = subtract_batch_effect(
                        data, coefficients.loc[data.index].to_numpy(), design)
                    expected = pd.read_csv(output / "only_batch_corrected_data.csv",
                                           sep="\t", index_col=0)
                    self.assertEqual(sorted(expected.index), sorted(hashes.values()))
                    pd.testing.assert_frame_equal(
                        corrected.rename(index=client.hash2feature).loc[expected.index],
                        expected, check_names=False, rtol=0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
