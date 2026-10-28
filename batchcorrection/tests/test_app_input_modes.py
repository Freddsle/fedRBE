"""Regression checks for design-free inputs and client ordering defaults."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import yaml

from batchcorrection.classes.client import Client
from batchcorrection.classes.coordinator_utils import (
    batch_coefficients,
    compute_beta,
    create_beta_mask,
    create_feature_presence_matrix,
    select_common_features_variables,
)


class AppInputModeTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.names = ["client_a", "client_b", "client_c"]
        self.covariate = np.array([0, 1] * 4, dtype=float)
        self.expression = []
        for number, name in enumerate(self.names):
            values = np.array([
                10 + 3 * number + 2 * self.covariate + np.arange(8) / 10,
                20 - 2 * number + self.covariate / 2 + np.arange(8) ** 2 / 20,
            ])
            if number == 1:
                values[1, 2] = np.nan
            self.expression.append(pd.DataFrame(
                values, index=["feature_a", "feature_b"],
                columns=[f"{name}_sample_{index}" for index in range(8)],
            ))

    def initialize(self, expression_layout, covariates):
        clients = []
        for name, expression in zip(self.names, self.expression):
            directory = self.root / name
            directory.mkdir(exist_ok=True)
            data = expression.copy()
            if covariates:
                data.loc["condition"] = self.covariate
            (data if expression_layout else data.T).to_csv(directory / "data.tsv", sep="\t")
            config = {
                "data_filename": "data.tsv", "expression_file_flag": expression_layout,
                "index_col": 0, "covariates": covariates, "separator": "\t",
            }
            (directory / "config.yml").write_text(
                yaml.safe_dump({"flimmaBatchCorrection": config}), encoding="utf-8"
            )
            client = Client()
            client.config_based_init(
                clientname=name, input_folder=str(directory), use_hashing=True
            )
            clients.append(client)
        return clients

    def pooled_expected(self, covariates):
        # Build the pooled model independently of the client's design matrices.
        batch = np.repeat([[1, 0], [0, 1], [-1, -1]], 8, axis=0)
        columns = [np.ones(24)]
        if covariates:
            columns.append(np.tile(self.covariate, 3))
        design = np.column_stack(columns + [batch])
        observed = pd.concat(self.expression, axis=1)
        corrected = observed.copy()
        for feature, values in observed.iterrows():
            finite = np.isfinite(values)
            coefficients = np.linalg.lstsq(design[finite], values[finite], rcond=None)[0]
            corrected.loc[feature] = values - batch @ coefficients[-2:]
        return corrected

    def test_design_free_modes_match_pooled_regression(self):
        for expression_layout in (True, False):
            for covariates in ([], ["condition"]):
                with self.subTest(layout=expression_layout, covariates=covariates), \
                        contextlib.redirect_stdout(io.StringIO()):
                    clients = self.initialize(expression_layout, covariates)
                    information = [
                        (client.client_name, client.position, client.reference_batch,
                         client.get_batch_feature_presence_info(4 + len(covariates)))
                        for client in clients
                    ]
                    features, presence, cohorts = select_common_features_variables(
                        information, default_order=self.names
                    )
                    statistics = []
                    for client in clients:
                        client.validate_inputs(list(clients[0].hash2variable))
                        client.set_data(features)
                        self.assertIsNone(client.create_design(cohorts))
                        xtx, xty, error = client.compute_XtX_XtY()
                        self.assertFalse(error)
                        statistics.append((xtx, xty))
                    width = clients[0].design.shape[1]
                    mask = create_beta_mask(presence, len(features), width)
                    full = compute_beta(statistics, len(features), width, mask,
                                        n_batch_columns=presence.shape[1] - 1)
                    betas = batch_coefficients(full, presence.shape[1] - 1)
                    with self.assertRaisesRegex(ValueError, "Expected coefficients for 2 batch columns"):
                        clients[0].remove_batch_effects(full)
                    expected = self.pooled_expected(covariates)
                    for client, expression in zip(clients, self.expression):
                        client.remove_batch_effects(betas)
                        pd.testing.assert_frame_equal(
                            client.data_corrected.sort_index(), expected[expression.columns],
                            atol=1e-11, rtol=1e-12,
                        )
                        if covariates:
                            np.testing.assert_array_equal(
                                client.full_corrected_data.loc["condition"], self.covariate
                            )


class ClientOrderingTests(unittest.TestCase):
    def information(self, positions=(None, None, None), references=(None, None, None),
                    multiple_batches=False):
        information = []
        for index, name in enumerate(("a", "b", "c")):
            batches = [f"{name}|alpha", f"{name}|beta"] if multiple_batches and name == "b" else [name]
            information.append((name, positions[index], references[index],
                                {batch: ["feature"] for batch in batches}))
        return information

    def ordered(self, information, default_order):
        original = list(default_order)
        with contextlib.redirect_stdout(io.StringIO()):
            _, cohorts = create_feature_presence_matrix(information, ["feature"], default_order)
        self.assertEqual(default_order, original)
        return cohorts

    def test_omitted_or_mixed_positions_keep_default_order(self):
        for positions in ((None, None, None), (2, None, 0)):
            with self.subTest(positions=positions):
                self.assertEqual(self.ordered(
                    self.information(positions), ["c", "a", "b"]
                ), ["c", "a", "b"])

    def test_boolean_and_multibatch_references_move_to_end(self):
        cases = [
            ((True, None, None), False, ["a", "b", "c"], ["b", "c", "a"]),
            ((None, "alpha", None), True, ["c", "b", "a"],
             ["c", "a", "b|beta", "b|alpha"]),
        ]
        for references, multiple_batches, default_order, expected in cases:
            with self.subTest(references=references):
                self.assertEqual(self.ordered(self.information(
                    references=references, multiple_batches=multiple_batches
                ), default_order), expected)

    def test_integer_positions_determine_order_and_reject_conflicting_reference(self):
        for references in ((None, None, None), (True, None, None)):
            with self.subTest(references=references):
                self.assertEqual(self.ordered(
                    self.information((2, 0, 1), references), ["a", "b", "c"]
                ), ["b", "c", "a"])
        with self.assertRaisesRegex(ValueError, "Reference client is not the last"):
            self.ordered(self.information((0, 1, 2), (True, None, None)), ["a", "b", "c"])
        with self.assertRaisesRegex(ValueError, r"distinct integer position, got \[1, 1, 2\]"):
            self.ordered(self.information((1, 1, 2)), ["a", "b", "c"])

    def test_reference_client_missing_from_default_order_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Reference client a not found"):
            self.ordered(self.information(references=(True, None, None)), ["b", "c"])


if __name__ == "__main__":
    unittest.main()
