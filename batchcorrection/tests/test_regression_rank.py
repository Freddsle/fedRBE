"""Reject unidentified regression models while retaining valid masked fits."""

import unittest

import numpy as np

from batchcorrection.classes.coordinator_utils import compute_beta, create_beta_mask


class RegressionRankTests(unittest.TestCase):
    def test_aliased_covariate_is_rejected_in_plaintext_and_rounded_smpc(self):
        membership = np.repeat(np.eye(3), 4, axis=0)
        batches = membership[:, :2] - membership[:, -1:]
        weights = np.random.default_rng(501).uniform(0.1, 2, 3)
        design = np.column_stack([
            np.ones(12), weights[0] + batches @ weights[1:], batches,
        ])
        response = design @ np.array([30., 20., 5., 7.])
        for xtx_error in (0.0, 3e-12):
            with self.subTest(xtx_error=xtx_error):
                gram = design.T @ design
                target = design.T @ response
                if xtx_error:
                    # Apply fixed-point-scale perturbations before aggregation.
                    gram = sum(np.round(part.T @ part, 12) for part in np.split(design, 3))
                    target = sum(
                        np.round(part.T @ values, 12)
                        for part, values in zip(np.split(design, 3), np.split(response, 3))
                    )
                with self.assertRaisesRegex(ValueError, "rank-deficient design"):
                    compute_beta(
                        [[gram[None], target[None]]], 1, 4, np.zeros((1, 4), bool),
                        xtx_error=xtx_error, n_batch_columns=2,
                    )

    def test_featurewise_zero_and_one_covariates_preserve_batch_coefficients(self):
        membership = np.repeat(np.eye(3), 6, axis=0)
        batches = membership[:, :2] - membership[:, -1:]
        covariate = np.tile([0., 0., 0., 1., 1., 1.], 3)
        design = np.column_stack([np.ones(18), covariate, batches])
        response = design @ np.array([30., 2., 5., 7.])
        observed = [np.ones(18, bool), covariate == 0, covariate == 1]
        mask = create_beta_mask(np.ones((3, 3), int), 3, 4)
        expected = [[30., 2., 5., 7.], [30., 0., 5., 7.], [32., 0., 5., 7.]]
        for xtx_error in (0.0, 3e-12):
            with self.subTest(xtx_error=xtx_error):
                statistics = []
                for indices in np.split(np.arange(18), 3):
                    rows = [indices[keep[indices]] for keep in observed]
                    gram = np.array([design[row].T @ design[row] for row in rows])
                    target = np.array([design[row].T @ response[row] for row in rows])
                    if xtx_error:
                        gram = np.trunc(gram * 1e12) / 1e12
                        target = np.trunc(target * 1e12) / 1e12
                    statistics.append([gram, target])
                if xtx_error:
                    statistics = [[sum(values) for values in zip(*statistics)]]
                result = compute_beta(statistics, 3, 4, mask, xtx_error=xtx_error,
                                      n_batch_columns=2)
                np.testing.assert_allclose(result, expected, atol=1e-9)

    def test_missing_batch_is_masked_before_rank_check(self):
        membership = np.repeat(np.eye(3), 6, axis=0)
        batches = membership[:, :2] - membership[:, -1:]
        covariate = np.tile([0., 0., 0., 1., 1., 1.], 3)
        design = np.column_stack([np.ones(18), covariate, batches])
        response = design @ np.array([30., 2., 5., 7.])
        observed = membership[:, 2] == 0
        gram = design[observed].T @ design[observed]
        target = design[observed].T @ response[observed]
        mask = create_beta_mask(np.array([[1, 1, 0]]), 1, 4)

        result = compute_beta([[gram[None], target[None]]], 1, 4, mask,
                              xtx_error=3e-12, n_batch_columns=2)
        np.testing.assert_allclose(result, [[37., 2., -2., 0.]], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
