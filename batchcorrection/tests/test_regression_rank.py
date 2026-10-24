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
                        xtx_error=xtx_error,
                    )

    def test_missing_observations_can_make_only_one_feature_aliased(self):
        membership = np.repeat(np.eye(3), 6, axis=0)
        batches = membership[:, :2] - membership[:, -1:]
        covariate = np.tile([0., 0., 0., 1., 1., 1.], 3)
        design = np.column_stack([np.ones(18), covariate, batches])
        response = design @ np.array([30., 2., 5., 7.])
        observed = [np.ones(18, bool), covariate == 0]
        gram = np.array([design[keep].T @ design[keep] for keep in observed])
        target = np.array([design[keep].T @ response[keep] for keep in observed])
        mask = create_beta_mask(np.ones((2, 3), int), 2, 4)
        with self.assertRaisesRegex(ValueError, "Cannot reliably fit feature index 1"):
            compute_beta([[gram, target]], 2, 4, mask)

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

        result = compute_beta([[gram[None], target[None]]], 1, 4, mask, xtx_error=3e-12)
        np.testing.assert_allclose(result, [[37., 2., -2., 0.]], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
