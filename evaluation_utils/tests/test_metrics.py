"""Regression checks for shared label-score definitions and confusion counts."""

import unittest

import numpy as np

from evaluation_utils.metrics import (
    classification_scores,
    classification_scores_from_confusion,
    clustering_scores,
)


class SharedMetricTests(unittest.TestCase):
    def assert_scores_close(self, actual, expected):
        self.assertEqual(actual.keys(), expected.keys())
        for name, value in expected.items():
            self.assertAlmostEqual(actual[name], value, places=12)

    def test_labels_and_confusion_counts_agree(self):
        expected = {'MCC': 0.4375, 'F1_score': 0.6}
        self.assert_scores_close(classification_scores(
            ['a', 'a', 'b', 'b', 'c'], ['a', 'b', 'b', 'c', 'c']), expected)
        self.assert_scores_close(classification_scores_from_confusion(
            [[1, 1, 0], [0, 1, 1], [0, 0, 1]]), expected)

    def test_partition_score_definitions(self):
        self.assert_scores_close(clustering_scores(
            [0, 0, 1, 1, 2, 2], [0, 0, 0, 1, 1, 1]), {
                'ARI': 0.24242424242424243,
                'AMI': 0.2987924581708903,
                'NMI': 0.5158037429793889,
            })

    def test_cluster_renumbering_preserves_partition_but_changes_classes(self):
        truth, swapped = [0, 0, 1, 1], [1, 1, 0, 0]
        expected = {'ARI': 1., 'AMI': 1., 'NMI': 1.}
        self.assert_scores_close(clustering_scores(truth, swapped), expected)
        self.assert_scores_close(clustering_scores(truth, ['B', 'B', 'A', 'A']), expected)
        self.assert_scores_close(classification_scores(truth, swapped),
                                 {'MCC': -1., 'F1_score': 0.})

    def test_invalid_confusion_counts(self):
        for counts in ([], [[1, 2]], [[0, 0], [0, 0]], [[1, -1], [0, 1]],
                       [[1, np.nan], [0, 1]], [[1, np.inf], [0, 1]]):
            with self.subTest(counts=counts):
                with self.assertRaises(ValueError):
                    classification_scores_from_confusion(counts)


if __name__ == '__main__':
    unittest.main()
