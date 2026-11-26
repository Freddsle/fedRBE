"""Regression checks for shared label-score definitions and confusion counts."""

import unittest

import numpy as np

from evaluation_utils.metrics import (
    classification_scores,
    classification_scores_from_confusion,
    clustering_scores,
    confusion_counts,
)
from sklearn.metrics import confusion_matrix


class SharedMetricTests(unittest.TestCase):
    def assert_scores_close(self, actual, expected):
        self.assertEqual(actual.keys(), expected.keys())
        for name, value in expected.items():
            self.assertAlmostEqual(actual[name], value, places=12)

    def test_labels_and_confusion_counts_agree(self):
        # per-class F1 2/3, 1/2, 2/3 with supports 2, 2, 1
        expected = {'MCC': 0.4375, 'weighted_F1_score': 0.6, 'macro_F1_score': 11 / 18}
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
                                 {'MCC': -1., 'weighted_F1_score': 0., 'macro_F1_score': 0.})

    def test_one_vs_rest_counts_match_the_confusion_matrix(self):
        truth = ['a', 'a', 'b', 'b', 'c']
        labels = ['a', 'b', 'b', 'c', 'c']
        classes = ['a', 'b', 'c', 'd']  # 'd' is absent from the data
        counts = confusion_counts(truth, labels, classes)
        matrix = confusion_matrix(truth, labels, labels=classes)
        self.assertEqual(len(counts), 4 * len(classes))
        for i, cls in enumerate(classes):
            with self.subTest(cls=cls):
                tp = matrix[i, i]
                fp = matrix[:, i].sum() - tp
                fn = matrix[i, :].sum() - tp
                self.assertEqual(counts, counts | {f'TP|{cls}': tp, f'FP|{cls}': fp,
                                                   f'FN|{cls}': fn,
                                                   f'TN|{cls}': len(truth) - tp - fp - fn})
        self.assertEqual([counts[f'{c}|d'] for c in ('TP', 'FP', 'FN', 'TN')], [0, 0, 0, 5])

    def test_binary_counts_and_labels_outside_the_classes(self):
        self.assertEqual(confusion_counts([0, 0, 1, 1, 1], [0, 1, 1, 1, 0], [0, 1]), {
            'TP|0': 1, 'TN|0': 2, 'FP|0': 1, 'FN|0': 1,
            'TP|1': 2, 'TN|1': 1, 'FP|1': 1, 'FN|1': 1,
        })
        with self.assertRaisesRegex(ValueError, "not among the classes"):
            confusion_counts([0, 2], [0, 0], [0, 1])
        with self.assertRaisesRegex(ValueError, "not among the classes"):
            confusion_counts(['0', '1'], ['0', '1'], [0, 1])

    def test_invalid_confusion_counts(self):
        for counts in ([], [[1, 2]], [[0, 0], [0, 0]], [[1, -1], [0, 1]],
                       [[1, np.nan], [0, 1]], [[1, np.inf], [0, 1]]):
            with self.subTest(counts=counts):
                with self.assertRaises(ValueError):
                    classification_scores_from_confusion(counts)


if __name__ == '__main__':
    unittest.main()
