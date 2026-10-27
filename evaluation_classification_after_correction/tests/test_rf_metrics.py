"""Regression checks for local and pooled random-forest evaluation metrics."""

import json
import unittest

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef

from evaluation_classification_after_correction.fc_fed_forest_simple_app.fedForestSimple import (
    _pool_confusion_matrices,
)
from evaluation_utils.metrics import classification_scores_from_confusion


class RandomForestMetricTests(unittest.TestCase):
    def assert_scores_match(self, true, predicted, labels):
        actual = classification_scores_from_confusion(
            confusion_matrix(true, predicted, labels=labels))
        expected = {
            'MCC': matthews_corrcoef(true, predicted),
            'F1_score': f1_score(true, predicted, average='weighted', zero_division=0),
        }
        self.assertEqual(actual.keys(), expected.keys())
        np.testing.assert_allclose(list(actual.values()), list(expected.values()),
                                   rtol=0, atol=1e-12)
        return actual

    def test_imbalanced_majority_prediction(self):
        true = [0] * 90 + [1] * 10
        scores = self.assert_scores_match(true, [0] * 100, [0, 1])
        self.assertEqual(scores['MCC'], 0)
        self.assertAlmostEqual(scores['F1_score'], 0.9 * (180 / 190))
        self.assertNotAlmostEqual(scores['MCC'], 0.8)
        self.assertNotAlmostEqual(scores['F1_score'], 0.9)

    def test_binary_perfect_constant_and_wrong_predictions(self):
        examples = [
            ([0, 0, 1, 1], [0, 0, 1, 1]),
            ([0, 0, 0], [0, 0, 0]),
            ([0, 0, 1, 1], [1, 1, 0, 0]),
            ([0, 1, 1, 1], [0, 0, 0, 0]),
        ]
        for true, predicted in examples:
            with self.subTest(true=true, predicted=predicted):
                self.assert_scores_match(true, predicted, [0, 1])

    def test_four_classes_and_reordered_vocabulary(self):
        true = ['D5'] * 8 + ['D6'] * 4 + ['F7'] * 3 + ['M8'] * 2
        predicted = (
            ['D5'] * 6 + ['D6', 'M8'] + ['D6'] * 2 + ['D5', 'F7']
            + ['F7', 'D5', 'D6'] + ['M8', 'D5']
        )
        self.assert_scores_match(true, predicted, ['D5', 'D6', 'F7', 'M8'])
        self.assert_scores_match(true, predicted, ['M8', 'F7', 'D6', 'D5'])

    def test_locally_absent_classes(self):
        self.assert_scores_match(
            ['D5', 'D5', 'D6'], ['D5', 'D6', 'D6'],
            ['M8', 'D6', 'F7', 'D5'],
        )

    def test_pooled_clients_align_vocabulary_and_score_all_samples(self):
        clients = [
            (['D5'] * 9 + ['D6'], ['D5'] * 10, ['D6', 'D5']),
            (['D6', 'F7', 'M8', 'M8'], ['D6', 'M8', 'M8', 'F7'],
             ['M8', 'F7', 'D6']),
            (['F7', 'F7'], ['F7', 'D5'], ['D5', 'F7']),
        ]
        payloads = [{
            'classes': classes,
            'matrix': confusion_matrix(true, predicted, labels=classes).tolist(),
        } for true, predicted, classes in clients]
        # The transmitted representation contains no numpy scalar/array objects.
        payloads = json.loads(json.dumps(payloads))
        classes, pooled = _pool_confusion_matrices(payloads)
        true = [value for y, _, _ in clients for value in y]
        predicted = [value for _, p, _ in clients for value in p]
        np.testing.assert_array_equal(
            pooled, confusion_matrix(true, predicted, labels=classes),
        )
        actual = classification_scores_from_confusion(pooled)
        expected = self.assert_scores_match(true, predicted, classes)
        np.testing.assert_allclose(list(actual.values()), list(expected.values()),
                                   rtol=0, atol=1e-12)
        local_average = np.mean([
            list(classification_scores_from_confusion(payload['matrix']).values())
            for payload in payloads
        ], axis=0)
        self.assertFalse(np.allclose(list(actual.values()), local_average))


if __name__ == '__main__':
    unittest.main()
