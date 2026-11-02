"""Regression checks for local and pooled random-forest evaluation metrics."""

import json
from pathlib import Path
import pickle
import tempfile
import unittest

import pandas as pd
import yaml

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef

from evaluation_classification_after_correction.fc_fed_forest_simple_app.fedForestSimple import (
    _pool_confusion_matrices, main,
)
from evaluation_utils.metrics import classification_scores_from_confusion
from evaluation_classification_after_correction.fc_fed_forest_simple_app.fc_fedlearnsim_helper.run_app_simulation import (
    run_simulation_native,
)


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


class RandomForestSeedTests(unittest.TestCase):
    def fit_fixture(self, seed, offset=0, clients=1, train_ratio=1.0):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        inputs, outputs = [], []
        rng = np.random.RandomState(17)
        for client in range(clients):
            input_folder, output_folder = root / f"input{client}", root / f"output{client}"
            input_folder.mkdir()
            output_folder.mkdir()
            features = rng.normal(size=(30, 6)) + offset
            frame = pd.DataFrame(features, columns=[f"x{i}" for i in range(6)])
            frame["label"] = np.tile([0, 1], 15)
            frame.index = [f"client{client}_sample{i}" for i in range(len(frame))]
            frame.to_csv(input_folder / "data.csv", index_label="sample_id")
            (input_folder / "config_forest.yaml").write_text(yaml.safe_dump({"simple_forest": {
                "data_filename": "data.csv", "csv_seperator": ",",
                "features_as_columns": True, "sample_col": "sample_id",
                "predicted_feature_name": "label", "random_state": seed,
                "num_estimators_total": 100, "max_depth": 4, "max_features": "sqrt",
                "train_test_ratio": train_ratio,
            }}))
            inputs.append(str(input_folder))
            outputs.append(str(output_folder))
        run_simulation_native(inputs, outputs, None, main)
        with (Path(outputs[0]) / "global_model.pkl").open("rb") as stream:
            forest = pickle.load(stream)
        predictions = [pd.read_csv(Path(folder) / "pred.csv", index_col=0) for folder in outputs]
        return [tree.random_state for tree in forest.estimators_], predictions

    def test_adjacent_run_seeds_do_not_reuse_tree_rng_blocks(self):
        first, _ = self.fit_fixture(42)
        second, _ = self.fit_fixture(43)
        self.assertEqual(len(set(first)), 100)
        self.assertEqual(len(set(second)), 100)
        self.assertTrue(set(first).isdisjoint(second))

    def test_zero_seed_reproduces_federated_trees_and_paired_splits(self):
        seeds, predictions = self.fit_fixture(0, clients=2, train_ratio=0.7)
        repeated_seeds, repeated = self.fit_fixture(0, clients=2, train_ratio=0.7)
        paired_seeds, paired = self.fit_fixture(0, offset=10, clients=2, train_ratio=0.7)
        self.assertTrue(all(isinstance(seed, int) for seed in seeds))
        self.assertEqual(seeds, repeated_seeds)
        self.assertEqual(seeds, paired_seeds)
        for original, repeat, corrected in zip(predictions, repeated, paired):
            pd.testing.assert_frame_equal(original, repeat)
            pd.testing.assert_frame_equal(original[["label", "dataset"]],
                                          corrected[["label", "dataset"]])
            self.assertEqual((original.dataset == "test").sum(), 9)


if __name__ == '__main__':
    unittest.main()
