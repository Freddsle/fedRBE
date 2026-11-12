"""Regression checks for the random-forest app and the pooled evaluation metrics."""

import contextlib
import io
from pathlib import Path
import sys
import pickle
import tempfile
import unittest

import pandas as pd
import yaml

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef
from sklearn.tree import DecisionTreeClassifier
from unittest.mock import patch

from evaluation_classification_after_correction.fc_fed_forest_simple_app.fedForestSimple import main
from evaluation_utils.metrics import classification_scores, classification_scores_from_confusion
from evaluation_classification_after_correction.fc_fed_forest_simple_app.fc_fedlearnsim_helper.run_app_simulation import (
    run_simulation_native,
)

COMPONENT = Path(__file__).resolve().parents[1]
with patch.object(sys, "path", [str(COMPONENT), *sys.path]):
    from helper import train_test_correction


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

    def test_pooled_scores_use_all_samples_and_each_cohort_is_scored(self):
        cohorts = {
            "A": (np.array(['D5'] * 9 + ['D6']), np.array(['D5'] * 10)),
            "B": (np.array(['D6', 'F7', 'M8', 'M8']), np.array(['D6', 'M8', 'M8', 'F7'])),
            "C": (np.array(['F7', 'F7']), np.array(['F7', 'D5'])),
        }
        true = np.concatenate([y for y, _ in cohorts.values()])
        predicted = np.concatenate([p for _, p in cohorts.values()])
        pooled = train_test_correction._score_rows(cohorts)
        per_cohort = train_test_correction._score_rows(cohorts, pooled=False)
        self.assertEqual(list(pooled), ["All", "A", "B", "C"])
        self.assertEqual(list(per_cohort), ["A", "B", "C"])
        self.assertEqual(pooled["All"], classification_scores(true, predicted))
        for cohort, (y, p) in cohorts.items():
            self.assertEqual(pooled[cohort], classification_scores(y, p))
            self.assertEqual(per_cohort[cohort], classification_scores(y, p))
        local_average = np.mean([list(pooled[c].values()) for c in cohorts], axis=0)
        self.assertFalse(np.allclose(list(pooled["All"].values()), local_average))


class RandomForestSeedTests(unittest.TestCase):
    def fit_fixture(self, seed, offset=0, clients=1):
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

    def test_zero_seed_reproduces_federated_trees_on_all_samples(self):
        seeds, predictions = self.fit_fixture(0, clients=2)
        repeated_seeds, repeated = self.fit_fixture(0, clients=2)
        paired_seeds, paired = self.fit_fixture(0, offset=10, clients=2)
        self.assertTrue(all(isinstance(seed, int) for seed in seeds))
        self.assertEqual(seeds, repeated_seeds)
        self.assertEqual(seeds, paired_seeds)
        for original, repeat, shifted in zip(predictions, repeated, paired):
            pd.testing.assert_frame_equal(original, repeat)
            # every sample is a training sample
            self.assertEqual(len(original), 30)
            pd.testing.assert_index_equal(original.index, shifted.index)

    def test_train_test_ratio_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            pd.DataFrame({"x0": [0., 1.], "label": [0, 1]}).to_csv(
                folder / "data.csv", index_label="sample_id")
            (folder / "config_forest.yaml").write_text(yaml.safe_dump({"simple_forest": {
                "data_filename": "data.csv", "csv_seperator": ",", "sample_col": "sample_id",
                "predicted_feature_name": "label", "train_test_ratio": 0.7,
            }}))
            with contextlib.redirect_stdout(io.StringIO()), \
                    self.assertRaisesRegex(ValueError, "train_test_ratio is not supported"):
                run_simulation_native([str(folder)], [str(folder)], None, main)


class RandomForestNumericInputTests(unittest.TestCase):
    def test_trees_match_named_dataframe_reference(self):
        # The reference keeps sklearn's DataFrame conversion inside every fit.
        # Check both the numeric fast path and coercion, including missing values.
        for kind in ("numeric", "coercion", "large_integer", "mixed_boolean"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                input_dir, output_dir = root / 'input', root / 'output'
                input_dir.mkdir()
                output_dir.mkdir()
                rng = np.random.RandomState(17)
                frame = pd.DataFrame(rng.normal(size=(30, 6)),
                                     columns=[f'x{i}' for i in range(6)])
                frame.loc[[0, 7], 'x1'] = np.nan
                if kind == "coercion":
                    frame['x2'] = frame.x2.astype(str)
                    frame.loc[4, 'x2'] = 'not_numeric'
                elif kind in ("large_integer", "mixed_boolean"):
                    frame['x2'] = np.uint64(2**63 + 2**39) + np.arange(30, dtype=np.uint64)
                    if kind == "mixed_boolean":
                        frame['x3'] = np.tile([True, False], 15)
                frame['label'] = np.tile([0, 1], 15)
                data = input_dir / 'data.csv'
                frame.to_csv(data, index_label='sample_id')
                params = dict(max_depth=4, max_features='sqrt',
                              min_samples_leaf=1, min_samples_split=2)
                config = dict(data_filename='data.csv', csv_seperator=',',
                              features_as_columns=True, sample_col='sample_id',
                              predicted_feature_name='label', random_state=42,
                              num_estimators_total=7, **params)
                (input_dir / 'config_forest.yaml').write_text(
                    yaml.safe_dump({'simple_forest': config}))
                run_simulation_native([str(input_dir)], [str(output_dir)], None, main)
                with (output_dir / 'global_model.pkl').open('rb') as stream:
                    forest = pickle.load(stream)
                original = pd.read_csv(data, index_col='sample_id')
                features = original.drop(columns='label').apply(pd.to_numeric, errors='coerce')
                features = features[sorted(features.columns)]
                labels = original.label
                train, target = features, labels
                seeds = np.random.RandomState(42).randint(np.iinfo(np.int32).max, size=7)
                self.assertEqual(len(forest.estimators_), len(seeds))
                for actual, seed in zip(forest.estimators_, seeds):
                    expected = DecisionTreeClassifier(random_state=int(seed), **params).fit(
                        train, target)
                    np.testing.assert_array_equal(actual.feature_names_in_, train.columns)
                    np.testing.assert_array_equal(actual.classes_, expected.classes_)
                    for field in ('nodes', 'values'):
                        np.testing.assert_array_equal(actual.tree_.__getstate__()[field],
                                                      expected.tree_.__getstate__()[field])
                    np.testing.assert_array_equal(actual.predict_proba(features),
                                                  expected.predict_proba(features))


class RandomForestArtifactTests(unittest.TestCase):
    def test_omitting_prediction_exports_preserves_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / 'input'
            input_dir.mkdir()
            rng = np.random.RandomState(19)
            frame = pd.DataFrame(rng.normal(size=(30, 6)),
                                 columns=[f'x{i}' for i in range(6)])
            frame['label'] = np.tile([0, 1], 15)
            frame.to_csv(input_dir / 'data.csv', index_label='sample_id')
            config = dict(data_filename='data.csv', csv_seperator=',',
                          features_as_columns=True, sample_col='sample_id',
                          predicted_feature_name='label', random_state=42,
                          num_estimators_total=7, max_depth=4, max_features='sqrt')
            for save in (True, False):
                output = root / str(save)
                output.mkdir()
                config['save_predictions'] = save
                (input_dir / 'config_forest.yaml').write_text(
                    yaml.safe_dump({'simple_forest': config}))
                run_simulation_native([str(input_dir)], [str(output)], None, main)
                self.assertEqual((output / 'pred.csv').exists(), save)
            for name in ('global_model.pkl', 'model_info.yaml'):
                self.assertEqual((root / 'True' / name).read_bytes(),
                                 (root / 'False' / name).read_bytes())


if __name__ == '__main__':
    unittest.main()
