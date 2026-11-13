"""Checks of the leave-one-cohort-out evaluation whose batch effect correction sees only the training cohorts."""

import contextlib
import copy
import hashlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from batchcorrection.align_new_batch import read_training_level

COMPONENT = Path(__file__).resolve().parents[1]
with patch.object(sys, "path", [str(COMPONENT), *sys.path]):
    from helper import common, loco
    from helper.experiment import run_units
    from helper.steps import correct, datasets, evaluate, prepare, train_forest
    from tests.test_train_test_split import write_fixture


def three_cohorts(root: Path) -> datasets.DatasetSpec:
    """
    The train/test fixture with client_a, client_b (two batches) and client_c,
    so every fold has two training clients. client_c is the explicit reference
    client, as the last Quartet client.
    """
    spec = write_fixture(root)
    federation = spec.federations[0]
    keep = slice(0, 3)
    configs = {arm: copy.deepcopy(arm_configs[keep])
               for arm, arm_configs in federation.configs.items()}
    for arm_configs in configs.values():
        arm_configs[2]["reference_batch"] = True
    federation = datasets.Federation(None, federation.cohorts[keep], federation.client_dirs[keep],
                                configs)
    return datasets.DatasetSpec("fixture", "Fixture", "label", [federation])


def feature_hash(name: str) -> str:
    return hashlib.sha3_256(name.encode()).hexdigest()


class FoldTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.spec = three_cohorts(self.root)
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))

    def test_fold_drops_the_held_out_cohort_and_allows_two_clients(self):
        original = copy.deepcopy(self.spec.federations[0].configs)
        for held_out in self.spec.cohorts:
            with self.subTest(held_out=held_out):
                fold = prepare.fold_spec(self.spec, held_out)
                federation = fold.federations[0]
                expected = [cohort for cohort in self.spec.cohorts if cohort != held_out]
                self.assertEqual(federation.cohorts, expected)
                self.assertEqual([folder.name for folder in federation.client_dirs], expected)
                for arm in common.ARMS:
                    for cohort, config in zip(expected, federation.configs[arm]):
                        source = original[arm][self.spec.cohorts.index(cohort)]
                        self.assertEqual(config, {**source, "min_clients": 2})
        # the dataset itself is unchanged
        self.assertEqual(self.spec.federations[0].configs, original)
        with self.assertRaisesRegex(ValueError, "unknown cohort"):
            prepare.fold_spec(self.spec, "client_x")

    def test_split_marks_the_held_out_cohort_as_test(self):
        inputs = datasets.read_inputs(self.spec)
        split = prepare.fold_split(self.spec, inputs, "client_b")
        self.assertEqual(set(split.loc[split.cohort == "client_b", "part"]), {"test"})
        self.assertEqual(set(split.loc[split.cohort != "client_b", "part"]), {"train"})
        self.assertEqual(len(split), sum(len(client.keys) for client in inputs[0]))


class HeldOutCorrectionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.spec = three_cohorts(self.root)
        self.inputs = datasets.read_inputs(self.spec)
        self.experiment = loco.ClassificationExperimentLeaveOneCohortOut(self.spec)
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))

    def prepare(self, held_out, arm, tag=""):
        held_out_idx = self.spec.cohorts.index(held_out)
        training = prepare.fold_spec(self.spec, held_out)
        training_inputs = [[client for idx, client in enumerate(fed_inputs) if idx != held_out_idx]
                           for fed_inputs in self.inputs]
        split = prepare.fold_split(self.spec, self.inputs, held_out)
        folders = prepare.write_split_inputs(self.spec, self.inputs, split,
                                         self.root / f"split_{held_out}_{arm}{tag}")
        tables, files = self.experiment._prepare(
            arm, held_out, training, training_inputs, self.inputs, split, folders,
            self.root / f"work_{held_out}_{arm}{tag}")
        return tables, files, folders

    def raw(self, folders, column_names):
        (folder,) = folders[(0, "test")]
        raw = pd.read_csv(folder / "data.tsv", sep="\t", index_col=0).T
        return raw.loc[column_names.index]

    def test_held_out_batches_are_shifted_onto_the_training_level(self):
        for arm in common.ARMS:
            with self.subTest(arm=arm):
                tables, files, folders = self.prepare("client_b", arm)
                self.assertEqual(list(tables["train"]), ["client_a", "client_c"])
                test = tables["test"]["client_b"]
                level = files["training_level.csv"]
                info = files["training_level_info.yml"]
                self.assertEqual(info["covariates_used"], arm == common.ARM_LABEL_AWARE)
                # each of client_b's batches has the training level as mean
                for batch in ("x", "y"):
                    samples = test.index[test.index.str.startswith(f"client_b_{batch}")]
                    means = test.loc[samples, ["feature_0", "feature_3"]].mean()
                    np.testing.assert_allclose(
                        means, level.loc[[feature_hash("feature_0"), feature_hash("feature_3")]])
                # the training data is fedRBE's corrected data
                train_mean = pd.concat(tables["train"].values())["feature_3"].mean()
                np.testing.assert_allclose(train_mean, level.loc[feature_hash("feature_3")])

    def test_sparse_held_out_features_stay_uncorrected(self):
        # client_a has 5 values of feature_7, fewer than its min_samples (6)
        tables, _, folders = self.prepare("client_a", common.ARM_LABEL_FREE)
        test = tables["test"]["client_a"]
        raw = self.raw(folders, test)
        pd.testing.assert_series_equal(test["feature_7"], raw["feature_7"], check_names=False)
        self.assertFalse(np.allclose(test["feature_0"], raw["feature_0"]))

    def test_held_out_labels_are_not_used(self):
        tables, _, folders = self.prepare("client_c", common.ARM_LABEL_AWARE)
        (folder,) = folders[(0, "test")]
        design = pd.read_csv(folder / "design.tsv", sep="\t", index_col=0)
        design.assign(label=1 - design.label).to_csv(folder / "design.tsv", sep="\t")
        held_out_idx = self.spec.cohorts.index("client_c")
        training = prepare.fold_spec(self.spec, "client_c")
        training_inputs = [[client for idx, client in enumerate(fed_inputs) if idx != held_out_idx]
                           for fed_inputs in self.inputs]
        split = prepare.fold_split(self.spec, self.inputs, "client_c")
        flipped, _ = self.experiment._prepare(
            common.ARM_LABEL_AWARE, "client_c", training, training_inputs, self.inputs, split,
            folders, self.root / "work_flipped")
        pd.testing.assert_frame_equal(tables["test"]["client_c"], flipped["test"]["client_c"])


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.spec = three_cohorts(self.root)
        self.results = evaluate.ResultFile(self.root / "results.csv")
        self.corrections = self.root / "corrections"
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))

    def experiment(self):
        return loco.ClassificationExperimentLeaveOneCohortOut(
            self.spec, resultfile=self.results, corrections_folder=self.corrections,
            work_folder=self.root / "work")

    def test_parallel_folds_give_the_same_rows_and_corrections(self):
        self.experiment().run_experiment(seeds=[5])
        parallel_results = evaluate.ResultFile(self.root / "parallel.csv")
        parallel = loco.ClassificationExperimentLeaveOneCohortOut(
            self.spec, resultfile=parallel_results,
            corrections_folder=self.root / "parallel_corrections", work_folder=self.root / "work")
        # one job per held-out cohort; spawned workers re-import helper
        with patch.object(sys, "path", [str(COMPONENT), *sys.path]):
            run_units(parallel.units([5]), jobs=3)
        keys = ["predicted_client_name", "data_preprocessing_name", "learning_type",
                "metric_name"]
        sequential_rows = pd.read_csv(self.results.path).sort_values(keys, ignore_index=True)
        parallel_rows = pd.read_csv(parallel_results.path).sort_values(keys, ignore_index=True)
        self.assertEqual(len(parallel_rows), 36)
        pd.testing.assert_frame_equal(parallel_rows, sequential_rows)
        for path in sorted(self.corrections.rglob("*.*")):
            other = self.root / "parallel_corrections" / path.relative_to(self.corrections)
            self.assertEqual(path.read_bytes(), other.read_bytes(), path)
        self.assertEqual(list((self.root / "work").iterdir()), [])

    def test_rows_corrections_and_caching(self):
        with patch.object(correct, "correct_training", wraps=correct.correct_training) as correction:
            self.experiment().run_experiment(seeds=[5, 6])
        # once per fold and arm, not per seed
        self.assertEqual(correction.call_count, 3 * 2)
        rows = pd.read_csv(self.results.path)
        # 3 folds x 2 seeds x 3 preprocessings x 2 learning types x 2 metrics
        self.assertEqual(len(rows), 72)
        self.assertEqual(set(rows.cross_validation_method), {"Leave-One-Cohort-Out"})
        self.assertEqual(set(rows.predicted_client_name), set(self.spec.cohorts))
        self.assertEqual(set(rows.data_preprocessing_name),
                         {"uncorrected", "label_free", "label_aware"})
        self.assertEqual(set(rows.learning_type), {"federated", "centralized"})
        self.assertTrue(np.isfinite(rows.metric_value).all())
        for held_out in self.spec.cohorts:
            for arm in common.ARMS:
                folder = self.corrections / "fixture" / held_out / arm
                self.assertEqual({path.name for path in folder.iterdir()},
                                 {"batch_coefficients.csv", "training_level.csv",
                                  "training_level_info.yml"})
                # readable by batchcorrection/align_new_batch.py
                level, info = read_training_level(str(folder))
                self.assertEqual(info["covariates_used"], arm == common.ARM_LABEL_AWARE)
                self.assertTrue(np.isfinite(level).all() and len(level) > 0)
        self.assertEqual(list((self.root / "work").iterdir()), [])
        with patch.object(correct, "correct_training") as correction:
            self.experiment().run_experiment(seeds=[5, 6])
            correction.assert_not_called()
        pd.testing.assert_frame_equal(pd.read_csv(self.results.path), rows)

    def test_failures_publish_nothing(self):
        def fail(*args, **kwargs):
            raise RuntimeError("forest failed")

        with patch.object(train_forest, "train", side_effect=fail), \
                self.assertRaisesRegex(RuntimeError, "forest failed"):
            self.experiment().run_experiment(seeds=[5])
        self.assertFalse(self.results.path.exists())
        self.assertFalse(self.corrections.exists())
        self.assertEqual(list((self.root / "work").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
