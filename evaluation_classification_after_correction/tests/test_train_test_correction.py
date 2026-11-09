"""Checks of the repeated train/test evaluation whose batch effect correction sees only training data."""

import contextlib
from dataclasses import replace
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

COMPONENT = Path(__file__).resolve().parents[1]
with patch.object(sys, "path", [str(COMPONENT), *sys.path]):
    from helper import helper_run_classification_report_metrics as helper
    from helper import train_test_correction as ttc

FEATURES = [f"feature_{i}" for i in range(8)]
BATCHES = {"client_a": [None], "client_b": ["x", "y"], "client_c": [None], "client_d": [None]}


def write_fixture(root: Path, seed: int = 3):
    """
    Four clients, client_b holds two batches. feature_7 is too sparse in
    client_a, fedRBE corrects it only for the other clients.
    """
    rng = np.random.RandomState(seed)
    federation_dirs, configs = [], {ttc.ARM_LABEL_FREE: [], ttc.ARM_LABEL_AWARE: []}
    for number, (name, batches) in enumerate(BATCHES.items()):
        folder = root / "before" / name
        folder.mkdir(parents=True)
        samples, labels, batch_values, shifts = [], [], [], []
        for b, batch in enumerate(batches):
            for i in range(20):
                samples.append(f"{name}_{batch or 'only'}_{i}")
                labels.append(i % 2)
                batch_values.append(batch or "only")
                shifts.append(3 * number + 2 * b)
        labels = np.array(labels)
        values = rng.normal(size=(len(FEATURES), len(samples))) + np.array(shifts)
        values[:2] += 2 * labels
        if name == "client_a":
            # five finite values: fewer than fedRBE needs (6 or 7) in any training split
            values[7, np.setdiff1d(np.arange(len(samples)), [0, 5, 10, 15, 19])] = np.nan
        pd.DataFrame(values, index=pd.Index(FEATURES, name="rowname"), columns=samples).to_csv(
            folder / "data.tsv", sep="\t")
        pd.DataFrame({"label": labels, "batch": batch_values},
                     index=pd.Index(samples, name="file")).to_csv(folder / "design.tsv", sep="\t")
        base = {"data_filename": "data.tsv", "design_filename": "design.tsv",
                "expression_file_flag": True, "index_col": "rowname", "separator": "\t",
                "design_separator": "\t", "normalizationMethod": None, "smpc": False,
                "batch_col": "batch" if len(batches) > 1 else None, "position": number}
        configs[ttc.ARM_LABEL_FREE].append({**base, "covariates": [], "min_samples": 6})
        configs[ttc.ARM_LABEL_AWARE].append({**base, "covariates": ["label"], "min_samples": 2})
        federation_dirs.append(folder)
    federation = ttc.Federation(None, list(BATCHES), federation_dirs, configs)
    return ttc.DatasetSpec("fixture", "Fixture", "label", [federation])


class SplitTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.spec = write_fixture(self.root)
        self.inputs = ttc.read_inputs(self.spec)

    def test_split_is_stratified_by_label_and_batch_and_deterministic(self):
        split = ttc.split_samples(self.spec, self.inputs, 42)
        pd.testing.assert_frame_equal(split, ttc.split_samples(self.spec, self.inputs, 42))
        self.assertFalse(split.equals(ttc.split_samples(self.spec, self.inputs, 43)))
        design = pd.concat([client.design for client in self.inputs[0]])
        split["batch"] = design.loc[split["sample"], "batch"].to_numpy()
        counts = split.groupby(["cohort", "batch", "label", "part"]).size().unstack()
        # 10 samples per batch and label: 7 train, 3 test
        self.assertTrue((counts["train"] == 7).all() and (counts["test"] == 3).all())

    def test_modalities_share_the_split_and_must_agree(self):
        federation = self.spec.federations[0]
        spec = ttc.DatasetSpec("two", "Two", "label",
                               [replace(federation, modality="one"),
                                replace(federation, modality="other")])
        inputs = [self.inputs[0], self.inputs[0]]
        pd.testing.assert_frame_equal(ttc.split_samples(spec, inputs, 7),
                                      ttc.split_samples(self.spec, self.inputs, 7))
        changed = [ttc.ClientInput(client.data, client.design.assign(label=1 - client.design.label),
                                   client.keys) for client in self.inputs[0]]
        with self.assertRaisesRegex(ValueError, "labels differ"):
            ttc.split_samples(spec, [self.inputs[0], changed], 7)

    def test_too_small_training_batches_are_rejected(self):
        split = ttc.split_samples(self.spec, self.inputs, 42)
        ttc.check_min_samples(self.spec, self.inputs, split, ttc.ARMS)
        self.spec.federations[0].configs[ttc.ARM_LABEL_FREE][1]["min_samples"] = 15
        with self.assertRaisesRegex(ValueError, "training batch client_b|x has 14 samples"):
            ttc.check_min_samples(self.spec, self.inputs, split, [ttc.ARM_LABEL_FREE])
        ttc.check_min_samples(self.spec, self.inputs, split, [ttc.ARM_LABEL_AWARE])

    def test_batch_design_codes_reference_and_rejects_unknown_batches(self):
        coefficients = pd.DataFrame(np.zeros((1, 2)), columns=["a", "b|x"])
        design = ttc.batch_design(["a", "b|x", "c"], ["s1", "s2", "s3"], coefficients,
                                  {"a", "b|x", "c"})
        np.testing.assert_array_equal(design.to_numpy(), [[1, 0], [0, 1], [-1, -1]])
        with self.assertRaisesRegex(ValueError, "not a training batch"):
            ttc.batch_design(["d"], ["s1"], coefficients, {"a", "b|x", "c"})
        with self.assertRaisesRegex(ValueError, "do not match the training batches"):
            ttc.batch_design(["a"], ["s1"], coefficients, {"a", "b|x", "c", "e"})


class CorrectionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.spec = write_fixture(self.root)
        self.inputs = ttc.read_inputs(self.spec)
        self.split = ttc.split_samples(self.spec, self.inputs, 42)
        self.folders = ttc.write_split_inputs(self.spec, self.inputs, self.split,
                                              self.root / "split")
        self.federation = self.spec.federations[0]
        self.batches = set(ttc.training_batches(self.federation, self.inputs[0], self.split))

    def correct(self, arm, part):
        outputs, coefficients = ttc.run_correction(
            self.federation, arm, self.folders[(0, "train")],
            Path(tempfile.mkdtemp(dir=self.root)) / "fedrbe")
        corrected = {}
        for idx, (cohort, output, folder) in enumerate(zip(
                self.federation.cohorts, outputs, self.folders[(0, part)])):
            config = self.federation.configs[arm][idx]
            training = ttc.read_corrected_data(output, config["separator"])
            sample_batches = ttc.sample_batches(cohort, self.inputs[0][idx],
                                                config.get("batch_col"), training.columns)
            client = ttc.load_client(folder, cohort, config)
            corrected[cohort] = (training, ttc.correct_new_samples(
                client, coefficients, self.batches, training, sample_batches))
        return coefficients, corrected

    def test_coefficients_reproduce_the_correction_of_the_training_samples(self):
        for arm in ttc.ARMS:
            with self.subTest(arm=arm):
                coefficients, corrected = self.correct(arm, "train")
                self.assertEqual(list(coefficients.columns),
                                 ["client_a", "client_b|x", "client_b|y", "client_c"])
                for cohort, (training, recomputed) in corrected.items():
                    recomputed.index = recomputed.index.astype(str)
                    self.assertTrue(recomputed.drop(index=training.index, errors="ignore")
                                    .isna().all().all())
                    pd.testing.assert_frame_equal(
                        recomputed.loc[training.index, training.columns], training,
                        check_names=False, rtol=0, atol=1e-12)
                # feature_7 is too sparse in client_a: not corrected there, but elsewhere
                self.assertTrue(corrected["client_a"][0].loc["feature_7"].isna().all())
                self.assertTrue(corrected["client_c"][0].loc["feature_7"].notna().all())

    def test_test_samples_are_corrected_without_their_labels(self):
        _, corrected = self.correct(ttc.ARM_LABEL_AWARE, "test")
        for folder in self.folders[(0, "test")]:
            design = pd.read_csv(folder / "design.tsv", sep="\t", index_col=0)
            design.assign(label=-design.label).to_csv(folder / "design.tsv", sep="\t")
        _, shuffled = self.correct(ttc.ARM_LABEL_AWARE, "test")
        for cohort in corrected:
            pd.testing.assert_frame_equal(corrected[cohort][1], shuffled[cohort][1])
            self.assertTrue(corrected[cohort][1].notna().any().all())
        # client_a has feature_7 values in test samples, they are not corrected
        test_input = pd.read_csv(self.folders[(0, "test")][0] / "data.tsv", sep="\t", index_col=0)
        self.assertTrue(test_input.loc["feature_7"].notna().any())
        self.assertTrue(corrected["client_a"][1].loc["feature_7"].isna().all())
        self.assertTrue(corrected["client_c"][1].loc["feature_7"].notna().all())


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.spec = write_fixture(self.root)
        self.results = helper.ResultFile(self.root / "results.csv")
        self.splits = self.root / "splits"
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))

    def experiment(self, arms=ttc.ARMS):
        return ttc.ClassificationExperimentTrainTestSplit(
            self.spec, arms=arms, resultfile=self.results, splits_folder=self.splits,
            work_folder=self.root / "work")

    def test_rows_splits_and_caching(self):
        self.experiment().run_experiment(5)
        rows = pd.read_csv(self.results.path)
        # 3 preprocessings x (federated All + 4 cohorts + centralized All) x 2 metrics
        self.assertEqual(len(rows), 36)
        self.assertEqual(set(rows.data_preprocessing_name),
                         {"uncorrected", "label_free", "label_aware"})
        self.assertEqual(set(rows.cross_validation_method), {"Train-Test-Split"})
        self.assertEqual(set(rows.seed), {5})
        self.assertTrue(np.isfinite(rows.metric_value).all())
        centralized = rows[rows.learning_type == "centralized"]
        self.assertEqual(set(centralized.predicted_client_name), {"All"})
        seed_folder = self.splits / "fixture" / "seed_5"
        split = pd.read_csv(seed_folder / "split.tsv", sep="\t")
        self.assertEqual(set(split.part), {"train", "test"})
        for arm in ttc.ARMS:
            coefficients = pd.read_csv(seed_folder / arm / "batch_coefficients.csv", index_col=0)
            self.assertEqual(coefficients.shape[1], 4)
        self.assertEqual(list((self.root / "work").iterdir()), [])
        with patch.object(ttc, "run_correction") as correction:
            self.experiment().run_experiment(5)
            correction.assert_not_called()
        pd.testing.assert_frame_equal(pd.read_csv(self.results.path), rows)

    def test_failures_publish_nothing(self):
        def fail(*args, **kwargs):
            raise RuntimeError("forest failed")

        with patch.object(ttc, "train_forest", side_effect=fail), \
                self.assertRaisesRegex(RuntimeError, "forest failed"):
            self.experiment().run_experiment(5)
        self.assertFalse(self.results.path.exists())
        self.assertFalse(self.splits.exists())
        self.assertEqual(list((self.root / "work").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
