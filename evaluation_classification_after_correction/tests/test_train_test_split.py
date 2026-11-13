"""Checks of the repeated train/test evaluation whose batch effect correction sees only training data."""

import contextlib
from dataclasses import replace
import hashlib
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
    from helper import common, train_test_split
    from helper.experiment import run_units
    from helper.steps import correct, datasets, evaluate, prepare, train_forest


def _has_rscript() -> bool:
    try:
        correct.rscript()
    except RuntimeError:
        return False
    return True


# the central correction of the centralized forest runs limma in R
requires_r = unittest.skipUnless(_has_rscript(), "Rscript with limma is unavailable")

FEATURES = [f"feature_{i}" for i in range(9)]
BATCHES = {"client_a": [None], "client_b": ["x", "y"], "client_c": [None], "client_d": [None]}


def write_fixture(root: Path, seed: int = 3):
    """
    Four clients, client_b holds two batches. feature_7 is too sparse in
    client_a, fedRBE corrects it only for the other clients. feature_8 only
    exists in client_a and client_b, too few clients for fedRBE to correct it.
    """
    rng = np.random.RandomState(seed)
    federation_dirs, configs = [], {common.ARM_LABEL_FREE: [], common.ARM_LABEL_AWARE: []}
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
        if name in ("client_c", "client_d"):
            values[8] = np.nan
        pd.DataFrame(values, index=pd.Index(FEATURES, name="rowname"), columns=samples).to_csv(
            folder / "data.tsv", sep="\t")
        pd.DataFrame({"label": labels, "batch": batch_values},
                     index=pd.Index(samples, name="file")).to_csv(folder / "design.tsv", sep="\t")
        base = {"data_filename": "data.tsv", "design_filename": "design.tsv",
                "expression_file_flag": True, "index_col": "rowname", "separator": "\t",
                "design_separator": "\t", "normalizationMethod": None, "smpc": False,
                "batch_col": "batch" if len(batches) > 1 else None, "position": number}
        configs[common.ARM_LABEL_FREE].append({**base, "covariates": [], "min_samples": 6})
        configs[common.ARM_LABEL_AWARE].append({**base, "covariates": ["label"], "min_samples": 2})
        federation_dirs.append(folder)
    federation = datasets.Federation(None, list(BATCHES), federation_dirs, configs)
    return datasets.DatasetSpec("fixture", "Fixture", "label", [federation])


class SplitTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.spec = write_fixture(self.root)
        self.inputs = datasets.read_inputs(self.spec)

    def test_split_is_stratified_by_label_and_batch_and_deterministic(self):
        split = prepare.split_samples(self.spec, self.inputs, 42)
        pd.testing.assert_frame_equal(split, prepare.split_samples(self.spec, self.inputs, 42))
        self.assertFalse(split.equals(prepare.split_samples(self.spec, self.inputs, 43)))
        design = pd.concat([client.design for client in self.inputs[0]])
        split["batch"] = design.loc[split["sample"], "batch"].to_numpy()
        counts = split.groupby(["cohort", "batch", "label", "part"]).size().unstack()
        # 10 samples per batch and label: 7 train, 3 test
        self.assertTrue((counts["train"] == 7).all() and (counts["test"] == 3).all())

    def test_modalities_share_the_split_and_must_agree(self):
        federation = self.spec.federations[0]
        spec = datasets.DatasetSpec("two", "Two", "label",
                               [replace(federation, modality="one"),
                                replace(federation, modality="other")])
        inputs = [self.inputs[0], self.inputs[0]]
        pd.testing.assert_frame_equal(prepare.split_samples(spec, inputs, 7),
                                      prepare.split_samples(self.spec, self.inputs, 7))
        changed = [datasets.ClientInput(client.data, client.design.assign(label=1 - client.design.label),
                                   client.keys) for client in self.inputs[0]]
        with self.assertRaisesRegex(ValueError, "labels differ"):
            prepare.split_samples(spec, [self.inputs[0], changed], 7)

    def test_too_small_training_batches_are_rejected(self):
        split = prepare.split_samples(self.spec, self.inputs, 42)
        prepare.check_min_samples(self.spec, self.inputs, split, common.ARMS)
        self.spec.federations[0].configs[common.ARM_LABEL_FREE][1]["min_samples"] = 15
        with self.assertRaisesRegex(ValueError, "training batch client_b|x has 14 samples"):
            prepare.check_min_samples(self.spec, self.inputs, split, [common.ARM_LABEL_FREE])
        prepare.check_min_samples(self.spec, self.inputs, split, [common.ARM_LABEL_AWARE])

    def test_batch_design_codes_reference_and_rejects_unknown_batches(self):
        coefficients = pd.DataFrame(np.zeros((1, 2)), columns=["a", "b|x"])
        design = correct.batch_design(["a", "b|x", "c"], ["s1", "s2", "s3"], coefficients,
                                  {"a", "b|x", "c"})
        np.testing.assert_array_equal(design.to_numpy(), [[1, 0], [0, 1], [-1, -1]])
        with self.assertRaisesRegex(ValueError, "not a training batch"):
            correct.batch_design(["d"], ["s1"], coefficients, {"a", "b|x", "c"})
        with self.assertRaisesRegex(ValueError, "do not match the training batches"):
            correct.batch_design(["a"], ["s1"], coefficients, {"a", "b|x", "c", "e"})


class CorrectionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.spec = write_fixture(self.root)
        self.inputs = datasets.read_inputs(self.spec)
        self.split = prepare.split_samples(self.spec, self.inputs, 42)
        self.folders = prepare.write_split_inputs(self.spec, self.inputs, self.split,
                                              self.root / "split")
        self.federation = self.spec.federations[0]
        self.batches = set(prepare.training_batches(self.federation, self.inputs[0], self.split))

    @requires_r
    def test_central_limma_matches_fedrbe_and_also_corrects_sparse_values(self):
        complete = [f"feature_{i}" for i in range(7)]
        hashes = [hashlib.sha3_256(name.encode()).hexdigest() for name in complete]
        for arm in common.ARMS:
            with self.subTest(arm=arm):
                work = Path(tempfile.mkdtemp(dir=self.root))
                arguments = (self.federation, self.inputs[0], arm, self.folders[(0, "train")],
                             self.split, work)
                fedrbe = correct.correct_training(*arguments)
                central = correct.correct_training_central(*arguments)
                self.assertEqual(central.batches, fedrbe.batches)
                self.assertEqual(list(central.coefficients.columns),
                                 list(fedrbe.coefficients.columns))
                np.testing.assert_allclose(central.coefficients.loc[hashes],
                                           fedrbe.coefficients.loc[hashes], rtol=0, atol=1e-10)
                np.testing.assert_allclose(central.level.loc[hashes], fedrbe.level.loc[hashes],
                                           rtol=0, atol=1e-10)
                for cohort_idx in range(len(self.federation.cohorts)):
                    pd.testing.assert_frame_equal(
                        central.matrices[cohort_idx].loc[complete],
                        fedrbe.matrices[cohort_idx].loc[complete], rtol=0, atol=1e-10,
                        check_names=False)
                # feature_7 is too sparse in client_a for fedRBE: uncorrected there,
                # while central limma corrects it
                raw = self.raw("train", 0).loc["feature_7"]
                pd.testing.assert_series_equal(fedrbe.matrices[0].loc["feature_7"], raw,
                                               check_names=False)
                corrected = central.matrices[0].loc["feature_7"].dropna()
                self.assertFalse(np.allclose(corrected, raw.loc[corrected.index]))

    def correct(self, arm, part):
        """Per cohort fedRBE's corrected training data, the forest's training data and part."""
        outputs, coefficients = correct.run_correction(
            self.federation, arm, self.folders[(0, "train")],
            Path(tempfile.mkdtemp(dir=self.root)) / "fedrbe")
        corrected = {}
        for idx, (cohort, output) in enumerate(zip(self.federation.cohorts, outputs)):
            config = self.federation.configs[arm][idx]
            fedrbe_training = correct.read_corrected_data(output, config["separator"])
            training = correct.training_matrix(fedrbe_training, prepare.load_client(
                self.folders[(0, "train")][idx], cohort, config))
            sample_batches = correct.sample_batches(cohort, self.inputs[0][idx],
                                                config.get("batch_col"), fedrbe_training.columns)
            client = prepare.load_client(self.folders[(0, part)][idx], cohort, config)
            corrected[cohort] = (fedrbe_training, training, correct.correct_new_samples(
                client, coefficients, self.batches, fedrbe_training, sample_batches))
        return coefficients, corrected

    def raw(self, part, cohort_idx):
        return pd.read_csv(self.folders[(0, part)][cohort_idx] / "data.tsv", sep="\t",
                           index_col=0)

    def test_coefficients_reproduce_the_correction_of_the_training_samples(self):
        for arm in common.ARMS:
            with self.subTest(arm=arm):
                coefficients, corrected = self.correct(arm, "train")
                self.assertEqual(list(coefficients.columns),
                                 ["client_a", "client_b|x", "client_b|y", "client_c"])
                for idx, (cohort, (fedrbe, training, recomputed)) in enumerate(corrected.items()):
                    raw = self.raw("train", idx).dropna(how="all")
                    # corrected where fedRBE corrected, uncorrected elsewhere
                    self.assertEqual(set(recomputed.index), set(raw.index))
                    expected = fedrbe.combine_first(raw).loc[recomputed.index, recomputed.columns]
                    pd.testing.assert_frame_equal(recomputed, expected, check_names=False,
                                                  rtol=0, atol=1e-12)
                    # applied to the training samples, the test rule gives the training data
                    pd.testing.assert_frame_equal(
                        recomputed, training.loc[recomputed.index, recomputed.columns],
                        check_names=False, rtol=0, atol=1e-12)
                # feature_7 is too sparse in client_a: uncorrected there, corrected elsewhere
                self.assertTrue(corrected["client_a"][0].loc["feature_7"].isna().all())
                pd.testing.assert_series_equal(corrected["client_a"][1].loc["feature_7"],
                                               self.raw("train", 0).loc["feature_7"],
                                               check_names=False)
                self.assertTrue(corrected["client_c"][0].loc["feature_7"].notna().all())
                # feature_8 is not corrected at all, but kept
                self.assertNotIn("feature_8", corrected["client_a"][0].index)
                pd.testing.assert_series_equal(corrected["client_a"][1].loc["feature_8"],
                                               self.raw("train", 0).loc["feature_8"],
                                               check_names=False)

    def test_test_samples_are_corrected_without_their_labels(self):
        _, corrected = self.correct(common.ARM_LABEL_AWARE, "test")
        for folder in self.folders[(0, "test")]:
            design = pd.read_csv(folder / "design.tsv", sep="\t", index_col=0)
            design.assign(label=-design.label).to_csv(folder / "design.tsv", sep="\t")
        _, shuffled = self.correct(common.ARM_LABEL_AWARE, "test")
        for idx, cohort in enumerate(corrected):
            pd.testing.assert_frame_equal(corrected[cohort][2], shuffled[cohort][2])
            raw = self.raw("test", idx).dropna(how="all")
            self.assertEqual(set(corrected[cohort][2].index), set(raw.index))
            # corrected features differ from the raw values
            self.assertFalse(np.allclose(corrected[cohort][2].loc["feature_0"],
                                         raw.loc["feature_0"]))
        # client_a's test values of feature_7 and feature_8 stay uncorrected
        raw = self.raw("test", 0)
        self.assertTrue(raw.loc["feature_7"].notna().any())
        for feature in ("feature_7", "feature_8"):
            pd.testing.assert_series_equal(corrected["client_a"][2].loc[feature],
                                           raw.loc[feature], check_names=False)
        # with feature_7 missing in client_a, fedRBE's coefficient for client_c is 0
        self.assertFalse(np.allclose(corrected["client_b"][2].loc["feature_7"],
                                     self.raw("test", 1).loc["feature_7"]))


@requires_r
class ExperimentTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.spec = write_fixture(self.root)
        self.results = evaluate.ResultFile(self.root / "results.csv")
        self.splits = self.root / "splits"
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))

    def experiment(self, arms=common.ARMS):
        return train_test_split.ClassificationExperimentTrainTestSplit(
            self.spec, arms=arms, resultfile=self.results, splits_folder=self.splits,
            work_folder=self.root / "work")

    def test_rows_splits_and_caching(self):
        self.experiment().run_experiment(5)
        rows = pd.read_csv(self.results.path)
        # 3 preprocessings x 2 learning types x (All + 4 cohorts) x 2 metrics
        self.assertEqual(len(rows), 60)
        self.assertEqual(set(rows.data_preprocessing_name),
                         {"uncorrected", "label_free", "label_aware"})
        self.assertEqual(set(rows.cross_validation_method), {"Train-Test-Split"})
        self.assertEqual(set(rows.seed), {5})
        self.assertTrue(np.isfinite(rows.metric_value).all())
        for learning_type in ("centralized", "federated"):
            scored = rows[rows.learning_type == learning_type]
            self.assertEqual(set(scored.predicted_client_name), {"All", *self.spec.cohorts})
        seed_folder = self.splits / "fixture" / "seed_5"
        split = pd.read_csv(seed_folder / "split.tsv", sep="\t")
        self.assertEqual(set(split.part), {"train", "test"})
        for arm in common.ARMS:
            coefficients = pd.read_csv(seed_folder / arm / "batch_coefficients.csv", index_col=0)
            self.assertEqual(coefficients.shape[1], 4)
        self.assertEqual(list((self.root / "work").iterdir()), [])
        with patch.object(correct, "run_correction") as correction:
            self.experiment().run_experiment(5)
            correction.assert_not_called()
        pd.testing.assert_frame_equal(pd.read_csv(self.results.path), rows)

    def test_parallel_seeds_give_the_same_rows_and_splits(self):
        run_units(self.experiment().units([5, 6]), jobs=1)
        parallel_results = evaluate.ResultFile(self.root / "parallel.csv")
        parallel = train_test_split.ClassificationExperimentTrainTestSplit(
            self.spec, resultfile=parallel_results, splits_folder=self.root / "parallel_splits",
            work_folder=self.root / "work")
        # spawned workers re-import helper from the component folder
        with patch.object(sys, "path", [str(COMPONENT), *sys.path]):
            run_units(parallel.units([5, 6]), jobs=2)
        keys = ["seed", "data_preprocessing_name", "learning_type", "predicted_client_name",
                "metric_name"]
        sequential_rows = pd.read_csv(self.results.path).sort_values(keys, ignore_index=True)
        parallel_rows = pd.read_csv(parallel_results.path).sort_values(keys, ignore_index=True)
        self.assertEqual(len(parallel_rows), 120)
        pd.testing.assert_frame_equal(parallel_rows, sequential_rows)
        for path in sorted(self.splits.rglob("*.*")):
            other = self.root / "parallel_splits" / path.relative_to(self.splits)
            self.assertEqual(path.read_bytes(), other.read_bytes(), path)
        self.assertEqual(list((self.root / "work").iterdir()), [])

    def test_failures_publish_nothing(self):
        def fail(*args, **kwargs):
            raise RuntimeError("forest failed")

        with patch.object(train_forest, "train", side_effect=fail), \
                self.assertRaisesRegex(RuntimeError, "forest failed"):
            self.experiment().run_experiment(5)
        self.assertFalse(self.results.path.exists())
        self.assertFalse(self.splits.exists())
        self.assertEqual(list((self.root / "work").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
