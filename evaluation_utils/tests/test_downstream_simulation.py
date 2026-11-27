"""Scientific invariants, metrics and paired summaries for downstream simulations."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import contextlib
import fcntl
import io
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from evaluation_classification_after_correction.fc_fed_forest_simple_app import fedForestSimple
from evaluation_data.simulated import downstream_simulation as downstream
from evaluation_data.simulated import training_correction
from evaluation_data.simulated.downstream_simulation import (
    ForestPredictor, GROUP, _aggregate, _paired, _validate, cluster, main,
    positive_integers, predict_forest,
    score_assignments, summarize,
)


WORKER_BARRIER = None


def both(matrix):
    """The same matrix for the central and the federated learner."""
    return dict.fromkeys(("central", "federated"), matrix)


def initialize_worker(barrier):
    global WORKER_BARRIER
    WORKER_BARRIER = barrier


def fit_isolated(payload):
    """Importable worker entry point: exercise both adapters in a fresh process."""
    if WORKER_BARRIER is not None:
        WORKER_BARRIER.wait(timeout=30)
    matrix, labs, labels = payload
    clustering = cluster(both(matrix), labs, 11)
    forest = predict_forest(matrix, labels, labs, "lab3", 11, "federated")
    return os.getpid(), {
        "central_clusters": clustering["central"]["labels"],
        "federated_clusters": clustering["federated"]["labels"],
        "predictions": forest["predictions"],
        "probability_b": forest["probability_b"],
    }


class DownstreamLearnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        samples = pd.Index([f"s{i}" for i in range(36)])
        cls.labs = pd.Series(np.repeat(["lab1", "lab2", "lab3"], 12), index=samples)
        cls.labels = pd.Series(np.tile(["A", "B"], 18), index=samples)
        rng = np.random.RandomState(102)
        cls.matrix = pd.DataFrame(rng.normal(size=(8, 36)),
                                  index=[f"f{i}" for i in range(8)], columns=samples)
        cls.matrix.loc["f0"] += 3 * cls.labels.eq("B")

    def test_cluster_reproducible_and_feature_order_invariant(self):
        original = cluster(both(self.matrix), self.labs, 11)
        repeated = cluster(both(self.matrix.iloc[::-1]), self.labs, 11)
        for learner in ("central", "federated"):
            pd.testing.assert_series_equal(original[learner]["labels"],
                                           repeated[learner]["labels"])
            self.assertTrue(original[learner]["labels"].index.equals(self.matrix.columns))
        self.assertEqual(original["federated"]["diagnostics"]["coordinator_aggregations"], 1)

    def test_forest_never_reads_held_out_labels(self):
        hidden = self.labels.copy()
        hidden.loc[self.labs.eq("lab3")] = "unavailable"
        for learner in ("central", "federated"):
            original = predict_forest(self.matrix, self.labels, self.labs, "lab3", 11, learner)
            repeated = predict_forest(self.matrix.iloc[::-1], hidden,
                                      self.labs, "lab3", 11, learner)
            pd.testing.assert_series_equal(original["predictions"], repeated["predictions"])
            pd.testing.assert_series_equal(original["probability_b"], repeated["probability_b"],
                                           check_exact=True)
            self.assertEqual(original["diagnostics"]["tree_fingerprints"],
                             repeated["diagnostics"]["tree_fingerprints"])
            self.assertEqual(original["diagnostics"]["n_train"], 24)
            self.assertEqual(len(original["predictions"]), 12)

    def test_cached_forests_exactly_match_native_app_across_folds_and_seeds(self):
        for seed in (11, 12):
            predictor = ForestPredictor(self.matrix.iloc[::-1], self.labels, self.labs, seed)
            for learner in ("central", "federated"):
                for held_out in sorted(self.labs.unique()):
                    with self.subTest(seed=seed, learner=learner, held_out=held_out):
                        native = predict_forest(self.matrix, self.labels, self.labs,
                                                held_out, seed, learner, native_io=True)
                        cached = predictor.predict(held_out, learner)
                        for key in ("predictions", "probability_b"):
                            pd.testing.assert_series_equal(native[key], cached[key],
                                                           check_exact=True)
                        self.assertEqual(native["diagnostics"]["tree_fingerprints"],
                                         cached["diagnostics"]["tree_fingerprints"])
                        self.assertEqual(len(cached["diagnostics"]["tree_fingerprints"]), 100)
                        expected_seeds = np.random.RandomState(seed).randint(
                            np.iinfo(np.int32).max,
                            size=100 if learner == "central" else 50).tolist()
                        for result in (native, cached):
                            self.assertEqual(result["diagnostics"]["tree_seeds_per_client"],
                                             expected_seeds)
                            self.assertNotIn("train_test_ratio",
                                             result["diagnostics"]["config"]["simple_forest"])

    def test_federated_reuses_each_local_forest_but_central_refits_each_fold(self):
        with patch.object(fedForestSimple, "train_local_trees",
                          wraps=fedForestSimple.train_local_trees) as fit:
            predictor = ForestPredictor(self.matrix, self.labels, self.labs, 11)
            for held_out in sorted(self.labs.unique()):
                predictor.predict(held_out, "federated")
            self.assertEqual(fit.call_count, 3)
            self.assertEqual([call.args[2] for call in fit.call_args_list], [50] * 3)
            fit.reset_mock()
            for held_out in sorted(self.labs.unique()):
                predictor.predict(held_out, "central")
            self.assertEqual(fit.call_count, 3)
            self.assertEqual([call.args[2] for call in fit.call_args_list], [100] * 3)

    def test_optimized_learners_do_not_start_external_processes(self):
        with patch("subprocess.run", side_effect=AssertionError("Unexpected external process")), \
                patch("subprocess.Popen", side_effect=AssertionError("Unexpected process")):
            cluster(both(self.matrix), self.labs, 11)
            predictor = ForestPredictor(self.matrix, self.labels, self.labs, 11)
            for learner in ("central", "federated"):
                predictor.predict("lab3", learner)

    def test_rejects_alignment_and_unsupported_inputs(self):
        with self.assertRaisesRegex(ValueError, "exactly match"):
            cluster(both(self.matrix), self.labs.iloc[::-1], 11)
        with self.assertRaisesRegex(ValueError, "positive learner seed"):
            cluster(both(self.matrix), self.labs, 0)
        incomplete = self.matrix.copy()
        incomplete.iloc[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "complete and finite"):
            cluster(both(incomplete), self.labs, 11)
        one_class = self.labels.copy()
        one_class.loc[self.labs.eq("lab1")] = "A"
        with self.assertRaisesRegex(ValueError, "both A and B"):
            predict_forest(self.matrix, one_class, self.labs, "lab3", 11, "federated")

    def test_spawned_workers_match_sequential_adapters(self):
        payload = (self.matrix, self.labs, self.labels)
        _, expected = fit_isolated(payload)
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=2, mp_context=context,
                                 initializer=initialize_worker,
                                 initargs=(context.Barrier(2),)) as pool:
            results = list(pool.map(fit_isolated, [payload, payload]))
        self.assertEqual(len({pid for pid, _ in results}), 2)
        for _, result in results:
            for name, values in expected.items():
                pd.testing.assert_series_equal(result[name], values, check_exact=True)

class MissingValueForestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        samples = pd.Index([f"s{i:02d}" for i in range(60)])
        cls.labs = pd.Series(np.repeat([f"lab{i}" for i in range(1, 6)], 12),
                             index=samples)
        cls.labels = pd.Series(np.tile(["A", "B"], 30), index=samples)
        values = np.random.default_rng(29).normal(size=(60, 4))
        values[:, 0] += 2 * cls.labels.eq("B").to_numpy()
        values[cls.labs.isin(["lab2", "lab5"]), 1] = np.nan
        values[cls.labs.isin(["lab1", "lab3"]), 2] = np.nan
        values[::3, 3] = np.nan
        cls.matrix = pd.DataFrame(values.T, index=[f"f{i}" for i in range(4)],
                                  columns=samples)

    def test_native_and_core_match_with_local_missing_columns_and_all_na_sample(self):
        matrix = self.matrix.copy()
        matrix.loc[:, matrix.columns[-1]] = np.nan
        hidden = self.labels.copy()
        hidden.loc[self.labs.eq("lab5")] = "unavailable"
        for learner in ("central", "federated"):
            with self.subTest(learner=learner):
                native = predict_forest(matrix, hidden, self.labs, "lab5", 101,
                                        learner, native_io=True)
                core = predict_forest(matrix.iloc[::-1], hidden, self.labs,
                                      "lab5", 101, learner)
                for name in ("predictions", "probability_b"):
                    pd.testing.assert_series_equal(native[name], core[name], check_exact=True)
                self.assertTrue(np.isfinite(core["probability_b"]).all())
                self.assertTrue(core["probability_b"].between(0, 1).all())
                self.assertEqual(native["diagnostics"]["tree_fingerprints"],
                                 core["diagnostics"]["tree_fingerprints"])
                self.assertEqual(core["diagnostics"]["n_features"], 4)
                self.assertEqual(core["diagnostics"]["num_trees"], 100)

    def test_within_lab_split_matches_native_app_with_local_missing_columns(self):
        split = downstream.within_lab_split(self.labels, self.labs, 101)
        for learner in ("central", "federated"):
            with self.subTest(learner=learner):
                native = predict_forest(self.matrix, self.labels, self.labs, None, 101,
                                        learner, native_io=True, split=split)
                core = predict_forest(self.matrix, self.labels, self.labs, None, 101,
                                      learner, split=split)
                for name in ("predictions", "probability_b"):
                    pd.testing.assert_series_equal(native[name], core[name], check_exact=True)
                self.assertTrue(core["predictions"].index.equals(split["test"]))
                self.assertEqual(core["diagnostics"]["num_trees"], 100)

    def test_invalid_rf_inputs_do_not_silently_change_feature_panel(self):
        for replacement in (np.inf, -np.inf):
            matrix = self.matrix.copy()
            matrix.iloc[0, 0] = replacement
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                ForestPredictor(matrix, self.labels, self.labs, 101)
        matrix = self.matrix.copy()
        matrix.loc["f0"] = np.nan
        with self.assertRaises(ValueError):
            ForestPredictor(matrix, self.labels, self.labs, 101)
        labels = self.labels.copy()
        labels.iloc[0] = "unknown"
        with self.assertRaises(ValueError):
            predict_forest(self.matrix, labels, self.labs, "lab5", 101, "federated")

    def test_input_loader_preserves_matched_na_and_rejects_mask_or_order_changes(self):
        samples = pd.Index([f"s{i:03d}" for i in range(600)])
        metadata = pd.DataFrame({"file": samples, "lab": np.repeat(
            [f"batch{i}" for i in range(1, 6)], 120), "condition": ["A", "B"] * 300})
        matrix = pd.DataFrame(np.ones((3, 600)), index=["f0", "f1", "f2"], columns=samples)
        matrix.iloc[1, :120] = np.nan
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            metadata.to_csv(directory / "metadata.tsv", sep="\t", index=False)
            pd.DataFrame({"feature": matrix.index}).to_csv(
                directory / "features.tsv", sep="\t", index=False)
            imputed = matrix.fillna(0.5).iloc[1:]
            for route, states in downstream.ROUTES.items():
                for state in states if route != "drop" else ():
                    (matrix if route == "keep" else imputed).to_csv(
                        directory / f"{state}.tsv.gz", sep="\t")
            # The central learners read their own correction output.
            for state, source in (("no_label", matrix), ("impute_no_label", imputed)):
                (source + 1).to_csv(directory / f"{state}_central.tsv.gz", sep="\t")
            _, loaded = downstream.load_inputs(directory, "missingness_overlap")
            self.assertEqual(set(loaded), set(downstream.ROUTE_STATES))
            for state, by_learner in loaded.items():
                expected = {"keep": matrix, "drop": matrix.drop(index="f1"),
                            "impute": imputed}[downstream.STATE_ROUTE[state]]
                central = expected + 1 if state.endswith("no_label") else expected
                pd.testing.assert_frame_equal(by_learner["federated"], expected)
                pd.testing.assert_frame_equal(by_learner["central"], central)
            metadata_frame = metadata.set_index("file")
            with contextlib.redirect_stdout(io.StringIO()):
                metrics, _, _ = downstream.evaluate_inputs(metadata_frame, loaded, [101], False,
                                                           tasks=("rf",))
            self.assertEqual(set(metrics.state), set(downstream.ROUTE_STATES))
            self.assertEqual(metrics.groupby("state").n_features.max().to_dict(), {
                **dict.fromkeys(downstream.ROUTES["keep"], 3),
                **dict.fromkeys(downstream.ROUTES["drop"], 2),
                **dict.fromkeys(downstream.ROUTES["impute"], 2)})
            for name in ("drop_dcombat", "drop_dcombat_central"):
                matrix.drop(index="f1").to_csv(directory / f"{name}.tsv.gz", sep="\t")
            for name in ("impute_dcombat", "impute_dcombat_central"):
                imputed.to_csv(directory / f"{name}.tsv.gz", sep="\t")
            _, selected = downstream.load_inputs(directory, "missingness_overlap",
                                                 list(downstream.DCOMBAT_STATES))
            self.assertEqual(list(selected), list(downstream.DCOMBAT_STATES))
            pd.testing.assert_frame_equal(selected["drop_dcombat"]["federated"],
                                          matrix.drop(index="f1"))
            matrix.fillna(0.5).to_csv(directory / "drop_dcombat.tsv.gz", sep="\t")
            with self.assertRaisesRegex(ValueError, "order mismatch in drop_dcombat"):
                downstream.load_inputs(directory, "missingness_overlap", ["drop_dcombat"])
            incomplete = imputed.copy()
            incomplete.iloc[0, 0] = np.nan
            incomplete.to_csv(directory / "impute_no_label.tsv.gz", sep="\t")
            with self.assertRaises(ValueError):
                downstream.load_inputs(directory, "missingness_overlap")
            imputed.to_csv(directory / "impute_no_label.tsv.gz", sep="\t")
            for corruption in ("mask", "order", "infinity"):
                changed = matrix.copy()
                if corruption == "mask":
                    changed.iloc[1, 0] = 1.0
                elif corruption == "order":
                    changed = changed.iloc[::-1]
                else:
                    changed.iloc[0, 0] = np.inf
                changed.to_csv(directory / "no_label.tsv.gz", sep="\t")
                with self.subTest(corruption=corruption), self.assertRaises(ValueError):
                    downstream.load_inputs(directory, "missingness_overlap")


class WithinLabForestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        DownstreamLearnerTests.setUpClass.__func__(cls)

    def test_stratified_splits_cover_each_lab_without_overlap(self):
        split = downstream.within_lab_split(self.labels, self.labs, 11)
        repeated = downstream.within_lab_split(self.labels, self.labs, 11)
        self.assertTrue(split["test"].equals(repeated["test"]))
        self.assertTrue(split["test"].equals(
            self.labels.index[self.labels.index.isin(split["test"])]))
        self.assertEqual(len(split["test"]), 12)
        for lab, train in split["train"].items():
            test = split["test"][self.labs.loc[split["test"]].eq(lab)]
            self.assertTrue(train.intersection(test).empty)
            self.assertEqual(set(train) | set(test), set(self.labs.index[self.labs.eq(lab)]))
            self.assertEqual(self.labels.loc[test].value_counts().to_dict(), {"A": 2, "B": 2})
            self.assertTrue(train.equals(repeated["train"][lab]))

    def test_within_lab_native_core_agree_and_ignore_test_labels(self):
        split = downstream.within_lab_split(self.labels, self.labs, 11)
        hidden = self.labels.copy()
        hidden.loc[split["test"]] = "unavailable"
        for learner, tree_count in (("central", 100), ("federated", 102)):
            with self.subTest(learner=learner):
                native = predict_forest(self.matrix, self.labels, self.labs,
                                        None, 11, learner, native_io=True, split=split)
                core = predict_forest(self.matrix.iloc[::-1], hidden, self.labs,
                                      None, 11, learner, split=split)
                self.assertTrue(core["predictions"].index.equals(split["test"]))
                for name in ("predictions", "probability_b"):
                    pd.testing.assert_series_equal(native[name], core[name], check_exact=True)
                self.assertEqual(native["diagnostics"]["tree_fingerprints"],
                                 core["diagnostics"]["tree_fingerprints"])
                self.assertEqual(core["diagnostics"]["num_trees"], tree_count)
                self.assertEqual(core["diagnostics"]["n_train"], 24)
                self.assertEqual(core["diagnostics"]["n_test"], 12)

    def test_explicit_split_rejects_test_leakage_and_foreign_samples(self):
        for corruption in ("overlap", "foreign", "wrong_lab"):
            split = downstream.within_lab_split(self.labels, self.labs, 11)
            if corruption == "overlap":
                split["train"]["lab1"] = split["train"]["lab1"].append(split["test"][:1])
            elif corruption == "foreign":
                split["test"] = pd.Index(["unknown_sample", *split["test"][1:]])
            else:
                split["train"]["lab1"], split["train"]["lab2"] = (
                    split["train"]["lab2"], split["train"]["lab1"])
            with self.subTest(corruption=corruption), self.assertRaises(ValueError):
                predict_forest(self.matrix, self.labels, self.labs, None, 11,
                               "federated", split=split)

    def test_evaluation_reuses_split_across_states_and_learners(self):
        split = downstream.within_lab_split(self.labels, self.labs, 11)
        matrices = {state: both(self.matrix.copy()) for state in downstream.STATES}
        metadata = pd.DataFrame({"condition": self.labels, "lab": self.labs})
        with contextlib.redirect_stdout(io.StringIO()):
            metrics, predictions, _ = downstream.evaluate_inputs(
                metadata, matrices, [11], False, rf_protocols=("within_lab_70_30",))
        self.assertEqual(set(metrics.loc[metrics.task.eq("rf"), "protocol"]),
                         {"within_lab_70_30"})
        self.assertEqual(set(metrics.loc[metrics.task.eq("kmeans"), "protocol"]),
                         {"all_samples"})
        rf = predictions.loc[predictions.task.eq("rf")]
        for _, subset in rf.groupby(["state", "learner"]):
            self.assertEqual(subset["sample"].tolist(), split["test"].tolist())


class DownstreamSimulationTests(unittest.TestCase):
    def setUp(self):
        samples = pd.Index([f"s{i}" for i in range(12)])
        self.target = pd.Series(list("AAAABBAABBBB"), index=samples)
        self.labs = pd.Series(["lab1"] * 6 + ["lab2"] * 6, index=samples)
        self.predictions = pd.Series(list("AAAAAAAABBBB"), index=samples)

    def test_pooled_metrics_are_recomputed_from_all_predictions(self):
        rows = pd.DataFrame(score_assignments(
            self.target, self.predictions, self.labs, "rf"))
        scores = rows.pivot(index="lab", columns="metric", values="value")
        # Lab 1 predicts only A, while lab 2 is perfect; pooled confusion is
        # [[6, 0], [2, 4]], so the pooled MCC is not the mean of the lab MCCs.
        self.assertEqual(set(rows.metric), {"MCC"})
        self.assertAlmostEqual(scores.loc["pooled", "MCC"], 1 / np.sqrt(2))
        self.assertAlmostEqual(scores.loc[["lab1", "lab2"], "MCC"].mean(), 0.5)
        pooled = rows.loc[rows.lab.eq("pooled")]
        self.assertTrue(pooled.n_samples.eq(12).all())
        self.assertTrue(pooled.n_a.eq(6).all())
        self.assertTrue(pooled.n_b.eq(6).all())

    def test_cluster_scores_use_global_assignment_subsets(self):
        assignments = self.target.map({"A": 1, "B": 0})
        rows = pd.DataFrame(score_assignments(
            self.target, assignments, self.labs, "kmeans"))
        self.assertEqual(set(rows.metric), {"ARI"})
        self.assertEqual(set(rows.lab), {"pooled", "lab1", "lab2"})
        np.testing.assert_allclose(rows.value, 1.0)

    def test_rejects_misaligned_duplicate_and_missing_predictions(self):
        with self.assertRaisesRegex(ValueError, "match exactly"):
            score_assignments(self.target, self.predictions.iloc[::-1], self.labs, "rf")
        missing = self.predictions.copy()
        missing.iloc[0] = None
        with self.assertRaisesRegex(ValueError, "Missing predictions"):
            score_assignments(self.target, missing, self.labs, "rf")
        duplicated = self.target.index.to_list()
        duplicated[-1] = duplicated[0]
        objects = [value.set_axis(duplicated) for value in
                   (self.target, self.predictions, self.labs)]
        with self.assertRaises(ValueError):
            score_assignments(*objects, "rf")

    def test_rejects_unsupported_targets_tasks_and_cluster_values(self):
        with self.assertRaises(ValueError):
            score_assignments(self.target, self.predictions, self.labs, "unsupported")
        invalid = self.predictions.copy()
        invalid.iloc[0] = "unknown"
        with self.assertRaisesRegex(ValueError, "condition A/B"):
            score_assignments(self.target, invalid, self.labs, "rf")
        constant = self.target.copy()
        constant.loc[self.labs.eq("lab1")] = "A"
        with self.assertRaisesRegex(ValueError, "Both classes"):
            score_assignments(constant, self.predictions, self.labs, "rf")
        infinite = self.target.map({"A": 0.0, "B": 1.0})
        infinite.iloc[0] = np.inf
        with self.assertRaises(ValueError):
            score_assignments(self.target, infinite, self.labs, "kmeans")

    def test_positive_ids_reject_duplicates_and_invalid_values(self):
        self.assertEqual(positive_integers("1,5,10"), [1, 5, 10])
        for value in ("", "0", "-1", "1.5", "1,,2", "1,1", str(2**31)):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                positive_integers(value)

    def test_cli_rejects_unsupported_or_repeated_generation_ids(self):
        common = ["--out", "unused_output", "--reference", "unused_reference"]
        cases = [
            (["--runs", "11"], "1:10"),
            (["--runs", "1,1"], "Repeated seeds/runs"),
            (["--scenarios", "confounded_p50", "confounded_p50"], "Duplicate scenarios"),
            (["--scenarios", "missingness"], "invalid choice"),
            (["--states", "truth", "truth"], "Duplicate states"),
            (["--states", "drop_dcombat"], "d-ComBat states need --dcombat"),
            (["--dcombat", "unused_upstream"], "needs missingness_overlap"),
        ]
        for arguments, message in cases:
            with self.subTest(arguments=arguments):
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
                    main(common + arguments)
                self.assertEqual(error.exception.code, 2)
                self.assertIn(message, stderr.getvalue())

def fixture():
    rows = []
    for scenario, data_seed in (("confounded_p50", 5), ("confounded_p30", 6)):
        for task, metric in (("kmeans", "ARI"), ("rf", "MCC")):
            for state, score in (("truth", .8), ("uncorrected", .4),
                                 ("no_label", .6), ("label_aware", .9)):
                for learner, adjustment in (("central", 0), ("federated", -.1)):
                    for learner_seed, random_effect in ((101, 0), (202, .04)):
                        rows.append(dict(
                            analysis="confounding", panel="full", target=0.0,
                            protocol="all_samples" if task == "kmeans" else "loco",
                            scenario=scenario, run=1, data_seed=data_seed, task=task,
                            metric=metric, state=state, learner=learner, lab="pooled",
                            learner_seed=learner_seed, value=score + adjustment + random_effect,
                            oracle=state == "label_aware", n_samples=10, n_a=5, n_b=5,
                            status="evaluable", n_features=8,
                        ))
    return pd.DataFrame(rows)


class DownstreamSummaryTests(unittest.TestCase):
    def test_protocols_are_separate_pairing_and_uncertainty_units(self):
        original = fixture()
        represented = original.loc[original.task.eq("rf")].assign(
            protocol="within_lab_70_30", value=lambda x: x.value - .3)
        metrics = pd.concat([original, represented], ignore_index=True)
        pairs = _paired(metrics)
        rf_pairs = pairs.loc[pairs.task.eq("rf")]
        for comparison, difference in (("no_label-minus-truth", -.2),
                                        ("no_label-minus-uncorrected", .2),
                                        ("federated-minus-central", -.1)):
            selected = rf_pairs.loc[rf_pairs.comparison.eq(comparison)]
            self.assertEqual(set(selected.protocol), {"loco", "within_lab_70_30"})
            np.testing.assert_allclose(selected.value, difference, atol=1e-15)
        _, summary = _aggregate(metrics, GROUP)
        selected = summary.loc[summary.task.eq("rf") & summary.state.eq("truth") &
                               summary.learner.eq("central")]
        np.testing.assert_allclose(selected.loc[selected.protocol.eq("loco"), "mean"], .82)
        np.testing.assert_allclose(
            selected.loc[selected.protocol.eq("within_lab_70_30"), "mean"], .52)
        self.assertTrue(summary.n_data_runs.eq(1).all())

    def test_missing_scores_are_rejected(self):
        metrics = fixture()
        metrics.loc[0, "value"] = np.nan
        with self.assertRaises(ValueError):
            _validate(metrics, {"confounded_p50": 0, "confounded_p30": .16})

    def test_two_learner_seeds_are_not_two_generated_datasets(self):
        with tempfile.TemporaryDirectory() as directory:
            summarize(fixture(), Path(directory), {"confounded_p50": 0, "confounded_p30": .16})
            summary = pd.read_csv(Path(directory) / "summary.tsv", sep="\t")
            generated = pd.read_csv(Path(directory) / "generated_run_means.tsv", sep="\t")
            self.assertTrue(summary.n_data_runs.eq(1).all())
            self.assertTrue(summary[["mcse", "ci_lower", "ci_upper"]].isna().all().all())
            self.assertTrue(generated.n_learner_seeds.eq(2).all())
            np.testing.assert_allclose(generated.within_run_learner_sd, .04 / np.sqrt(2))
            self.assertEqual({p.name for p in Path(directory).iterdir()}, {
                "metrics.tsv", "generated_run_means.tsv", "summary.tsv", "paired_differences.tsv",
                "paired_generated_run_means.tsv", "paired_summary.tsv", "ceiling_diagnostics.tsv",
                "endpoint_availability.tsv",
            })

    def test_pair_deltas_and_cross_scenario_isolation(self):
        metrics = fixture()
        metrics.loc[metrics.scenario.eq("confounded_p30"), "value"] -= .25
        pairs = _paired(metrics)
        for comparison, difference in (("no_label-minus-truth", -.2),
                                        ("no_label-minus-uncorrected", .2),
                                        ("federated-minus-central", -.1)):
            np.testing.assert_allclose(pairs.loc[pairs.comparison.eq(comparison), "value"],
                                       difference, atol=1e-15)
        self.assertEqual(len(pairs), 128)

    def test_routes_pair_within_and_across_feature_panels(self):
        metrics = fixture().query(
            "scenario == 'confounded_p50' and state != 'label_aware' and task == 'rf'")
        metrics = metrics.assign(analysis="missingness_overlap", scenario="missingness_full_30")
        routes = [metrics]
        for route, features in (("drop", 2), ("impute", 6)):
            states = [state.removeprefix(f"{route}_") for state in downstream.ROUTES[route]]
            routes.append(metrics.loc[metrics.task.eq("rf") & metrics.state.isin(states)].assign(
                state=lambda frame: f"{route}_" + frame.state, n_features=features,
                value=lambda frame: frame.value - .1))
        metrics = _validate(pd.concat(routes, ignore_index=True), {"missingness_full_30": 0})
        pairs = _paired(metrics)
        across = pairs.loc[pairs.comparison.isin(
            ["drop_no_label-minus-no_label", "impute_no_label-minus-no_label"])]
        self.assertEqual(set(across.task), {"rf"})
        np.testing.assert_allclose(across.value, -.1, atol=1e-15)
        np.testing.assert_allclose(pairs.loc[pairs.comparison.eq(
            "drop_no_label-minus-drop_uncorrected"), "value"], .2, atol=1e-15)
        self.assertFalse(pairs.comparison.str.contains("label_aware").any())
        dcombat = metrics.loc[metrics.state.isin(["drop_uncorrected", "impute_uncorrected"])].assign(
            state=lambda frame: frame.state.str.replace("uncorrected", "dcombat"),
            value=lambda frame: frame.value + .3)
        states = [*downstream.ROUTE_STATES, *downstream.DCOMBAT_STATES]
        compared = _paired(_validate(pd.concat([metrics, dcombat], ignore_index=True),
                                     {"missingness_full_30": 0}, states))
        for comparison, difference in (("drop_dcombat-minus-drop_uncorrected", .3),
                                       ("impute_dcombat-minus-impute_no_label", .1),
                                       ("drop_dcombat-minus-no_label", 0),
                                       ("impute_dcombat-minus-no_label", 0)):
            selected = compared.loc[compared.comparison.eq(comparison)]
            self.assertEqual(set(selected.task), {"rf"})
            np.testing.assert_allclose(selected.value, difference, atol=1e-15)
        changed = metrics.copy()
        changed.loc[changed.state.eq("drop_truth"), "n_features"] = 3
        with self.assertRaisesRegex(ValueError, "Feature count mismatch"):
            _paired(changed)
        with self.assertRaisesRegex(ValueError, "input states"):
            _validate(metrics.loc[~metrics.state.eq("impute_no_label")],
                      {"missingness_full_30": 0})

    def test_rejects_missing_pairs_and_duplicate_observations(self):
        metrics = fixture()
        with self.assertRaisesRegex(ValueError, "Missing paired"):
            _paired(metrics.iloc[1:])
        with self.assertRaisesRegex(ValueError, "Duplicate metric"):
            _validate(pd.concat([metrics, metrics.iloc[[0]]]),
                      {"confounded_p50": 0, "confounded_p30": .16})

    def test_uncertainty_uses_between_dataset_means(self):
        first = fixture()
        second = first.assign(run=2, data_seed=first.data_seed + 100, value=first.value + .1)
        _, summary = _aggregate(pd.concat([first, second]), GROUP)
        np.testing.assert_allclose(summary.mcse, .05, atol=1e-15)
        self.assertTrue(summary.n_data_runs.eq(2).all())
        self.assertTrue(summary.ci_upper.gt(summary["mean"]).all())

class DownstreamCheckpointTests(unittest.TestCase):
    def test_fresh_missingness_export_command_identifies_setting_and_saved_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "missingness.tsv"
            reference.write_text("reference\n")
            saved = root / "saved"
            saved.mkdir()
            saved_root = saved / "root008_inputs.rds"
            saved_root.write_bytes(b"fixed root")
            output = root / "output"
            output.mkdir()
            (output / "provenance.json").write_text("{}\n")
            args = argparse.Namespace(out=output, reference=reference,
                analysis="missingness_overlap", rf_protocols=["loco"], saved_inputs=saved,
                tasks=["rf"], correction="joint", learner_seeds=[101], runs=[8],
                repeat_check=False,
                source_sha256={"source": "original"},
                reference_sha256=downstream.sha256(reference),
                states=list(downstream.ROUTE_STATES), dcombat=None)
            validation = dict(data_seed=20269110, eta2_condition=0.0, elapsed_seconds=1.0)
            result = (pd.DataFrame({"learner_seed": [101], "value": [.5]}),
                      pd.DataFrame({"learner_seed": [101], "prediction": ["A"]}), [])

            def export(command, **unused):
                options = dict(zip(command[2::2], command[3::2]))
                self.assertEqual(options["--scenario"], "missingness_full_00")
                self.assertEqual(options["--analysis"], "missingness_overlap")
                self.assertEqual(options["--target"], "0")
                self.assertEqual(options["--panel"], "full")
                self.assertEqual(options["--run"], "8")
                self.assertEqual(Path(options["--inputs"]), saved_root)
                destination = Path(options["--out"])
                destination.mkdir()
                (destination / "validation.json").write_text(json.dumps(validation))

            with patch.object(downstream, "source_hashes", return_value=args.source_sha256), \
                    patch.object(downstream, "load_inputs", return_value=(None, None)), \
                    patch.object(downstream, "evaluate_inputs", return_value=result), \
                    patch.object(downstream.subprocess, "run", side_effect=export) as generation, \
                    contextlib.redirect_stdout(io.StringIO()):
                _, timing, _ = downstream.evaluate_dataset_job(("missingness_full_00", 8, args))
                generation.assert_called_once()
            self.assertFalse(timing["inputs_reused"])

    def test_dataset_lock_rejects_overlapping_writer_and_releases_after_close(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            destination = output / "confounded_p50_run001"
            destination.mkdir()
            job = ("confounded_p50", 1, argparse.Namespace(out=output))
            with patch.object(downstream, "_evaluate_dataset_job", return_value="completed") as fit:
                with (destination / ".dataset.lock").open("a") as held_lock:
                    fcntl.flock(held_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    with self.assertRaisesRegex(RuntimeError, "Another worker"):
                        downstream.evaluate_dataset_job(job)
                    fit.assert_not_called()
                self.assertEqual(downstream.evaluate_dataset_job(job), "completed")
                fit.assert_called_once_with(job)

    def test_checksum_rejects_changed_and_missing_checkpoint_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            artifact = directory / "metrics.tsv"
            artifact.write_text("original\n")
            (directory / "predictions.tsv.gz").write_text("predictions\n")
            (directory / "learner_diagnostics.json").write_text("[]\n")
            downstream.commit_checkpoint(directory, {"learner_seed": 101})
            self.assertEqual(downstream.verify_checkpoint(directory)["learner_seed"], 101)
            with self.assertRaisesRegex(ValueError, "identity differs"):
                downstream.verify_checkpoint(directory, {"learner_seed": 211})
            artifact.write_text("changed\n")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                downstream.verify_checkpoint(directory)
            artifact.unlink()
            with self.assertRaises(FileNotFoundError):
                downstream.verify_checkpoint(directory)

    def test_checkpoint_manifest_cannot_omit_required_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            downstream.write_json(directory / "completion.json", {"artifacts_sha256": {}})
            with self.assertRaisesRegex(ValueError, "Incomplete checkpoint"):
                downstream.verify_checkpoint(directory)

    def test_finished_campaign_resume_does_not_fit_and_rejects_mismatched_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "reference.tsv"
            reference.write_text("reference\n")
            output = root / "output"
            args = ["--out", str(output), "--reference", str(reference),
                    "--runs", "1", "--scenarios", "confounded_p50", "--learner-seeds", "101"]
            timing = dict(scenario="confounded_p50", run=1, total_seconds=1.0)

            def summary_files(*unused):
                (output / "summary.tsv").write_text("summary\n")
                (output / "metrics.tsv").write_text("metrics\n")

            with patch.object(downstream, "source_hashes", return_value={"source": "original"}), \
                    patch.object(downstream, "evaluate_dataset_job",
                                 return_value=(fixture(), timing, 0.0)) as fit, \
                    patch.object(downstream, "summarize", side_effect=summary_files), \
                    contextlib.redirect_stdout(io.StringIO()):
                main(args)
                self.assertEqual(fit.call_count, 1)
                fit.reset_mock()
                main(args + ["--resume", "--workers", "2"])
                fit.assert_not_called()
                for changed, message in ((args + ["--learner-seeds", "211"], "learner_seeds"),
                                         (args + ["--rf-protocols", "within_lab_70_30"],
                                          "rf_protocols"),
                                         (args + ["--runs", "2"], "runs")):
                    error = io.StringIO()
                    with contextlib.redirect_stderr(error), self.assertRaises(SystemExit):
                        main(changed + ["--resume"])
                    self.assertIn(message, error.getvalue())
                with patch.object(downstream, "source_hashes", return_value={"source": "new"}):
                    error = io.StringIO()
                    with contextlib.redirect_stderr(error), self.assertRaises(SystemExit):
                        main(args + ["--resume"])
                    self.assertIn("source_sha256", error.getvalue())
                reference.write_text("different reference\n")
                error = io.StringIO()
                with contextlib.redirect_stderr(error), self.assertRaises(SystemExit):
                    main(args + ["--resume"])
                self.assertIn("reference_sha256", error.getvalue())
                fit.assert_not_called()

    def test_failed_second_seed_resumes_first_seed_without_generation_or_refit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "reference.tsv"
            reference.write_text("reference\n")
            output = root / "output"
            inputs = output / "confounded_p50_run001" / "inputs"
            inputs.mkdir(parents=True)
            (output / "provenance.json").write_text("{}\n")
            validation = dict(data_seed=20267017, eta2_condition=0.0, elapsed_seconds=1.0)
            (inputs / "validation.json").write_text(json.dumps(validation))
            args = argparse.Namespace(out=output, reference=reference,
                                      analysis="confounding", rf_protocols=["loco"],
                                      saved_inputs=None, tasks=["kmeans", "rf"],
                                      correction="joint",
                                      learner_seeds=[101, 211], repeat_check=False,
                                      source_sha256={"source": "original"},
                                      reference_sha256=downstream.sha256(reference),
                                      states=list(downstream.STATES), dcombat=None)
            exact_probability = 0.12345678901234567

            def result(seed):
                return (pd.DataFrame({"learner_seed": [seed], "value": [exact_probability]}),
                        pd.DataFrame({"learner_seed": [seed], "prediction": ["A"],
                                      "probability_b": [exact_probability]}),
                        [{"learner_seed": seed}])

            with patch.object(downstream, "source_hashes", return_value=args.source_sha256), \
                    patch.object(downstream, "load_inputs", return_value=(None, None)), \
                    patch.object(downstream.subprocess, "run") as generation, \
                    contextlib.redirect_stdout(io.StringIO()):
                with patch.object(downstream, "evaluate_inputs",
                                  side_effect=[result(101), RuntimeError("interruption")]) as fit:
                    with self.assertRaisesRegex(RuntimeError, "interruption"):
                        downstream.evaluate_dataset_job(("confounded_p50", 1, args))
                    self.assertEqual(fit.call_count, 2)
                seed_one = inputs.parent / "seed_101"
                preserved = {p.name: p.read_bytes() for p in seed_one.iterdir()}
                self.assertFalse((inputs.parent / "completion.json").exists())
                with patch.object(downstream, "evaluate_inputs", return_value=result(211)) as fit:
                    metrics, _, _ = downstream.evaluate_dataset_job(("confounded_p50", 1, args))
                    self.assertEqual(fit.call_count, 1)
                    self.assertEqual(fit.call_args.args[2], [211])
                self.assertEqual(set(metrics.learner_seed), {101, 211})
                self.assertTrue(metrics.value.eq(exact_probability).all())
                predictions = pd.read_csv(inputs.parent / "predictions.tsv.gz", sep="\t",
                                          float_precision="round_trip")
                self.assertTrue(predictions.probability_b.eq(exact_probability).all())
                self.assertEqual(preserved, {p.name: p.read_bytes() for p in seed_one.iterdir()})
                generation.assert_not_called()
                with patch.object(downstream, "evaluate_inputs") as fit:
                    downstream.evaluate_dataset_job(("confounded_p50", 1, args))
                    fit.assert_not_called()

@unittest.skipUnless(shutil.which("Rscript"), "Rscript with limma is unavailable")
class TrainingOnlyCorrectionTests(unittest.TestCase):
    """The correction of the learning inputs sees the training samples only."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.work = Path(directory.name)
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))

    @staticmethod
    def data(n_labs, per_lab=24, features=12, b_per_lab=None):
        rng = np.random.RandomState(7)
        samples = pd.Index([f"s.{i}" for i in range(n_labs * per_lab)])
        metadata = pd.DataFrame({
            "condition": np.concatenate([np.repeat(["B", "A"], [b, per_lab - b])
                                         for b in b_per_lab or [per_lab // 2] * n_labs]),
            "lab": np.repeat([f"batch{i + 1}" for i in range(n_labs)], per_lab)}, index=samples)
        matrix = pd.DataFrame(rng.normal(size=(features, len(samples))),
                              index=[f"prt{i}" for i in range(features)], columns=samples)
        matrix.iloc[:3] += 2 * metadata.condition.eq("B").to_numpy()
        matrix += np.repeat(rng.normal(scale=2, size=(features, n_labs)), per_lab, axis=1)
        return matrix, metadata

    @staticmethod
    def held_out(metadata, lab):
        train = {name: metadata.index[metadata.lab.eq(name)]
                 for name in sorted(metadata.lab.unique()) if name != lab}
        return train, metadata.index[metadata.lab.eq(lab)]

    def test_held_out_lab_is_aligned_to_the_training_level_without_its_labels(self):
        matrix, metadata = self.data(3, b_per_lab=[6, 12, 18])
        train, test = self.held_out(metadata, "batch2")
        states = ("truth", "uncorrected", "no_label", "label_aware")
        inputs = {"truth": matrix, "uncorrected": matrix}
        fold = training_correction.confounding_split(inputs, metadata, train, test,
                                                     self.work / "a", states)
        self.assertIs(fold["uncorrected"]["central"], matrix)
        for state in ("no_label", "label_aware"):
            for learner in ("federated", "central"):
                corrected = fold[state][learner]
                self.assertTrue(corrected.index.equals(matrix.index))
                self.assertTrue(corrected.columns.equals(matrix.columns))
                level = corrected.loc[:, [s for ids in train.values() for s in ids]].mean(axis=1)
                np.testing.assert_allclose(corrected.loc[:, test].mean(axis=1), level, atol=1e-10)
                # Within a lab every sample receives the same shift per feature.
                for lab in ("batch1", "batch2", "batch3"):
                    shift = (corrected - matrix).loc[:, metadata.lab.eq(lab)]
                    np.testing.assert_allclose(shift.sub(shift.iloc[:, 0], axis=0), 0, atol=1e-10)
            np.testing.assert_allclose(fold[state]["federated"], fold[state]["central"],
                                       atol=training_correction.TOLERANCE)
        no_label = fold["no_label"]["federated"]
        np.testing.assert_allclose(no_label.loc[:, train["batch3"]].mean(axis=1),
                                   no_label.loc[:, train["batch1"]].mean(axis=1), atol=1e-10)
        # The training labels enter the label-aware fit: the class mix differs between labs.
        self.assertFalse(np.allclose(no_label, fold["label_aware"]["federated"]))
        # Other labels of the held-out lab do not change any corrected value.
        relabelled = metadata.copy()
        relabelled.loc[test, "condition"] = np.where(metadata.condition.loc[test].eq("A"), "B", "A")
        again = training_correction.confounding_split(inputs, relabelled, train, test,
                                                      self.work / "b", ("label_aware",))
        pd.testing.assert_frame_equal(again["label_aware"]["federated"],
                                      fold["label_aware"]["federated"])

    def test_test_samples_take_the_shift_of_their_training_lab_without_their_labels(self):
        matrix, metadata = self.data(3)
        split = downstream.within_lab_split(metadata.condition, metadata.lab, 11)
        inputs = {"truth": matrix, "uncorrected": matrix}
        fold = training_correction.confounding_split(
            inputs, metadata, split["train"], split["test"], self.work / "a", ("label_aware",))
        corrected = fold["label_aware"]["federated"]
        for lab, training in split["train"].items():
            shift = (corrected - matrix).loc[:, metadata.lab.eq(lab)]
            np.testing.assert_allclose(shift.sub(shift[training[0]], axis=0), 0, atol=1e-10)
        relabelled = metadata.copy()
        relabelled.loc[split["test"], "condition"] = "A"
        again = training_correction.confounding_split(
            inputs, relabelled, split["train"], split["test"], self.work / "b", ("label_aware",))
        pd.testing.assert_frame_equal(again["label_aware"]["federated"], corrected)

    def test_missing_value_routes_follow_the_training_samples(self):
        truth, metadata = self.data(5, features=10)
        matrix = truth.copy()
        lab = metadata.lab
        matrix.loc["prt0", lab.eq("batch1")] = np.nan                    # three training labs
        matrix.loc["prt1", lab.isin(["batch1", "batch2"])] = np.nan      # two: not kept
        matrix.loc["prt2", lab.index[lab.eq("batch5")][5:]] = np.nan     # held-out lab: five values
        matrix.loc["prt3", lab.index[lab.eq("batch4")][5:]] = np.nan     # training lab: five values
        matrix.loc["prt4", lab.index[lab.eq("batch1")][:3]] = np.nan     # single missing values
        matrix.loc["prt5", lab.index[lab.eq("batch5")][:3]] = np.nan
        truth = truth.mask(matrix.isna())
        minimum = matrix.min(axis=0)
        train, test = self.held_out(metadata, "batch5")
        fold = training_correction.missingness_split(
            {"truth": truth, "uncorrected": matrix}, metadata, minimum, train, test, self.work,
            downstream.ROUTE_STATES)
        fold = {state: by_learner["federated"] for state, by_learner in fold.items()}
        for states in downstream.ROUTES.values():
            for state in states:
                pd.testing.assert_frame_equal(fold[state].isna(), fold[states[0]].isna())
        keep = fold["no_label"]
        self.assertEqual(sorted(set(matrix.index) - set(keep.index)), ["prt1"])
        self.assertTrue(keep.loc["prt2", test].isna().all())
        self.assertTrue(keep.loc["prt3", lab.eq("batch4")].isna().all())
        self.assertEqual(keep.loc["prt5", test].notna().sum(), 21)
        level = keep.loc[:, lab.ne("batch5")].mean(axis=1)
        aligned = keep.index.difference(["prt2"])
        np.testing.assert_allclose(keep.loc[aligned, test].mean(axis=1), level[aligned], atol=1e-10)
        # Drop: complete in the training samples; the held-out lab keeps its missing values.
        self.assertEqual(list(fold["drop_no_label"].index),
                         ["prt2", "prt5", "prt6", "prt7", "prt8", "prt9"])
        self.assertEqual(fold["drop_no_label"].loc["prt5", test].isna().sum(), 3)
        # Impute: the minimum in every training lab; a missing value takes its sample's minimum.
        imputed = fold["impute_uncorrected"]
        self.assertEqual(list(imputed.index), ["prt2", "prt4", "prt5", "prt6", "prt7", "prt8", "prt9"])
        missing = matrix.loc["prt4"].isna()
        np.testing.assert_array_equal(imputed.loc["prt4", missing], minimum[missing])
        self.assertTrue(imputed.loc["prt2", test].isna().all())
        self.assertEqual(imputed.drop(index="prt2").isna().sum().sum(), 0)

    def test_evaluation_scores_training_only_splits_per_lab(self):
        matrix, metadata = self.data(3)
        args = argparse.Namespace(analysis="confounding", states=list(downstream.STATES))
        inputs = {"truth": matrix, "uncorrected": matrix}
        split_inputs = downstream.training_only_inputs(args, metadata, inputs)
        metrics, predictions, _ = downstream.evaluate_inputs(
            metadata, inputs, [11], False, rf_protocols=("loco", "within_lab_70_30"),
            tasks=("rf",), split_inputs=split_inputs)
        self.assertIs(split_inputs("batch1", None), split_inputs("batch1", None))
        self.assertFalse(metrics.oracle.any())
        self.assertEqual(set(metrics.state), set(downstream.STATES))
        counts = metrics.groupby(["protocol", "state", "learner"]).lab.agg(sorted)
        self.assertTrue(counts.map(lambda labs: labs == ["batch1", "batch2", "batch3",
                                                         "pooled"]).all())
        self.assertEqual(len(counts), 16)
        held_out = predictions.loc[predictions.protocol.eq("loco")]
        self.assertTrue(held_out.groupby(["state", "learner"])["sample"].agg(list).map(
            lambda samples: samples == list(metadata.index)).all())


if __name__ == "__main__":
    unittest.main()
