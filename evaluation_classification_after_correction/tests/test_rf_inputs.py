"""Check the experiment launchers and their dataset selection."""

import contextlib
import io
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

COMPONENT = Path(__file__).resolve().parents[1]
with patch.object(sys, "path", [str(COMPONENT), *sys.path]):
    from helper import loco, train_test_split
    from helper.steps import datasets


class RandomForestInputTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_entrypoints_run_their_experiments(self):
        for script, module, runner in (
                ("run_classification_leave_one_cohort_out.py", loco,
                 "run_loco_experiments"),
                ("run_classification_train_test_split.py", train_test_split,
                 "run_train_test_split_experiments")):
            with self.subTest(script=script), patch.object(module, runner) as run:
                runpy.run_path(str(COMPONENT / script), run_name="import_check")
                run.assert_not_called()
                runpy.run_path(str(COMPONENT / script), run_name="__main__")
                run.assert_called_once_with()

    def test_loco_cli_selects_datasets_and_saves_corrections_next_to_the_results(self):
        registry = {"first": "spec_1", "second": "spec_2"}
        results = self.root / "results" / "metrics.csv"
        with patch.object(datasets, "dataset_registry", return_value=registry), \
                patch.object(loco, "ClassificationExperimentLeaveOneCohortOut") as experiment:
            loco.run_loco_experiments([
                "--datasets", "second", "--arms", "label_free", "--seeds", "1", "2",
                "--results-file", str(results), "--force-run"])
        experiment.assert_called_once()
        self.assertEqual(experiment.call_args.args, ("spec_2",))
        self.assertEqual(experiment.call_args.kwargs["arms"], ["label_free"])
        self.assertEqual(experiment.call_args.kwargs["resultfile"].path, results)
        self.assertEqual(experiment.call_args.kwargs["corrections_folder"],
                         results.parent / "loco_corrections")
        experiment.return_value.run_experiment.assert_called_once_with(seeds=[1, 2],
                                                                       force_run=True)
        self.assertTrue(results.parent.is_dir())
        with patch.object(datasets, "dataset_registry", return_value=registry), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            loco.run_loco_experiments(["--datasets", "third"])


if __name__ == "__main__":
    unittest.main()
