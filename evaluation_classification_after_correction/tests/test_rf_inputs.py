"""Check configurable RF inputs through the existing experiment launchers."""

import contextlib
import importlib
import io
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import pandas as pd
import yaml

COMPONENT = Path(__file__).resolve().parents[1]
REPO = COMPONENT.parent
with patch.object(sys, "path", [str(COMPONENT), *sys.path]):
    from helper import helper_run_classification_report_metrics as helper


class RandomForestInputTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def descriptor(self, name):
        folder = self.root / name / "site"
        folder.mkdir(parents=True)
        pd.DataFrame({"s1": [1., float("nan")], "s2": [2., float("nan")]},
                     index=["observed", "missing"]).to_csv(folder / "data.tsv", sep="\t")
        pd.DataFrame({"file": ["s1", "s2"], "Condition": [0, 1]}).to_csv(
            folder / "design.tsv", sep="\t", index=False)
        path = folder.parent / "datainfo.json"
        path.write_text(json.dumps({
            "data_name": "Fixture", "covariates": [], "prediction_targets": ["Condition"],
            "datafile": {"filename": "data.tsv", "separator": "\t", "rotation": "features x samples"},
            "cohorts": [{"name": "site", "folder": "site", "designfile": {
                "filename": "design.tsv", "separator": "\t", "rotation": "samples x features",
                "samplename_column": "file",
            }}],
        }))
        return path

    def test_registry_selection_shares_baseline_and_preserves_normal_inputs(self):
        before, first, second = [self.descriptor(name) for name in ("before", "first", "second")]
        config = self.root / "datasets.yaml"
        config.write_text(yaml.safe_dump({
            "first": {"classification": {"before": str(before), "corrected": str(first)}},
            "second": {"classification": {"before": str(before),
                                           "corrected": os.path.relpath(second, REPO),
                                           "preprocessing_name": "alternative"}},
        }))
        output = self.root / "results/metrics.csv"
        args = ["--config", str(config), "--results-file", str(output),
                "--seeds", "42", "43", "--force-run"]
        constructor = Mock()
        helper.run_classification_experiments(constructor, args)
        self.assertEqual([call.args[0] for call in constructor.call_args_list],
                         ["uncorrected", "corrected", "alternative"])
        self.assertEqual(constructor.return_value.run_experiment.call_count, 6)
        self.assertEqual([call.kwargs["seed"] for call in constructor.return_value.run_experiment.call_args_list],
                         [42, 42, 42, 43, 43, 43])
        self.assertTrue(all(call.kwargs["force_run"] is True
                            for call in constructor.return_value.run_experiment.call_args_list))
        for call in constructor.call_args_list:
            self.assertEqual(call.kwargs["resultfile"].path, output)
        info = constructor.call_args_list[0].args[1]
        frame = info.receive_data(info.cohorts[0])
        self.assertEqual(list(frame.columns), ["observed", "missing", "Condition"])
        self.assertTrue(frame.missing.isna().all())
        self.assertTrue(output.parent.is_dir())
        constructor.reset_mock()
        helper.run_classification_experiments(constructor, [*args, "--datasets", "second"])
        self.assertEqual([call.args[0] for call in constructor.call_args_list],
                         ["uncorrected", "alternative"])

    def test_defaults_and_entrypoints_keep_the_existing_experiment_types(self):
        constructor = Mock()
        with patch.object(helper, "DataInfo") as data, \
                patch.object(helper, "build_quartet_matrices"), \
                patch.object(helper.quartet_exporter, "create_merged_omics_folder"):
            helper.run_classification_experiments(
                constructor, ["--results-file", str(self.root / "metrics.csv")])
        self.assertEqual(constructor.call_count, 10)  # Five datasets, before and after.
        self.assertEqual(len({call.args[0] for call in data.call_args_list}), 10)
        self.assertEqual(constructor.return_value.run_experiment.call_count, 100)
        for script, experiment in (
            ("run_classification_leave_one_cohort_out.py", helper.ClassificationExperimentLeaveOneCohortOut),
            ("run_classification_train_test_split.py", helper.ClassificationExperimentTrainTestSplit),
        ):
            with patch.object(helper, "run_classification_experiments") as runner:
                runpy.run_path(str(COMPONENT / script), run_name="import_check")
                runner.assert_not_called()
                runpy.run_path(str(COMPONENT / script), run_name="__main__")
                runner.assert_called_once_with(experiment)

    def test_quartet_export_uses_named_variants_and_explicit_sample_mapping(self):
        exporter = importlib.import_module("evaluation_data.quartet_multiomics.05_create_merged_omics_folder")
        matrix = pd.DataFrame([[1., 2., 3., 4.], [float("nan"), float("nan"), 5., 6.]],
                              index=["gene_1", "gene_2"], columns=["s1", "s2", "s3", "s4"])
        matrix_path = self.root / "matrix.tsv"
        matrix.to_csv(matrix_path, sep="\t", index_label="rowname")
        metadata = pd.DataFrame({"pseudo_sample": matrix.columns, "client": ["B", "B", "A", "A"],
                                 "condition": ["D5", "D6", "D5", "D6"]})
        metadata_path = self.root / "metadata.tsv"
        metadata.to_csv(metadata_path, sep="\t", index=False)
        output = self.root / "rf"
        before = self.root / "before/all_modalities_before_kmeans_matrix.tsv"
        before.parent.mkdir()
        matrix.to_csv(before, sep="\t", index_label="rowname")
        variants = [("alternative", {"datafiles": {"alternative": matrix_path}}, matrix)]
        for offset, arm in enumerate(("label_aware", "no_label"), start=1):
            corrected = self.root / f"after/{arm}/all_modalities_fedapp_kmeans_matrix.tsv"
            corrected.parent.mkdir(parents=True)
            (matrix + offset).to_csv(corrected, sep="\t", index_label="rowname")
            variants.append((f"after/{arm}", {"correction_arm": arm}, matrix + offset))
        for folder, inputs, expected_matrix in variants:
            with self.subTest(folder=folder), patch.object(exporter, "__file__", str(self.root / "export.py")), \
                    contextlib.redirect_stdout(io.StringIO()):
                exporter.create_merged_omics_folder(output_folder=output,
                                                   metadata_file=metadata_path, **inputs)
            info = helper.DataInfo(output / folder / "datainfo.json")
            self.assertEqual([cohort.name for cohort in info.cohorts], ["B", "A"])
            self.assertEqual(info.prediction_targets, ["condition"])
            for cohort in info.cohorts:
                labels = metadata.loc[metadata.client.eq(cohort.name)].set_index("pseudo_sample")
                expected = expected_matrix.T.loc[labels.index].copy()
                expected["condition"] = labels.condition
                pd.testing.assert_frame_equal(info.receive_data(cohort), expected.rename_axis("sample_id"))

    def test_correction_arm_selects_real_data_folders_and_distinct_result_keys(self):
        for arm, preprocessing in (("label_aware", "corrected"), ("no_label", "no_label")):
            with self.subTest(arm=arm), patch.object(helper, "DataInfo") as data, \
                    patch.object(helper, "build_quartet_matrices"), \
                    patch.object(helper.quartet_exporter, "create_merged_omics_folder"):
                constructor = Mock()
                helper.run_classification_experiments(constructor, [
                    "--correction-arm", arm, "--datasets", "ecoli", "quartet_multiomics", "balanced",
                    "--seeds", "42", "--results-file", str(self.root / "metrics.csv"),
                ])
                self.assertEqual([call.args[0] for call in data.call_args_list], [
                    REPO / "evaluation_data/ecoli/before/datainfo.json",
                    REPO / f"evaluation_data/ecoli/after/{arm}/datainfo.json",
                    REPO / "evaluation_data/quartet_multiomics/merged_omics/before/datainfo.json",
                    REPO / f"evaluation_data/quartet_multiomics/merged_omics/after/{arm}/datainfo.json",
                    REPO / "evaluation_data/simulated/balanced/before/datainfo.json",
                    REPO / "evaluation_data/simulated/balanced/after/datainfo.json",
                ])
                self.assertEqual([call.args[0] for call in constructor.call_args_list],
                                 ["uncorrected", preprocessing, "uncorrected", preprocessing,
                                  "uncorrected", "corrected"])
                self.assertEqual(constructor.return_value.run_experiment.call_count, 6)

    def test_canonical_quartet_prepares_both_arms_from_modality_inputs(self):
        quartet = self.root / "evaluation_data/quartet_multiomics"
        pseudo = ["B_D5", "B_D6", "A_D5", "A_D6"]
        metadata = pd.DataFrame({"pseudo_sample": pseudo, "client": ["B", "B", "A", "A"],
                                 "condition": ["D5", "D6", "D5", "D6"], "rep": 1})
        expected = []
        for i, modality in enumerate(("Transcriptomics", "Proteomics", "Metabolomics")):
            raw_samples = [f"{modality}_{sample}" for sample in pseudo]
            matrix = pd.DataFrame([[10. * i + j for j in range(4)],
                                   [float("nan"), 1., 2., float("nan")]],
                                  index=["measured", "missing"], columns=raw_samples)
            before = quartet / "before" / modality
            before.mkdir(parents=True)
            matrix.to_csv(before / "central_intensities_log_UNION.tsv", sep="\t", index_label="rowname")
            metadata.assign(file=raw_samples).to_csv(before / "metadata.tsv", sep="\t", index=False)
            for arm, offset in (("label_aware", 10.), ("no_label", 100.)):
                corrected = quartet / "after" / arm / modality
                corrected.mkdir(parents=True)
                (matrix + offset).to_csv(corrected / "FedApp_corrected_data_smpc.tsv",
                                         sep="\t", index_label="rowname")
            matrix.index = [f"{modality}__{feature}" for feature in matrix.index]
            matrix.columns = pseudo
            expected.append(matrix)
        expected = pd.concat(expected).T.sort_index()
        expected.index.name = "sample_id"
        self.assertFalse((quartet / "merged_omics").exists())
        self.assertFalse((quartet / "all_modalities_metadata.tsv").exists())
        for arm, offset in (("label_aware", 10.), ("no_label", 100.)):
            constructor = Mock()
            with patch.object(helper, "SCRIPT_FOLDER", self.root / "classification/helper"), \
                    contextlib.redirect_stdout(io.StringIO()):
                helper.run_classification_experiments(constructor, [
                    "--correction-arm", arm, "--datasets", "quartet_multiomics",
                    "--seeds", "42", "--results-file", str(self.root / "metrics.csv"),
                ])
            self.assertEqual(constructor.call_count, 2)
            self.assertEqual([call.args[0] for call in constructor.call_args_list],
                             ["uncorrected", "corrected" if arm == "label_aware" else "no_label"])
            for call, delta in zip(constructor.call_args_list, (0., offset)):
                info = call.args[1]
                actual = pd.concat([info.receive_data(cohort) for cohort in info.cohorts]).sort_index()
                wanted = expected + delta
                wanted["condition"] = metadata.set_index("pseudo_sample").condition
                pd.testing.assert_frame_equal(actual, wanted)
            self.assertEqual(constructor.return_value.run_experiment.call_count, 2)
        self.assertEqual({path.name for path in (self.root / "evaluation_data").iterdir()},
                         {"quartet_multiomics"})

    def test_correction_arm_cannot_override_explicit_registry_paths(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            helper.run_classification_experiments(
                Mock(), ["--config", "datasets.yaml", "--correction-arm", "no_label"])
        self.assertEqual(error.exception.code, 2)
        before, corrected = self.descriptor("before"), self.descriptor("corrected")
        registry = self.root / "registry.yaml"
        registry.write_text(yaml.safe_dump({"quartet_multiomics": {"classification": {
            "before": str(before), "corrected": str(corrected),
        }}}))
        with patch.object(helper, "build_quartet_matrices") as builder, \
                patch.object(helper.quartet_exporter, "create_merged_omics_folder") as exporter:
            helper.run_classification_experiments(Mock(), [
                "--config", str(registry), "--seeds", "42",
                "--results-file", str(self.root / "metrics.csv"),
            ])
        builder.assert_not_called()
        exporter.assert_not_called()


if __name__ == "__main__":
    unittest.main()
