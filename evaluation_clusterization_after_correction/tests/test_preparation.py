"""Check automatic clustering preparation from canonical matrices and metadata."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import pandas as pd

from evaluation_utils.real_datasets_utils import (
    DatasetConfig,
    load_dataset_metadata,
    prepare_dataset,
)
from evaluation_utils import real_datasets_utils as shared


class ClusteringPreparationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.before = self.root / "data/before"
        self.before.mkdir(parents=True)
        self.matrix = pd.DataFrame(
            [[4., 1., 3., 2.], [4., float("nan"), 3., 2.]],
            index=["complete", "partial"], columns=["s4", "s1", "s3", "s2"],
        )
        self.matrix.to_csv(self.before / "matrix.tsv", sep="\t")
        for name, samples in (("A", ["s2", "s1"]), ("B", ["s4", "s3"])):
            client = self.before / name
            client.mkdir()
            pd.DataFrame({"file": samples, "condition": ["case", "control"]}).to_csv(
                client / "design.tsv", sep="\t", index=False
            )
        self.central = self.root / "central.tsv"
        self.federated = self.root / "federated.tsv"
        (self.matrix + 5).to_csv(self.central, sep="\t")
        (self.matrix + 10).to_csv(self.federated, sep="\t")
        self.cfg = DatasetConfig(
            name="fixture", before_matrix=self.before / "matrix.tsv",
            before_site_matrix_file="matrix.tsv", corrected_central=self.central,
            corrected_federated=self.federated, sites_root=self.before,
            site_glob="[AB]", design_file="design.tsv", sample_col="file",
            condition_col="condition",
        )
        self.prepared = self.root / "outputs/prepared"

    def prepare(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return prepare_dataset(self.cfg, self.prepared, **kwargs)

    def test_prepares_current_central_inputs_and_plot_intermediates(self):
        data = self.prepare()
        samples = ["s2", "s1", "s4", "s3"]
        expected = self.matrix.loc[["complete"], samples].rename_axis("rowname")
        pd.testing.assert_frame_equal(data["before"], expected)
        pd.testing.assert_frame_equal(data["corrected"], expected + 5)
        self.assertEqual(data["metadata"].file.tolist(), samples)
        self.assertEqual([path.name for path in data["clients"]], ["A", "B"])
        self.assertEqual(data["k_values"], [2])
        saved = pd.read_csv(self.prepared / "corrected_matrix.tsv", sep="\t", index_col=0)
        pd.testing.assert_frame_equal(saved, expected + 5)
        self.assertEqual(sorted(path.name for path in self.prepared.iterdir()),
                         ["before_matrix.tsv", "corrected_matrix.tsv", "metadata.tsv"])
        self.assertFalse((self.prepared.parent / "inputs").exists())

    def test_preserves_source_and_missing_value_options_and_refreshes_outputs(self):
        first = self.prepare(corrected_source="auto", remove_na=False)
        self.assertEqual(first["corrected_source"], "federated")
        self.assertEqual(first["corrected"].index.tolist(), ["complete", "partial"])
        self.assertTrue(pd.isna(first["corrected"].loc["partial", "s1"]))
        (self.matrix + 20).to_csv(self.federated, sep="\t")
        second = self.prepare(corrected_source="federated", remove_na=False)
        pd.testing.assert_frame_equal(second["corrected"], first["corrected"] + 10)
        saved = pd.read_csv(self.prepared / "corrected_matrix.tsv", sep="\t", index_col=0)
        pd.testing.assert_frame_equal(saved, second["corrected"])

    def test_missing_sample_stops_before_writing_intermediates(self):
        self.matrix.drop(columns="s1").to_csv(self.central, sep="\t")
        with self.assertRaisesRegex(ValueError, "samples are missing"):
            self.prepare()
        self.assertFalse(self.prepared.exists())

    def test_unavailable_corrected_matrix_never_uses_uncorrected_client_values(self):
        for name, samples in (("A", ["s2", "s1"]), ("B", ["s4", "s3"])):
            self.matrix[samples].to_csv(self.before / name / "matrix.tsv", sep="\t")
        pointer = "version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 100\n"
        for source, path in (("central", self.central), ("federated", self.federated)):
            for state in ("LFS pointer", "missing"):
                with self.subTest(source=source, state=state):
                    if state == "LFS pointer":
                        path.write_text(pointer)
                    else:
                        path.unlink()
                    with self.assertRaises(FileNotFoundError):
                        self.prepare(corrected_source=source)
                    self.assertFalse(self.prepared.exists())

    def test_quartet_uses_joint_metadata_without_generated_client_folders(self):
        mapping = self.before.parent / "all_modalities_metadata.tsv"
        pd.DataFrame({
            "pseudo_sample": ["s4", "s2", "s3", "s1"],
            "condition": ["D5", "D6", "D6", "D5"],
            "client": ["B", "A", "B", "A"],
        }).to_csv(mapping, sep="\t", index=False)
        cfg = DatasetConfig(**{
            **self.cfg.__dict__, "name": "quartet_multiomics_no_label",
            "extra": {"base_dataset": "quartet_multiomics"},
            "sites_root": self.root / "generated_clients_not_present",
        })
        metadata, clients = load_dataset_metadata(cfg)
        self.assertEqual(metadata.file.tolist(), ["s2", "s1", "s4", "s3"])
        self.assertEqual(metadata.condition.tolist(), ["D6", "D5", "D5", "D6"])
        self.assertEqual([path.name for path in clients], ["A", "B"])
        self.assertTrue(all(not path.exists() for path in clients))
        with contextlib.redirect_stdout(io.StringIO()):
            data = prepare_dataset(cfg, self.prepared)
        self.assertEqual(data["before"].columns.tolist(), metadata.file.tolist())

    def test_evaluation_notebooks_start_without_prepared_outputs(self):
        component = Path(__file__).resolve().parents[1] / "real_datasets"
        for filename, index in (("02_central_kmeans.ipynb", 6),
                                ("03_federated_runs.ipynb", 6),
                                ("05_multiple_runs.ipynb", 8)):
            output = self.root / filename
            self.assertFalse(output.exists())
            notebook = json.loads((component / filename).read_text())
            source = "".join(notebook["cells"][index]["source"])
            namespace = {
                **vars(shared), "pd": pd, "prepare_dataset": prepare_dataset,
                "dataset_configs": lambda *args, **kwargs: {"fixture": self.cfg},
                "DATASET_CONFIG": None, "DATASETS": None, "REPO_ROOT": self.root,
                "CORRECTION_ARM": "no_label",
                "OUTPUT_ROOT": output, "CORRECTED_SOURCE": "central",
                "REMOVE_NA": True, "SEED": 11, "display": lambda frame: None,
            }
            with self.subTest(notebook=filename), contextlib.redirect_stdout(io.StringIO()):
                exec(compile(source, filename, "exec"), namespace)
                if filename.startswith("02"):
                    self.assertTrue((output / "fixture/kmeans_res/runs/"
                                     "1_metadata_cntrl_kmeans_res.tsv").exists())
                    self.assertFalse((output / "fixture/inputs").exists())
                elif filename.startswith("03"):
                    self.assertTrue((output / "fixture/inputs/corrected/A/intensities.tsv").exists())
                else:
                    data = namespace["dataset_data"]["fixture"]
                    self.assertEqual(data["before"].columns.tolist(), ["s2", "s1", "s4", "s3"])
                    self.assertEqual(data["central_corr"].shape, (1, 4))
                    self.assertEqual(data["fed_corr"].shape, (1, 4))
                    self.assertEqual([path.name for path in data["clients"]], ["A", "B"])

    def test_canonical_quartet_prepares_inputs_before_single_and_multirun_evaluation(self):
        quartet = self.root / "quartet"
        pseudo = ["s4", "s1", "s3", "s2"]
        for modality in ("Transcriptomics", "Proteomics", "Metabolomics"):
            folder = quartet / "before" / modality
            folder.mkdir(parents=True)
            self.matrix.to_csv(folder / "central_intensities_log_UNION.tsv", sep="\t")
            pd.DataFrame({"file": pseudo, "pseudo_sample": pseudo,
                          "client": ["B", "A", "B", "A"],
                          "condition": ["D5", "D5", "D6", "D6"], "rep": 1}).to_csv(
                folder / "metadata.tsv", sep="\t", index=False)
            for arm, offset in (("label_aware", 5.), ("no_label", 10.)):
                after = quartet / "after" / arm / modality
                after.mkdir(parents=True)
                (self.matrix + offset).to_csv(after / "intensities_log_Rcorrected_UNION.tsv", sep="\t")
                (self.matrix + offset + 20).to_csv(after / "FedApp_corrected_data_smpc.tsv", sep="\t")
        self.assertFalse((quartet / "all_modalities_metadata.tsv").exists())
        notebook = Path(__file__).resolve().parents[1] / "real_datasets/05_multiple_runs.ipynb"
        source = "".join(json.loads(notebook.read_text())["cells"][8]["source"])
        for arm, offset in (("label_aware", 5.), ("no_label", 10.)):
            cfg = DatasetConfig(**{**self.cfg.__dict__, "name": "quartet_multiomics",
                "before_matrix": quartet / "before/all_modalities_before_kmeans_matrix.tsv",
                "corrected_central": quartet / f"after/{arm}/all_modalities_corrected_kmeans_matrix.tsv",
                "corrected_federated": quartet / f"after/{arm}/all_modalities_fedapp_kmeans_matrix.tsv",
                "sites_root": self.root / "unused_client_exports", "correction_arm": arm})
            with self.subTest(arm=arm), contextlib.redirect_stdout(io.StringIO()):
                data = prepare_dataset(cfg, self.prepared)
                self.assertEqual(data["before"].shape, (3, 4))
                pd.testing.assert_frame_equal(data["corrected"], data["before"] + offset)
                # Start multirun from modality inputs too, rather than files made by 02.
                cfg.before_matrix.unlink()
                cfg.corrected_central.unlink()
                namespace = {**vars(shared), "pd": pd, "REPO_ROOT": self.root,
                             "DATASET_CONFIG": None, "DATASETS": None,
                             "CORRECTION_ARM": arm,
                             "dataset_configs": lambda *args, **kwargs: {cfg.name: cfg}}
                exec(compile(source, str(notebook), "exec"), namespace)
                multirun = namespace["dataset_data"][cfg.name]
                pd.testing.assert_frame_equal(multirun["before"], data["before"])
                pd.testing.assert_frame_equal(multirun["central_corr"], data["corrected"])
                pd.testing.assert_frame_equal(multirun["fed_corr"], data["corrected"] + 20)
                self.assertEqual(multirun["metadata"].condition.tolist(), ["D5", "D6", "D6", "D5"])
        (quartet / "after/no_label/Proteomics/FedApp_corrected_data_smpc.tsv").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "federated.*incomplete"):
            shared.build_quartet_matrices(quartet, "no_label")
        shared.build_quartet_matrices(quartet, "no_label", "central")
        metadata_path = quartet / "before/Proteomics/metadata.tsv"
        metadata = pd.read_csv(metadata_path, sep="\t")
        metadata.loc[0, "condition"] = "wrong"
        metadata.to_csv(metadata_path, sep="\t", index=False)
        with self.assertRaisesRegex(ValueError, "labels differ"):
            shared.build_quartet_matrices(quartet, "no_label", "central")

    def test_correction_arm_changes_only_canonical_corrected_folders(self):
        aware = shared.dataset_configs(self.root)
        no_label = shared.dataset_configs(self.root, correction_arm="no_label")
        self.assertEqual(set(aware), set(no_label))
        for name, cfg in aware.items():
            expected = {**cfg.__dict__, "correction_arm": "no_label"}
            for key in ("corrected_central", "corrected_federated"):
                expected[key] = Path(str(expected[key]).replace("/label_aware/", "/no_label/"))
            self.assertEqual(no_label[name].__dict__, expected)
        with self.assertRaisesRegex(ValueError, "correction_arm"):
            shared.dataset_configs(self.root, correction_arm="unknown")

    def test_custom_registry_paths_are_explicit(self):
        registry = self.root / "datasets.yaml"
        contents = shared.yaml.safe_load(Path(shared.__file__).with_name("datasets.yaml").read_text())
        contents["ecoli"]["corrected_central"] = "chosen/corrected.tsv"
        registry.write_text(shared.yaml.safe_dump(contents))
        cfg = shared.dataset_configs(self.root, registry)["ecoli"]
        self.assertEqual(cfg.corrected_central, self.root / "chosen/corrected.tsv")
        self.assertIsNone(cfg.correction_arm)
        with self.assertRaisesRegex(ValueError, "custom registry"):
            shared.dataset_configs(self.root, registry, correction_arm="no_label")

    def test_notebook_defaults_keep_correction_arms_in_separate_output_folders(self):
        component = Path(__file__).resolve().parents[1] / "real_datasets"
        for filename in ("02_central_kmeans.ipynb", "03_federated_runs.ipynb",
                         "04_analysis_metrics_plots.ipynb", "05_multiple_runs.ipynb"):
            notebook = json.loads((component / filename).read_text())
            source = "".join(notebook["cells"][4]["source"])
            for arm in ("label_aware", "no_label"):
                namespace = {"NOTEBOOK_DIR": self.root,
                             "sns": SimpleNamespace(set_theme=lambda **kwargs: None),
                             "plt": SimpleNamespace(rcParams={})}
                with self.subTest(notebook=filename, arm=arm), contextlib.redirect_stdout(io.StringIO()):
                    exec(source.replace('CORRECTION_ARM = "label_aware"',
                                        f'CORRECTION_ARM = "{arm}"'), namespace)
                    expected = self.root if arm == "label_aware" else self.root / arm
                    self.assertEqual(namespace["OUTPUT_ROOT"], expected)


if __name__ == "__main__":
    unittest.main()
