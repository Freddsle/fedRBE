"""Check automatic clustering preparation from canonical matrices and metadata."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from evaluation_utils.feature_panel import load_panel_matrices
from evaluation_utils.real_datasets_utils import (
    DatasetConfig,
    load_dataset_metadata,
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

    def load(self, cfg=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return load_panel_matrices(cfg or self.cfg, canonical_before=True)

    def test_canonical_ovarian_metadata_contains_six_distinct_studies(self):
        root = Path(__file__).resolve().parents[2]
        cfg = shared.dataset_configs(root)["ovarian_cancer"]
        clients = shared.discover_clients(cfg)
        self.assertEqual([client.name for client in clients],
                         ["GSE10971", "GSE26712", "GSE38666", "GSE40595", "GSE6008", "GSE69428"])
        metadata = shared.load_metadata(cfg, clients)
        self.assertEqual(len(metadata), 331)
        self.assertEqual(metadata.condition.value_counts().to_dict(), {"1": 278, "0": 53})
        self.assertTrue(metadata.file.is_unique)

    def test_loads_complete_case_matrices_in_metadata_order(self):
        metadata, clients, matrices = self.load()
        samples = ["s2", "s1", "s4", "s3"]
        expected = self.matrix.loc[["complete"], samples].rename_axis("rowname")
        pd.testing.assert_frame_equal(matrices["before"], expected)
        pd.testing.assert_frame_equal(matrices["central_corr"], expected + 5)
        pd.testing.assert_frame_equal(matrices["fed_corr"], expected + 10)
        self.assertEqual(metadata.file.tolist(), samples)
        self.assertEqual([path.name for path in clients], ["A", "B"])

    def test_missing_sample_stops_loading(self):
        self.matrix.drop(columns="s1").to_csv(self.central, sep="\t")
        with self.assertRaisesRegex(ValueError, "samples are missing"):
            self.load()

    def test_unavailable_corrected_matrix_never_uses_uncorrected_client_values(self):
        for name, samples in (("A", ["s2", "s1"]), ("B", ["s4", "s3"])):
            self.matrix[samples].to_csv(self.before / name / "matrix.tsv", sep="\t")
        pointer = "version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 100\n"
        for source, path in (("central", self.central), ("federated", self.federated)):
            for state in ("LFS pointer", "missing"):
                with self.subTest(source=source, state=state):
                    (self.matrix + 5).to_csv(self.central, sep="\t")
                    (self.matrix + 10).to_csv(self.federated, sep="\t")
                    if state == "LFS pointer":
                        path.write_text(pointer)
                    else:
                        path.unlink()
                    with self.assertRaises(FileNotFoundError):
                        self.load()

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
        self.assertEqual(self.load(cfg)[2]["before"].columns.tolist(), metadata.file.tolist())

    def test_canonical_quartet_loads_from_modality_inputs(self):
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
        for arm, offset in (("label_aware", 5.), ("no_label", 10.)):
            cfg = DatasetConfig(**{**self.cfg.__dict__, "name": "quartet_multiomics",
                "before_matrix": quartet / "before/all_modalities_before_kmeans_matrix.tsv",
                "corrected_central": quartet / f"after/{arm}/all_modalities_corrected_kmeans_matrix.tsv",
                "corrected_federated": quartet / f"after/{arm}/all_modalities_fedapp_kmeans_matrix.tsv",
                "sites_root": self.root / "unused_client_exports", "correction_arm": arm})
            with self.subTest(arm=arm):
                shared.build_quartet_matrices(quartet, arm)
                metadata, _, matrices = self.load(cfg)
                self.assertEqual(matrices["before"].shape, (3, 4))
                pd.testing.assert_frame_equal(matrices["central_corr"], matrices["before"] + offset)
                pd.testing.assert_frame_equal(matrices["fed_corr"], matrices["before"] + offset + 20)
                self.assertEqual(metadata.condition.tolist(), ["D5", "D6", "D6", "D5"])
                # Missing joint files are rebuilt from the modality inputs.
                cfg.before_matrix.unlink()
                cfg.corrected_central.unlink()
                shared.build_quartet_matrices(quartet, arm)
                for state, matrix in self.load(cfg)[2].items():
                    pd.testing.assert_frame_equal(matrix, matrices[state])
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


if __name__ == "__main__":
    unittest.main()
