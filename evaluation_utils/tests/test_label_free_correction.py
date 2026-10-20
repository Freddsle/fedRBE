"""Correction-experiment and numerical checks on small temporary fixtures."""

import ast
import contextlib
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch
import zipfile

import numpy as np
import pandas as pd
import yaml

from batchcorrection.classes.client import Client
from batchcorrection.classes.coordinator_utils import (
    compute_beta, create_beta_mask, select_common_features_variables,
)
from evaluation_utils import run_correction_experiments as correction
from evaluation_utils.real_datasets_utils import (
    read_correction_matrix as read_matrix,
    validate_corrected_matrix as validate_corrected,
    validate_experiment_dir,
    write_feature_matrix,
)

REPO = correction.REPO_ROOT
RSCRIPT = REPO / ".pixi/envs/default/bin/Rscript"
if not RSCRIPT.is_file():
    RSCRIPT = Path(shutil.which("Rscript") or "/unavailable/Rscript")
requires_r = unittest.skipUnless(RSCRIPT.is_file(), "Rscript is unavailable; limma check unrun")
DATASET = "ccRCC_proteomics"
RETAINED = ["feature_z", "feature_a", "feature_m", "feature_c", "feature_n", "feature_d"]
SEED = 73



def client_correction(clients, threshold):
    """Exercise the app's client/coordinator lifecycle without transport."""
    information = []
    variables = list(clients[0].hash2variable)
    for client in clients:
        presence = client.get_batch_feature_presence_info(threshold)
        information.append((client.client_name, client.position, client.reference_batch, presence))
    features, presence, cohorts = select_common_features_variables(
        information, default_order=[c.client_name for c in clients],
    )
    statistics = []
    for client in clients:
        client.validate_inputs(variables)
        client.set_data(features)
        error = client.create_design(cohorts)
        if error:
            raise AssertionError(error)
        xtx, xty, error = client.compute_XtX_XtY()
        if error:
            raise AssertionError(error)
        statistics.append((xtx, xty))
    n, k = len(features), clients[0].design.shape[1]
    beta = compute_beta(statistics, n, k, create_beta_mask(presence, n, k))
    for client in clients:
        client.remove_batch_effects(beta)
    return pd.concat([c.data_corrected for c in clients], axis=1)


class FrozenCorrectionTests(unittest.TestCase):
    def setUp(self):
        self.frozen = pd.DataFrame(
            [[11., np.nan, 10.], [3., 4., 2.]],
            index=["feature_a", "feature_b"],
            columns=["client3_1", "client2_1", "client1_1"],
        )

    def test_corrected_result_realigns_to_prepared_order(self):
        frozen = self.frozen
        corrected = (frozen + 0.25).iloc[::-1, ::-1]
        result = validate_corrected(corrected, frozen)
        pd.testing.assert_frame_equal(result, frozen + 0.25)

    def test_corrected_result_rejects_changed_coverage_and_mask(self):
        frozen = self.frozen
        for matrix in (frozen.iloc[:-1], frozen.iloc[:, :-1], frozen.fillna(0.)):
            with self.subTest(shape=matrix.shape), self.assertRaises(ValueError):
                validate_corrected(matrix, frozen)


class ArchiveCoverageTests(unittest.TestCase):
    def setUp(self):
        from evaluation_utils import featurecloud_api_extension as api
        with patch.object(api.Experiment, "_startup", side_effect=AssertionError(
            "Importing the federated launcher must not start canonical experiments"
        )):
            import generate_fedrbe_corrected_datasets as federated
        self.federated = federated
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.clients, self.paths, self.inputs, configs, changes = [], [], [], [], []
        self.calls, self.startups, self.failure, self.archive_override = [], [], None, None
        for i in range(3):
            client = self.root / f"client_{i}"
            client.mkdir()
            matrix = pd.DataFrame(
                np.arange(12, dtype=float).reshape(2, 6) + i,
                index=["feature_z", "feature_a"],
                columns=[f"client{i}_sample{j}" for j in range(6)],
            )
            matrix.iloc[0, 1] = np.nan
            write_feature_matrix(matrix, client / "expression.tsv")
            pd.DataFrame({"file": matrix.columns, "Condition": [0, 1] * 3}).to_csv(
                client / "client_design.tsv", sep="\t", index=False)
            if i < 2:
                (client / "config.yml").write_bytes(b"# Preserve exact source bytes\nother: true\n")
            configs.append({"flimmaBatchCorrection": {
                "batch_col": None, "covariates": ["unused"],
                "data_filename": "expression.tsv", "design_filename": "client_design.tsv",
                "separator": "\t", "design_separator": "\t", "expression_file_flag": True,
                "index_col": "rowname", "position": None, "smpc": False,
                "normalizationMethod": None, "min_samples": 0,
            }})
            changes.append({"flimmaBatchCorrection": {
                "covariates": ["Condition"], "smpc": True, "position": i,
            }})
            self.clients.append(client)
            self.inputs.append(matrix)
            self.paths.append(self.make_archive(i))
        (self.root / "datainfo.json").write_text(json.dumps({"cohorts": [
            {"name": client.name, "folder": client.name,
             "designfile": {"filename": "client_design.tsv"}} for client in self.clients
        ]}))
        self.sources = {p: p.read_bytes() for client in self.clients
                        for p in client.iterdir() if p.is_file()}
        fixture = self

        class TransportFixture(api.Experiment):
            def _startup(self):
                fixture.startups.append(self.fc_data_dir)

            def run_test(self, retry=0):
                self._set_config_files()
                active = [yaml.safe_load((Path(p) / "config.yml").read_text())
                          for p in self.clients]
                fixture.calls.append((self.clients, active, self.app_image_name))
                if fixture.failure == "transport":
                    raise RuntimeError("fixture transport failed")
                if fixture.archive_override is not None:
                    return [str(p) for p in fixture.archive_override], 0, None
                paths = []
                for i, client in enumerate(self.clients):
                    details = active[i]["flimmaBatchCorrection"]
                    matrix = read_matrix(Path(client) / details["data_filename"])
                    matrix += .25 * len(fixture.calls)
                    if fixture.failure == "mask" and not details["covariates"]:
                        matrix.iloc[1, 0] = np.nan
                    paths.append(str(fixture.make_archive(i, matrix, str(len(fixture.calls)))))
                return paths, 0, None

        self.experiment = TransportFixture(
            name="fixture", clients=[str(p) for p in self.clients], app_image_name="baseline",
            fc_data_dir=str(self.root), config_files=configs, config_file_changes=changes,
        )
        self.enterContext(patch.object(federated.time, "sleep"))
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))

    def make_archive(self, index, matrix=None, suffix="raw"):
        path = self.root / f"result_{index}_{suffix}.zip"
        with zipfile.ZipFile(path, "w") as archive:
            for name in ("only_batch_corrected_data.csv", "full_corrected_data.csv"):
                archive.writestr(name, (self.inputs[index] + .25 if matrix is None
                                       else matrix).to_csv(sep="\t"))
        return path

    def run_campaign(self, folder="after", paired=True, protocols=(True,), modality=None):
        selected = []
        for smpc in protocols:
            baseline = deepcopy(self.experiment)
            for changes in baseline.config_file_changes:
                changes["flimmaBatchCorrection"]["smpc"] = smpc
            filename = "FedApp_corrected_data_smpc.tsv" if smpc else "FedApp_corrected_data.tsv"
            relative = Path(modality) if modality else Path()
            selected.append((baseline, str(Path(folder) / "label_aware" / relative / filename)))
            if paired:
                selected.append((self.federated.without_covariates(baseline),
                                 str(Path(folder) / "no_label" / relative / filename)))
        # Execute the actual script body with fixture selections, without a second runner.
        path = Path(self.federated.__file__)
        tree = ast.parse(path.read_text())
        guard = next(node for node in tree.body if isinstance(node, ast.If)
                     and isinstance(node.test, ast.Compare)
                     and isinstance(node.test.left, ast.Name)
                     and node.test.left.id == "__name__")
        body = ast.Module(body=guard.body, type_ignores=[])
        with patch.object(self.federated, "experiments", selected), \
                patch.object(self.federated, "data_dir", str(self.root)), \
                patch.object(self.federated.sys, "argv", [str(path)]):
            exec(compile(body, str(path), "exec"), dict(vars(self.federated)))

    def output(self, folder="after", no_label=False, smpc=True, modality=None):
        parent = self.root / folder / ("no_label" if no_label else "label_aware")
        if modality:
            parent /= modality
        filename = "FedApp_corrected_data_smpc.tsv" if smpc else "FedApp_corrected_data.tsv"
        return parent / filename

    def assert_sources_preserved(self):
        self.assertEqual(self.sources, {p: p.read_bytes() for p in self.sources})
        self.assertFalse((self.clients[-1] / "config.yml").exists())

    def test_no_covariate_variant_retains_label_aware_filtering_and_settings(self):
        for smpc in (False, True):
            for changes in self.experiment.config_file_changes:
                changes["flimmaBatchCorrection"]["smpc"] = smpc
            variant = self.federated.without_covariates(self.experiment)
            self.assertIsNot(variant, self.experiment)
            self.assertEqual(variant.clients, self.experiment.clients)
            self.assertEqual(variant.fc_data_dir, self.experiment.fc_data_dir)
            self.assertIsNone(variant.config_file_changes)
            self.assertEqual(self.experiment.config_files[0]["flimmaBatchCorrection"]["covariates"],
                             ["unused"])
            self.assertEqual(self.experiment.config_file_changes[0]["flimmaBatchCorrection"]["covariates"],
                             ["Condition"])
            for i, config in enumerate(variant.config_files):
                details = config["flimmaBatchCorrection"]
                self.assertEqual(details["covariates"], [])
                self.assertEqual(details["min_samples"], 5)
                self.assertEqual(details["position"], i)
                self.assertEqual(details["smpc"], smpc)
                self.assertEqual(details["data_filename"], "expression.tsv")
        for client, changes in zip(self.clients, self.experiment.config_file_changes):
            design = pd.read_csv(client / "client_design.tsv", sep="\t")
            design["Batch"] = ["A", "B"] * 3
            design.to_csv(client / "client_design.tsv", sep="\t", index=False)
            changes["flimmaBatchCorrection"]["batch_col"] = "Batch"
        self.experiment.config_file_changes[-1]["flimmaBatchCorrection"]["min_samples"] = 10
        batched = self.federated.without_covariates(self.experiment)
        self.assertEqual([c["flimmaBatchCorrection"]["min_samples"]
                          for c in batched.config_files], [8, 8, 10])

    def test_selected_variants_use_the_same_campaign_and_preserve_source_files(self):
        self.run_campaign(protocols=(False, True))
        self.assertEqual(len(self.calls), 4)
        self.assertEqual(self.startups, [str(self.root)])
        raw = pd.concat(self.inputs, axis=1)
        for i, (clients, configs, image) in enumerate(self.calls):
            self.assertEqual(clients, [str(p) for p in self.clients])
            self.assertEqual(image, "baseline")
            for position, config in enumerate(configs):
                details = config["flimmaBatchCorrection"]
                self.assertEqual(details["covariates"], ["Condition"] if i % 2 == 0 else [])
                self.assertEqual(details["min_samples"], 0 if i % 2 == 0 else 5)
                self.assertEqual(details["position"], position)
                self.assertEqual(details["smpc"], i >= 2)
            matrix = read_matrix(self.output(no_label=bool(i % 2), smpc=i >= 2))
            pd.testing.assert_frame_equal(matrix, (raw + .25 * (i + 1)).rename_axis("rowname"))
        self.assertFalse((self.root / "inputs").exists())
        for no_label in (False, True):
            descriptor = self.output(no_label=no_label).with_name("datainfo.json")
            cohorts = json.loads(descriptor.read_text())["cohorts"]
            folders = [descriptor.parent / cohort["folder"] for cohort in cohorts]
            self.assertTrue(all((folder / "full_corrected_data.csv").exists() for folder in folders))
            self.assertEqual(
                [(folder / cohort["designfile"]["filename"]).resolve()
                 for folder, cohort in zip(folders, cohorts)],
                [(client / "client_design.tsv").resolve() for client in self.clients])
        self.assert_sources_preserved()

    def test_incomplete_repeated_or_unordered_results_are_rejected_before_publication(self):
        cases = [self.paths[:-1], [self.paths[0]] * 3, self.paths[::-1]]
        raw = self.inputs[0]
        for i, matrix in enumerate((raw.iloc[:, :-1], raw.iloc[:, ::-1],
                                   raw.iloc[:, [0, 0, 2, 3, 4, 5]])):
            cases.append([self.make_archive(0, matrix, f"invalid_{i}"), *self.paths[1:]])
        for i, paths in enumerate(cases):
            self.archive_override = paths
            folder = f"invalid_{i}"
            with self.subTest(names=[p.name for p in paths]), self.assertRaises(ValueError):
                self.run_campaign(folder, paired=False)
            self.assertFalse(self.output(folder).exists())
            self.assert_sources_preserved()

    def test_transport_and_paired_mask_failures_restore_existing_and_absent_configs(self):
        self.failure = "transport"
        self.run_campaign("failed_transport")
        self.assertEqual(len(self.calls), 2)
        self.assertFalse(self.output("failed_transport").exists())
        self.assertFalse(self.output("failed_transport", no_label=True).exists())
        self.assert_sources_preserved()
        self.calls.clear()
        self.failure = "mask"
        with self.assertRaisesRegex(ValueError, "missing-value"):
            self.run_campaign("failed_mask")
        self.assertTrue(self.output("failed_mask").exists())
        self.assertFalse(self.output("failed_mask", no_label=True).exists())
        self.assert_sources_preserved()

    def test_quartet_arm_first_outputs_keep_paired_missing_value_validation(self):
        self.failure = "mask"
        with self.assertRaisesRegex(ValueError, "missing-value"):
            self.run_campaign("quartet/after", modality="Proteomics")
        self.assertTrue(self.output("quartet/after", modality="Proteomics").exists())
        self.assertFalse(self.output("quartet/after", no_label=True,
                                     modality="Proteomics").exists())
        self.assert_sources_preserved()


class DestinationTests(unittest.TestCase):
    def test_canonical_destinations_are_rejected_and_isolated_root_is_allowed(self):
        canonical = (
            "evaluation_data", "evaluation", "evaluation_utils", "batchcorrection",
            "evaluation_classification_after_correction",
            "evaluation_clusterization_after_correction", ".git",
        )
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory) / "repository"
            run = Path(directory) / "run"
            for path in [repo, *(repo / name / "new_run" for name in canonical)]:
                with self.subTest(path=path), self.assertRaises(ValueError):
                    validate_experiment_dir(repo, path)
            self.assertEqual(validate_experiment_dir(repo, run), run)

    def test_canonical_outputs_use_sibling_arms_before_quartet_modalities(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            ecoli_after = repo / "evaluation_data/ecoli/after"
            config = Mock(sites_root=repo / "evaluation_data/ecoli/before",
                          corrected_central=ecoli_after / "label_aware/central.tsv")
            with patch.object(correction, "REPO_ROOT", repo), patch.object(
                correction, "dataset_configs", return_value={"ecoli": config,
                                                              "quartet_multiomics": Mock()},
            ):
                units = correction._source_units(["ecoli", "quartet_multiomics"])
                matrix = pd.DataFrame([[1., np.nan]], index=["gene"], columns=["a", "b"])
                for unit in units:
                    write_feature_matrix(matrix, unit["central_input"])
                with patch.object(correction, "_unit_sources", return_value=({}, {}, {})), \
                        patch.object(correction, "_central_designs"), \
                        patch.object(correction, "_run_limma") as run_limma:
                    correction.correct_canonical(
                        ["ecoli", "quartet_multiomics"], rscript="unused",
                        arms=["label_aware", "no_label"])
            expected = [ecoli_after / arm / "central.tsv" for arm in ("label_aware", "no_label")]
            quartet_after = repo / "evaluation_data/quartet_multiomics/after"
            expected.extend(
                quartet_after / arm / modality / "intensities_log_Rcorrected_UNION.tsv"
                for modality in ("Transcriptomics", "Proteomics", "Metabolomics")
                for arm in ("label_aware", "no_label"))
            self.assertEqual([call.args[3] for call in run_limma.call_args_list], expected)
            self.assertFalse(list(repo.rglob("metadata.tsv")))


class NumericalSmokeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="label-free-numerical-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.repo = self.root / "repository"
        before = self.repo / "evaluation_data/fixture/before"
        after_root = before.parent / "after"
        after = after_root / "label_aware"
        before.mkdir(parents=True)
        after.mkdir(parents=True)
        rng = np.random.default_rng(SEED)
        self.names = ["client_a", "client_b", "client_c"]
        self.samples = [f"{name}_sample_{i:02d}" for name in self.names for i in range(12)]
        targets = np.tile([0, 1] * 6, 3)
        values = rng.normal(size=(7, 36)) + targets * np.arange(1, 8)[:, None]
        values += np.repeat([2., -1., 4.], 12)
        values[-1, :] = np.nan
        for i in range(3):
            values[-1, i * 12:i * 12 + 4] = rng.normal(size=4)
        values[1, 2], values[2, 16], values[3, 31] = np.nan, np.nan, np.nan
        self.raw = pd.DataFrame(values, index=[*RETAINED, "weak_probe"], columns=self.samples)
        write_feature_matrix(self.raw, before / "raw.tsv")
        write_feature_matrix(self.raw.loc[RETAINED], before / "central_input.tsv")
        historical = self.raw.loc[RETAINED].copy() + 999.
        for backend in ("central", "federated"):
            write_feature_matrix(historical, after / f"{backend}.tsv")
        cohorts = []
        for i, name in enumerate(self.names):
            client = before / name
            client.mkdir()
            samples = self.samples[i * 12:(i + 1) * 12]
            write_feature_matrix(self.raw[samples], client / "data.tsv")
            pd.DataFrame({"file": samples, "Condition": targets[i * 12:(i + 1) * 12],
                          "extra_target_copy": targets[i * 12:(i + 1) * 12]}).to_csv(
                client / "design.tsv", sep="\t", index=False,
            )
            config = {"flimmaBatchCorrection": {
                "batch_col": None, "covariates": ["Condition"],
                "data_filename": "data.tsv", "design_filename": "design.tsv",
                "design_separator": "\t", "expression_file_flag": True,
                "index_col": "rowname", "min_samples": 0,
                "normalizationMethod": None, "position": i, "separator": "\t", "smpc": True,
            }}
            (client / "config.yml").write_text(yaml.safe_dump(config))
            cohorts.append({"name": name, "folder": name, "designfile": {
                "filename": "design.tsv", "separator": "\t", "rotation": "samples x features",
                "samplename_column": "file",
            }})
        (before / "datainfo.json").write_text(json.dumps({
            "data_name": "Numerical fixture", "covariates": ["Condition"],
            "prediction_targets": ["Condition"], "cohorts": cohorts,
            "datafile": {"filename": "data.tsv", "separator": "\t",
                         "rotation": "features x samples", "featurename_column": "rowname"},
        }))
        self.source = {"dataset": DATASET, "modality": "main", "source_root": before,
                       "raw": before / "raw.tsv", "central_input": before / "central_input.tsv",
                       "after_root": after_root,
                       "central": after / "central.tsv",
                       "federated": after / "federated.tsv"}
        # Replace repository discovery only; numerical functions remain real.
        self.enterContext(patch.object(correction, "REPO_ROOT", self.repo))
        self.enterContext(patch.object(correction, "_source_units", return_value=[self.source]))
        self.canonical_hashes = self.hashes(before.parent)
        self.run = self.root / "run"

    def hashes(self, root):
        return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in root.rglob("*") if p.is_file()}

    def output(self, arm):
        return self.run / "correction" / DATASET / "main" / "central" / arm / "matrix.tsv"

    def correct(self, arms=None):
        return correction.correct_canonical(
            [DATASET], arms=arms or ["label_aware", "no_label"],
            rscript=str(RSCRIPT), output_dir=self.run,
        )

    def clients(self, arm):
        clients = []
        for name in self.names:
            path = self.root / "client_inputs" / name
            if not path.exists():
                shutil.copytree(self.source["source_root"] / name, path)
            config = yaml.safe_load((path / "config.yml").read_text())
            details = config["flimmaBatchCorrection"]
            details["covariates"] = ["Condition"] if arm == "label_aware" else []
            details["min_samples"] = 5
            (path / "config.yml").write_text(yaml.safe_dump(config))
            client = Client()
            client.config_based_init(name, input_folder=str(path), use_hashing=True)
            clients.append(client)
        return clients

    def central_inputs(self):
        directory = self.root / "central_inputs"
        matrix = read_matrix(self.source["central_input"])
        write_feature_matrix(matrix, directory / "data.tsv")
        _, configs, designs = correction._unit_sources(self.source)
        correction._central_designs(directory, matrix, configs, designs, self.names)
        return directory

    @requires_r
    def test_actual_client_matches_limma_for_both_arms(self):
        self.source["central"].unlink()
        with patch("docker.from_env") as docker, contextlib.redirect_stdout(io.StringIO()):
            correction.main(["correct", "--datasets", DATASET, "--rscript", str(RSCRIPT)])
            docker.assert_not_called()
        canonical = read_matrix(self.source["central"])
        self.canonical_hashes = self.hashes(self.source["source_root"].parent)
        with patch("docker.from_env") as docker, contextlib.redirect_stdout(io.StringIO()):
            correction.main(["correct", "--datasets", DATASET, "--output-dir", str(self.run),
                             "--rscript", str(RSCRIPT)])
            docker.assert_not_called()
        self.assertFalse((self.run / "run.json").exists())
        self.assertFalse((self.run / "datasets.yaml").exists())
        self.assertFalse(list(self.run.rglob("federated")))
        with contextlib.redirect_stdout(io.StringIO()):
            for arm in ("label_aware", "no_label"):
                clients = self.clients(arm)
                result = client_correction(clients, 5)
                central = read_matrix(self.output(arm))
                result = validate_corrected(result, central)
                np.testing.assert_allclose(result, central, atol=1e-10, rtol=0, equal_nan=True)
                if arm == "label_aware":
                    pd.testing.assert_frame_equal(central, canonical)
                self.assertEqual(set(result.index), set(RETAINED))
                if arm == "no_label":
                    self.assertTrue(all(c.hash2variable == {} for c in clients))
                    self.assertTrue(all(list(c.design.columns) == ["intercept", *self.names[:-1]]
                                        for c in clients))
        saved = self.hashes(self.run)
        with self.assertRaisesRegex(ValueError, "fresh run"):
            self.correct()
        self.assertEqual(saved, self.hashes(self.run))
        self.assertEqual(self.canonical_hashes, self.hashes(self.source["source_root"].parent))

    @requires_r
    def test_matched_arms_share_a_snapshot_when_source_values_change(self):
        self.correct()
        baseline = {arm: read_matrix(self.output(arm)) for arm in ("label_aware", "no_label")}
        self.run = self.root / "source_changed"
        frozen = read_matrix(self.source["central_input"])
        run_limma = correction._run_limma
        inputs = []

        def correct_from_snapshot(input_file, *args):
            self.assertNotEqual(input_file.resolve(), self.source["central_input"].resolve())
            pd.testing.assert_frame_equal(read_matrix(input_file), frozen)
            inputs.append(input_file)
            run_limma(input_file, *args)
            if len(inputs) == 1:
                write_feature_matrix(frozen + 400., self.source["central_input"])

        with patch.object(correction, "_run_limma", side_effect=correct_from_snapshot):
            self.correct()
        self.assertEqual(len(inputs), 2)
        self.assertEqual(inputs[0], inputs[1])
        for arm, expected in baseline.items():
            pd.testing.assert_frame_equal(read_matrix(self.output(arm)), expected)

    @requires_r
    def test_no_label_correction_requires_no_targets_smpc_or_evaluation_imports(self):
        descriptor_path = self.source["source_root"] / "datainfo.json"
        descriptor = json.loads(descriptor_path.read_text())
        descriptor.pop("prediction_targets")
        descriptor_path.write_text(json.dumps(descriptor))
        for name in self.names:
            folder = self.source["source_root"] / name
            config_path = folder / "config.yml"
            config = yaml.safe_load(config_path.read_text())
            config["flimmaBatchCorrection"]["smpc"] = False
            config_path.write_text(yaml.safe_dump(config))
            design_path = folder / "design.tsv"
            pd.read_csv(design_path, sep="\t")[["file"]].to_csv(
                design_path, sep="\t", index=False,
            )
        source_bytes = self.hashes(self.source["source_root"].parent)
        original_import = __import__

        def import_correction_only(name, *args, **kwargs):
            if name.startswith(("evaluation_classification_after_correction",
                                "evaluation_clusterization_after_correction")):
                raise AssertionError(f"Central correction imported evaluation code: {name}")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=import_correction_only):
            self.correct(arms=["no_label"])
        result = read_matrix(self.output("no_label"))
        self.assertEqual(set(result.index), set(RETAINED))
        self.assertFalse(self.output("label_aware").exists())
        self.assertEqual(source_bytes, self.hashes(self.source["source_root"].parent))

    @requires_r
    def test_shared_correction_matches_limma_for_notebook_designs_and_no_covariates(self):
        prepared = self.central_inputs()
        batches = pd.read_csv(prepared / "batch.tsv", sep="\t")
        batches["batch_order"] = batches.batch_order.max() + 1 - batches.batch_order
        batches.to_csv(prepared / "batch.tsv", sep="\t", index=False)
        matrix = read_matrix(prepared / "data.tsv")
        matrix.loc[matrix.index[0], batches.loc[batches.batch.eq(batches.batch.iloc[0]), "file"]] = np.nan
        write_feature_matrix(matrix, prepared / "data.tsv")
        script = r'''
args <- commandArgs(trailingOnly = TRUE)
source(args[1])
expr <- as.matrix(read.delim(file.path(args[2], "data.tsv"), row.names = 1,
                            check.names = FALSE))
batch_table <- read.delim(file.path(args[2], "batch.tsv"))
batch <- batch_table$batch
condition <- factor(rep(c("Normal", "Tumor"), length.out = ncol(expr)))
donor <- factor(rep(c("D6", "D5", "F7", "M8"), length.out = ncol(expr)),
                levels = c("D6", "D5", "F7", "M8"))
for (design in list(model.matrix(~ condition), model.matrix(~ donor), matrix(1, ncol(expr), 1))) {
  for (batches in list(batch, factor(batch, levels = rev(unique(batch))))) {
    expected <- limma::removeBatchEffect(expr, batch = batches, design = design)
    actual <- central_batch_correction(expr, batches, design)
    stopifnot(isTRUE(all.equal(actual, expected, tolerance = 1e-12)),
              identical(is.na(actual), is.na(expr)),
              identical(dimnames(actual), dimnames(expr)))
  }
  levels <- unique(batch[order(batch_table$batch_order)])
  expected <- limma::removeBatchEffect(expr, batch = factor(batch, levels = levels), design = design)
  actual <- central_batch_correction(expr, batch, design, batch_levels = levels)
  stopifnot(isTRUE(all.equal(actual, expected, tolerance = 1e-12)))
}
write.table(data.frame(rowname = rownames(expected), expected, check.names = FALSE),
            file.path(args[2], "expected.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
'''
        completed = subprocess.run(
            [str(RSCRIPT), "-e", script, str(Path(correction.__file__).with_name("central_correction.R")),
             str(prepared)], capture_output=True, text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        output = prepared / "actual.tsv"
        correction._run_limma(prepared / "data.tsv", prepared, "no_label", output, str(RSCRIPT), matrix)
        np.testing.assert_allclose(read_matrix(output),
                                   read_matrix(prepared / "expected.tsv"),
                                   atol=1e-12, rtol=0, equal_nan=True)

    def test_fixed_threshold_and_label_shuffle_boundary(self):
        with contextlib.redirect_stdout(io.StringIO()):
            for arm, threshold, expected in (("label_aware", 5, 6), ("no_label", 5, 6),
                                              ("no_label", 4, 7)):
                result = client_correction(self.clients(arm), threshold)
                self.assertEqual(len(result), expected)
            original = client_correction(self.clients("no_label"), 5)
            for name in self.names:
                path = self.root / "client_inputs" / name / "design.tsv"
                design = pd.read_csv(path, sep="\t")
                design["Condition"] = 1 - design.Condition
                design["extra_target_copy"] = design.Condition
                design.to_csv(path, sep="\t", index=False)
            shuffled = client_correction(self.clients("no_label"), 5)
        pd.testing.assert_frame_equal(original, shuffled)
        central = self.central_inputs() / "no_label_design.tsv"
        self.assertEqual(list(pd.read_csv(central, sep="\t").columns), ["file", "intercept"])
        self.assertEqual(self.canonical_hashes, self.hashes(self.source["source_root"].parent))

    def test_external_failure_leaves_no_partial_correction_matrix(self):
        def fail_correction(command, **kwargs):
            Path(command[-1]).write_text("partial external output\n")
            return subprocess.CompletedProcess(command, 7, "", "fixture external failure\n")

        with patch.object(correction.subprocess, "run", side_effect=fail_correction):
            with self.assertRaisesRegex(RuntimeError, "fixture external failure"):
                self.correct(arms=["no_label"])
        output = self.output("no_label")
        self.assertFalse(output.exists())
        self.assertFalse(output.with_suffix(".pending.tsv").exists())
        self.assertEqual(self.canonical_hashes, self.hashes(self.source["source_root"].parent))


if __name__ == "__main__":
    unittest.main()
