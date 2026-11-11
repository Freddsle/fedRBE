"""
FeatureCloud federated k-means infrastructure: test execution, zip extraction,
and result aggregation.

This module isolates all FeatureCloud-specific logic so that the evaluation
notebooks can run central k-means without any Docker/FC dependency.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pandas as pd

from evaluation_utils.real_datasets_utils import aggregate_fed_clusters


# ---------------------------------------------------------------------------
# Zip / result helpers
# ---------------------------------------------------------------------------

def wait_for_paths(paths: Iterable[Path], timeout: int = 120) -> None:
    """Block until all *paths* exist on disk, or raise after *timeout* seconds."""
    deadline = time.time() + timeout
    missing = [p for p in paths if not p.exists()]
    while missing and time.time() < deadline:
        time.sleep(2)
        missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Result files not found: {missing}")


def parse_test_id(paths: Iterable[Path]) -> Optional[int]:
    """Extract the numeric test ID from FeatureCloud result zip filenames."""
    test_ids = set()
    for path in paths:
        match = re.search(r"results_test_(\d+)_client_", path.name)
        if match:
            test_ids.add(int(match.group(1)))
    if len(test_ids) == 1:
        return test_ids.pop()
    return None


def extract_clustering(
    zip_path: Path,
    output_dir: Path,
    client_idx: int,
    k_values: Sequence[int],
) -> None:
    """Extract clustering CSVs from a FeatureCloud result zip.

    For each k, extracts ``kmeans/K_{k}/clustering.csv`` and renames it to
    ``{client_idx}_{k}_clustering.csv`` in *output_dir*.
    """
    with zipfile.ZipFile(zip_path, "r") as zh:
        members = set(zh.namelist())
        for k in k_values:
            member = f"kmeans/K_{k}/clustering.csv"
            if member not in members:
                raise FileNotFoundError(f"Missing {member} in {zip_path}")
            zh.extract(member, path=output_dir)
            extracted = output_dir / member
            shutil.move(str(extracted), str(output_dir / f"{client_idx}_{k}_clustering.csv"))

    kmeans_dir = output_dir / "kmeans"
    if kmeans_dir.exists():
        shutil.rmtree(kmeans_dir)


def validate_client_samples(
    run_output: Path,
    metadata: pd.DataFrame,
    client_names: Sequence[str],
    k_values: Sequence[int],
) -> None:
    """Require exactly the requested samples in every client's clustering."""
    for idx, client in enumerate(client_names, start=1):
        expected = set(metadata.loc[metadata.lab.eq(client), "file"].astype(str))
        for k in k_values:
            path = run_output / f"{idx}_{k}_clustering.csv"
            frame = pd.read_csv(path, sep=";", index_col=0)
            samples = frame.index.astype(str)
            if not expected or not samples.is_unique or set(samples) != expected:
                raise ValueError(f"Incomplete or unexpected samples for {client} in {path}")
            if frame.shape[1] != 1 or frame.isna().any().any():
                raise ValueError(f"Missing clustering labels in {path}")


# ---------------------------------------------------------------------------
# FeatureCloud SDK integration
# ---------------------------------------------------------------------------

def import_featurecloud_utils():
    """Import the featurecloud_api_extension module, or raise with guidance."""
    try:
        from evaluation_utils import featurecloud_api_extension as fc_utils
    except Exception as exc:
        raise RuntimeError(
            "FeatureCloud SDK or dependencies are unavailable. "
            "Install FeatureCloud to run federated k-means."
        ) from exc
    return fc_utils


def run_single_federated_variant(
    dataset_name: str,
    variant_label: str,
    variant_input_dir: Path,
    dataset_root: Path,
    metadata: pd.DataFrame,
    client_names: Sequence[str],
    k_values: Sequence[int],
    app_image: str,
    controller_host: str,
    query_interval: int,
    timeout: int,
    seed: int,
) -> Path:
    """Run a federated k-means variant (before or after correction).

    Extracted clustering files and the aggregated metadata TSV are written to
    seed-specific paths, so each seed of a repeated run remains auditable.

    Returns the path to the saved metadata-with-clusters TSV.
    """
    run_dir_name = f"seed_{seed:03d}"

    run_output = dataset_root / "fc_kmeans_res" / variant_label / run_dir_name
    run_output.mkdir(parents=True, exist_ok=True)

    kmeans_run_dir = dataset_root / "kmeans_res" / "runs"
    kmeans_run_dir.mkdir(parents=True, exist_ok=True)
    output_path = kmeans_run_dir / f"{run_dir_name}_metadata_{variant_label}_fedclusters.tsv"

    # Launch a real FeatureCloud test
    fc_utils = import_featurecloud_utils()
    staged_output: Optional[Path] = None
    aggregated: Optional[pd.DataFrame] = None

    class KMeansExperiment(fc_utils.Experiment):
        def _set_config_files(self) -> None:
            return

        def _check_test(self, exp_id: int) -> List[dict]:
            nonlocal staged_output, aggregated
            instances = super()._check_test(exp_id)
            attempt_dir = Path(tempfile.mkdtemp(prefix=".attempt_", dir=run_output))
            try:
                if sorted(info["id"] for info in instances) != list(range(len(client_names))):
                    raise ValueError("FeatureCloud did not return every requested client")
                zip_paths = [
                    variant_input_dir / "tests" / fc_utils.TESTS_DIR /
                    f"results_test_{exp_id}_client_{info['id']}_{info['name']}.zip"
                    for info in sorted(instances, key=lambda info: info["id"])
                ]
                wait_for_paths(zip_paths, timeout=timeout)
                for idx, path in enumerate(zip_paths, start=1):
                    copied = attempt_dir / path.name
                    shutil.copy2(path, copied)
                    extract_clustering(copied, attempt_dir, idx, k_values)
                validate_client_samples(attempt_dir, metadata, client_names, k_values)
                result = aggregate_fed_clusters(
                    metadata, attempt_dir, client_names, k_values,
                )
            except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
                failures = run_output / "failed_attempts"
                failures.mkdir(exist_ok=True)
                saved = attempt_dir.rename(failures / attempt_dir.name.removeprefix("."))
                raise fc_utils._RetryableTestError(
                    f"Incomplete k-means test {exp_id}: {exc}. Evidence saved in {saved}"
                ) from exc
            staged_output, aggregated = attempt_dir, result
            return instances

        def _coordinator_index(self, instances: List[dict]) -> int:
            coord_idx: Optional[int] = None
            for idx, info in enumerate(instances):
                if bool(info.get("coordinator", False)):
                    coord_idx = idx
            if coord_idx is None:
                print(
                    "[WARN] No coordinator reported by controller metadata. "
                    "Continuing with fallback coordinator index 0."
                )
                coord_idx = 0 if instances else -1
            return coord_idx

    client_dirs = [variant_input_dir / client for client in client_names]
    for cd in client_dirs:
        if not (cd / "config_kmeans.yml").exists():
            raise FileNotFoundError(f"Missing config_kmeans.yml in {cd}")

    experiment = KMeansExperiment(
        name=f"real_{dataset_name}_{variant_label}_kmeans",
        clients=[str(p) for p in client_dirs],
        app_image_name=app_image,
        fc_data_dir=str(variant_input_dir),
        controller_host=controller_host,
        query_interval=query_interval,
        timeout=timeout,
    )

    print(
        f"[{dataset_name}] Starting FeatureCloud controller "
        f"for '{variant_label}' data"
    )
    experiment._startup()

    print(f"[{dataset_name}] Running federated k-means for '{variant_label}' data")
    zip_files_str, _, exp_meta = experiment.run_test()
    zip_paths = [Path(p) for p in zip_files_str]
    test_id = parse_test_id(zip_paths)

    if staged_output is None or aggregated is None:
        raise RuntimeError("FeatureCloud returned without validated k-means results")
    for idx in range(1, len(client_names) + 1):
        for k in k_values:
            name = f"{idx}_{k}_clustering.csv"
            (staged_output / name).replace(run_output / name)
    shutil.rmtree(staged_output)
    aggregated.to_csv(output_path, sep="\t", index=False)

    manifest = {
        "dataset": dataset_name,
        "variant": variant_label,
        "run_label": run_dir_name,
        "seed": seed,
        "test_id": test_id,
        "k_values": list(k_values),
        "client_names": list(client_names),
        "data_dir": str(variant_input_dir),
        "app_image": app_image,
        "zip_files": [p.name for p in zip_paths],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "experiment_meta": {
            "experiment_name": exp_meta.experiment_name,
            "input_hashes": exp_meta.input_hashes,
            "config": exp_meta.config,
        },
    }
    (run_output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )
    print(f"[{dataset_name}] Federated output saved: {output_path}")
    return output_path
