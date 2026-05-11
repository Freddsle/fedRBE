"""
FeatureCloud federated k-means infrastructure: Docker image management,
test execution, zip extraction, and result aggregation.

This module isolates all FeatureCloud-specific logic so that the evaluation
notebooks can run central k-means without any Docker/FC dependency.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from evaluation_utils.real_datasets_utils import aggregate_fed_clusters


# ---------------------------------------------------------------------------
# Docker helpers
# ---------------------------------------------------------------------------

def docker_image_exists(image_name: str) -> bool:
    """Return True if a Docker image with *image_name* exists locally."""
    cmd = ["docker", "image", "inspect", image_name]
    proc = subprocess.run(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
    )
    return proc.returncode == 0


def ensure_app_image(
    app_image: str,
    app_source_dir: Path,
    auto_build: bool,
    build_timeout: int,
) -> None:
    """Build the FeatureCloud app Docker image if it doesn't exist."""
    if docker_image_exists(app_image):
        return

    if not auto_build:
        raise RuntimeError(
            f"App image '{app_image}' not found. Build it first with:\n"
            f"  featurecloud app build {app_source_dir} {app_image}"
        )

    if not app_source_dir.exists():
        raise FileNotFoundError(f"App source directory not found: {app_source_dir}")

    print(
        f"App image '{app_image}' not found. "
        f"Building from local source: {app_source_dir}"
    )
    build_cmd = ["featurecloud", "app", "build", str(app_source_dir), app_image]
    try:
        subprocess.run(build_cmd, check=True, timeout=build_timeout)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "FeatureCloud CLI is not installed or not in PATH. "
            "Cannot auto-build missing app image."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Timed out while building app image '{app_image}' "
            f"(timeout={build_timeout}s)."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Failed to build app image '{app_image}' from {app_source_dir}."
        ) from exc

    if not docker_image_exists(app_image):
        raise RuntimeError(
            f"Image '{app_image}' is still missing after build. "
            "Please verify Docker/FeatureCloud build output."
        )


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


def has_extracted_clustering(
    run_output: Path, num_clients: int, k_values: Sequence[int]
) -> bool:
    """Check whether all expected per-client clustering CSVs already exist."""
    for idx in range(1, num_clients + 1):
        for k in k_values:
            if not (run_output / f"{idx}_{k}_clustering.csv").exists():
                return False
    return True


def select_latest_test_zip_files(zip_files: Sequence[Path]) -> List[Path]:
    """From a list of FC result zips, return those belonging to the latest test ID."""
    if not zip_files:
        return []
    grouped: Dict[int, List[Path]] = {}
    for zf in zip_files:
        match = re.search(r"results_test_(\d+)_client_", zf.name)
        if not match:
            continue
        test_id = int(match.group(1))
        grouped.setdefault(test_id, []).append(zf)
    if not grouped:
        return sorted(zip_files, key=lambda p: p.name)
    latest_test_id = max(grouped)
    return sorted(grouped[latest_test_id], key=lambda p: p.name)


def extract_clustering(
    zip_path: Path,
    output_dir: Path,
    client_idx: int,
    k_values: Sequence[int],
    keep_extracted: bool,
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
            target = output_dir / f"{client_idx}_{k}_clustering.csv"
            if keep_extracted:
                shutil.copy2(extracted, target)
            else:
                shutil.move(str(extracted), str(target))

    if not keep_extracted:
        kmeans_dir = output_dir / "kmeans"
        if kmeans_dir.exists():
            shutil.rmtree(kmeans_dir)


# ---------------------------------------------------------------------------
# Aggregate existing federated outputs
# ---------------------------------------------------------------------------

def aggregate_existing_federated_output(
    dataset_name: str,
    variant_label: str,
    variant_input_dir: Path,
    run_output: Path,
    k_values: Sequence[int],
    num_clients: int,
    keep_extracted: bool,
) -> None:
    """Re-use existing federated zip outputs without running a new FC test."""
    if has_extracted_clustering(run_output, num_clients, k_values):
        print(
            f"[{dataset_name}] Using existing extracted clustering "
            f"files for '{variant_label}'"
        )
        return

    # Look for zip files in run_output first, then in the inputs/tests dir
    zip_candidates = select_latest_test_zip_files(
        list(run_output.glob("results_test_*_client_*.zip"))
    )
    if not zip_candidates:
        tests_dir = variant_input_dir / "tests" / "tests"
        zip_candidates = select_latest_test_zip_files(
            list(tests_dir.glob("results_test_*_client_*.zip"))
        )

    if not zip_candidates:
        raise FileNotFoundError(
            f"No existing federated outputs found for '{variant_label}'. Checked:\n"
            f" - {run_output}\n"
            f" - {variant_input_dir / 'tests' / 'tests'}\n"
            "Use the federated runs notebook to launch a new FeatureCloud run."
        )

    print(
        f"[{dataset_name}] Aggregating existing federated zip outputs "
        f"for '{variant_label}' ({len(zip_candidates)} zip files)"
    )

    for client_idx, zf in enumerate(
        sorted(zip_candidates, key=lambda p: p.name), start=1
    ):
        source_zip = zf
        if zf.parent != run_output:
            target_zip = run_output / zf.name
            shutil.copy2(zf, target_zip)
            source_zip = target_zip

        extract_clustering(
            source_zip,
            run_output,
            client_idx=client_idx,
            k_values=k_values,
            keep_extracted=keep_extracted,
        )
        if source_zip.parent == run_output and source_zip.exists():
            source_zip.unlink()

    if not has_extracted_clustering(run_output, num_clients, k_values):
        raise FileNotFoundError(
            f"Could not build all expected clustering files "
            f"for '{variant_label}' in {run_output}"
        )


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
    keep_extracted: bool,
    aggregate_only: bool,
) -> Path:
    """Run or aggregate a federated k-means variant (before or after correction).

    When *aggregate_only* is True, existing zip outputs are re-used without
    launching a new FeatureCloud test.

    Returns the path to the saved metadata-with-clusters TSV.
    """
    run_output = dataset_root / "fc_kmeans_res" / variant_label / "1_fed_clustering"
    run_output.mkdir(parents=True, exist_ok=True)

    kmeans_run_dir = dataset_root / "kmeans_res" / "runs"
    kmeans_run_dir.mkdir(parents=True, exist_ok=True)
    output_path = kmeans_run_dir / (
        "1_metadata_before_fedclusters.tsv"
        if variant_label == "before"
        else "1_metadata_after_fedclusters.tsv"
    )

    if aggregate_only:
        aggregate_existing_federated_output(
            dataset_name=dataset_name,
            variant_label=variant_label,
            variant_input_dir=variant_input_dir,
            run_output=run_output,
            k_values=k_values,
            num_clients=len(client_names),
            keep_extracted=keep_extracted,
        )
        aggregated = aggregate_fed_clusters(
            metadata=metadata,
            run_output=run_output,
            client_names=client_names,
            k_values=k_values,
        )
        aggregated.to_csv(output_path, sep="\t", index=False)
        print(
            f"[{dataset_name}] Aggregated existing federated output saved: {output_path}"
        )
        return output_path

    # Launch a real FeatureCloud test
    fc_utils = import_featurecloud_utils()

    class KMeansExperiment(fc_utils.Experiment):
        def _set_config_files(self) -> None:
            return

        def run_test(self, retry: int = 0):
            if retry == 0:
                print("_______________EXPERIMENT_______________")

            exp_id, exp_meta = self._start_test(retry=retry)
            instances, _dirs = self._check_test(exp_id=exp_id, retry=retry)
            print(f"instances: {instances}")
            print("TEST DONE:")

            instances.sort(key=lambda item: item["id"])
            result_files: List[str] = []
            coord_idx: Optional[int] = None
            tests_subdir = getattr(fc_utils, "TESTS_DIR", "tests")

            for idx, info in enumerate(instances):
                if bool(info.get("coordinator", False)):
                    coord_idx = idx
                file_path = (
                    Path(self.fc_data_dir)
                    / "tests"
                    / tests_subdir
                    / f"results_test_{exp_id}_client_{info['id']}_{info['name']}.zip"
                )
                result_files.append(str(file_path))

            if coord_idx is None:
                print(
                    "[WARN] No coordinator reported by controller metadata. "
                    "Continuing with fallback coordinator index 0."
                )
                coord_idx = 0 if result_files else -1

            print("_______________EXPERIMENT FINISHED SUCCESSFULLY_______________")
            return result_files, coord_idx, exp_meta

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
    wait_for_paths(zip_paths, timeout=timeout)
    test_id = parse_test_id(zip_paths)

    for idx, zp in enumerate(zip_paths, start=1):
        copied_zip = run_output / zp.name
        shutil.copy2(zp, copied_zip)
        extract_clustering(
            copied_zip,
            run_output,
            client_idx=idx,
            k_values=k_values,
            keep_extracted=keep_extracted,
        )
        if copied_zip.exists():
            copied_zip.unlink()

    aggregated = aggregate_fed_clusters(
        metadata=metadata,
        run_output=run_output,
        client_names=client_names,
        k_values=k_values,
    )
    aggregated.to_csv(output_path, sep="\t", index=False)

    manifest = {
        "dataset": dataset_name,
        "variant": variant_label,
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
