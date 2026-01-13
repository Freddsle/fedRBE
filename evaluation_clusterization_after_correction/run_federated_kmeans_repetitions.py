#!/usr/bin/env python3
"""
Run federated k-means clustering via FeatureCloud across multiple repetitions.

This script:
1) (Optional) prepares per-run input data from simulated_rotation intermediate files.
2) Runs the fc_kmeans_upd app on lab1/lab2/lab3.
3) Extracts clustering.csv for requested K values and stores them in a run folder.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
import zipfile
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd

try:
    from FeatureCloud.api.imp.controller import commands as fc_controller
    from FeatureCloud.api.imp.test import commands as fc_test
except ImportError as exc:  # pragma: no cover - only needed when running the script
    raise SystemExit(
        "FeatureCloud SDK is not available. Install FeatureCloud to run this script."
    ) from exc

DEFAULT_CONTROLLER_HOST = "http://localhost:8000"
DEFAULT_MODES = ("balanced", "mild_imbalanced", "strong_imbalanced")
DEFAULT_CLIENT_DIRS = ("lab1", "lab2", "lab3")
DEFAULT_K_VALUES = (2, 3)


def parse_csv_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_k_values(value: str) -> List[int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    return [int(item) for item in items]


def read_metadata(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", index_col=0)
    missing = {"file", "condition", "lab"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in metadata {path}: {sorted(missing)}")
    return df


def load_intensities(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    if "rowname" not in df.columns:
        raise ValueError(f"Missing 'rowname' column in {path}")
    return df.set_index("rowname")


def write_intensities(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    out.insert(0, "rowname", out.index)
    out.to_csv(path, sep="\t", index=False, quoting=csv.QUOTE_NONNUMERIC)


def write_design(df: pd.DataFrame, path: Path, condition_levels: Sequence[str]) -> None:
    out = df[["file", "condition", "lab"]].copy()
    out["A"] = pd.Categorical(out["condition"], categories=condition_levels).codes
    out.to_csv(path, sep="\t", index=False, quoting=csv.QUOTE_NONNUMERIC)


def prepare_inputs(
    mode_root: Path,
    run_id: int,
    input_variant: str,
    client_dirs: Sequence[str],
) -> Path:
    metadata = read_metadata(mode_root / "all_metadata.tsv")
    condition_levels = sorted(metadata["condition"].dropna().unique())

    if input_variant == "before":
        intensities_path = (
            mode_root
            / "before"
            / "intermediate"
            / f"{run_id}_intensities_data.tsv"
        )
        target_base = mode_root / "before"
    elif input_variant == "before_corrected":
        intensities_path = mode_root / "after" / "runs" / f"{run_id}_FedSim_corrected.tsv"
        target_base = mode_root / "before_corrected"
    else:
        raise ValueError(f"Unknown input variant: {input_variant}")

    if not intensities_path.exists():
        raise FileNotFoundError(f"Missing intensities file: {intensities_path}")

    intensities = load_intensities(intensities_path)
    target_base.mkdir(parents=True, exist_ok=True)

    for lab in client_dirs:
        lab_meta = metadata[metadata["lab"] == lab].copy()
        if lab_meta.empty:
            raise ValueError(f"No metadata rows found for lab '{lab}' in {mode_root}")

        missing_cols = [f for f in lab_meta["file"] if f not in intensities.columns]
        if missing_cols:
            raise ValueError(
                f"Missing columns in intensities for lab '{lab}': {missing_cols[:5]}"
            )

        lab_dir = target_base / lab
        lab_dir.mkdir(parents=True, exist_ok=True)

        # Ensure config is present (the app looks for config_kmeans*.yml)
        config_dest = lab_dir / "config_kmeans.yml"
        if not config_dest.exists():
            config_src = mode_root / "before" / lab / "config_kmeans.yml"
            if not config_src.exists():
                raise FileNotFoundError(f"Missing config_kmeans.yml in {config_src}")
            shutil.copy2(config_src, config_dest)

        write_intensities(intensities[lab_meta["file"]], lab_dir / "intensities.tsv")
        write_design(lab_meta, lab_dir / "design.tsv", condition_levels)

    return target_base


def start_controller(data_dir: Path) -> None:
    fc_controller.stop(name=fc_controller.DEFAULT_CONTROLLER_NAME)
    time.sleep(5)
    fc_controller.start(
        name=fc_controller.DEFAULT_CONTROLLER_NAME,
        port=8000,
        data_dir=str(data_dir),
        controller_image="",
        with_gpu=False,
        mount="",
        blockchain_address="",
    )
    time.sleep(5)


def run_featurecloud_test(
    app_image: str,
    data_dir: Path,
    client_dirs: Sequence[str],
    controller_host: str,
    query_interval: int,
    timeout: int,
) -> tuple[int, List[Path]]:
    exp_id = fc_test.start(
        controller_host=controller_host,
        client_dirs=",".join(client_dirs),
        generic_dir="",
        app_image=app_image,
        channel="local",
        query_interval=query_interval,
        download_results="tests",
    )
    exp_id = int(exp_id)
    start_time = time.time()

    while True:
        info = fc_test.info(controller_host=controller_host, test_id=exp_id)
        status = info.iloc[0]["status"]
        if status == "finished":
            instances = info.iloc[0]["instances"]
            break
        if status in ("error", "stopped"):
            raise RuntimeError(f"FeatureCloud test {exp_id} ended with status: {status}")
        if time.time() - start_time > timeout:
            fc_test.stop(controller_host=controller_host, test_id=exp_id)
            raise RuntimeError(f"FeatureCloud test {exp_id} timed out after {timeout}s")
        time.sleep(query_interval)

    instances = sorted(instances, key=lambda item: item.get("id", 0))
    result_files = []
    for inst in instances:
        filename = f"results_test_{exp_id}_client_{inst['id']}_{inst['name']}.zip"
        result_files.append(data_dir / "tests" / "tests" / filename)
    return exp_id, result_files


def wait_for_paths(paths: Iterable[Path], timeout: int = 120) -> None:
    deadline = time.time() + timeout
    missing = [p for p in paths if not p.exists()]
    while missing and time.time() < deadline:
        time.sleep(2)
        missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Result files not found: {missing}")


def extract_clustering(
    zip_path: Path,
    output_dir: Path,
    client_idx: int,
    k_values: Sequence[int],
) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = set(zf.namelist())
        for k in k_values:
            member = f"kmeans/K_{k}/clustering.csv"
            if member not in members:
                raise FileNotFoundError(f"Missing {member} in {zip_path}")
            zf.extract(member, path=output_dir)
            extracted = output_dir / member
            target = output_dir / f"{client_idx}_{k}_clustering.csv"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(extracted), str(target))

    kmeans_dir = output_dir / "kmeans"
    if kmeans_dir.exists():
        shutil.rmtree(kmeans_dir)


def ensure_configs(data_dir: Path, client_dirs: Sequence[str]) -> None:
    for lab in client_dirs:
        config_path = data_dir / lab / "config_kmeans.yml"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing config_kmeans.yml in {config_path}")


def run_repetitions(
    data_root: Path,
    modes: Sequence[str],
    input_variant: str,
    start: int,
    end: int,
    app_image: str,
    client_dirs: Sequence[str],
    k_values: Sequence[int],
    controller_host: str,
    query_interval: int,
    timeout: int,
    prepare: bool,
    restart_controller: bool,
    skip_existing: bool,
) -> None:
    current_data_dir: Path | None = None

    for mode in modes:
        mode_root = data_root / mode
        if not mode_root.exists():
            raise FileNotFoundError(f"Mode not found: {mode_root}")

        if input_variant == "before":
            output_base = mode_root / "after" / "fc_kmeans_res"
        else:
            output_base = mode_root / "before_corrected" / "fc_kmeans_res"
        output_base.mkdir(parents=True, exist_ok=True)

        for run_id in range(start, end + 1):
            run_output = output_base / f"{run_id}_fed_clustering"
            if skip_existing and run_output.exists():
                existing = list(run_output.glob("*_clustering.csv"))
                if existing:
                    print(f"[{mode} run {run_id}] Skipping existing output: {run_output}")
                    continue

            if prepare:
                prepare_inputs(mode_root, run_id, input_variant, client_dirs)

            data_dir = (mode_root / input_variant).resolve()
            ensure_configs(data_dir, client_dirs)

            if restart_controller or data_dir != current_data_dir:
                print(f"Starting controller for {data_dir}")
                start_controller(data_dir)
                current_data_dir = data_dir

            print(f"[{mode} run {run_id}] Starting FeatureCloud test")
            exp_id, zip_paths = run_featurecloud_test(
                app_image=app_image,
                data_dir=data_dir,
                client_dirs=client_dirs,
                controller_host=controller_host,
                query_interval=query_interval,
                timeout=timeout,
            )
            wait_for_paths(zip_paths)

            run_output.mkdir(parents=True, exist_ok=True)
            for idx, zip_path in enumerate(zip_paths, start=1):
                shutil.copy2(zip_path, run_output / zip_path.name)
                extract_clustering(zip_path, run_output, idx, k_values)

            manifest = {
                "mode": mode,
                "run_id": run_id,
                "input_variant": input_variant,
                "app_image": app_image,
                "test_id": exp_id,
                "client_dirs": list(client_dirs),
                "k_values": list(k_values),
                "data_dir": str(data_dir),
                "zip_files": [p.name for p in zip_paths],
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            (run_output / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True)
            )
            print(f"[{mode} run {run_id}] Done: {run_output}")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    default_data_root = repo_root / "evaluation_data" / "simulated_rotation"

    parser = argparse.ArgumentParser(
        description="Run federated k-means clustering across repetitions."
    )
    parser.add_argument(
        "--modes",
        default=",".join(DEFAULT_MODES),
        help="Comma-separated modes (default: balanced,mild_imbalanced,strong_imbalanced)",
    )
    parser.add_argument(
        "--input-variant",
        choices=("before", "before_corrected"),
        default="before",
        help="Input folder to use for FeatureCloud tests",
    )
    parser.add_argument("--start", type=int, default=1, help="First run id (inclusive)")
    parser.add_argument("--end", type=int, default=1, help="Last run id (inclusive)")
    parser.add_argument(
        "--app-image",
        default="fc_kmeans_upd",
        help="FeatureCloud app image name (default: fc_kmeans_upd)",
    )
    parser.add_argument(
        "--client-dirs",
        default=",".join(DEFAULT_CLIENT_DIRS),
        help="Comma-separated client dirs (default: lab1,lab2,lab3)",
    )
    parser.add_argument(
        "--k-values",
        default=",".join(str(k) for k in DEFAULT_K_VALUES),
        help="Comma-separated K values to extract (default: 2,3)",
    )
    parser.add_argument(
        "--data-root",
        default=str(default_data_root),
        help="Path to evaluation_data/simulated_rotation",
    )
    parser.add_argument(
        "--controller-host",
        default=DEFAULT_CONTROLLER_HOST,
        help="Controller host URL",
    )
    parser.add_argument("--query-interval", type=int, default=5, help="Poll interval")
    parser.add_argument("--timeout", type=int, default=1800, help="Test timeout (s)")
    parser.add_argument(
        "--prepare-inputs",
        action="store_true",
        help="Regenerate inputs from intermediate or FedSim outputs per run",
    )
    parser.add_argument(
        "--restart-controller",
        action="store_true",
        help="Restart controller before every run (default: only when data dir changes)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs with existing clustering outputs",
    )

    args = parser.parse_args()

    run_repetitions(
        data_root=Path(args.data_root),
        modes=parse_csv_list(args.modes),
        input_variant=args.input_variant,
        start=args.start,
        end=args.end,
        app_image=args.app_image,
        client_dirs=parse_csv_list(args.client_dirs),
        k_values=parse_k_values(args.k_values),
        controller_host=args.controller_host,
        query_interval=args.query_interval,
        timeout=args.timeout,
        prepare=args.prepare_inputs,
        restart_controller=args.restart_controller,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
