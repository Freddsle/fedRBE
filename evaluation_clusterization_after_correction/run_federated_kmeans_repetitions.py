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
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from evaluation_utils import featurecloud_api_extension as fc_utils
except ImportError as exc:  # pragma: no cover - only needed when running the script
    raise SystemExit(
        "FeatureCloud SDK or its dependencies are not available. Install FeatureCloud to run this script."
    ) from exc

DEFAULT_CONTROLLER_HOST = fc_utils.DEFAULT_CONTROLLER_HOST
DEFAULT_MODES = ("balanced", "mild_imbalanced", "strong_imbalanced")
DEFAULT_CLIENT_DIRS = ("lab1", "lab2", "lab3")
DEFAULT_K_VALUES = (2, 3)
KMEANS_CONFIG_PATTERNS = ("config_kmeans*.yml", "config_kmeans*.yaml")


class KMeansExperiment(fc_utils.Experiment):
    def _set_config_files(self) -> None:
        # KMeans uses config_kmeans*.yml/.yaml; avoid overwriting config.yml.
        return


def parse_csv_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_k_values(value: str) -> List[int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    return [int(item) for item in items]


def find_kmeans_config(directory: Path) -> Path:
    candidates = []
    for pattern in KMEANS_CONFIG_PATTERNS:
        candidates.extend(directory.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"Missing config_kmeans*.yml/.yaml in {directory}")

    candidates = sorted(candidates, key=lambda path: path.name)
    for name in ("config_kmeans.yml", "config_kmeans.yaml"):
        for candidate in candidates:
            if candidate.name == name:
                return candidate
    return candidates[0]


def read_metadata(path: Path) -> pd.DataFrame:
    with path.open(newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        rows = []
        for row in reader:
            if len(row) == len(header) + 1:
                row = row[1:]
            if len(row) != len(header):
                raise ValueError(
                    f"Unexpected column count in metadata {path}: {len(row)}"
                )
            rows.append(dict(zip(header, row)))

    df = pd.DataFrame(rows)
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
    corrected_suffixes: Sequence[str],
) -> Path:
    metadata = read_metadata(mode_root / "all_metadata.tsv")
    condition_levels = sorted(metadata["condition"].dropna().unique())

    if input_variant == "before_corrected":
        intensities_path = (
            mode_root
            / "before"
            / "intermediate"
            / f"{run_id}_intensities_data.tsv"
        )
        target_base = mode_root / "before_corrected"
    elif input_variant == "before":
        intensities_path = None
        for suffix in corrected_suffixes:
            candidate = mode_root / "after" / "runs" / f"{run_id}_{suffix}.tsv"
            if candidate.exists():
                intensities_path = candidate
                break
        if intensities_path is None:
            checked = ", ".join(
                f"{run_id}_{suffix}.tsv" for suffix in corrected_suffixes
            )
            raise FileNotFoundError(
                f"Missing corrected intensities for run {run_id} in {mode_root / 'after' / 'runs'} "
                f"(checked: {checked})"
            )
        target_base = mode_root / "before"
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

        # Ensure config is present (the app looks for config_kmeans*.yml/.yaml)
        try:
            find_kmeans_config(lab_dir)
        except FileNotFoundError:
            if input_variant == "before":
                raise
            config_src = find_kmeans_config(mode_root / "before" / lab)
            shutil.copy2(config_src, lab_dir / config_src.name)

        write_intensities(intensities[lab_meta["file"]], lab_dir / "intensities.tsv")
        write_design(lab_meta, lab_dir / "design.tsv", condition_levels)

    return target_base


def wait_for_paths(paths: Iterable[Path], timeout: int = 120) -> None:
    deadline = time.time() + timeout
    missing = [p for p in paths if not p.exists()]
    while missing and time.time() < deadline:
        time.sleep(2)
        missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Result files not found: {missing}")


def parse_test_id(paths: Iterable[Path]) -> int | None:
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
    keep_extracted: bool,
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
            if keep_extracted:
                shutil.copy2(extracted, target)
            else:
                shutil.move(str(extracted), str(target))

    if not keep_extracted:
        kmeans_dir = output_dir / "kmeans"
        if kmeans_dir.exists():
            shutil.rmtree(kmeans_dir)


def aggregate_fed_clusters(
    mode_root: Path,
    run_id: int,
    run_output: Path,
    client_dirs: Sequence[str],
    k_values: Sequence[int],
    input_variant: str,
    column_prefix: str,
) -> Path:
    metadata = read_metadata(mode_root / "all_metadata.tsv").set_index("file")

    aggregated = []
    for k in k_values:
        lab_frames = []
        for idx, _lab in enumerate(client_dirs, start=1):
            clustering_path = run_output / f"{idx}_{k}_clustering.csv"
            if not clustering_path.exists():
                raise FileNotFoundError(f"Missing clustering file: {clustering_path}")
            df = pd.read_csv(clustering_path, sep=";", index_col=0)
            if df.empty:
                raise ValueError(f"Empty clustering file: {clustering_path}")
            df.columns = [f"{column_prefix}_{k}clusters"]
            lab_frames.append(df)
        aggregated.append(pd.concat(lab_frames, axis=0))

    merged = metadata.join(pd.concat(aggregated, axis=1), how="left")
    variant_label = "before" if input_variant == "before_corrected" else "after"
    output_root = mode_root / "after" / "kmeans_res" / "runs"
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"{run_id}_metadata_{variant_label}_fedclusters.tsv"
    merged.reset_index().to_csv(output_path, sep="\t", index=False)
    return output_path


def ensure_configs(data_dir: Path, client_dirs: Sequence[str]) -> None:
    for lab in client_dirs:
        lab_dir = data_dir / lab
        try:
            find_kmeans_config(lab_dir)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Missing config_kmeans*.yml/.yaml in {lab_dir}"
            ) from exc


def run_repetitions(
    data_root: Path,
    modes: Sequence[str],
    input_variant: str,
    start: int,
    end: int,
    app_image: str,
    client_dirs: Sequence[str],
    k_values: Sequence[int],
    corrected_suffixes: Sequence[str],
    controller_host: str,
    query_interval: int,
    timeout: int,
    prepare: bool,
    restart_controller: bool,
    skip_existing: bool,
    aggregate_output: bool,
    column_prefix: str,
    debug: bool,
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
                prepare_inputs(
                    mode_root,
                    run_id,
                    input_variant,
                    client_dirs,
                    corrected_suffixes,
                )

            data_dir = (mode_root / input_variant).resolve()
            ensure_configs(data_dir, client_dirs)

            experiment = KMeansExperiment(
                name=f"fed_kmeans_{mode}_run_{run_id}",
                clients=[str(data_dir / lab) for lab in client_dirs],
                app_image_name=app_image,
                fc_data_dir=str(data_dir),
                controller_host=controller_host,
                query_interval=query_interval,
                timeout=timeout,
            )

            if restart_controller or data_dir != current_data_dir:
                print(f"Starting controller for {data_dir}")
                experiment._startup()
                current_data_dir = data_dir

            print(f"[{mode} run {run_id}] Starting FeatureCloud test")
            zip_files, _, exp_meta = experiment.run_test()
            zip_paths = [Path(path) for path in zip_files]
            wait_for_paths(zip_paths, timeout=timeout)
            test_id = parse_test_id(zip_paths)

            run_output.mkdir(parents=True, exist_ok=True)
            for idx, zip_path in enumerate(zip_paths, start=1):
                copied_zip = run_output / zip_path.name
                shutil.copy2(zip_path, copied_zip)
                extract_clustering(
                    copied_zip,
                    run_output,
                    idx,
                    k_values,
                    keep_extracted=debug,
                )
                if copied_zip.exists():
                    copied_zip.unlink()

            aggregated_path = None
            if aggregate_output:
                aggregated_path = aggregate_fed_clusters(
                    mode_root=mode_root,
                    run_id=run_id,
                    run_output=run_output,
                    client_dirs=client_dirs,
                    k_values=k_values,
                    input_variant=input_variant,
                    column_prefix=column_prefix,
                )

            manifest = {
                "mode": mode,
                "run_id": run_id,
                "input_variant": input_variant,
                "app_image": app_image,
                "test_id": test_id,
                "client_dirs": list(client_dirs),
                "k_values": list(k_values),
                "data_dir": str(data_dir),
                "zip_files": [p.name for p in zip_paths],
                "aggregated_result": str(aggregated_path) if aggregated_path else None,
                "experiment_meta": {
                    "experiment_name": exp_meta.experiment_name,
                    "input_hashes": exp_meta.input_hashes,
                    "config": exp_meta.config,
                },
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

    # run_federated_kmeans_repetitions.py --modes balanced --start 1 --end 1 --prepare-inputs

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
        "--corrected-suffixes",
        default="R_corrected",
        help="Comma-separated suffixes for corrected inputs (default: R_corrected)",
    )
    parser.add_argument(
        "--data-root",
        default=str(default_data_root),
        help="Path to evaluation_data/simulated_rotation",
    )
    parser.add_argument(
        "--controller-host",
        default=DEFAULT_CONTROLLER_HOST,
        help="Controller host URL. Default: " + DEFAULT_CONTROLLER_HOST,
    )
    parser.add_argument("--query-interval", type=int, default=5, help="Poll interval. Default: 5s")
    parser.add_argument("--timeout", type=int, default=1800, help="Test timeout (s). Default: 1800s")
    parser.add_argument(
        "--prepare-inputs",
        action="store_true",
        help="Regenerate inputs from intermediate or corrected outputs per run. Default: False",
    )
    parser.add_argument(
        "--restart-controller",
        action="store_true",
        help="Restart controller before every run (default: only when data dir changes)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs with existing clustering outputs. Default: False",
    )
    parser.add_argument(
        "--aggregate-output",
        action="store_true",
        help="Write per-run aggregated metadata with Fed clusters. Default: False",
    )
    parser.add_argument(
        "--cluster-prefix",
        default="Fed",
        help="Prefix for aggregated cluster columns (default: Fed)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Keep extracted kmeans folders in the run output for inspection.",
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
        corrected_suffixes=parse_csv_list(args.corrected_suffixes),
        controller_host=args.controller_host,
        query_interval=args.query_interval,
        timeout=args.timeout,
        prepare=args.prepare_inputs,
        restart_controller=args.restart_controller,
        skip_existing=args.skip_existing,
        aggregate_output=args.aggregate_output,
        column_prefix=args.cluster_prefix,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
