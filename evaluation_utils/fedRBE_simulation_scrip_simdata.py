#!/usr/bin/env python3
"""
Runs fedRBE locally without docker or the FeatureCloud controller.
Each client runs the app's actual protocol (batchcorrection/fedrbe.py run_fedrbe)
in its own thread, data is exchanged via run_simulation_native of the
fc_fedlearnsim_helper. SMPC is not emulated, values are aggregated in plain.

Two modes:
1. Folder mode, used by evaluation_data/simulated/robustness_simulation.R and
   evaluation_data/simulated/01_data_prep_and_central_RBE.ipynb:
       python3 evaluation_utils/fedRBE_simulation_scrip_simdata.py <input_dir> <output_file>
   input_dir contains one subfolder per client (config.yml, data, design).
   The clients are the sorted subfolders, the first is the coordinator.
   The merged corrected data of all clients is saved as output_file.
2. Experiment mode, to quickly check that a change did not break anything:
       python3 evaluation_utils/fedRBE_simulation_scrip_simdata.py --experiments [--only <text>]
   Runs the experiments of generate_fedrbe_corrected_datasets.py (experiments
   using SMPC are skipped) and compares the merged federated result to the
   central limma result used by analyse_fedvscentral.py. The experiment's
   config.yml is written into the client folders for the run and restored
   afterwards; no results are written, published results come from the app.
--verbose shows the output of the clients in both modes.
"""
import argparse
import contextlib
from copy import deepcopy
import io
import os
from pathlib import Path
import sys
import tempfile
import traceback
from typing import List

import pandas as pd
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "evaluation_classification_after_correction"))

from batchcorrection.fedrbe import run_fedrbe  # noqa: E402
from fc_fed_forest_simple_app.fc_fedlearnsim_helper.run_app_simulation import run_simulation_native  # noqa: E402

OUTPUT_FILES = ["only_batch_corrected_data.csv", "full_corrected_data.csv", "report.txt"]


def run_clients(client_dirs: List[str], verbose: bool, log: io.StringIO) -> List[dict]:
    """
    Runs run_fedrbe for all clients, the first client is the coordinator.
    The client IDs are the folder names.
    Returns:
        Per client, a dict mapping the output file names of the app to their content as bytes
    """
    client_names = [os.path.basename(os.path.normpath(client)) for client in client_dirs]
    with tempfile.TemporaryDirectory() as tmp:
        output_dirs = [os.path.join(tmp, name) for name in client_names]
        for output_dir in output_dirs:
            os.makedirs(output_dir)
        with contextlib.redirect_stdout(sys.stdout if verbose else log):
            run_simulation_native(client_dirs, output_dirs, None, run_fedrbe,
                                  client_ids=client_names)
        return [{fname: Path(output_dir, fname).read_bytes()
                 for fname in OUTPUT_FILES if Path(output_dir, fname).exists()}
                for output_dir in output_dirs]


def merge_corrected_data(client_files: List[dict], separators: List[str]) -> pd.DataFrame:
    """Concatenates the clients' corrected data (features x samples) along the samples."""
    return pd.concat([pd.read_csv(io.BytesIO(files["only_batch_corrected_data.csv"]),
                                  sep=separator, index_col=0)
                      for files, separator in zip(client_files, separators)], axis=1)


### FOLDER MODE
def run_folder(input_dir: str, output_file: str, verbose: bool) -> None:
    """Runs fedRBE on the client subfolders of input_dir and saves the merged result."""
    client_dirs = sorted(os.path.join(input_dir, name) for name in os.listdir(input_dir)
                         if os.path.isfile(os.path.join(input_dir, name, "config.yml")))
    if len(client_dirs) < 3:
        raise ValueError(f"Expected at least three client folders with config.yml in {input_dir}")
    print("Clients: ", [os.path.basename(client) for client in client_dirs])
    separators = []
    for client in client_dirs:
        with open(os.path.join(client, "config.yml"), "r") as f:
            config = yaml.safe_load(f)["flimmaBatchCorrection"]
        separators.append(config.get("separator", "\t"))
    log = io.StringIO()
    try:
        client_files = run_clients(client_dirs, verbose, log)
    except Exception:
        print("\n".join(log.getvalue().splitlines()[-20:]))
        raise
    # rows sorted by feature name, the app's own order follows the hashed names
    federated_df = merge_corrected_data(client_files, separators).sort_index()
    federated_df.to_csv(output_file, sep="\t")
    print(f"Shape of federated data: {federated_df.shape}")
    print(f"Results saved to: {output_file}")


### EXPERIMENT MODE
def central_result_file(result_filename: str) -> str:
    """Finds the central result file belonging to a federated result file."""
    import analyse_fedvscentral  # pylint: disable=import-outside-toplevel
    for experiment in analyse_fedvscentral.experiment_results:
        if os.path.abspath(experiment.federated_result_file) == os.path.abspath(result_filename):
            return experiment.central_result_file
    raise ValueError(f"No central result registered in analyse_fedvscentral.py for {result_filename}")


def run_experiment(exp, result_filename: str, verbose: bool) -> dict:
    """
    Runs one experiment of generate_fedrbe_corrected_datasets.py locally and
    compares it to the central result. Returns a summary row.
    """
    # pylint: disable=import-outside-toplevel
    from evaluation_utils import utils_analyse
    import generate_fedrbe_corrected_datasets as generate

    exp = deepcopy(exp)
    row = {"Experiment": exp.name, "Result": os.path.relpath(result_filename, generate.data_dir)}
    config_paths = [Path(client) / "config.yml" for client in exp.clients]
    original_configs = {path: path.read_bytes() if path.exists() else None for path in config_paths}
    log = io.StringIO()
    try:
        # merges the config changes and writes config.yml into the client folders
        with contextlib.redirect_stdout(log):
            exp._set_config_files()  # pylint: disable=protected-access
        if any(config["flimmaBatchCorrection"].get("smpc") for config in exp.config_files):
            row["Status"] = "skipped (SMPC)"
            return row
        # the first client is the coordinator, as in the docker runs
        client_files = run_clients(exp.clients, verbose, log)

        # compare the merged corrected data to the central result
        separators = [config["flimmaBatchCorrection"].get("separator", "\t")
                      for config in exp.config_files]
        with tempfile.TemporaryDirectory() as tmp:
            federated_file = os.path.join(tmp, "FedApp_corrected_data.tsv")
            merge_corrected_data(client_files, separators).to_csv(federated_file, sep="\t")
            comparison = utils_analyse.ExperimentResult(
                name=exp.name,
                federated_result_file=federated_file,
                central_result_file=central_result_file(result_filename),
            )
            with contextlib.redirect_stdout(sys.stdout if verbose else log):
                result_df = utils_analyse.compare_experiments([comparison])
        if result_df is None or result_df.empty:
            row["Status"] = "FAILED (no comparison)"
            return row
        result = result_df.iloc[0]
        row["Mean difference"] = result["Mean difference"]
        row["Max difference"] = result["Max difference"]
        failed = result["Max difference"] > utils_analyse.FAILURE_THRESHOLD
        row["Status"] = "FAILED (difference)" if failed else "ok"
    except Exception as error:  # pylint: disable=broad-except
        row["Status"] = f"FAILED: {error}"
        if not verbose:
            print(f"--- log of {exp.name} ({row['Result']}) ---")
            print("\n".join(log.getvalue().splitlines()[-20:]))
        traceback.print_exc()
    finally:
        for path, content in original_configs.items():
            if content is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(content)
    return row


def run_experiments(only: str, verbose: bool) -> None:
    """Runs the selected experiments and prints a summary, exits with 1 on failures."""
    import generate_fedrbe_corrected_datasets as generate  # pylint: disable=import-outside-toplevel

    # result paths are given relative to evaluation_data, as in the generate script
    selected = [(exp, os.path.join(generate.data_dir, result_filename))
                for exp, result_filename in generate.experiments
                if only is None or only in exp.name or only in result_filename]
    if not selected:
        raise SystemExit(f"No experiment matches {only!r}")

    rows = []
    for exp, result_filename in selected:
        print(f"Running {exp.name} ({os.path.relpath(result_filename, generate.data_dir)})")
        rows.append(run_experiment(exp, result_filename, verbose))
        print(f"  -> {rows[-1]['Status']}")

    summary = pd.DataFrame(rows)
    with pd.option_context("display.width", 200, "display.max_colwidth", 80):
        print(summary.to_string(index=False))
    if summary["Status"].str.startswith("FAILED").any():
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Run fedRBE locally (no docker, no SMPC), either on the client folders "
                    "of input_dir or on the experiments of generate_fedrbe_corrected_datasets.py.")
    parser.add_argument("input_dir", nargs="?",
                        help="Folder with one subfolder (config.yml, data, design) per client")
    parser.add_argument("output_file", nargs="?", help="TSV file for the merged corrected data")
    parser.add_argument("--experiments", action="store_true",
                        help="Run the experiments of generate_fedrbe_corrected_datasets.py and "
                             "compare them to the central results")
    parser.add_argument("--only", default=None,
                        help="With --experiments: only run experiments whose name or result path "
                             "contains this text")
    parser.add_argument("--verbose", action="store_true", help="Show the output of the clients")
    args = parser.parse_args()

    if args.experiments:
        if args.input_dir or args.output_file:
            parser.error("--experiments does not take input_dir or output_file")
        run_experiments(args.only, args.verbose)
    else:
        if not (args.input_dir and args.output_file):
            parser.error("input_dir and output_file are required unless --experiments is given")
        if args.only:
            parser.error("--only requires --experiments")
        run_folder(args.input_dir, args.output_file, args.verbose)


if __name__ == "__main__":
    main()
