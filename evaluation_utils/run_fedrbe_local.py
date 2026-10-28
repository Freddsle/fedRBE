#!/usr/bin/env python3
"""
Runs the fedRBE experiments of generate_fedrbe_corrected_datasets.py locally
without docker or the FeatureCloud controller, to quickly check that a change
did not break anything.
Each client runs the app's actual protocol (batchcorrection/fedrbe.py run_fedrbe)
in its own thread, data is exchanged via run_simulation_native of the
fc_fedlearnsim_helper. SMPC is not simulated, experiments using SMPC are skipped.
For every experiment, the merged federated result is compared to the central
limma result used by analyse_fedvscentral.py. With --write, the results are
additionally saved into evaluation_data exactly like generate_fedrbe_corrected_datasets.py
does.
Usage (from the repository root):
    python3 evaluation_utils/run_fedrbe_local.py [--only <name part>] [--write] [--verbose]
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

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "evaluation_classification_after_correction"))

from batchcorrection.fedrbe import run_fedrbe  # noqa: E402
from fc_fed_forest_simple_app.fc_fedlearnsim_helper.run_app_simulation import run_simulation_native  # noqa: E402
from evaluation_utils import utils_analyse  # noqa: E402
import generate_fedrbe_corrected_datasets as generate  # noqa: E402
import analyse_fedvscentral  # noqa: E402

OUTPUT_FILES = ["only_batch_corrected_data.csv", "full_corrected_data.csv", "report.txt"]


def central_result_file(result_filename: str) -> str:
    """Finds the central result file belonging to a federated result file."""
    for experiment in analyse_fedvscentral.experiment_results:
        if os.path.abspath(experiment.federated_result_file) == os.path.abspath(result_filename):
            return experiment.central_result_file
    raise ValueError(f"No central result registered in analyse_fedvscentral.py for {result_filename}")


def run_experiment(exp, result_filename: str, write: bool, verbose: bool) -> dict:
    """
    Runs one experiment locally and compares it to the central result.
    Returns a summary row.
    """
    exp = deepcopy(exp)
    row = {"Experiment": exp.name, "Result": os.path.relpath(result_filename, generate.data_dir)}
    config_paths = [Path(client) / "config.yml" for client in exp.clients]
    original_configs = {path: path.read_bytes() if path.exists() else None for path in config_paths}
    log = io.StringIO()
    try:
        # merges the config changes and writes config.yml into the client folders
        with contextlib.redirect_stdout(log):
            exp._set_config_files()
        if any(config["flimmaBatchCorrection"].get("smpc") for config in exp.config_files):
            row["Status"] = "skipped (SMPC)"
            return row
        client_names = [os.path.basename(client) for client in exp.clients]
        with tempfile.TemporaryDirectory() as tmp:
            output_dirs = [os.path.join(tmp, name) for name in client_names]
            for output_dir in output_dirs:
                os.makedirs(output_dir)
            # the first client is the coordinator, as in the docker runs
            with contextlib.redirect_stdout(sys.stdout if verbose else log):
                run_simulation_native(exp.clients, output_dirs, None, run_fedrbe,
                                      client_ids=client_names)
            client_files = [{fname: Path(output_dir, fname).read_bytes()
                             for fname in OUTPUT_FILES if Path(output_dir, fname).exists()}
                            for output_dir in output_dirs]
            if write:
                with contextlib.redirect_stdout(sys.stdout if verbose else log):
                    generate.save_experiment_results(exp, result_filename, client_files)

            # merge the clients' corrected data and compare it to the central result
            frames = []
            for config, files in zip(exp.config_files, client_files):
                separator = config["flimmaBatchCorrection"].get("separator", "\t")
                frames.append(pd.read_csv(io.BytesIO(files["only_batch_corrected_data.csv"]),
                                          sep=separator, index_col=0))
            federated_file = os.path.join(tmp, "FedApp_corrected_data.tsv")
            pd.concat(frames, axis=1).to_csv(federated_file, sep="\t")
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
        row["Status"] = "FAILED (difference)" if failed else ("ok, written" if write else "ok")
    except Exception as error:  # pylint: disable=broad-except
        row["Status"] = f"FAILED: {error}"
        if verbose:
            traceback.print_exc()
        else:
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


def main():
    parser = argparse.ArgumentParser(
        description="Run the fedRBE experiments locally (no docker, no SMPC) and "
                    "compare them to the central results.")
    parser.add_argument("--only", default=None,
                        help="Only run experiments whose name or result path contains this text")
    parser.add_argument("--write", action="store_true",
                        help="Also save the results into evaluation_data like generate_fedrbe_corrected_datasets.py")
    parser.add_argument("--verbose", action="store_true",
                        help="Show the output of the clients")
    args = parser.parse_args()

    # result paths are given relative to evaluation_data, as in the generate script
    selected = [(exp, os.path.join(generate.data_dir, result_filename))
                for exp, result_filename in generate.experiments
                if args.only is None or args.only in exp.name or args.only in result_filename]
    if not selected:
        raise SystemExit(f"No experiment matches {args.only!r}")

    rows = []
    for exp, result_filename in selected:
        print(f"Running {exp.name} ({os.path.relpath(result_filename, generate.data_dir)})")
        rows.append(run_experiment(exp, result_filename, args.write, args.verbose))
        print(f"  -> {rows[-1]['Status']}")

    summary = pd.DataFrame(rows)
    with pd.option_context("display.width", 200, "display.max_colwidth", 80):
        print(summary.to_string(index=False))
    if summary["Status"].str.startswith("FAILED").any():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
