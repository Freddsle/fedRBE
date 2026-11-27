"""Output lists and result checks for reproduce.sh.

The paths come from the experiment definitions of the scripts that write them,
so reproduce.sh checks the same files the pipeline uses:

    python -m evaluation_utils.reproduce_checks central-outputs [--simulated | --real]
    python -m evaluation_utils.reproduce_checks federated-outputs [--compared]
    python -m evaluation_utils.reproduce_checks check-fedvscentral

Paths are printed one per line, relative to the repository root. federated-outputs
lists every result generate_fedrbe_corrected_datasets.py writes; --compared
only those analyse_fedvscentral.py reads.
check-fedvscentral exits with 1 when evaluation_data/fed_vc_cent_results.tsv
misses an experiment or any maximum difference exceeds FAILURE_THRESHOLD.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def _relative(path: str | os.PathLike) -> str:
    return os.path.relpath(Path(path).resolve(), REPO_ROOT)


def _comparisons():
    import analyse_fedvscentral  # pylint: disable=import-outside-toplevel
    return analyse_fedvscentral.experiment_results


def central_outputs(subset: str | None) -> list[str]:
    paths = []
    for experiment in _comparisons():
        simulated = experiment.name.startswith("Simulated")
        if subset == "simulated" and not simulated or subset == "real" and simulated:
            continue
        path = _relative(experiment.central_result_file)
        if path not in paths:
            paths.append(path)
    return paths


def federated_outputs(compared: bool) -> list[str]:
    if compared:
        return [_relative(experiment.federated_result_file) for experiment in _comparisons()]
    import generate_fedrbe_corrected_datasets as generate  # pylint: disable=import-outside-toplevel
    return [_relative(os.path.join(generate.data_dir, result_filename))
            for _, result_filename in generate.experiments]


def check_fedvscentral() -> int:
    from evaluation_utils import utils_analyse  # pylint: disable=import-outside-toplevel
    result_file = REPO_ROOT / "evaluation_data" / "fed_vc_cent_results.tsv"
    results = pd.read_csv(result_file, sep="\t")
    expected = [experiment.name for experiment in _comparisons()]
    missing = sorted(set(expected) - set(results["Experiment"]))
    max_column = utils_analyse.RESULT_DF_COLUMNS[3]
    failed = results[[not (math.isfinite(value) and value <= utils_analyse.FAILURE_THRESHOLD)
                      for value in results[max_column]]]
    if missing:
        print(f"Experiments missing from {_relative(result_file)}:", *missing, sep="\n  ")
    if not failed.empty:
        print(f"Federated and central corrections differ by more than "
              f"{utils_analyse.FAILURE_THRESHOLD:g}:")
        print(failed.to_string(index=False))
    if missing or not failed.empty:
        return 1
    print(f"All {len(results)} comparisons within {utils_analyse.FAILURE_THRESHOLD:g} "
          f"(largest {max_column.lower()}: {results[max_column].max():.3g})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    central = commands.add_parser("central-outputs")
    subset = central.add_mutually_exclusive_group()
    subset.add_argument("--simulated", dest="subset", action="store_const", const="simulated")
    subset.add_argument("--real", dest="subset", action="store_const", const="real")
    federated = commands.add_parser("federated-outputs")
    federated.add_argument("--compared", action="store_true")
    commands.add_parser("check-fedvscentral")
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT))
    if args.command == "check-fedvscentral":
        return check_fedvscentral()
    if args.command == "central-outputs":
        paths = central_outputs(args.subset)
    else:
        paths = federated_outputs(args.compared)
    print("\n".join(paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())
