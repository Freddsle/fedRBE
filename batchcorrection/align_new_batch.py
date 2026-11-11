"""
Corrects a new batch with the output of a finished fedRBE run, without another
federated run.

The coordinator of a fedRBE run writes training_level.csv, the level per feature
that the corrected training data sits at, and training_level_info.yml, the
settings needed to bring a new batch to the same scale. This script loads the
new batch like a fedRBE client and shifts each of its batches per feature so
that its mean equals the training level (see
classes.client.align_to_training_level). The aligned batch can then be used
with models trained on the corrected data of the run. No labels are needed.

Usage:
    python batchcorrection/align_new_batch.py --fedrbe-output <coordinator output> \\
        --input <folder with config.yml and data> --output <folder>
"""
import argparse
import os
import tempfile
from typing import Dict, Optional, Tuple

import pandas as pd
import yaml

try:
    # inside the app container, the app folder is the working directory
    from classes.client import Client, align_to_training_level
    from fedrbe import TRAINING_LEVEL_FILE, TRAINING_LEVEL_INFO_FILE
except ImportError:
    # imported from the repository, e.g. by tests or the evaluation
    from batchcorrection.classes.client import Client, align_to_training_level
    from batchcorrection.fedrbe import TRAINING_LEVEL_FILE, TRAINING_LEVEL_INFO_FILE

ALIGNED_DATA_FILE = "aligned_data.csv"
REPORT_FILE = "report.txt"
CONFIG_KEY = "flimmaBatchCorrection"


def read_training_level(fedrbe_output: str) -> Tuple[pd.Series, dict]:
    """The training level per hashed feature and the settings of a fedRBE run."""
    info_path = os.path.join(fedrbe_output, TRAINING_LEVEL_INFO_FILE)
    level_path = os.path.join(fedrbe_output, TRAINING_LEVEL_FILE)
    for path in (info_path, level_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{path} is missing; use the output folder of the "
                                    "coordinator of a fedRBE run")
    with open(info_path, encoding="utf-8") as info_file:
        info = yaml.safe_load(info_file)
    # the index holds hashed feature names, read as text
    level = pd.read_csv(level_path, sep=info["separator"], index_col=0, dtype=str)
    return level.iloc[:, 0].astype(float), info


def load_new_batch(input_folder: str, info: dict, name: Optional[str] = None) -> Client:
    """
    Loads the new batch as fedRBE loads a client, from input_folder/config.yml
    (or config.yaml). The covariates are ignored and the normalization of the
    fedRBE run is used.
    """
    for filename in ("config.yml", "config.yaml"):
        path = os.path.join(input_folder, filename)
        if os.path.isfile(path):
            break
    else:
        raise FileNotFoundError(f"No config.yml or config.yaml in {input_folder}")
    with open(path, encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict) or CONFIG_KEY not in config:
        raise ValueError(f"{path} needs the key {CONFIG_KEY}, as for fedRBE")
    settings = dict(config[CONFIG_KEY])
    recorded = info["normalizationMethod"]
    if "normalizationMethod" in settings and (settings["normalizationMethod"] or None) != recorded:
        raise ValueError(f"The new batch uses normalizationMethod "
                         f"{settings['normalizationMethod']!r}, but the fedRBE run used "
                         f"{recorded!r}; the training level is only valid on that scale")
    settings["normalizationMethod"] = recorded
    settings["covariates"] = []  # labels are not needed for the alignment
    client = Client()
    with tempfile.TemporaryDirectory() as tmp:
        # the data paths stay relative to input_folder, an absolute config
        # path is used as given
        adjusted = os.path.join(tmp, "config.yml")
        with open(adjusted, "w", encoding="utf-8") as config_file:
            yaml.safe_dump({CONFIG_KEY: settings}, config_file)
        client.config_based_init(
            clientname=name or os.path.basename(os.path.normpath(input_folder)),
            config_filename=adjusted, input_folder=input_folder)
    return client


def report_text(client: Client, info: dict, aligned_per_batch: Dict[str, int],
                num_level_features: int) -> str:
    num_features = len(client.feature_names)
    lines = [
        f"Aligned {client.client_name} to the training level of a fedRBE run "
        f"({info['n_batches']} batches).",
        f"normalizationMethod: {info['normalizationMethod']}",
        f"Features: {num_features} in the new batch, {num_level_features} with a "
        "training level.",
    ]
    for batch, count in aligned_per_batch.items():
        lines.append(f"Batch {batch}: {count} features aligned, "
                     f"{num_features - count} left as loaded (no training level or fewer "
                     f"than {max(client.min_samples, 1)} finite values).")
    if info["covariates_used"]:
        lines.append("The fedRBE run used covariates: the training level includes the "
                     "covariate effects at the training data's covariate mix. If the new "
                     "batch has a different mix (e.g. of labels), part of that difference "
                     "is removed as batch effect.")
    return "\n".join(lines) + "\n"


def align_new_batch(fedrbe_output: str, input_folder: str, output_folder: str,
                    name: Optional[str] = None) -> pd.DataFrame:
    """Aligns a new batch, writes the aligned data and a report, returns the aligned data."""
    level, info = read_training_level(fedrbe_output)
    client = load_new_batch(input_folder, info, name)
    aligned, aligned_per_batch = align_to_training_level(client, level)
    os.makedirs(output_folder, exist_ok=True)
    aligned.to_csv(os.path.join(output_folder, ALIGNED_DATA_FILE), sep=client.separator)
    with open(os.path.join(output_folder, REPORT_FILE), "w", encoding="utf-8") as report:
        report.write(report_text(client, info, aligned_per_batch, len(level)))
    return aligned


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fedrbe-output", required=True,
                        help=f"Output folder of the coordinator, with {TRAINING_LEVEL_FILE} "
                             f"and {TRAINING_LEVEL_INFO_FILE}")
    parser.add_argument("--input", required=True,
                        help="Folder of the new batch with config.yml and the data (and "
                             "design) file, laid out like a fedRBE client")
    parser.add_argument("--output", required=True,
                        help=f"Folder for {ALIGNED_DATA_FILE} and {REPORT_FILE}")
    parser.add_argument("--name", default=None,
                        help="Name of the new batch (default: the input folder's name)")
    args = parser.parse_args(argv)
    align_new_batch(args.fedrbe_output, args.input, args.output, args.name)


if __name__ == "__main__":
    main()
