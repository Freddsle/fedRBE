#!/usr/bin/env python3
"""Rebuild the simulation Results figures, plotted statistics and reported tables."""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
THREE = ("confounding_correction.py", "build_sr2_boxplots.R")
FIVE = ("build_sr3_retention.py", "build_sr4_boxplots.R", "build_sr5_dcombat_boxplots.R")
SCRIPTS = {"all": (*THREE, *FIVE, "table_values.py"), "three-client": THREE,
           "five-client": FIVE, "sr1": THREE[:1], "sr3": FIVE[:1]}
# Figures the supplement does not show.
CANDIDATES = {"dcombat_learning_training_only.png", "dcombat_learning_70_30_training_only.png"}


def main(output=None, block="all"):
    if output is not None:
        output = output.resolve()
    if output is not None and output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    sibling = Path(sys.executable).with_name("Rscript")
    rscript = sibling if sibling.is_file() else shutil.which("Rscript")
    if not rscript and any(script.endswith(".R") for script in SCRIPTS[block]):
        raise FileNotFoundError("Rscript is required. Run in the repository's pixi environment.")
    parent = output.parent if output is not None else ROOT
    with tempfile.TemporaryDirectory(dir=parent, prefix=".simulation-report-") as tmp:
        artifacts = Path(tmp) / "render"
        artifacts.mkdir()
        for script in SCRIPTS[block]:
            binary = rscript if script.endswith(".R") else sys.executable
            subprocess.run([binary, HERE / script, artifacts], cwd=tmp, check=True)

        target = Path(tmp) / "release" if output is not None else ROOT
        destinations = {".png": target / "figures/suppl", "candidate": target / "figures/candidate",
                        ".tsv": target / "evaluation_data/simulated/results/reported"}
        files = [(path, destinations["candidate" if path.name in CANDIDATES else path.suffix]
                  / path.name) for path in artifacts.iterdir()]
        for directory in destinations.values():
            directory.mkdir(parents=True, exist_ok=True)
        for source, destination in files:
            source.replace(destination)
        if output is not None:
            if output.exists():
                raise FileExistsError(output)
            target.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path,
                        help="Export the same directory layout under a new root (default: repository)")
    parser.add_argument("--block", choices=SCRIPTS, default="all")
    args = parser.parse_args()
    main(args.output, args.block)
