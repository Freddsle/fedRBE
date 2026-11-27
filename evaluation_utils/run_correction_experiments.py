"""Run central limma correction with or without biological covariates.

Prepared real-data inputs are corrected into canonical outputs or an isolated
experiment directory. Input preparation and downstream evaluation have separate
entrypoints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import pandas as pd
import yaml

from evaluation_utils.real_datasets_utils import (
    dataset_configs, read_correction_matrix as read_matrix,
    validate_corrected_matrix as validate_corrected, validate_experiment_dir,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("ecoli", "ovarian_cancer", "ccRCC_proteomics", "quartet_multiomics")
ARMS = ("label_aware", "no_label")
MODALITIES = ("Transcriptomics", "Proteomics", "Metabolomics")


def _source_units(names: list[str]) -> list[dict]:
    configs = dataset_configs(REPO_ROOT)
    units = []
    for name in names:
        cfg = configs[name]
        for modality in MODALITIES if name == "quartet_multiomics" else ("main",):
            source_root = (REPO_ROOT / "evaluation_data/quartet_multiomics/before" / modality
                           if name == "quartet_multiomics" else cfg.sites_root)
            central_input = source_root / "central_input.tsv"
            if not central_input.exists() and central_input.with_suffix(".zip").exists():
                central_input = central_input.with_suffix(".zip")
            after_root = (REPO_ROOT / "evaluation_data/quartet_multiomics/after"
                          if name == "quartet_multiomics" else cfg.corrected_central.parent.parent)
            output = (after_root / "label_aware" / modality /
                      "intensities_log_Rcorrected_UNION.tsv"
                      if name == "quartet_multiomics" else cfg.corrected_central)
            units.append(dict(dataset=name, modality=modality, source_root=source_root,
                              central_input=central_input, after_root=after_root, central=output))
    return units


def _unit_sources(unit: dict) -> tuple[dict, dict, dict]:
    """Read the cohort configuration and sample-indexed central design metadata."""
    source_root = unit["source_root"]
    descriptor = json.loads((source_root / "datainfo.json").read_text())
    configs, designs = {}, {}
    for cohort in descriptor["cohorts"]:
        name = cohort["name"]
        client = source_root / cohort["folder"]
        config = yaml.safe_load((client / "config.yml").read_text())
        details = config["flimmaBatchCorrection"]
        design = pd.read_csv(client / details["design_filename"],
                             sep=details.get("design_separator", "\t"), index_col=0)
        design.index = design.index.astype(str)
        configs[name], designs[name] = config, design
    sample_ids = [sample for design in designs.values() for sample in design.index]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"Duplicate correction sample IDs for {unit['dataset']}")
    return descriptor, configs, designs


def _central_designs(directory: Path, matrix: pd.DataFrame, configs: dict,
                     designs: dict, clients: list[str], arms=ARMS) -> None:
    """Write aligned batch and selected biological/intercept-only design tables."""
    directory.mkdir(parents=True, exist_ok=True)
    details = configs[clients[0]]["flimmaBatchCorrection"]
    batch_col = details.get("batch_col")
    if any(configs[name]["flimmaBatchCorrection"].get("batch_col") != batch_col
           for name in clients):
        raise ValueError("Central batch columns differ across cohorts")
    combined = pd.concat([designs[name] for name in clients])
    if not combined.index.is_unique:
        raise ValueError("Duplicate correction sample IDs in the central design")
    missing = matrix.columns.difference(combined.index)
    if len(missing):
        raise ValueError(f"Central design is missing samples: {missing.tolist()}")
    batches = (combined[batch_col].astype(str) if batch_col else
               pd.concat([pd.Series(name, index=designs[name].index) for name in clients]))
    batches = batches.loc[matrix.columns]
    if batch_col:
        # Preserve the source notebooks' Quartet batch-code ordering.
        order = combined[["batch_code", batch_col]].drop_duplicates().sort_values("batch_code")
        batch_order = order[batch_col].astype(str).tolist()
    else:
        batch_order = sorted(clients)
    if len(batch_order) != len(set(batch_order)) or set(batch_order) != set(batches):
        raise ValueError("Explicit batch levels do not match the correction inputs")
    ranks = {name: rank for rank, name in enumerate(batch_order, start=1)}
    pd.DataFrame({"file": matrix.columns, "batch": batches.values,
                  "batch_order": batches.map(ranks).values}).to_csv(
        directory / "batch.tsv", sep="\t", index=False)
    for arm in arms:
        design = pd.DataFrame({"intercept": 1}, index=matrix.columns)
        if arm == "label_aware":
            covariates = details.get("covariates") or []
            if any((configs[name]["flimmaBatchCorrection"].get("covariates") or []) != covariates
                   for name in clients):
                raise ValueError("Central correction covariates differ across cohorts")
            if not set(covariates).issubset(combined.columns):
                raise ValueError("Central design is missing biological covariates")
            values = combined.loc[matrix.columns, covariates]
            if values.isna().any().any():
                raise ValueError("Central biological covariates contain missing values")
            design[covariates] = values
        design.rename_axis("file").to_csv(directory / f"{arm}_design.tsv", sep="\t")


def _run_limma(input_file: Path, prepared: Path, arm: str, output: Path,
               rscript: str, expected: pd.DataFrame) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = output.with_suffix(".pending.tsv")
    try:
        completed = subprocess.run([
            rscript, str(Path(__file__).with_name("central_correction.R")),
            str(input_file), str(prepared / f"{arm}_design.tsv"),
            str(prepared / "batch.tsv"), str(pending),
        ], capture_output=True, text=True)
        if completed.returncode:
            raise RuntimeError(f"Central correction failed for {output}:\n{completed.stderr}")
        validate_corrected(read_matrix(pending), expected)
        pending.replace(output)
    finally:
        pending.unlink(missing_ok=True)


def correct_canonical(datasets: list[str], *, rscript: str,
                      arms: list[str] | None = None, output_dir: Path | None = None) -> None:
    """Correct each central input once per selected design, preserving its source."""
    arms = list(dict.fromkeys(arms if arms is not None else
                             ARMS if output_dir is not None else ["label_aware"]))
    if not arms or not set(arms).issubset(ARMS):
        raise ValueError("Select label_aware and/or no_label correction")
    root = validate_experiment_dir(REPO_ROOT, output_dir) if output_dir is not None else None
    sources = _source_units(datasets)
    jobs = []
    for source in sources:
        outputs = {}
        for arm in arms:
            output = (root / "correction" / source["dataset"] / source["modality"] /
                      "central" / arm / "matrix.tsv" if root is not None else
                      source["after_root"] / arm /
                      source["central"].relative_to(source["after_root"] / "label_aware"))
            if root is not None and output.exists():
                raise ValueError("Correction outputs already exist; use a fresh run directory")
            outputs[arm] = output
        jobs.append((source, outputs))
    records = []
    for source, outputs in jobs:
        _, configs, designs = _unit_sources(source)
        matrix = read_matrix(source["central_input"])
        with tempfile.TemporaryDirectory(prefix="central-correction-") as directory:
            prepared = Path(directory)
            _central_designs(prepared, matrix, configs, designs, list(configs), arms)
            input_file = prepared / "data.tsv"
            header = pd.read_csv(source["central_input"], sep="\t", nrows=0).columns[0]
            matrix.to_csv(input_file, sep="\t", index_label=header)
            for arm, output in outputs.items():
                _run_limma(input_file, prepared, arm, output, rscript, matrix)
        records.append({"dataset": source["dataset"], "modality": source["modality"],
                        "input": str(source["central_input"]),
                        "outputs": {arm: str(path) for arm, path in outputs.items()}})
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
        (root / "central.json").write_text(json.dumps(
            {"rscript": rscript, "arms": arms, "inputs": records}, indent=2) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("correct",))
    parser.add_argument("--output-dir", type=Path, help="Isolated central experiment outputs")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS)
    parser.add_argument("--arms", nargs="+", choices=ARMS,
                        help="Default: label_aware for canonical outputs, both arms if isolated")
    parser.add_argument("--rscript", default="Rscript")
    args = parser.parse_args(argv)
    correct_canonical(args.datasets or list(DATASETS), rscript=args.rscript,
                      arms=args.arms, output_dir=args.output_dir)
    print("Central correction complete")


if __name__ == "__main__":
    main()
