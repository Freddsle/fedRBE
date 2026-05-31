"""Utilities for Quartet multiomics FedRBE-style batch correction.

The FeatureCloud FedRBE app requires every client to have more samples than the
global design width. For the full Quartet design this is
1 intercept + 3 donor covariates + 14 non-reference batch columns = 18 columns.
Single 12-sample source batches therefore cannot be clients. The helpers below
create deterministic multi-batch clients while keeping the original source
batch as ``batch_col`` inside each client.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd


MODALITIES = ["Transcriptomics", "Proteomics", "Metabolomics"]
DONOR_LEVELS = ["D5", "D6", "F7", "M8"]
COVARIATES = ["D6", "F7", "M8"]


@dataclass(frozen=True)
class ClientGroup:
    name: str
    batch_codes: List[str]
    batches: List[str]
    reference_batch: str | bool
    position: int


def read_feature_matrix(path: Path) -> pd.DataFrame:
    """Read a feature-by-sample TSV matrix with the first column as row index."""
    df = pd.read_csv(path, sep="\t", index_col=0, low_memory=False)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df.apply(pd.to_numeric, errors="coerce")


def write_feature_matrix(df: pd.DataFrame, path: Path) -> None:
    """Write a feature-by-sample matrix using the repo's quoted TSV convention."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out.insert(0, "rowname", out.index)
    out.to_csv(path, sep="\t", index=False, quoting=csv.QUOTE_NONNUMERIC)


def write_table(df: pd.DataFrame, path: Path, quote: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    quoting = csv.QUOTE_NONNUMERIC if quote else csv.QUOTE_MINIMAL
    df.to_csv(path, sep="\t", index=False, quoting=quoting)


def load_prepared_modality(base_dir: Path, modality: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load prepared central matrix and metadata for one modality."""
    before_dir = base_dir / "before" / modality
    expr = read_feature_matrix(before_dir / "central_intensities_log_UNION.tsv")
    metadata = pd.read_csv(before_dir / "metadata.tsv", sep="\t")
    metadata["file"] = metadata["file"].astype(str)
    metadata["condition"] = pd.Categorical(
        metadata["condition"], categories=DONOR_LEVELS, ordered=True
    )
    metadata["batch_code"] = metadata["batch_code"].astype(str)
    metadata["batch"] = metadata["batch"].astype(str)
    metadata["rep"] = metadata["rep"].astype(int)

    missing = [sample for sample in metadata["file"] if sample not in expr.columns]
    if missing:
        raise ValueError(f"{modality}: samples missing from matrix: {missing[:5]}")

    metadata = metadata.sort_values(["batch_code", "condition", "rep"]).reset_index(drop=True)
    expr = expr.loc[:, metadata["file"]]
    return expr, metadata


def make_client_groups(metadata: pd.DataFrame) -> List[ClientGroup]:
    """Group sorted source batches into privacy-valid 2-batch clients.

    With 15 source batches, this yields six 2-batch clients and one 3-batch
    reference client. Each client has at least 24 samples.
    """
    batch_table = (
        metadata[["batch_code", "batch"]]
        .drop_duplicates()
        .sort_values("batch_code")
        .reset_index(drop=True)
    )
    rows = batch_table.to_dict("records")
    grouped_rows: List[List[dict]] = []
    i = 0
    while i < len(rows):
        remaining = len(rows) - i
        if remaining == 3:
            grouped_rows.append(rows[i : i + 3])
            i += 3
        elif remaining == 1 and grouped_rows:
            grouped_rows[-1].append(rows[i])
            i += 1
        else:
            grouped_rows.append(rows[i : i + 2])
            i += 2

    groups: List[ClientGroup] = []
    for idx, group_rows in enumerate(grouped_rows):
        batch_codes = [row["batch_code"] for row in group_rows]
        batches = [row["batch"] for row in group_rows]
        name = f"client_{idx + 1:02d}_{batch_codes[0]}_{batch_codes[-1]}"
        is_last = idx == len(grouped_rows) - 1
        reference_batch: str | bool = batches[-1] if is_last else False
        groups.append(
            ClientGroup(
                name=name,
                batch_codes=batch_codes,
                batches=batches,
                reference_batch=reference_batch,
                position=idx,
            )
        )
    return groups


def design_for_client(metadata: pd.DataFrame) -> pd.DataFrame:
    out = metadata.copy()
    out["D6"] = (out["condition"].astype(str) == "D6").astype(int)
    out["F7"] = (out["condition"].astype(str) == "F7").astype(int)
    out["M8"] = (out["condition"].astype(str) == "M8").astype(int)
    columns = [
        "file",
        "D6",
        "F7",
        "M8",
        "batch",
        "batch_code",
        "condition",
        "lab",
        "platform",
        "protocol",
        "datatype",
        "rep",
        "date",
        "pseudo_sample",
    ]
    return out[columns]


def config_text(group: ClientGroup, smpc: bool = True) -> str:
    reference = (
        f'"{group.reference_batch}"'
        if isinstance(group.reference_batch, str)
        else str(group.reference_batch).lower()
    )
    smpc_value = str(smpc).lower()
    return "\n".join(
        [
            "flimmaBatchCorrection:",
            "  batch_col: batch",
            "  covariates:",
            "  - D6",
            "  - F7",
            "  - M8",
            "  data_filename: intensities_log_UNION.tsv",
            "  design_filename: design.tsv",
            '  design_separator: "\\t"',
            "  expression_file_flag: true",
            "  index_col: rowname",
            "  min_samples: 0",
            "  normalizationMethod: null",
            f"  position: {group.position}",
            f"  reference_batch: {reference}",
            '  separator: "\\t"',
            f"  smpc: {smpc_value}",
            "",
        ]
    )


def prepare_fedrbe_clients(base_dir: Path, modality: str) -> pd.DataFrame:
    """Create FedRBE client folders for one modality."""
    expr, metadata = load_prepared_modality(base_dir, modality)
    groups = make_client_groups(metadata)
    clients_root = base_dir / "before_fedrbe" / modality
    clients_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for group in groups:
        client_meta = (
            metadata[metadata["batch_code"].isin(group.batch_codes)]
            .sort_values(["batch_code", "condition", "rep"])
            .reset_index(drop=True)
        )
        if len(client_meta) <= 18:
            raise ValueError(f"{modality}/{group.name}: not enough samples for FedRBE privacy.")

        client_dir = clients_root / group.name
        write_feature_matrix(expr.loc[:, client_meta["file"]], client_dir / "intensities_log_UNION.tsv")
        write_table(design_for_client(client_meta), client_dir / "design.tsv", quote=True)
        (client_dir / "config.yml").write_text(config_text(group, smpc=True))

        rows.append(
            {
                "modality": modality,
                "client": group.name,
                "position": group.position,
                "n_batches": len(group.batches),
                "batch_codes": ",".join(group.batch_codes),
                "batches": ",".join(group.batches),
                "n_samples": len(client_meta),
                "reference_batch": group.reference_batch,
                "path": str(client_dir),
            }
        )

    summary = pd.DataFrame(rows)
    write_table(summary, clients_root / "fedrbe_client_groups.tsv", quote=False)
    write_table(summary, base_dir / "after" / modality / "fedrbe_client_groups.tsv", quote=False)
    return summary


def cohorts_order_from_groups(groups: Sequence[ClientGroup]) -> List[str]:
    """Replicate the app's deterministic cohort ordering."""
    cohorts: List[str] = []
    for group in sorted(groups, key=lambda item: item.position):
        labels = sorted([f"{group.name}|{batch}" for batch in group.batches])
        if isinstance(group.reference_batch, str):
            reference_label = f"{group.name}|{group.reference_batch}"
            labels = [label for label in labels if label != reference_label] + [reference_label]
        cohorts.extend(labels)
    return cohorts


def build_design_matrix(client_meta: pd.DataFrame, client_name: str, cohorts: Sequence[str]) -> pd.DataFrame:
    """Build the same design columns as the FedRBE app."""
    design = pd.DataFrame(index=client_meta["file"].astype(str))
    design["intercept"] = 1.0
    for covariate in COVARIATES:
        design[covariate] = (client_meta["condition"].astype(str).values == covariate).astype(float)

    design_cohorts = list(cohorts[:-1])
    for cohort in design_cohorts:
        design[cohort] = 0.0

    for cohort in design_cohorts:
        cohort_client, cohort_batch = cohort.split("|", 1)
        if cohort_client == client_name:
            sample_mask = client_meta["batch"].astype(str).values == cohort_batch
            design.loc[sample_mask, cohort] = 1.0

    reference_client, reference_batch = cohorts[-1].split("|", 1)
    if reference_client == client_name:
        reference_mask = client_meta["batch"].astype(str).values == reference_batch
        design.loc[reference_mask, design_cohorts] = -1.0

    return design[["intercept", *COVARIATES, *design_cohorts]]


def fedrbe_simulate_modality(base_dir: Path, modality: str) -> Dict[str, float | int | str]:
    """Run local XTX/XTY aggregation equivalent to FedRBE for one modality.

    The local simulation is used for reproducible validation in environments
    where FeatureCloud Python dependencies are not installed. It uses the same
    client grouping, design coding, covariates, and batch-effect subtraction as
    the app.
    """
    expr, metadata = load_prepared_modality(base_dir, modality)
    if expr.isna().to_numpy().any():
        raise ValueError(f"{modality}: FedRBE simulation expects a complete feature matrix.")

    groups = make_client_groups(metadata)
    cohorts = cohorts_order_from_groups(groups)
    design_cohorts = cohorts[:-1]
    n_batches = len(cohorts)
    min_samples = n_batches + len(COVARIATES) + 1

    clients = []
    for group in groups:
        client_meta = (
            metadata[metadata["batch_code"].isin(group.batch_codes)]
            .sort_values(["batch_code", "condition", "rep"])
            .reset_index(drop=True)
        )
        if len(client_meta) < min_samples:
            raise ValueError(
                f"{modality}/{group.name}: {len(client_meta)} samples < required {min_samples}."
            )
        client_expr = expr.loc[:, client_meta["file"]]
        design = build_design_matrix(client_meta, group.name, cohorts)
        if design.shape[0] <= design.shape[1]:
            raise ValueError(
                f"{modality}/{group.name}: design has {design.shape[0]} rows "
                f"and {design.shape[1]} columns."
            )
        clients.append((group, client_meta, client_expr, design))

    feature_names = expr.index.astype(str).tolist()
    k = clients[0][3].shape[1]
    xtx_global = np.zeros((k, k), dtype=float)
    xty_global = np.zeros((len(feature_names), k), dtype=float)

    for _group, _client_meta, client_expr, design in clients:
        x = design.to_numpy(dtype=float)
        y = client_expr.to_numpy(dtype=float)
        xty_global += y @ x
        xtx_global += x.T @ x

    try:
        beta = np.linalg.solve(xtx_global, xty_global.T).T
    except np.linalg.LinAlgError:
        beta = (np.linalg.pinv(xtx_global) @ xty_global.T).T

    corrected_parts = []
    out_dir = base_dir / "after" / modality
    individual_dir = out_dir / "individual_results_fedsim"
    individual_dir.mkdir(parents=True, exist_ok=True)

    batch_beta = beta[:, len(["intercept", *COVARIATES]) :]
    for group, client_meta, client_expr, design in clients:
        batch_design = design.loc[:, design_cohorts].to_numpy(dtype=float)
        batch_effect = batch_beta @ batch_design.T
        corrected = client_expr.to_numpy(dtype=float) - batch_effect
        corrected_df = pd.DataFrame(corrected, index=feature_names, columns=client_expr.columns)
        corrected_parts.append(corrected_df)

        client_dir = individual_dir / group.name
        write_feature_matrix(corrected_df, client_dir / "only_batch_corrected_data.csv")
        covariates = design_for_client(client_meta).set_index("file")[COVARIATES].T
        full_corrected = pd.concat([corrected_df, covariates.loc[:, corrected_df.columns]])
        write_feature_matrix(full_corrected, client_dir / "full_corrected_data.csv")
        (client_dir / "report.txt").write_text(
            "\n".join(
                [
                    f"Client {group.name}:",
                    "Local FedRBE-equivalent XTX/XTY simulation.",
                    f"Cohorts order: {cohorts}",
                    f"Design columns: {design.columns.tolist()}",
                    f"Corrected shape: {corrected_df.shape}",
                    "",
                ]
            )
        )

    corrected_all = pd.concat(corrected_parts, axis=1).loc[:, metadata["file"]]
    write_feature_matrix(corrected_all, out_dir / "FedSim_corrected_data.tsv")

    central_path = out_dir / "intensities_log_Rcorrected_UNION.tsv"
    max_abs_diff = math.nan
    mean_abs_diff = math.nan
    max_abs_diff_row_centered = math.nan
    if central_path.exists():
        central = read_feature_matrix(central_path)
        central = central.loc[corrected_all.index, corrected_all.columns]
        diff = corrected_all - central
        max_abs_diff = float(np.nanmax(np.abs(diff.to_numpy())))
        mean_abs_diff = float(np.nanmean(np.abs(diff.to_numpy())))
        fed_centered = corrected_all.sub(corrected_all.mean(axis=1), axis=0)
        central_centered = central.sub(central.mean(axis=1), axis=0)
        centered_diff = fed_centered - central_centered
        max_abs_diff_row_centered = float(np.nanmax(np.abs(centered_diff.to_numpy())))

    return {
        "modality": modality,
        "features": int(corrected_all.shape[0]),
        "samples": int(corrected_all.shape[1]),
        "clients": int(len(groups)),
        "batches": int(n_batches),
        "design_columns": int(k),
        "required_min_samples_per_client": int(min_samples),
        "max_abs_diff_vs_central_limma": max_abs_diff,
        "mean_abs_diff_vs_central_limma": mean_abs_diff,
        "max_abs_diff_vs_central_after_row_centering": max_abs_diff_row_centered,
    }


def row_zscore(df: pd.DataFrame) -> pd.DataFrame:
    values = df.to_numpy(dtype=float)
    means = np.nanmean(values, axis=1, keepdims=True)
    sds = np.nanstd(values, axis=1, ddof=1, keepdims=True)
    sds[~np.isfinite(sds) | (sds == 0)] = 1.0
    scaled = (values - means) / sds
    return pd.DataFrame(scaled, index=df.index, columns=df.columns)


def build_fedsim_combined_kmeans_matrix(base_dir: Path) -> pd.DataFrame:
    """Build equal-weight all-modality matrix from FedSim-corrected outputs."""
    blocks = []
    sample_keys: List[str] | None = None
    combined_metadata = None

    for modality in MODALITIES:
        corrected = read_feature_matrix(base_dir / "after" / modality / "FedSim_corrected_data.tsv")
        metadata = pd.read_csv(base_dir / "before" / modality / "metadata.tsv", sep="\t")
        metadata = metadata.sort_values(["batch_code", "condition", "rep"]).reset_index(drop=True)
        corrected = corrected.loc[:, metadata["file"]]
        corrected.columns = metadata["pseudo_sample"]

        if sample_keys is None:
            sample_keys = sorted(metadata["pseudo_sample"].unique().tolist())
            combined_metadata = metadata[["pseudo_sample", "batch_code", "condition", "rep"]].drop_duplicates()
        elif sorted(metadata["pseudo_sample"].unique().tolist()) != sample_keys:
            raise ValueError(f"{modality}: pseudo-sample keys do not match previous modalities.")

        corrected = corrected.loc[:, sample_keys]
        block = row_zscore(corrected) / math.sqrt(corrected.shape[0])
        block.index = [f"{modality}__{feature}" for feature in block.index]
        blocks.append(block)

    if sample_keys is None or combined_metadata is None:
        raise ValueError("No modalities were available for combined matrix.")

    combined = pd.concat(blocks, axis=0)
    write_feature_matrix(combined, base_dir / "after" / "all_modalities_fedsim_kmeans_matrix.tsv")
    return combined


def write_fedsim_datainfo(base_dir: Path, summaries: pd.DataFrame) -> None:
    datainfo = {
        "method": "Local FedRBE-equivalent XTX/XTY simulation",
        "reason": (
            "FeatureCloud Python dependencies are optional; this output uses the same "
            "client grouping and linear batch-effect removal as the FedRBE app."
        ),
        "covariates": COVARIATES,
        "batch_col": "batch",
        "modalities": summaries.to_dict(orient="records"),
    }
    (base_dir / "after" / "fedsim_datainfo.json").write_text(json.dumps(datainfo, indent=2))


def run_all_fedsim(base_dir: Path) -> pd.DataFrame:
    """Prepare clients and run local FedRBE-equivalent correction for all modalities."""
    client_summaries = []
    correction_summaries = []
    for modality in MODALITIES:
        client_summaries.append(prepare_fedrbe_clients(base_dir, modality))
        correction_summaries.append(fedrbe_simulate_modality(base_dir, modality))

    client_summary = pd.concat(client_summaries, axis=0, ignore_index=True)
    correction_summary = pd.DataFrame(correction_summaries)
    write_table(client_summary, base_dir / "before_fedrbe" / "fedrbe_client_groups.tsv", quote=False)
    write_table(correction_summary, base_dir / "after" / "fedsim_correction_summary.tsv", quote=False)
    write_fedsim_datainfo(base_dir, correction_summary)
    build_fedsim_combined_kmeans_matrix(base_dir)
    return correction_summary
