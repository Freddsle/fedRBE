"""Utilities for Quartet multiomics FedRBE-style batch correction.

Lab/client structure (fixed; see ``02_prepare_RBE_inputs.ipynb`` for the
selection rationale): four clients are used per modality.  Only labs that
appear in all three modalities of the Quartet figshare release form
separate clients (L01, L02, L05); a fourth synthetic client combines L03
(Metab+RNA) with L14 (Protein) to cover all three layers.

Within each (lab, modality) we keep a deterministic subset of batches so
that every client has 12 or 24 libraries per modality. The selection rule
is hard-coded in ``SELECTED_BATCHES`` (see notebook README for details).

The biological covariate is the donor (D5/D6/F7/M8) with **D6 as the
reference level**, as in the Quartet figshare 22188349 design.
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
DONOR_REFERENCE = "D6"
COVARIATES = [donor for donor in DONOR_LEVELS if donor != DONOR_REFERENCE]

# Per-modality registry of which lab contributes which batches. The keys
# of each inner dict are the source ``lab`` IDs from the figshare metadata
# (L01..L15, enumerated independently within each modality).
SELECTED_BATCHES: Dict[str, Dict[str, List[str]]] = {
    "Transcriptomics": {
        "L01": ["P_ILM_L1_B1"],
        "L02": ["P_ILM_L2_B1"],
        "L05": ["P_ILM_L5_B1", "R_ILM_L5_B2"],
        "L03": ["P_BGI_L3_B1", "R_BGI_L3_B1"],
    },
    "Proteomics": {
        "L01": ["ABS_QTOF6600_1"],
        "L02": ["APT_QE-HFX_1"],
        "L05": ["FDU_Lumos_1", "FDU_QE-HFX_4"],
        "L14": ["TMO_Exploris480_1", "TMO_QE-HFX_1"],
    },
    "Metabolomics": {
        "L01": ["U_L1_01"],
        "L02": ["U_L2_01"],
        "L05": ["U_L5_01"],
        "L04": ["T_L4_01"],
        "L03": ["U_L3_01", "U_L3_02"],
    },
}

# Mapping from (modality, lab) -> client name. Two synthetic clients combine
# multiple labs:
#   client_03_L05_L04 = L05 (all modalities) + L04 (only Metabolomics)
#   client_04_L03_L14 = L03 (Metab+RNA) + L14 (Proteomics)
CLIENT_LAB_MAPS: Dict[str, Dict[str, str]] = {
    "Transcriptomics": {
        "L01": "client_01_L01",
        "L02": "client_02_L02",
        "L05": "client_03_L05_L04",
        "L03": "client_04_L03_L14",
    },
    "Proteomics": {
        "L01": "client_01_L01",
        "L02": "client_02_L02",
        "L05": "client_03_L05_L04",
        "L14": "client_04_L03_L14",
    },
    "Metabolomics": {
        "L01": "client_01_L01",
        "L02": "client_02_L02",
        "L05": "client_03_L05_L04",
        "L04": "client_03_L05_L04",
        "L03": "client_04_L03_L14",
    },
}

CLIENT_NAMES: List[str] = [
    "client_01_L01",
    "client_02_L02",
    "client_03_L05_L04",
    "client_04_L03_L14",
]


@dataclass(frozen=True)
class ClientGroup:
    name: str
    labs: List[str]
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
    out.to_csv(path, sep="\t", index=False, quoting=csv.QUOTE_NONNUMERIC, na_rep="NA")


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
    metadata["lab"] = metadata["lab"].astype(str)
    metadata["rep"] = metadata["rep"].astype(int)

    missing = [sample for sample in metadata["file"] if sample not in expr.columns]
    if missing:
        raise ValueError(f"{modality}: samples missing from matrix: {missing[:5]}")

    metadata = metadata.sort_values(["lab", "batch_code", "condition", "rep"]).reset_index(drop=True)
    expr = expr.loc[:, metadata["file"]]
    return expr, metadata


def make_client_groups(metadata: pd.DataFrame, modality: str) -> List[ClientGroup]:
    """Build the four fixed clients for ``modality`` from its prepared metadata.

    Lab → client mapping is taken from ``CLIENT_LAB_MAPS[modality]``. Each
    client carries 1 or 2 source batches (depending on the lab) and the
    original ``batch`` column is preserved as the within-client batch factor.
    The privacy requirement
    ``n_samples >= 1 + |covariates| + (n_total_batches - 1) + 1``
    is checked but never triggers a merge—this layout is designed so the
    smallest client (12 libraries) already exceeds the threshold.

    The reference batch (used by the FedRBE coordinator and by central limma
    for exact alignment) is the alphabetically-last batch of the last client.
    """
    lab_to_client = CLIENT_LAB_MAPS[modality]
    n_total_batches = metadata["batch"].nunique()
    min_samples_required = 1 + len(COVARIATES) + (n_total_batches - 1) + 1

    # Group rows by client according to the modality-specific lab map.
    by_client: Dict[str, Dict[str, list]] = {}
    for _, row in metadata.iterrows():
        client_name = lab_to_client.get(str(row["lab"]))
        if client_name is None:
            raise ValueError(
                f"{modality}: lab {row['lab']!r} has no client mapping. "
                f"Update CLIENT_LAB_MAPS or filter the metadata in 02_prepare_RBE_inputs."
            )
        bucket = by_client.setdefault(
            client_name,
            {"labs": [], "batch_codes": [], "batches": [], "n_samples": 0},
        )
        if row["lab"] not in bucket["labs"]:
            bucket["labs"].append(str(row["lab"]))
        if row["batch_code"] not in bucket["batch_codes"]:
            bucket["batch_codes"].append(str(row["batch_code"]))
        if row["batch"] not in bucket["batches"]:
            bucket["batches"].append(str(row["batch"]))
        bucket["n_samples"] += 1

    ordered = [name for name in CLIENT_NAMES if name in by_client]
    if not ordered:
        raise ValueError(f"{modality}: no clients produced from CLIENT_LAB_MAPS.")

    groups: List[ClientGroup] = []
    for idx, client_name in enumerate(ordered):
        bucket = by_client[client_name]
        if bucket["n_samples"] < min_samples_required:
            raise ValueError(
                f"{modality}/{client_name}: {bucket['n_samples']} samples < "
                f"required {min_samples_required} for FedRBE privacy."
            )
        is_last = idx == len(ordered) - 1
        reference_batch: str | bool = (
            sorted(bucket["batches"])[-1] if is_last else False
        )
        groups.append(
            ClientGroup(
                name=client_name,
                labs=sorted(bucket["labs"]),
                batch_codes=sorted(bucket["batch_codes"]),
                batches=sorted(bucket["batches"]),
                reference_batch=reference_batch,
                position=idx,
            )
        )
    return groups


def design_for_client(metadata: pd.DataFrame) -> pd.DataFrame:
    out = metadata.copy()
    for covariate in COVARIATES:
        out[covariate] = (out["condition"].astype(str) == covariate).astype(int)
    columns = [
        "file",
        *COVARIATES,
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
    cov_lines = [f"  - {cov}" for cov in COVARIATES]
    return "\n".join(
        [
            "flimmaBatchCorrection:",
            "  batch_col: batch",
            "  covariates:",
            *cov_lines,
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


def _client_membership(metadata: pd.DataFrame, modality: str) -> pd.Series:
    """Return the client name (one of CLIENT_NAMES) for each metadata row."""
    lab_to_client = CLIENT_LAB_MAPS[modality]
    membership = metadata["lab"].astype(str).map(lab_to_client)
    if membership.isna().any():
        bad = metadata.loc[membership.isna(), "lab"].unique().tolist()
        raise ValueError(
            f"{modality}: labs {bad} have no client mapping in CLIENT_LAB_MAPS."
        )
    return membership


def prepare_fedrbe_clients(base_dir: Path, modality: str) -> pd.DataFrame:
    """Create FedRBE client folders for one modality."""
    expr, metadata = load_prepared_modality(base_dir, modality)
    groups = make_client_groups(metadata, modality)
    membership = _client_membership(metadata, modality)

    clients_root = base_dir / "before" / modality
    clients_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for group in groups:
        client_meta = (
            metadata[membership == group.name]
            .sort_values(["lab", "batch_code", "condition", "rep"])
            .reset_index(drop=True)
        )
        client_expr = expr.loc[:, client_meta["file"]]
        all_na_rows = client_expr.isna().all(axis=1)
        n_all_na_rows = int(all_na_rows.sum())
        client_expr = client_expr.loc[~all_na_rows]

        client_dir = clients_root / group.name
        write_feature_matrix(client_expr, client_dir / "intensities_log_UNION.tsv")
        write_table(design_for_client(client_meta), client_dir / "design.tsv", quote=True)
        (client_dir / "config.yml").write_text(config_text(group, smpc=True))

        rows.append(
            {
                "modality": modality,
                "client": group.name,
                "position": group.position,
                "labs": ",".join(group.labs),
                "n_labs": len(group.labs),
                "n_batches": len(group.batches),
                "batch_codes": ",".join(group.batch_codes),
                "batches": ",".join(group.batches),
                "n_samples": len(client_meta),
                "n_features": int(client_expr.shape[0]),
                "all_na_feature_rows_dropped": n_all_na_rows,
                "reference_batch": group.reference_batch,
                "path": str(Path("before") / modality / group.name),
            }
        )

    summary = pd.DataFrame(rows)
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

    groups = make_client_groups(metadata, modality)
    membership = _client_membership(metadata, modality)
    cohorts = cohorts_order_from_groups(groups)
    design_cohorts = cohorts[:-1]
    n_batches = len(cohorts)

    clients = []
    for group in groups:
        client_meta = (
            metadata[membership == group.name]
            .sort_values(["lab", "batch_code", "condition", "rep"])
            .reset_index(drop=True)
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
    n_features = len(feature_names)
    k = clients[0][3].shape[1]
    xtx_global = np.zeros((n_features, k, k), dtype=float)
    xty_global = np.zeros((n_features, k), dtype=float)

    for _group, _client_meta, client_expr, design in clients:
        x = design.to_numpy(dtype=float)
        y = client_expr.to_numpy(dtype=float)
        observed = np.isfinite(y)
        y_filled = np.where(observed, y, 0.0)
        xty_global += y_filled @ x
        xtx_global += np.einsum("ni,nj,fn->fij", x, x, observed.astype(float), optimize=True)

    beta = np.zeros((n_features, k), dtype=float)
    for feature_idx in range(n_features):
        xtx = xtx_global[feature_idx]
        xty = xty_global[feature_idx]
        if not np.any(np.isfinite(xty)) or np.allclose(xtx, 0.0):
            beta[feature_idx, :] = np.nan
            continue
        try:
            beta[feature_idx, :] = np.linalg.solve(xtx, xty)
        except np.linalg.LinAlgError:
            beta[feature_idx, :] = np.linalg.pinv(xtx) @ xty

    corrected_parts = []
    out_dir = base_dir / "after" / modality
    out_dir.mkdir(parents=True, exist_ok=True)

    batch_beta = beta[:, len(["intercept", *COVARIATES]) :]
    for group, client_meta, client_expr, design in clients:
        batch_design = design.loc[:, design_cohorts].to_numpy(dtype=float)
        batch_effect = batch_beta @ batch_design.T
        raw_values = client_expr.to_numpy(dtype=float)
        corrected = raw_values - batch_effect
        corrected[~np.isfinite(raw_values)] = np.nan
        corrected_df = pd.DataFrame(corrected, index=feature_names, columns=client_expr.columns)
        corrected_parts.append(corrected_df)

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
        "missing_cells": int(corrected_all.isna().to_numpy().sum()),
        "rows_with_any_missing": int(corrected_all.isna().any(axis=1).sum()),
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
    """Run local FedRBE-equivalent correction using clients prepared by step 02."""
    groups_path = base_dir / "before" / "fedrbe_client_groups.tsv"
    if not groups_path.exists():
        raise FileNotFoundError(f"Missing {groups_path}. Run 02_prepare_RBE_inputs.ipynb first.")
    client_summary = pd.read_csv(groups_path, sep="\t")
    for client_path in client_summary["path"]:
        expected = base_dir / client_path / "intensities_log_UNION.tsv"
        if not expected.exists():
            raise FileNotFoundError(f"Missing prepared client matrix: {expected}")

    correction_summaries = []
    for modality in MODALITIES:
        correction_summaries.append(fedrbe_simulate_modality(base_dir, modality))

    correction_summary = pd.DataFrame(correction_summaries)
    write_table(correction_summary, base_dir / "after" / "fedsim_correction_summary.tsv", quote=False)
    write_fedsim_datainfo(base_dir, correction_summary)
    build_fedsim_combined_kmeans_matrix(base_dir)
    return correction_summary
