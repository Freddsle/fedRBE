"""Prepare and evaluate Quartet multiomics k-means inputs.

The source matrices are the all-modality k-means-ready matrices produced by
``evaluation_data/multiomics/03_central_RBE.ipynb`` and
``evaluation_data/multiomics/04_run_fedrbe.ipynb``. They are already row-scaled
within modality and block-weighted, so this script runs k-means directly on the
sample columns instead of applying another global feature scaler.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Sequence

import pandas as pd
try:
    from sklearn.cluster import KMeans
except ModuleNotFoundError as exc:
    raise SystemExit(
        "scikit-learn is required. In this workspace, run with "
        "/home/yuliya-cosybio/miniforge3/bin/python."
    ) from exc

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation_utils.real_datasets_utils import (  # noqa: E402
    calculate_metrics,
    load_feature_matrix,
    write_design,
    write_feature_matrix,
    write_intensities,
)


DATASET = "multiomics"
SEED = 11
N_INIT = 50
K_CONDITION = 4
K_BATCH = 4  # one cluster per FedRBE client (the technical structure to be removed)
K_VALUES = [K_CONDITION, K_BATCH]

CLIENT_NAMES = [
    "client_01_L01",
    "client_02_L02",
    "client_03_L05_L04",
    "client_04_L03_L14",
]

DATA_DIR = REPO_ROOT / "evaluation_data" / "multiomics" / "after"
OUT_ROOT = REPO_ROOT / "evaluation_clusterization_after_correction" / "real_datasets"
DATASET_ROOT = OUT_ROOT / DATASET

MATRIX_SOURCES = {
    "before": DATA_DIR / "all_modalities_before_kmeans_matrix.tsv",
    "corrected": DATA_DIR / "all_modalities_corrected_kmeans_matrix.tsv",
    "corrected_fed": DATA_DIR / "all_modalities_fedsim_kmeans_matrix.tsv",
}

PREPARED_NAMES = {
    "before": "before_matrix.tsv",
    "corrected": "corrected_matrix.tsv",
    "corrected_fed": "corrected_fedrbe_matrix.tsv",
}


def load_metadata() -> pd.DataFrame:
    """Convert multiomics pseudo-sample metadata to the real-dataset schema."""
    path = DATA_DIR / "all_modalities_metadata.tsv"
    meta = pd.read_csv(path, sep="\t")
    required = {"pseudo_sample", "condition", "client", "rep"}
    missing = required.difference(meta.columns)
    if missing:
        raise ValueError(f"Missing metadata columns in {path}: {sorted(missing)}")

    out = (
        meta.loc[:, ["pseudo_sample", "condition", "client", "rep"]]
        .rename(columns={"pseudo_sample": "file", "client": "lab"})
        .sort_values(["lab", "condition", "rep"])
        .reset_index(drop=True)
    )
    out["condition"] = pd.Categorical(
        out["condition"], categories=["D5", "D6", "F7", "M8"], ordered=True
    )
    out["lab"] = pd.Categorical(out["lab"], categories=CLIENT_NAMES, ordered=True)
    out["condition"] = out["condition"].astype(str)
    out["lab"] = out["lab"].astype(str)
    return out


def align_matrix(matrix: pd.DataFrame, metadata: pd.DataFrame, label: str) -> pd.DataFrame:
    samples = metadata["file"].tolist()
    missing = [sample for sample in samples if sample not in matrix.columns]
    if missing:
        raise ValueError(f"{label}: missing samples: {missing[:5]}")
    aligned = matrix.loc[:, samples]
    if aligned.isna().to_numpy().any():
        raise ValueError(f"{label}: matrix contains NA values")
    if not pd.Index(aligned.columns).is_unique:
        raise ValueError(f"{label}: duplicate sample IDs")
    if not pd.Index(aligned.index).is_unique:
        raise ValueError(f"{label}: duplicate feature IDs")
    return aligned


def run_kmeans_ready_matrix(
    feature_by_sample: pd.DataFrame, k_values: Sequence[int], seed: int = SEED
) -> Dict[int, pd.Series]:
    """Run k-means on already scaled samples x features data."""
    data = feature_by_sample.to_numpy(dtype=float).T
    samples = list(feature_by_sample.columns)
    outputs: Dict[int, pd.Series] = {}
    for k in sorted(set(k_values)):
        km = KMeans(n_clusters=k, random_state=seed, n_init=N_INIT)
        outputs[k] = pd.Series(km.fit_predict(data), index=samples)
    return outputs


def add_clusters(
    metadata: pd.DataFrame, labels_by_variant: Dict[str, Dict[int, pd.Series]]
) -> pd.DataFrame:
    out = metadata.loc[:, ["file", "condition", "lab"]].copy()
    prefixes = {
        "before": "Before_CtrlKm",
        "corrected": "Cor_CtrlKm",
        "corrected_fed": "FedRBE_CtrlKm",
    }
    for variant, labels_by_k in labels_by_variant.items():
        for k, labels in labels_by_k.items():
            out[f"{prefixes[variant]}_{k}clusters"] = out["file"].map(labels)
    return out


def metric_record(
    dataset: str,
    target: str,
    k: int,
    method: str,
    truth: pd.Series,
    predicted: pd.Series,
) -> dict:
    metrics = calculate_metrics(truth, predicted)
    return {
        "Dataset": dataset,
        "Target": target,
        "K": int(k),
        "Method": method,
        **metrics,
    }


def evaluate(clustered: pd.DataFrame) -> pd.DataFrame:
    records = []
    tasks = [
        ("condition", K_CONDITION, "condition"),
        ("client", K_BATCH, "lab"),
    ]
    methods = [
        ("BC_Cntrl", "Before_CtrlKm"),
        ("AC_Cntrl", "Cor_CtrlKm"),
        ("AC_FedRBE_Cntrl", "FedRBE_CtrlKm"),
    ]
    for target, k, truth_col in tasks:
        truth = clustered[truth_col]
        for method_label, column_prefix in methods:
            column = f"{column_prefix}_{k}clusters"
            records.append(
                metric_record(
                    dataset=DATASET,
                    target=target,
                    k=k,
                    method=f"{DATASET}_{target}_{method_label}_{k}cls",
                    truth=truth,
                    predicted=clustered[column],
                )
            )
    return pd.DataFrame.from_records(records)


def prepare_fc_inputs(
    variant: str,
    matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    labs: Iterable[str],
) -> None:
    """Write per-batch FeatureCloud k-means inputs for one matrix variant."""
    variant_dir = DATASET_ROOT / "inputs" / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    for lab in labs:
        lab_meta = metadata.loc[metadata["lab"] == lab, ["file", "condition", "lab"]].copy()
        if lab_meta.empty:
            raise ValueError(f"{variant}: no samples for {lab}")
        lab_dir = variant_dir / lab
        lab_dir.mkdir(parents=True, exist_ok=True)
        write_intensities(matrix.loc[:, lab_meta["file"]], lab_dir / "intensities.tsv")
        write_design(lab_meta, lab_dir / "design.tsv")
        write_multiomics_kmeans_config(lab_dir / "config_kmeans.yml")


def write_multiomics_kmeans_config(path: Path) -> None:
    """Write fc_kmeans config without a second scaling step.

    The input matrices are already k-means-ready: each modality block was
    row-standardized and divided by sqrt(n_features) before concatenation.
    """
    content = (
        "fc_kmeans:\n"
        "  algorithm:\n"
        f"    k_max: {K_BATCH}\n"
        f"    k_min: {K_CONDITION}\n"
        "    k_step: 1\n"
        "    cluster_on: column\n"
        f"    seed: {SEED}\n"
        f"    n_init_local: {N_INIT}\n"
        f"    n_init_global: {N_INIT}\n"
        "    max_global_iter: 1\n"
        "  input:\n"
        "    delimiter: \"\\t\"\n"
        "    dir: \"\"\n"
        "    file: intensities.tsv\n"
        "  output:\n"
        "    centroids: centroids.csv\n"
        "    clustering: clustering.csv\n"
        "    delimiter: ;\n"
        "    dir: kmeans\n"
        "    silhouette: silhouette.csv\n"
        "  scaling:\n"
        "    center: false\n"
        "    log_transform: false\n"
        "    max_nan_fraction: 1\n"
        "    variance: false\n"
    )
    path.write_text(content)


def update_aggregate_metrics(metrics: pd.DataFrame) -> None:
    """Append/replace the multiomics rows in the aggregate real-dataset table."""
    aggregate_path = OUT_ROOT / "metrics_ari.tsv"
    if aggregate_path.exists():
        aggregate = pd.read_csv(aggregate_path, sep="\t")
        aggregate = aggregate.loc[aggregate["Dataset"] != DATASET].copy()
        aggregate = pd.concat([aggregate, metrics], ignore_index=True)
    else:
        aggregate = metrics
    aggregate.to_csv(aggregate_path, sep="\t", index=False)


def write_manifest(metadata: pd.DataFrame, matrices: Dict[str, pd.DataFrame]) -> None:
    manifest = {
        "dataset": DATASET,
        "samples": int(metadata.shape[0]),
        "condition_levels": sorted(metadata["condition"].unique().tolist()),
        "batch_levels": sorted(metadata["lab"].unique().tolist()),
        "k_condition": K_CONDITION,
        "k_batch": K_BATCH,
        "seed": SEED,
        "n_init": N_INIT,
        "scaling": "already row-scaled within modality and equal block-weighted",
        "matrices": {
            name: {"features": int(df.shape[0]), "samples": int(df.shape[1])}
            for name, df in matrices.items()
        },
    }
    (DATASET_ROOT / "kmeans_manifest.json").write_text(json.dumps(manifest, indent=2))


def main() -> None:
    metadata = load_metadata()
    labs = sorted(metadata["lab"].unique().tolist())
    matrices: Dict[str, pd.DataFrame] = {}

    for variant, source in MATRIX_SOURCES.items():
        if not source.exists():
            raise FileNotFoundError(f"Missing {variant} matrix: {source}")
        matrix = align_matrix(load_feature_matrix(source), metadata, variant)
        matrices[variant] = matrix
        write_feature_matrix(matrix, DATASET_ROOT / "prepared" / PREPARED_NAMES[variant])
        prepare_fc_inputs(variant, matrix, metadata, labs)

    metadata.loc[:, ["file", "condition", "lab"]].to_csv(
        DATASET_ROOT / "prepared" / "metadata.tsv", sep="\t", index=False
    )

    labels = {
        variant: run_kmeans_ready_matrix(matrix, K_VALUES, seed=SEED)
        for variant, matrix in matrices.items()
    }
    clustered = add_clusters(metadata, labels)

    runs_dir = DATASET_ROOT / "kmeans_res" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    clustered.loc[
        :,
        [
            "file",
            "condition",
            "lab",
            "Before_CtrlKm_4clusters",
            "Cor_CtrlKm_4clusters",
            "FedRBE_CtrlKm_4clusters",
        ],
    ].to_csv(runs_dir / "1_metadata_cntrl_kmeans_res.tsv", sep="\t", index=False)

    metrics = evaluate(clustered)
    metrics_dir = DATASET_ROOT / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metrics_dir / "metrics_ari.tsv", sep="\t", index=False)
    update_aggregate_metrics(metrics)
    write_manifest(metadata, matrices)

    print(f"Wrote {DATASET_ROOT}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
