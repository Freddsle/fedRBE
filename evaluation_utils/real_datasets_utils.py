"""
Shared Python utilities for real-dataset evaluation (data loading, alignment,
k-means, metrics, FeatureCloud input preparation).

Dataset configurations are defined once in ``evaluation_utils/datasets.yaml``
and loaded by ``dataset_configs()``.
"""

from __future__ import annotations

import csv
import itertools
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetConfig:
    """Immutable configuration for one dataset (paths, column names, glob patterns)."""

    name: str
    before_matrix: Path
    before_site_matrix_file: str
    corrected_central: Path
    corrected_federated: Path
    sites_root: Path
    site_glob: str
    design_file: str
    sample_col: str
    condition_col: str
    n_init: int = 10
    extra: Dict[str, str] = None  # type: ignore[assignment]


def dataset_configs(repo_root: Path) -> Dict[str, DatasetConfig]:
    """Return a dict of dataset name -> DatasetConfig, loaded from datasets.yaml."""
    yaml_path = Path(__file__).resolve().parent / "datasets.yaml"
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)
    configs = {}
    for name, d in raw.items():
        configs[name] = DatasetConfig(
            name=name,
            before_matrix=repo_root / d["before_matrix"],
            before_site_matrix_file=d["before_site_matrix_file"],
            corrected_central=repo_root / d["corrected_central"],
            corrected_federated=repo_root / d["corrected_federated"],
            sites_root=repo_root / d["sites_root"],
            site_glob=d["site_glob"],
            design_file=d["design_file"],
            sample_col=d["sample_col"],
            condition_col=d["condition_col"],
            n_init=int(d.get("n_init", 10)),
            extra={k: str(d[k]) for k in d.get("extra", {})},
        )
    return configs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_token(value: object) -> str:
    """Strip whitespace and surrounding quotes from a value."""
    return str(value).strip().strip('"').strip("'")


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def choose_corrected_path(cfg: DatasetConfig, source: str) -> Tuple[Path, str]:
    """Resolve the path to the corrected matrix (central, federated, or auto).

    Returns (path, kind) where kind is "central" or "federated".
    """
    if source == "central":
        if not cfg.corrected_central.exists():
            raise FileNotFoundError(f"Missing central corrected matrix: {cfg.corrected_central}")
        return cfg.corrected_central, "central"

    if source == "federated":
        if not cfg.corrected_federated.exists():
            raise FileNotFoundError(f"Missing federated corrected matrix: {cfg.corrected_federated}")
        return cfg.corrected_federated, "federated"

    # auto: prefer federated, fall back to central
    if cfg.corrected_federated.exists():
        return cfg.corrected_federated, "federated"
    if cfg.corrected_central.exists():
        return cfg.corrected_central, "central"
    raise FileNotFoundError(
        f"No corrected matrix found for {cfg.name} "
        f"(checked {cfg.corrected_central} and {cfg.corrected_federated})"
    )


# ---------------------------------------------------------------------------
# Metadata loading
# ---------------------------------------------------------------------------

def discover_clients(cfg: DatasetConfig) -> List[Path]:
    """Find client (site) directories under sites_root matching site_glob."""
    clients = []
    for candidate in sorted(cfg.sites_root.glob(cfg.site_glob), key=lambda path: path.name):
        if candidate.is_dir() and (candidate / cfg.design_file).exists():
            clients.append(candidate)
    if not clients:
        raise FileNotFoundError(
            f"No client directories with {cfg.design_file} found in {cfg.sites_root}"
        )
    return clients


def load_metadata(cfg: DatasetConfig, clients: Sequence[Path]) -> pd.DataFrame:
    """Load and merge design files from all client directories into a unified metadata table.

    Returns a DataFrame with columns: file, condition, lab.
    """
    rows = []
    sample_col = normalize_token(cfg.sample_col)
    condition_col = normalize_token(cfg.condition_col)

    for client in clients:
        design_path = client / cfg.design_file
        design = pd.read_csv(design_path, sep="\t")
        design.columns = [normalize_token(column) for column in design.columns]

        if sample_col not in design.columns:
            raise ValueError(f"Column '{sample_col}' not found in {design_path}")
        if condition_col not in design.columns:
            raise ValueError(f"Column '{condition_col}' not found in {design_path}")

        subset = design[[sample_col, condition_col]].copy()
        subset.columns = ["file", "condition"]
        subset["file"] = subset["file"].map(normalize_token)
        subset["condition"] = subset["condition"].map(normalize_token)
        subset["lab"] = client.name
        rows.append(subset)

    metadata = pd.concat(rows, axis=0, ignore_index=True)
    metadata = metadata.dropna(subset=["file", "condition", "lab"])
    if metadata["file"].duplicated().any():
        dup = metadata.loc[metadata["file"].duplicated(), "file"].head(5).tolist()
        raise ValueError(f"Duplicate sample IDs in metadata: {dup}")
    return metadata


# ---------------------------------------------------------------------------
# Matrix loading
# ---------------------------------------------------------------------------

def load_feature_matrix(path: Path) -> pd.DataFrame:
    """Load a TSV expression matrix (features x samples) with the first column as row index."""
    df = pd.read_csv(path, sep="\t", low_memory=False)
    if df.shape[1] < 2:
        raise ValueError(f"Unexpected matrix shape for {path}: {df.shape}")
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "rowname"})
    df["rowname"] = df["rowname"].map(normalize_token)
    df = df.set_index("rowname")
    df.columns = [normalize_token(column) for column in df.columns]
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=0, how="all")
    if df.empty:
        raise ValueError(f"Matrix became empty after dropping all-NA rows: {path}")
    return df


def load_before_matrix_from_sites(cfg: DatasetConfig, clients: Sequence[Path]) -> pd.DataFrame:
    """Merge per-site uncorrected matrices into a single features x samples DataFrame."""
    parts: List[pd.DataFrame] = []
    for client in clients:
        matrix_path = client / cfg.before_site_matrix_file
        if not matrix_path.exists():
            raise FileNotFoundError(f"Missing per-site matrix file: {matrix_path}")
        parts.append(load_feature_matrix(matrix_path))

    if not parts:
        raise FileNotFoundError(f"No per-site matrices found for {cfg.name} under {cfg.sites_root}")

    merged = pd.concat(parts, axis=1, join="outer")
    dup_cols = merged.columns[merged.columns.duplicated()].unique().tolist()
    if dup_cols:
        raise ValueError(
            f"Duplicate sample columns while merging per-site before matrices for {cfg.name}. "
            f"Examples: {dup_cols[:5]}"
        )
    if merged.empty:
        raise ValueError(f"Merged per-site before matrix is empty for {cfg.name}")
    return merged


def load_before_matrix_from_dir(before_dir: Path, filename: str = "intensities.tsv") -> pd.DataFrame:
    """Merge per-subdirectory matrices into a single features x samples DataFrame.

    Iterates direct subdirectories of *before_dir* in sorted order, loads *filename*
    from each, and concatenates column-wise.  Suitable for simulated data where labs are
    plain sub-folders (no DatasetConfig required).
    """
    parts: List[pd.DataFrame] = []
    for sub in sorted(before_dir.iterdir()):
        matrix_path = sub / filename
        if sub.is_dir() and matrix_path.exists():
            parts.append(load_feature_matrix(matrix_path))

    if not parts:
        raise FileNotFoundError(f"No {filename!r} files found in subdirectories of {before_dir}")

    merged = pd.concat(parts, axis=1, join="outer")
    dup_cols = merged.columns[merged.columns.duplicated()].unique().tolist()
    if dup_cols:
        raise ValueError(
            f"Duplicate sample columns while merging matrices under {before_dir}. "
            f"Examples: {dup_cols[:5]}"
        )
    if merged.empty:
        raise ValueError(f"Merged matrix is empty for {before_dir}")
    return merged


def filter_matrix_to_available(
    matrix: pd.DataFrame, metadata: pd.DataFrame, label: str
) -> pd.DataFrame:
    """Restrict matrix columns to samples listed in metadata, then drop any-NA feature rows.

    Unlike :func:`align_matrix_to_metadata`, this does NOT raise when samples are absent
    from the matrix — it silently keeps only the intersection.  Use this when the matrix
    may contain a subset of all samples (e.g. per-run simulated data).
    """
    available = [s for s in metadata["file"] if s in matrix.columns]
    return drop_rows_with_any_na(matrix.loc[:, available], label)


# ---------------------------------------------------------------------------
# Matrix alignment and filtering
# ---------------------------------------------------------------------------

def align_matrix_to_metadata(
    matrix: pd.DataFrame, metadata: pd.DataFrame, matrix_label: str
) -> pd.DataFrame:
    """Reorder matrix columns to match metadata sample order; raise if any missing."""
    samples = metadata["file"].tolist()
    missing = [sample for sample in samples if sample not in matrix.columns]
    if missing:
        raise ValueError(
            f"{matrix_label}: {len(missing)} samples are missing in matrix columns. "
            f"Examples: {missing[:5]}"
        )
    return matrix.loc[:, samples]


def drop_rows_with_any_na(matrix: pd.DataFrame, matrix_label: str) -> pd.DataFrame:
    """Remove feature rows containing any NA value (required for k-means)."""
    total_rows = matrix.shape[0]
    filtered = matrix.dropna(axis=0, how="any")
    removed = total_rows - filtered.shape[0]
    print(
        f"[{matrix_label}] remove_na=True dropped {removed} of {total_rows} feature rows containing NA"
    )
    if filtered.empty:
        raise ValueError(
            f"{matrix_label}: all feature rows were removed by --remove-na. "
            "No data left for k-means."
        )
    return filtered


# ---------------------------------------------------------------------------
# Scaling and k-means
# ---------------------------------------------------------------------------

def scale_like_federated(feature_by_sample: pd.DataFrame) -> np.ndarray:
    """Center and variance-scale each feature row, then transpose to samples x features."""
    values = feature_by_sample.to_numpy(dtype=float)
    means = np.nanmean(values, axis=1, keepdims=True)
    centered = values - means
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        std = np.nanstd(values, axis=1, ddof=1, keepdims=True)
    std[~np.isfinite(std)] = 1.0
    std[std == 0.0] = 1.0
    scaled = centered / std
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
    return scaled.T


def run_central_kmeans(
    feature_by_sample: pd.DataFrame,
    k_values: Sequence[int],
    seed: int,
    n_init: int = 10,
) -> Dict[int, pd.Series]:
    """Run k-means for each k and return {k: Series of cluster labels indexed by sample name}.

    Applies a per-feature ``StandardScaler`` pass (centering + scale by population std,
    ddof=0) before running k-means, matching the federated ``fc_kmeans`` app.

    Parameters
    ----------
    n_init
        Number of k-means random initializations.
    """
    array = feature_by_sample.to_numpy(dtype=float).T
    # StandardScaler: center + scale by population std (ddof=0), per feature.
    data = StandardScaler().fit_transform(array)
    samples = list(feature_by_sample.columns)
    outputs: Dict[int, pd.Series] = {}
    for k in sorted(set(k_values)):
        km = KMeans(n_clusters=k, random_state=seed, n_init=n_init)
        labels = km.fit_predict(data)
        outputs[k] = pd.Series(labels, index=samples)
    return outputs


def merge_central_clusters(
    metadata: pd.DataFrame,
    before_clusters: Dict[int, pd.Series],
    corrected_clusters: Dict[int, pd.Series],
) -> pd.DataFrame:
    """Join before/after cluster assignments onto metadata as new columns."""
    out = metadata.copy()
    for k, labels in before_clusters.items():
        out[f"Before_CtrlKm_{k}clusters"] = out["file"].map(labels)
    for k, labels in corrected_clusters.items():
        out[f"Cor_CtrlKm_{k}clusters"] = out["file"].map(labels)
    return out


# ---------------------------------------------------------------------------
# Prediction alignment and metrics
# ---------------------------------------------------------------------------

def align_predictions_to_truth(predicted: pd.Series, truth: pd.Series) -> pd.Series:
    """Map cluster labels to truth labels via best-match permutation (<=7 classes) or majority vote."""
    mask = predicted.notna() & truth.notna()
    aligned = pd.Series(index=predicted.index, dtype=object)
    if not mask.any():
        return aligned

    pred_values = predicted[mask].astype(str)
    truth_values = truth[mask].astype(str)

    pred_levels = sorted(pred_values.unique())
    truth_levels = sorted(truth_values.unique())
    mapping: Dict[str, str] = {}

    if len(pred_levels) == len(truth_levels) and len(pred_levels) <= 7:
        best_acc = -1.0
        for perm in itertools.permutations(truth_levels):
            candidate = dict(zip(pred_levels, perm))
            mapped = pred_values.map(candidate)
            acc = float((mapped == truth_values).mean())
            if acc > best_acc:
                best_acc = acc
                mapping = candidate
    else:
        for level in pred_levels:
            votes = truth_values[pred_values == level]
            if votes.empty:
                continue
            mapping[level] = votes.mode().iloc[0]

    aligned.loc[mask] = pred_values.map(mapping)
    return aligned


def calculate_metrics(true_labels: pd.Series, predicted_labels: pd.Series) -> Dict[str, float]:
    """Compute ARI from aligned labels."""
    mask = true_labels.notna() & predicted_labels.notna()
    if not mask.any():
        return {"ARI": np.nan, "N": 0}

    y_true = true_labels[mask].astype(str)
    y_pred = predicted_labels[mask].astype(str)
    ari = float(adjusted_rand_score(y_true, y_pred))
    return {"ARI": ari, "N": int(mask.sum())}


# ---------------------------------------------------------------------------
# FeatureCloud k-means input generation
# ---------------------------------------------------------------------------

def write_feature_matrix(df: pd.DataFrame, path: Path) -> None:
    """Write a feature-by-sample matrix as an unquoted TSV with a ``rowname`` index column."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out.insert(0, "rowname", out.index)
    out.to_csv(path, sep="\t", index=False)


def write_intensities(df: pd.DataFrame, path: Path) -> None:
    """Write a feature-by-sample matrix as a quoted TSV for the fc_kmeans app.

    Adds a ``rowname`` column as the first field (gene/feature name).
    """
    out = df.copy()
    out.insert(0, "rowname", out.index)
    out.to_csv(path, sep="\t", index=False, quoting=csv.QUOTE_NONNUMERIC)


def write_design(df: pd.DataFrame, path: Path) -> None:
    """Write a design table with file, condition, lab, and integer-coded A column.

    The ``A`` column is a 0-based integer encoding of the condition factor.
    """
    out = df[["file", "condition", "lab"]].copy()
    out["A"] = pd.Categorical(out["condition"]).codes
    out.to_csv(path, sep="\t", index=False, quoting=csv.QUOTE_NONNUMERIC)


def write_kmeans_config(
    path: Path,
    k_values: Sequence[int],
    n_init_local: int = 50,
    n_init_global: int = 50,
    max_global_iter: int = 1,
    seed: int = 11,
) -> None:
    """Write a ``config_kmeans.yml`` for the FeatureCloud fc_kmeans app.

    The app applies its own per-feature centering and variance scaling on each
    client's intensities before running k-means.
    """
    k_min = min(k_values)
    k_max = max(k_values)
    content = (
        "fc_kmeans:\n"
        "  algorithm:\n"
        f"    k_max: {k_max}\n"
        f"    k_min: {k_min}\n"
        "    k_step: 1\n"
        "    cluster_on: column\n"
        f"    seed: {seed}\n"
        f"    n_init_local: {n_init_local}\n"
        f"    n_init_global: {n_init_global}\n"
        f"    max_global_iter: {max_global_iter}\n"
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
        "    center: true\n"
        "    log_transform: false\n"
        "    max_nan_fraction: 1\n"
        "    variance: true\n"
    )
    path.write_text(content)


def prepare_variant_inputs(
    dataset_root: Path,
    variant_name: str,
    matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    clients: Sequence[Path],
    k_values: Sequence[int],
) -> Path:
    """Build per-site input directories for the FeatureCloud fc_kmeans app.

    Creates ``<dataset_root>/inputs/<variant_name>/<lab>/`` with
    ``intensities.tsv``, ``design.tsv``, and ``config_kmeans.yml`` per site.

    Returns the path to the variant directory.
    """
    variant_dir = dataset_root / "inputs" / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)

    for client in clients:
        lab = client.name
        lab_meta = metadata[metadata["lab"] == lab].copy()
        if lab_meta.empty:
            raise ValueError(f"No metadata rows for lab '{lab}'")

        missing = [s for s in lab_meta["file"] if s not in matrix.columns]
        if missing:
            raise ValueError(f"Missing samples for lab '{lab}': {missing[:5]}")

        lab_dir = variant_dir / lab
        lab_dir.mkdir(parents=True, exist_ok=True)
        write_intensities(matrix[lab_meta["file"]], lab_dir / "intensities.tsv")
        write_design(lab_meta, lab_dir / "design.tsv")
        write_kmeans_config(
            lab_dir / "config_kmeans.yml",
            k_values,
        )

    return variant_dir


# ---------------------------------------------------------------------------
# Federated cluster aggregation
# ---------------------------------------------------------------------------

def aggregate_fed_clusters(
    metadata: pd.DataFrame,
    run_output: Path,
    client_names: Sequence[str],
    k_values: Sequence[int],
) -> pd.DataFrame:
    """Read per-client clustering CSVs and join them onto metadata.

    Expects files named ``{idx}_{k}_clustering.csv`` (1-indexed) in *run_output*.
    Returns metadata with added ``Fed_{k}clusters`` columns.
    """
    frames = []
    for k in k_values:
        lab_frames = []
        for idx, _client in enumerate(client_names, start=1):
            clustering_path = run_output / f"{idx}_{k}_clustering.csv"
            if not clustering_path.exists():
                raise FileNotFoundError(f"Missing clustering file: {clustering_path}")
            df = pd.read_csv(clustering_path, sep=";", index_col=0)
            if df.empty:
                raise ValueError(f"Empty clustering file: {clustering_path}")
            df.columns = [f"Fed_{k}clusters"]
            lab_frames.append(df)
        frames.append(pd.concat(lab_frames, axis=0))

    merged = metadata.set_index("file").join(pd.concat(frames, axis=1), how="left")
    return merged.reset_index()


# ---------------------------------------------------------------------------
# Metrics evaluation, I/O and saving
# ---------------------------------------------------------------------------

def load_fed_metadata(path: Optional[Path]) -> Optional[pd.DataFrame]:
    """Load a federated metadata TSV, or return None if path is missing."""
    if path is None or not path.exists():
        return None
    return pd.read_csv(path, sep="\t")


def evaluate_metrics(
    dataset_name: str,
    central_res: pd.DataFrame,
    before_fed_res: Optional[pd.DataFrame],
    after_fed_res: Optional[pd.DataFrame],
    k_condition: int,
    k_batch: int,
) -> pd.DataFrame:
    """Compute ARI for central and federated k-means results.

    Returns a DataFrame with one row per (target, method) combination,
    containing ARI and sample count.
    """
    records: List[Dict] = []
    by_file_before_fed = (
        before_fed_res.set_index("file") if before_fed_res is not None else None
    )
    by_file_after_fed = (
        after_fed_res.set_index("file") if after_fed_res is not None else None
    )

    def fed_prediction(
        fed_df: Optional[pd.DataFrame], column: str, file_order: Sequence[str]
    ) -> Optional[pd.Series]:
        if fed_df is None or column not in fed_df.columns:
            return None
        return fed_df[column].reindex(file_order).reset_index(drop=True)

    tasks = [("condition", k_condition), ("batch", k_batch)]
    for target, k in tasks:
        truth_col = "condition" if target == "condition" else "lab"
        file_order = central_res["file"].tolist()
        truth = central_res[truth_col]

        method_specs: List[Tuple[str, Optional[pd.Series]]] = [
            (
                f"{dataset_name}_{target}_BC_Cntrl_{k}cls",
                central_res.get(f"Before_CtrlKm_{k}clusters"),
            ),
            (
                f"{dataset_name}_{target}_AC_Cntrl_{k}cls",
                central_res.get(f"Cor_CtrlKm_{k}clusters"),
            ),
            (
                f"{dataset_name}_{target}_BC_Fed_{k}cls",
                fed_prediction(by_file_before_fed, f"Fed_{k}clusters", file_order),
            ),
            (
                f"{dataset_name}_{target}_AC_Fed_{k}cls",
                fed_prediction(by_file_after_fed, f"Fed_{k}clusters", file_order),
            ),
        ]

        for method_name, predicted in method_specs:
            if predicted is None:
                continue
            metrics = calculate_metrics(truth, predicted)
            records.append(
                {
                    "Dataset": dataset_name,
                    "Target": target,
                    "K": int(k),
                    "Method": method_name,
                    **metrics,
                }
            )

    return pd.DataFrame(records)


def save_metrics_tables(metrics_df: pd.DataFrame, output_dir: Path) -> None:
    """Save ARI metrics table as a TSV file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_dir / "metrics_ari.tsv", sep="\t", index=False)


# ---------------------------------------------------------------------------
# Git LFS fallback helpers
# ---------------------------------------------------------------------------

def is_git_lfs_pointer(path: Path) -> bool:
    """Return True if *path* is an unfetched Git LFS pointer (< 200 bytes, starts with 'version')."""
    if not path.exists() or path.stat().st_size > 200:
        return False
    try:
        header = path.read_text(errors="replace")[:50]
        return header.startswith("version https://git-lfs")
    except Exception:
        return False


def load_matrix_with_lfs_fallback(
    path: Path, label: str, cfg: Optional[DatasetConfig] = None
) -> pd.DataFrame:
    """Load a feature matrix, falling back to merging per-site files if *path* is a Git LFS pointer.

    Fallback strategies (tried in order):
    1. ``individual_results/only_batch_corrected_data_*.csv`` next to *path* (corrected matrices).
    2. Per-site before matrices via ``load_before_matrix_from_sites(cfg, ...)`` (if *cfg* given).

    Parameters
    ----------
    path : Path
        Primary path to the matrix TSV.
    label : str
        Human-readable label for log messages.
    cfg : DatasetConfig, optional
        Dataset config for per-site fallback when the before matrix is LFS.
    """
    if path.exists() and not is_git_lfs_pointer(path):
        return load_feature_matrix(path)

    # Fallback 1: numbered per-site corrected CSVs
    indiv_dir = path.parent / "individual_results"
    if indiv_dir.exists():
        csvs = sorted(indiv_dir.glob("only_batch_corrected_data_*.csv"))
        real_csvs = [f for f in csvs if not is_git_lfs_pointer(f)]
        if real_csvs:
            print(f"[{label}] LFS pointer at {path}, merging {len(real_csvs)} per-site files")
            parts = [load_feature_matrix(f) for f in real_csvs]
            return pd.concat(parts, axis=1, join="outer")

    # Fallback 2: merge per-site before matrices
    if cfg is not None:
        clients = discover_clients(cfg)
        print(f"[{label}] LFS pointer at {path}, merging per-site before matrices")
        return load_before_matrix_from_sites(cfg, clients)

    raise FileNotFoundError(
        f"[{label}] File is an LFS pointer and no fallback data found: {path}"
    )
