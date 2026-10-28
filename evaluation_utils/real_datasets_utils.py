"""
Shared Python utilities for real-dataset evaluation (data loading, alignment,
k-means, metrics, FeatureCloud input preparation).

Dataset configurations are defined once in ``evaluation_utils/datasets.yaml``
and loaded by ``dataset_configs()``.
"""

from __future__ import annotations

import csv
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

QUARTET_LAYERS = ("Transcriptomics", "Proteomics", "Metabolomics")


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
    extra: Dict[str, str] = None  # type: ignore[assignment]
    correction_arm: Optional[str] = None


def dataset_configs(
    repo_root: Path, yaml_path: Optional[Path] = None,
    correction_arm: str = "label_aware",
) -> Dict[str, DatasetConfig]:
    """Load canonical paths for an arm, or use a custom registry's explicit paths."""
    if correction_arm not in ("label_aware", "no_label"):
        raise ValueError("correction_arm must be 'label_aware' or 'no_label'")
    if yaml_path is not None and correction_arm != "label_aware":
        raise ValueError("Set correction paths in the custom registry instead of correction_arm")
    canonical = yaml_path is None
    yaml_path = (Path(__file__).resolve().parent / "datasets.yaml"
                 if yaml_path is None else repo_root / yaml_path)
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)
    if canonical:
        for d in raw.values():
            for key in ("corrected_central", "corrected_federated"):
                d[key] = d[key].replace("/after/label_aware/", f"/after/{correction_arm}/")
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
            extra={k: str(value) for k, value in d.get("extra", {}).items()},
            correction_arm=correction_arm if canonical else None,
        )
    return configs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_token(value: object) -> str:
    """Strip whitespace and surrounding quotes from a value."""
    return str(value).strip().strip('"').strip("'")


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

def read_correction_matrix(path: Path) -> pd.DataFrame:
    """Read numeric correction inputs without dropping all-missing feature rows."""
    matrix = pd.read_csv(path, sep="\t", index_col=0)
    matrix.index = matrix.index.map(normalize_token)
    matrix.index.name = "rowname"
    matrix.columns = matrix.columns.map(normalize_token)
    matrix = matrix.apply(pd.to_numeric)
    if not matrix.index.is_unique or not matrix.columns.is_unique:
        raise ValueError(f"Duplicate feature or sample identifiers: {path}")
    if np.isinf(matrix.to_numpy(dtype=float)).any():
        raise ValueError(f"Non-finite observed values: {path}")
    return matrix


def validate_corrected_matrix(matrix: pd.DataFrame, expected: pd.DataFrame) -> pd.DataFrame:
    """Require unchanged coverage and missingness; return the expected identity order."""
    if set(matrix.index) != set(expected.index) or set(matrix.columns) != set(expected.columns):
        raise ValueError("Correction changed sample or feature coverage")
    matrix = matrix.loc[expected.index, expected.columns]
    if not matrix.isna().equals(expected.isna()):
        raise ValueError("Correction changed the missing-value mask")
    return matrix


def validate_experiment_dir(repo_root: Path, path: Path) -> Path:
    """Keep isolated experiment outputs outside canonical source and data directories."""
    root = path.expanduser().resolve()
    repo_root = repo_root.resolve()
    canonical = ("evaluation_data", "evaluation", "evaluation_utils", "batchcorrection",
                 "evaluation_classification_after_correction",
                 "evaluation_clusterization_after_correction", ".git")
    if root == repo_root or root in repo_root.parents:
        raise ValueError("Use a separate experiment output directory")
    if any(root == repo_root / name or repo_root / name in root.parents for name in canonical):
        raise ValueError("Output would modify canonical artifacts")
    return root


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


def pooled_inertia(data: np.ndarray, labels: np.ndarray) -> float:
    """Return the within-cluster sum of squares of samples x features ``data``."""
    return sum(
        float(np.square(data[labels == value] - data[labels == value].mean(axis=0)).sum())
        for value in pd.unique(labels)
    )


def run_central_kmeans(
    feature_by_sample: pd.DataFrame,
    k_values: Sequence[int],
    seed: int,
    n_init: int = 1,
    *,
    scaling: str = "zscore",
) -> Dict[int, dict]:
    """Run k-means for each k; return {k: labels indexed by sample name, inertia, iterations}.

    ``zscore`` uses population standard deviations (ddof=0); ``centered`` only
    subtracts feature means. The app uses sample standard deviations (ddof=1),
    giving a common scale factor on complete-case inputs and different inertia units.

    Parameters
    ----------
    n_init
        Number of k-means random initializations.
    """
    array = feature_by_sample.to_numpy(dtype=float).T
    data = StandardScaler(with_std=scaling == "zscore").fit_transform(array)
    samples = list(feature_by_sample.columns)
    outputs = {}
    for k in sorted(set(k_values)):
        km = KMeans(n_clusters=k, random_state=seed, n_init=n_init)
        labels = km.fit_predict(data)
        labels = pd.Series(labels, index=samples)
        outputs[k] = {"labels": labels, "inertia": float(km.inertia_), "n_iter": int(km.n_iter_)}
    return outputs


# ---------------------------------------------------------------------------
# FeatureCloud k-means input generation
# ---------------------------------------------------------------------------

def write_feature_matrix(df: pd.DataFrame, path: Path) -> None:
    """Write a feature-by-sample matrix as an unquoted TSV with a ``rowname`` index column."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out.insert(0, "rowname", out.index)
    out.to_csv(path, sep="\t", index=False)


def build_quartet_matrices(
    quartet_root: Path,
    correction_arm: str = "label_aware",
    corrected_source: str = "auto",
) -> Dict[str, Path]:
    """Stack matched Quartet modalities for the existing downstream evaluators.

    Explicit sources require only that correction backend. ``auto`` builds each
    correction backend available for all modalities.
    Current outputs are reused until a per-modality input changes.
    """
    if correction_arm not in ("label_aware", "no_label"):
        raise ValueError("correction_arm must be 'label_aware' or 'no_label'")
    if corrected_source not in ("central", "federated", "auto"):
        raise ValueError("corrected_source must be 'central', 'federated', or 'auto'")
    quartet_root = Path(quartet_root)
    before_dir = quartet_root / "before"
    after_dir = quartet_root / "after" / correction_arm
    modalities = QUARTET_LAYERS
    source_files = {"central": "intensities_log_Rcorrected_UNION.tsv",
                    "federated": "FedApp_corrected_data_smpc.tsv"}
    sources = [corrected_source] if corrected_source != "auto" else [
        source for source, filename in source_files.items()
        if all((after_dir / modality / filename).exists() for modality in modalities)
    ]
    if not sources:
        raise FileNotFoundError(f"No complete Quartet correction backend in {after_dir}")
    outputs = {
        "before": before_dir / "all_modalities_before_kmeans_matrix.tsv",
        "metadata": quartet_root / "all_modalities_metadata.tsv",
    }
    filenames = {"central": "all_modalities_corrected_kmeans_matrix.tsv",
                 "federated": "all_modalities_fedapp_kmeans_matrix.tsv"}
    if corrected_source == "auto":
        for source, filename in filenames.items():
            if source not in sources and (after_dir / filename).exists():
                raise FileNotFoundError(
                    f"Cannot refresh Quartet {source} joint matrix: modality inputs are "
                    f"incomplete in {after_dir}; restore them or select an available corrected_source"
                )
    outputs.update({source: after_dir / filenames[source] for source in sources})
    inputs = [before_dir / modality / filename for modality in modalities
              for filename in ("central_intensities_log_UNION.tsv", "metadata.tsv")]
    inputs += [after_dir / modality / source_files[source]
               for modality in modalities for source in sources]
    newest_input = max(path.stat().st_mtime_ns for path in [*inputs, Path(__file__)])
    if all(path.exists() and path.stat().st_mtime_ns >= newest_input
           for path in outputs.values()):
        return outputs

    common_columns = ["pseudo_sample", "client", "condition", "rep"]
    modality_columns = ["file", "batch", "lab", "platform", "protocol", "date"]
    blocks = {source: [] for source in ("before", *sources)}
    combined_metadata = None
    for modality in modalities:
        metadata_path = before_dir / modality / "metadata.tsv"
        metadata = pd.read_csv(metadata_path, sep="\t")
        required = common_columns + ["file"]
        missing = set(required).difference(metadata.columns)
        if missing or metadata[required].isna().any().any():
            raise ValueError(f"Missing Quartet sample metadata in {metadata_path}: {sorted(missing)}")
        metadata["rep"] = pd.to_numeric(metadata["rep"], errors="raise").astype(int)
        for column in ("file", "pseudo_sample", "client", "condition"):
            metadata[column] = metadata[column].map(normalize_token)
        if not metadata.pseudo_sample.is_unique or not metadata.file.is_unique:
            raise ValueError(f"Duplicate Quartet sample identifiers in {metadata_path}")
        metadata = metadata.sort_values("pseudo_sample").set_index("pseudo_sample", drop=False)
        shared = metadata[common_columns]
        if combined_metadata is None:
            combined_metadata = shared.copy()
        elif not combined_metadata[common_columns].equals(shared):
            raise ValueError(f"Quartet pseudo-samples or labels differ across modalities: {modality}")
        combined_metadata = combined_metadata.join(
            metadata[[column for column in modality_columns if column in metadata]]
            .add_prefix(f"{modality}_")
        )
        before = read_correction_matrix(
            before_dir / modality / "central_intensities_log_UNION.tsv"
        )
        matrices = {"before": before}
        for source in sources:
            matrices[source] = validate_corrected_matrix(
                read_correction_matrix(after_dir / modality / source_files[source]), before
            )
        for source, matrix in matrices.items():
            block = matrix.loc[:, metadata.file].copy()
            block.columns = metadata.pseudo_sample
            block.index = f"{modality}__" + block.index
            blocks[source].append(block)
    for source, parts in blocks.items():
        write_feature_matrix(pd.concat(parts, axis=0), outputs[source])
    combined_metadata.to_csv(outputs["metadata"], sep="\t", index=False)
    return outputs


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
    n_init_local: int = 1,
    n_init_global: int = 1,
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


def load_dataset_metadata(cfg: DatasetConfig):
    """Read site labels or the joint Quartet sample mapping."""
    base_dataset = (cfg.extra or {}).get("base_dataset", cfg.name)
    if base_dataset != "quartet_multiomics":
        clients = discover_clients(cfg)
        return load_metadata(cfg, clients), clients

    mapping = cfg.before_matrix.parent.parent / "all_modalities_metadata.tsv"
    if not mapping.exists():
        raise FileNotFoundError(
            f"Missing Quartet joint metadata: {mapping}. "
            "Canonical evaluators prepare this automatically; custom inputs must supply the joint mapping."
        )
    metadata = pd.read_csv(mapping, sep="\t", dtype=str)[
        ["pseudo_sample", "condition", "client"]
    ].rename(columns={"pseudo_sample": "file", "client": "lab"})
    if metadata.isna().any().any():
        raise ValueError(f"Missing Quartet evaluation labels: {mapping}")
    metadata = metadata.apply(lambda column: column.map(normalize_token))
    if not metadata.file.is_unique:
        raise ValueError(f"Duplicate Quartet pseudo-samples: {mapping}")
    names = sorted(metadata.lab.unique())
    metadata = pd.concat(
        [metadata.loc[metadata.lab.eq(name)] for name in names], ignore_index=True
    )
    return metadata, [cfg.sites_root / name for name in names]



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
    if merged.filter(like="Fed_").isna().any().any():
        raise ValueError(f"FeatureCloud clustering is missing samples in {run_output}")
    return merged.reset_index()


# ---------------------------------------------------------------------------
# Assignment tables
# ---------------------------------------------------------------------------

ASSIGNMENT_METADATA = ["file", "condition", "lab"]


def assignment_path(root: Path, dataset: str) -> Path:
    """Return a dataset's table of sample metadata and ``<method>_seed_NNN_kK`` labels."""
    return Path(root) / dataset / "assignments.tsv"


def arm_root(base: Path, arm: str) -> Path:
    """Return the output root for a correction arm: ``base``, or ``base/no_label``."""
    return Path(base) / "no_label" if arm == "no_label" else Path(base)


def read_metadata(root: Path, dataset: str) -> pd.DataFrame:
    """Read the sample metadata stored once in a dataset's assignment table."""
    return pd.read_csv(assignment_path(root, dataset), sep="\t", dtype=str,
                       keep_default_na=False, usecols=ASSIGNMENT_METADATA)


def _assignment_table(path: Path, metadata: pd.DataFrame) -> pd.DataFrame:
    """Read a table, or start one, after checking it holds the same ordered metadata."""
    expected = metadata[ASSIGNMENT_METADATA].astype(str).reset_index(drop=True)
    table = (pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
             if path.exists() else expected)
    if not table[ASSIGNMENT_METADATA].equals(expected):
        raise ValueError(f"{path} holds other samples or metadata; use a separate OUTPUT_ROOT")
    return table


def write_assignments(
    root: Path, dataset: str, metadata: pd.DataFrame, method: str, seed: int,
    labels_by_k: Dict[int, pd.Series],
) -> Path:
    """Add or replace one method/seed's labels, given as Series indexed by sample."""
    path = assignment_path(root, dataset)
    table = _assignment_table(path, metadata)
    for k, labels in labels_by_k.items():
        table[f"{method}_seed_{seed:03d}_k{k}"] = labels.loc[table.file].to_numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, sep="\t", index=False)
    return path


def read_assignments(
    root: Path, dataset: str, method: str, seed: int, k_values: Sequence[int],
    metadata: pd.DataFrame,
) -> Dict[int, pd.Series]:
    """Return {k: integer labels indexed by sample} for one method and seed."""
    table = _assignment_table(assignment_path(root, dataset), metadata).set_index("file")
    return {k: table[f"{method}_seed_{seed:03d}_k{k}"].astype(int) for k in k_values}


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
