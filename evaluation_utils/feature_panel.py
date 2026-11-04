"""Compute a correction-invariant, within-batch variance panel on log-scale inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import yaml

from evaluation_utils.real_datasets_utils import (
    QUARTET_LAYERS, align_matrix_to_metadata, dataset_configs, drop_rows_with_any_na,
    load_before_matrix_from_sites, load_dataset_metadata, load_feature_matrix,
    load_matrix_with_lfs_fallback, normalize_token,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PANEL_DIR = REPO_ROOT / "evaluation_clusterization_after_correction/real_datasets/panel5000"


def correction_batches(cfg, metadata):
    """Return correction-batch labels per layer, in evaluation sample order.

    Batches are clients; Quartet uses client × batch from each layer's metadata.
    """
    if cfg.name != "quartet_multiomics":
        return {"all": metadata.set_index("file")["lab"]}
    groups = {}
    for layer in QUARTET_LAYERS:
        frame = pd.read_csv(cfg.before_matrix.parent / layer / "metadata.tsv", sep="\t", dtype=str)
        frame = frame.apply(lambda col: col.map(normalize_token))
        batches = pd.Series(list(zip(frame.client, frame.batch)), index=frame.pseudo_sample)
        groups[layer] = batches.loc[metadata.file]
    return groups


def compute_feature_panel(matrix, groups, top_n=5000, ranking="pooled"):
    """Rank complete-case features per layer, with feature-name tie breaking.

    ``groups`` maps ``all`` or a ``<layer>__`` feature prefix to batch labels.
    Each correction batch supplies its sample count and centred sum of squares.
    ``pooled`` divides their summed SS by summed degrees of freedom. ``equal``
    averages batch variances; ``total`` uses total log-scale variance. The latter
    two options support sensitivity analysis, not the shared invariant panel.
    """
    if ranking not in ("pooled", "equal", "total"):
        raise ValueError("ranking must be pooled, equal, or total")
    rows = []
    for layer, batches in groups.items():
        block = matrix if layer == "all" else matrix.loc[
            matrix.index.str.startswith(layer + "__")]
        values = block.loc[:, batches.index].to_numpy(float)
        sums = np.zeros(len(block))
        variances = []
        degrees = 0
        for batch in batches.unique():
            mask = batches.map(lambda value: value == batch).to_numpy()
            local = values[:, mask]
            ss = np.square(local - local.mean(axis=1, keepdims=True)).sum(axis=1)
            sums += ss
            degrees += local.shape[1] - 1
            if local.shape[1] > 1:
                variances.append(ss / (local.shape[1] - 1))
        pooled = sums / degrees
        score = (pooled if ranking == "pooled" else
                 np.mean(variances, axis=0) if ranking == "equal" else
                 values.var(axis=1, ddof=1))
        table = pd.DataFrame({"feature": block.index, "layer": layer,
                              "pooled_variance": pooled, "score": score})
        table = table.sort_values(["score", "feature"], ascending=[False, True]).head(top_n)
        table.insert(2, "rank", range(1, len(table) + 1))
        rows.append(table)
    return pd.concat(rows, ignore_index=True)


def load_panel_matrices(cfg, *, canonical_before=False):
    """Read complete-case matrices in evaluation sample order without regenerating inputs.

    Quartet requires the joint metadata/matrices prepared by the ordinary data
    preparation route. Its uncorrected matrix is reconstructed from each layer's
    per-client inputs and mapped to the same pseudo-samples, entirely in memory.
    ``canonical_before=True`` instead reads the exact serialized before matrix
    used by `multiple_runs.py`, for numerical reproduction in downstream analyses.
    """
    metadata, clients = load_dataset_metadata(cfg)
    if canonical_before:
        before = load_matrix_with_lfs_fallback(cfg.before_matrix, f"{cfg.name} before", cfg)
    elif cfg.name == "quartet_multiomics":
        blocks = []
        for layer in QUARTET_LAYERS:
            folder = cfg.before_matrix.parent / layer
            mapping = pd.read_csv(folder / "metadata.tsv", sep="\t", dtype=str)
            mapping = mapping.set_index("pseudo_sample").loc[metadata.file]
            parts = [load_feature_matrix(folder / client.name / "intensities_log_UNION.tsv")
                     for client in clients]
            block = pd.concat(parts, axis=1).loc[:, mapping.file].copy()
            block.columns = metadata.file
            block.index = layer + "__" + block.index
            blocks.append(block)
        before = pd.concat(blocks)
    else:
        before = load_before_matrix_from_sites(cfg, clients)
    matrices = {"before": before}
    for source, path in (("central_corr", cfg.corrected_central),
                         ("fed_corr", cfg.corrected_federated)):
        matrices[source] = load_matrix_with_lfs_fallback(path, f"{cfg.name} {source}")
    matrices = {state: drop_rows_with_any_na(
        align_matrix_to_metadata(matrix, metadata, state), state)
        for state, matrix in matrices.items()}
    return metadata, clients, matrices


def read_panel(path, matrix):
    """Subset a feature-by-sample matrix to a saved panel, in panel order."""
    return matrix.loc[pd.read_csv(path, sep="\t", dtype={"feature": str}).feature]


def export_rf_panel(cfg, features):
    """Export panel-only copies of a dataset's canonical RF states.

    Descriptors and design files are copied. Data files keep their cell text,
    sample order and target columns, with features in panel order. Returns the
    rf_datasets.yaml entries for the unchanged RF launchers.
    """
    quartet = cfg.name == "quartet_multiomics"
    dataset_root = cfg.before_matrix.parent.parent if quartet else cfg.sites_root.parent
    source_root = dataset_root / "merged_omics" if quartet else dataset_root
    destination = dataset_root / "panel5000"
    for state in ("before", "after/label_aware", "after/no_label"):
        info = json.loads((source_root / state / "datainfo.json").read_text())
        spec = info["datafile"]
        by_feature = spec["rotation"] == "features x samples"
        for cohort in info["cohorts"]:
            source = source_root / state / cohort["folder"]
            target = destination / state / cohort["folder"]
            target.mkdir(parents=True, exist_ok=True)
            frame = pd.read_csv(
                source / spec["filename"], sep=spec["separator"], dtype=str,
                keep_default_na=False,
                index_col=(spec.get("featurename_column") or 0) if by_feature
                else spec["samplename_column"])
            frame = (frame.loc[features] if by_feature
                     else frame[features + info["prediction_targets"]])
            frame.to_csv(target / spec["filename"], sep=spec["separator"])
            if cohort.get("designfile"):
                # Corrected states refer back to the before cohorts' design files.
                design = cohort["designfile"]["filename"]
                (target / design).resolve().parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / design, (target / design).resolve())
        shutil.copyfile(source_root / state / "datainfo.json",
                        destination / state / "datainfo.json")
    before = destination / "before/datainfo.json"
    return {f"{cfg.name}_{arm}": {"classification": {
        "before": str(before.relative_to(REPO_ROOT)),
        "corrected": str((destination / "after" / arm / "datainfo.json").relative_to(REPO_ROOT)),
        "preprocessing_name": "corrected" if arm == "label_aware" else "no_label",
    }} for arm in ("label_aware", "no_label")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-rf", action="store_true",
                        help="Also export panel-only RF cohorts and rf_datasets.yaml")
    args = parser.parse_args()
    no_label = dataset_configs(REPO_ROOT, correction_arm="no_label")
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    registry = {}
    for name, cfg in dataset_configs(REPO_ROOT).items():
        metadata, _, matrices = load_panel_matrices(cfg)
        groups = correction_batches(cfg, metadata)
        panel = compute_feature_panel(matrices["before"], groups)
        # Location-only correction keeps within-batch variances, so all six states agree.
        states = {"label_aware": matrices, "no_label": load_panel_matrices(no_label[name])[2]}
        for arm, arm_matrices in states.items():
            for state, matrix in arm_matrices.items():
                if compute_feature_panel(matrix, groups).feature.tolist() != \
                        panel.feature.tolist():
                    raise ValueError(f"Panel differs for {name}/{arm}/{state}")
        panel.drop(columns="score").to_csv(
            PANEL_DIR / f"panel_top5000_{name}.tsv", sep="\t", index=False)
        print(f"{name}: {panel.groupby('layer', sort=False).size().to_dict()}; all states match")
        if args.prepare_rf:
            registry.update(export_rf_panel(cfg, panel.feature.tolist()))
    if args.prepare_rf:
        (PANEL_DIR / "rf_datasets.yaml").write_text(yaml.safe_dump(registry, sort_keys=False))


if __name__ == "__main__":
    main()
