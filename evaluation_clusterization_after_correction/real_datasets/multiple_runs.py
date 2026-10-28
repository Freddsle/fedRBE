"""Repeated-seed central and federated k-means before and after batch correction.

Run from the repository root. Each feature space and correction arm has its own
output root: ``--root`` for all complete-case features and ``<root>/panel5000``
for the shared panel, with ``no_label/`` appended for that arm. Every root holds
``<dataset>/assignments.tsv`` and ``metrics_ari_multirun.tsv``. Saved federated
assignments are reused unless ``--rerun-federated`` launches new FeatureCloud
tests, which require Docker and the ``fc_kmeans_upd`` app image.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation_utils.feature_panel import PANEL_DIR, load_panel_matrices, read_panel
from evaluation_utils.metrics import clustering_scores
from evaluation_utils.real_datasets_utils import (
    arm_root, build_quartet_matrices, dataset_configs, pooled_inertia,
    prepare_variant_inputs, read_assignments, run_central_kmeans, scale_like_federated,
    write_assignments, write_kmeans_config,
)

SEEDS = list(range(1, 92, 10))
N_INIT = 10  # central, local federated and coordinator k-means starts
# Method -> (matrix state, FeatureCloud input variant or None for central k-means).
METHODS = {
    "BC_Cntrl": ("before", None),
    "AC_Cntrl": ("central_corr", None),
    "BC_Fed": ("before", "before_multi"),
    "AC_Fed": ("fed_corr", "corrected_fed"),
}
APP_IMAGE = "fc_kmeans_upd"
CONTROLLER_HOST = "http://localhost:8000"
QUERY_INTERVAL = 5
TIMEOUT = 1800  # seconds per FeatureCloud test


def run_federated(root, name, variant, inputs, metadata, clients, k, seed):
    """Launch one FeatureCloud test on prepared inputs; return labels indexed by sample."""
    from evaluation_utils.featurecloud_kmeans_utils import run_single_federated_variant

    for client in clients:
        write_kmeans_config(inputs / client.name / "config_kmeans.yml", [k], seed=seed,
                            n_init_local=N_INIT, n_init_global=N_INIT)
    export = run_single_federated_variant(
        dataset_name=name, variant_label=variant, variant_input_dir=inputs,
        dataset_root=root / name, metadata=metadata,
        client_names=[client.name for client in clients], k_values=[k],
        app_image=APP_IMAGE, controller_host=CONTROLLER_HOST,
        query_interval=QUERY_INTERVAL, timeout=TIMEOUT, seed=seed)
    frame = pd.read_csv(export, sep="\t", dtype={"file": str})
    return frame.set_index("file")[f"Fed_{k}clusters"]


def run_dataset(cfg, root, space, seeds, rerun_federated, rows):
    """Fit or reuse every method and seed for one dataset; append metric rows per method."""
    if cfg.name == "quartet_multiomics" and cfg.correction_arm is not None:
        build_quartet_matrices(cfg.before_matrix.parent.parent, cfg.correction_arm)
    metadata, clients, matrices = load_panel_matrices(cfg, canonical_before=True)
    if space == "panel":
        panel = PANEL_DIR / f"panel_top5000_{cfg.name}.tsv"
        matrices = {state: read_panel(panel, matrix) for state, matrix in matrices.items()}
    k = int(metadata.condition.nunique())
    for method, (state, variant) in METHODS.items():
        matrix = matrices[state]
        if variant:
            # Federated inertia is recomputed with the app's sample-SD scaling.
            scaled = scale_like_federated(matrix)
        if variant and rerun_federated:
            inputs = prepare_variant_inputs(root / cfg.name, variant, matrix, metadata,
                                            clients, [k])
        for seed in seeds:
            print(f"[{cfg.name}] {method} seed={seed}", flush=True)
            if variant is None:
                fit = run_central_kmeans(matrix, [k], seed, n_init=N_INIT)[k]
                labels, inertia = fit["labels"], fit["inertia"]
            elif rerun_federated:
                labels = run_federated(root, cfg.name, variant, inputs, metadata, clients, k, seed)
            else:
                labels = read_assignments(root, cfg.name, method, seed, [k], metadata)[k]
            if variant is None or rerun_federated:
                write_assignments(root, cfg.name, metadata, method, seed, {k: labels})
            labels = labels.loc[metadata.file].to_numpy()
            if variant:
                inertia = pooled_inertia(scaled, labels)
            scores = clustering_scores(metadata.condition.to_numpy(), labels)
            rows[method].append({
                "Dataset": cfg.name, "Target": "condition", "K": k, "Method": method,
                "Seed": seed, "ARI": scores["ARI"], "AMI": scores["AMI"], "Inertia": inertia,
            })


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arms", nargs="+", choices=("label_aware", "no_label"),
                        default=["label_aware", "no_label"])
    parser.add_argument("--spaces", nargs="+", choices=("all", "panel"), default=["all", "panel"])
    parser.add_argument("--config", type=Path,
                        help="Custom dataset registry; use with --arms label_aware")
    parser.add_argument("--datasets", nargs="+", help="Dataset IDs; default: whole registry")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--rerun-federated", action="store_true",
                        help="Launch new FeatureCloud tests instead of reusing saved assignments")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent,
                        help="Output base (default: this directory)")
    args = parser.parse_args()
    for space in args.spaces:
        for arm in args.arms:
            root = arm_root(args.root / "panel5000" if space == "panel" else args.root, arm)
            configs = dataset_configs(REPO_ROOT, args.config, correction_arm=arm)
            rows = {method: [] for method in METHODS}
            for name in args.datasets or list(configs):
                run_dataset(configs[name], root, space, args.seeds, args.rerun_federated, rows)
            table = pd.DataFrame([row for method in METHODS for row in rows[method]])
            path = root / "metrics_ari_multirun.tsv"
            if path.exists():
                # Replace this run's rows; keep method, registry and seed order.
                keys = ["Dataset", "Method", "Seed"]
                saved = pd.read_csv(path, sep="\t", float_precision="round_trip")
                saved = saved[~saved.set_index(keys).index.isin(table.set_index(keys).index)]
                table = pd.concat([saved, table], ignore_index=True)
                rank = pd.DataFrame({
                    "method": table.Method.map({m: i for i, m in enumerate(METHODS)}),
                    "dataset": table.Dataset.map({d: i for i, d in enumerate(configs)}),
                    "seed": table.Seed})
                table = table.loc[rank.sort_values(["method", "dataset", "seed"]).index]
            table.to_csv(path, sep="\t", index=False)
            print(f"Saved {path}")


if __name__ == "__main__":
    main()
