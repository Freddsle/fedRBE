"""Central sensitivity analyses and descriptive tables for real-data clustering.

Run from the repository root; use ``--help`` on a subcommand for its inputs.
Offline centroid reconstruction describes an optimizer replica, not a new
FeatureCloud run. Sampling intervals condition on the existing correction.
"""

from __future__ import annotations

import argparse
import hashlib
from itertools import islice, permutations
import json
import math
from pathlib import Path
from statistics import NormalDist
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation_utils.feature_panel import (
    PANEL_DIR, compute_feature_panel, correction_batches, load_panel_matrices, read_panel,
)
from evaluation_utils.metrics import clustering_scores
from evaluation_utils.real_datasets_utils import (
    arm_root, dataset_configs, pooled_inertia,
    read_assignments as read_saved_assignments, read_metadata, run_central_kmeans,
    scale_like_federated,
)

SEEDS = list(range(1, 92, 10))


def digest(values):
    return hashlib.sha256(("\n".join(map(str, values)) + "\n").encode()).hexdigest()


def partition_id(labels):
    # First-occurrence coding makes the identity invariant to label permutations.
    return digest(pd.factorize(np.asarray(labels), sort=False)[0])


def fit(matrix, truth, seed, n_init=10, scaling="zscore"):
    k = len(pd.unique(truth))
    result = run_central_kmeans(matrix, [k], seed, n_init=n_init, scaling=scaling)[k]
    labels = result["labels"].reindex(matrix.columns).to_numpy()
    row = {**clustering_scores(truth, labels), "inertia": result["inertia"],
           "n_iter": result["n_iter"], "partition": partition_id(labels),
           "cluster_sizes": json.dumps(sorted(pd.Series(labels).value_counts().tolist())),
           "n_features": len(matrix), "n_samples": matrix.shape[1]}
    return row, labels


def subset(matrix, groups, size, ranking="pooled", ranking_matrix=None):
    # Saved panels use layer/rank order; use that same order for the fitted matrix.
    if size == "all":
        return matrix
    source = matrix if ranking_matrix is None else ranking_matrix
    if size == "10%":
        tables = []
        for layer, labels in groups.items():
            block = source if layer == "all" else source.loc[
                source.index.str.startswith(layer + "__")]
            tables.append(compute_feature_panel(
                block, {layer: labels}, max(1, int(len(block) * .1)), ranking))
        features = pd.concat(tables).feature
    else:
        features = compute_feature_panel(source, groups, int(size), ranking).feature
    return matrix.loc[features]


def spaces(name, matrices, selected):
    for space in selected:
        if space == "all":
            yield space, matrices
        else:
            panel = PANEL_DIR / f"panel_top5000_{name}.tsv"
            yield space, {key: read_panel(panel, value) for key, value in matrices.items()}


def datasets(args):
    for arm in args.arms:
        configs = dataset_configs(REPO_ROOT, args.config, correction_arm=arm)
        for name in args.datasets or list(configs):
            cfg = configs[name]
            metadata, clients, matrices = load_panel_matrices(cfg, canonical_before=True)
            yield name, arm, cfg, metadata, clients, matrices, correction_batches(cfg, metadata)


def save(args, name, rows):
    table = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    table.to_csv(args.output_dir / f"{name}.tsv", sep="\t", index=False)
    return table


def sensitivity(args):
    """E1: complete central feature-count/ranking/scaling grid and paired changes."""
    rows = []
    for name, arm, _, meta, _, matrices, groups in datasets(args):
        truth = meta.condition.to_numpy()
        for size in args.sizes:
            for ranking in args.rankings:
                for scaling in args.scalings:
                    for state in ("before", "central_corr"):
                        source = matrices[state] if ranking == "total" else matrices["before"]
                        matrix = subset(matrices[state], groups, size, ranking, source)
                        for seed in args.seeds:
                            row, _ = fit(matrix, truth, seed, args.n_init, scaling)
                            rows.append({"dataset": name, "arm": arm, "size": size,
                                         "ranking": ranking, "scaling": scaling, "state": state,
                                         "selection_source": state if ranking == "total" else "before",
                                         "seed": seed, "n_init": args.n_init, **row})
    table = save(args, "sensitivity", rows)
    keys = ["dataset", "arm", "size", "ranking", "scaling", "seed"]
    wide = table.pivot(index=keys, columns="state", values=["ARI", "AMI"])
    paired = wide.xs("central_corr", level="state", axis=1) - wide.xs(
        "before", level="state", axis=1)
    summary = paired.groupby(keys[:-1], sort=False).median().add_prefix("median_delta_").reset_index()
    summary["positive_ARI_cell"] = summary.median_delta_ARI.gt(0)
    save(args, "sensitivity_summary", summary)
    save(args, "sensitivity_positive_counts", summary.groupby(["dataset", "arm"], sort=False)
         .positive_ARI_cell.agg(positive_cells="sum", total_cells="size").reset_index())


def starts(args):
    """E1b: individual local optima, restart counts and the pooled best solution."""
    rows, starts_rows = [], []
    for name, arm, _, meta, _, matrices, groups in datasets(args):
        truth = meta.condition.to_numpy()
        for space, selected in spaces(name, matrices, args.spaces):
            states = args.states or (["before"] if name == "quartet_multiomics" else
                                     ["before", "central_corr"])
            for state in states:
                matrix = selected[state]
                for seed in args.seeds:
                    rng = np.random.RandomState(seed)
                    stream = []
                    for index in range(max(args.n_inits)):
                        row, labels = fit(matrix, truth, rng, 1)
                        row = {"dataset": name, "arm": arm, "space": space, "state": state,
                               "seed": seed, "start": index + 1, **row}
                        starts_rows.append(row)
                        stream.append((row, labels))
                    for n_init in args.n_inits:
                        row, labels = fit(matrix, truth, seed, n_init)
                        best_inertia = min(value[0]["inertia"] for value in stream[:n_init])
                        matches = any(np.isclose(row["inertia"], value[0]["inertia"], rtol=1e-10)
                                      and partition_id(labels) == value[0]["partition"]
                                      for value in stream[:n_init])
                        if not matches or not np.isclose(row["inertia"], best_inertia, rtol=1e-10):
                            raise ValueError("Individual starts do not reproduce the library multi-start fit")
                        rows.append({"dataset": name, "arm": arm, "space": space, "state": state,
                                     "seed": seed, "n_init": n_init, **row})
    fits = save(args, "start_count_fits", rows)
    individual = save(args, "individual_starts", starts_rows)
    keys = ["dataset", "arm", "space", "state", "partition"]
    save(args, "distinct_start_partitions", individual.groupby(keys, sort=False).agg(
        frequency=("start", "size"), inertia=("inertia", "min"), ARI=("ARI", "first"),
        AMI=("AMI", "first")).reset_index())
    best = individual.sort_values("inertia").drop_duplicates(keys[:-1]).set_index(keys[:-1])[
        ["inertia", "ARI", "AMI", "partition"]].add_prefix("pooled_best_")
    comparison = fits.join(best, on=keys[:-1])
    comparison["excess_inertia_over_pooled_best"] = comparison.inertia - comparison.pooled_best_inertia
    save(args, "start_count_best_comparison", comparison)


def subsample(args):
    """E1c: paired, stratified subsampling with fresh scaling and clustering."""
    rows = []
    for name, arm, _, meta, _, matrices, groups in datasets(args):
        rng = np.random.default_rng(args.resample_seed)
        strata = list(meta.groupby(["lab", "condition"], sort=False).indices.values())
        selected_spaces = list(spaces(name, matrices, args.spaces))
        for repeat in range(args.repeats):
            indices = np.sort(np.concatenate([
                rng.choice(values, max(1, int(len(values) * args.fraction)), replace=False)
                for values in strata]))
            sampled = meta.iloc[indices]
            for space, selected in selected_spaces:
                for seed in args.seeds:
                    scores = {}
                    for state in ("before", "central_corr"):
                        scores[state], _ = fit(selected[state].iloc[:, indices], sampled.condition,
                                               seed, args.n_init)
                    rows.append({"dataset": name, "arm": arm, "space": space,
                                 "repeat": repeat, "seed": seed, "n_samples": len(indices),
                                 "delta_ARI": scores["central_corr"]["ARI"] - scores["before"]["ARI"],
                                 "delta_AMI": scores["central_corr"]["AMI"] - scores["before"]["AMI"]})
    table = save(args, "subsample_paired_deltas", rows)
    median = table.groupby(["dataset", "arm", "space", "repeat"], sort=False)[
        ["delta_ARI", "delta_AMI"]].median().reset_index()
    save(args, "subsample_seed_medians", median)
    intervals = []
    for key, block in median.groupby(["dataset", "arm", "space"], sort=False):
        for metric in ("delta_ARI", "delta_AMI"):
            values = block[metric].to_numpy()
            intervals.append(dict(zip(("dataset", "arm", "space"), key)) | {
                "metric": metric, "median": np.median(values), "lower": np.quantile(values, args.alpha / 2),
                "upper": np.quantile(values, 1 - args.alpha / 2), "repeats": len(values),
                "fraction": args.fraction, "scope": "reclustering; correction and feature panel fixed"})
    save(args, "subsample_intervals", intervals)


def arrival_order(args):
    """Reconstruct local centroids and permute coordinator inputs offline."""
    rows, summaries = [], []
    for name, arm, _, meta, clients, matrices, groups in datasets(args):
        truth = meta.condition.to_numpy()
        k = len(pd.unique(truth))
        parts = [np.flatnonzero(meta.lab.eq(client.name)) for client in clients]
        for space, selected in spaces(name, matrices, args.spaces):
            for state in args.states:
                data = scale_like_federated(selected[state])
                for seed in args.seeds:
                    centroids = [KMeans(k, n_init=args.n_init_local,
                                        random_state=np.random.RandomState(seed)).fit(data[part]).cluster_centers_
                                 for part in parts]
                    solutions = {}
                    orders = permutations(range(len(clients)))
                    if args.max_orders is not None:
                        orders = islice(orders, args.max_orders)
                    for index, order in enumerate(orders):
                        coordinator = KMeans(k, n_init=args.n_init_global,
                                             random_state=np.random.RandomState(seed)).fit(
                            np.concatenate([centroids[i] for i in order]))
                        labels = np.empty(len(meta), dtype=int)
                        for part in parts:
                            labels[part] = KMeans(k, n_init=1, init=coordinator.cluster_centers_,
                                                  random_state=seed, max_iter=1).fit(data[part]).labels_
                        identity = partition_id(labels)
                        if identity not in solutions:
                            solutions[identity] = {**clustering_scores(truth, labels),
                                                   "pooled_inertia": pooled_inertia(data, labels)}
                        rows.append({"dataset": name, "arm": arm, "space": space,
                                     "state": state, "seed": seed, "order_index": index,
                                     "client_order": json.dumps([clients[i].name for i in order]),
                                     "n_init_local": args.n_init_local, "n_init_global": args.n_init_global,
                                     "partition": identity, "coordinator_inertia": coordinator.inertia_,
                                     **solutions[identity]})
                    summaries.append({"dataset": name, "arm": arm, "space": space,
                                      "state": state, "seed": seed, "distinct_partitions": len(solutions),
                                      "orders_checked": index + 1,
                                      "exhaustive": index + 1 == math.factorial(len(clients)),
                                      "ARI_min": min(row["ARI"] for row in solutions.values()),
                                      "ARI_max": max(row["ARI"] for row in solutions.values()),
                                      "scope": "local sklearn optimizer replica; message order unobserved"})
    save(args, "arrival_orders", rows)
    save(args, "arrival_order_summary", summaries)


def assignment_datasets(args):
    """Load retained labels/metadata without requiring any correction matrices."""
    for arm in args.arms:
        root = arm_root(args.run_root, arm)
        names = args.datasets or list(dataset_configs(
            REPO_ROOT, args.config, correction_arm=arm))
        for name in names:
            yield name, arm, read_metadata(root, name)


def read_assignments(args, name, arm, metadata, seed):
    root = arm_root(args.run_root, arm)
    k = metadata.condition.nunique()
    labels = {}
    for method in ("BC_Cntrl", "AC_Cntrl", "BC_Fed", "AC_Fed"):
        labels[method] = read_saved_assignments(
            root, name, method, seed, [k], metadata)[k].to_numpy()
    return labels


def contingency(args):
    """Tabulate final assignments by study/class and class-pair co-assignment."""
    rows, coassignments = [], []
    for name, arm, meta in assignment_datasets(args):
        for seed in args.seeds:
            for method, labels in read_assignments(args, name, arm, meta, seed).items():
                frame = meta.assign(cluster=pd.factorize(labels, sort=False)[0])
                base = {"dataset": name, "arm": arm, "seed": seed, "method": method}
                for (client, condition, cluster), count in frame.groupby(
                        ["lab", "condition", "cluster"], sort=False).size().items():
                    rows.append({**base, "client": client, "condition": condition,
                                 "cluster": cluster, "count": count})
                classes = sorted(frame.condition.unique())
                for i, left in enumerate(classes):
                    a = frame.loc[frame.condition.eq(left), "cluster"].value_counts()
                    for right in classes[i:]:
                        b = frame.loc[frame.condition.eq(right), "cluster"].value_counts()
                        numerator = ((a * (a - 1)).sum() / 2 if left == right else
                                     a.mul(b, fill_value=0).sum())
                        denominator = (a.sum() * (a.sum() - 1) / 2 if left == right else
                                       a.sum() * b.sum())
                        coassignments.append({**base, "class_a": left, "class_b": right,
                                              "coassigned_pairs": numerator, "possible_pairs": denominator,
                                              "fraction": numerator / denominator if denominator else np.nan})
    save(args, "contingencies", rows)
    save(args, "class_coassignment", coassignments)


def rf_cohorts(args):
    """Summarize held-out-cohort MCC and support-weighted F1 by seed and method."""
    rows = []
    for path in args.reports:
        report = pd.read_csv(path)
        keys = ["data_name", "data_preprocessing_name", "predicted_client_name",
                "cross_validation_method", "seed", "predicted_target", "learning_type"]
        report = report.loc[report.cross_validation_method.eq("Leave-One-Cohort-Out") &
                            report.metric_name.isin(["MCC", "F1_score"])]
        table = report.pivot(index=keys, columns="metric_name", values="metric_value").reset_index()
        table.insert(0, "report", str(path))
        rows.append(table)
    combined = save(args, "rf_per_cohort_seed", pd.concat(rows, ignore_index=True))
    keys = [key for key in combined.columns if key not in ("seed", "MCC", "F1_score")]
    save(args, "rf_per_cohort_summary", combined.groupby(keys, sort=False).agg(
        MCC_median=("MCC", "median"), MCC_min=("MCC", "min"), MCC_max=("MCC", "max"),
        F1_median=("F1_score", "median"), F1_min=("F1_score", "min"),
        F1_max=("F1_score", "max"), seeds=("seed", "nunique")).reset_index())


def class_r2(args):
    """Summarize unadjusted pooled per-feature class R-squared by correction state."""
    rows = []
    for name, arm, _, meta, _, matrices, groups in datasets(args):
        for space, selected in spaces(name, matrices, args.spaces):
            for state, matrix in selected.items():
                values = matrix.to_numpy(float)
                total = np.square(values - values.mean(axis=1, keepdims=True)).sum(axis=1)
                within = np.zeros(len(matrix))
                for condition in meta.condition.unique():
                    block = values[:, meta.condition.eq(condition)]
                    within += np.square(block - block.mean(axis=1, keepdims=True)).sum(axis=1)
                r2 = np.divide(total - within, total, out=np.full(len(matrix), np.nan), where=total > 0)
                rows.append({"dataset": name, "arm": arm, "space": space, "state": state,
                             "mean_class_r2": np.nanmean(r2), "median_class_r2": np.nanmedian(r2),
                             "n_features": len(matrix), "undefined_constant_features": int((total == 0).sum()),
                             "scope": "pooled unadjusted class association; correction fixed"})
    save(args, "class_r2", rows)


def ari_leave_one_out(truth, labels):
    """ARI after deleting each observation; deletions from one contingency cell agree."""
    cells = list(zip(truth, labels))
    scores = {}
    for index, cell in enumerate(cells):
        if cell not in scores:
            scores[cell] = adjusted_rand_score(np.delete(truth, index), np.delete(labels, index))
    return np.array([scores[cell] for cell in cells])


def intervals(args):
    """Fixed-assignment intervals: jackknife-normal ARI and paired study bootstrap."""
    rows = []
    quantile = NormalDist().inv_cdf(1 - args.alpha / 2)
    for name, arm, meta in assignment_datasets(args):
        truth = meta.condition.to_numpy()
        strata = list(meta.groupby("lab", sort=False).indices.values())
        for seed in args.seeds:
            labels = read_assignments(args, name, arm, meta, seed)
            scores = {key: clustering_scores(truth, value) for key, value in labels.items()}
            jack = {key: ari_leave_one_out(truth, value) for key, value in labels.items()}
            rng = np.random.default_rng(args.resample_seed)
            samples = []
            for _ in range(args.repeats):
                indices = np.concatenate([rng.choice(part, len(part), replace=True) for part in strata])
                samples.append({key: clustering_scores(truth[indices], value[indices]) for key, value in labels.items()})
            for method in (*labels, "delta_Cntrl", "delta_Fed"):
                paired = method.startswith("delta_")
                after, before = ("AC_" + method[6:], "BC_" + method[6:]) if paired else (method, None)
                for metric in ("ARI", "AMI"):
                    estimate = scores[after][metric] - (scores[before][metric] if paired else 0)
                    bootstrap = np.array([row[after][metric] - (row[before][metric] if paired else 0)
                                          for row in samples])
                    row = {"dataset": name, "arm": arm, "seed": seed, "method": method,
                           "metric": metric, "estimate": estimate, "bootstrap_repeats": args.repeats,
                           "bootstrap_lower": np.quantile(bootstrap, args.alpha / 2),
                           "bootstrap_upper": np.quantile(bootstrap, 1 - args.alpha / 2),
                           "scope": "evaluation uncertainty; assignments and correction fixed",
                           "normal_estimator": "delete-one jackknife ARI variance; iid arrays",
                           "bootstrap_design": "paired arrays sampled within study"}
                    if metric == "ARI":
                        deleted = jack[after] - (jack[before] if paired else 0)
                        se = np.sqrt((len(meta) - 1) / len(meta) * np.square(deleted - deleted.mean()).sum())
                        limit = 2 if paired else 1
                        row.update(normal_se=se, normal_lower=max(-limit, estimate - quantile * se),
                                   normal_upper=min(limit, estimate + quantile * se))
                    else:
                        row["normal_estimator"] = "not computed for AMI"
                    rows.append(row)
    save(args, "fixed_partition_intervals", rows)


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    sub = command.add_subparsers(dest="command", required=True)
    actions = {"sensitivity": sensitivity, "starts": starts, "subsample": subsample,
               "arrival-order": arrival_order, "contingency": contingency,
               "rf-cohorts": rf_cohorts, "class-r2": class_r2, "intervals": intervals}
    for name, action in actions.items():
        item = sub.add_parser(name, help=action.__doc__ or name.replace("-", " "))
        item.set_defaults(action=action)
        item.add_argument("--output-dir", type=Path, required=True)
        if name == "rf-cohorts":
            item.add_argument("--reports", nargs="+", type=Path, required=True)
            continue
        item.add_argument("--config", type=Path)
        item.add_argument("--datasets", nargs="+")
        item.add_argument("--arms", nargs="+", choices=("label_aware", "no_label"),
                          default=["label_aware", "no_label"])
        item.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
        if name in ("starts", "subsample", "arrival-order", "class-r2"):
            item.add_argument("--spaces", nargs="+", choices=("all", "panel"), default=["all", "panel"])
        if name in ("sensitivity", "subsample"):
            item.add_argument("--n-init", type=int, default=10)
        if name in ("starts", "arrival-order"):
            item.add_argument("--states", nargs="+", choices=("before", "central_corr", "fed_corr"),
                              default=None if name == "starts" else ["before", "fed_corr"])
        if name in ("subsample", "intervals"):
            item.add_argument("--repeats", type=int, default=100 if name == "subsample" else 1000)
            item.add_argument("--alpha", type=float, default=.05)
            item.add_argument("--resample-seed", type=int, default=20261022)
        if name in ("contingency", "intervals"):
            item.add_argument("--run-root", type=Path, required=True,
                              help="Notebook output base; no_label is appended for that arm")
        if name == "sensitivity":
            item.add_argument("--sizes", nargs="+", default=["500", "1000", "2000", "3000", "5000", "10%", "all"])
            item.add_argument("--rankings", nargs="+", choices=("pooled", "equal", "total"), default=["pooled", "equal", "total"])
            item.add_argument("--scalings", nargs="+", choices=("zscore", "centered"), default=["zscore", "centered"])
        elif name == "starts":
            item.set_defaults(datasets=["ovarian_cancer", "quartet_multiomics"])
            item.add_argument("--n-inits", nargs="+", type=int, default=[1, 10, 50])
        elif name == "subsample":
            item.add_argument("--fraction", type=float, default=.8)
        elif name == "arrival-order":
            item.add_argument("--n-init-local", type=int, default=10)
            item.add_argument("--n-init-global", type=int, default=10)
            item.add_argument("--max-orders", type=int, help="Bound an explicitly partial smoke check")
    return command


def main():
    args = parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        args.action(args)


if __name__ == "__main__":
    main()
