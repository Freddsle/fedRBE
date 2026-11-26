#!/usr/bin/env python3
"""Matched downstream evaluation of confounding and missingness simulations."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import io
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import hashlib
import json
from math import ceil
import os
from pathlib import Path
import pickle
import platform
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import t
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from evaluation_data.simulated import training_correction
from evaluation_utils.metrics import classification_scores, clustering_scores
from evaluation_classification_after_correction.fc_fed_forest_simple_app import (
    fedForestSimple,
)
from evaluation_classification_after_correction.fc_fed_forest_simple_app.fc_fedlearnsim_helper.run_app_simulation import (
    run_simulation_native,
)
from evaluation_utils.real_datasets_utils import (
    pooled_inertia,
    run_central_kmeans,
    scale_like_federated,
)

APP_ROOT = (REPO /
            "evaluation_clusterization_after_correction" / "federated_kmeans_upd")
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))
from kmeans.Clustering import Clustering


SCENARIOS = ("confounded_p50", "confounded_p30", "confounded_p15", "confounded_p05")
STATES = ("truth", "uncorrected", "no_label", "label_aware")
# Missingness learning inputs by missing-value route. Keep: matched features with missing
# values; drop: their complete features; impute: per-sample minimum, then no-label
# correction. Learning inputs use no-label correction only.
ROUTES = {"keep": ("truth", "uncorrected", "no_label"),
          "drop": ("drop_truth", "drop_uncorrected", "drop_no_label"),
          "impute": ("impute_uncorrected", "impute_no_label")}
ROUTE_STATES = tuple(state for states in ROUTES.values() for state in states)
# d-ComBat (Chen et al. 2022) corrects the raw drop and impute matrices; exports with
# --dcombat only.
DCOMBAT_STATES = {"drop_dcombat": "drop", "impute_dcombat": "impute"}
STATE_ROUTE = {state: route for route, states in ROUTES.items() for state in states}
STATE_ROUTE.update(DCOMBAT_STATES)
# Corrected states whose central learners read <state>_central.tsv.gz (central limma or ComBat).
CENTRAL_STATES = ("no_label", "label_aware", "impute_no_label", *DCOMBAT_STATES)
STATE_PAIRS = [
    ("uncorrected", "truth"), ("no_label", "truth"), ("label_aware", "truth"),
    ("no_label", "uncorrected"), ("label_aware", "uncorrected"), ("label_aware", "no_label"),
    ("drop_uncorrected", "drop_truth"), ("drop_no_label", "drop_truth"),
    ("drop_no_label", "drop_uncorrected"), ("impute_no_label", "impute_uncorrected"),
    ("drop_truth", "truth"), ("drop_uncorrected", "uncorrected"), ("drop_no_label", "no_label"),
    ("impute_uncorrected", "uncorrected"), ("impute_no_label", "no_label"),
    ("drop_dcombat", "drop_uncorrected"), ("drop_dcombat", "drop_no_label"),
    ("impute_dcombat", "impute_uncorrected"), ("impute_dcombat", "impute_no_label"),
    ("drop_dcombat", "no_label"), ("impute_dcombat", "no_label"),
]
GROUP = ["analysis", "protocol", "panel", "target", "scenario", "task", "learner",
         "state", "lab", "metric"]
RUN = ["run", "data_seed"]
COUNTS = ["n_samples", "n_a", "n_b"]
KEY = GROUP + RUN + ["learner_seed"]
PAIR_GROUP = GROUP + ["comparison", "reference_state", "reference_learner"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".pending_",
                                     delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
    temporary.replace(path)


def source_hashes():
    paths = [
        "evaluation_data/simulated/robustness_simulation.R",
        "evaluation_data/simulated/missingness_overlap.R",
        "evaluation_data/simulated/export_downstream_inputs.R",
        "evaluation_data/simulated/downstream_simulation.py",
        "evaluation_data/simulated/training_correction.py",
        "evaluation_data/simulated/dcombat_training.R",
        "evaluation_utils/central_correction.R",
        "evaluation_utils/simulation_func.R", "evaluation_utils/filtering.R",
        "evaluation_utils/evaluation_funcs.R", "evaluation_utils/real_datasets_utils.py",
        "evaluation_utils/metrics.py", "evaluation_utils/fedRBE_simulation_scrip_simdata.py",
        "evaluation_clusterization_after_correction/federated_kmeans_upd/states.py",
        "pixi.lock",
    ]
    for directory in ("batchcorrection", "evaluation_clusterization_after_correction/"
                      "federated_kmeans_upd/kmeans", "evaluation_classification_after_correction/"
                      "fc_fed_forest_simple_app",
                      "evaluation_classification_after_correction/helper"):
        paths.extend(str(p.relative_to(REPO)) for p in (REPO / directory).rglob("*.py"))
    return {p: sha256(REPO / p) for p in sorted(set(paths))}


def positive_integers(value):
    try:
        values = [int(v) for v in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use comma-separated positive integers") from exc
    if not values or any(v <= 0 or v > 2**31 - 101 for v in values):
        raise argparse.ArgumentTypeError("Seeds/runs must be positive and within signed int32")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("Repeated seeds/runs would duplicate evidence")
    return values


def load_inputs(directory, analysis="confounding", states=None):
    """Load and check exported inputs as {state: {learner: matrix}}: a corrected state has
    the fedRBE or d-ComBat output for the federated learner and the central limma or ComBat
    output for the central one. Return the requested states (default: all but d-ComBat)."""
    if analysis not in ("confounding", "missingness_overlap"):
        raise ValueError("Unknown analysis")
    metadata = pd.read_csv(directory / "metadata.tsv", sep="\t").set_index("file")
    features = pd.read_csv(directory / "features.tsv", sep="\t")
    files = STATES if analysis == "confounding" else ROUTES["keep"] + ROUTES["impute"]
    files += tuple(state for state in DCOMBAT_STATES if state in (states or ()))
    read = lambda name: pd.read_csv(directory / f"{name}.tsv.gz", sep="\t", index_col=0)
    matrices = {s: read(s) for s in files}
    central = {s: read(f"{s}_central") for s in files if s in CENTRAL_STATES}
    expected = matrices["truth"]
    if not metadata.index.is_unique or not expected.index.is_unique:
        raise ValueError("Duplicate sample or feature identifiers")
    if not expected.columns.equals(metadata.index):
        raise ValueError("Matrix sample order must match metadata")
    if (expected.empty or expected.shape[1] != 600 or not expected.columns.is_unique or
            features.feature.tolist() != expected.index.tolist()):
        raise ValueError("Expected a nonempty matched simulation feature/sample panel")
    n_labs = 3 if analysis == "confounding" else 5
    if (set(metadata.condition) != {"A", "B"} or metadata.condition.isna().any() or
            set(metadata.lab) != {f"batch{i}" for i in range(1, n_labs + 1)}):
        raise ValueError("Unexpected labs or condition labels; require condition A/B")
    if analysis == "confounding" and expected.shape[0] != 6000:
        raise ValueError("Expected the complete, matched 6000-feature simulation panel")
    imputed = matrices.get("impute_uncorrected")
    complete = ~expected.isna().any(axis=1)
    references = {"drop": expected.loc[complete], "impute": imputed}
    for state, matrix in matrices.items():
        reference = references.get(STATE_ROUTE.get(state), expected)
        if not matrix.index.equals(reference.index) or not matrix.columns.equals(expected.columns):
            raise ValueError(f"Feature/sample order mismatch in {state}")
        values = matrix.to_numpy(dtype=float)
        if np.isinf(values).any() or np.isnan(values).all(axis=1).any():
            raise ValueError(f"Infinite or globally unobserved feature in {state}")
        if ((analysis == "confounding" or STATE_ROUTE.get(state) in ("drop", "impute")) and
                np.isnan(values).any()):
            raise ValueError(f"Incomplete input in state {state}")
        if not np.array_equal(matrix.isna(), reference.isna()):
            raise ValueError(f"Effective scoring masks differ in {state}")
    matrices = {s: {"central": central.get(s, m), "federated": m} for s, m in matrices.items()}
    if analysis == "missingness_overlap":
        if imputed.empty or not imputed.index.is_unique:
            raise ValueError("Expected a nonempty impute route with unique features")
        for state in ROUTES["keep"]:
            matrices[f"drop_{state}"] = {learner: matrix.loc[complete]
                                         for learner, matrix in matrices[state].items()}
    if states is not None:
        matrices = {state: matrix for state, matrix in matrices.items() if state in states}
    return metadata, matrices


def _validate_learner_inputs(matrix: pd.DataFrame, labs: pd.Series, seed: int,
                             *, allow_na=False) -> list[str]:
    if not isinstance(seed, (int, np.integer)) or not 0 < seed < 2**32 - 100:
        raise ValueError("Use a positive learner seed below 2**32 - 100")
    if matrix.empty or not matrix.index.is_unique or not matrix.columns.is_unique:
        raise ValueError("Feature and sample identifiers must be nonempty and unique")
    if not labs.index.equals(matrix.columns) or labs.isna().any():
        raise ValueError("Lab metadata must exactly match matrix sample order")
    if not all(isinstance(value, str) for value in matrix.index):
        raise ValueError("Feature identifiers must be strings")
    if not all(isinstance(value, str) for value in labs):
        raise ValueError("Lab identifiers must be strings")
    values = matrix.to_numpy(dtype=float)
    if allow_na and (np.isinf(values).any() or np.isnan(values).all(axis=1).any()):
        raise ValueError("RF inputs cannot contain infinities or globally unobserved features")
    if not allow_na and not np.isfinite(values).all():
        raise ValueError("Downstream inputs must be complete and finite")
    clients = sorted(labs.unique())
    if len(clients) < 2:
        raise ValueError("At least two labs are required")
    return clients


def _tree_fingerprint(tree) -> str:
    """Hash fitted values explicitly, excluding structured-array padding bytes."""
    digest = hashlib.sha256()
    digest.update(str(tree.random_state).encode())
    state = tree.tree_.__getstate__()
    for name in state["nodes"].dtype.names:
        digest.update(np.ascontiguousarray(state["nodes"][name]).tobytes())
    digest.update(np.ascontiguousarray(state["values"]).tobytes())
    return digest.hexdigest()


def cluster(matrices: dict, labs: pd.Series, seed: int) -> dict:
    """Run k=2, one-start central and locally simulated federated k-means, each on the
matrix of its learner in ``matrices`` ({"central": ..., "federated": ...}).

The app protocol clusters local centroids without sample-count weights, then
performs one local Lloyd update. It is not a pooled Lloyd implementation.
Pooled scaling uses every sample but no condition labels.
"""
    clients = _validate_learner_inputs(matrices["federated"], labs, seed)
    if any((labs == client).sum() < 2 for client in clients):
        raise ValueError("Each k-means client needs at least two samples")
    # Explicit feature and client order replaces the deployed app's set/message order.
    central_matrix, matrix = (matrices[name].sort_index() for name in ("central", "federated"))
    with threadpool_limits(limits=1), contextlib.redirect_stdout(io.StringIO()):
        central = run_central_kmeans(central_matrix, [2], int(seed), n_init=1)[2]
        data = scale_like_federated(matrix)
        parts = [np.flatnonzero(labs.eq(client)) for client in clients]
        fits = [Clustering.init_local_clustering(
            SimpleNamespace(data=data[part].T, scaled=data[part].T),
            2, 2, int(seed), n_init_local=1, n_init_global=1,
        ) for part in parts]
        fits[0].aggregate_centroids({2: [fit.centroids[2] for fit in fits]})
        centroids = fits[0].centroids
        assignments = np.empty(len(labs), dtype=int)
        for part, fit in zip(parts, fits):
            fit.update_local_centroid(centroids, converged=True)
            assignments[part] = fit.labels[2]
    common = {
        "k": 2, "n_init": 1, "seed": int(seed), "feature_order": "lexicographic",
        "scaling_samples": "all samples, no condition labels",
        "n_features": matrix.shape[0], "n_samples": matrix.shape[1],
    }
    return {
        "central": {
            "labels": central["labels"],
            "diagnostics": {
                **common, "inertia": central["inertia"], "n_iter": central["n_iter"],
                "scaling": "pooled feature z-scores, population SD (ddof=0)",
                "implementation": "run_central_kmeans",
            },
        },
        "federated": {
            "labels": pd.Series(assignments, index=matrix.columns),
            "diagnostics": {
                **common, "inertia": pooled_inertia(data, assignments),
                "client_order": clients, "coordinator_aggregations": 1,
                "local_updates": 1, "scaling": "pooled feature z-scores, sample SD (ddof=1)",
                "implementation": "native Clustering app core; fixed message order",
            },
        },
    }


def within_lab_split(labels, labs, seed):
    """Select one stratified split per lab, shared by learners and input states."""
    if not labels.index.equals(labs.index) or not labels.index.is_unique:
        raise ValueError("Split labels/labs must have identical unique sample IDs")
    if labels.isna().any() or set(labels) != {"A", "B"} or labs.isna().any():
        raise ValueError("Stratified splits require valid labs and condition A/B")
    if not isinstance(seed, (int, np.integer)) or not 0 < seed < 2**32 - 100:
        raise ValueError("Use a positive learner seed below 2**32 - 100")
    train, test = {}, []
    for lab in sorted(labs.unique()):
        ids = labs.index[labs.eq(lab)]
        if set(labels.loc[ids]) != {"A", "B"}:
            raise ValueError(f"Lab {lab} needs both A and B for a stratified split")
        training, testing = train_test_split(ids, test_size=0.3, random_state=int(seed),
                                             stratify=labels.loc[ids])
        # Preserve original order for native/core and cross-state consistency.
        train[lab] = ids[ids.isin(training)]
        test.extend(testing)
    return {"train": train, "test": labs.index[labs.index.isin(test)]}


def _forest_parts(labels, labs, clients, held_out, learner, split):
    if learner not in ("central", "federated"):
        raise ValueError("Use central/federated and a held-out lab present in metadata")
    if split is None:
        if held_out not in clients:
            raise ValueError("Use central/federated and a held-out lab present in metadata")
        training_labs = [client for client in clients if client != held_out]
        parts = [labs.index[labs.eq(client)] for client in training_labs]
        test = labs.index[labs.eq(held_out)]
    else:
        if held_out is not None or set(split) != {"train", "test"}:
            raise ValueError("An explicit split requires held_out=None and train/test IDs")
        training_labs = clients
        if set(split["train"]) != set(clients):
            raise ValueError("Explicit split must train on every represented lab")
        parts = [pd.Index(split["train"][client]) for client in clients]
        test = pd.Index(split["test"])
        all_train = pd.Index([sample for part in parts for sample in part])
        if (not all_train.is_unique or not test.is_unique or not len(test) or
                len(all_train.intersection(test)) or
                set(all_train).union(test) != set(labs.index)):
            raise ValueError("Train/test IDs must be disjoint and cover all samples exactly")
        for client, part in zip(clients, parts):
            if not len(part) or not labs.loc[part].eq(client).all():
                raise ValueError("Training partitions must preserve their lab membership")
        if set(labs.loc[test]) != set(clients):
            raise ValueError("Explicit test split must represent every lab")
    for client, part in zip(training_labs, parts):
        if labels.loc[part].isna().any() or set(labels.loc[part]) != {"A", "B"}:
            raise ValueError(f"Training lab {client} must contain both A and B")
    if learner == "central":
        parts = [pd.Index([sample for part in parts for sample in part])]
    return training_labs, parts, test


def _predict_forest_native(matrix: pd.DataFrame, labels: pd.Series, labs: pd.Series,
                           held_out: str, seed: int, learner: str, *, split=None) -> dict:
    """Train the existing forest app on other labs and predict the held-out lab.

Only training labels are read. Central learning uses one pooled training client;
federated learning uses one client per training lab. The app trains 100
non-bootstrap trees with depth 10, sqrt feature subsampling, at least five samples
per leaf and balanced class weights.
"""
    clients = _validate_learner_inputs(matrix, labs, seed, allow_na=True)
    if not labels.index.equals(matrix.columns):
        raise ValueError("Label metadata must exactly match matrix sample order")
    training_labs, train_parts, test = _forest_parts(
        labels, labs, clients, held_out, learner, split)
    matrix = matrix.sort_index()
    config = _forest_config(seed)
    if "condition" in matrix.index or "sample" in matrix.index:
        raise ValueError("Feature identifiers may not be reserved names condition/sample")
    with tempfile.TemporaryDirectory(prefix="simulation_forest_") as temporary:
        root = Path(temporary)
        inputs, outputs = [], []
        for index, part in enumerate(train_parts):
            input_dir, output_dir = root / f"input_{index}", root / f"output_{index}"
            input_dir.mkdir()
            output_dir.mkdir()
            frame = matrix.loc[:, part].T.copy()
            frame["condition"] = labels.loc[part]
            frame.to_csv(input_dir / "data.csv", index_label="sample")
            (input_dir / "config_forest.yaml").write_text(yaml.safe_dump(config))
            inputs.append(str(input_dir))
            outputs.append(str(output_dir))
        with threadpool_limits(limits=1), contextlib.redirect_stdout(io.StringIO()):
            run_simulation_native(inputs, outputs, None, fedForestSimple.main)
        with (Path(outputs[0]) / "global_model.pkl").open("rb") as stream:
            model = pickle.load(stream)
        model_info = yaml.safe_load((Path(outputs[0]) / "model_info.yaml").read_text())
    if model_info["used_features"] != list(matrix.index):
        raise ValueError("Forest app changed the requested feature panel/order")
    return _forest_result(model, matrix, labs, held_out, seed, learner,
                          training_labs, train_parts, config, "native fedForestSimple app", test)


def _forest_config(seed):
    return {
        "simple_forest": {
            "data_filename": "data.csv", "csv_seperator": ",",
            "features_as_columns": True, "sample_col": "sample",
            "predicted_feature_name": "condition", "save_predictions": False,
            "num_estimators_total": 100, "max_depth": 10,
            "min_samples_split": 2, "min_samples_leaf": 5,
            "max_features": "sqrt", "random_state": int(seed),
        },
    }


def _forest_result(model, matrix, labs, held_out, seed, learner, training_labs,
                   train_parts, config, implementation, test_samples):
    # Thread arrival changes tree order. Canonicalize the sum order, preserving all
    # app-fitted trees and their weights, so probabilities are reproducible too.
    trees_per_client = len(model.estimators_) // len(train_parts)
    tree_seeds = [int(tree.random_state) for tree in model.estimators_[:trees_per_client]]
    fingerprints = [(_tree_fingerprint(tree), tree) for tree in model.estimators_]
    fingerprints.sort(key=lambda pair: pair[0])
    model.estimators_ = [tree for _, tree in fingerprints]
    test = matrix.loc[:, test_samples].T
    with threadpool_limits(limits=1):
        prediction = model.predict(test)
        probability = model.predict_proba(test)[:, list(model.classes_).index("B")]
    return {
        "predictions": pd.Series(prediction, index=test_samples),
        "probability_b": pd.Series(probability, index=test_samples),
        "diagnostics": {
            "implementation": implementation,
            "learner": learner, "held_out": held_out, "training_labs": training_labs,
            "n_train": sum(len(part) for part in train_parts), "n_test": len(test_samples),
            "n_features": matrix.shape[0], "num_trees": len(model.estimators_),
            "train_samples": [str(sample) for part in train_parts for sample in part],
            "test_samples": list(test_samples),
            "seed": int(seed), "tree_seeds_per_client": tree_seeds,
            "tree_seed_generation": "numpy RandomState(seed).randint(int32 max)",
            "held_out_labels_used_for_fit": False,
            "feature_order": "lexicographic", "tree_order": "SHA256 of fitted tree values",
            "tree_fingerprints": [fingerprint for fingerprint, _ in fingerprints],
            "config": config,
        },
    }


class ForestPredictor:
    """Reuse native RF cores and identical local fits within one matrix/seed only.

    Training numbers pass through the native CSV parser once in memory. This
    retains its rounding before sklearn's float32 conversion without repeated
    disk exports, model serialization, or unused training-data predictions.
    """

    def __init__(self, matrix, labels, labs, seed):
        self.clients = _validate_learner_inputs(matrix, labs, seed, allow_na=True)
        if not labels.index.equals(matrix.columns):
            raise ValueError("Label metadata must exactly match matrix sample order")
        if "condition" in matrix.index or "sample" in matrix.index:
            raise ValueError("Feature identifiers may not be reserved names condition/sample")
        self.matrix = matrix.sort_index()
        self.labels, self.labs = labels.copy(), labs.copy()
        self.seed = int(seed)
        self.config = _forest_config(seed)
        self.local_trees = {}
        # Parsing values once preserves the native app's CSV numerical semantics.
        # Identifiers remain original; the parser may infer integer sample IDs.
        with io.StringIO() as buffer:
            self.matrix.T.to_csv(buffer, index_label="sample")
            buffer.seek(0)
            self.training = pd.read_csv(buffer, index_col=0)
        if list(self.training.columns) != list(self.matrix.index):
            raise ValueError("Forest CSV parser changed the requested feature panel/order")
        self.training.index = self.matrix.columns

    def predict(self, held_out, learner, *, split=None):
        training_labs, parts, test = _forest_parts(
            self.labels, self.labs, self.clients, held_out, learner, split)
        config = self.config["simple_forest"]
        num_trees = ceil(config["num_estimators_total"] / len(parts))
        trees = []
        with threadpool_limits(limits=1):
            for client, part in zip(training_labs, parts):
                # Pooled folds differ; only per-lab trees recur identically.
                cache_key = (client, tuple(part), num_trees)
                local = self.local_trees.get(cache_key) if learner == "federated" else None
                if local is None:
                    local = fedForestSimple.train_local_trees(
                        self.training.loc[part], self.labels.loc[part], num_trees,
                        config["max_depth"], config["min_samples_split"],
                        config["min_samples_leaf"], config["max_features"], self.seed)
                    if learner == "federated":
                        self.local_trees[cache_key] = local
                trees.extend(local)
            first = parts[0]
            model = fedForestSimple.assemble_forest(
                trees, self.training.loc[first], self.labels.loc[first])
        return _forest_result(
            model, self.matrix, self.labs, held_out, self.seed, learner,
            training_labs, parts, self.config, "in-memory fedForestSimple cores", test)


def predict_forest(matrix: pd.DataFrame, labels: pd.Series, labs: pd.Series,
                   held_out: str, seed: int, learner: str, *, native_io=False,
                   split=None) -> dict:
    """Predict one held-out lab; ``native_io`` retains the app validation reference."""
    if native_io:
        return _predict_forest_native(matrix, labels, labs, held_out, seed, learner, split=split)
    return ForestPredictor(matrix, labels, labs, seed).predict(held_out, learner, split=split)


def score_assignments(target, assignments, labs, task):
    """Score pooled out-of-lab RF predictions or subsets of the global partition."""
    if task not in ("kmeans", "rf"):
        raise ValueError("Use task kmeans or rf")
    if not target.index.equals(assignments.index) or not target.index.equals(labs.index):
        raise ValueError("Prediction, label and lab indices must match exactly")
    if not target.index.is_unique:
        raise ValueError("Duplicate sample identifiers")
    if assignments.isna().any():
        raise ValueError("Missing predictions")
    if task == "kmeans" and not np.isfinite(assignments.to_numpy(dtype=float)).all():
        raise ValueError("Cluster assignments must be finite")
    rows = []
    for lab in ["pooled", *sorted(labs.unique())]:
        mask = np.ones(len(labs), dtype=bool) if lab == "pooled" else labs.eq(lab).to_numpy()
        actual, predicted = target.iloc[mask], assignments.iloc[mask]
        if actual.nunique() != 2:
            raise ValueError(f"Both classes are required in {lab}")
        if task == "kmeans":
            scores = {"ARI": clustering_scores(actual, predicted)["ARI"]}
        else:
            if not set(predicted).issubset({"A", "B"}):
                raise ValueError("RF predictions must use condition A/B")
            scores = {"MCC": classification_scores(actual, predicted)["MCC"]}
        for metric, value in scores.items():
            rows.append(dict(lab=lab, metric=metric, value=value, n_samples=len(actual),
                             n_a=int(actual.eq("A").sum()), n_b=int(actual.eq("B").sum())))
    return rows


def evaluate_inputs(metadata, matrices, seeds, repeat_check,
                    rf_protocols=("loco",), tasks=("kmeans", "rf"), split_inputs=None):
    """Score the learners on ``matrices``, {state: {learner: matrix}}.

    With ``split_inputs`` the forests use its matrices instead: per split (held-out lab
    or within-lab split) and state the matrix of each learner, with the correction
    fitted on the training samples (training_only_inputs).
    """
    target, labs = metadata.condition, metadata.lab
    if target.isna().any() or set(target) != {"A", "B"}:
        raise ValueError("Evaluation requires condition A/B")
    if (not tasks or not set(tasks).issubset({"kmeans", "rf"}) or
            not rf_protocols or not set(rf_protocols).issubset({"loco", "within_lab_70_30"})):
        raise ValueError("Unsupported task or RF protocol")
    metrics, predictions, diagnostics = [], [], []
    for seed in seeds:
        split = within_lab_split(target, labs, seed) if "within_lab_70_30" in rf_protocols else None
        for state, by_learner in matrices.items() if "kmeans" in tasks else ():
            started = time.perf_counter()
            common = dict(state=state, learner_seed=seed, oracle=state == "label_aware")
            partitions = cluster(by_learner, labs, seed)
            repeat = cluster(by_learner, labs, seed) if repeat_check and state == "truth" else None
            for learner, fit in partitions.items():
                assignment = fit["labels"]
                if repeat is not None and not assignment.equals(repeat[learner]["labels"]):
                    raise AssertionError("Repeated k-means assignments differ")
                key = dict(common, task="kmeans", protocol="all_samples", learner=learner,
                           status="evaluable", n_features=by_learner[learner].shape[0])
                metrics.extend(dict(key, **row) for row in
                               score_assignments(target, assignment, labs, "kmeans"))
                predictions.append(pd.DataFrame(dict(
                    key, sample=target.index, lab=labs.to_numpy(), truth=target.to_numpy(),
                    prediction=assignment.to_numpy(), probability_b=np.nan)))
                diagnostics.append(dict(key, **{k: v for k, v in fit["diagnostics"].items()
                                                if k not in key},
                                        repeat_checked=repeat is not None))
            print(f"  {state} seed {seed} k-means: {time.perf_counter()-started:.1f}s", flush=True)
        if "rf" not in tasks:
            continue
        forests = {}
        for protocol in rf_protocols:
            started = time.perf_counter()
            active_split = split if protocol == "within_lab_70_30" else None
            test = target.index if active_split is None else active_split["test"]
            folds = sorted(labs.unique()) if active_split is None else [None]
            results = {}
            for held_out in folds:
                if split_inputs is None:
                    fold = matrices
                else:
                    # A split has its own matrices, so forests are not reused across splits.
                    fold, forests = split_inputs(held_out, active_split), {}
                ids = target.index[labs.eq(held_out)] if active_split is None else test
                for state, by_learner in fold.items():
                    common = dict(state=state, learner_seed=seed,
                                  oracle=state == "label_aware" and split_inputs is None)
                    for learner in ("central", "federated"):
                        matrix = by_learner[learner]
                        # One predictor per matrix; the matrices outlive ``forests``.
                        forest = forests.get(id(matrix)) or forests.setdefault(
                            id(matrix), ForestPredictor(matrix, target, labs, seed))
                        fit = forest.predict(held_out, learner, split=active_split)
                        fit["diagnostics"]["split_class_counts"] = {
                            portion: pd.crosstab(labs.loc[samples], target.loc[samples]).to_dict("index")
                            for portion, samples in (("train", fit["diagnostics"]["train_samples"]),
                                                     ("test", fit["diagnostics"]["test_samples"]))}
                        if not fit["predictions"].index.equals(ids):
                            raise AssertionError("RF prediction IDs differ from held-out IDs")
                        result = results.setdefault((state, learner), dict(
                            common=common, predicted=pd.Series(index=test, dtype=object),
                            probability=pd.Series(index=test, dtype=float),
                            n_features=pd.Series(0, index=test)))
                        result["predicted"].loc[ids] = fit["predictions"]
                        result["probability"].loc[ids] = fit["probability_b"]
                        result["n_features"].loc[ids] = matrix.shape[0]
                        checked = repeat_check and state == "truth" and held_out == folds[0]
                        if checked:
                            changed = target.copy()
                            changed.loc[ids] = "unavailable"
                            again = predict_forest(matrix, changed, labs, held_out, seed, learner,
                                                   split=active_split)
                            if not fit["predictions"].equals(again["predictions"]):
                                raise AssertionError("RF depends on test labels or is not repeatable")
                            np.testing.assert_allclose(fit["probability_b"], again["probability_b"],
                                                       rtol=0, atol=1e-14)
                        diagnostics.append(dict(common, task="rf", protocol=protocol,
                                                **fit["diagnostics"],
                                                repeat_and_label_invariance_checked=checked))
            for (state, learner), result in results.items():
                probability, n_features = result["probability"], result["n_features"]
                if probability.isna().any() or not probability.between(0, 1).all():
                    raise ValueError("Invalid RF probabilities")
                key = dict(result["common"], task="rf", protocol=protocol, learner=learner,
                           status="evaluable")
                # The features of a held-out lab's split; pooled rows give the smallest split.
                metrics.extend(
                    dict(key, n_features=int(n_features.loc[
                        slice(None) if row["lab"] == "pooled" else labs.loc[test].eq(row["lab"])
                    ].min()), **row)
                    for row in score_assignments(target.loc[test], result["predicted"],
                                                 labs.loc[test], "rf"))
                predictions.append(pd.DataFrame(dict(
                    key, n_features=n_features.to_numpy(), sample=test,
                    lab=labs.loc[test].to_numpy(), truth=target.loc[test].to_numpy(),
                    prediction=result["predicted"].to_numpy(),
                    probability_b=probability.to_numpy())))
            print(f"  seed {seed} RF {protocol}: {time.perf_counter()-started:.1f}s", flush=True)
    columns = ["state", "learner_seed", "oracle", "task", "protocol", "learner", "status",
               "n_features", "sample", "lab", "truth", "prediction", "probability_b"]
    predicted = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(columns=columns)
    return pd.DataFrame(metrics), predicted, diagnostics


def training_only_inputs(args, metadata, matrices, sample_minimum=None):
    """Inputs per split whose correction is fitted on the training samples.

    Returns the ``split_inputs`` function of evaluate_inputs, see training_correction.py.
    A held-out lab's inputs do not depend on the learner seed and are kept for the
    following seeds.
    """
    labs, held_out_inputs = metadata.lab, {}

    def split_inputs(held_out, split):
        if held_out in held_out_inputs:
            return held_out_inputs[held_out]
        if split is None:
            train = {lab: labs.index[labs.eq(lab)] for lab in sorted(labs.unique())
                     if lab != held_out}
            test = labs.index[labs.eq(held_out)]
        else:
            train, test = split["train"], split["test"]
        with tempfile.TemporaryDirectory() as work:
            if args.analysis == "confounding":
                fold = training_correction.confounding_split(
                    matrices, metadata, train, test, Path(work), args.states)
            else:
                fold = training_correction.missingness_split(
                    matrices, metadata, sample_minimum, train, test, Path(work), args.states,
                    args.dcombat)
        if split is None:
            held_out_inputs[held_out] = fold
        return fold

    return split_inputs


def verify_checkpoint(directory, expected=None):
    """A completion marker commits an immutable set of checkpoint files."""
    completion = json.loads((directory / "completion.json").read_text())
    required = ({"provenance.json", "metrics.tsv", "timings.tsv", "summary.tsv"}
                if completion.get("status") == "complete" else
                {"metrics.tsv", "predictions.tsv.gz", "learner_diagnostics.json"})
    if not required.issubset(completion.get("artifacts_sha256", {})):
        raise ValueError(f"Incomplete checkpoint manifest: {directory}")
    if expected and any(completion.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Checkpoint identity differs: {directory}")
    for name, digest in completion["artifacts_sha256"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError(f"Invalid checkpoint artifact path: {name}")
        if sha256(directory / name) != digest:
            raise ValueError(f"Checkpoint checksum mismatch: {directory / name}")
    return completion


def commit_checkpoint(directory, details):
    write_json(directory / "completion.json", dict(
        details, artifacts_sha256={p.name: sha256(p) for p in sorted(directory.iterdir())
                                   if p.is_file() and p.name != "completion.json"
                                   and not p.name.endswith(".lock")}))


def check_campaign_sources(args):
    if source_hashes() != args.source_sha256 or sha256(args.reference) != args.reference_sha256:
        raise RuntimeError("Sources/reference changed; checkpoint reuse is not valid")


def missingness_setting(scenario):
    prefix, panel, percent = scenario.split("_")
    if prefix != "missingness" or panel not in ("full", "overlap") or percent not in ("00", "10", "30", "50"):
        raise ValueError("Invalid missingness setting")
    return panel, int(percent) / 100


def evaluate_dataset_job(job):
    """Keep a dataset locked even if its worker outlives the campaign parent."""
    scenario, run, args = job
    destination = args.out / f"{scenario}_run{run:03d}"
    destination.mkdir(exist_ok=True)
    with (destination / ".dataset.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another worker is using {destination}") from error
        return _evaluate_dataset_job(job)


def _evaluate_dataset_job(job):
    """Checkpoint each learner seed; spawned workers never share learner state."""
    scenario, run, args = job
    check_campaign_sources(args)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    run_start = time.perf_counter()
    destination = args.out / f"{scenario}_run{run:03d}"
    destination.mkdir(exist_ok=True)
    inputs = destination / "inputs"
    saved_input = (args.saved_inputs / f"root{run:03d}_inputs.rds"
                   if args.saved_inputs is not None else None)
    panel, target = missingness_setting(scenario) if args.analysis == "missingness_overlap" else ("full", 0.0)
    input_start = time.perf_counter()
    reused = inputs.exists()
    if not inputs.exists():
        # Publish inputs only after generation and validation succeed. Interrupted
        # staging directories remain available for inspection and are never reused.
        staging = Path(tempfile.mkdtemp(prefix="input_pending_", dir=destination))
        candidate = staging / "inputs"
        command = ["Rscript", str(REPO / "evaluation_data/simulated/export_downstream_inputs.R"),
                   "--scenario", scenario, "--run", str(run), "--out", str(candidate.resolve()),
                   "--reference", str(args.reference.resolve())]
        if args.analysis == "missingness_overlap":
            command.extend(["--analysis", args.analysis, "--target", f"{target:g}",
                            "--panel", panel, "--inputs", str(saved_input.resolve())])
        if args.dcombat is not None:
            command.extend(["--dcombat", str(args.dcombat.resolve())])
        with (staging / "generation.log").open("w") as log:
            subprocess.run(command, cwd=REPO, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        check_campaign_sources(args)
        candidate.rename(inputs)
        (staging / "generation.log").rename(destination / "generation.log")
        staging.rmdir()
    validation = json.loads((inputs / "validation.json").read_text())
    input_seconds = time.perf_counter() - input_start
    identity = dict(analysis=args.analysis, scenario=scenario, panel=panel, target=target, run=run,
                    rf_protocols=args.rf_protocols, tasks=args.tasks, correction=args.correction,
                    input_sha256=sha256(inputs / "validation.json"),
                    campaign_sha256=sha256(args.out / "provenance.json"))
    if (destination / "completion.json").exists():
        complete = verify_checkpoint(destination, identity)
        for seed in args.learner_seeds:
            verify_checkpoint(destination / f"seed_{seed}", dict(identity, learner_seed=seed))
        print(f"Reusing completed {scenario} run {run}", flush=True)
        return (pd.read_csv(destination / "metrics.tsv", sep="\t", float_precision="round_trip"),
                complete["timing"], validation["eta2_condition"])
    split_inputs = None
    if args.correction == "joint":
        metadata, matrices = load_inputs(inputs, args.analysis, args.states)
    else:
        metadata, matrices = load_inputs(inputs, args.analysis, ["truth", "uncorrected"])
        sample_minimum = (pd.read_csv(inputs / "sample_minimum.tsv", sep="\t", index_col="file",
                                      float_precision="round_trip").minimum
                          if args.analysis == "missingness_overlap" else None)
        split_inputs = training_only_inputs(
            args, metadata, {state: by_learner["federated"] for state, by_learner in matrices.items()},
            sample_minimum)
    print(f"{scenario} run {run}: {'reused' if reused else 'generated'} validated inputs "
          f"in {input_seconds:.1f}s", flush=True)
    all_metrics, all_predictions, all_diagnostics = [], [], []
    learning_seconds = 0.0
    for seed in args.learner_seeds:
        checkpoint = destination / f"seed_{seed}"
        if checkpoint.exists():
            complete = verify_checkpoint(checkpoint, dict(identity, learner_seed=seed))
            print(f"  Reusing {scenario} run {run} seed {seed}", flush=True)
        else:
            fit_start = time.perf_counter()
            with threadpool_limits(limits=1):
                metrics, predictions, diagnostics = evaluate_inputs(
                    metadata, matrices, [seed], args.repeat_check,
                    rf_protocols=args.rf_protocols, tasks=args.tasks, split_inputs=split_inputs)
            elapsed = time.perf_counter() - fit_start
            for frame in (metrics, predictions):
                frame.insert(0, "scenario", scenario)
                frame.insert(1, "run", run)
                frame.insert(2, "data_seed", validation["data_seed"])
                frame["analysis"], frame["panel"], frame["target"] = args.analysis, panel, target
            for item in diagnostics:
                item.update(analysis=args.analysis, scenario=scenario, panel=panel,
                            target=target, run=run, data_seed=validation["data_seed"])
            staging = Path(tempfile.mkdtemp(prefix=f"seed_{seed}_pending_", dir=destination))
            metrics.to_csv(staging / "metrics.tsv", sep="\t", index=False)
            predictions.to_csv(staging / "predictions.tsv.gz", sep="\t", index=False)
            write_json(staging / "learner_diagnostics.json", diagnostics)
            check_campaign_sources(args)
            commit_checkpoint(staging, dict(identity, learner_seed=seed, learning_seconds=elapsed))
            staging.rename(checkpoint)
            complete = verify_checkpoint(checkpoint)
        learning_seconds += complete["learning_seconds"]
        all_metrics.append(pd.read_csv(checkpoint / "metrics.tsv", sep="\t", float_precision="round_trip"))
        all_predictions.append(pd.read_csv(checkpoint / "predictions.tsv.gz", sep="\t",
                                           float_precision="round_trip"))
        all_diagnostics.extend(json.loads((checkpoint / "learner_diagnostics.json").read_text()))
    metrics = pd.concat(all_metrics, ignore_index=True)
    metrics.to_csv(destination / "metrics.tsv", sep="\t", index=False)
    pd.concat(all_predictions, ignore_index=True).to_csv(
        destination / "predictions.tsv.gz", sep="\t", index=False)
    write_json(destination / "learner_diagnostics.json", all_diagnostics)
    timing = dict(analysis=args.analysis, scenario=scenario, panel=panel, target=target, run=run,
                  generation_seconds=validation["elapsed_seconds"],
                  input_seconds=input_seconds, inputs_reused=reused,
                  learning_seconds=learning_seconds, total_seconds=time.perf_counter()-run_start)
    check_campaign_sources(args)
    commit_checkpoint(destination, dict(identity, timing=timing))
    print(f"Completed {scenario} run {run}: {timing['total_seconds']:.1f}s", flush=True)
    return metrics, timing, validation["eta2_condition"]


def _validate(metrics: pd.DataFrame, eta: dict, states=None) -> pd.DataFrame:
    required = KEY + ["value", "oracle", "status", "n_features"] + COUNTS
    if set(required) - set(metrics.columns):
        raise ValueError(f"Missing metric columns: {sorted(set(required) - set(metrics.columns))}")
    if metrics.empty or metrics[[key for key in required if key != "value"]].isna().any().any():
        raise ValueError("Metric rows must be nonempty and complete")
    if metrics.duplicated(KEY).any():
        raise ValueError("Duplicate metric keys would duplicate paired evidence")
    if states is None:
        states = STATES if metrics.analysis.eq("confounding").all() else ROUTE_STATES
    if set(metrics.state) != set(states) or set(metrics.learner) != {"central", "federated"}:
        raise ValueError("All input states of the analysis and both learners are required")
    if (not metrics.status.eq("evaluable").all() or not np.isfinite(metrics.value).all() or
            not metrics.n_features.gt(0).all()):
        raise ValueError("Every metric must be an evaluable, finite score with features")
    if not ((metrics.task.eq("kmeans") & metrics.protocol.eq("all_samples")) |
            (metrics.task.eq("rf") & metrics.protocol.isin(["loco", "within_lab_70_30"]))).all():
        raise ValueError("Task and protocol must agree")
    # Label-aware inputs are an oracle only when the correction saw the test labels.
    if (metrics.oracle & metrics.state.ne("label_aware")).any():
        raise ValueError("Only label-aware input rows may be marked oracle")
    if not metrics.n_samples.eq(metrics.n_a + metrics.n_b).all():
        raise ValueError("Class counts must sum to the sample count")
    seeds_per_run = metrics.groupby(["scenario", "run"]).data_seed.nunique()
    if not seeds_per_run.eq(1).all():
        raise ValueError("Each scenario/run must identify exactly one generated-data seed")
    if not metrics.groupby(["scenario", "data_seed"]).run.nunique().eq(1).all():
        raise ValueError("Distinct generated runs may not reuse the same data seed")
    if not set(metrics.scenario).issubset(eta) or any(
            not np.isfinite(eta[name]) or not 0 <= eta[name] <= 1
            for name in metrics.scenario.unique()):
        raise ValueError("Provide a finite confounding index in [0,1] for each scenario")
    return metrics.sort_values(KEY).reset_index(drop=True)


def _paired(metrics: pd.DataFrame) -> pd.DataFrame:
    """Form explicit one-to-one pairs; incomplete or nonmatching pairs are errors."""
    rows = []
    states = set(metrics.state)
    state_pairs = [pair for pair in STATE_PAIRS if states.issuperset(pair)]
    for dimension, pairs in (("state", state_pairs),
                             ("learner", [("federated", "central")])):
        matching = [column for column in KEY if column != dimension]
        for left, right in pairs:
            a = metrics.loc[metrics[dimension].eq(left)]
            b = metrics.loc[metrics[dimension].eq(right)]
            # Routes differ in features.
            across = (dimension == "state" and
                      STATE_ROUTE.get(left, "keep") != STATE_ROUTE.get(right, "keep"))
            joined = a.merge(b, on=matching, how="outer", suffixes=("", "_reference"),
                             validate="one_to_one", indicator=True)
            if not joined._merge.eq("both").all():
                raise ValueError(f"Missing paired metric rows for {left} minus {right}")
            for count in COUNTS:
                if not joined[count].eq(joined[f"{count}_reference"]).all():
                    raise ValueError(f"Sample/class count mismatch for {left} minus {right}")
            if not across and not joined.n_features.eq(joined.n_features_reference).all():
                raise ValueError(f"Feature count mismatch for {left} minus {right}")
            result = joined[KEY + COUNTS + ["n_features", "status"]].copy()
            result["comparison"] = f"{left}-minus-{right}"
            result["reference_state"] = right if dimension == "state" else result.state
            result["reference_learner"] = right if dimension == "learner" else result.learner
            result["oracle"] = joined.oracle | joined.oracle_reference
            result["value"] = joined.value - joined.value_reference
            rows.append(result)
    return pd.concat(rows, ignore_index=True).sort_values(PAIR_GROUP + RUN + ["learner_seed"])


def _aggregate(metrics: pd.DataFrame, groups: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average learner seeds before estimating uncertainty across generated datasets."""
    generated = metrics.groupby(groups + RUN, sort=True, dropna=False).agg(
        mean=("value", "mean"), within_run_learner_sd=("value", "std"),
        n_learner_seeds=("value", "count"), n_requested_learner_seeds=("learner_seed", "nunique"),
        oracle=("oracle", "max"),
    ).reset_index()
    generated["n_unavailable_learner_seeds"] = (generated.n_requested_learner_seeds -
                                               generated.n_learner_seeds)
    summary = generated.groupby(groups, sort=True, dropna=False).agg(
        mean=("mean", "mean"), between_run_sd=("mean", "std"),
        n_data_runs=("mean", "count"), n_requested_data_runs=("run", "size"),
        oracle=("oracle", "max"),
        min_learner_seeds=("n_learner_seeds", "min"),
        max_learner_seeds=("n_learner_seeds", "max"),
        mean_within_run_learner_sd=("within_run_learner_sd", "mean"),
    ).reset_index()
    summary["n_unavailable_data_runs"] = summary.n_requested_data_runs - summary.n_data_runs
    summary["mcse"] = summary.between_run_sd / np.sqrt(summary.n_data_runs)
    half_width = t.ppf(0.975, summary.n_data_runs - 1) * summary.mcse
    summary["ci_lower"] = summary["mean"] - half_width
    summary["ci_upper"] = summary["mean"] + half_width
    return generated, summary


def summarize(metrics: pd.DataFrame, out: Path, eta: dict, states=None) -> None:
    """Write the eight metric, uncertainty and diagnostic tables into ``out``."""
    metrics = _validate(metrics, eta, states)
    pairs = _paired(metrics)
    generated, summary = _aggregate(metrics, GROUP)
    paired_generated, paired_summary = _aggregate(pairs, PAIR_GROUP)
    ceilings = metrics.assign(
        exact_one=np.isclose(metrics.value, 1, rtol=0, atol=1e-12),
        near_ceiling=metrics.value.ge(.99),
    ).groupby(GROUP, sort=True).agg(
        n_fits=("value", "count"), n_requested_fits=("value", "size"),
        n_exact_one=("exact_one", "sum"),
        n_at_least_099=("near_ceiling", "sum"), minimum=("value", "min"),
        maximum=("value", "max"), n_data_runs=("run", "nunique"),
        n_learner_seeds=("learner_seed", "nunique"),
    ).reset_index()
    ceilings["n_unavailable_fits"] = ceilings.n_requested_fits - ceilings.n_fits
    availability = metrics.groupby(GROUP + ["status"], sort=True).agg(
        n_fits=("learner_seed", "size"), n_data_runs=("run", "nunique"),
        min_features=("n_features", "min"), max_features=("n_features", "max"),
    ).reset_index()
    artifacts = {
        "metrics": metrics, "generated_run_means": generated, "summary": summary,
        "paired_differences": pairs, "paired_generated_run_means": paired_generated,
        "paired_summary": paired_summary, "ceiling_diagnostics": ceilings,
        "endpoint_availability": availability,
    }
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in artifacts.items():
        if "scenario" in frame:
            frame = frame.assign(eta2_condition=frame.scenario.map(eta))
        frame.to_csv(out / f"{name}.tsv", sep="\t", index=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", choices=("confounding", "missingness_overlap"),
                        default="confounding")
    parser.add_argument("--runs", type=positive_integers, default=[1])
    parser.add_argument("--learner-seeds", type=positive_integers, default=[101])
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS)
    parser.add_argument("--rf-protocols", nargs="+", choices=("loco", "within_lab_70_30"),
                        default=["loco"])
    parser.add_argument("--correction", choices=("training_only", "joint"),
                        default="training_only",
                        help="Fit the correction of the forest inputs on the training samples "
                             "of every split (default), or once on all samples")
    parser.add_argument("--tasks", nargs="+", choices=("kmeans", "rf"),
                        help="Default: rf; for the confounding analysis with --correction joint "
                             "also kmeans, which clusters all samples")
    parser.add_argument("--targets", nargs="+", type=float, choices=(0, 0.1, 0.3, 0.5))
    parser.add_argument("--panels", nargs="+", choices=("full", "overlap"))
    parser.add_argument("--saved-inputs", type=Path,
                        help="Saved missingness rootNNN_inputs.rds directory")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=1,
                        help="Independent dataset processes (1-4); consider available memory")
    parser.add_argument("--out", type=Path, required=True, help="Fresh output directory, or an existing campaign with --resume")
    parser.add_argument("--reference", type=Path, required=True,
                        help="Retained simulation run-level metrics TSV or TSV.gz")
    parser.add_argument("--repeat-check", action="store_true",
                        help="Repeat truth clustering and a truth RF fold with held-out labels flipped")
    parser.add_argument("--resume", action="store_true",
                        help="Resume only matching, checksum-verified seed checkpoints")
    parser.add_argument("--dcombat", type=Path,
                        help="Distributed-ComBat reference code directory (missingness_overlap); "
                             "adds the drop_dcombat and impute_dcombat states")
    parser.add_argument("--states", nargs="+", choices=(*STATES, *ROUTE_STATES, *DCOMBAT_STATES),
                        help="Learn only these input states (default: all of the analysis)")
    args = parser.parse_args(argv)
    if any(r > 10 for r in args.runs):
        parser.error("Generated run IDs must lie within the retained campaign's 1:10")
    if args.scenarios is not None and len(set(args.scenarios)) != len(args.scenarios):
        parser.error("Duplicate scenarios")
    joint_confounding = args.correction == "joint" and args.analysis == "confounding"
    if args.tasks is None:
        args.tasks = ["kmeans", "rf"] if joint_confounding else ["rf"]
    if "kmeans" in args.tasks and not joint_confounding:
        parser.error("k-means clusters all samples of the confounding analysis and needs "
                     "--correction joint")
    for name in ("rf_protocols", "tasks", "targets", "panels", "states"):
        values = getattr(args, name)
        if values is not None and len(values) != len(set(values)):
            parser.error(f"Duplicate {name}")
    if args.analysis == "confounding":
        if args.saved_inputs is not None or args.targets is not None or args.panels is not None:
            parser.error("--saved-inputs/--targets/--panels apply only to missingness_overlap")
        args.scenarios = list(SCENARIOS) if args.scenarios is None else args.scenarios
        args.targets, args.panels = [], []
    else:
        if args.scenarios is not None:
            parser.error("Use --targets and --panels for missingness_overlap")
        if args.saved_inputs is None or not args.saved_inputs.is_dir():
            parser.error("missingness_overlap requires --saved-inputs DIR")
        if any(not (args.saved_inputs / f"root{run:03d}_inputs.rds").is_file() for run in args.runs):
            parser.error("Missing selected rootNNN_inputs.rds")
        args.targets = [0.0, 0.1, 0.3, 0.5] if args.targets is None else args.targets
        args.panels = ["full", "overlap"] if args.panels is None else args.panels
        args.scenarios = [f"missingness_{panel}_{round(target * 100):02d}"
                          for target in args.targets for panel in args.panels]
    missingness = args.analysis == "missingness_overlap"
    if args.dcombat is not None and (not missingness or not args.dcombat.is_dir()):
        parser.error("--dcombat takes the reference code directory and needs missingness_overlap")
    available = (ROUTE_STATES + (tuple(DCOMBAT_STATES) if args.dcombat else ())
                 if missingness else STATES)
    args.states = list(available) if args.states is None else args.states
    if not set(args.states) <= set(available):
        parser.error("--states must be states of the analysis; d-ComBat states need --dcombat")
    if not args.reference.is_file():
        parser.error("--reference must exist")
    if args.out.exists() and not args.resume:
        parser.error("--out already exists; choose a fresh output or use --resume")
    if args.resume and not (args.out / "provenance.json").is_file():
        parser.error("--resume requires an existing campaign provenance.json")
    run_campaign(args, parser)


def run_campaign(args, parser):
    started = time.perf_counter()
    hashes = source_hashes()
    reference_hash = sha256(args.reference)
    args.source_sha256, args.reference_sha256 = hashes, reference_hash
    provenance = dict(
        protocol_version=4, source_sha256=hashes, analysis=args.analysis,
        tasks=args.tasks, rf_protocols=args.rf_protocols, states=args.states,
        correction=args.correction, targets=args.targets, panels=args.panels,
        reference_sha256=reference_hash, reference_path=str(args.reference.resolve()),
        runs=args.runs, scenarios=args.scenarios, learner_seeds=args.learner_seeds,
        repeat_check=args.repeat_check, workers=args.workers, command=sys.argv,
        environment=dict(python=sys.version, platform=platform.platform(),
                         numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__,
                         sklearn=sklearn.__version__),
        protocol=dict(target="condition", k=2,
                      feature_panel=("all 6000; matched across states" if args.analysis == "confounding"
                                     else "keep: observed_ge6_clients_ge3_native, matched names and "
                                          "NA masks; drop: its complete features; impute: "
                                          "observed_ge6_every_client, per-sample minimum"
                                          + ("" if args.correction == "joint" else
                                             "; counted on the training samples of every split, "
                                             "labs below the minimum masked")),
                      seed_unit="generated dataset independent from learner seed",
                      no_label=("joint correction uses unlabeled held-out expression and lab identity"
                                if args.correction == "joint" else
                                "correction fitted on the training samples of every split; "
                                "held-out lab shifted onto the training level, test samples of "
                                "training labs corrected with their lab's coefficients"),
                      label_aware=("not evaluated" if args.analysis != "confounding" else
                                   "oracle: joint correction uses held-out labels"
                                   if args.correction == "joint" else
                                   "training labels as covariate of the training-only "
                                   "correction; test samples as for no_label, without labels"),
                      rf="selected matched LOCO/within-lab 70:30; training labels only; native NA; no scaling",
                      kmeans="complete inputs; all-sample pooled scaling; one start",
                      federation="local native learner reuse, no deployed or SMPC validation",
                      inference="average learner seeds within dataset, then independent run MCSE"))
    if args.resume:
        previous = json.loads((args.out / "provenance.json").read_text())
        for key in ("protocol_version", "source_sha256", "reference_sha256", "runs",
                    "scenarios", "learner_seeds", "repeat_check", "environment", "protocol",
                    "analysis", "tasks", "rf_protocols", "targets", "panels", "states",
                    "correction"):
            if previous.get(key) != provenance[key]:
                parser.error(f"Cannot resume: campaign {key} differs; use a fresh output")
    else:
        args.out.mkdir(parents=True)
    # flock is released even after interruption; a stale file never blocks resume.
    with (args.out / ".campaign.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another process is already using this campaign output")
        if args.resume and (args.out / "completion.json").exists():
            verify_checkpoint(args.out)
            print(f"Campaign already complete and checksums verified: {args.out}", flush=True)
            return
        if not args.resume:
            write_json(args.out / "provenance.json", provenance)
        jobs = [(scenario, run, args) for scenario in args.scenarios for run in args.runs]
        if args.workers == 1:
            results = list(map(evaluate_dataset_job, jobs))
        else:
            with ProcessPoolExecutor(max_workers=args.workers,
                                     mp_context=multiprocessing.get_context("spawn")) as pool:
                # map preserves requested order regardless of completion order.
                results = list(pool.map(evaluate_dataset_job, jobs))
        metrics = pd.concat([result[0] for result in results], ignore_index=True)
        timings = [result[1] for result in results]
        eta = {result[1]["scenario"]: result[2] for result in results}
        summarize(metrics, args.out, eta, args.states)
        if args.dcombat is not None:
            pd.concat([pd.read_csv(args.out / f"{scenario}_run{run:03d}" / "inputs" /
                                   "dcombat_diagnostics.tsv", sep="\t")
                       for scenario, run, _ in jobs]).to_csv(
                args.out / "dcombat_diagnostics.tsv", sep="\t", index=False)
        pd.DataFrame(timings).to_csv(args.out / "timings.tsv", sep="\t", index=False)
        check_campaign_sources(args)
        write_json(args.out / "completion.json", dict(
            status="complete", total_seconds=time.perf_counter()-started,
            n_generated_datasets=len(timings), n_learner_seeds=len(args.learner_seeds),
            artifacts_sha256={str(p.relative_to(args.out)): sha256(p)
                              for p in sorted(args.out.rglob("*"))
                              if p.is_file() and not p.name.endswith(".lock")}))
        print(f"Completed {len(timings)} generated datasets in "
              f"{time.perf_counter()-started:.1f}s: {args.out}", flush=True)



if __name__ == "__main__":
    main()
