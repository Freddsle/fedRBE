"""
Learning inputs whose batch effect correction sees only the training samples.

For one split, a held-out lab or a 70:30 split within every lab, the functions
build the input matrices of every state. They follow the random forest
evaluation of the real datasets and reuse its steps
(evaluation_classification_after_correction/helper/steps):

1. fedRBE, run in the native simulation, corrects the training samples for the
   federated forest; central limma::removeBatchEffect (central_batch_fit of
   evaluation_utils/central_correction.R) corrects the pooled training samples
   for the centralized forest. The label-aware state uses the labels of the
   training samples as covariate.
2. A held-out lab is shifted per feature onto the training level
   (batchcorrection.classes.client.align_to_training_level). Test samples of a
   training lab are corrected with the batch coefficients of their lab. Neither
   step uses the labels of the test samples.

With missing values, the features of a split follow from its training samples:
the keep route retains a feature with at least MIN_OBSERVATIONS observed values
in at least MIN_CLIENTS training labs. A lab with fewer observed values of a
feature, among its training samples or as held-out lab, is masked for that
feature. The drop route uses the retained features without missing values in
the training samples; the impute route uses the features with the minimum in
every training lab and fills a missing value with the lowest observed value of
its sample. d-ComBat for the federated forest and centralized ComBat for the
centralized forest are fitted on the training samples of the drop and impute
routes (dcombat_training.R); a held-out lab is shifted onto the level of the
corrected training samples like for fedRBE.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
COMPONENT = REPO / "evaluation_classification_after_correction"
for path in (REPO, COMPONENT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from batchcorrection.classes.client import Client  # noqa: E402
from helper.common import ARM_LABEL_AWARE, ARM_LABEL_FREE  # noqa: E402
from helper.steps import correct, prepare  # noqa: E402
from helper.steps.datasets import ClientInput, DatasetSpec, Federation  # noqa: E402

ARMS = {"no_label": ARM_LABEL_FREE, "label_aware": ARM_LABEL_AWARE}
# fedRBE's minimum of observed values per lab with five labs and no covariates
MIN_OBSERVATIONS = 6
MIN_CLIENTS = 3
# Largest accepted difference between a federated correction and its central counterpart
TOLERANCE = 1e-8
DCOMBAT_SCRIPT = Path(__file__).with_name("dcombat_training.R")


def _config(position, covariates, min_samples, min_clients):
    return {"batch_col": None, "covariates": covariates, "data_filename": "intensities.tsv",
            "design_filename": "design.tsv", "design_separator": "\t",
            "expression_file_flag": True, "index_col": "rowname", "min_samples": min_samples,
            "min_clients": min_clients, "normalizationMethod": None, "position": position,
            "separator": "\t", "smpc": False}


def _combat_training(matrix, metadata, train, test, dcombat, work):
    """
    d-ComBat (federated) and ComBat (central) fitted on the training samples
    (dcombat_training.R). Each matrix holds the training samples and the
    adjusted test samples of the training labs; its level is the mean of its
    corrected training samples per feature.
    """
    training = metadata.index.isin([sample for samples in train.values() for sample in samples])
    selected = training | (metadata.index.isin(test) & metadata.lab.isin(list(train)))
    folder = work / "dcombat"
    folder.mkdir(parents=True)
    matrix.loc[:, selected].to_csv(folder / "input.tsv", sep="\t", na_rep="NA")
    pd.DataFrame({"file": metadata.index[selected], "lab": metadata.lab[selected],
                  "part": np.where(training[selected], "train", "test")}).to_csv(
        folder / "samples.tsv", sep="\t", index=False)
    learners = ("federated", "central")
    process = subprocess.run(
        [correct.rscript(), str(DCOMBAT_SCRIPT), str(dcombat), str(folder / "input.tsv"),
         str(folder / "samples.tsv"), *(str(folder / f"{name}.tsv") for name in learners)],
        capture_output=True, text=True, check=False)
    if process.returncode != 0:
        raise RuntimeError(f"ComBat on the training samples failed:\n{process.stderr[-3000:]}")
    hashed = Client().hash_names(matrix.index)[0]
    corrections = {}
    for name in learners:
        corrected = pd.read_csv(folder / f"{name}.tsv", sep="\t", index_col=0)
        level = corrected.loc[:, metadata.index[training]].mean(axis=1)
        level.index = level.index.map(hashed)
        corrections[name] = correct.TrainingCorrection(
            None, level, {"normalizationMethod": None}, set(train), [], [corrected])
    return corrections


def corrected_split(matrix, metadata, train, test, work, *, arm=ARM_LABEL_FREE, min_samples=0,
                    min_clients=MIN_CLIENTS, dcombat=None):
    """
    The matrix with its training samples corrected and its test samples
    brought onto their scale.
    Args:
        matrix: Uncorrected values, features x all samples
        metadata: Per sample (index) its lab and condition (A or B)
        train: Per training lab its training samples
        test: The test samples
        work: New folder for the inputs and outputs of the corrections
        arm: ARM_LABEL_FREE or ARM_LABEL_AWARE, the covariates of the training fit
        min_samples, min_clients: fedRBE settings of the training fit
        dcombat: Folder of the d-ComBat reference code; d-ComBat and ComBat then
            replace fedRBE and central limma
    Returns:
        {"federated": matrix after fedRBE (d-ComBat), "central": matrix after
        central limma (ComBat)}
    """
    labs = sorted(metadata.lab.unique())
    training_labs = [lab for lab in labs if lab in train]
    covariates = {ARM_LABEL_FREE: [], ARM_LABEL_AWARE: ["B"]}
    configs = {name: [_config(position, covariates[name], min_samples, min_clients)
                      for position in range(len(labs))] for name in covariates}
    data = matrix.rename_axis("rowname")
    inputs = []
    for lab in labs:
        samples = metadata.index[metadata.lab.eq(lab)]
        condition = metadata.condition.loc[samples]
        design = pd.DataFrame({"condition": condition, "B": condition.eq("B").astype(int)},
                              index=samples.rename("file"))
        inputs.append(ClientInput(data.loc[:, samples], design, design.index))
    spec = DatasetSpec("simulation", "simulation", "condition",
                       [Federation(None, labs, [], configs)])
    split = pd.DataFrame({"cohort": metadata.lab.to_numpy(), "sample": metadata.index,
                          "label": metadata.condition.to_numpy(),
                          "part": np.where(metadata.index.isin(test), "test", "train")})
    folders = prepare.write_split_inputs(spec, [inputs], split, work / "split")
    train_folders, test_folders = folders[(0, "train")], folders[(0, "test")]
    if dcombat is not None:
        corrections = _combat_training(matrix, metadata, train, test, dcombat, work)
    else:
        keep = [labs.index(lab) for lab in training_labs]
        training = Federation(None, training_labs, [],
                              {name: [configs[name][index] for index in keep]
                               for name in configs})
        fit = (training, [inputs[index] for index in keep], arm, train_folders, split, work)
        corrections = {"federated": correct.correct_training(*fit),
                       "central": correct.correct_training_central(*fit)}
    test_labs = [lab for lab in labs if metadata.lab.loc[test].eq(lab).any()]
    result = {}
    for name, correction in corrections.items():
        parts = list(correction.matrices)
        for lab, folder in zip(test_labs, test_folders):
            config = configs[ARM_LABEL_FREE][labs.index(lab)]
            if lab not in training_labs:
                parts.append(correct.align_held_out(prepare.load_client(folder, lab, config),
                                                    config, correction))
            elif dcombat is None:
                corrected = correction.corrected[training_labs.index(lab)]
                parts.append(correct.correct_new_samples(
                    prepare.load_client(folder, lab, config), correction.coefficients,
                    correction.batches, corrected,
                    correct.sample_batches(lab, inputs[labs.index(lab)], None,
                                           corrected.columns)))
        result[name] = pd.concat(parts, axis=1).reindex(index=matrix.index,
                                                        columns=matrix.columns)
    difference = (result["federated"] - result["central"]).abs().max().max()
    if not difference <= TOLERANCE:
        raise ValueError(f"The federated and central corrections differ by {difference}")
    return result


def confounding_split(matrices, metadata, train, test, work, states):
    """
    The states of the confounding simulation for one split, {state: {learner:
    matrix}}. Truth and uncorrected values are the same for both learners.
    """
    # Two training labs remain when one of the three labs is held out.
    min_clients = min(MIN_CLIENTS, len(train))
    fold = {}
    for state in states:
        if state in ARMS:
            fold[state] = corrected_split(matrices["uncorrected"], metadata, train, test,
                                          work / state, arm=ARMS[state],
                                          min_clients=min_clients)
        else:
            fold[state] = dict.fromkeys(("central", "federated"), matrices[state])
    return fold


def missingness_split(matrices, metadata, sample_minimum, train, test, work, states,
                      dcombat=None):
    """
    The states of the missingness simulation for one split, {state: {learner:
    matrix}}, see the module docstring. matrices holds the truth and the
    uncorrected values of the features that the keep route retains over all
    labs; the features of a split are among them.
    """
    labs = sorted(metadata.lab.unique())
    observed = matrices["uncorrected"].notna()
    # A training lab counts its training samples, a held-out lab all its samples.
    eligible = pd.DataFrame({
        lab: observed.loc[:, train.get(lab, metadata.index[metadata.lab.eq(lab)])].sum(axis=1)
        >= MIN_OBSERVATIONS for lab in labs})
    in_training = eligible[list(train)]
    masked = ~eligible[metadata.lab.tolist()].to_numpy()
    training_samples = [sample for samples in train.values() for sample in samples]
    settings = dict(min_samples=MIN_OBSERVATIONS)

    keep = in_training.sum(axis=1) >= MIN_CLIENTS
    route = {state: matrices[state].mask(masked).loc[keep] for state in ("truth", "uncorrected")}
    corrected = None
    if {"no_label", "drop_no_label"} & set(states):
        corrected = corrected_split(route["uncorrected"], metadata, train, test, work / "keep",
                                    **settings)
    complete = route["uncorrected"].loc[:, training_samples].notna().all(axis=1)
    impute = in_training.all(axis=1)
    imputed = matrices["uncorrected"].loc[impute]
    imputed = imputed.mask(imputed.isna(), sample_minimum, axis=1).mask(
        masked[impute.to_numpy()])

    fold = {}
    for state in states:
        if state in ("truth", "uncorrected", "drop_truth", "drop_uncorrected"):
            matrix = route[state.removeprefix("drop_")]
            fold[state] = dict.fromkeys(("central", "federated"),
                                        matrix.loc[complete] if "drop" in state else matrix)
        elif state in ("no_label", "drop_no_label"):
            fold[state] = {learner: matrix.loc[complete] if "drop" in state else matrix
                           for learner, matrix in corrected.items()}
        elif state == "impute_uncorrected":
            fold[state] = dict.fromkeys(("central", "federated"), imputed)
        elif state == "impute_no_label":
            fold[state] = corrected_split(imputed, metadata, train, test, work / "impute",
                                          **settings)
        else:
            source = imputed if state == "impute_dcombat" else route["uncorrected"].loc[complete]
            fold[state] = corrected_split(source, metadata, train, test, work / state,
                                          dcombat=dcombat, **settings)
    return fold
