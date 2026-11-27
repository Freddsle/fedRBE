#!/usr/bin/env python3
"""Build the numerical cells of simulation Tables SR1–SR4 and S4 from saved results."""

import argparse
import csv
import gzip
import math
from collections import defaultdict
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
FIVE = STUDY / "results/missingness_overlap"
SUMMARY = STUDY / "robustness/summary.tsv"
EQUIVALENCE = STUDY / "robustness/equivalence.tsv"
METRICS = FIVE / "informative_metrics.tsv.gz"
DCOMBAT = FIVE / "informative_dcombat_diagnostics.tsv"
LABEL_AWARE = FIVE / "informative_label_aware_diagnostics.tsv"
REFERENCE = FIVE / "informative_reference_scores.tsv"
MISSING_FRACTION = STUDY / "results/missing_fraction_slopes.tsv"
MISSING_FRACTION_KEEP = FIVE / "informative_common_cell_slopes.tsv"

SCENARIOS = ("balanced", "mild_imbalanced", "strong_imbalanced", "confounded_p50",
             "confounded_p30", "confounded_p15", "confounded_p05", "intensity_lower",
             "intensity_dependent")
ARMS = ("without_condition", "with_condition")
TARGETS = (0, 0.1, 0.3, 0.5)
PANELS = ("full", "overlap")
ROOTS = set(range(1, 11))


def rows(path):
    with (gzip.open(path, "rt") if path.suffix == ".gz" else path.open()) as stream:
        yield from csv.DictReader(stream, delimiter="\t")


def summarize(records, column, unit, maximum=False):
    ids = [int(row[unit]) for row in records]
    if len(ids) != len(ROOTS) or set(ids) != ROOTS:
        raise ValueError(f"Expected one record per dataset 1–10 for {column}, got {ids}")
    values = [float(r[column]) for r in records if r[column] not in ("", "NA")]
    if not all(math.isfinite(v) for v in values):
        raise ValueError(f"Non-finite {column} for datasets {ids}")
    if not values:
        return None, 0
    return (max(values) if maximum else math.fsum(values) / len(values)), len(values)


def recovery():
    summary = {(r["scenario"], r["arm"], r["stage"], r["metric"]): r for r in rows(SUMMARY)}
    for scenario in SCENARIOS:
        selections = [(ARMS[0], "before", "residual_lab_rms")]
        for arm in ARMS:
            for metric in ("attenuation_condition", "attenuation_latent"):
                selections.append((arm, "after", metric))
            selections.append((arm, "after", "residual_lab_rms"))
        if scenario in SCENARIOS[:7]:
            selections.append(("none", "input", "eta2_condition"))
        for arm, stage, metric in selections:
            row = summary[scenario, arm, stage, metric]
            value = float(row["mean"])
            if (int(row["n_runs"]), int(row["n_missing"])) != (10, 0) or not math.isfinite(value):
                raise ValueError(f"Incomplete recovery summary for {scenario}/{arm}/{stage}/{metric}")
            yield dict(scenario=scenario, arm=arm, stage=stage, metric=metric, datasets=10, value=value)


def agreement(block="combined"):
    groups = defaultdict(list)
    designs = []
    if block in ("three-client", "combined"):
        for r in rows(EQUIVALENCE):
            groups[(r["scenario"], "", "", r["arm"], r["metric"])].append(r)
        designs.extend((s, "", "") for s in SCENARIOS)
    if block in ("five-client", "combined"):
        for r in rows(METRICS):
            if r["stage"] == "agreement" and r["presence"] == "all":
                groups[("", r["panel"], float(r["target"]), r["arm"], r["metric"])].append(r)
        designs.extend(("", panel, target) for panel in PANELS for target in TARGETS)
    for scenario, panel, target in designs:
        for statistic, metric in (("mean", "mean_diff"), ("maximum", "max_diff")):
            for arm in ARMS:
                records = groups[scenario, panel, target, arm, metric]
                value, n = summarize(records, "value", "run" if scenario else "root",
                                     maximum=statistic == "maximum")
                if n != 10 or any(not 0 <= float(r["value"]) <= 1e-8 for r in records):
                    raise ValueError(f"Correction agreement failed for {scenario or panel}/{target}/{arm}")
                yield dict(scenario=scenario, panel=panel, target=target, arm=arm,
                           statistic=statistic, datasets=n, value=value)


def missing_fraction():
    groups = defaultdict(list)
    for path in (MISSING_FRACTION, MISSING_FRACTION_KEEP):
        for r in rows(path):
            arm = "keep_cells" if path == MISSING_FRACTION_KEEP else r["arm"]
            groups[r["design"], r["missing_percent"], r["contrast"], arm].append(r)
    columns = [("with_condition", "features"), ("with_condition", "masked_truth"),
               ("keep_cells", "corrected"), ("with_condition", "corrected")]
    for design in ("five_client_full", "five_client_overlap"):
        for group in ("0-5", "5-10", "10-20", "20-30", "30-50", "50-100"):
            for contrast in ("condition", "latent"):
                for arm, column in columns:
                    records = groups[design, group, contrast, arm]
                    value, n = summarize(records, column, "dataset")
                    yield dict(design=design, missing_percent=group, contrast=contrast, arm=arm,
                               column=column, datasets=n, value=value)


def correction():
    groups = defaultdict(list)

    def collect(r, route, method, stage):
        groups[(r["panel"], route, method, stage, r["metric"], float(r["target"]))].append(r)

    for path in (DCOMBAT, LABEL_AWARE):
        for r in rows(path):
            if path != LABEL_AWARE or r["method"] != "uncorrected":
                collect(r, r["route"], r["method"], r["stage"])
    for r in rows(REFERENCE):
        if r["setting"] == "matched" and (r["route"] == "keep" or r["method"] == "masked_truth"):
            collect(r, r["route"], r["method"], r["stage"])
    stages = {"fedrbe": "masked_truth", "fedrbe_vs_complete": "complete_truth"}
    for r in rows(METRICS):
        if r["arm"] == "with_condition" and r["presence"] == "all" and r["stage"] in stages:
            collect(r, "keep", "fedrbe_label_aware", stages[r["stage"]])
    for metric, stage in (("residual_lab_rms", "masked_truth"),
                          ("attenuation_condition", "complete_truth")):
        methods = ["uncorrected", "fedrbe", "fedrbe_label_aware", "dcombat", "dcombat_label_aware"]
        if metric == "attenuation_condition":
            methods.insert(0, "masked_truth")
        for panel in PANELS:
            for route in ("keep", "drop", "impute"):
                for target in TARGETS:
                    for method in methods:
                        n, value = None, None
                        if route != "keep" or not method.startswith("dcombat"):
                            value, n = summarize(groups[panel, route, method, stage, metric, target],
                                                 "value", "root")
                        yield dict(panel=panel, target=target, route=route, method=method,
                                   stage=stage, metric=metric, datasets=n, value=value)


def main(output):
    output.mkdir(parents=True, exist_ok=True)
    for name, cells in (("sr1_recovery", recovery), ("s4_agreement", agreement),
                        ("sr2_missing_fraction", missing_fraction), ("sr3_sr4_correction", correction)):
        records = list(cells())
        with (output / f"{name}.tsv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, records[0].keys(), delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    main(parser.parse_args().output)
