#!/usr/bin/env python3
"""Estimate the between-feature to within-feature spread ratio for each E. coli lab."""

import argparse
import csv
import math
import statistics
from pathlib import Path


def spread(before):
    sample_lab = {}
    labs = [f"lab_{letter}" for letter in "ABCDE"]
    for lab in labs:
        with (before / lab / "design.tsv").open() as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                sample = row["file"]
                if sample in sample_lab:
                    raise ValueError(f"Duplicate design entry for {sample}")
                sample_lab[sample] = lab
    means = {lab: [] for lab in labs}
    sds = {lab: [] for lab in labs}
    with (before / "central_intensities_log_UNION.tsv").open() as stream:
        reader = csv.reader(stream, delimiter="\t")
        samples = next(reader)[1:]
        if len(samples) != len(set(samples)) or set(samples) != set(sample_lab):
            raise ValueError("The intensity matrix and lab designs must contain the same samples")
        indices = {lab: [i for i, sample in enumerate(samples, 1) if sample_lab[sample] == lab]
                   for lab in labs}
        for row in reader:
            if len(row) != len(samples) + 1:
                raise ValueError(f"Wrong sample count for feature {row[0]}")
            for lab, columns in indices.items():
                observed = [float(row[i]) for i in columns if row[i] not in ("", "NA", "NaN", "nan")]
                if not all(math.isfinite(x) for x in observed):
                    raise ValueError(f"Non-finite intensity for {row[0]} in {lab}")
                if len(observed) >= 10:
                    means[lab].append(statistics.mean(observed))
                    sds[lab].append(statistics.stdev(observed))
    for lab in labs:
        between = statistics.stdev(means[lab])
        within = statistics.median(sds[lab])
        yield dict(lab=lab, features=len(means[lab]), sd_feature_means=between,
                   median_within_feature_sd=within, ratio=between / within)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--before", type=Path,
                        default=Path(__file__).resolve().parents[2] / "ecoli/before")
    args = parser.parse_args()
    records = list(spread(args.before))
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, records[0].keys(), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
