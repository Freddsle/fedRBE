"""Build the selected ovarian clinical table, counts and plot from source annotations."""

import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
STUDIES = ("GSE6008", "GSE26712", "GSE40595", "GSE69428", "GSE38666")


def build():
    with (ROOT / "source_cohort_metadata.tsv").open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = [name for name in reader.fieldnames if name != "shared_donor_14407_38666"]
        source = list(reader)
    with (ROOT / "duplicate_cel_pairs.tsv").open() as handle:
        pairs = list(csv.DictReader(handle, delimiter="\t"))
    by_sample = {row["sample_id"]: row for row in source}
    if len(by_sample) != len(source):
        raise ValueError("Source clinical table contains duplicate sample IDs.")
    excluded = {pair["excluded_sample"] for pair in pairs}
    if excluded != {row["sample_id"] for row in source if row["study"] == "GSE14407"}:
        raise ValueError("Raw CEL mapping must cover every excluded GSE14407 array.")
    for pair in pairs:
        left, right = by_sample[pair["excluded_sample"]], by_sample[pair["retained_sample"]]
        if (pair["byte_for_byte_equal"] != "True"
                or left["condition"] != right["condition"]
                or left["study"] != pair["excluded_study"]
                or right["study"] != pair["retained_study"]):
            raise ValueError("Inconsistent raw CEL duplicate mapping.")
    counterparts = {pair["retained_sample"]: pair["excluded_sample"] for pair in pairs}
    selected = []
    for row in source:
        if row["study"] in STUDIES and row["sample_id"] not in excluded:
            selected.append({**{name: row[name] for name in fields},
                             "excluded_duplicate_sample": counterparts.get(row["sample_id"], "NA")})
    before = ROOT.parent / "before"
    descriptor = json.loads((before / "datainfo.json").read_text())
    if {cohort["name"] for cohort in descriptor["cohorts"]} != set(STUDIES):
        raise ValueError("Selected clinical studies do not match the active cohort descriptor.")
    for cohort in descriptor["cohorts"]:
        spec = cohort["designfile"]
        with (before / cohort["folder"] / spec["filename"]).open() as handle:
            design = list(csv.DictReader(handle, delimiter=spec["separator"]))
        actual = {row[spec["samplename_column"]]: row["HGSC"] for row in design}
        expected = {row["sample_id"]: row["HGSC"] for row in selected
                    if row["study"] == cohort["name"]}
        if actual != expected or len(actual) != len(design):
            raise ValueError(f"Clinical sample IDs or labels differ from {cohort['name']} design.")
    fields.append("excluded_duplicate_sample")
    with (ROOT / "cohort_metadata.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(selected)

    groups = [[row for row in selected if row["study"] == study] for study in STUDIES]
    summary = []
    for study, rows in zip(STUDIES, groups):
        classes = Counter(row["condition"] for row in rows)
        summary.append(dict(study=study, arrays=len(rows), HGSC=classes["HGSC"],
                            controls=classes["Control"],
                            ages_available=sum(row["age_years"] != "NA" for row in rows)))
    with (ROOT / "selected_cohort_summary.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, list(summary[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)
    plot(groups)


def plot(groups):
    orange, blue, green = "#D55E00", "#0072B2", "#009E73"
    missing, undefined = "#BDBDBD", "#EEEEEE"
    panels = [
        ("condition", "HGSC/control", [("HGSC", orange), ("Control", blue)]),
        ("stage_group", "tumor stage", [("Early I/II", orange), ("Late III/IV", blue),
                                         ("Missing", missing), ("Control: N/A", undefined)]),
        ("grade_numeric", "numeric tumor grade", [("2", orange), ("3", blue), ("2-3", green),
                                                   ("Missing", missing), ("Control: N/A", undefined)]),
        ("surgery_outcome", "surgery outcome", [("Optimal", orange), ("Suboptimal", blue),
                                                ("Missing", missing), ("Control: N/A", undefined)]),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(18, 5.5), sharey=True)
    for axis, (field, title, categories) in zip(axes, panels):
        left = [0.0] * len(groups)
        for category, color in categories:
            values = [sum(row[field] == category for row in rows) / len(rows) for rows in groups]
            axis.barh(range(len(groups)), values, left=left, height=0.8, color=color,
                      label=category)
            left = [a + b for a, b in zip(left, values)]
        if any(abs(total - 1) > 1e-12 for total in left):
            raise ValueError(f"Unmapped metadata category in {field}.")
        axis.set(xlim=(0, 1), title=title)
        axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2,
                    fontsize=7, frameon=False)
    for i, rows in enumerate(groups):
        ages = [float(row["age_years"]) for row in rows if row["age_years"] != "NA"]
        if ages:
            axes[4].boxplot([ages], positions=[i], orientation="horizontal", widths=0.48,
                            showfliers=False, medianprops={"color": orange})
            axes[4].text(0.98, i + 0.32, f"n={len(ages)}", ha="right", va="center",
                         transform=axes[4].get_yaxis_transform(), fontsize=8)
        else:
            axes[4].text(0.5, i, "unavailable", ha="center", va="center", color="grey",
                         transform=axes[4].get_yaxis_transform(), fontsize=8)
    axes[4].set_title("age (years)")
    axes[0].set_yticks(range(len(groups)),
                      [f"{study} (n={len(rows)})" for study, rows in zip(STUDIES, groups)])
    axes[0].set_ylim(len(groups) - 0.5, -0.5)
    for axis in axes:
        axis.tick_params(labelsize=8)
        axis.title.set_fontsize(10)
    total = sum(map(len, groups))
    fig.suptitle(f"Ovarian microarray evaluation: metadata for {total} arrays from five studies",
                 fontsize=13, y=0.98)
    fig.text(0.5, 0.025,
             "Bars: fraction of arrays. Light grey = undefined in controls; grey = unavailable. "
             "No imputation.\nCounts are arrays, not unique patients. "
             "GSE14407 is excluded using exact raw CEL duplication.",
             ha="center", fontsize=8)
    fig.subplots_adjust(left=0.10, right=0.99, top=0.80, bottom=0.27, wspace=0.14)
    fig.savefig(ROOT / "cohort_metadata.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    build()
