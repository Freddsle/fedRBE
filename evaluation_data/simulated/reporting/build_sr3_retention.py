#!/usr/bin/env python3
"""Build Figure SR3 from the five-client retention and learner feature counts."""

import csv
from decimal import Decimal, ROUND_HALF_UP
import gzip
import argparse
from pathlib import Path

from figure_png import save_png
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "results/missingness_overlap"
ROUTE_COUNTS = SOURCE / "informative_learner_eligibility.tsv"
LEVELS = (0., .1, .3, .5)
ROUTES = (("Keep", "rf_features"), ("Drop", "kmeans_features"),
          ("Impute", "impute_features"))
PANELS = (("overlap", "Partially overlapping panels"), ("full", "Full panels"))
INK = "#262e36"
CMAP = LinearSegmentedColormap.from_list("retained", ["#f6f8fa", "#75b4c6"])


def read(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_rows(path, rows):
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, rows[0].keys(), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def count_summary(rows):
    result = []
    for panel, _ in PANELS:
        for group, roles, total in (("all", ("all",), 6000),
                                    ("A/B", ("A_DE", "B_DE"), 100)):
            for route, column in ROUTES:
                for target in LEVELS:
                    selected = [r for r in rows if r["panel"] == panel
                                and r["role"] in roles and float(r["target"]) == target]
                    assert len(selected) == 10 * len(roles)
                    values = []
                    for root in range(1, 11):
                        unit = [r for r in selected if int(r["root"]) == root]
                        assert len(unit) == len(roles)
                        assert sum(int(r["total_features"]) for r in unit) == total
                        values.append(sum(int(r[column]) for r in unit))
                    result.append(dict(panel=panel, group=group, route=route, target=target,
                                       total=total, mean_count=mean(values), n=10))
    return result


def count_matrix(ax, rows, group, letter, title):
    ax.set(xlim=(-1.12, 4), ylim=(8.05, -1.3))
    ax.axis("off")
    ax.set_title(f"{letter}  {title}", loc="left", fontsize=10, fontweight="bold", pad=16)
    ax.text(2, -1.55, "Target missingness (%)", ha="center", va="center", fontsize=8.5)
    for col, level in enumerate(LEVELS):
        ax.text(col + .5, -.77, f"{level:.0%}", ha="center", va="center", fontsize=9)
    for panel_index, (panel, label) in enumerate(PANELS):
        start = panel_index * 4.3
        ax.text(-1.1, start + .02, label, fontsize=8.5, fontweight="bold", va="center")
        for route_index, (route, _) in enumerate(ROUTES):
            y = start + route_index + 1
            ax.text(-.15, y, route, ha="right", va="center", fontsize=8.5)
            for col, target in enumerate(LEVELS):
                row, = [r for r in rows if (r["panel"], r["group"], r["route"], r["target"])
                        == (panel, group, route, target)]
                value = row["mean_count"]
                ax.add_patch(Rectangle((col, y - .46), .98, .92,
                                       facecolor=CMAP(value / row["total"]),
                                       edgecolor="white", lw=.5))
                if group == "all":
                    label = f"{int(Decimal(str(value)).quantize(Decimal('1'), rounding=ROUND_HALF_UP)):,}"
                else:
                    label = f"{value:.1f}"
                ax.text(col + .49, y, label, ha="center", va="center", fontsize=8.6)


def retention_plot(ax, metrics):
    records = []
    ax.set_title("c  Keep route, by assigned labs",
                 loc="left", fontsize=10, fontweight="bold", pad=10)
    for labs, color, marker in ((5, INK, "o"), (4, "#687680", "s"),
                                (3, "#98a6af", "^")):
        averages = []
        for target in LEVELS:
            selected = sorted([r for r in metrics if r["panel"] == "overlap"
                               and r["arm"] == "without_condition"
                               and r["stage"] == "eligibility"
                               and r["metric"] == "retained_fraction"
                               and r["presence"] == str(labs)
                               and float(r["target"]) == target], key=lambda r: int(r["root"]))
            assert [int(r["root"]) for r in selected] == list(range(1, 11))
            values = [100 * float(r["value"]) for r in selected]
            averages.append(mean(values))
            ax.scatter(target * 100 + np.linspace(-.45, .45, 10), values,
                       s=8, color=color, alpha=.45, edgecolors="none", zorder=2)
            records.extend(dict(labs=labs, target=target, root=int(r["root"]),
                                retained_percent=v) for r, v in zip(selected, values))
        ax.plot([100 * v for v in LEVELS], averages, color=color, marker=marker,
                markersize=4.5, markeredgecolor="white", markeredgewidth=.5, lw=1.2)
        ax.text(52, averages[-1], f"{labs} labs",
                color=INK, va="center", fontsize=8.5)
    ax.set(xlim=(-1.5, 58), ylim=(50, 103), xticks=(0, 10, 30, 50),
           yticks=(50, 60, 70, 80, 90, 100), xlabel="Target missingness (%)",
           ylabel="Features retained (%)")
    ax.grid(axis="y", color="#e7e9ec", lw=.6)
    ax.set_axisbelow(True)
    return records


def plotted_means(counts, retention, output):
    rows = [dict(figure="SR3", panel="a" if r["group"] == "all" else "b",
                 scenario=f"{r['panel']}_{100 * r['target']:.0f}", state=r["route"],
                 metric="feature_count", statistic="mean", center=r["mean_count"])
            for r in counts]
    for labs in (3, 4, 5):
        for target in LEVELS:
            values = [r["retained_percent"] for r in retention
                      if r["labs"] == labs and r["target"] == target]
            rows.append(dict(figure="SR3", panel="c", scenario=f"overlap_{100 * target:.0f}",
                             state=f"{labs} labs", metric="retained_percent",
                             statistic="mean", center=mean(values)))
    write_rows(output / "sr3_plotted_values.tsv", rows)


def main(output):
    output.mkdir(parents=True, exist_ok=True)
    route_counts = read(ROUTE_COUNTS)
    rows = count_summary(route_counts)
    metrics = read(SOURCE / "informative_metrics.tsv.gz")
    # The keep route is the no-label native filter (six observations in three labs).
    # Reuse filtering counts only, and verify every dataset against the route totals.
    keep = {(r["panel"], float(r["target"]), r["root"]): int(r["rf_features"])
            for r in route_counts if r["role"] == "all"}
    native = {(r["panel"], float(r["target"]), r["root"]): int(float(r["value"]))
              for r in metrics if r["arm"] == "without_condition"
              and r["stage"] == "eligibility" and r["presence"] == "all"
              and r["metric"] == "retained_features"}
    assert len(keep) == 80 and keep == native, "Keep/native filtering counts differ"
    write_rows(output / "sr3_feature_counts.tsv", rows)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "text.color": INK, "axes.labelcolor": INK,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": .65, "axes.edgecolor": "#68727a",
                         "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig = plt.figure(figsize=(180 / 25.4, 122 / 25.4))
    count_matrix(fig.add_axes((.03, .48, .45, .43)), rows, "all", "a", "All features (6,000)")
    count_matrix(fig.add_axes((.53, .48, .45, .43)), rows, "A/B", "b", "A/B features (100)")
    colorbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=Normalize(0, 100), cmap=CMAP),
                           cax=fig.add_axes((.70, .425, .27, .015)), orientation="horizontal",
                           ticks=(0, 50, 100))
    colorbar.outline.set_visible(False)
    colorbar.solids.set_rasterized(False)
    colorbar.ax.tick_params(labelsize=8, length=2)
    fig.text(.68, .426, "Retained (%)", ha="right", fontsize=8.5)
    records = retention_plot(fig.add_axes((.11, .105, .86, .275)), metrics)
    write_rows(output / "sr3_retention_by_labs.tsv", records)
    plotted_means(rows, records, output)
    save_png(fig, output / "sr3_five_client.png")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Output directory")
    main(parser.parse_args().output)
