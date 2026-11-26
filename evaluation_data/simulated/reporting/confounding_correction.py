#!/usr/bin/env python3
"""Draw Figure SR1 from the saved three-client run metrics and recovery summary."""

import argparse
import csv
import gzip
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from figure_png import save_png

STUDY = Path(__file__).resolve().parents[1]
SCENARIOS = ("confounded_p50", "confounded_p30", "confounded_p15", "confounded_p05")
PLOTTED_FIELDS = ("figure", "panel", "scenario", "task", "learner", "lab", "state",
                  "metric", "statistic", "center")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5,
    "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8.2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .65, "axes.edgecolor": "#68727a", "text.color": "#262e36",
    "axes.labelcolor": "#262e36", "xtick.color": "#262e36", "ytick.color": "#262e36",
    "figure.facecolor": "white", "axes.facecolor": "white",
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "svg.hashsalt": "fedrbe-robustness-candidate-20261028",
    "savefig.facecolor": "white", "savefig.dpi": 300,
})


def series(label, color, keys, marker="o", open_=False):
    return dict(label=label, color=color, keys=keys, marker=marker, open=open_)


def draw(ax, xs, items, letter, title, ylabel, *, summary, raw, plotted, width, offsets):
    """Saved means over every finite run value."""
    ax.text(-.085, 1.13, letter, transform=ax.transAxes, fontweight="bold", fontsize=11,
            va="bottom")
    ax.set_title(title, loc="left", y=1.13, pad=0, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#e7e9ec", linewidth=.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=.6)
    for i, x in enumerate(xs):
        present = [s for s in items if s["keys"][i] is not None]
        for j, s in enumerate(present):
            row = summary[s["keys"][i]]
            values = [(r, v) for r, v in raw[s["keys"][i]] if math.isfinite(v)]
            assert [run for run, _ in values] == list(range(1, 11)) and int(row["n_runs"]) == 10
            xi = x + (offsets[j] if offsets else (j - (len(present) - 1) / 2) * width)
            jitter = np.linspace(-width * .32, width * .32, len(values))
            ax.scatter(xi + jitter, [v for _, v in values], s=10, color=s["color"],
                       alpha=.7, edgecolors="none", zorder=2)
            avg = float(row["mean"])
            ax.scatter([xi], [avg], s=38, marker=s["marker"], zorder=4, **face(s))
            plotted.append(dict(figure="SR1", panel=letter, scenario=s["keys"][i][0], state=s["label"],
                   metric=s["keys"][i][3], statistic="mean", center=repr(avg)))


def face(s):
    if s["open"]:
        return dict(facecolors="white", edgecolors=s["color"], linewidths=1.1)
    return dict(color=s["color"], edgecolors="white", linewidths=.4)


def handle(s):
    if s["marker"] == "_":
        return Line2D([], [], ls="none", marker="_", markersize=12, markeredgewidth=1.3,
                      color=s["color"], label=s["label"])
    return Line2D([], [], ls="none", marker=s["marker"], markersize=6,
                  markerfacecolor="white" if s["open"] else s["color"],
                  markeredgecolor=s["color"] if s["open"] else "white", label=s["label"])


def save(fig, output, stem, items, ncol):
    labels = dict((s["label"], handle(s)) for s in items)
    fig.legend(list(labels.values()), list(labels), loc="lower center", ncol=ncol,
               frameon=False, handletextpad=.3, columnspacing=1.4)
    save_png(fig, output / f"{stem}.png")
    plt.close(fig)


TRUTH = "#5b6168"
PINK = "#CC79A7"  # batch-free truth bars, as in SR2 and the paper's Figure 4
STATE_COLORS = {"uncorrected": "#8f979e", "no_label": "#bc543b", "label_aware": "#176788",
                "joint": "#009E73"}
STATE_NAMES = {"uncorrected": "Uncorrected", "no_label": "No-label",
               "label_aware": "Label-aware", "joint": "Joint model, uncorrected data"}


def scenario_axis(ax, summary):
    """The four confounding scenarios, evenly spaced and labelled as in SR2."""
    labels = []
    for s in SCENARIOS:
        eta = round(float(summary[s, "none", "input", "eta2_condition"]["mean"]), 3)
        p = f"{int(s[-2:]) / 100:.2f}"
        labels.append(f"p = {p}\nη² = {'0' if eta == 0 else f'{eta:.2f}'}")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_xlim(-.55, len(labels) - .45)


def truth_marks(ax, xs, values, half=.38):
    """Truth across each scenario's states, below the points."""
    for x, value in zip(xs, values):
        ax.plot([x - half, x + half], [value, value], color=PINK, lw=1.6,
                solid_capstyle="butt", zorder=1.5)


def confounding_correction(output, summary, raw, plotted):
    """False positives and recall against η² (Figure SR1)."""
    fig, axes = plt.subplots(2, 1, figsize=(180 / 25.4, 130 / 25.4))
    fig.subplots_adjust(left=.095, right=.985, bottom=.13, top=.91, hspace=.62)
    xs = list(range(len(SCENARIOS)))
    # Three condition-only states, then a gap and the joint model of the uncorrected data.
    offsets = [-.29, -.13, .03, .29]
    states = {"uncorrected": ("with_condition", "before"),
              "no_label": ("without_condition", "after"),
              "label_aware": ("with_condition", "after"),
              "joint": ("with_condition", "joint")}
    panels = [("de_condition_fp", "a", "False positives", "Null features called",
               "3,600 null features"),
              ("de_condition_recall", "b", "Recall", "Recall",
               "2,250 A and B features")]
    legend = {}
    for ax, (metric, letter, title, ylabel, subtitle) in zip(axes, panels):
        items = [series(STATE_NAMES[st], STATE_COLORS[st],
                        [(s, *states[st], metric) for s in SCENARIOS],
                        {"uncorrected": "D", "no_label": "s", "label_aware": "o",
                         "joint": "^"}[st],
                        open_=st == "uncorrected") for st in states]
        draw(ax, xs, items, letter, title, ylabel, summary=summary, raw=raw, plotted=plotted,
             width=.14, offsets=offsets)
        ax.text(0, 1.035, subtitle, transform=ax.transAxes, fontsize=8, va="bottom")
        truth = [float(summary[s, "with_condition", "truth", metric]["mean"]) for s in SCENARIOS]
        truth_marks(ax, xs, truth)
        for scenario, value in zip(SCENARIOS, truth):
            plotted.append(dict(figure="SR1", panel=letter, scenario=scenario, state="Batch-free truth",
                   metric=metric, statistic="mean", center=repr(value)))
        scenario_axis(ax, summary)
        legend.update((item["label"], item) for item in items)
    axes[0].axhline(3600, color=TRUTH, lw=.7, ls=(0, (1, 1.5)), zorder=1)
    axes[0].set_yscale("symlog", linthresh=10)
    axes[0].set_yticks([0, 5, 10, 100, 1000, 3600],
                      ["0", "5", "10", "100", "1,000", "3,600"])
    axes[0].set_ylim(-1, 6000)
    axes[0].minorticks_off()
    axes[0].annotate("All null features", xy=(.5, 3600), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom",
                     color=TRUTH, fontsize=7)
    axes[0].get_ygridlines()[-1].set_visible(False)
    axes[1].set_ylim(-.04, 1.08)
    fig.align_ylabels(axes)
    legend = [legend[STATE_NAMES[st]] for st in states] + [
        series("Batch-free truth", PINK, [], "_")]
    save(fig, output, "sr1_confounding_correction", legend, 5)


def write_plotted(output, plotted):
    with (output / "plotted_values_three_client.tsv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, PLOTTED_FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(plotted)


def main(output):
    output.mkdir(parents=True, exist_ok=True)
    fields = ("scenario", "arm", "stage", "metric")
    with (STUDY / "robustness/summary.tsv").open() as stream:
        summary = {tuple(r[k] for k in fields): r for r in csv.DictReader(stream, delimiter="\t")
                   if r["scenario"] in SCENARIOS}
    raw = defaultdict(list)
    with gzip.open(STUDY / "results/confounding_downstream/reference_metrics.tsv.gz", "rt") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if row["scenario"] in SCENARIOS and row["metric"] in ("de_condition_fp", "de_condition_recall"):
                raw[tuple(row[k] for k in fields)].append((int(row["run"]), float(row["value"])))
    for runs in raw.values():
        runs.sort()
    plotted = []
    confounding_correction(output, summary, raw, plotted)
    write_plotted(output, plotted)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Output directory")
    main(parser.parse_args().output)
