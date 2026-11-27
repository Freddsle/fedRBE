#!/usr/bin/env python3
"""Supplementary Figure S11: feature overlap across clients in the real datasets.

Counts the features each client observes (at least one finite value) in the prepared
inputs, compares them with the features fedRBE corrected (the label-free federated
output), writes both count tables under figures/ and draws figures/suppl/Figure_S11.png.

Run from the repository root with Liberation Sans installed:
    pixi run python evaluation/plot_feature_overlap.py
"""

import argparse
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import pandas as pd


DATA = Path("evaluation_data")
# Layer: client inputs, their feature column, and the corrected output.
LAYERS = {
    "E. coli": ("ecoli/before/lab_*/intensities_log_UNION.tsv", "rowname",
                "ecoli/after/no_label/FedApp_corrected_data.tsv"),
    "ccRCC": ("ccRCC_studies/before/P*/report_filtered.tsv", "Gene",
              "ccRCC_studies/after/no_label/FedApp_corrected_data.tsv"),
    "Ovarian cancer": ("ovarian_cancer/before/GSE*/expr_for_correction_UNION.tsv", "rowname",
                       "ovarian_cancer/after/no_label/FedApp_corrected_data.tsv"),
    **{f"Quartet {modality.lower()}": (
        f"quartet_multiomics/before/{modality}/client_*/intensities_log_UNION.tsv", "rowname",
        f"quartet_multiomics/after/no_label/{modality}/FedApp_corrected_data.tsv")
       for modality in ("Transcriptomics", "Proteomics", "Metabolomics")},
}
UPSET_LAYERS = ("E. coli", "ccRCC", "Ovarian cancer")

WIDTH, HEIGHT = 159.2, 193.0  # mm, the supplement's text width
PNG_DPI = 600
MIN_FONT_PT = 8
INK = "#202020"
DATASET_LABELS = {
    "E. coli": r"$\it{E.}$ $\it{coli}$ (proteomics)",
    "ccRCC": "ccRCC (proteomics)",
    "Ovarian cancer": "Ovarian cancer (microarrays)",
}
QUARTET_LABEL = "Quartet (multi-omics)"
COLORS = {
    "all": "#1c5cab",
    "partial": "#86b6ef",
    "two": "#9a9893",
    "one": "#d6d4ce",
}
LEGEND = {
    "all": "All clients",
    "partial": "≥3 clients, fewer than all",
    "two": "2 clients",
    "one": "1 client",
}
LOG_MIN, LOG_MAX = 0.65, 80_000
COUNT_HEIGHT = 32  # mm in each UpSet panel


def present_features(path, feature_column):
    """Features with at least one finite value, read in row blocks."""
    present = set()
    for block in pd.read_csv(path, sep="\t", chunksize=512):
        block = block.set_index(feature_column if feature_column in block.columns
                                else block.columns[0])
        values = block.apply(pd.to_numeric, errors="coerce")
        present.update(values.index[values.notna().any(axis=1)].astype(str))
    return present


def corrected_features(path):
    """Number of feature rows in a corrected output (features x samples)."""
    with path.open() as stream:
        return sum(1 for _ in stream) - 1


def count_features():
    """Per layer: the overview counts and the exact client intersections."""
    layers, intersections = {}, {}
    for layer, (pattern, feature_column, corrected) in LAYERS.items():
        clients = {path.parent.name: present_features(path, feature_column)
                   for path in sorted(DATA.glob(pattern))}
        if not clients:
            raise FileNotFoundError(f"No client inputs match {DATA / pattern}")
        union = set().union(*clients.values())
        memberships = Counter(tuple(client for client in clients if feature in clients[client])
                              for feature in union)
        by_clients = Counter(len(members) for members in memberships.elements())
        retained = corrected_features(DATA / corrected)
        layers[layer] = {
            "clients": len(clients),
            "union": len(union),
            "by_number_of_clients": {n: by_clients[n] for n in sorted(by_clients)},
            "presence_excluded": by_clients[1] + by_clients[2],
            "retained": retained,
            "final_excluded": len(union) - retained,
        }
        if layer in UPSET_LAYERS:
            intersections[layer] = sorted(memberships.items(), key=lambda item: (-item[1], item[0]))
    return layers, intersections


def write_tables(layers, intersections, directory):
    summary = pd.DataFrame([
        {"layer": layer, **{key: value for key, value in row.items()
                            if key != "by_number_of_clients"},
         **{f"in_{n}_clients": count for n, count in row["by_number_of_clients"].items()}}
        for layer, row in layers.items()])
    by_clients = [column for column in summary if column.startswith("in_")]
    summary[by_clients] = summary[by_clients].fillna(0).astype(int)
    summary.to_csv(directory / "feature_overlap_summary.tsv", sep="\t", index=False)
    pd.DataFrame([
        {"layer": layer, "clients": "+".join(members), "n_clients": len(members),
         "features": count}
        for layer, patterns in intersections.items() for members, count in patterns
    ]).to_csv(directory / "feature_overlap_intersections.tsv", sep="\t", index=False)


def axes_at(figure, x, y, width, height):
    return figure.add_axes((x / WIDTH, y / HEIGHT, width / WIDTH, height / HEIGHT))


def panel_title(figure, letter, title, x, y):
    figure.text(x / WIDTH, y / HEIGHT, letter, weight="bold", fontsize=11, va="baseline")
    figure.text((x + 5) / WIDTH, y / HEIGHT, title, fontsize=10, va="baseline")


def plot_overview(figure, layers):
    panel_title(figure, "a", "Feature overlap across clients", 3, 188)
    figure.legend(
        [Patch(facecolor=COLORS[key]) for key in LEGEND],
        list(LEGEND.values()),
        loc="upper left",
        bbox_to_anchor=(3 / WIDTH, 184 / HEIGHT),
        borderaxespad=0,
        frameon=False,
        ncol=4,
        fontsize=MIN_FONT_PT,
        handlelength=1.4,
    )

    chart_left, chart_width = 44, 43
    ax = axes_at(figure, chart_left, 139, chart_width, 30)
    ax.set(xlim=(0, 100), ylim=(len(layers) + 0.5, -0.5), xticks=[0, 25, 50, 75, 100])
    ax.tick_params(axis="x", labelsize=MIN_FONT_PT, length=2, pad=2)
    ax.set_xlabel("Share of input features (%)", labelpad=3)
    ax.set_yticks([])
    ax.spines[["left", "top", "right"]].set_visible(False)

    columns = [(106, "Input features"), (129.5, "In 1–2 clients"), (156, "Total excluded")]
    for x, label in columns:
        figure.text(x / WIDTH, 171 / HEIGHT, label, ha="right", weight="bold")

    row_transform = ax.get_yaxis_transform()
    for i, (layer, row) in enumerate(layers.items()):
        is_quartet = layer.startswith("Quartet ")
        if layer == "Quartet transcriptomics":
            ax.text((3 - chart_left) / chart_width, i, QUARTET_LABEL,
                    transform=row_transform, va="center")
        position = i + 1 if is_quartet else i
        label = layer.removeprefix("Quartet ").capitalize() if is_quartet else DATASET_LABELS[layer]
        label_left = 6 if is_quartet else 3
        ax.text((label_left - chart_left) / chart_width, position, label,
                transform=row_transform, va="center")

        counts = {int(n): count for n, count in row["by_number_of_clients"].items()}
        union = row["union"]
        all_clients = counts.get(row["clients"], 0)
        shares = (
            all_clients,
            union - all_clients - row["presence_excluded"],
            counts.get(2, 0),
            counts.get(1, 0),
        )
        left = 0
        for key, count in zip(LEGEND, shares):
            width = 100 * count / union
            if width:
                ax.barh(position, width, left=left, height=0.68, color=COLORS[key])
            left += width

        values = [f"{union:,}"]
        for count in (row["presence_excluded"], row["final_excluded"]):
            values.append(f"{count:,} ({count / union:.1%})" if count else "0 (0%)")
        for (x, _), value in zip(columns, values):
            ax.text((x - chart_left) / chart_width, position, value,
                    transform=row_transform, ha="right", va="center")


def plot_upset(figure, patterns, letter, title, x, y, width, title_x, title_y):
    panel_title(figure, letter, title, title_x, title_y)
    clients = sorted({client for members, _ in patterns for client in members})
    positions = range(len(patterns))
    counts = [count for _, count in patterns]
    colors = []
    for members, _ in patterns:
        if len(members) == len(clients):
            colors.append(COLORS["all"])
        elif len(members) >= 3:
            colors.append(COLORS["partial"])
        else:
            colors.append(COLORS["two" if len(members) == 2 else "one"])

    bars = axes_at(figure, x, y, width, COUNT_HEIGHT)
    bars.bar(
        positions,
        [count - LOG_MIN for count in counts],
        bottom=LOG_MIN,
        color=colors,
        width=min(0.67, 4.0 * len(patterns) / width),
    )
    bars.set(yscale="log", ylim=(LOG_MIN, LOG_MAX), xlim=(-0.55, len(patterns) - 0.45))
    bars.set_axisbelow(True)
    bars.yaxis.set_major_locator(FixedLocator([1, 10, 100, 1_000, 10_000]))
    bars.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    bars.yaxis.set_minor_locator(NullLocator())
    bars.tick_params(axis="y", labelsize=MIN_FONT_PT, length=2, pad=2)
    bars.grid(axis="y", color="#e5e5e5", linewidth=0.55)
    bars.set_xticks([])
    bars.spines[["top", "right", "bottom"]].set_visible(False)
    for position, count in zip(positions, counts):
        # Give neighboring three-digit labels in the dense panel separate baselines.
        label_offset = 3.9 if len(patterns) > 20 and count >= 100 and position % 2 else 1.18
        bars.text(position, count * label_offset, f"{count:,}",
                  ha="center", va="bottom")

    matrix_height = 3.45 * len(clients)
    matrix = axes_at(figure, x, y - matrix_height - 0.9, width, matrix_height)
    matrix.set(xlim=bars.get_xlim(), ylim=(len(clients) - 0.5, -0.5), xticks=[])
    matrix.set_yticks(range(len(clients)), [client.removeprefix("lab_") for client in clients])
    matrix.tick_params(axis="y", labelsize=MIN_FONT_PT, length=0, pad=3)
    matrix.spines[:].set_visible(False)
    for i in range(0, len(clients), 2):
        matrix.axhspan(i - 0.5, i + 0.5, color="#f3f4f5", linewidth=0)
    client_rows = {client: i for i, client in enumerate(clients)}
    for position, (members, _) in zip(positions, patterns):
        rows = sorted(client_rows[client] for client in members)
        matrix.plot([position, position], [min(rows), max(rows)], color=INK, linewidth=0.85)
        matrix.scatter([position] * len(clients), range(len(clients)), s=10,
                       facecolors="white", edgecolors="#8c8c8c", linewidths=0.55, zorder=3)
        matrix.scatter([position] * len(rows), rows, s=13, color=INK, linewidths=0, zorder=4)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=Path("figures"),
                        help="Directory for the count tables; the PNG goes to its suppl/")
    args = parser.parse_args()
    layers, intersections = count_features()
    figure_dir = args.output_dir / "suppl"
    figure_dir.mkdir(parents=True, exist_ok=True)
    write_tables(layers, intersections, args.output_dir)
    plt.rcParams.update({
        "font.family": "Liberation Sans",
        "font.size": MIN_FONT_PT,
        "text.color": INK,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Liberation Sans",
        "mathtext.it": "Liberation Sans:italic",
        "mathtext.cal": "Liberation Sans",
        "mathtext.bf": "Liberation Sans:bold",
        "pdf.fonttype": 42,
        "axes.linewidth": 0.55,
    })
    figure = plt.figure(figsize=(WIDTH / 25.4, HEIGHT / 25.4))
    plot_overview(figure, layers)
    plot_upset(figure, intersections["E. coli"], "b", DATASET_LABELS["E. coli"],
               16, 91, 140, 3, 128)
    figure.text(156 / WIDTH, 128 / HEIGHT, "Feature counts", ha="right")
    plot_upset(figure, intersections["ccRCC"], "c", DATASET_LABELS["ccRCC"], 24, 26, 55, 3, 63)
    plot_upset(figure, intersections["Ovarian cancer"], "d", DATASET_LABELS["Ovarian cancer"],
               109, 26, 47, 84, 63)
    figure.savefig(figure_dir / "Figure_S11.png", dpi=PNG_DPI)
    plt.close(figure)


if __name__ == "__main__":
    main()
