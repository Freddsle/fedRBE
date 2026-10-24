"""Plot saved RF metrics with the shared Figure 4 R style, split by correction arm."""

import argparse
from pathlib import Path
import subprocess


def main():
    component = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-file", type=Path,
        default=component / "results" / "classification_metric_report.csv",
        help="RF metric CSV (default: the component's canonical results)",
    )
    parser.add_argument(
        "--plots-dir", type=Path, default=component / "plots",
        help="Output directory for label_aware/ and no_label/ PNGs",
    )
    args = parser.parse_args()
    if not args.results_file.is_file():
        parser.error(f"RF report does not exist: {args.results_file}")
    helper = component.parent / "evaluation_utils" / "figure_aggregate_utils.R"
    expression = (
        "args <- commandArgs(TRUE); source(args[[1]]); "
        "export_classification_plots(args[[2]], args[[3]])"
    )
    subprocess.run(
        ["Rscript", "-e", expression, str(helper), str(args.results_file), str(args.plots_dir)],
        check=True,
    )


if __name__ == "__main__":
    main()
