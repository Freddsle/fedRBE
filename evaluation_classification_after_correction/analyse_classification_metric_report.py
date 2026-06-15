import os
from pathlib import Path
import json

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(Path(os.path.abspath(__file__)).parent)
COLOUR_SCHEMA_FILE = os.path.join(ROOT_DIR, "evaluation_utils", "colour_schema.json")
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
RESULTS_FILE = os.path.join(RESULTS_DIR, "classification_metric_report.csv")
AVERAGE_CLIENT_NAME = "All"
FEDERATED_LEARNING_TYPE = "federated"
    # in the predicted_client_name column, the AVERAGE_CLIENT_NAME for the FEDERATED_LEARNING_TYPE
    # is treated as being the average over multiple runs that are otherwise identical
    # We therefore ignore it in the plots as it's just an aggregation of the other rows
    # in centralized results, this is the only results row tho!
DATANAME_TO_LABEL = {
    "Balanced Simulated Data": "Simulated Additive Batch",
    "Balanced Simulated Data (Rotational Batch Effect)": "Simulated Rotational Batch",
}
# these datanames will be displayed with the corresponding label in the plots.
# If a dataname is not in this dict, it will be displayed as is.
DATANAME_TO_INCLUDE = [
    "Ovarian cancer",
    "E. coli",
    "ccRCC",
]
# These datanames will be included in the plots; everything else is filtered out.

# file system management
if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)
if not os.path.exists(RESULTS_FILE):
    print(f"Error: Results file {RESULTS_FILE} does not exist. Please run the classification experiments first.")
    print("python3 run_classification_train_test_split.py")
    print("python3 run_classification_leave_one_cohort_out.py")
    exit(1)

# Load classification metric report
df = pd.read_csv(RESULTS_FILE)
if 'learning_type' not in df.columns:
    raise ValueError(f"Expected 'learning_type' column in {RESULTS_FILE} not found. Please check the file.")
# remove duplicates if any
df = df.drop_duplicates()
print(f"Loaded {len(df)} rows from {RESULTS_FILE} after removing duplicates.")

# Load colour schema
colour_schema = json.loads(Path(COLOUR_SCHEMA_FILE).read_text(encoding='utf-8'))
palette = colour_schema.get('palette', sns.color_palette())
background_color = colour_schema.get('background', '#ffffff')
grid_color = colour_schema.get('grid', '#d3d3d3')

sns.set_palette(palette)
sns.set_style(
    "whitegrid",
    {
        'grid.color': grid_color,
        'axes.facecolor': background_color,
        'figure.facecolor': background_color,
    },
)
plt.rcParams.update(
    {
        'figure.facecolor': background_color,
        'savefig.facecolor': background_color,
        'axes.facecolor': background_color,
    }
)

# ensure no duplicate of all columns except metric_value
duplicates = df[df.duplicated(subset=df.columns.difference(['metric_value']), keep=False)]
if len(duplicates) > 0:
    print(f"Error: Found duplicate rows in {RESULTS_FILE} differing only in metric_value. Please check the file.")
    print("Duplicates:")
    print(duplicates)
    exit(1)

# filter to datanames that should be included
df = df[df['data_name'].isin(DATANAME_TO_INCLUDE)]
print(f"Using {len(df)} rows after filtering to included datanames: {DATANAME_TO_INCLUDE}")

# metric name x target x cross validation method plots
for metric_name in df['metric_name'].unique():
    for target in df['predicted_target'].unique():
        for cv_method in df['cross_validation_method'].unique():
            df_subset = df[(df['metric_name'] == metric_name) & \
                           (df['cross_validation_method'] == cv_method) & \
                            (df['predicted_target'] == target)].copy()

            # filter out average rows of the federated learning type
            rows_to_exclude = (df_subset['learning_type'] == FEDERATED_LEARNING_TYPE) & (df_subset['predicted_client_name'] == AVERAGE_CLIENT_NAME)
            df_subset = df_subset[~rows_to_exclude]
            print(f"Using {len(df_subset)} rows for metric '{metric_name}' and CV method '{cv_method}'")

            # we add the predicted target to the data name
            df_subset['data_name'] = df_subset.apply(lambda row: f"{row['data_name']} (Predicted Target: {row['predicted_target']})", axis=1)
            df_subset['plot_hue'] = df_subset.apply(
                lambda row: f"{row['data_preprocessing_name']} / {row['learning_type']}",
                axis=1,
            )
            output_csv = os.path.join(RESULTS_DIR, f"plotting_data_{metric_name.replace(' ', '_').lower()}_{cv_method.replace(' ', '_').lower()}.csv")
            df_subset.to_csv(output_csv, index=False)
            print(f"Saved plotting data to {output_csv}")
            # swarmplot
            plt.figure(figsize=(14, 8))
            axes = sns.swarmplot(
                data=df_subset,
                x='data_name',
                y='metric_value',
                hue='plot_hue',
            )
            axes.set_facecolor(background_color)
            plt.xlabel("")
            plt.ylabel(metric_name)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

            output_png = os.path.join(PLOTS_DIR, f"classification_analysis_swarmplot_{metric_name.replace(' ', '_').lower()}_{cv_method.replace(' ', '_').lower()}.png")
            figure = axes.get_figure()
            assert figure is not None
            figure.savefig(output_png, bbox_inches='tight', dpi=100)
            plt.close()
            print(f"Saved plot to {output_png}")

            # boxplot
            plt.figure(figsize=(14, 8))
            axes = sns.boxplot(
                data=df_subset,
                x='data_name',
                y='metric_value',
                hue='plot_hue',
                medianprops=dict(color=colour_schema.get('boxplot_median_marker_color', '#FF0000'), linewidth=2)
            )
            axes.set_facecolor(background_color)
            plt.xlabel("")
            plt.ylabel(metric_name)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

            output_png = os.path.join(PLOTS_DIR, f"classification_analysis_boxplot_{metric_name.replace(' ', '_').lower()}_{cv_method.replace(' ', '_').lower()}.png")
            figure = axes.get_figure()
            assert figure is not None
            figure.savefig(output_png, bbox_inches='tight', dpi=100)
            plt.close()
            print(f"Saved plot to {output_png}")

