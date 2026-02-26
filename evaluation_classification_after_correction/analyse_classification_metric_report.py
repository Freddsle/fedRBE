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
RESULTS_FILE = os.path.join(SCRIPT_DIR, "results", "classification_metric_report.csv")
AVERAGE_CLIENT_NAME = "All"
    # in the predicted_client_name column, this value is treated as being the average
    # over multiple runs that are otherwise identical
DATANAME_TO_LABEL = {
    "Balanced Simulated Data": "Simulated Additive Batch",
    "Balanced Simulated Data (Rotational Batch Effect)": "Simulated Rotational Batch",
}
# these datanames will be displayed with the corresponding label in the plots.
# If a dataname is not in this dict, it will be displayed as is.
DATANAME_TO_EXCLUDE = [
    "Mildly Imbalanced Simulated Data",
    "Strongly Imbalanced Simulated Data",
    "Microbiome Data",
    "Mildly Imbalanced Simulated Data (Rotational Batch Effect)",
    "Strongly Imbalanced Simulated Data (Rotational Batch Effect)",
]
# These datanames will be excluded from the plots, even if they are in the results file.

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
# remove duplicates if any
df = df.drop_duplicates()

# Load colour schema
with open(COLOUR_SCHEMA_FILE, 'r') as f:
    colour_schema = json.load(f)
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

# filter out datanames that should be excluded
df = df[~df['data_name'].isin(DATANAME_TO_EXCLUDE)]
# apply dataname labels
df['data_name'] = df['data_name'].apply(lambda x: DATANAME_TO_LABEL.get(x, x))

# metric name x cross validation method plots
for metric_name in df['metric_name'].unique():
    for cv_method in df['cross_validation_method'].unique():
        df_subset = df[(df['metric_name'] == metric_name) & (df['cross_validation_method'] == cv_method)]

        # filter out average rows
        df_subset = df_subset[df_subset['predicted_client_name'] != AVERAGE_CLIENT_NAME]
        print(f"Using {len(df_subset)} rows for metric '{metric_name}' and CV method '{cv_method}'")

        # swarmplot
        plt.figure(figsize=(14, 8))
        axes = sns.swarmplot(
            data=df_subset,
            x='data_name',
            y='metric_value',
            hue='data_preprocessing_name',
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
            hue='data_preprocessing_name',
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

# summary statistics
summary = df.groupby(['data_preprocessing_name', 'cross_validation_method', 'metric_name', 'data_name', 'predicted_client_name'])['metric_value'].agg(['mean', 'std', 'count'])
summary_output_file = os.path.join(SCRIPT_DIR, "classification_metric_report_summary.csv")
summary.to_csv(summary_output_file)
print(f"\nSummary statistics saved to {summary_output_file}")
