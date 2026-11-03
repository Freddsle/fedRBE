# Aggregate performance figures
#
# Build the runtime and memory figures of the dockerized fedRBE runs from the
# *_performance_clients.csv tables that generate_fedrbe_corrected_datasets.py
# writes next to each corrected dataset.
# Run from the repository root:
#   pixi run Rscript evaluation/aggregate_performance.R
# PERFORMANCE_DATA_DIR overrides the folder searched for the tables (default
# evaluation_data). PERFORMANCE_RUNTIME_OUTPUT and PERFORMANCE_MEMORY_OUTPUT
# override the output paths; an output path ending in .svg writes an SVG.

source(file.path("evaluation_utils", "figure_aggregate_utils.R"))
require_figure_packages(c("ggplot2", "grid", "rlang", "scales"))

# Source and output files, overrides are absolute or relative to the repository root.

env_or_default <- function(variable, default) {
  value <- Sys.getenv(variable, "")
  if (!nzchar(value)) return(default)
  value
}
data_dir <- env_or_default("PERFORMANCE_DATA_DIR", "evaluation_data")
runtime_output_path <- env_or_default(
  "PERFORMANCE_RUNTIME_OUTPUT", file.path("figures", "performance_runtime.png")
)
memory_output_path <- env_or_default(
  "PERFORMANCE_MEMORY_OUTPUT", file.path("figures", "performance_memory.png")
)

# Figure parameters: PNG sizes in inches.

runtime_width <- 7.5
runtime_height_per_row <- 2.4
memory_width <- 6
memory_height <- 4.5
figure_dpi <- 600

# Load data: one row per participant and run, all datasets and correction arms.

clients <- read_performance_clients(data_dir)
cat("Read", nrow(clients), "participant runs of", length(unique(clients$dataset)), "datasets\n")

# Build and save the figures.

runtime_plot <- plot_performance_runtime(clients)
runtime_rows <- ceiling(length(unique(clients$dataset)) / 3)
saved_path <- save_grid_png(
  function() print(runtime_plot),
  output_path = runtime_output_path,
  width = runtime_width,
  height = runtime_height_per_row * runtime_rows + 1.6,
  dpi = figure_dpi
)
cat("Saved the runtime figure to", saved_path, "\n")

memory_plot <- plot_performance_memory(clients)
saved_path <- save_grid_png(
  function() print(memory_plot),
  output_path = memory_output_path,
  width = memory_width,
  height = memory_height,
  dpi = figure_dpi
)
cat("Saved the memory figure to", saved_path, "\n")
