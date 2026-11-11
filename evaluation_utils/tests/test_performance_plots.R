# Run from the repository root: pixi run Rscript evaluation_utils/tests/test_performance_plots.R
# Build the performance figures from a synthetic per client table.
source("evaluation_utils/figure_aggregate_utils.R")

set.seed(3)
grid <- expand.grid(
  dataset = c("E. coli", "Simulated Balanced"), arm = c("label_aware", "no_label"),
  smpc = c(FALSE, TRUE), client = c("lab1", "lab2", "lab3"), stringsAsFactors = FALSE
)
clients <- data.frame(
  grid,
  coordinator = grid$client == "lab1",
  run_duration = 60 + 20 * grid$smpc + stats::runif(nrow(grid), 0, 5),
  client_compute = stats::runif(nrow(grid), 1, 3),
  n_samples = ifelse(grid$dataset == "E. coli", 50, 200),
  n_features = ifelse(grid$dataset == "E. coli", 2000, 6000),
  app_peak_rss_bytes = stats::runif(nrow(grid), 1.5e8, 2.5e8),
  container_mem_peak_bytes = stats::runif(nrow(grid), 3e8, 5e8),
  stringsAsFactors = FALSE
)
clients$aggregation <- ifelse(clients$coordinator, stats::runif(nrow(clients), 2, 4), 0)
clients$waiting <- clients$run_duration - clients$client_compute - clients$aggregation

# The data sets are ordered as in the paper, not alphabetically.
stopifnot(identical(performance_dataset_levels(clients$dataset),
                    c("Simulated Balanced", "E. coli")))

# Every stacked coordinator bar reaches the median coordinator runtime, its parts
# are the medians of the coordinator's computation and aggregation.
prepared <- prepare_performance_clients(clients)
stack <- performance_coordinator_stack(prepared)
stopifnot(nrow(stack) == 2 * 2 * 3)
for (key in unique(paste(stack$dataset, stack$SMPC))) {
  parts <- stack[paste(stack$dataset, stack$SMPC) == key, ]
  runs <- prepared[prepared$coordinator & paste(prepared$dataset, prepared$SMPC) == key, ]
  stopifnot(
    isTRUE(all.equal(max(parts$ymax), stats::median(runs$run_duration))),
    isTRUE(all.equal(parts$ymax[parts$Part == "Coordinator: client computation"],
                     stats::median(runs$client_compute))),
    all(parts$ymax >= parts$ymin)
  )
}

# Both figures build and save, with memory values that were not recorded skipped.
clients$container_mem_peak_bytes[1] <- NA
output_dir <- tempfile("performance_plots")
runtime_plot <- plot_performance_runtime(clients)
memory_plot <- suppressMessages(plot_performance_memory(clients))
stopifnot(inherits(runtime_plot, "ggplot"), inherits(memory_plot, "ggplot"))
for (item in list(list(runtime_plot, "runtime.png"), list(memory_plot, "memory.png"))) {
  path <- save_grid_png(function() print(item[[1]]), file.path(output_dir, item[[2]]),
                        width = 7.5, height = 4, dpi = 100)
  stopifnot(file.size(path) > 0)
}
memory_data <- ggplot2::ggplot_build(memory_plot)$data[[1]]
stopifnot(nrow(memory_data) == 2 * nrow(clients) - 1)

# A table without a required column is rejected.
missing_column <- tryCatch(
  plot_performance_runtime(clients[, setdiff(names(clients), "waiting")]),
  error = function(error) conditionMessage(error)
)
stopifnot(grepl("waiting", missing_column))

cat("Performance plot tests passed\n")
