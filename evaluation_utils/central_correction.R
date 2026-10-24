# Central limma correction for standard datasets and prepared experiments.

central_batch_correction <- function(expr, batch, design, batch_levels = NULL) {
  expr <- as.matrix(expr)
  batch <- if (is.null(batch_levels)) as.factor(batch) else factor(batch, levels = batch_levels)
  if (qr(cbind(design, model.matrix(~ batch)[, -1, drop = FALSE]))$rank !=
      ncol(design) + nlevels(batch) - 1L) {
    stop("Correction design is rank deficient")
  }
  limma::removeBatchEffect(expr, batch = batch, design = design)
}

# Run the same correction on prepared files when invoked with Rscript.
# run_correction_experiments.py validates the corrected output.
if (sys.nframe() == 0L) {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) != 4L) {
    stop("Expected input, design, batch and output TSV")
  }
  read_matrix <- function(path) {
    as.matrix(read.delim(path, row.names = 1, check.names = FALSE))
  }
  expr <- read_matrix(args[1])
  design <- read_matrix(args[2])
  batch_table <- read.delim(args[3], check.names = FALSE,
                           colClasses = c("character", "character", "integer"))
  if (!identical(colnames(expr), rownames(design)) ||
      !identical(colnames(expr), batch_table$file)) {
    stop("Input, design, and batch sample order must match")
  }
  batch_levels <- unique(batch_table$batch[order(batch_table$batch_order)])
  corrected <- central_batch_correction(expr, batch_table$batch, design, batch_levels)
  output <- data.frame(rowname = rownames(corrected), corrected, check.names = FALSE)
  names(output)[1] <- names(read.delim(args[1], nrows = 0, check.names = FALSE))[1]
  write.table(output, args[4], sep = "\t", quote = FALSE, row.names = FALSE, na = "NA")
}
