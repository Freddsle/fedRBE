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

# The fit of limma::removeBatchEffect(expr, batch = batch, design = design), with
# its batch coefficients: the batches are coded as sum-to-zero contrasts, the last
# level is the reference with -1 in every batch column (as in fedRBE). Batch
# coefficients that cannot be estimated (e.g. a batch without values of a
# feature) are 0, as in removeBatchEffect. The coefficients allow correcting new
# samples of the same batches later on.
# Returns a list with the corrected matrix and the coefficients
# (features x batch levels except the last).
central_batch_fit <- function(expr, batch, design, batch_levels = NULL) {
  expr <- as.matrix(expr)
  batch <- if (is.null(batch_levels)) as.factor(batch) else factor(batch, levels = batch_levels)
  if (nlevels(batch) < 2L) {
    stop("Central correction needs at least two batches")
  }
  contrasts(batch) <- contr.sum(levels(batch))
  x_batch <- model.matrix(~batch)[, -1, drop = FALSE]
  colnames(x_batch) <- head(levels(batch), -1L)
  fit <- limma::lmFit(expr, cbind(design, x_batch))
  beta <- fit$coefficients[, -seq_len(ncol(design)), drop = FALSE]
  beta[is.na(beta)] <- 0
  colnames(beta) <- colnames(x_batch)
  list(corrected = expr - beta %*% t(x_batch), coefficients = beta)
}

# Run the same correction on prepared files when invoked with Rscript.
# run_correction_experiments.py validates the corrected output.
# "fit" as the first argument runs central_batch_fit instead and also writes the
# batch coefficients (used by the classification evaluations).
if (sys.nframe() == 0L) {
  args <- commandArgs(trailingOnly = TRUE)
  fit_mode <- length(args) >= 1L && args[1] == "fit"
  if (fit_mode) args <- args[-1]
  if (length(args) != 4L) {
    stop("Expected [fit] input, design, batch and output TSV (output prefix for fit)")
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
  feature_column <- names(read.delim(args[1], nrows = 0, check.names = FALSE))[1]
  write_matrix <- function(matrix, path) {
    output <- data.frame(rowname = rownames(matrix), matrix, check.names = FALSE)
    names(output)[1] <- feature_column
    write.table(output, path, sep = "\t", quote = FALSE, row.names = FALSE, na = "NA")
  }
  if (fit_mode) {
    result <- central_batch_fit(expr, batch_table$batch, design, batch_levels)
    write_matrix(result$corrected, paste0(args[4], "corrected.tsv"))
    write_matrix(result$coefficients, paste0(args[4], "coefficients.tsv"))
  } else {
    write_matrix(central_batch_correction(expr, batch_table$batch, design, batch_levels),
                 args[4])
  }
}
