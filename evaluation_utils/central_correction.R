# Central limma correction for standard datasets and prepared experiments.

central_batch_correction <- function(expr, batch, design, batch_levels = NULL, weights = NULL) {
  expr <- as.matrix(expr)
  batch <- if (is.null(batch_levels)) as.factor(batch) else factor(batch, levels = batch_levels)
  if (qr(cbind(design, model.matrix(~ batch)[, -1, drop = FALSE]))$rank !=
      ncol(design) + nlevels(batch) - 1L) {
    stop("Correction design is rank deficient")
  }
  # lmFit leaves values with weight 0 out of the fit, removeBatchEffect still corrects them
  limma::removeBatchEffect(expr, batch = batch, design = design, weights = weights)
}

# Mirrors fedRBE's local leverage check (exclude_high_leverage in
# batchcorrection/classes/client.py), like filtering.R mirrors min_samples.
# Per client and feature, a value whose leverage in the client's own fit (one
# intercept per batch plus the covariates) exceeds max_leverage gets weight 0:
# it is left out of the fit but still corrected. A batch left with fewer than
# min_samples fitted values is masked, and the check is repeated until stable.
# Features that are then observed in fewer than min_clients clients are masked,
# as the app drops them.
# Args: covariates is samples x covariates; batch and client are given per
# sample, min_samples and max_leverage per sample (constant per client) or once.
# Returns the masked expr and the weights (1 or 0).
fedrbe_leverage_weights <- function(expr, covariates, batch, client, min_samples,
                                    max_leverage = 0.5, min_clients = 3L, tolerance = 1e-8) {
  expr <- as.matrix(expr)
  covariates <- as.matrix(covariates)
  min_samples <- rep_len(min_samples, ncol(expr))
  max_leverage <- rep_len(max_leverage, ncol(expr))
  weights <- matrix(1, nrow(expr), ncol(expr), dimnames = dimnames(expr))
  if (ncol(covariates) == 0L) {
    return(list(expr = expr, weights = weights))
  }
  masked <- rep(FALSE, nrow(expr))
  for (name in unique(client)) {
    cols <- which(client == name)
    threshold <- max_leverage[cols[1]] + tolerance  # h = max_leverage exactly is kept
    if (max_leverage[cols[1]] >= 1) next
    local_batch <- factor(batch[cols])
    indicators <- outer(as.integer(local_batch), seq_len(nlevels(local_batch)), "==") * 1
    x <- cbind(indicators, covariates[cols, , drop = FALSE])
    for (f in seq_len(nrow(expr))) {
      fitted <- !is.na(expr[f, cols])
      repeat {
        for (level in levels(local_batch)) {
          in_batch <- local_batch == level
          count <- sum(fitted & in_batch)
          if (count < min_samples[cols[1]] && any(!is.na(expr[f, cols[in_batch]]))) {
            expr[f, cols[in_batch]] <- NA
            fitted[in_batch] <- FALSE
            masked[f] <- TRUE
          }
        }
        if (!any(fitted)) break
        # stats::hat uses a pivoted QR, aliased covariates are dropped
        high <- stats::hat(x[fitted, , drop = FALSE], intercept = FALSE) > threshold
        if (!any(high)) break
        excluded <- which(fitted)[high]
        weights[f, cols[excluded]] <- 0
        fitted[excluded] <- FALSE
      }
    }
  }
  observed_clients <- Reduce(`+`, lapply(unique(client), function(name) {
    rowSums(!is.na(expr[, client == name, drop = FALSE])) > 0
  }))
  expr[masked & observed_clients < min_clients, ] <- NA
  message("Leverage mirror: ", sum(weights == 0 & !is.na(expr)), " values left out of the fit, ",
          sum(masked), " features with masked batches")
  list(expr = expr, weights = weights)
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
                           colClasses = c(file = "character", batch = "character",
                                          batch_order = "integer", client = "character"))
  if (!identical(colnames(expr), rownames(design)) ||
      !identical(colnames(expr), batch_table$file)) {
    stop("Input, design, and batch sample order must match")
  }
  batch_levels <- unique(batch_table$batch[order(batch_table$batch_order)])
  weights <- NULL
  if ("max_leverage" %in% names(batch_table)) {
    # mirror the app's leverage check, it only applies with covariates
    mirrored <- fedrbe_leverage_weights(
      expr, design[, colnames(design) != "intercept", drop = FALSE], batch_table$batch,
      batch_table$client, batch_table$min_samples, batch_table$max_leverage)
    expr <- mirrored$expr
    weights <- mirrored$weights
  }
  corrected <- central_batch_correction(expr, batch_table$batch, design, batch_levels, weights)
  output <- data.frame(rowname = rownames(corrected), corrected, check.names = FALSE)
  names(output)[1] <- names(read.delim(args[1], nrows = 0, check.names = FALSE))[1]
  write.table(output, args[4], sep = "\t", quote = FALSE, row.names = FALSE, na = "NA")
}
