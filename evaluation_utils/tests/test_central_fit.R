# central_batch_fit reproduces limma::removeBatchEffect and returns its batch
# coefficients. Run from the repository root:
#   pixi run Rscript evaluation_utils/tests/test_central_fit.R

source(file.path("evaluation_utils", "central_correction.R"))

set.seed(7)
samples <- paste0("s", 1:30)
batch <- rep(c("lab_b", "lab_a|x", "lab_a|y"), each = 10)
label <- rep(c(0, 1), 15)
expr <- matrix(rnorm(5 * 30), nrow = 5, dimnames = list(paste0("f", 1:5), samples))
expr <- expr + outer(rep(1, 5), rep(c(2, -1, 3), each = 10)) + outer(1:5, label)
expr[2, c(3, 17)] <- NA          # scattered missing values
expr[4, batch == "lab_a|y"] <- NA  # a batch without values of a feature
levels <- c("lab_a|x", "lab_a|y", "lab_b")

for (with_label in c(FALSE, TRUE)) {
  design <- if (with_label) cbind(intercept = 1, label = label) else cbind(intercept = rep(1, 30))
  result <- central_batch_fit(expr, batch, design, levels)
  expected <- limma::removeBatchEffect(expr, batch = factor(batch, levels = levels),
                                       design = design)
  stopifnot(
    isTRUE(all.equal(result$corrected, expected, tolerance = 1e-12)),
    identical(colnames(result$coefficients), c("lab_a|x", "lab_a|y")),
    identical(rownames(result$coefficients), rownames(expr)),
    !anyNA(result$coefficients),
    # the batch without values of f4 gets no coefficient
    result$coefficients["f4", "lab_a|y"] == 0
  )
  # the coefficients reproduce the correction: -1 in all columns for the reference
  coding <- t(sapply(batch, function(b) {
    if (b == "lab_b") c(-1, -1) else as.numeric(c("lab_a|x", "lab_a|y") == b)
  }))
  stopifnot(isTRUE(all.equal(expr - result$coefficients %*% t(coding), result$corrected,
                             tolerance = 1e-12, check.attributes = FALSE)))
}
cat("central_batch_fit reproduces removeBatchEffect and its coefficients\n")
