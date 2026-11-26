#!/usr/bin/env Rscript
# d-ComBat (Chen et al. 2022) and centralized ComBat fitted on training samples only,
# called by training_correction.py:
#   Rscript dcombat_training.R UPSTREAM_DIR INPUT.tsv SAMPLES.tsv FEDERATED.tsv CENTRAL.tsv
# INPUT holds features x samples, SAMPLES the columns file, lab and part (train or test).
# d-ComBat runs its documented protocol on the training samples, each lab a site, for the
# federated learner; the reference code's neuroCombat fits the pooled training samples for
# the central learner. A test sample of a training lab is adjusted with its lab's estimates
# of the respective fit (neuroCombatFromTraining). Both outputs hold all samples of INPUT.
script <- if (sys.nframe() == 0L) {
  sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE))
} else sys.frame(1)$ofile
exporter <- new.env()
source(file.path(dirname(normalizePath(script)), "export_downstream_inputs.R"),
       local = exporter)

combat_training <- function(x, lab, training, upstream) {
  test <- !training
  federated <- central <- x
  sites <- exporter$dcombat_sites(x[, training, drop = FALSE], lab[training], upstream)
  for (site in names(sites)) {
    federated[, colnames(sites[[site]]$dat.combat)] <- sites[[site]]$dat.combat
    own <- test & lab == site
    federated[, own] <- upstream$neuroCombatFromTraining(
      x[, own, drop = FALSE], lab[own], sites[[site]]$estimates, verbose = FALSE)
  }
  fit <- upstream$neuroCombat(x[, training, drop = FALSE],
                              factor(lab[training], levels = names(sites)), verbose = FALSE)
  central[, training] <- fit$dat.combat
  central[, test] <- upstream$neuroCombatFromTraining(x[, test, drop = FALSE], lab[test],
                                                      fit$estimates, verbose = FALSE)
  list(federated = federated, central = central)
}

if (sys.nframe() == 0L) {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) != 5L) {
    stop("Expected UPSTREAM_DIR INPUT.tsv SAMPLES.tsv FEDERATED.tsv CENTRAL.tsv")
  }
  x <- as.matrix(read.delim(args[2], row.names = 1, check.names = FALSE))
  samples <- read.delim(args[3], stringsAsFactors = FALSE)
  stopifnot(identical(colnames(x), samples$file), all(samples$part %in% c("train", "test")))
  corrected <- combat_training(x, samples$lab, samples$part == "train",
                               exporter$load_dcombat(args[1]))
  for (i in 1:2) {
    write.table(data.frame(feature = rownames(x), corrected[[i]], check.names = FALSE),
                args[3 + i], sep = "\t", row.names = FALSE, quote = FALSE, na = "NA")
  }
}
