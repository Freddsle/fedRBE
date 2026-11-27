#!/usr/bin/env Rscript
# Five-client correction scores for the reported tables. Run from the repository root.
# Rscript evaluation_data/simulated/five_client_diagnostics.R ROOTS EXPORTS UPSTREAM NEW_DIR [ROOT ...]
# ROOTS contains rootNNN_inputs.rds. EXPORTS contains <dataset>/inputs from the
# six-observation no-label campaign. UPSTREAM is Distributed-ComBat commit 45b5a25.

exporter <- new.env()
source("evaluation_data/simulated/export_downstream_inputs.R", local = exporter)
slopes <- new.env()
source("evaluation_data/simulated/missing_fraction_slopes.R", local = slopes)
generator <- exporter$missingness
simulation <- exporter$simulation

read_matrix <- function(directory, state) {
  path <- file.path(directory, paste0(state, ".tsv.gz"))
  as.matrix(read.delim(path, check.names = FALSE, row.names = 1))
}

# Fit the no-label model on its own six-observation inputs (already exported).
# Score it only on rows and individual lab measurements also retained with labels.
on_common_cells <- function(corrected, common) {
  stopifnot(identical(colnames(common), colnames(corrected)),
            all(rownames(common) %in% rownames(corrected)))
  y <- corrected[rownames(common), , drop = FALSE]
  y[is.na(common)] <- NA_real_
  stopifnot(identical(is.na(y), is.na(common)))
  y
}

# The current export records the root and the checksums of the matrices it wrote.
# A corrected matrix with the right dimensions can still belong to a different fit.
check_export <- function(directory, root, panel, target, root_hash) {
  record <- jsonlite::fromJSON(file.path(directory, "validation.json"), simplifyVector = FALSE)
  stopifnot(identical(as.numeric(record$root), as.numeric(root)), identical(record$panel, panel),
            identical(as.numeric(record$target), as.numeric(target)),
            identical(record$saved_input_sha256, root_hash),
            identical(record$feature_mask_policy, "observed_ge6_clients_ge3_native"))
  files <- c("uncorrected.tsv.gz", "no_label.tsv.gz", "impute_uncorrected.tsv.gz")
  expected <- vapply(files, function(name) record$output_md5[[name]], character(1))
  if (!identical(unname(tools::md5sum(file.path(directory, files))), unname(expected))) {
    stop("Exported matrix checksum mismatch in ", directory)
  }
}

score_dataset <- function(d, panel, target, exports, upstream, root_hash) {
  root <- d$root
  dataset <- sprintf("missingness_%s_%02d_run%03d", panel, round(100 * target), root)
  directory <- file.path(exports, dataset, "inputs")
  check_export(directory, root, panel, target, root_hash)
  x <- generator$panel_data(generator$mask_root(d, target), d, panel)
  keep <- read_matrix(directory, "uncorrected")
  corrected <- read_matrix(directory, "no_label")
  impute <- read_matrix(directory, "impute_uncorrected")
  meta <- d$meta
  native_input <- function(covariates) {
    suppressMessages(simulation$filter_fedrbe_client(x, meta, "file", "lab",
                     num_batches = length(unique(meta$lab)), covariates = covariates))
  }
  native <- native_input(character())
  common <- native_input("condition")
  stopifnot(identical(colnames(keep), meta$file), identical(colnames(impute), meta$file),
            setequal(rownames(keep), rownames(native)), all(is.finite(impute)),
            identical(dimnames(keep), dimnames(corrected)),
            identical(is.na(keep), is.na(corrected)),
            isTRUE(all.equal(keep, native[rownames(keep), , drop = FALSE], tolerance = 1e-12)))
  common_corrected <- on_common_cells(corrected, common)
  drop <- keep[rowSums(is.na(keep)) == 0L, , drop = FALSE]
  # Verify that the impute export uses this root, its six-observation filter and
  # sample minima, including cells outside a feature's designed panel.
  eligible <- generator$downstream_panel(x, meta)$table
  expected <- x[eligible$impute_retained, , drop = FALSE]
  stopifnot(setequal(rownames(expected), rownames(impute)))
  expected <- expected[rownames(impute), , drop = FALSE]
  missing <- is.na(expected)
  expected[missing] <- apply(x, 2, min, na.rm = TRUE)[col(expected)[missing]]
  stopifnot(isTRUE(all.equal(impute, expected, tolerance = 1e-12)))

  truth <- function(y) d$truth[rownames(y), colnames(y), drop = FALSE]
  score <- function(routes, observed) {
    cbind(root, panel, target, exporter$correction_diagnostics(routes, observed, d))
  }
  references <- function(setting, routes, observed) {
    value <- score(routes, observed)
    data.frame(value[1:3], setting, value[-(1:3)], check.names = FALSE)
  }
  reference <- rbind(
    references("no_label", list(keep = list(uncorrected = native, masked_truth = truth(native))), native),
    references("label_aware", list(keep = list(uncorrected = common, masked_truth = truth(common))), common),
    references("matched", list(keep = list(uncorrected = common, masked_truth = truth(common),
                                          fedrbe = common_corrected)), common),
    references("matched", list(drop = list(uncorrected = drop, masked_truth = truth(drop)),
                               impute = list(uncorrected = impute, masked_truth = truth(impute))), x))

  mod <- cbind(condition = as.numeric(meta$condition == "B"))
  rownames(mod) <- meta$file
  with_condition <- list(
    distributedCombat_site = function(dat, batch, ...) {
      upstream$distributedCombat_site(dat, batch, mod = mod[colnames(dat), , drop = FALSE], ...)
    },
    distributedCombat_central = upstream$distributedCombat_central)
  batch <- factor(meta$lab, levels = sort(unique(meta$lab)))
  checks <- list()
  routes <- list()
  raw <- list(drop = drop, impute = impute)
  for (route in names(raw)) {
    m <- raw[[route]]
    fed <- simulation$correct_both(m, meta, "with_condition")
    dc <- exporter$dcombat_correct(m, meta$lab, with_condition)
    central <- upstream$neuroCombat(m, batch, mod = mod, verbose = FALSE)$dat.combat
    stopifnot(setequal(rownames(fed$input), rownames(m)), identical(dimnames(dc), dimnames(central)))
    check <- data.frame(root, panel, target, route, features = nrow(m),
                        fedrbe_minus_limma = fed$differences[["max_diff"]],
                        dcombat_distributed_minus_central = max(abs(dc - central)))
    stopifnot(check$fedrbe_minus_limma <= generator$campaign$tolerance,
              check$dcombat_distributed_minus_central <= generator$campaign$tolerance)
    checks[[route]] <- check
    routes[[route]] <- list(uncorrected = m,
                            fedrbe_label_aware = fed$corrected[rownames(m), meta$file, drop = FALSE],
                            dcombat_label_aware = dc)
  }
  list(reference = reference, diagnostics = score(routes, x), checks = do.call(rbind, checks),
       fit = if (target > 0) list(input = common, corrected = common_corrected))
}

main <- function(args) {
  if (length(args) < 4L) stop("Expected ROOTS EXPORTS UPSTREAM NEW_DIR [ROOT ...]")
  requested <- args[-(1:4)]
  if (any(!grepl("^[0-9]+$", requested))) stop("Supply integer root identifiers from 1 to 10")
  roots <- if (length(requested)) as.integer(requested) else 1:10
  stopifnot(!anyNA(roots), all(roots %in% 1:10), !anyDuplicated(roots))
  out <- args[4]
  if (file.exists(out) || !dir.exists(dirname(out))) stop("Supply a new directory under an existing parent")
  upstream <- exporter$load_dcombat(args[3])
  tables <- list(reference_scores = list(), label_aware_diagnostics = list(),
                 label_aware_agreement = list(), common_cell_slopes = list())
  for (root in roots) {
    path <- file.path(args[1], sprintf("root%03d_inputs.rds", root))
    d <- readRDS(path)
    stopifnot(identical(d, generator$make_root(root)))
    root_hash <- strsplit(system2("sha256sum", shQuote(path), stdout = TRUE), " +")[[1]][1]
    for (panel in generator$campaign$panels) {
      fits <- list()
      for (target in generator$campaign$targets) {
        name <- paste(root, panel, target)
        result <- score_dataset(d, panel, target, args[2], upstream, root_hash)
        tables$reference_scores[[name]] <- result$reference
        tables$label_aware_diagnostics[[name]] <- result$diagnostics
        tables$label_aware_agreement[[name]] <- result$checks
        if (target > 0) fits[[as.character(target)]] <- result$fit
      }
      grouped <- slopes$slopes_by_missing_fraction(fits, d, if (panel == "overlap") d$panel)
      grouped[c("masked_truth", "corrected")] <- round(grouped[c("masked_truth", "corrected")], 6)
      tables$common_cell_slopes[[paste(root, panel)]] <- cbind(
        design = paste0("five_client_", panel), dataset = root, arm = "without_condition", grouped)
    }
    message("Scored root ", root)
  }
  staging <- tempfile(pattern = ".diagnostics-", tmpdir = dirname(out))
  stopifnot(dir.create(staging))
  on.exit(unlink(staging, recursive = TRUE), add = TRUE)
  for (name in names(tables)) {
    write.table(do.call(rbind, tables[[name]]), file.path(staging, paste0("informative_", name, ".tsv")),
                sep = "\t", row.names = FALSE, quote = FALSE,
                na = if (name == "common_cell_slopes") "NA" else "")
  }
  if (file.exists(out) || !file.rename(staging, out)) stop("Could not publish output directory: ", out)
}

if (sys.nframe() == 0L) main(commandArgs(trailingOnly = TRUE))
