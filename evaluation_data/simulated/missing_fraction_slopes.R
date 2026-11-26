#!/usr/bin/env Rscript
# Contrast slopes by per-feature missing fraction in the five-client panels; see README.md.
script <- if (sys.nframe() == 0L) {
  sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE))
} else sys.frame(1)$ofile
script <- normalizePath(script)
campaign <- new.env()
source(file.path(dirname(script), "missingness_overlap.R"), local = campaign)
shared <- campaign$shared

# Groups of the missing fraction in percent, closed on the right; the first includes 0.
breaks <- c(0, 5, 10, 20, 30, 50, 100)
groups <- paste(head(breaks, -1), breaks[-1], sep = "-")

# Slopes of the masked truth and of the corrected data against the complete truth, over
# the retained features of `fits` pooled within each group. A feature's missing fraction
# counts the missing cells of the correction input among the labs assigned the feature
# (`present`, features by labs; every lab when NULL).
slopes_by_missing_fraction <- function(fits, d, present = NULL) {
  parts <- lapply(fits, function(fit) {
    kept <- match(rownames(fit$input), rownames(d$truth))
    cells <- if (is.null(present)) ncol(fit$input) else rowSums(present[kept, d$meta$lab])
    # Multiplying the missing count before dividing keeps the group limits exact.
    list(truth = d$truth[kept, , drop = FALSE], corrected = fit$corrected, role = d$role[kept],
         missing = 100 * (cells - rowSums(!is.na(fit$input))) / cells)
  })
  bind <- function(name, combine) do.call(combine, lapply(parts, `[[`, name))
  truth <- bind("truth", rbind)
  corrected <- bind("corrected", rbind)
  role <- bind("role", c)
  group <- cut(bind("missing", c), breaks, groups, include.lowest = TRUE)
  masked <- truth
  masked[is.na(corrected)] <- NA
  do.call(rbind, lapply(c(groups, "all"), function(g) {
    rows <- g == "all" | group == g
    slope <- function(x) {
      shared$contrast_recovery(x[rows, , drop = FALSE], truth[rows, , drop = FALSE], d$meta,
                               role[rows])[c("attenuation_condition", "attenuation_latent")]
    }
    data.frame(missing_percent = g, contrast = c("condition", "latent"),
               features = c(sum(role[rows] %in% c("A_DE", "B_DE")), sum(role[rows] == "latent")),
               masked_truth = slope(masked), corrected = slope(corrected))
  }))
}

# Five-client informative design: the nonzero missingness levels of a root share its
# cohort and nest their masks, so their features are pooled within the root and panel.
five_client <- function(root) {
  d <- campaign$make_root(root)
  targets <- campaign$campaign$targets[campaign$campaign$targets > 0]
  masked <- lapply(targets, function(target) campaign$mask_root(d, target))
  do.call(rbind, lapply(campaign$campaign$panels, function(panel) {
    do.call(rbind, lapply(shared$arms, function(arm) {
      fits <- lapply(masked, function(x) {
        shared$correct_both(campaign$panel_data(x, d, panel), d$meta, arm)
      })
      cbind(design = paste0("five_client_", panel), dataset = root, arm,
            slopes_by_missing_fraction(fits, d, if (panel == "overlap") d$panel))
    }))
  }))
}

main <- function(args = commandArgs(trailingOnly = TRUE)) {
  out <- if (length(args)) args[[1]] else {
    file.path(dirname(script), "results", "missing_fraction_slopes.tsv")
  }
  values <- do.call(rbind, lapply(seq_len(campaign$campaign$roots), five_client))
  numeric <- c("masked_truth", "corrected")
  values[numeric] <- round(values[numeric], 6)
  write.table(values, out, sep = "\t", quote = FALSE, row.names = FALSE)
}

if (sys.nframe() == 0L) main()
