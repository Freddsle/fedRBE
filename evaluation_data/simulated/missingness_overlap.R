#!/usr/bin/env Rscript
# Five-client panel overlap and abundance-dependent missingness; see README.md.
campaign_script <- if (sys.nframe() == 0L) {
  sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE))
} else {
  source_paths <- lapply(sys.frames(), function(frame) frame$ofile)
  tail(Filter(Negate(is.null), source_paths), 1L)[[1L]]
}
campaign_script <- normalizePath(campaign_script)
shared <- new.env()
source(file.path(dirname(campaign_script), "robustness_simulation.R"), local = shared)
campaign_source_md5 <- unname(tools::md5sum(campaign_script))

campaign <- list(seed = 20261110L, roots = 10L, clients = 5L, samples = 600L,
                 features = 6000L, targets = c(0, 0.1, 0.3, 0.5),
                 panels = c("full", "overlap"), tolerance = 1e-8,
                 # Detection probability (Li & Smyth 2023): logit P = b0_lab + b1 * y with
                 # b1 times the median within-feature SD of the InvGamma(2, 3) variances
                 # equal to 0.5; lab missing shares are 0.6-1.4 times the target.
                 within_sd = sqrt(3 / qgamma(0.5, 2)), detection_slope_sd = 0.5,
                 lab_rate = setNames(c(0.6, 0.8, 1, 1.2, 1.4), paste0("batch", 1:5)),
                 spread_ratio = 10,
                 mean_additive = setNames(c(0, 0.7, -1.5, 1.2, -0.7), paste0("batch", 1:5)),
                 sd_additive = setNames(c(0.5, 1, 1.5, 0.75, 1.25), paste0("batch", 1:5)),
                 shape = setNames(c(3, 1, 5, 2, 4), paste0("batch", 1:5)),
                 mult_scale = 1, noise_sd = 0.5, min_pc1_var = 0.2,
                 # On/off features per class and their off-class down-shift in
                 # within-feature SDs; the other A/B features carry no class signal.
                 onoff = 50L, onoff_shift_sd = 1)

campaign_options <- function(args) {
  flags <- args[seq_along(args) %% 2L == 1L]
  values <- args[seq_along(args) %% 2L == 0L]
  if (length(args) %% 2L || any(!flags %in% c("--roots", "--out")) ||
      anyDuplicated(flags) || any(!nzchar(args)) || any(startsWith(values, "--"))) {
    stop("Use --roots 1 or --roots 1,2,...,10 and --out DIR; one process per directory.")
  }
  option <- function(flag, default) {
    i <- match(flag, flags)
    if (is.na(i)) default else values[i]
  }
  roots <- option("--roots", "1")
  if (!grepl("^[1-9][0-9]*(,[1-9][0-9]*)*$", roots)) stop("Invalid --roots")
  roots <- suppressWarnings(as.integer(strsplit(roots, ",", fixed = TRUE)[[1]]))
  if (anyNA(roots) || any(roots > campaign$roots)) stop("--roots must be within 1:10")
  list(roots = unique(roots), out = option("--out",
       file.path(dirname(campaign_script), "missingness_overlap", "validation")))
}

# Complementary A/B rounding gives exact overall fractions; 1125 is not divisible by 10.
make_panel <- function(role, labs) {
  allocation <- rbind(A_DE = c(112, 225, 225, 225, 338),
                      B_DE = c(113, 225, 225, 225, 337),
                      latent = c(15, 30, 30, 30, 45),
                      null = c(360, 720, 720, 720, 1080))
  stopifnot(length(labs) == 5L,
            identical(as.integer(table(factor(role, levels = rownames(allocation)))),
                      as.integer(rowSums(allocation))))
  presence <- integer(length(role))
  for (r in rownames(allocation)) {
    rows <- which(role == r)
    presence[sample(rows)] <- rep(1:5, allocation[r, ])
  }
  panel <- matrix(FALSE, length(role), length(labs), dimnames = list(NULL, labs))
  for (i in seq_along(role)) panel[i, sample(seq_along(labs), presence[i])] <- TRUE
  panel
}

# The design changes a generated root after its draws: the first `onoff` A and B features
# are shifted down in their off class, and the other A/B features take their values
# without class signal (`null`) and count as null. Batch component, panel, detection draws
# and lab intercepts stay those of the generated root, so unchanged features keep their
# masks.
informative_root <- function(d, null) {
  in_b <- d$meta$condition == "B"
  de <- d$role %in% c("A_DE", "B_DE")
  # The class signal lies only in the on-class cells of A/B features.
  stopifnot(identical(null[!de, ], d$truth[!de, ]),
            identical(null[d$role == "A_DE", in_b], d$truth[d$role == "A_DE", in_b]),
            identical(null[d$role == "B_DE", !in_b], d$truth[d$role == "B_DE", !in_b]))
  onoff <- de & ave(seq_along(d$role), d$role, FUN = seq_along) <= campaign$onoff
  change <- matrix(0, nrow(null), ncol(null))
  change[de & !onoff, ] <- (d$truth - null)[de & !onoff, ]
  shift <- campaign$onoff_shift_sd * campaign$within_sd
  change[onoff & d$role == "A_DE", in_b] <- shift
  change[onoff & d$role == "B_DE", !in_b] <- shift
  targets <- campaign$targets[campaign$targets > 0]
  d$intercept <- t(sapply(targets, function(target) lab_intercepts(d$observed, d$meta, target)))
  rownames(d$intercept) <- targets
  d$truth <- d$truth - change
  d$observed <- d$observed - change
  d$role[de & !onoff] <- "null"
  d
}

make_root <- function(root) {
  seed <- campaign$seed + 1000L * root
  simulate <- function(mu_1) {
    set.seed(seed)
    meta <- data.frame(file = paste0("s.", seq_len(campaign$samples)))
    rownames(meta) <- meta$file
    meta <- shared$generate_metadata(meta, "balanced", num_batches = campaign$clients)
    list(meta = meta, truth = shared$generate_data(0.5, 0.5, 0.45, 0.15, mu_1 = mu_1,
           mu_4 = 1.25, batch_info = meta, m = campaign$samples))
  }
  base <- simulate(1.45)
  meta <- base$meta
  truth <- base$truth
  role <- attr(truth, "feature_role")
  meta$latent <- meta$file %in% attr(truth, "latent_samples")
  meta$lab <- as.character(meta$batch)
  truth <- matrix(truth, nrow(truth), dimnames = list(paste0("prt", seq_len(nrow(truth))),
                                                    meta$file))
  observed <- shared$add_batch_effect(truth, meta, mean_additive = campaign$mean_additive,
      sd_additive = campaign$sd_additive, shape_multiplicative = campaign$shape,
      mult_scale = campaign$mult_scale, noise_sd = campaign$noise_sd,
      min_pc1_var = campaign$min_pc1_var)
  stopifnot(identical(dim(truth), c(6000L, 600L)), all(table(meta$lab, meta$condition) == 60),
            all(table(meta$lab[meta$latent]) == 36))
  labs <- sort(unique(meta$lab))
  set.seed(seed + 1L)
  panel <- make_panel(role, labs)
  rownames(panel) <- rownames(truth)
  set.seed(seed + 2L)
  uniform <- matrix(runif(length(observed)), nrow(observed), dimnames = dimnames(observed))
  # Feature abundances span spread_ratio within-feature SDs, as in label-free proteomics
  # (base generator: 0.75). Correction, contrasts, DE, RF and scaled k-means ignore
  # per-feature constants of observed values, so the offset changes the detection mask
  # and, in the impute route, how far the filling sample minimum lies below a feature.
  set.seed(seed + 3L)
  offset <- rnorm(nrow(truth), 0, sqrt((campaign$spread_ratio * campaign$within_sd)^2 - 1))
  d <- list(root = root, seed = seed, variant = "informative", meta = meta,
            truth = truth + offset, observed = observed + offset, role = role, panel = panel,
            uniform = uniform)
  informative_root(d, matrix(simulate(0)$truth, nrow(truth), dimnames = dimnames(truth)) +
                        offset)
}

# One uniform draw per cell gives identical masks for shared cells in both panels and
# nested masks across levels. Each lab's intercept sets its expected missing share over
# all its cells; designed absence ignores intensity, so the within-panel share is the same
# in expectation. The intercepts are those of the generated root before its change.
lab_intercepts <- function(observed, meta, target) {
  slope <- campaign$detection_slope_sd / campaign$within_sd
  sapply(names(campaign$lab_rate), function(lab) {
    eta <- slope * observed[, meta$lab == lab]
    rate <- target * campaign$lab_rate[[lab]]
    uniroot(function(b) mean(plogis(b + eta, lower.tail = FALSE)) - rate,
            c(-100, 100), tol = 1e-10)$root
  })
}

mask_root <- function(d, target) {
  stopifnot(target %in% campaign$targets)
  x <- d$observed
  if (target == 0) return(x)
  slope <- campaign$detection_slope_sd / campaign$within_sd
  intercept <- if (is.null(d$intercept)) lab_intercepts(x, d$meta, target)
               else d$intercept[as.character(target), ]
  for (lab in names(intercept)) {
    cols <- d$meta$lab == lab
    x[, cols][d$uniform[, cols] >= plogis(intercept[[lab]] + slope * x[, cols])] <- NA
  }
  x
}

panel_data <- function(masked, d, panel) {
  stopifnot(panel %in% campaign$panels)
  if (panel == "full") return(masked)
  for (lab in colnames(d$panel)) masked[!d$panel[, lab], d$meta$lab == lab] <- NA
  masked
}

# The keep route is the no-label correction's own filter, from observation counts only,
# independently of condition: the native minimum in at least three clients, with the
# cells of clients below it masked.
downstream_panel <- function(x, meta) {
  labs <- sort(unique(meta$lab))
  stopifnot(length(labs) == 5L, identical(colnames(x), meta$file),
            !anyDuplicated(rownames(x)), !anyDuplicated(meta$file),
            !any(is.infinite(x)))
  counts <- sapply(labs, function(lab) rowSums(!is.na(x[, meta$lab == lab, drop = FALSE])))
  eligible <- counts >= shared$fedrbe_min_samples(length(labs))
  retained <- rowSums(eligible) >= 3L
  matched <- x
  for (lab in labs) matched[!eligible[, lab], meta$lab == lab] <- NA_real_
  complete <- retained & rowSums(is.na(matched)) == 0L
  # A complete-data correction needs every client's values, so imputation requires the
  # minimum in each client.
  table <- data.frame(feature = rownames(x), matched_retained = retained,
                      matched_clients = rowSums(eligible), complete_case = complete,
                      impute_retained = rowSums(eligible) == length(labs),
                      matched_blanked_cells = rowSums(ifelse(eligible, 0L, counts)))
  matched <- matched[sort(rownames(matched)[retained]), , drop = FALSE]
  list(input = matched, table = table)
}

downstream_eligibility_counts <- function(table, roles) {
  stopifnot(nrow(table) == length(roles))
  do.call(rbind, lapply(c("all", sort(unique(roles))), function(role) {
    rows <- if (role == "all") rep(TRUE, length(roles)) else roles == role
    data.frame(role, total_features = sum(rows),
      rf_features = sum(table$matched_retained[rows]),
      kmeans_features = sum(table$complete_case[rows]),
      impute_features = sum(table$impute_retained[rows]),
      matched_blanked_cells = sum(table$matched_blanked_cells[rows]))
  }))
}

downstream_inputs <- function(d, target, panel) {
  x <- panel_data(mask_root(d, target), d, panel)
  matched <- downstream_panel(x, d$meta)
  if (!nrow(matched$input)) stop("No matched features for downstream learning")
  if (!any(matched$table$impute_retained)) stop("No features for the impute route")
  redacted_meta <- d$meta
  redacted_meta$condition <- rep("redacted", nrow(redacted_meta))
  # Learning inputs use no-label correction only: with missing values, label-aware
  # correction of a held-out client depends on its own labels.
  native <- shared$correct_both(x, redacted_meta, "without_condition")
  features <- rownames(matched$input)
  masks <- is.na(matched$input)
  stopifnot(all(features %in% rownames(native$input)),
            identical(is.na(native$input), is.na(native$corrected)),
            all(is.finite(native$corrected[!is.na(native$corrected)])),
            native$differences[["max_diff"]] <= campaign$tolerance)
  matrices <- list(truth = d$truth[features, , drop = FALSE], uncorrected = matched$input,
                   no_label = native$corrected[features, d$meta$file, drop = FALSE],
                   no_label_central = native$central[features, d$meta$file, drop = FALSE])
  matched$table$no_label_native_retained <- matched$table$feature %in% rownames(native$input)
  matched$table$no_label_native_observed_cells <-
    rowSums(!is.na(native$input))[match(matched$table$feature, rownames(native$input))]
  for (state in names(matrices)) {
    matrices[[state]][masks] <- NA_real_
    stopifnot(identical(dimnames(matrices[[state]]), dimnames(matched$input)),
              identical(is.na(matrices[[state]]), masks),
              all(is.finite(matrices[[state]][!masks])))
  }
  # Impute route (MinDet): a missing cell takes the lowest observed value of its sample,
  # which needs no other client; the complete matrix then takes the same correction path.
  imputed <- x[matched$table$impute_retained, , drop = FALSE]
  gaps <- is.na(imputed)
  minimum <- apply(x, 2, min, na.rm = TRUE)
  imputed[gaps] <- minimum[col(imputed)[gaps]]
  filled <- shared$correct_both(imputed, redacted_meta, "without_condition")
  impute <- list(uncorrected = imputed,
                 no_label = filled$corrected[rownames(imputed), d$meta$file, drop = FALSE],
                 no_label_central = filled$central[rownames(imputed), d$meta$file, drop = FALSE])
  stopifnot(setequal(rownames(filled$input), rownames(imputed)),
            all(is.finite(impute$no_label)), all(is.finite(impute$no_label_central)),
            filled$differences[["max_diff"]] <= campaign$tolerance)
  matched$table$role <- d$role[match(matched$table$feature, rownames(d$truth))]
  list(matrices = matrices, impute = impute, imputed_cells = sum(gaps),
       sample_minimum = minimum, eligibility = matched$table,
       corrections = list(without_condition = native, impute_without_condition = filled))
}

# Per feature, the share of the condition indicator's variance that lies between labs
# among the cells of x: shared$eta_squared on each feature's own cells. On the cells a
# correction fits, no-label correction keeps 1 - eta2_obs of the within-lab condition
# contrast.
feature_eta2 <- function(x, meta) {
  observed <- t(!is.na(x))
  n <- t(rowsum(observed * 1, meta$lab))
  b <- t(rowsum(observed * (meta$condition == "B"), meta$lab))
  share <- rowSums(b) / rowSums(n)
  rowSums(n * (b / n - share)^2, na.rm = TRUE) / (rowSums(n) * share * (1 - share))
}

# The identity is checked on features whose fitted cells agree in both arms.
eta2_identity <- function(no_label, label_aware, meta) {
  features <- intersect(rownames(no_label$input), rownames(label_aware$input))
  agree <- rowSums(is.na(no_label$input[features, , drop = FALSE]) !=
                     is.na(label_aware$input[features, , drop = FALSE])) == 0L
  features <- features[agree]
  in_b <- meta$condition == "B"
  contrast <- function(m) {
    rowMeans(m[features, in_b, drop = FALSE], na.rm = TRUE) -
      rowMeans(m[features, !in_b, drop = FALSE], na.rm = TRUE)
  }
  retained <- 1 - feature_eta2(no_label$input[features, , drop = FALSE], meta)
  deviation <- abs(contrast(no_label$corrected) - contrast(label_aware$corrected) * retained)
  ok <- is.finite(deviation)
  c(features = sum(ok), max_deviation = if (any(ok)) max(deviation[ok]) else NA_real_)
}

# Check the effective present-client model, not the globally absent batch columns.
# A globally constant condition is omitted, matching production's covariate rule.
eligibility_audit <- function(x, meta, arm, designed_presence) {
  labs <- sort(unique(meta$lab))
  covariates <- if (arm == "with_condition") "condition" else character()
  threshold <- shared$fedrbe_min_samples(length(labs), covariates)
  min_clients <- eval(formals(shared$filter_fedrbe_client)$min_clients)
  counts <- sapply(labs, function(lab) rowSums(!is.na(x[, meta$lab == lab, drop = FALSE])))
  effective <- rowSums(counts >= threshold)
  blanked <- rowSums(ifelse(counts < threshold, counts, 0L))
  input <- suppressMessages(shared$filter_fedrbe_client(x, meta, "file", "lab",
                              num_batches = length(labs), covariates = covariates))
  retained <- rownames(x) %in% rownames(input)
  table <- data.frame(feature = rownames(x), designed_presence, effective_presence = effective,
                      retained, threshold, blanked_cells = blanked,
                      eta2_obs = unname(feature_eta2(input, meta)[rownames(x)]),
                      reason = ifelse(designed_presence < min_clients, "designed_panel_loss",
                               ifelse(retained, "retained", "additional_exclusion")),
                      rank = NA_integer_, required_rank = NA_integer_,
                      rank_failure = FALSE, constant_condition = FALSE)
  stopifnot(identical(retained, unname(effective >= min_clients)))
  for (i in which(retained)) {
    observed <- !is.na(input[table$feature[i], ])
    lab <- droplevels(factor(meta$lab[observed]))
    design <- model.matrix(~lab)
    condition <- as.integer(meta$condition[observed] == "B")
    constant <- length(unique(condition)) == 1L
    if (arm == "with_condition" && !constant) design <- cbind(design, condition)
    singular <- svd(crossprod(design), nu = 0, nv = 0)$d
    tolerance <- ncol(design) * singular[1] * .Machine$double.eps
    table$rank[i] <- sum(singular > tolerance)
    table$required_rank[i] <- ncol(design)
    table$rank_failure[i] <- table$rank[i] < ncol(design)
    table$constant_condition[i] <- arm == "with_condition" && constant
  }
  list(input = input, table = table)
}

score_metrics <- function(x, truth, meta, role) {
  values <- shared$recovery_metrics(x, truth, meta, role)
  contrast_valid <- function(group, control) {
    rowSums(!is.na(x[, group, drop = FALSE])) > 0 &
      rowSums(!is.na(x[, control, drop = FALSE])) > 0
  }
  b <- meta$condition == "B"
  c(values, condition_contrast_missing = sum(!contrast_valid(b, !b) &
                                             role %in% c("A_DE", "B_DE")),
    latent_contrast_missing = sum(!contrast_valid(b & meta$latent, b & !meta$latent) &
                                  role == "latent"))
}

# Achieved missing shares of each lab within its designed panel and over all features.
missingness_rates <- function(x, d, panel, target) {
  present <- if (panel == "full") d$panel | TRUE else d$panel
  do.call(rbind, lapply(colnames(present), function(lab) {
    cols <- d$meta$lab == lab
    data.frame(root = d$root, panel, target, lab, panel_features = sum(present[, lab]),
               within_panel_rate = mean(is.na(x[present[, lab], cols, drop = FALSE])),
               total_rate = mean(is.na(x[, cols, drop = FALSE])),
               designed_absence_rate = mean(!present[, lab]))
  }))
}

evaluate_root <- function(d) {
  metrics <- eligibility <- missingness <- status <- learner_eligibility <- list()
  add <- function(panel, target, arm, presence, stage, values) {
    metrics[[length(metrics) + 1L]] <<- data.frame(root = d$root, panel, target, arm,
                   presence = as.character(presence), stage, metric = names(values),
                   value = as.numeric(values))
  }
  for (target in campaign$targets) {
    masked <- mask_root(d, target)
    for (panel in campaign$panels) {
      x <- panel_data(masked, d, panel)
      learner_eligibility[[length(learner_eligibility) + 1L]] <-
        cbind(root = d$root, panel, target,
              downstream_eligibility_counts(downstream_panel(x, d$meta)$table, d$role))
      designed <- if (panel == "full") rep(5L, nrow(x)) else rowSums(d$panel)
      missingness[[length(missingness) + 1L]] <- missingness_rates(x, d, panel, target)
      fits <- list()
      for (arm in shared$arms) {
        started <- Sys.time()
        audit <- eligibility_audit(x, d$meta, arm, designed)
        audit$table$role <- d$role
        eligibility[[length(eligibility) + 1L]] <- cbind(root = d$root, panel, target,
                                                        arm, audit$table)
        input <- audit$input
        groups <- list(condition = c("A_DE", "B_DE"), latent = "latent", null = "null")
        add(panel, target, arm, "all", "eta2_obs", unlist(lapply(names(groups), function(g) {
          v <- audit$table$eta2_obs[d$role %in% groups[[g]] & is.finite(audit$table$eta2_obs)]
          setNames(c(median(v), quantile(v, 0.9, names = FALSE), if (length(v)) max(v) else NA),
                   paste0(c("median_", "p90_", "max_"), g))
        })))
        for (p in c("all", as.character(1:5))) {
          rows <- if (p == "all") rep(TRUE, nrow(x)) else designed == as.integer(p)
          a <- audit$table[rows, ]
          add(panel, target, arm, p, "eligibility", c(total_features = nrow(a),
              retained_features = sum(a$retained),
              retained_fraction = if (nrow(a)) mean(a$retained) else NA_real_,
              designed_panel_losses = sum(a$reason == "designed_panel_loss"),
              additional_exclusions = sum(a$reason == "additional_exclusion"),
              blanked_cells = sum(a$blanked_cells), rank_failures = sum(a$rank_failure)))
        }
        fit <- tryCatch({
          if (!nrow(input)) stop("No eligible features")
          if (any(audit$table$rank_failure)) stop("Rank failure in eligible feature model")
          central <- shared$correct_central(x, d$meta, arm)
          stopifnot(identical(central$input, input))
          federated <- shared$correct_fedrbe(x, d$meta, arm)
          stopifnot(setequal(rownames(federated), rownames(input)),
                    setequal(colnames(federated), colnames(input)))
          federated <- federated[rownames(input), colnames(input), drop = FALSE]
          stopifnot(identical(is.na(federated), is.na(input)),
                    identical(is.na(central$corrected), is.na(input)))
          list(central = central$corrected, fedrbe = federated)
        }, error = identity)
        failed <- inherits(fit, "error")
        detail <- if (failed) conditionMessage(fit) else ""
        if (!failed) fits[[arm]] <- list(input = input, corrected = fit$fedrbe)
        idx <- match(rownames(input), rownames(d$truth))
        for (p in c("all", "3", "4", "5")) {
          rows <- if (p == "all") seq_along(idx) else which(designed[idx] == as.integer(p))
          ref <- d$truth[idx[rows], , drop = FALSE]
          observed <- input[rows, , drop = FALSE]
          roles <- d$role[idx[rows]]
          masked_truth <- ref
          masked_truth[is.na(observed)] <- NA
          add(panel, target, arm, p, "before", score_metrics(observed, ref, d$meta, roles))
          add(panel, target, arm, p, "observed_truth",
              shared$contrast_recovery(masked_truth, ref, d$meta, roles))
          for (method in c("central", "fedrbe")) {
            after <- if (failed) observed * NA_real_ else fit[[method]][rows, , drop = FALSE]
            add(panel, target, arm, p, method, score_metrics(after, ref, d$meta, roles))
            add(panel, target, arm, p, paste0(method, "_vs_complete"),
                shared$contrast_recovery(after, ref, d$meta, roles))
          }
          difference <- if (failed || !length(rows)) c(mean_diff = NA_real_, max_diff = NA_real_)
                        else shared$calculated_differences(fit$central[rows, , drop = FALSE],
                                                           fit$fedrbe[rows, , drop = FALSE])
          add(panel, target, arm, p, "agreement", difference)
        }
        max_diff <- if (failed) NA_real_ else max(abs(fit$central - fit$fedrbe), na.rm = TRUE)
        success <- !failed && is.finite(max_diff) && max_diff <= campaign$tolerance
        if (!failed && !success) detail <- "Numerical agreement exceeds tolerance"
        status[[length(status) + 1L]] <- data.frame(root = d$root, panel, target, arm,
          status = if (success) "ok" else "failed", detail, max_diff,
          retained = nrow(input), rank_failures = sum(audit$table$rank_failure),
          seconds = as.numeric(difftime(Sys.time(), started, units = "secs")))
        cat(sprintf("root %d %s target %.0f%% %s: %s, %d retained, %.1fs\n",
                    d$root, panel, 100 * target, arm, tail(status, 1)[[1]]$status,
                    nrow(input), tail(status, 1)[[1]]$seconds))
      }
      add(panel, target, "without_condition", "all", "eta2_identity",
          if (length(fits) == 2L) {
            eta2_identity(fits$without_condition, fits$with_condition, d$meta)
          } else c(features = 0, max_deviation = NA_real_))
    }
  }
  list(metrics = do.call(rbind, metrics), eligibility = do.call(rbind, eligibility),
       missingness = do.call(rbind, missingness), status = do.call(rbind, status),
       learner_eligibility = do.call(rbind, learner_eligibility))
}

campaign_fingerprint <- function() {
  # Dependency hashes come from the shared simulation driver.
  c(capture.output(dput(campaign)),
    paste("md5", basename(campaign_script), campaign_source_md5),
    grep("^md5 ", shared$fingerprint(), value = TRUE),
    paste("R", getRversion()), paste("limma", packageVersion("limma")))
}

overlap_summary_values <- function(values, deviation = FALSE) {
  result <- shared$summarise_values(values, deviation)
  names(result)[names(result) == "n_runs"] <- "n_finite"
  result$n_roots <- length(values)
  result$sd <- sd(values[is.finite(values)])
  result[c("n_roots", "n_finite", "n_missing", "mean", "median", "sd", "mcse",
           "ci_low", "ci_high", "mean_abs_dev_1")]
}

overlap_group_summary <- function(metrics) {
  keys <- setdiff(names(metrics), c("root", "value"))
  if (!nrow(metrics)) return(data.frame())
  groups <- split(seq_len(nrow(metrics)), interaction(metrics[keys], drop = TRUE))
  result <- lapply(groups, function(rows) {
    key <- metrics[rows[1], keys, drop = FALSE]
    cbind(key, overlap_summary_values(metrics$value[rows],
                                      startsWith(key$metric, "attenuation_")))
  })
  result <- do.call(rbind, result)
  rownames(result) <- NULL
  result
}

summarise_overlap <- function(out) {
  files <- list.files(file.path(out, "runs"), pattern = "^root[0-9]{3}\\.rds$",
                      full.names = TRUE)
  if (!length(files)) stop("No completed roots in ", out)
  input_hashes <- tools::md5sum(files)
  roots <- lapply(files, readRDS)
  metrics <- do.call(rbind, lapply(roots, `[[`, "metrics"))
  expected <- c("root", "panel", "target", "arm", "presence", "stage", "metric", "value")
  if (!identical(names(metrics), expected) ||
      anyNA(metrics[setdiff(expected, "value")]) ||
      anyDuplicated(metrics[setdiff(expected, "value")])) {
    stop("Root metrics must have the documented columns and unique, nonmissing keys")
  }
  metrics <- metrics[do.call(order, metrics[setdiff(expected, "value")]), ]
  summary <- overlap_group_summary(metrics)
  write_tsv <- function(data, name) {
    write.table(data, file.path(out, name), sep = "\t", row.names = FALSE,
                 quote = FALSE, na = "NA")
  }
  write_tsv(metrics, "metrics.tsv")
  write_tsv(summary, "summary.tsv")
  for (name in c("eligibility", "missingness", "learner_eligibility")) {
    pieces <- lapply(roots, `[[`, name)
    pieces <- Filter(function(x) is.data.frame(x) && nrow(x) > 0L, pieces)
    if (!length(pieces)) next
    data <- do.call(rbind, pieces)
    if (name == "eligibility") {
      connection <- gzfile(file.path(out, "eligibility.tsv.gz"), "wt")
      tryCatch(write.table(data, connection, sep = "\t", row.names = FALSE, quote = FALSE),
               finally = close(connection))
    } else write_tsv(data, paste0(name, ".tsv"))
  }
  hashes <- data.frame(kind = c("report_source", rep("root_result", length(files))),
    path = c(basename(campaign_script), file.path("runs", basename(files))),
    md5 = c(campaign_source_md5, unname(input_hashes)))
  numerical_provenance <- file.path(out, "provenance.txt")
  if (file.exists(numerical_provenance)) {
    hashes <- rbind(hashes, data.frame(kind = "numerical_provenance", path = "provenance.txt",
                                      md5 = unname(tools::md5sum(numerical_provenance))))
  }
  writeLines(c(paste("generated_at_utc", format(Sys.time(), tz = "UTC", usetz = TRUE)),
    paste("R", getRversion()),
    capture.output(write.table(hashes, sep = "\t", row.names = FALSE, quote = FALSE))),
    file.path(out, "report_provenance.txt"))
  invisible(summary)
}


campaign_main <- function(args = commandArgs(trailingOnly = TRUE)) {
  opts <- campaign_options(args)
  dir.create(file.path(opts$out, "runs"), recursive = TRUE, showWarnings = FALSE)
  provenance <- file.path(opts$out, "provenance.txt")
  fingerprint <- campaign_fingerprint()
  if (file.exists(provenance)) {
    if (!identical(readLines(provenance, n = length(fingerprint)), fingerprint)) {
      stop("Code or parameters changed; use a new --out directory.")
    }
  } else writeLines(c(fingerprint, "", shared$session_lines()), provenance)
  for (root in opts$roots) {
    output <- file.path(opts$out, "runs", sprintf("root%03d.rds", root))
    if (file.exists(output)) next
    started <- Sys.time()
    d <- make_root(root)
    # Paired inputs shared by all settings of this root.
    saveRDS(d, file.path(opts$out, "runs", sprintf("root%03d_inputs.rds", root)))
    result <- evaluate_root(d)
    result$seconds <- as.numeric(difftime(Sys.time(), started, units = "secs"))
    saveRDS(result, output)
    rm(d)
    gc()
  }
  summarise_overlap(opts$out)
  results <- lapply(list.files(file.path(opts$out, "runs"), "^root[0-9]+[.]rds$",
                              full.names = TRUE), readRDS)
  status <- do.call(rbind, lapply(results, `[[`, "status"))
  write.table(status, file.path(opts$out, "status.tsv"), sep = "\t", quote = FALSE,
              row.names = FALSE, na = "NA")
  runtime <- data.frame(root = vapply(results, function(x) x$metrics$root[1], integer(1)),
                         seconds = vapply(results, `[[`, numeric(1), "seconds"))
  write.table(runtime, file.path(opts$out, "runtime.tsv"), sep = "\t", quote = FALSE,
              row.names = FALSE)
  cat(sprintf("Measured %.1f min/root; ten-root estimate %.1f min (serial, same host).\n",
              mean(runtime$seconds) / 60, mean(runtime$seconds) / 6))
  if (any(status$status != "ok")) stop("Recorded failures; inspect status.tsv and eligibility.tsv.gz")
}

if (sys.nframe() == 0L) campaign_main()
