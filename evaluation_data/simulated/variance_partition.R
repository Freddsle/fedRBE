# Variance partitioning of a simulated dataset before and after correction; see
# README.md, "Variance partitioning". Callers supply each correction arm's native input
# and output (as returned by correct_both in robustness_simulation.R) and the batch-free
# truth; the functions return per-feature fractions and dataset-level summaries.

# The interaction model of the variancePartition vignette ("Including interaction terms").
vp_formula <- ~ (1 | lab) + (1 | condition) + (1 | lab:condition)
vp_components <- c(lab = "lab", condition = "condition", interaction = "lab:condition",
                   residual = "Residuals")
vp_states <- c("uncorrected", "no_label", "label_aware", "truth")
# Descriptive threshold for a zero (boundary) component; zeros are valid estimates.
vp_zero <- 1e-8
# variancePartition's lme4 settings (its internal `vpcontrol`).
vp_control <- lme4::lmerControl(calc.derivs = FALSE, check.rankX = "stop.deficient",
                                check.conv.singular = "ignore")
# Starting values of the three relative standard deviations: lme4's default and every
# corner of {0.05, 2}^3. The likelihood can have separate maxima that place a lab shift on
# lab or on lab × condition, so a single start can stop at the lower one.
vp_starts <- c(list(NULL), lapply(1:8, function(i) {
  unname(as.matrix(expand.grid(c(0.05, 2), c(0.05, 2), c(0.05, 2)))[i, ])
}))

# Common features and observed cells of both arms' native inputs, applied after the
# corrections so that neither fit population changes. `native` has elements no_label and
# label_aware, each with the filtered `input` and its `corrected` matrix.
matched_states <- function(native, truth) {
  features <- intersect(rownames(truth),
                        intersect(rownames(native$no_label$input),
                                  rownames(native$label_aware$input)))
  observed <- !is.na(native$no_label$input[features, colnames(truth), drop = FALSE]) &
    !is.na(native$label_aware$input[features, colnames(truth), drop = FALSE])
  mask <- function(x) {
    x <- x[features, colnames(truth), drop = FALSE]
    x[!observed] <- NA_real_
    x
  }
  losses <- unlist(lapply(c("no_label", "label_aware"), function(arm) {
    input <- native[[arm]]$input
    kept <- rownames(input) %in% features
    setNames(c(nrow(input), sum(!kept), sum(!is.na(input)), sum(!is.na(input[!kept, ])),
               sum(!is.na(input[features, colnames(truth), drop = FALSE])) - sum(observed)),
             paste0(c("native_features_", "matching_excluded_features_",
                      "native_observed_cells_", "matching_excluded_cells_",
                      "matching_blanked_cells_"), arm))
  }))
  list(states = list(uncorrected = mask(native$no_label$input),
                     no_label = mask(native$no_label$corrected),
                     label_aware = mask(native$label_aware$corrected),
                     truth = mask(truth)),
       matching = c(losses, common_features = length(features),
                    common_observed_cells = sum(observed)))
}

# Observed levels per feature. lme4 needs two levels of each factor and more
# observations than random effects (labs + conditions + lab-condition cells).
observation_structure <- function(x, meta) {
  levels_seen <- function(group) {
    group <- factor(group)
    seen <- vapply(levels(group), function(level) {
      rowSums(!is.na(x[, group == level, drop = FALSE])) > 0
    }, logical(nrow(x)))
    rowSums(matrix(seen, nrow(x)))
  }
  structure <- data.frame(feature = rownames(x), n_obs = rowSums(!is.na(x)),
                          labs = levels_seen(meta$lab),
                          conditions = levels_seen(meta$condition),
                          cells = levels_seen(paste(meta$lab, meta$condition)))
  structure$crossed <- structure$cells == structure$labs * structure$conditions
  structure$estimable <- structure$labs >= 2L & structure$conditions >= 2L &
    structure$n_obs > structure$labs + structure$conditions + structure$cells
  structure
}

# Maximum-likelihood fit of one feature (column y of `data`) on its observed samples: one
# lme4 fit per start, keeping the highest likelihood. As in variancePartition, a warning
# or a failed model check of calcVarPart() rejects a fit, and near-constant responses
# fail. Each feature is fitted anew: variancePartition 1.40.2 refits a template model,
# which with lme4 2.0.6 can stop away from the optimum. The fits are lmer()'s own steps
# (lme4 2.0.6) with the start-independent set-up done once per feature; they equal
# separate lmer() calls. The default start comes first: it begins at the set-up's values.
fit_feature <- function(data) {
  if (var(data$y, na.rm = TRUE) < 1e-5) stop("response variance < 1e-05")
  # A set-up warning would reject every start.
  setup <- tryCatch({
    lmod <- lme4::lFormula(update(vp_formula, y ~ .), data, REML = FALSE,
                           control = vp_control)[c("fr", "X", "reTrms", "REML")]
    c(lmod, devfun = do.call(lme4::mkLmerDevfun, c(lmod, list(control = vp_control))))
  }, warning = function(w) stop(conditionMessage(w)))
  rho <- environment(setup$devfun)
  fits <- lapply(vp_starts, function(start) tryCatch({
    opt <- lme4::optimizeLmer(setup$devfun, optimizer = vp_control$optimizer,
                              restart_edge = vp_control$restart_edge,
                              boundary.tol = vp_control$boundary.tol,
                              start = if (!is.null(start)) list(theta = start),
                              control = vp_control$optCtrl,
                              calc.derivs = vp_control$calc.derivs,
                              force.calc.derivs = vp_control$calc.derivs,
                              use.last.params = vp_control$use.last.params)
    conv <- lme4::checkConv(attr(opt, "derivs"), opt$par, ctrl = vp_control$checkConv,
                            lbound = rho$lower, ubound = rho$upper, nobs = nrow(setup$fr),
                            ndim = length(rho$lower))
    model <- lme4::mkMerMod(rho, opt, setup$reTrms, fr = setup$fr, lme4conv = conv)
    list(fractions = variancePartition::calcVarPart(model),
         log_likelihood = as.numeric(stats::logLik(model)))
  }, warning = identity, error = identity))
  ok <- !vapply(fits, inherits, logical(1), "condition")
  # Every start rejected: the feature fails with the error of lme4's default start.
  if (!any(ok)) stop(conditionMessage(fits[[1]]))
  fits[ok][[which.max(vapply(fits[ok], `[[`, numeric(1), "log_likelihood"))]]
}

# Per-feature fractions and maximised log-likelihoods; failed features keep the error.
fit_fractions <- function(x, meta) {
  if (!identical(colnames(x), rownames(meta)) || !all(c("lab", "condition") %in% names(meta))) {
    stop("Samples must match metadata rows, and metadata needs lab and condition")
  }
  data <- data.frame(lab = factor(meta$lab), condition = factor(meta$condition),
                     row.names = rownames(meta))
  result <- data.frame(feature = rownames(x), status = "failed", lab = NA_real_,
                       condition = NA_real_, interaction = NA_real_, residual = NA_real_,
                       log_likelihood = NA_real_, error = NA_character_)
  for (i in seq_len(nrow(x))) {
    data$y <- x[i, ]
    fit <- tryCatch(fit_feature(data), error = identity)
    if (inherits(fit, "error")) {
      # One line per message, so tab-separated outputs keep one row per feature.
      result$error[i] <- trimws(gsub("[[:space:]]+", " ", conditionMessage(fit)))
      next
    }
    result[i, names(vp_components)] <- as.list(unname(fit$fractions[vp_components]))
    result$log_likelihood[i] <- fit$log_likelihood
    result$status[i] <- "ok"
  }
  result
}

# Fit the four matched states; return per-feature results and long dataset summaries.
# The primary statistic is the mean of each component over the features fitted in every
# available state (equal feature weights); medians and quartiles are kept alongside.
analyse_dataset <- function(states, meta, role) {
  stopifnot(identical(names(states), vp_states))
  structure <- observation_structure(states$truth, meta)
  structure$role <- role
  seconds <- numeric()
  fits <- lapply(vp_states, function(state) {
    started <- proc.time()[["elapsed"]]
    fit <- fit_fractions(states[[state]], meta)
    seconds[[state]] <<- proc.time()[["elapsed"]] - started
    stopifnot(identical(fit$feature, structure$feature))
    fit
  })
  features <- do.call(rbind, lapply(seq_along(vp_states), function(i) {
    cbind(structure, state = vp_states[i], fits[[i]][-1])
  }))
  ok <- vapply(fits, function(fit) fit$status == "ok", logical(nrow(structure)))
  available <- colSums(ok) > 0L
  valid <- any(available) & rowSums(ok[, available, drop = FALSE]) == sum(available)
  rows <- list()
  add <- function(state, component, statistic, value) {
    rows[[length(rows) + 1L]] <<- data.frame(state, component, statistic, value)
  }
  for (i in seq_along(vp_states)) {
    fit <- fits[[i]]
    state <- vp_states[i]
    used <- fit[valid & available[i], names(vp_components)]
    for (component in names(vp_components)) {
      v <- used[[component]]
      add(state, component, c("mean", "median", "q1", "q3", "n_features"),
          if (length(v)) c(mean(v), quantile(v, c(0.5, 0.25, 0.75), names = FALSE), length(v))
          else c(rep(NA_real_, 4L), 0))
    }
    good <- fit[ok[, i], names(vp_components)]
    add(state, "all", c("available", "fitted_features", "ok_features", "failed_features",
                        "failed_estimable_features", "excluded_by_other_states", "zero_lab",
                        "zero_condition", "zero_interaction", "max_sum_error", "seconds"),
        c(available[i], nrow(fit), sum(ok[, i]), sum(!ok[, i]),
          sum(!ok[, i] & structure$estimable), sum(ok[, i] & !valid),
          colSums(good[c("lab", "condition", "interaction")] < vp_zero),
          if (nrow(good)) max(abs(rowSums(good) - 1)) else NA_real_, seconds[[state]]))
  }
  roles <- sort(unique(role))
  add("matched", "all",
      c("valid_features", "crossed_features", "unestimable_features",
        paste0("common_role_", roles), paste0("valid_role_", roles)),
      c(sum(valid), sum(structure$crossed), sum(!structure$estimable),
        table(factor(role, roles)), table(factor(role[valid], roles))))
  list(features = features, summary = do.call(rbind, rows))
}
