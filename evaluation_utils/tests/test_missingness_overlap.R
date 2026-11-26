# Run from the repository root: pixi run Rscript evaluation_utils/tests/test_missingness_overlap.R
campaign <- new.env()
source("evaluation_data/simulated/missingness_overlap.R", local = campaign)
same <- function(actual, expected) {
  isTRUE(all.equal(actual, expected, tolerance = 1e-12, check.attributes = FALSE))
}
fails <- function(expr) inherits(tryCatch(force(expr), error = identity), "error")

# Options accept comma-separated roots within 1:10.
stopifnot(identical(campaign$campaign_options(character())$roots, 1L))
options <- campaign$campaign_options(c("--roots", "1,5,10", "--out", "/tmp/data,cohort"))
stopifnot(identical(options$roots, c(1L, 5L, 10L)),
          identical(options$out, "/tmp/data,cohort"))
for (args in list(c("--root", "1"), "--roots", c("--out", "--roots"),
                  c("--variant", "informative"),
                  c("--roots", "1", "--roots", "2"), c("--roots", "1.5"),
                  c("--roots", "0"), c("--roots", "11"), c("--roots", "1,,2"),
                  c("--roots", ""))) {
  stopifnot(fails(campaign$campaign_options(args)))
}

# Stratification retains exact overall fractions and complementary A/B rounding.
role <- rep(c("A_DE", "B_DE", "latent", "null"), c(1125, 1125, 150, 3600))
labs <- paste0("batch", 1:5)
set.seed(1001)
panel <- campaign$make_panel(role, labs)
set.seed(1001)
repeated_panel <- campaign$make_panel(role, labs)
stopifnot(identical(panel, repeated_panel), identical(dim(panel), c(6000L, 5L)),
          identical(colnames(panel), labs),
          identical(tabulate(rowSums(panel), 5L), c(600L, 1200L, 1200L, 1200L, 1800L)))
allocation <- rbind(A_DE = c(112, 225, 225, 225, 338),
                    B_DE = c(113, 225, 225, 225, 337),
                    latent = c(15, 30, 30, 30, 45),
                    null = c(360, 720, 720, 720, 1080))
for (r in rownames(allocation)) {
  stopifnot(same(tabulate(rowSums(panel[role == r, ]), 5L), allocation[r, ]))
}
set.seed(1002)
stopifnot(!identical(panel, campaign$make_panel(role, labs)),
          fails(campaign$make_panel(role, labs[1:3])),
          fails(campaign$make_panel(role[-1], labs)))

# Five-client eligibility must use the production minima of 6/7 observed samples.
meta <- data.frame(file = paste0("s", 1:100), lab = rep(labs, each = 20),
                   condition = factor(rep(rep(c("A", "B"), each = 10), 5)),
                   latent = rep(c(rep(FALSE, 14), rep(TRUE, 6)), 5))
rownames(meta) <- meta$file
features <- c("complete", "three", "two", "six", "seven", "three_low",
              "four_low", "confounded", "no_condition")
input <- matrix(NA_real_, length(features), nrow(meta), dimnames = list(features, meta$file))
put <- function(feature, clients, positions) {
  for (client in clients) {
    columns <- (client - 1L) * 20L + positions
    input[feature, columns] <<- seq_along(columns)
  }
}

put("complete", 1:5, 1:20)
put("three", 1:3, 1:20)
put("two", 1:2, 1:20)
put("six", 1:3, c(1:3, 11:13))
put("seven", 1:3, c(1:4, 11:13))
put("three_low", 1:2, c(1:4, 11:13))
put("three_low", 3, c(1:3, 11:12))
put("four_low", 1:3, c(1:4, 11:13))
put("four_low", 4, c(1:3, 11:12))
put("confounded", 1, 1:7)
put("confounded", 2:3, 11:17)
put("no_condition", 1:3, 1:7)
presence <- c(5L, 3L, 2L, 3L, 3L, 3L, 4L, 3L, 3L)
original <- input
audits <- setNames(lapply(c("without_condition", "with_condition"), function(arm) {
  campaign$eligibility_audit(input, meta, arm, presence)
}), c("without_condition", "with_condition"))
for (arm in names(audits)) {
  audit <- audits[[arm]]
  rows <- audit$table[match(features, audit$table$feature), ]
  stopifnot(identical(rows$feature, features), identical(input, original),
            all(rows$threshold == if (arm == "with_condition") 7L else 6L),
            rows$effective_presence[rows$feature == "four_low"] == 3L,
            rows$blanked_cells[rows$feature == "four_low"] == 5L,
            rows$effective_presence[rows$feature == "three_low"] == 2L,
            !rows$retained[rows$feature %in% c("two", "three_low")],
            rows$retained[rows$feature %in% c("complete", "three", "seven", "four_low")],
            !rows$rank_failure[rows$feature %in% c("complete", "three", "seven", "four_low")],
            rows$rank[rows$feature == "three"] ==
              if (arm == "with_condition") 4L else 3L,
            all(is.na(audit$input["four_low", meta$lab %in% labs[4:5]])))
  if (arm == "with_condition") {
    stopifnot(!rows$retained[rows$feature == "six"],
              rows$rank_failure[rows$feature == "confounded"],
              !rows$rank_failure[rows$feature == "no_condition"],
              rows$constant_condition[rows$feature == "no_condition"],
              rows$required_rank[rows$feature == "no_condition"] == 3L)
  } else {
    stopifnot(rows$retained[rows$feature == "six"], !any(rows$rank_failure))
  }
}

# The keep route is the no-label filter: observation counts and the six-sample minimum;
# it must not use condition values or remove every structurally absent column.
matched <- campaign$downstream_panel(input, meta)
redacted_meta <- meta
redacted_meta$condition <- "redacted"
stopifnot(identical(matched, campaign$downstream_panel(input, redacted_meta)),
          identical(rownames(matched$input), sort(rownames(audits$without_condition$input))),
          identical(matched$input, audits$without_condition$input[rownames(matched$input), ]),
          all(is.na(matched$input["four_low", meta$lab %in% labs[4:5]])),
          sum(matched$table$complete_case) == 1L,
          identical(matched$table$feature[matched$table$impute_retained], "complete"),
          matched$table$matched_retained[matched$table$feature == "six"])
without_complete <- input
without_complete[, 1L] <- NA_real_
stopifnot(!any(campaign$downstream_panel(without_complete, meta)$table$complete_case),
          sum(campaign$downstream_panel(without_complete, meta)$table$matched_retained) > 0L)
invalid <- input
invalid[1, 1] <- Inf
stopifnot(fails(campaign$downstream_panel(invalid, meta)))
# The impute route needs six observations in every client, the keep route in three.
edge <- input[c("complete", "complete"), ]
rownames(edge) <- c("six_each", "five_once")
edge[, rep(7:20, 5) + rep(0:4 * 20, each = 14)] <- NA
edge["five_once", 6] <- NA
edge <- campaign$downstream_panel(edge, meta)$table
stopifnot(identical(edge$impute_retained, c(TRUE, FALSE)), all(edge$matched_retained),
          identical(edge$matched_clients, c(5, 4)))

# Saved uniform draws pair severities and panels without consuming further RNG draws.
truth <- rbind(condition_b = rep(c(rep(0, 10), rep(2, 5), rep(4, 5)), 5),
               condition_a = rep(c(rep(2, 10), rep(0, 10)), 5),
               latent = rep(c(rep(0, 14), rep(-2, 6)), 5),
               null = rep(c(-1, 0, 1, 0), 25))
colnames(truth) <- meta$file
roles <- c("B_DE", "A_DE", "latent", "null")
set.seed(40)
uniform <- matrix(runif(length(truth)), nrow(truth), dimnames = dimnames(truth))
small_panel <- matrix(TRUE, 4, 5, dimnames = list(rownames(truth), labs))
small_panel[1, 4:5] <- FALSE
small_panel[2, 5] <- FALSE
data <- list(root = 1L, observed = truth + 0.25, truth = truth, role = roles, meta = meta,
             uniform = uniform, panel = small_panel)
original <- data
rng_before <- .Random.seed
stopifnot(identical(campaign$mask_root(data, 0), data$observed))
masked <- campaign$mask_root(data, 0.1)
severe <- campaign$mask_root(data, 0.5)
stopifnot(identical(.Random.seed, rng_before), identical(data, original),
          anyNA(masked), sum(is.na(severe)) > sum(is.na(masked)),
          all(is.na(severe)[is.na(masked)]),
          identical(campaign$mask_root(data, 0.1), masked),
          identical(campaign$panel_data(masked, data, "full"), masked))
partial <- campaign$panel_data(masked, data, "overlap")
for (lab in labs) {
  columns <- meta$lab == lab
  stopifnot(identical(partial[small_panel[, lab], columns, drop = FALSE],
                      masked[small_panel[, lab], columns, drop = FALSE]),
            all(is.na(partial[!small_panel[, lab], columns, drop = FALSE])))
}

# The informative variant shifts the first on/off feature of each class down in its off
# class, turns the other A/B features into null features and keeps the base root's batch
# component and lab intercepts, so unchanged features keep their masks.
signal <- rbind(a_on = truth["condition_a", ], a_other = truth["condition_a", ],
                b_on = truth["condition_b", ], b_other = truth["condition_b", ],
                truth[c("latent", "null"), ])
toy <- list(meta = meta, truth = signal, observed = signal + 0.25,
            role = c("A_DE", "A_DE", "B_DE", "B_DE", "latent", "null"),
            uniform = matrix(runif(length(signal)), nrow(signal)))
no_signal <- signal
no_signal[1:4, ] <- 0
campaign$campaign$onoff <- 1L
variant <- campaign$informative_root(toy, no_signal)
in_b <- meta$condition == "B"
shift <- campaign$campaign$onoff_shift_sd * campaign$campaign$within_sd
masks <- lapply(list(base = toy, variant = variant), function(d) is.na(campaign$mask_root(d, 0.5)))
stopifnot(identical(variant$role, c("A_DE", "null", "B_DE", "null", "latent", "null")),
          same(variant$truth["a_on", in_b], signal["a_on", in_b] - shift),
          identical(variant$truth["a_on", !in_b], signal["a_on", !in_b]),
          same(variant$truth["b_on", !in_b], signal["b_on", !in_b] - shift),
          identical(variant$truth["b_on", in_b], signal["b_on", in_b]),
          all(variant$truth[c("a_other", "b_other"), ] == 0),
          identical(variant$truth[5:6, ], signal[5:6, ]),
          same(variant$observed - variant$truth, toy$observed - toy$truth),
          same(variant$intercept["0.5", ], campaign$lab_intercepts(toy$observed, meta, 0.5)),
          identical(masks$variant[5:6, ], masks$base[5:6, ]),
          sum(masks$variant["a_on", in_b]) > sum(masks$base["a_on", in_b]),
          fails(campaign$informative_root(toy, 0 * signal)))
campaign$campaign$onoff <- 50L

# Each lab's expected missing share is its multiple of the target, and low-intensity
# features are missed more often.
set.seed(42)
wide <- matrix(rnorm(400 * 100, sd = 10), 400, 100) + rnorm(400, sd = 15)
wide_data <- list(observed = wide, meta = meta, uniform = matrix(runif(length(wide)), 400))
wide_masked <- campaign$mask_root(wide_data, 0.3)
rates <- sapply(labs, function(lab) mean(is.na(wide_masked[, meta$lab == lab])))
abundance <- rowMeans(wide)
stopifnot(all(abs(rates - 0.3 * campaign$lab_rate[labs]) < 0.02),
          mean(is.na(wide_masked[abundance < median(abundance), ])) >
            2 * mean(is.na(wide_masked[abundance >= median(abundance), ])))

# A per-feature constant changes only the detection mask: with a fixed mask, correction
# shifts by the constant while recovery scores and DE statistics are unchanged.
set.seed(43)
shifted <- truth + matrix(rnorm(length(truth), sd = 0.3), nrow(truth)) +
  rep(c(0, 0.7, -1.5, 1.2, -0.7), each = 20)[col(truth)]
shifted[1, c(1, 25, 47)] <- NA
constant <- c(30, -12, 5, 0)
for (arm in campaign$shared$arms) {
  plain <- campaign$shared$correct_central(shifted, meta, arm)$corrected
  offset <- campaign$shared$correct_central(shifted + constant, meta, arm)$corrected
  stopifnot(identical(is.na(plain), is.na(offset)), max(abs(offset - constant - plain),
                                                        na.rm = TRUE) < 1e-10,
            same(campaign$score_metrics(plain, truth, meta, roles),
                 campaign$score_metrics(offset, truth + constant, meta, roles)))
}
design <- model.matrix(~condition, meta)
stopifnot(same(limma::eBayes(limma::lmFit(shifted, design))$t[, 2],
               limma::eBayes(limma::lmFit(shifted + constant, design))$t[, 2]))

# Per-feature eta2_obs is the confounding index on the observed cells, and no-label
# correction keeps 1 - eta2_obs of the label-aware condition contrast.
eta2 <- campaign$feature_eta2(shifted, meta)
seen <- !is.na(shifted[1, ])
fits <- lapply(setNames(nm = campaign$shared$arms), function(arm) {
  campaign$shared$correct_central(shifted, meta, arm)
})
checked <- campaign$eta2_identity(fits$without_condition, fits$with_condition, meta)
uneven <- fits$with_condition
uneven$input[2, 1] <- NA
stopifnot(eta2[[1]] > 0, all(eta2[-1] == 0),
          same(eta2[[1]], campaign$shared$eta_squared(meta$condition[seen] == "B",
                                                      meta$lab[seen])),
          checked[["features"]] == 4, checked[["max_deviation"]] < 1e-12,
          campaign$eta2_identity(fits$without_condition, uneven, meta)[["features"]] == 3)

# Observed-cell and complete-truth references answer different recovery questions.
selected <- truth
selected["condition_b", rep(15:20, 5) + rep(0:4 * 20, each = 6)] <- NA
observed <- campaign$score_metrics(selected, truth, meta, roles)
complete <- campaign$shared$contrast_recovery(selected, truth, meta, roles)
stopifnot(same(observed["attenuation_condition_b_de"], 1),
          same(complete["attenuation_condition_b_de"], 2 / 3),
          same(observed["rmse_to_truth"], 0), same(observed["residual_lab_rms"], 0))
observed <- campaign$score_metrics(0.5 * selected, truth, meta, roles)
complete <- campaign$shared$contrast_recovery(0.5 * selected, truth, meta, roles)
stopifnot(same(observed["attenuation_condition_b_de"], 0.5),
          same(complete["attenuation_condition_b_de"], 1 / 3),
          same(observed["rmse_to_truth"]^2,
               observed["residual_lab_rms"]^2 + observed["within_lab_rmse"]^2))
selected["condition_b", meta$condition == "B"] <- NA
undefined <- campaign$score_metrics(selected, truth, meta, roles)
stopifnot(is.na(undefined["attenuation_condition_b_de"]),
          undefined["condition_contrast_missing"] == 1)

# Exercise all settings with tiny inputs and stubbed correction, never a native campaign.
calls <- character()
matched_input <- function(x, meta, arm) {
  suppressMessages(campaign$shared$filter_fedrbe_client(x, meta, "file", "lab", 5L,
    covariates = if (arm == "with_condition") "condition" else character()))
}
campaign$shared$correct_central <- function(x, meta, arm) {
  calls <<- c(calls, paste("central", arm))
  input <- matched_input(x, meta, arm)
  list(input = input, corrected = input)
}
campaign$shared$correct_fedrbe <- function(x, meta, arm) {
  calls <<- c(calls, paste("fedrbe", arm))
  input <- matched_input(x, meta, arm)
  input[rev(seq_len(nrow(input))), rev(seq_len(ncol(input))), drop = FALSE]
}
# Keep a generous observed-cell margin: one sample per lab is always undetected.
data$uniform[] <- 0
data$uniform[, c(1, 21, 41, 61, 81)] <- 1
result <- campaign$evaluate_root(data)
stopifnot(nrow(result$status) == 16L, length(calls) == 32L,
          all(result$status$status == "ok"), all(result$status$max_diff == 0),
          nrow(unique(result$status[c("panel", "target")])) == 8L,
          !anyDuplicated(result$metrics[setdiff(names(result$metrics), "value")]),
          nrow(result$missingness) == 40L,
          all(is.na(result$metrics$value[result$metrics$panel == "full" &
            result$metrics$presence == "3" & result$metrics$stage == "fedrbe" &
            result$metrics$metric == "attenuation_condition"])))
agreement <- result$metrics[result$metrics$stage == "agreement", ]
stopifnot(all(agreement$value[is.finite(agreement$value)] == 0))

# Export preparation reuses each native fit, then masks all scoring states alike.
# Stubs above exercise the boundary between native retention and learner matching.
# Imputation needs one observed value per sample: the undetected samples keep "null".
data$uniform["null", c(1, 21, 41, 61, 81)] <- 0
data$observed["null", c(1, 21, 41, 61, 81)] <- -(1:5)
exported <- campaign$downstream_inputs(data, 0.5, "overlap")
scoring_mask <- is.na(exported$matrices$uncorrected)
for (state in names(exported$matrices)) {
  stopifnot(identical(is.na(exported$matrices[[state]]), scoring_mask),
            identical(dimnames(exported$matrices[[state]]),
                      dimnames(exported$matrices$uncorrected)))
}
# The stubbed fits agree, so the central learners' matrix equals the federated one.
stopifnot(identical(exported$matrices$no_label_central, exported$matrices$no_label),
          identical(exported$impute$no_label_central, exported$impute$no_label),
          identical(exported$matrices$truth[!scoring_mask],
                    data$truth[rownames(exported$matrices$truth), ][!scoring_mask]),
          nrow(result$learner_eligibility) == 8L * 5L,
          all(result$learner_eligibility$rf_features <=
                result$learner_eligibility$total_features))

# The impute route keeps features observed in every client and fills each gap with the
# sample's own lowest observed value, so other clients' values cannot change it.
filled <- exported$impute$uncorrected
elsewhere <- data
elsewhere$observed[, meta$lab != "batch1"] <- data$observed[, meta$lab != "batch1"] + 100
stopifnot(identical(rownames(filled), c("latent", "null")), !anyNA(filled),
          identical(dimnames(exported$impute$no_label), dimnames(filled)),
          exported$imputed_cells == 5L,
          identical(filled["latent", c(1, 21, 41, 61, 81)],
                    data$observed["null", c(1, 21, 41, 61, 81)]),
          identical(campaign$downstream_inputs(elsewhere, 0.5, "overlap")$impute$uncorrected[
                      , meta$lab == "batch1"], filled[, meta$lab == "batch1"]),
          setequal(names(exported$corrections),
                   c("without_condition", "impute_without_condition")))

# The exporter validates the selected analysis and retained mask records strictly.
exporter <- new.env()
source("evaluation_data/simulated/export_downstream_inputs.R", local = exporter)
reference_path <- tempfile(fileext = ".tsv")
input_path <- tempfile(fileext = ".rds")
stopifnot(file.create(input_path))
current <- result$missingness[result$missingness$panel == "overlap" &
                                result$missingness$target == 0.5, ]
write.table(current, reference_path, sep = "\t", row.names = FALSE, quote = FALSE)
comparison <- exporter$compare_missingness_reference(current, reference_path)
stopifnot(comparison$matched_metrics == 20L, comparison$max_absolute_difference < 1e-12)
args <- c("--analysis", "missingness_overlap", "--scenario", "missingness_overlap_50",
          "--run", "1", "--out", tempfile(), "--reference", reference_path,
          "--target", "0.5", "--panel", "overlap", "--inputs", input_path)
stopifnot(exporter$parse_options(args)$target == 0.5)
invalid <- args
invalid[which(invalid == "0.5")] <- "0.4"
stopifnot(fails(exporter$parse_options(invalid)))
invalid <- args
invalid[which(invalid == "missingness_overlap_50")] <- "missingness_full_50"
stopifnot(fails(exporter$parse_options(invalid)))
changed <- current
changed$total_rate[1] <- changed$total_rate[1] + 0.01
stopifnot(fails(exporter$compare_missingness_reference(changed, reference_path)))
write.table(rbind(current, current[1, ]), reference_path, sep = "\t",
            row.names = FALSE, quote = FALSE)
stopifnot(fails(exporter$compare_missingness_reference(current, reference_path)))
unlink(c(reference_path, input_path))

# Uncertainty is across roots; slope summaries add the mean deviation from unit gain.
report <- campaign
summary <- report$overlap_summary_values(c(1, 2, 3, NA_real_, Inf))
half_width <- qt(0.975, 2) / sqrt(3)
stopifnot(summary$n_roots == 5L, summary$n_finite == 3L, summary$n_missing == 2L,
          summary$mean == 2, summary$sd == 1, same(summary$mcse, 1 / sqrt(3)),
          same(summary$ci_low, 2 - half_width), same(summary$ci_high, 2 + half_width))
for (values in list(numeric(), NA_real_, c(4, NA_real_))) {
  summary <- report$overlap_summary_values(values)
  stopifnot(is.na(summary$mcse), is.na(summary$ci_low), is.na(summary$ci_high))
}
synthetic <- data.frame(root = 1:3, panel = "full", target = 0.1,
                        arm = "without_condition", presence = "5", stage = "fedrbe",
                        metric = "attenuation_condition", value = c(0.9, 1, 1.1))
summary <- report$overlap_group_summary(synthetic)
stopifnot(nrow(summary) == 1L, summary$n_finite == 3L, same(summary$mean, 1),
          same(summary$mean_abs_dev_1, 0.2 / 3))
