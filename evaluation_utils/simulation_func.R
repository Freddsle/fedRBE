library(invgamma)
library(RobNorm)
library(tidyverse)

normalize_count <- function(n, max_n = NULL) {
    out <- max(0L, as.integer(round(n)))
    if (!is.null(max_n)) {
        out <- min(out, max_n)
    }
    out
}

take_first_values <- function(x, n) {
    n <- normalize_count(n, length(x))
    if (n == 0L) {
        return(x[0])
    }
    x[seq_len(n)]
}

take_first_rows <- function(x, n) {
    n <- normalize_count(n, nrow(x))
    if (n == 0L) {
        return(x[0, , drop = FALSE])
    }
    x[seq_len(n), , drop = FALSE]
}

take_last_rows <- function(x, n) {
    n <- normalize_count(n, nrow(x))
    if (n == 0L) {
        return(x[0, , drop = FALSE])
    }
    x[seq.int(nrow(x) - n + 1L, nrow(x)), , drop = FALSE]
}

drop_first_rows <- function(x, n) {
    n <- normalize_count(n, nrow(x))
    if (n >= nrow(x)) {
        return(x[0, , drop = FALSE])
    }
    x[seq.int(n + 1L, nrow(x)), , drop = FALSE]
}

drop_last_rows <- function(x, n) {
    keep_n <- nrow(x) - normalize_count(n, nrow(x))
    if (keep_n <= 0L) {
        return(x[0, , drop = FALSE])
    }
    x[seq_len(keep_n), , drop = FALSE]
}

sample_effects_with_min_spacing <- function(first_param, second_param, sampler, min_pct_diff = 20) {
    pct_diff <- function(a, b) abs(a - b) / max(abs(b), .Machine$double.eps) * 100

    repeat {
        effects <- mapply(sampler, first_param, second_param)
        if (length(effects) < 2) {
            return(effects)
        }

        sorted_effects <- sort(effects)
        pairwise_diffs <- mapply(
            pct_diff,
            sorted_effects[-1],
            sorted_effects[-length(sorted_effects)]
        )

        if (all(pairwise_diffs >= min_pct_diff)) {
            return(effects)
        }
    }
}

expand_effects_by_sample <- function(effect_values, batch_vec, n_rows, col_names = NULL) {
    batch_chr <- as.character(batch_vec)
    missing_batches <- setdiff(unique(batch_chr), names(effect_values))
    if (length(missing_batches) > 0) {
        stop(
            "Missing effect values for batches: ",
            paste(missing_batches, collapse = ", "),
            call. = FALSE
        )
    }

    matrix(
        rep(unname(effect_values[batch_chr]), each = n_rows),
        nrow = n_rows,
        ncol = length(batch_chr),
        dimnames = list(NULL, col_names)
    )
}

################################################################################################
# Adapted from RobNorm sim.dat.fn with modified IG parameters
sim.dat.fn = function(row.frac, col.frac, mu.up, mu.down, n, m, nu.fix=TRUE) {
    mu.00 = rnorm(n, 0, 1)
    var.00 = rinvgamma(n, 2, 3)

    X.0 = matrix(rnorm(n*m, 
					outer(mu.00, rep(1, m)),
					sqrt(outer(var.00, rep(1, m)))), 
				n, m) # the null matrix
    
	S = matrix(0, n, m)
	signal_rows <- normalize_count(n * row.frac, n)
	signal_cols <- normalize_count(m * col.frac, m)
	if (signal_rows > 0L && signal_cols > 0L) {
		   bk.nm <- signal_rows * signal_cols
		   a = rbinom(1, bk.nm, 0.8)
		   S[seq_len(signal_rows), seq_len(signal_cols)] =
               sample(c(rep(mu.up, a), rep(0, bk.nm-a)), bk.nm) # the shifted mean of the signal mx
		   a = rbinom(1, bk.nm, 0.8)
		   S[seq.int(n - signal_rows + 1L, n), seq.int(m - signal_cols + 1L, m)] =
               sample(c(rep(mu.down, a), rep(0, bk.nm-a)), bk.nm) # the signal mx of shifted mean
	}

	X = X.0 + S 
	
	rownames(X) = paste("prt", 1:nrow(X), sep=".")
	colnames(X) = paste("s", 1:ncol(X), sep=".")
    return(list(dat=X, raw=X.0))
}


select_proportions <- function(mode_version){
    proportions <- switch(
        mode_version,
        balanced = list(0.6, 0.6, 0.6),
        imbalanced_balanced = list(0.6, 0.6, 0.6),
        mild_imbalanced = list(0.4, 0.5, 0.66),
        strong_imbalanced = list(0.2, 0.5, 0.7),
        NULL
    )

    if (is.null(proportions)) {
        stop("Unknown mode_version: ", mode_version, call. = FALSE)
    }

    return(proportions)
}


generate_data <- function(
    col_frac_A, col_frac_B,
    frac_1, frac_7,
    mu_1 = 1.25, mu_4 = 1.25,
    batch_info = NULL, nu.fix = TRUE,
    mode_version = "balanced",
    m = 600
    ){
    if (is.null(batch_info)) {
        stop("batch_info must be provided.", call. = FALSE)
    }

    # add A condition
    data_mu1 <- sim.dat.fn(row.frac=frac_1, col.frac=col_frac_A, mu.up=mu_1, mu.down=0, n=2500, m=m, nu.fix=nu.fix)
    # add B condition
    data_mu2 <- sim.dat.fn(row.frac=frac_1, col.frac=col_frac_B, mu.up=0, mu.down=mu_1, n=2500, m=m, nu.fix=nu.fix)

    # % of confounder in batches
    b_proportions <- select_proportions(mode_version)
    get_condition_files <- function(batch_name, condition_name) {
        batch_info$file[batch_info$batch == batch_name & batch_info$condition == condition_name]
    }

    # rearrange the data7
    batch1_files <- get_condition_files("batch1", "B")
    batch2_files <- get_condition_files("batch2", "B")
    batch3_files <- get_condition_files("batch3", "B")
    batch1_size <- normalize_count(b_proportions[[1]] * length(batch1_files), length(batch1_files))
    batch2_size <- normalize_count(b_proportions[[2]] * length(batch2_files), length(batch2_files))
    batch3_size <- normalize_count(b_proportions[[3]] * length(batch3_files), length(batch3_files))
    need_to_generate <- (batch1_size + batch2_size + batch3_size) / m
    # add confounder
    data_mu7 <- sim.dat.fn(row.frac=frac_7, col.frac=need_to_generate, mu.up=0, mu.down=-1*mu_4, n=1000, m=m,  nu.fix=nu.fix)
    cat(ncol(data_mu7$dat) * need_to_generate, "\n")

    new_names <- c(
        take_first_values(batch1_files, batch1_size),
        take_first_values(batch2_files, batch2_size),
        take_first_values(batch3_files, batch3_size)
    )

    a_condition_files <- batch_info$file[batch_info$condition == "A"]
    b_condition_files <- batch_info$file[batch_info$condition == "B"]
    all_other_names <- c(
        a_condition_files,
        setdiff(b_condition_files, new_names)
    )
    
    colnames(data_mu7$dat) <- c(all_other_names, new_names)
    data_mu7$dat <- data_mu7$dat[, batch_info$file]

    mu1_signal_rows <- normalize_count(frac_1 * nrow(data_mu1$dat), nrow(data_mu1$dat))
    mu7_signal_rows <- normalize_count(frac_7 * nrow(data_mu7$dat), nrow(data_mu7$dat))

    # Combine all the data
    data_allmu <- rbind(
        take_first_rows(data_mu1$dat, mu1_signal_rows),
        take_last_rows(data_mu2$dat, mu1_signal_rows),
        take_last_rows(data_mu7$dat, mu7_signal_rows),
        drop_first_rows(data_mu1$dat, mu1_signal_rows),
        drop_last_rows(data_mu2$dat, mu1_signal_rows),
        drop_last_rows(data_mu7$dat, mu7_signal_rows)
        )

    return(data_allmu)
}


apply_rotation_effect_1pc <- function(X, batch_vec, target_batch, angle_deg = 25, pc = 1) {
    # X: proteins x samples matrix
    # batch_vec: length == ncol(X)
    # target_batch: which batch label to rotate (e.g., "batch2")
    # angle_deg: rotation angle (degrees); 0 => no-op
    # pc: axis PC index (default 1). Rotation is done in (PCpc, PCpc+1) plane.

    if (is.null(target_batch) || is.na(target_batch) || angle_deg == 0) return(X)

    batch_chr <- as.character(batch_vec)
    idx <- which(batch_chr == target_batch)
    if (length(idx) == 0) return(X)

    # Center by protein means (keeps per-protein mean structure stable)
    mu <- rowMeans(X, na.rm = TRUE)
    Xc <- sweep(X, 1, mu, "-")

    # Neutralize NAs for SVD math (if you ever call after missingness)
    if (anyNA(Xc)) Xc[is.na(Xc)] <- 0

    pc <- as.integer(pc)
    k <- max(2L, pc + 1L)  # need pc and pc+1 to define a rotation plane

    # Truncated SVD in protein space
    sv <- svd(Xc, nu = k, nv = 0)

    if (ncol(sv$u) < (pc + 1L)) return(X)  # degenerate edge case

    u1 <- sv$u[, pc]
    u2 <- sv$u[, pc + 1L]  # <-- key fix: use the *actual* next PC direction

    # Fix signs deterministically (avoid run-to-run random flips)
    j1 <- which.max(abs(u1)); if (u1[j1] < 0) u1 <- -u1
    j2 <- which.max(abs(u2)); if (u2[j2] < 0) u2 <- -u2

    # Rotation in (u1, u2) plane
    theta <- angle_deg * pi / 180
    ct <- cos(theta); st <- sin(theta)

    # Project target batch columns to (u1, u2) coordinates
    a1 <- as.numeric(crossprod(u1, Xc[, idx, drop = FALSE]))
    a2 <- as.numeric(crossprod(u2, Xc[, idx, drop = FALSE]))

    # Rotate coordinates
    a1p <- ct * a1 - st * a2
    a2p <- st * a1 + ct * a2

    # Rank-2 update only on target batch columns
    Xc[, idx] <- Xc[, idx, drop = FALSE] +
        u1 %*% t(a1p - a1) +
        u2 %*% t(a2p - a2)

    # Add means back
    Xrot <- sweep(Xc, 1, mu, "+")
    rownames(Xrot) <- rownames(X)
    colnames(Xrot) <- colnames(X)
    Xrot
}



pc1_var_explained_fast <- function(X, iters = 15L) {
    Xc <- sweep(X, 1, rowMeans(X, na.rm = TRUE), "-")
    if (anyNA(Xc)) Xc[is.na(Xc)] <- 0
    if (ncol(Xc) == 0 || nrow(Xc) == 0) return(0)
    v <- rep(1, ncol(Xc))
    v <- v / sqrt(sum(v^2))
    for (i in seq_len(iters)) {
        w <- Xc %*% v
        v <- as.numeric(crossprod(Xc, w))
        nv <- sqrt(sum(v^2))
        if (nv == 0) return(0)
        v <- v / nv
    }
    w <- Xc %*% v
    lambda <- sum(w^2)
    total <- sum(Xc^2)
    if (total == 0) return(0)
    lambda / total
}


add_batch_effect <- function(
    result_two, 
    batch_info,
    mean_additive = sample(c(0, 0.7, -1.5)),
    sd_additive = sample(c(0.5, 1, 1.5)),
    shape_multiplicative = sample(c(3, 1, 5)),
    min_pc1_var = 0.2,
    pc1_iters = 15L
    ){

    # Assuming 'result_two' is your data matrix and 'batch_info' is a vector indicating the batch for each column
    batch_levels <- unique(as.character(batch_info$batch))
    n_batches <- length(batch_levels)
    n_cols <- ncol(result_two)
    n_proteins <- nrow(result_two)

    # For Additive Effects
    additive_params <- data.frame(
        mean = rep(mean_additive, length.out = n_batches),
        sd = rep(sd_additive, length.out = n_batches),
        row.names = batch_levels
    ) 
    # For Multiplicative Effects
    mult_scale <- sample(c(2, 1, 0.5), 1)
    multiplicative_params <- data.frame(
        shape = rep(shape_multiplicative, length.out = n_batches),
        scale = rep(mult_scale, length.out = n_batches),
        row.names = batch_levels
    ) 

    repeat {
        add_batch_effects <- sample_effects_with_min_spacing(
            additive_params$mean,
            additive_params$sd,
            function(mean, sd) rnorm(1, mean, sd)
        )
        names(add_batch_effects) <- batch_levels
        print(add_batch_effects)

        mult_batch_effects <- sample_effects_with_min_spacing(
            multiplicative_params$shape,
            multiplicative_params$scale,
            function(shape, scale) rinvgamma(1, shape, scale)
        )
        names(mult_batch_effects) <- batch_levels
        print(mult_batch_effects)

        additive_effects_matrix <- expand_effects_by_sample(
            add_batch_effects,
            batch_info$batch,
            n_proteins,
            colnames(result_two)
        )
        multiplicative_effects_matrix <- expand_effects_by_sample(
            mult_batch_effects,
            batch_info$batch,
            n_proteins,
            colnames(result_two)
        )

        # Noise
        noise_effect <- matrix(rnorm(n_cols * n_proteins, mean = 0, sd = 0.5), nrow = n_proteins, ncol = n_cols)

        # Apply batch effects
        batch_effect_component <- additive_effects_matrix + multiplicative_effects_matrix * noise_effect
        data_with_batch_effects <- result_two + batch_effect_component

        if (is.null(min_pc1_var) || min_pc1_var <= 0) {
            break
        }
        if (pc1_var_explained_fast(batch_effect_component, pc1_iters) >= min_pc1_var) {
            break
        }
    }

    # ---- Rotation effect (optional): one batch, 1 PC ----
    # Enable by setting either:
    #   attr(batch_info, "rotation") <- list(batch="batch2", angle_deg=25, pc=1)
    # or by adding columns rotation_batch + rotation_angle_deg (single unique value each).
    rot <- attr(batch_info, "rotation")

    if (is.null(rot) &&
        ("rotation_batch" %in% names(batch_info)) &&
        ("rotation_angle_deg" %in% names(batch_info))) {

        rb <- unique(na.omit(as.character(batch_info$rotation_batch)))
        ra <- unique(na.omit(as.numeric(batch_info$rotation_angle_deg)))
        if (length(rb) >= 1 && length(ra) >= 1) {
            rot <- list(batch = rb[1], angle_deg = ra[1], pc = 1)
        }
    }

    if (!is.null(rot) && !is.null(rot$batch) && !is.null(rot$angle_deg) && rot$angle_deg != 0) {
        data_with_batch_effects <- apply_rotation_effect_1pc(
            X = data_with_batch_effects,
            batch_vec = batch_info$batch,
            target_batch = as.character(rot$batch),
            angle_deg = as.numeric(rot$angle_deg),
            pc = ifelse(is.null(rot$pc), 1, as.integer(rot$pc))
        )
    }

    return(data_with_batch_effects)
}


simulateMissingValues <- function(df, alpha, beta) {
  df <- as.matrix(df)

  # Number of total values
  N <- nrow(df) * ncol(df)

  # Step 1: Creating a threshold matrix T
  alpha_quantile <- quantile(as.vector(df), probs = alpha, na.rm = TRUE)
  T <- matrix(rnorm(n = N, mean = alpha_quantile, sd = 0.3), nrow = nrow(df), ncol = ncol(df))

  # Step 2: Creating a probability matrix P
  P <- matrix(rbinom(n = N, size = 1, prob = beta), nrow = nrow(df), ncol = ncol(df))

  # Step 3: Vectorized operation to simulate MNAR
  mnar_mask <- (df < T) & (P == 1)
  df[mnar_mask] <- NA

  # Simulate MAR: Randomly replace additional values without touching MNAR values
  num_MAR <- round(N * alpha * (1 - beta))
  available_indices <- which(!is.na(df), arr.ind = TRUE)

  if (nrow(available_indices) > 0) {
    num_MAR <- min(num_MAR, nrow(available_indices))
    if (num_MAR > 0) {
      selected_indices <- available_indices[sample(nrow(available_indices), num_MAR), , drop = FALSE]
      df[selected_indices] <- NA
    }
  }

  return(df)
}
