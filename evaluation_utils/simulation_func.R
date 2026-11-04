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
sim.dat.fn = function(row.frac, col.frac, mu.up, mu.down, n, m) {
    mu.00 = rnorm(n, 0, 1)
    var.00 = 1 / rgamma(n, 2, 3) # InvGamma(shape 2, scale 3); same draws as invgamma::rinvgamma

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
    return(list(dat=X))
}


# Graded batch-condition confounding: 150/300/150 samples per lab with condition-B
# fractions p, 1 - p and p. The names are simulation scenarios.
confounded_b_fraction <- c(confounded_p50 = 0.5, confounded_p30 = 0.3,
                           confounded_p15 = 0.15, confounded_p05 = 0.05)

select_proportions <- function(mode_version){
    proportions <- switch(
        mode_version,
        balanced = list(0.6, 0.6, 0.6),
        mild_imbalanced = list(0.4, 0.5, 0.66),
        strong_imbalanced = list(0.2, 0.5, 0.7),
        confounded_p50 = , confounded_p30 = , confounded_p15 = ,
        confounded_p05 = list(0.6, 0.6, 0.6),
        NULL
    )

    if (is.null(proportions)) {
        stop("Unknown mode_version: ", mode_version, call. = FALSE)
    }

    return(proportions)
}


generate_metadata <- function(batch_info, mode){

    if (mode == "balanced"){
        batches <- rep(c("batch1", "batch2", "batch3"), each = 100)
        batch_info$condition <- as.factor(c(rep("A", 300), rep("B", 300)))
        batch_info$batch <- as.factor(c(batches[sample(1:300)], batches[sample(1:300)]))

    } else if (mode == "strong_imbalanced") {
        batches_A <- c(rep("batch1", 32), rep("batch2", 28), rep("batch3", 288))
        batches_B <- c(rep("batch1", 8), rep("batch2", 52), rep("batch3", 192))
        batch_info$condition <- as.factor(c(rep("A", length(batches_A)), rep("B", length(batches_B))))
        batch_info$batch <- as.factor(c(
            batches_A[sample(1:length(batches_A))],
            batches_B[sample(1:length(batches_B))]))

    } else if(mode == "mild_imbalanced") {
        batches_A <- c(rep("batch1", 36), rep("batch2", 91), rep("batch3", 185))
        batches_B <- c(rep("batch1", 54), rep("batch2", 49), rep("batch3", 185))
        batch_info$condition <- as.factor(c(rep("A", length(batches_A)), rep("B", length(batches_B))))
        batch_info$batch <- as.factor(c(
            batches_A[sample(1:length(batches_A))],
            batches_B[sample(1:length(batches_B))]))
    } else if (mode %in% names(confounded_b_fraction)) {
        p <- confounded_b_fraction[[mode]]
        sizes <- c(150L, 300L, 150L)
        n_B <- c(normalize_count(sizes[1] * p), normalize_count(sizes[2] * (1 - p)),
                 normalize_count(sizes[3] * p))
        batches_A <- rep(c("batch1", "batch2", "batch3"), sizes - n_B)
        batches_B <- rep(c("batch1", "batch2", "batch3"), n_B)
        batch_info$condition <- as.factor(c(rep("A", length(batches_A)), rep("B", length(batches_B))))
        batch_info$batch <- as.factor(c(
            batches_A[sample(1:length(batches_A))],
            batches_B[sample(1:length(batches_B))]))
    } else {
        stop("Unknown mode: ", mode, call. = FALSE)
    }
    return(batch_info)

}


generate_data <- function(
    col_frac_A, col_frac_B,
    frac_1, frac_7,
    mu_1 = 1.25, mu_4 = 1.25,
    batch_info = NULL,
    mode_version = "balanced",
    m = 600
    ){
    if (is.null(batch_info)) {
        stop("batch_info must be provided.", call. = FALSE)
    }

    # add A condition
    data_mu1 <- sim.dat.fn(row.frac=frac_1, col.frac=col_frac_A, mu.up=mu_1, mu.down=0, n=2500, m=m)
    # add B condition
    data_mu2 <- sim.dat.fn(row.frac=frac_1, col.frac=col_frac_B, mu.up=0, mu.down=mu_1, n=2500, m=m)

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
    data_mu7 <- sim.dat.fn(row.frac=frac_7, col.frac=need_to_generate, mu.up=0, mu.down=-1*mu_4, n=1000, m=m)
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
    # Ground truth for evaluation: row blocks follow the rbind order above.
    attr(data_allmu, "feature_role") <- rep(
        c("A_DE", "B_DE", "latent", "null"),
        c(mu1_signal_rows, mu1_signal_rows, mu7_signal_rows,
          nrow(data_allmu) - 2 * mu1_signal_rows - mu7_signal_rows)
    )
    attr(data_allmu, "latent_samples") <- new_names

    return(data_allmu)
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
            # Rate `scale` gives InvGamma(shape, scale), as invgamma::rinvgamma did.
            function(shape, scale) 1 / rgamma(1, shape, scale)
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

    # Realized lab parameters for provenance; the added noise is N(0, 0.5) scaled per lab.
    attr(data_with_batch_effects, "batch_params") <- list(
        additive = add_batch_effects, multiplicative = mult_batch_effects,
        noise_sd = 0.5 * mult_batch_effects)
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
