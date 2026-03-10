# =============================================================================
# stationarity.R
# Stationarity testing and transformation functions.
# =============================================================================

library(dplyr)
library(tseries)
library(forecast)

# Transformation code labels (index = code + 1)
TRANSFORM_LABELS <- c(
  "none", "log", "diff", "logdiff",
  "seasonal_diff", "log_seasonal_diff",
  "diff_seasonal_diff", "logdiff_seasonal_diff",
  "detrend"
)

# -----------------------------------------------------------------------------
# test_transform()
# Run ADF, KPSS, and STL seasonality tests on a single series and recommend
# a transformation code (0–8).
#
# Returns: string label (e.g. "logdiff", "logdiff+seasonal_diff", "none")
# -----------------------------------------------------------------------------
test_transform <- function(x, freq = 12, alpha = 0.05) {
  x_clean <- stats::na.omit(x)
  if (length(x_clean) < 24) return("too_short")

  # 1. Seasonality via STL
  has_seasonality <- FALSE
  seas_strength   <- NA
  if (freq > 1 && length(x_clean) > freq * 2) {
    stl_obj <- tryCatch(
      stats::stl(ts(x_clean, frequency = freq), s.window = "periodic"),
      error = \(e) NULL
    )
    if (!is.null(stl_obj)) {
      seas  <- stl_obj$time.series[, "seasonal"]
      resid <- stl_obj$time.series[, "remainder"]
      seas_strength   <- max(0, 1 - stats::var(resid) / stats::var(seas + resid))
      has_seasonality <- seas_strength > 0.3
    }
  }

  # 2. Unit root tests
  adf_p  <- tryCatch(tseries::adf.test(x_clean)$p.value,  error = \(e) NA)
  kpss_p <- tryCatch(tseries::kpss.test(x_clean)$p.value, error = \(e) NA)

  # 3. Log feasibility
  positive_only <- all(x_clean > 0)

  # 4. Decision logic
  adf_stationary  <- !is.na(adf_p)  && adf_p  < alpha
  kpss_stationary <- !is.na(kpss_p) && kpss_p > alpha

  base <- if (adf_stationary && kpss_stationary) {
    "none"
  } else if (!adf_stationary && !kpss_stationary) {
    "diff"
  } else if (adf_stationary && !kpss_stationary) {
    "detrend"
  } else {
    "diff"
  }

  if (base == "diff" && positive_only) base <- "logdiff"

  if (has_seasonality) {
    base <- if (base == "none") "seasonal_diff" else paste0(base, "+seasonal_diff")
  }

  base
}

# -----------------------------------------------------------------------------
# test_transform_full()
# Like test_transform() but also returns p-values and seasonality strength
# for display in the analyst-facing table.
# -----------------------------------------------------------------------------
test_transform_full <- function(x, freq = 12, alpha = 0.05) {
  x_clean <- stats::na.omit(x)

  if (length(x_clean) < 24) {
    return(list(
      recommendation = "too_short",
      seasonal       = FALSE,
      seas_strength  = NA,
      adf_p          = NA,
      kpss_p         = NA,
      positive_only  = all(x_clean > 0)
    ))
  }

  has_seasonality <- FALSE
  seas_strength   <- NA
  if (freq > 1 && length(x_clean) > freq * 2) {
    stl_obj <- tryCatch(
      stats::stl(ts(x_clean, frequency = freq), s.window = "periodic"),
      error = \(e) NULL
    )
    if (!is.null(stl_obj)) {
      seas          <- stl_obj$time.series[, "seasonal"]
      resid         <- stl_obj$time.series[, "remainder"]
      seas_strength <- max(0, 1 - stats::var(resid) / stats::var(seas + resid))
      has_seasonality <- seas_strength > 0.3
    }
  }

  adf_p  <- tryCatch(tseries::adf.test(x_clean)$p.value,  error = \(e) NA)
  kpss_p <- tryCatch(tseries::kpss.test(x_clean)$p.value, error = \(e) NA)

  positive_only   <- all(x_clean > 0)
  adf_stationary  <- !is.na(adf_p)  && adf_p  < alpha
  kpss_stationary <- !is.na(kpss_p) && kpss_p > alpha

  base <- if (adf_stationary && kpss_stationary) "none"
  else if (!adf_stationary && !kpss_stationary)  "diff"
  else if (adf_stationary && !kpss_stationary)    "detrend"
  else                                             "diff"

  if (base == "diff" && positive_only) base <- "logdiff"
  if (has_seasonality) {
    base <- if (base == "none") "seasonal_diff" else paste0(base, "+seasonal_diff")
  }

  list(
    recommendation = base,
    seasonal       = has_seasonality,
    seas_strength  = round(seas_strength, 3),
    adf_p          = round(adf_p, 4),
    kpss_p         = round(kpss_p, 4),
    positive_only  = positive_only
  )
}

# -----------------------------------------------------------------------------
# label_to_code()
# Convert a transformation label string to its numeric code (0–8).
# -----------------------------------------------------------------------------
label_to_code <- function(label) {
  # Strip compound suffix for matching base
  base_label <- sub("\\+seasonal_diff$", "", label)

  code <- switch(base_label,
    "none"      = 0,
    "log"       = 1,
    "diff"      = 2,
    "logdiff"   = 3,
    "seasonal_diff"          = 4,
    "log_seasonal_diff"      = 5,
    "diff_seasonal_diff"     = 6,
    "logdiff_seasonal_diff"  = 7,
    "detrend"   = 8,
    0  # fallback
  )

  # Compound: logdiff + seasonal_diff → code 7
  if (grepl("\\+seasonal_diff$", label) && base_label == "logdiff") code <- 7
  if (grepl("\\+seasonal_diff$", label) && base_label == "diff")    code <- 6

  code
}

# -----------------------------------------------------------------------------
# transform_block()
# Apply a vector of transformation codes to all non-Date columns of a block.
#
# Args:
#   block_df     : data.frame with 'Date' column
#   codes_vector : integer vector (0–8), one per data column
#   freq         : seasonal frequency (default 12)
#
# Returns: list with $data (transformed data.frame) and $info (metadata)
# -----------------------------------------------------------------------------
transform_block <- function(block_df, codes_vector, freq = 12) {
  block_df <- as.data.frame(block_df)
  if (!"Date" %in% names(block_df)) stop("Date column missing.")

  vars <- setdiff(names(block_df), "Date")
  if (length(vars) != length(codes_vector)) stop("codes_vector length mismatch.")

  out_df    <- block_df
  info_list <- list()

  for (i in seq_along(vars)) {
    varname <- vars[i]
    code    <- codes_vector[i]
    x       <- out_df[[varname]]
    n       <- length(x)
    result  <- rep(NA_real_, n)

    transformed <- tryCatch({
      if      (code == 0) { x }
      else if (code == 1) { log(x) }
      else if (code == 2) { result[2:n] <- diff(x); result }
      else if (code == 3) { lx <- log(x); result[2:n] <- diff(lx); result }
      else if (code == 4) {
        if (n > freq) result[(freq+1):n] <- x[(freq+1):n] - x[1:(n-freq)]
        result
      }
      else if (code == 5) {
        if (n > freq) {
          sd <- x[(freq+1):n] - x[1:(n-freq)]
          result[(freq+1):n] <- log(sd)
        }
        result
      }
      else if (code == 6) {
        if (n > freq+1) {
          sd <- x[(freq+1):n] - x[1:(n-freq)]
          result[(freq+2):n] <- diff(sd)
        }
        result
      }
      else if (code == 7) {
        if (n > freq+1) {
          lx <- log(x); ld <- diff(lx)
          result[(freq+2):n] <- ld[(freq+1):(n-1)] - ld[1:(n-freq-1)]
        }
        result
      }
      else if (code == 8) {
        fnna <- min(which(!is.na(x))); lnna <- max(which(!is.na(x)))
        x_seg <- x[fnna:lnna]; t_seg <- seq_len(length(x_seg))
        detr  <- stats::residuals(stats::lm(x_seg ~ t_seg))
        result[fnna:lnna] <- detr
        result
      }
      else stop("Unknown code: ", code)

    }, error = function(e) {
      warning("Failed '", varname, "': ", e$message)
      rep(NA_real_, n)
    })

    # Enforce correct length
    if (length(transformed) != n) {
      tmp <- rep(NA_real_, n)
      tmp[seq_along(transformed)] <- transformed
      transformed <- tmp
    }

    out_df[[varname]] <- transformed

    info_list[[varname]] <- data.frame(
      variable       = varname,
      code           = code,
      transformation = TRANSFORM_LABELS[code + 1],
      stringsAsFactors = FALSE
    )
  }

  list(data = out_df, info = dplyr::bind_rows(info_list))
}

# -----------------------------------------------------------------------------
# test_block_stationarity()
# Run test_transform_full() on every column of a data.frame.
# Returns a summary data.frame suitable for display in the Shiny table.
# -----------------------------------------------------------------------------
test_block_stationarity <- function(block_df, freq = 12, alpha = 0.05) {
  vars <- setdiff(names(block_df), "Date")

  rows <- lapply(vars, function(v) {
    res <- test_transform_full(block_df[[v]], freq = freq, alpha = alpha)
    data.frame(
      Variable       = v,
      Seasonal       = res$seasonal,
      Seas_Strength  = res$seas_strength,
      ADF_p          = res$adf_p,
      KPSS_p         = res$kpss_p,
      Recommended    = res$recommendation,
      stringsAsFactors = FALSE
    )
  })

  dplyr::bind_rows(rows)
}
