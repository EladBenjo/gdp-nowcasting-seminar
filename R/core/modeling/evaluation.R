# =============================================================================
# evaluation.R
# Forecast evaluation metrics: RMSE, MAE, directional accuracy.
# =============================================================================

library(dplyr)

# -----------------------------------------------------------------------------
# compute_metrics()
# Calculate RMSE and MAE for one forecast column vs. actuals.
# -----------------------------------------------------------------------------
compute_metrics <- function(actual, forecast) {
  valid <- !is.na(actual) & !is.na(forecast)
  if (sum(valid) == 0) return(list(RMSE = NA, MAE = NA, n = 0L))

  err  <- forecast[valid] - actual[valid]
  rmse <- sqrt(mean(err^2))
  mae  <- mean(abs(err))

  list(RMSE = round(rmse, 6), MAE = round(mae, 6), n = sum(valid))
}

# -----------------------------------------------------------------------------
# compute_all_metrics()
# Compute RMSE, MAE, and directional accuracy for all forecast horizons.
#
# Args:
#   quarterly_fcst : data.frame with columns Date, GDP (actual),
#                    h0_fcst, h1_fcst, h2_fcst, h3_fcst
#   target_var     : name of the actual target column (default "GDP")
#
# Returns: data.frame with one row per horizon
# -----------------------------------------------------------------------------
compute_all_metrics <- function(quarterly_fcst, target_var = "GDP") {
  horizons <- c("h0", "h1", "h2", "h3")
  actual   <- quarterly_fcst[[target_var]]

  # Directional accuracy helper
  dir_acc <- function(act, fcst) {
    act_d  <- sign(act  - dplyr::lag(act))
    fcst_d <- sign(fcst - dplyr::lag(fcst))
    mean(act_d == fcst_d, na.rm = TRUE)
  }

  rows <- lapply(horizons, function(h) {
    col  <- paste0(h, "_fcst")
    fcst <- quarterly_fcst[[col]]
    m    <- compute_metrics(actual, fcst)
    data.frame(
      Horizon            = h,
      RMSE               = m$RMSE,
      MAE                = m$MAE,
      N_obs              = m$n,
      Directional_Acc    = round(dir_acc(actual, fcst), 4),
      stringsAsFactors   = FALSE
    )
  })

  dplyr::bind_rows(rows)
}

# -----------------------------------------------------------------------------
# compute_factor_loadings()
# Extract factor loadings from a DFM object and return a tidy data.frame.
#
# Args:
#   dfm_obj   : DFM result object
#   var_names : character vector of variable names (colnames of the panel)
#   n_factors : number of factors to extract (default 4)
#
# Returns: data.frame with columns variable, factor, loading, abs_loading
# -----------------------------------------------------------------------------
compute_factor_loadings <- function(dfm_obj, var_names, n_factors = 4) {
  rows <- lapply(seq_len(n_factors), function(f) {
    data.frame(
      variable    = var_names,
      factor      = paste0("F", f),
      loading     = dfm_obj$C[, f],
      abs_loading = abs(dfm_obj$C[, f]),
      stringsAsFactors = FALSE
    )
  })

  dplyr::bind_rows(rows)
}
