# =============================================================================
# dfm.R
# Rolling-window Dynamic Factor Model (DFM) nowcasting.
# =============================================================================

library(dfms)
library(xts)
library(lubridate)
library(dplyr)

# -----------------------------------------------------------------------------
# unscale_var()
# Reverse the standardisation applied internally by DFM to recover real units.
# -----------------------------------------------------------------------------
unscale_var <- function(dfm_obj, var_name, std_value) {
  stats_mat <- attr(dfm_obj$X_imp, "stats")
  idx       <- which(colnames(dfm_obj$X_imp) == var_name)
  mu        <- stats_mat[idx, "Mean"]
  sigma     <- stats_mat[idx, "SD"]
  std_value * sigma + mu
}

# -----------------------------------------------------------------------------
# dynamic_horizon()
# Return the forecast horizon h for a given cutoff month.
#   Month in (1,4,7,10) → h = 3  (three months before quarter end)
#   Month in (2,5,8,11) → h = 2
#   Month in (3,6,9,12) → h = 1  (end of quarter)
# -----------------------------------------------------------------------------
dynamic_horizon <- function(date) {
  m <- lubridate::month(date)
  if      (m %in% c(1, 4, 7, 10)) 3L
  else if (m %in% c(2, 5, 8, 11)) 2L
  else                              1L
}

# -----------------------------------------------------------------------------
# run_rolling_dfm()
# Execute the rolling-window DFM nowcast over a date range.
#
# Args:
#   df            : data.frame with Date column + all predictors + target
#   target_var    : name of the quarterly target variable (default "GDP")
#   n_factors     : number of latent factors r (default 4)
#   n_lags        : number of lags p (default 2)
#   em_method     : EM algorithm ("BM" or "DK")
#   start_date    : first date in the rolling window
#   end_date      : last date in the rolling window
#   progress_fn   : optional function(i, total) for progress updates in Shiny
#
# Returns: data.frame with columns Date, h0_fcst, h1_fcst, h2_fcst, h3_fcst
# -----------------------------------------------------------------------------
run_rolling_dfm <- function(df,
                            target_var  = "GDP",
                            n_factors   = 4L,
                            n_lags      = 2L,
                            em_method   = "BM",
                            start_date  = as.Date("2021-01-01"),
                            end_date    = as.Date("2025-01-01"),
                            progress_fn = NULL) {

  all_months <- seq(start_date, end_date, by = "month")

  results <- data.frame(
    Date     = all_months,
    h0_fcst  = NA_real_,
    h1_fcst  = NA_real_,
    h2_fcst  = NA_real_,
    h3_fcst  = NA_real_
  )

  loop_months <- utils::head(all_months, -1)
  total       <- length(loop_months)

  for (i in seq_along(loop_months)) {
    if (is.function(progress_fn)) progress_fn(i, total)

    cutoff <- loop_months[i]
    h_val  <- dynamic_horizon(cutoff)

    df_sub <- df[df$Date <= cutoff, ]
    X_xts  <- xts::xts(
      as.matrix(df_sub[, !names(df_sub) %in% "Date"]),
      order.by = df_sub$Date
    )

    # Mask current-period GDP so the model must forecast it
    X_xts[nrow(X_xts), target_var] <- NA

    dfm_curr <- tryCatch(
      dfms::DFM(X = X_xts, r = n_factors, p = n_lags,
                quarterly.vars = target_var, em.method = em_method),
      error = function(e) { message("DFM failed at ", cutoff, ": ", e$message); NULL }
    )
    if (is.null(dfm_curr)) next

    pred           <- predict(dfm_curr, h = h_val, standardized = FALSE)
    gdp_fcst       <- pred$X_fcst[h_val, target_var]
    x_std          <- utils::tail(dfm_curr$X_imp[, target_var], 1)
    gdp_now_day_of <- unscale_var(dfm_curr, target_var, x_std)

    results$h0_fcst[i] <- gdp_now_day_of
    if (h_val == 1) results$h1_fcst[i + h_val] <- gdp_fcst
    if (h_val == 2) results$h2_fcst[i + h_val] <- gdp_fcst
    if (h_val == 3) results$h3_fcst[i + h_val] <- gdp_fcst
  }

  results
}

# -----------------------------------------------------------------------------
# extract_quarterly_forecasts()
# Filter the rolling results to quarter-end months and join actual target values.
# -----------------------------------------------------------------------------
extract_quarterly_forecasts <- function(results, df, target_var = "GDP") {
  results$month <- as.integer(format(results$Date, "%m"))
  qfcst <- results[results$month %in% c(1, 4, 7, 10), ]

  qfcst <- dplyr::left_join(
    qfcst,
    df[, c("Date", target_var)],
    by = "Date"
  )

  qfcst$month <- NULL
  qfcst
}

# -----------------------------------------------------------------------------
# run_rolling_dfm_factors()
# Rolling-window run that extracts and returns factor values per horizon
# (used as inputs to the XGBoost bridge model).
# -----------------------------------------------------------------------------
run_rolling_dfm_factors <- function(df,
                                    target_var  = "GDP",
                                    n_factors   = 4L,
                                    n_lags      = 2L,
                                    em_method   = "BM",
                                    start_date  = as.Date("2020-12-01"),
                                    end_date    = as.Date("2025-01-01"),
                                    progress_fn = NULL) {

  all_months <- seq(start_date, end_date, by = "month")

  f_cols <- unlist(lapply(0:3, function(h)
    paste0("h", h, "_f", seq_len(n_factors))
  ))

  results <- as.data.frame(matrix(NA_real_, nrow = length(all_months),
                                   ncol = 1 + length(f_cols)))
  names(results) <- c("Date", f_cols)
  results$Date   <- all_months

  loop_months <- utils::head(all_months, -1)
  total        <- length(loop_months)

  for (i in seq_along(loop_months)) {
    if (is.function(progress_fn)) progress_fn(i, total)

    cutoff <- loop_months[i]
    h_val  <- dynamic_horizon(cutoff)

    df_sub <- df[df$Date <= cutoff, ]
    X_xts  <- xts::xts(
      as.matrix(df_sub[, !names(df_sub) %in% "Date"]),
      order.by = df_sub$Date
    )
    X_xts[nrow(X_xts), target_var] <- NA

    dfm_curr <- tryCatch(
      dfms::DFM(X = X_xts, r = n_factors, p = n_lags,
                quarterly.vars = target_var, em.method = em_method),
      error = function(e) NULL
    )
    if (is.null(dfm_curr)) next

    pred  <- predict(dfm_curr, h = h_val, standardized = TRUE)
    F_h0  <- as.numeric(utils::tail(dfm_curr$F_qml, 1))
    F_h   <- as.numeric(pred$F[h_val, ])

    if (h_val == 1) results[i + h_val, paste0("h1_f", seq_len(n_factors))] <- F_h
    if (h_val == 2) results[i + h_val, paste0("h2_f", seq_len(n_factors))] <- F_h
    if (h_val == 3) {
      results[i + h_val, paste0("h3_f", seq_len(n_factors))] <- F_h
      results[i,         paste0("h0_f", seq_len(n_factors))] <- F_h0
    }
  }

  results
}
