# =============================================================================
# release_lags.R
# Release lag adjustment: shift data forward to reflect real publication delays.
# =============================================================================

# -----------------------------------------------------------------------------
# shift_by_vector()
# Shift each data column forward by k months (insert k NAs at top, drop tail).
#
# Args:
#   df  : data.frame where column 1 is 'Date'
#   vec : integer vector of lag values, one per data column (0 = no shift)
#
# Returns: list with $data (shifted data.frame) and $report (lag metadata)
# -----------------------------------------------------------------------------
shift_by_vector <- function(df, vec) {
  if (length(vec) != (ncol(df) - 1)) {
    stop("vec length must match the number of data columns (excluding Date).")
  }

  df_shifted <- df

  report <- data.frame(
    variable    = colnames(df)[-1],
    lag_applied = vec,
    stringsAsFactors = FALSE
  )

  for (i in seq_along(vec)) {
    k   <- vec[i]
    var <- colnames(df)[i + 1]

    if (!is.numeric(df[[var]])) {
      warning("Column '", var, "' is not numeric — skipped.")
      next
    }

    if (k > 0) {
      df_shifted[[var]] <- c(rep(NA_real_, k), df[[var]][1:(nrow(df) - k)])
    }
    # k == 0: no change needed
  }

  list(data = df_shifted, report = report)
}

# -----------------------------------------------------------------------------
# days_to_months()
# Helper: convert release lag in days to shift months using the standard rule:
#   < 30 days  → 1 month
#   30–60 days → 2 months
#   > 60 days  → 3 months
# -----------------------------------------------------------------------------
days_to_months <- function(days) {
  dplyr::case_when(
    days <  30 ~ 1L,
    days <= 60 ~ 2L,
    TRUE       ~ 3L
  )
}
