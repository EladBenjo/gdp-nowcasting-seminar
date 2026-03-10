# =============================================================================
# hyperparameters.R
# Factor and lag selection via ICr (information criteria) and VARselect.
# =============================================================================

library(dfms)
library(vars)
library(xts)

# -----------------------------------------------------------------------------
# run_icr()
# Run the ICr information criteria to help select the number of factors (r).
#
# Args:
#   xts_data : xts object — the full data panel
#
# Returns: ICr result object (printable, plottable via plot() and screeplot())
# -----------------------------------------------------------------------------
run_icr <- function(xts_data) {
  dfms::ICr(xts_data)
}

# -----------------------------------------------------------------------------
# run_varselect()
# Run VARselect on the first n_factors PCA factors to help select lag order p.
#
# Args:
#   icr_result : output of run_icr()
#   n_factors  : number of factors to pass to VARselect (default 4)
#
# Returns: VARselect result (contains AIC, HQ, SC, FPE criteria)
# -----------------------------------------------------------------------------
run_varselect <- function(icr_result, n_factors = 4) {
  vars::VARselect(icr_result$F_pca[, seq_len(n_factors)])
}

# -----------------------------------------------------------------------------
# suggest_hyperparams()
# Extract a simple summary of recommended r and p from ICr and VARselect.
# Returns a list with $r_suggestion and $p_suggestion (numeric).
# -----------------------------------------------------------------------------
suggest_hyperparams <- function(icr_result, var_result) {
  # IC1 criterion from ICr (column 1) — pick the minimising index
  ic1 <- icr_result$IC[, 1]
  r_suggestion <- which.min(ic1)

  # SC criterion from VARselect (most parsimonious)
  p_suggestion <- var_result$selection["SC(n)"]

  list(r = r_suggestion, p = p_suggestion)
}

# -----------------------------------------------------------------------------
# df_to_xts()
# Convert a plain data.frame (with a Date column) to an xts object
# suitable for dfms::DFM().
# -----------------------------------------------------------------------------
df_to_xts <- function(df) {
  xts::xts(
    x        = as.matrix(df[, !names(df) %in% "Date"]),
    order.by = df$Date
  )
}
