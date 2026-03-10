# =============================================================================
# price_adjustment.R
# CPI and FX price adjustment functions.
# =============================================================================

library(dplyr)

# -----------------------------------------------------------------------------
# adjust_block_for_cpi()
# Convert nominal values to real values by dividing by (CPI / 100).
#
# Args:
#   block_df : data.frame with a 'Date' column and numeric variable columns
#   cpi_df   : data.frame with 'Date' and 'CPI' columns
#   cols     : "all" to adjust every non-Date column, or a character vector
#              of specific column names to adjust
#
# Returns: block_df with selected columns adjusted to real values
# -----------------------------------------------------------------------------
adjust_block_for_cpi <- function(block_df, cpi_df, cols = "all") {
  df <- block_df |>
    dplyr::left_join(cpi_df |> dplyr::select(Date, CPI), by = "Date")

  if (any(is.na(df$CPI))) {
    warning("NA values in CPI after merging — check date alignment.")
  }

  cols_to_adjust <- if (identical(cols, "all")) {
    setdiff(names(df), c("Date", "CPI"))
  } else {
    cols
  }

  for (col in cols_to_adjust) {
    if (!col %in% names(df)) {
      warning("Column '", col, "' not found — skipped.")
      next
    }
    df[[col]] <- df[[col]] / (df$CPI / 100)
  }

  df |> dplyr::select(-CPI)
}

# -----------------------------------------------------------------------------
# apply_price_adjustments()
# Apply all CPI / FX adjustments to a named list of blocks.
# Mirrors the logic from transformations.qmd Section 2.
#
# Args:
#   blocks_raw : named list of raw data.frames (output of read_raw_data())
#
# Returns: named list with price-adjusted blocks (blocks_real)
# -----------------------------------------------------------------------------
apply_price_adjustments <- function(blocks_raw) {
  blocks_real <- blocks_raw

  # Tax blocks — full CPI adjustment
  tax_blocks <- c(
    "personal_labor_income_taxes",
    "corporate_business_tax",
    "consumption_tax",
    "import_trade_tax"
  )

  for (blk in tax_blocks) {
    if (blk %in% names(blocks_raw)) {
      blocks_real[[blk]] <- adjust_block_for_cpi(
        blocks_raw[[blk]],
        blocks_raw$adjusters
      )
    }
  }

  # Real estate — partial CPI adjustment (tax columns only)
  if ("real_estate" %in% names(blocks_raw)) {
    re_cols <- c(
      "Real estate taxation",
      "Property tax",
      "praise tax",
      "Real estate purchase tax",
      "praise tax returns",
      "purchase returns"
    )
    blocks_real$real_estate <- adjust_block_for_cpi(
      blocks_raw$real_estate,
      blocks_raw$adjusters,
      cols = re_cols
    )
  }

  # Oil: convert to ILS using FX, then CPI-adjust
  if ("real_activity" %in% names(blocks_raw) && "FX_liqudity" %in% names(blocks_raw)) {
    blocks_real$real_activity$Oil <-
      blocks_raw$real_activity$Oil * blocks_raw$FX_liqudity$Dollar

    blocks_real$real_activity <- adjust_block_for_cpi(
      blocks_real$real_activity,
      blocks_raw$adjusters,
      cols = "Oil"
    )
  }

  # FX reserves: convert to ILS using FX, then CPI-adjust
  if ("FX_liqudity" %in% names(blocks_raw)) {
    fx_col <- "Foreign exchange reserves (millions of dollars)"
    blocks_real$FX_liqudity[[fx_col]] <-
      blocks_raw$FX_liqudity[[fx_col]] * blocks_raw$FX_liqudity$Dollar

    blocks_real$FX_liqudity <- adjust_block_for_cpi(
      blocks_real$FX_liqudity,
      blocks_raw$adjusters,
      cols = fx_col
    )
  }

  blocks_real
}
