# =============================================================================
# seasonal_adjustment.R
# X-13 seasonal adjustment via the {seasonal} package.
# =============================================================================

library(dplyr)
library(lubridate)
library(seasonal)
library(readxl)

# -----------------------------------------------------------------------------
# load_sa_regressors()
# Load trading-day and holiday regressors used by X-13.
#
# Args:
#   td_path  : path to td_var.csv
#   hol_path : path to hol_preadj.xlsx
#
# Returns: list with $td_ts and $hag_ts (ts objects, frequency = 12)
# -----------------------------------------------------------------------------
load_sa_regressors <- function(td_path, hol_path) {
  td <- utils::read.csv(td_path)
  td_ts <- ts(
    td[, -1],
    start     = c(lubridate::year(min(as.Date(td[[1]]))),
                  lubridate::month(min(as.Date(td[[1]])))),
    frequency = 12
  )

  preadj <- readxl::read_excel(hol_path) |>
    dplyr::mutate(date = as.Date(date))

  hag_ts <- ts(
    preadj[, -1],
    start     = c(lubridate::year(min(preadj$date)),
                  lubridate::month(min(preadj$date))),
    frequency = 12
  )

  list(td_ts = td_ts, hag_ts = hag_ts)
}

# -----------------------------------------------------------------------------
# seasonal_adjust_block()
# Apply X-13 seasonal adjustment to selected columns of a block.
#
# Args:
#   block_df      : data.frame with 'Date' column
#   columns       : "all" or character vector of column names
#   hag_ts        : holiday regressor ts object
#   td_ts         : trading-day regressor ts object
#   x11           : logical — use X-11 filter (TRUE) or SEATS (FALSE)
#   outlier_types : X-13 outlier type string, default "ao"
#
# Returns: list with $data (adjusted data.frame) and $info (metadata data.frame)
# -----------------------------------------------------------------------------
seasonal_adjust_block <- function(block_df,
                                  columns       = "all",
                                  hag_ts,
                                  td_ts,
                                  x11           = TRUE,
                                  outlier_types = "ao") {
  if (!"Date" %in% names(block_df)) stop("Block must contain a 'Date' column.")

  vars <- if (identical(columns, "all")) setdiff(names(block_df), "Date") else columns

  info_list <- list()
  df_out    <- block_df

  for (v in vars) {
    message("SA: ", v)
    x <- df_out[[v]]

    # Skip all-NA
    if (all(is.na(x))) {
      info_list[[v]] <- .sa_info_row(v, "Skipped - all NA")
      next
    }

    first_non_na <- min(which(!is.na(x)))
    last_non_na  <- max(which(!is.na(x)))
    segment      <- x[first_non_na:last_non_na]

    # Skip series with internal NAs (X-13 cannot handle them)
    if (any(is.na(segment))) {
      info_list[[v]] <- .sa_info_row(v, "Skipped - internal NA")
      next
    }

    y <- ts(
      segment,
      start     = c(lubridate::year(block_df$Date[first_non_na]),
                    lubridate::month(block_df$Date[first_non_na])),
      frequency = 12
    )

    fit <- tryCatch({
      if (x11) {
        seasonal::seas(y, x11 = "", outlier.types = outlier_types,
                       transform.function = "auto",
                       xreg = cbind(hag_ts, td_ts),
                       regression.usertype = rep("holiday", ncol(hag_ts)))
      } else {
        seasonal::seas(y, outlier.types = outlier_types,
                       transform.function = "auto",
                       xreg = cbind(hag_ts, td_ts),
                       regression.usertype = rep("holiday", ncol(hag_ts)))
      }
    }, error = function(e) {
      info_list[[v]] <<- .sa_info_row(v, paste("SA failed:", e$message))
      NULL
    })

    if (is.null(fit)) next

    trans_used   <- tryCatch(as.character(fit$series$transfunc)[1], error = \(e) NA)
    arima_used   <- tryCatch(paste(as.character(fit$est$arima), collapse = "; "), error = \(e) NA)
    outliers_used <- tryCatch(paste(as.character(fit$outlier$Type), collapse = ", "), error = \(e) NA)

    info_list[[v]] <- data.frame(
      variable = v, status = "OK",
      transform = trans_used, arima = arima_used, outliers = outliers_used,
      stringsAsFactors = FALSE
    )

    adj <- as.numeric(predict(fit))
    result_vec <- x
    result_vec[first_non_na:last_non_na] <- adj
    df_out[[v]] <- result_vec
  }

  list(data = df_out, info = dplyr::bind_rows(info_list))
}

# Internal helper
.sa_info_row <- function(v, status) {
  data.frame(variable = v, status = status,
             transform = NA, arima = NA, outliers = NA,
             stringsAsFactors = FALSE)
}
