# =============================================================================
# bridge_model.R
# XGBoost bridge model: DFM factors → GDP nowcast.
# =============================================================================

library(xgboost)
library(dplyr)
library(lubridate)

# -----------------------------------------------------------------------------
# train_bridge_model()
# Train an XGBoost model using DFM factors as predictors of the target variable.
#
# Args:
#   dfm_obj        : DFM object from the final training-period run (provides F_qml)
#   df             : full panel data.frame (provides target values)
#   target_var     : name of the quarterly target column (default "GDP")
#   train_start_row: first row of F_qml to use for training
#   train_end_row  : last row of F_qml to use for training
#   data_start_date: date of the first row in F_qml (used to reconstruct dates)
#   n_factors      : number of factors (default 4)
#   params         : named list of XGBoost parameters
#   nrounds        : number of boosting rounds
#
# Returns: list with $model (xgb model) and $train_dates (data.frame)
# -----------------------------------------------------------------------------
train_bridge_model <- function(dfm_obj,
                               df,
                               target_var      = "GDP",
                               train_start_row = 5L,
                               train_end_row   = 309L,
                               data_start_date = as.Date("1995-07-01"),
                               n_factors       = 4L,
                               params          = list(
                                 objective        = "reg:squarederror",
                                 eval_metric      = "rmse",
                                 eta              = 0.015,
                                 max_depth        = 4L,
                                 subsample        = 0.8,
                                 colsample_bytree = 0.8
                               ),
                               nrounds = 300L) {

  f_names <- paste0("f", seq_len(n_factors))

  # Extract factor values for training rows
  x_train_raw <- dfm_obj$F_qml[train_start_row:train_end_row, ]
  n_train     <- nrow(x_train_raw)
  dates_train <- seq(data_start_date, by = "month", length.out = n_train)

  x_train <- data.frame(Date = dates_train, x_train_raw)
  colnames(x_train)[-1] <- f_names

  # Keep only quarter-end months
  x_train$month <- as.integer(format(x_train$Date, "%m"))
  x_train <- x_train[x_train$month %in% c(1, 4, 7, 10), ]

  # Match target values
  y_train <- df[df$Date <= df$Date[train_end_row] & !is.na(df[[target_var]]),
                c("Date", target_var)]

  X_mat <- as.matrix(x_train[, f_names])
  y_vec <- y_train[[target_var]]

  dtrain  <- xgboost::xgb.DMatrix(data = X_mat, label = y_vec)
  model   <- xgboost::xgb.train(params = params, data = dtrain, nrounds = nrounds)

  list(model = model, train_dates = x_train[, c("Date", f_names)])
}

# -----------------------------------------------------------------------------
# predict_bridge_model()
# Use a trained XGBoost bridge model to predict from factor factor data.
#
# Args:
#   model         : xgb model object from train_bridge_model()
#   factors_report: data.frame from run_rolling_dfm_factors() — h0_f1..h0_f4 cols
#   n_factors     : number of factors used (default 4)
#   df            : full panel data.frame (for joining actual target values)
#   target_var    : name of target column (default "GDP")
#
# Returns: data.frame with Date, GDP_FCST, and actual target column
# -----------------------------------------------------------------------------
predict_bridge_model <- function(model,
                                 factors_report,
                                 n_factors  = 4L,
                                 df         = NULL,
                                 target_var = "GDP") {

  f_cols  <- paste0("h0_f", seq_len(n_factors))
  col_map <- paste0("f",    seq_len(n_factors))

  # Keep only rows where h0 factors are available
  valid <- factors_report[
    rowSums(is.na(factors_report[, f_cols])) < n_factors, ]

  X_test <- as.matrix(valid[, f_cols])
  colnames(X_test) <- col_map

  preds <- predict(model, xgboost::xgb.DMatrix(data = X_test))

  results <- data.frame(Date = valid$Date, GDP_FCST = preds)

  if (!is.null(df)) {
    results <- dplyr::left_join(results, df[, c("Date", target_var)], by = "Date")
  }

  results
}
