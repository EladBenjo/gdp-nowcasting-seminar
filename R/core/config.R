# =============================================================================
# config.R
# Central configuration: all file paths and model parameters.
# Edit these values to match your environment before running the app.
# =============================================================================

# -----------------------------------------------------------------------------
# Data Paths
# -----------------------------------------------------------------------------

# Raw input data (multiple sheets / blocks)
RAW_DATA_PATH <- ""

# Pre-processed monthly-quarterly panel (used directly by the DFM)
PANEL_DATA_PATH <- ""

# Seasonal adjustment external regressors (trading day + holidays)
SA_TD_PATH  <- ""
SA_HOL_PATH <- ""

# Output directory for all generated Excel files
OUTPUT_DIR <- getwd()

# -----------------------------------------------------------------------------
# DFM Model Parameters (defaults — analyst can override in the app)
# -----------------------------------------------------------------------------

DFM_N_FACTORS    <- 4          # Number of latent factors (r)
DFM_N_LAGS       <- 2          # Number of lags (p)
DFM_EM_METHOD    <- "BM"       # EM algorithm: "BM" (Behrens-Maurer) or "DK"
DFM_QUARTERLY_VAR <- "GDP"     # Name of the quarterly target variable

# Rolling-window date range for nowcasting evaluation
NOWCAST_START <- as.Date("2021-01-01")
NOWCAST_END   <- as.Date("2025-01-01")

# -----------------------------------------------------------------------------
# XGBoost Bridge Model Parameters (defaults)
# -----------------------------------------------------------------------------

XGB_ETA             <- 0.015
XGB_MAX_DEPTH       <- 4
XGB_SUBSAMPLE       <- 0.8
XGB_COLSAMPLE       <- 0.8
XGB_NROUNDS         <- 300
XGB_TRAIN_END_ROW   <- 309    # Last row index used for training
XGB_TRAIN_START_ROW <- 5      # First row index used for training

# -----------------------------------------------------------------------------
# Stationarity Testing Parameters
# -----------------------------------------------------------------------------

STAT_ALPHA         <- 0.05    # Significance level for ADF/KPSS tests
STAT_SEAS_STRENGTH <- 0.3     # STL seasonality strength threshold
STAT_FREQ          <- 12      # Default time series frequency (monthly)
