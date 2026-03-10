# =============================================================================
# server.R
# Shiny app server — wires all 5 modules together with reactive data flow.
# =============================================================================

library(shiny)

# Core functions (sourced once so all modules can access them)
source("R/core/config.R")
source("R/core/utils/io.R")
source("R/core/transformations/price_adjustment.R")
source("R/core/transformations/seasonal_adjustment.R")
source("R/core/transformations/stationarity.R")
source("R/core/transformations/release_lags.R")
source("R/core/modeling/hyperparameters.R")
source("R/core/modeling/dfm.R")
source("R/core/modeling/bridge_model.R")
source("R/core/modeling/evaluation.R")

# Shiny modules
source("R/shiny/modules/mod_upload.R")
source("R/shiny/modules/mod_freq_target.R")
source("R/shiny/modules/mod_transforms.R")
source("R/shiny/modules/mod_modeling.R")
source("R/shiny/modules/mod_results.R")

server <- function(input, output, session) {

  # ── Step 1: Upload & Validate ─────────────────────────────────────────
  # Returns: reactive data.frame (validated, Date column normalised)
  uploaded_data <- mod_upload_server("upload")

  # ── Step 2: Frequency Detection & Target Selection ────────────────────
  # Returns: reactive list(df, target_var, freq_table)
  freq_target_data <- mod_freq_target_server("freq_target", uploaded_data)

  # ── Step 3: Transformation Recommendations & Apply ───────────────────
  # Returns: reactive list(df, target_var, info)
  transform_data <- mod_transforms_server("transforms", freq_target_data)

  # ── Step 4: DFM Configuration & Run ──────────────────────────────────
  # Returns: reactive list(results, quarterly_fcst, bridge_results, ...)
  model_data <- mod_modeling_server("modeling", transform_data)

  # ── Step 5: Results, Metrics & Export ────────────────────────────────
  mod_results_server("results", model_data)
}
