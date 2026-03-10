# =============================================================================
# mod_modeling.R
# Step 4 — DFM Configuration & Run
# Displays ICr / VARselect diagnostics to guide hyperparameter choice,
# then runs the rolling-window DFM (and optionally the XGBoost bridge model).
# =============================================================================

library(shiny)
library(DT)
library(xts)

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
mod_modeling_ui <- function(id) {
  ns <- NS(id)

  tagList(
    h3("Step 4 — DFM Configuration & Run"),

    fluidRow(
      # --- Left panel: hyperparameter selection ---
      column(4,
        wellPanel(
          h4("Hyperparameter Selection"),
          p("Run the diagnostics below to help choose the number of factors (r)
             and lags (p). The ICr plot shows an 'elbow' indicating ideal r;
             VARselect criteria indicate ideal p."),

          numericInput(ns("icr_n_factors"), "Factors to test in VARselect (n)",
                       value = 4, min = 1, max = 10),
          actionButton(ns("run_icr"), "Run ICr & VARselect",
                       class = "btn-secondary", icon = icon("search")),

          hr(),
          h4("Model Parameters"),
          numericInput(ns("r"), "Number of factors (r)", value = 4, min = 1, max = 20),
          numericInput(ns("p"), "Number of lags (p)",    value = 2, min = 1, max = 20),
          dateInput(ns("start_date"), "Nowcast window start", value = "2021-01-01"),
          dateInput(ns("end_date"),   "Nowcast window end",   value = "2025-01-01"),

          hr(),
          h4("Bridge Model (XGBoost)"),
          checkboxInput(ns("use_bridge"), "Enable XGBoost bridge model", value = TRUE),

          hr(),
          actionButton(ns("run_model"), "Run Model",
                       class = "btn-primary btn-lg btn-block",
                       icon  = icon("play")),
          uiOutput(ns("run_status"))
        )
      ),

      # --- Right panel: diagnostics ---
      column(8,
        h4("ICr — Factor Selection"),
        p("The 'knee' in the plot below suggests the ideal number of factors."),
        plotOutput(ns("icr_plot"), height = "250px"),

        h4("VARselect — Lag Selection"),
        p("Look for the minimum of AIC / HQ to choose p."),
        plotOutput(ns("varselect_plot"), height = "200px"),

        verbatimTextOutput(ns("varselect_text"))
      )
    )
  )
}

# -----------------------------------------------------------------------------
# Server
# -----------------------------------------------------------------------------
mod_modeling_server <- function(id, transform_data) {
  moduleServer(id, function(input, output, session) {
    ns <- session$ns

    icr_result_rv <- reactiveVal(NULL)
    model_out_rv  <- reactiveVal(NULL)

    # ------------------------------------------------------------------
    # Run ICr and VARselect
    # ------------------------------------------------------------------
    observeEvent(input$run_icr, {
      td <- transform_data()
      req(td)

      df         <- td$df
      target_var <- td$target_var

      xts_data <- xts::xts(
        as.matrix(df[, !names(df) %in% "Date"]),
        order.by = df$Date
      )

      withProgress(message = "Running ICr...", value = 0.5, {
        icr <- tryCatch(run_icr(xts_data), error = \(e) { showNotification(e$message, type = "error"); NULL })
      })

      req(icr)
      icr_result_rv(icr)
    })

    output$icr_plot <- renderPlot({
      req(icr_result_rv())
      plot(icr_result_rv())
    })

    output$varselect_plot <- renderPlot({
      req(icr_result_rv())
      vs <- tryCatch(run_varselect(icr_result_rv(), input$icr_n_factors), error = \(e) NULL)
      req(vs)
      plot(vs$criteria["AIC(n)", ], type = "b", main = "VARselect AIC",
           xlab = "Lags (p)", ylab = "AIC")
    })

    output$varselect_text <- renderPrint({
      req(icr_result_rv())
      vs <- tryCatch(run_varselect(icr_result_rv(), input$icr_n_factors), error = \(e) NULL)
      req(vs)
      cat("VARselect criteria summary:\n")
      print(vs$selection)
    })

    # ------------------------------------------------------------------
    # Run the DFM (and optionally the bridge model)
    # ------------------------------------------------------------------
    observeEvent(input$run_model, {
      td <- transform_data()
      req(td)

      df         <- td$df
      target_var <- td$target_var

      start_date <- input$start_date
      end_date   <- input$end_date
      r          <- input$r
      p          <- input$p

      # Progress callback for reactive updates
      progress <- shiny::Progress$new()
      on.exit(progress$close())
      progress$set(message = "Running rolling-window DFM...", value = 0)

      progress_fn <- function(i, total) {
        progress$set(value = i / total,
                     detail = paste0(i, " / ", total, " windows"))
      }

      results <- tryCatch(
        run_rolling_dfm(
          df         = df,
          target_var = target_var,
          n_factors  = r,
          n_lags     = p,
          start_date = start_date,
          end_date   = end_date,
          progress_fn = progress_fn
        ),
        error = function(e) {
          showNotification(paste("DFM error:", e$message), type = "error")
          NULL
        }
      )
      req(results)

      q_fcst <- extract_quarterly_forecasts(results, df, target_var)

      bridge_results <- NULL
      if (input$use_bridge) {
        progress$set(message = "Running bridge model (DFM factors)...", value = 0)
        fac_report <- tryCatch(
          run_rolling_dfm_factors(
            df         = df,
            target_var = target_var,
            n_factors  = r,
            n_lags     = p,
            start_date = start_date - months(12),
            end_date   = end_date,
            progress_fn = progress_fn
          ),
          error = \(e) NULL
        )

        if (!is.null(fac_report)) {
          # Train on the full data up to start_date
          df_train <- df[df$Date < start_date, ]
          xts_train <- xts::xts(
            as.matrix(df_train[, !names(df_train) %in% "Date"]),
            order.by = df_train$Date
          )
          dfm_train <- tryCatch(
            dfms::DFM(X = xts_train, r = r, p = p,
                      quarterly.vars = target_var, em.method = "BM"),
            error = \(e) NULL
          )

          if (!is.null(dfm_train)) {
            train_rows <- nrow(dfm_train$F_qml)
            bridge_fit <- tryCatch(
              train_bridge_model(
                dfm_obj         = dfm_train,
                df              = df,
                target_var      = target_var,
                train_start_row = 1L,
                train_end_row   = train_rows,
                data_start_date = df$Date[1],
                n_factors       = r
              ),
              error = \(e) NULL
            )

            if (!is.null(bridge_fit)) {
              bridge_results <- tryCatch(
                predict_bridge_model(bridge_fit$model, fac_report,
                                     n_factors = r, df = df,
                                     target_var = target_var),
                error = \(e) NULL
              )
            }
          }
        }
      }

      model_out_rv(list(
        results        = results,
        quarterly_fcst = q_fcst,
        bridge_results = bridge_results,
        target_var     = target_var,
        r              = r,
        p              = p
      ))
    })

    output$run_status <- renderUI({
      req(model_out_rv())
      div(class = "alert alert-success mt-2",
          icon("check-circle"), " Model run complete. Proceed to Step 5.")
    })

    return(reactive({ model_out_rv() }))
  })
}
