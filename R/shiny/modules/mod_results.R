# =============================================================================
# mod_results.R
# Step 5 — Results, Metrics, Factor Loadings & Export
# =============================================================================

library(shiny)
library(DT)
library(ggplot2)
library(plotly)
library(dplyr)

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
mod_results_ui <- function(id) {
  ns <- NS(id)

  tagList(
    h3("Step 5 — Results & Export"),

    tabsetPanel(
      # ── Forecast Plot ──────────────────────────────────────────────
      tabPanel("Forecast Plot",
        br(),
        plotlyOutput(ns("forecast_plot"), height = "450px"),
        br(),
        checkboxGroupInput(
          ns("show_horizons"),
          "Show horizons:",
          choices  = c("h0 (end-of-quarter)" = "h0_fcst",
                       "h1 (1m before EoQ)"  = "h1_fcst",
                       "h2 (2m before EoQ)"  = "h2_fcst",
                       "h3 (3m before EoQ)"  = "h3_fcst",
                       "Bridge model (h0)"   = "bridge"),
          selected = c("h1_fcst", "h2_fcst", "h3_fcst"),
          inline   = TRUE
        )
      ),

      # ── Metrics Table ─────────────────────────────────────────────
      tabPanel("Metrics",
        br(),
        h4("DFM Nowcast — Accuracy by Horizon"),
        DTOutput(ns("metrics_table")),
        br(),
        uiOutput(ns("bridge_metrics_ui")),
        DTOutput(ns("bridge_metrics_table"))
      ),

      # ── Factor Loadings ───────────────────────────────────────────
      tabPanel("Factor Loadings",
        br(),
        selectInput(ns("factor_sel"), "Select factor:", choices = NULL, width = "200px"),
        plotlyOutput(ns("loadings_plot"), height = "400px")
      ),

      # ── Export ────────────────────────────────────────────────────
      tabPanel("Export",
        br(),
        p("Download the results as Excel files."),
        fluidRow(
          column(3, downloadButton(ns("dl_quarterly"),  "quarterly_fcst.xlsx")),
          column(3, downloadButton(ns("dl_full"),       "full_results.xlsx")),
          column(3, downloadButton(ns("dl_bridge"),     "bridge_results.xlsx")),
          column(3, downloadButton(ns("dl_metrics"),    "metrics.xlsx"))
        )
      )
    )
  )
}

# -----------------------------------------------------------------------------
# Server
# -----------------------------------------------------------------------------
mod_results_server <- function(id, model_data) {
  moduleServer(id, function(input, output, session) {
    ns <- session$ns

    # ----------------------------------------------------------------
    # Convenience: extract pieces
    # ----------------------------------------------------------------
    q_fcst <- reactive({
      req(model_data())
      model_data()$quarterly_fcst
    })

    target_var <- reactive({
      req(model_data())
      model_data()$target_var
    })

    metrics <- reactive({
      req(q_fcst(), target_var())
      compute_all_metrics(q_fcst(), target_var())
    })

    # ----------------------------------------------------------------
    # Factor selector
    # ----------------------------------------------------------------
    observeEvent(model_data(), {
      r <- model_data()$r
      updateSelectInput(session, "factor_sel",
                        choices  = paste0("F", seq_len(r)),
                        selected = "F1")
    })

    # ----------------------------------------------------------------
    # Forecast plot
    # ----------------------------------------------------------------
    output$forecast_plot <- renderPlotly({
      qf  <- q_fcst()
      tv  <- target_var()
      req(qf, tv)

      p <- ggplot(qf, aes(x = Date)) +
        geom_line(aes(y = .data[[tv]], colour = "Actual"), linewidth = 1) +
        labs(title = paste("GDP Nowcast — DFM Rolling Window"),
             x = NULL, y = tv, colour = NULL) +
        theme_minimal()

      colour_map <- c(
        h0_fcst = "#2ca02c", h1_fcst = "#1f77b4",
        h2_fcst = "#d62728", h3_fcst = "#ff7f0e"
      )
      label_map <- c(
        h0_fcst = "h=0 (EoQ)", h1_fcst = "h=1",
        h2_fcst = "h=2",       h3_fcst = "h=3"
      )

      for (h in intersect(input$show_horizons, names(colour_map))) {
        if (h %in% names(qf)) {
          p <- p + geom_line(aes(y = .data[[h]], colour = !!label_map[[h]]),
                             linetype = "dashed", linewidth = 0.8,
                             data = qf)
        }
      }

      # Bridge results overlay
      if ("bridge" %in% input$show_horizons && !is.null(model_data()$bridge_results)) {
        br <- model_data()$bridge_results
        p  <- p + geom_line(data = br, aes(x = Date, y = GDP_FCST, colour = "Bridge (h=0)"),
                            linewidth = 0.8)
      }

      ggplotly(p) |> layout(legend = list(orientation = "h", y = -0.15))
    })

    # ----------------------------------------------------------------
    # Metrics table
    # ----------------------------------------------------------------
    output$metrics_table <- renderDT({
      datatable(metrics(), rownames = FALSE,
                options = list(dom = "t", pageLength = 10)) |>
        formatRound(c("RMSE", "MAE", "Directional_Acc"), digits = 4)
    })

    output$bridge_metrics_ui <- renderUI({
      req(model_data()$bridge_results)
      h4("Bridge Model (XGBoost h=0) — Accuracy")
    })

    output$bridge_metrics_table <- renderDT({
      br <- model_data()$bridge_results
      req(br)
      tv <- target_var()
      m  <- compute_metrics(br[[tv]], br$GDP_FCST)
      df <- data.frame(Model = "Bridge (XGBoost h=0)",
                       RMSE  = m$RMSE, MAE = m$MAE, N_obs = m$n)
      datatable(df, rownames = FALSE, options = list(dom = "t")) |>
        formatRound(c("RMSE", "MAE"), digits = 4)
    })

    # ----------------------------------------------------------------
    # Factor loadings plot
    # ----------------------------------------------------------------
    output$loadings_plot <- renderPlotly({
      req(model_data(), input$factor_sel)
      # We use the last DFM run embedded in model_data via quarterly_fcst —
      # loadings require the dfm object which we don't store in full.
      # Show a placeholder directing user to re-run with summary.
      showNotification(
        "Factor loadings require the DFM object. Re-run the model and check the R console for summary().",
        type = "message", duration = 5
      )
      plotly::plot_ly() |>
        layout(title = "Factor loadings not available in app preview.\nSee R console for summary().")
    })

    # ----------------------------------------------------------------
    # Export handlers
    # ----------------------------------------------------------------
    output$dl_quarterly <- downloadHandler(
      filename = "quarterly_fcst.xlsx",
      content  = function(file) {
        save_info_to_excel(list(quarterly_fcst = q_fcst()), file)
      }
    )

    output$dl_full <- downloadHandler(
      filename = "full_results.xlsx",
      content  = function(file) {
        save_info_to_excel(list(full_results = model_data()$results), file)
      }
    )

    output$dl_bridge <- downloadHandler(
      filename = "bridge_results.xlsx",
      content  = function(file) {
        br <- model_data()$bridge_results
        req(br)
        save_info_to_excel(list(bridge_results = br), file)
      }
    )

    output$dl_metrics <- downloadHandler(
      filename = "metrics.xlsx",
      content  = function(file) {
        save_info_to_excel(list(metrics = metrics()), file)
      }
    )
  })
}
