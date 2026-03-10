# =============================================================================
# mod_transforms.R
# Step 3 — Transformation Recommendations & Analyst Approval
# Runs ADF / KPSS / STL tests on each column, displays recommendations,
# allows per-column overrides, then applies the approved transformations.
# =============================================================================

library(shiny)
library(DT)
library(dplyr)

# Source core stationarity functions
# (sourced in server.R — assumed available in the session)

TRANSFORM_CHOICES <- c(
  "none", "log", "diff", "logdiff",
  "seasonal_diff", "log_seasonal_diff",
  "diff_seasonal_diff", "logdiff_seasonal_diff",
  "detrend"
)

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
mod_transforms_ui <- function(id) {
  ns <- NS(id)

  tagList(
    h3("Step 3 — Transformation Recommendations"),
    p("The table below shows the recommended stationarity transformation for each
       variable, based on ADF and KPSS unit-root tests and STL seasonality detection.
       You can override any recommendation before applying."),

    fluidRow(
      column(3,
        wellPanel(
          numericInput(ns("freq"), "Series frequency (months)", value = 12, min = 1),
          numericInput(ns("alpha"), "Significance level (α)", value = 0.05,
                       min = 0.01, max = 0.20, step = 0.01),
          actionButton(ns("run_tests"), "Run Tests",
                       class = "btn-secondary", icon = icon("flask")),
          hr(),
          p(strong("Override a variable:")),
          selectInput(ns("override_var"), "Variable", choices = NULL),
          selectInput(ns("override_trans"), "Set transformation to",
                      choices = TRANSFORM_CHOICES),
          actionButton(ns("apply_override"), "Apply override",
                       class = "btn-secondary", icon = icon("edit")),
          hr(),
          actionButton(ns("apply_transforms"), "Apply Transformations",
                       class = "btn-primary", icon = icon("check")),
          uiOutput(ns("apply_status"))
        )
      ),
      column(9,
        h4("Test Results & Recommendations"),
        DTOutput(ns("rec_table")),
        br(),
        h4("Post-Transformation Stationarity Check"),
        uiOutput(ns("post_check_ui")),
        DTOutput(ns("post_table"))
      )
    )
  )
}

# -----------------------------------------------------------------------------
# Server
# -----------------------------------------------------------------------------
mod_transforms_server <- function(id, freq_target_data) {
  moduleServer(id, function(input, output, session) {
    ns <- session$ns

    rec_table_rv   <- reactiveVal(NULL)  # recommendations (modifiable)
    trans_data_rv  <- reactiveVal(NULL)  # transformed data.frame
    post_table_rv  <- reactiveVal(NULL)  # post-transform test results

    # ----------------------------------------------------------------
    # Run stationarity tests
    # ----------------------------------------------------------------
    observeEvent(input$run_tests, {
      fd <- freq_target_data()
      req(fd)

      df         <- fd$df
      target_var <- fd$target_var

      # Exclude target variable from transformation (it is the output)
      vars <- setdiff(names(df), c("Date", target_var))

      withProgress(message = "Running stationarity tests...", value = 0, {
        rows <- lapply(seq_along(vars), function(i) {
          v <- vars[i]
          incProgress(1 / length(vars), detail = v)
          res <- test_transform_full(df[[v]], freq = input$freq, alpha = input$alpha)
          data.frame(
            Variable      = v,
            Seasonal      = res$seasonal,
            Seas_Strength = res$seas_strength,
            ADF_p         = res$adf_p,
            KPSS_p        = res$kpss_p,
            Recommended   = res$recommendation,
            Override      = res$recommendation,  # editable copy
            stringsAsFactors = FALSE
          )
        })
      })

      rt <- dplyr::bind_rows(rows)
      rec_table_rv(rt)
      updateSelectInput(session, "override_var", choices = rt$Variable)
    })

    # ----------------------------------------------------------------
    # Apply per-variable override
    # ----------------------------------------------------------------
    observeEvent(input$apply_override, {
      req(input$override_var, rec_table_rv())
      rt <- rec_table_rv()
      rt$Override[rt$Variable == input$override_var] <- input$override_trans
      rec_table_rv(rt)
    })

    # ----------------------------------------------------------------
    # Render recommendation table (with colour coding)
    # ----------------------------------------------------------------
    output$rec_table <- renderDT({
      req(rec_table_rv())
      datatable(
        rec_table_rv(),
        options  = list(pageLength = 20, dom = "ftp", scrollX = TRUE),
        rownames = FALSE
      ) |>
        formatStyle("Seasonal",
          backgroundColor = styleEqual(c(TRUE, FALSE), c("#fff3cd", "white"))) |>
        formatStyle("ADF_p",
          backgroundColor = styleInterval(c(0.05), c("#d4edda", "#f8d7da"))) |>
        formatStyle("KPSS_p",
          backgroundColor = styleInterval(c(0.05), c("#f8d7da", "#d4edda"))) |>
        formatStyle("Override",
          fontWeight = "bold")
    })

    # ----------------------------------------------------------------
    # Apply transformations
    # ----------------------------------------------------------------
    observeEvent(input$apply_transforms, {
      rt <- rec_table_rv()
      fd <- freq_target_data()
      req(rt, fd)

      df         <- fd$df
      target_var <- fd$target_var
      vars       <- rt$Variable

      codes <- sapply(rt$Override, label_to_code)

      withProgress(message = "Applying transformations...", value = 0.5, {
        block <- df[, c("Date", vars)]
        out   <- transform_block(block, codes, freq = input$freq)
      })

      # Attach the target variable (untransformed) back
      trans_df <- out$data
      trans_df[[target_var]] <- df[[target_var]]

      trans_data_rv(list(
        df         = trans_df,
        target_var = target_var,
        info       = out$info
      ))

      # Post-transformation stationarity re-check
      withProgress(message = "Re-running stationarity tests...", value = 0.8, {
        post_rows <- lapply(vars, function(v) {
          res <- test_transform_full(trans_df[[v]], freq = input$freq, alpha = input$alpha)
          data.frame(
            Variable   = v,
            ADF_p      = res$adf_p,
            KPSS_p     = res$kpss_p,
            Stationary = res$recommendation == "none",
            stringsAsFactors = FALSE
          )
        })
      })

      post_table_rv(dplyr::bind_rows(post_rows))
    })

    # Status message after apply
    output$apply_status <- renderUI({
      req(trans_data_rv())
      div(class = "alert alert-success mt-2",
          icon("check-circle"), " Transformations applied.")
    })

    # Post-transform table
    output$post_check_ui <- renderUI({
      req(post_table_rv())
      p("Re-ran ADF/KPSS on transformed series. Green = stationary, red = still non-stationary.")
    })

    output$post_table <- renderDT({
      req(post_table_rv())
      datatable(
        post_table_rv(),
        options  = list(pageLength = 20, dom = "ftp"),
        rownames = FALSE
      ) |>
        formatStyle("Stationary",
          backgroundColor = styleEqual(c(TRUE, FALSE), c("#d4edda", "#f8d7da")))
    })

    return(reactive({
      trans_data_rv()
    }))
  })
}
