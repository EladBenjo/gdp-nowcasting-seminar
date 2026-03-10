# =============================================================================
# mod_freq_target.R
# Step 2 — Frequency Detection & Target Variable Selection
# Auto-detects per-column frequency (monthly / quarterly / other),
# lets analyst confirm and select the quarterly target variable.
# =============================================================================

library(shiny)
library(DT)
library(dplyr)

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
mod_freq_target_ui <- function(id) {
  ns <- NS(id)

  tagList(
    h3("Step 2 — Frequency Detection & Target Variable"),

    fluidRow(
      column(4,
        wellPanel(
          h4("Target Variable"),
          p("Select the quarterly variable you want to nowcast (e.g. GDP)."),
          selectInput(ns("target_var"), "Target variable", choices = NULL),
          hr(),
          h4("Frequency Override"),
          p("The table on the right shows auto-detected frequencies.
             If any column is mis-detected, use the override below."),
          selectInput(ns("override_col"), "Column to override", choices = NULL),
          selectInput(ns("override_freq"), "Set frequency to",
                      choices = c("monthly", "quarterly", "annual", "other")),
          actionButton(ns("apply_override"), "Apply override", class = "btn-secondary"),
          hr(),
          actionButton(ns("confirm"), "Confirm & Continue",
                       class = "btn-primary", icon = icon("check"))
        )
      ),
      column(8,
        DTOutput(ns("freq_table"))
      )
    )
  )
}

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

# Detect frequency of a single column by analysing median gap between non-NA dates
detect_freq <- function(dates, values) {
  non_na_dates <- dates[!is.na(values)]
  if (length(non_na_dates) < 3) return("unknown")

  gaps <- as.numeric(diff(sort(non_na_dates)))
  med_gap <- stats::median(gaps)

  if      (med_gap <= 35)  "monthly"
  else if (med_gap <= 100) "quarterly"
  else if (med_gap <= 370) "annual"
  else                     "other"
}

# -----------------------------------------------------------------------------
# Server
# -----------------------------------------------------------------------------
mod_freq_target_server <- function(id, uploaded_data) {
  moduleServer(id, function(input, output, session) {
    ns <- session$ns

    # Frequency table (mutable via overrides)
    freq_table_rv <- reactiveVal(NULL)

    # Initialise when data arrives
    observeEvent(uploaded_data(), {
      df <- uploaded_data()
      req(df)

      vars <- setdiff(names(df), "Date")

      freq_df <- data.frame(
        Column    = vars,
        Frequency = sapply(vars, \(v) detect_freq(df$Date, df[[v]])),
        N_obs     = sapply(vars, \(v) sum(!is.na(df[[v]]))),
        stringsAsFactors = FALSE
      )

      freq_table_rv(freq_df)

      # Pre-select target: first col containing "gdp" / "GDP" / "growth", else last col
      gdp_match <- vars[grepl("gdp|growth|target", vars, ignore.case = TRUE)]
      selected  <- if (length(gdp_match)) gdp_match[1] else utils::tail(vars, 1)

      updateSelectInput(session, "target_var",  choices = vars, selected = selected)
      updateSelectInput(session, "override_col", choices = vars)
    })

    # Apply manual override
    observeEvent(input$apply_override, {
      req(input$override_col, input$override_freq, freq_table_rv())
      ft <- freq_table_rv()
      ft$Frequency[ft$Column == input$override_col] <- input$override_freq
      freq_table_rv(ft)
    })

    # Render frequency table
    output$freq_table <- renderDT({
      req(freq_table_rv())
      ft <- freq_table_rv()

      # Colour-code frequencies
      datatable(ft,
        options  = list(pageLength = 20, dom = "tp"),
        rownames = FALSE
      ) |>
        formatStyle("Frequency",
          backgroundColor = styleEqual(
            c("monthly", "quarterly", "annual", "other", "unknown"),
            c("#d4edda",  "#cce5ff",   "#fff3cd", "#f8d7da", "#e2e3e5")
          )
        )
    })

    # ------------------------------------------------------------------
    # Confirmed output: list(df = ..., target_var = ..., freq_table = ...)
    # ------------------------------------------------------------------
    confirmed <- eventReactive(input$confirm, {
      df        <- uploaded_data()
      req(df, input$target_var, freq_table_rv())

      list(
        df         = df,
        target_var = input$target_var,
        freq_table = freq_table_rv()
      )
    })

    return(confirmed)
  })
}
