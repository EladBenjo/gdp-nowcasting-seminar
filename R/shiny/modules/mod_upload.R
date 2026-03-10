# =============================================================================
# mod_upload.R
# Step 1 — Data Upload & Validation
# Analyst uploads an Excel or CSV file. The module validates structure,
# detects the date column, reports missing values, and passes a clean
# data.frame downstream.
# =============================================================================

library(shiny)
library(DT)

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
mod_upload_ui <- function(id) {
  ns <- NS(id)

  tagList(
    h3("Step 1 — Upload Your Data"),
    p("Upload a single Excel (.xlsx) or CSV file. The file should contain one
       date column and one or more numeric indicator columns. The target variable
       (e.g. GDP) should be included as one of the columns."),

    fluidRow(
      column(4,
        fileInput(ns("file"), "Choose file",
                  accept = c(".xlsx", ".xls", ".csv"),
                  buttonLabel = "Browse...",
                  placeholder = "No file selected"),

        # Excel sheet selector (hidden until file loaded)
        conditionalPanel(
          condition = paste0("output['", ns("is_excel"), "']"),
          selectInput(ns("sheet"), "Sheet", choices = NULL)
        ),

        # Date column selector
        selectInput(ns("date_col"), "Date column", choices = NULL),

        actionButton(ns("validate"), "Validate & Continue",
                     class = "btn-primary", icon = icon("check"))
      ),
      column(8,
        uiOutput(ns("validation_banner")),
        DTOutput(ns("column_summary"))
      )
    )
  )
}

# -----------------------------------------------------------------------------
# Server
# -----------------------------------------------------------------------------
mod_upload_server <- function(id) {
  moduleServer(id, function(input, output, session) {
    ns <- session$ns

    # ------------------------------------------------------------------
    # Reactive: raw file read
    # ------------------------------------------------------------------
    raw_data <- reactive({
      req(input$file)
      ext <- tools::file_ext(input$file$name)

      if (ext %in% c("xlsx", "xls")) {
        sheets <- readxl::excel_sheets(input$file$datapath)
        updateSelectInput(session, "sheet", choices = sheets, selected = sheets[1])
        sheet_choice <- isolate(input$sheet) %||% sheets[1]
        readxl::read_excel(input$file$datapath, sheet = sheet_choice)
      } else if (ext == "csv") {
        utils::read.csv(input$file$datapath, stringsAsFactors = FALSE)
      } else {
        NULL
      }
    })

    # Update sheet choices when Excel file changes
    observeEvent(input$file, {
      ext <- tools::file_ext(input$file$name)
      if (ext %in% c("xlsx", "xls")) {
        sheets <- readxl::excel_sheets(input$file$datapath)
        updateSelectInput(session, "sheet", choices = sheets)
      }
    })

    # Re-read when sheet changes
    data_for_sheet <- reactive({
      req(input$file)
      ext <- tools::file_ext(input$file$name)
      if (ext %in% c("xlsx", "xls") && !is.null(input$sheet)) {
        df <- readxl::read_excel(input$file$datapath, sheet = input$sheet)
      } else {
        df <- raw_data()
      }
      df
    })

    # Tell UI whether it's Excel (for conditional panel)
    output$is_excel <- reactive({
      req(input$file)
      tools::file_ext(input$file$name) %in% c("xlsx", "xls")
    })
    outputOptions(output, "is_excel", suspendWhenHidden = FALSE)

    # ------------------------------------------------------------------
    # Populate date column selector
    # ------------------------------------------------------------------
    observeEvent(data_for_sheet(), {
      df <- data_for_sheet()
      req(df)
      # Guess date column: prefer cols named "date" / "Date" / parseable as date
      date_candidates <- names(df)[sapply(names(df), function(col) {
        grepl("^date$", col, ignore.case = TRUE) ||
          tryCatch({ as.Date(df[[col]][1]); TRUE }, error = \(e) FALSE)
      })]
      choices <- names(df)
      selected <- if (length(date_candidates)) date_candidates[1] else choices[1]
      updateSelectInput(session, "date_col", choices = choices, selected = selected)
    })

    # ------------------------------------------------------------------
    # Validation logic
    # ------------------------------------------------------------------
    validation_result <- eventReactive(input$validate, {
      df  <- data_for_sheet()
      req(df, input$date_col)

      warnings <- character(0)
      errors   <- character(0)

      # Parse date column
      date_parsed <- tryCatch(as.Date(df[[input$date_col]]), error = \(e) NULL)
      if (is.null(date_parsed)) {
        errors <- c(errors, paste0("Cannot parse '", input$date_col, "' as dates."))
      } else {
        df[[input$date_col]] <- date_parsed
      }

      # Rename date col to "Date" for downstream use
      names(df)[names(df) == input$date_col] <- "Date"

      # Duplicate column names
      if (any(duplicated(names(df)))) {
        errors <- c(errors, "Duplicate column names detected.")
      }

      # All-NA columns
      all_na_cols <- names(df)[sapply(names(df), \(c) all(is.na(df[[c]])))]
      all_na_cols <- setdiff(all_na_cols, "Date")
      if (length(all_na_cols)) {
        warnings <- c(warnings,
                      paste("All-NA columns (will be dropped):", paste(all_na_cols, collapse = ", ")))
        df <- df[, !names(df) %in% all_na_cols]
      }

      # Column summary table
      summary_df <- do.call(rbind, lapply(names(df), function(col) {
        x         <- df[[col]]
        pct_na    <- round(mean(is.na(x)) * 100, 1)
        col_type  <- class(x)[1]
        date_rng  <- if (col == "Date") paste(min(x, na.rm=TRUE), "to", max(x, na.rm=TRUE)) else ""
        data.frame(Column = col, Type = col_type, Missing_pct = pct_na,
                   Date_range = date_rng, stringsAsFactors = FALSE)
      }))

      list(df = df, warnings = warnings, errors = errors, summary = summary_df)
    })

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------
    output$validation_banner <- renderUI({
      res <- validation_result()
      req(res)

      msgs <- tagList()
      if (length(res$errors)) {
        msgs <- tagAppendChild(msgs,
          div(class = "alert alert-danger",
              icon("times-circle"), strong(" Errors: "),
              tags$ul(lapply(res$errors, tags$li))))
      }
      if (length(res$warnings)) {
        msgs <- tagAppendChild(msgs,
          div(class = "alert alert-warning",
              icon("exclamation-triangle"), strong(" Warnings: "),
              tags$ul(lapply(res$warnings, tags$li))))
      }
      if (!length(res$errors) && !length(res$warnings)) {
        msgs <- tagAppendChild(msgs,
          div(class = "alert alert-success",
              icon("check-circle"), " Data validated successfully. Proceed to Step 2."))
      }
      msgs
    })

    output$column_summary <- renderDT({
      res <- validation_result()
      req(res)
      datatable(res$summary, options = list(pageLength = 15, dom = "tp"),
                rownames = FALSE)
    })

    # ------------------------------------------------------------------
    # Return the validated data.frame (NULL until validated)
    # ------------------------------------------------------------------
    return(reactive({
      res <- validation_result()
      req(res, length(res$errors) == 0)
      res$df
    }))
  })
}

# Null-coalescing operator
`%||%` <- function(a, b) if (!is.null(a) && length(a) > 0 && a != "") a else b
