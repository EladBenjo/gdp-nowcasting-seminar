# =============================================================================
# ui.R
# Shiny app UI — 5-step wizard layout using bslib.
# =============================================================================

library(shiny)
library(bslib)

source("R/shiny/modules/mod_upload.R")
source("R/shiny/modules/mod_freq_target.R")
source("R/shiny/modules/mod_transforms.R")
source("R/shiny/modules/mod_modeling.R")
source("R/shiny/modules/mod_results.R")

ui <- page_navbar(
  title = "GDP Nowcasting",
  theme = bs_theme(bootswatch = "flatly", base_font = font_google("Inter")),
  window_title = "GDP Nowcasting",

  # ── Step 1 ───────────────────────────────────────────────────────────────
  nav_panel("1 · Upload Data",
    icon = icon("upload"),
    br(),
    mod_upload_ui("upload")
  ),

  # ── Step 2 ───────────────────────────────────────────────────────────────
  nav_panel("2 · Frequency & Target",
    icon = icon("calendar-check"),
    br(),
    mod_freq_target_ui("freq_target")
  ),

  # ── Step 3 ───────────────────────────────────────────────────────────────
  nav_panel("3 · Transformations",
    icon = icon("sliders"),
    br(),
    mod_transforms_ui("transforms")
  ),

  # ── Step 4 ───────────────────────────────────────────────────────────────
  nav_panel("4 · Model",
    icon = icon("brain"),
    br(),
    mod_modeling_ui("modeling")
  ),

  # ── Step 5 ───────────────────────────────────────────────────────────────
  nav_panel("5 · Results",
    icon = icon("chart-line"),
    br(),
    mod_results_ui("results")
  ),

  nav_spacer(),
  nav_item(
    tags$a(icon("github"), "Source",
           href = "https://github.com", target = "_blank")
  )
)
