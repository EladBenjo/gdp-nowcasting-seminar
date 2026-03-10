# =============================================================================
# app.R — GDP Nowcasting Shiny App
#
# HOW TO RUN:
#   In RStudio: click "Run App" button, or run:
#     shiny::runApp()
#
#   From the R console:
#     setwd("path/to/gdp-nowcasting-seminar")
#     shiny::runApp()
#
# REQUIRED PACKAGES:
#   install.packages(c(
#     "shiny", "bslib", "DT", "ggplot2", "plotly", "dplyr",
#     "readxl", "openxlsx", "zoo", "xts", "lubridate",
#     "tseries", "forecast", "seasonal", "dfms", "vars", "xgboost"
#   ))
# =============================================================================

library(shiny)

source("R/shiny/ui.R")
source("R/shiny/server.R")

shinyApp(ui = ui, server = server)
