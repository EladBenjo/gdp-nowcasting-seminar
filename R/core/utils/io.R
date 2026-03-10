# =============================================================================
# io.R
# I/O helpers: reading data files, writing multi-sheet Excel outputs.
# =============================================================================

library(readxl)
library(openxlsx)

# -----------------------------------------------------------------------------
# save_list_to_excel()
# Write a named list of data.frames to a multi-sheet Excel file.
# -----------------------------------------------------------------------------
save_list_to_excel <- function(lst, file_path = "output.xlsx") {
  wb <- openxlsx::createWorkbook()

  for (sheet_name in names(lst)) {
    openxlsx::addWorksheet(wb, sheet_name)
    openxlsx::writeData(wb, sheet = sheet_name, x = lst[[sheet_name]])
  }

  openxlsx::saveWorkbook(wb, file = file_path, overwrite = TRUE)
  message("Saved: ", file_path)
}

# -----------------------------------------------------------------------------
# save_info_to_excel()
# Write a named list of metadata data.frames to a multi-sheet Excel file.
# (Used for SA metadata, transformation info, shift reports.)
# -----------------------------------------------------------------------------
save_info_to_excel <- function(info_list, file_path = "info.xlsx") {
  wb <- openxlsx::createWorkbook()

  for (block_name in names(info_list)) {
    openxlsx::addWorksheet(wb, block_name)
    openxlsx::writeData(wb, sheet = block_name, x = info_list[[block_name]])
  }

  openxlsx::saveWorkbook(wb, file = file_path, overwrite = TRUE)
  message("Saved: ", file_path)
}

# -----------------------------------------------------------------------------
# read_raw_data()
# Load all sheets from the raw Excel file into a named list of data.frames.
# Converts the Date column to Date type automatically.
# -----------------------------------------------------------------------------
read_raw_data <- function(path) {
  sheets <- readxl::excel_sheets(path)

  blocks <- sheets |>
    stats::setNames(sheets) |>
    lapply(\(s) {
      df <- readxl::read_excel(path, sheet = s)
      if ("Date" %in% names(df)) df$Date <- as.Date(df$Date)
      df
    })

  blocks
}

# -----------------------------------------------------------------------------
# read_panel_data()
# Load the pre-processed monthly-quarterly panel used by the DFM.
# -----------------------------------------------------------------------------
read_panel_data <- function(path) {
  df <- readxl::read_excel(path)
  df$Date <- as.Date(df$Date)
  df
}

# -----------------------------------------------------------------------------
# read_uploaded_file()
# Read an analyst-uploaded file (Excel or CSV) into a data.frame.
# Returns NULL and a message if the file format is unsupported.
# -----------------------------------------------------------------------------
read_uploaded_file <- function(file_path, sheet = 1) {
  ext <- tools::file_ext(file_path)

  df <- tryCatch({
    if (ext %in% c("xlsx", "xls")) {
      readxl::read_excel(file_path, sheet = sheet)
    } else if (ext == "csv") {
      utils::read.csv(file_path, stringsAsFactors = FALSE)
    } else {
      stop("Unsupported file format: ", ext)
    }
  }, error = function(e) {
    message("Error reading file: ", e$message)
    NULL
  })

  df
}
