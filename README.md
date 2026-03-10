# GDP Nowcasting — Israeli Ministry of Finance

Real-time quarterly GDP growth forecasting using mixed-frequency macroeconomic data.
Developed in collaboration with the Chief Economist's office at the Israeli Ministry of Finance.

---

## Overview

GDP figures are published with a significant lag — policymakers need estimates *before* the official release. This project builds a nowcasting pipeline that ingests high-frequency indicators (monthly, weekly) and produces quarterly GDP growth estimates in real time.

The core challenge: combining data that arrives at different frequencies, with missing values, under economic constraints — and doing it in a way that's actually useful to analysts.

The pipeline is packaged as an **analyst-friendly Shiny web app** — no R knowledge required to use it.

---

## Architecture

The pipeline combines two modeling stages:

1. **DFM (Dynamic Factor Model)** — extracts latent factors from mixed-frequency indicators; used directly for multi-horizon forecasting (h=1,2,3)
2. **DFM + XGBoost (Bridge Model)** — at h=0, DFM factors are used as regressors in XGBoost, compensating for DFM's weaker nowcasting performance at the current quarter

---

## Results

All metrics are reported as **ratio to Random Walk baseline** (lower = better). A value of 0.80 means the model achieves 80% of the baseline's error.

**DFM — Multi-Horizon Forecasting**

| Horizon | RMSE / Baseline | MAE / Baseline | Directional Accuracy |
|---|---|---|---|
| h=1 | 0.336 | 0.407 | 0.67 |
| h=2 | 0.265 | 0.280 | **0.93** |
| h=3 | 0.818 | 0.783 | 0.67 |

**Model Comparison at h=0 (Nowcasting)**

| Model | RMSE / Baseline | MAE / Baseline |
|---|---|---|
| DFM | 0.965 | 1.002 |
| Bridge (DFM → XGBoost) | **0.802** | **0.849** |

Key takeaways:
- DFM outperforms Random Walk significantly at h=1 and h=2
- At h=0, the Bridge model (DFM factors as XGBoost regressors) is the strongest performer
- Directional accuracy of 0.93 at h=2 is particularly strong for a policy-relevant signal

---

## Shiny App — Analyst Workflow

The app guides analysts through 5 steps:

| Step | Description |
| ---- | ----------- |
| **1 · Upload** | Upload any Excel or CSV file with indicators + target variable |
| **2 · Frequency & Target** | Auto-detects column frequency (monthly/quarterly); analyst selects the target variable |
| **3 · Transformations** | Runs ADF, KPSS, and STL tests; recommends transformations per variable; analyst can override before applying |
| **4 · Model** | ICr and VARselect diagnostics guide factor/lag selection; runs rolling-window DFM with optional XGBoost bridge |
| **5 · Results** | Interactive forecast plot, accuracy metrics by horizon, and Excel export |

### Running locally

```r
# Install required packages (first time only)
install.packages(c(
  "shiny", "bslib", "DT", "ggplot2", "plotly", "dplyr",
  "readxl", "openxlsx", "zoo", "xts", "lubridate",
  "tseries", "forecast", "seasonal", "dfms", "vars", "xgboost"
))

# Launch the app
shiny::runApp()
```

---

## Repository Structure

```
gdp-nowcasting-seminar/
├── app.R                       # Shiny entry point — run this
├── R/
│   ├── core/                   # Pure R functions (no Shiny dependency)
│   │   ├── config.R            # All paths and model parameters
│   │   ├── utils/io.R          # Excel read/write helpers
│   │   ├── transformations/    # Price adjustment, X-13 SA, stationarity, release lags
│   │   └── modeling/           # DFM, XGBoost bridge, evaluation metrics
│   ├── shiny/
│   │   ├── ui.R                # App layout
│   │   ├── server.R            # Reactive data flow
│   │   └── modules/            # One module per workflow step
│   ├── DFM.qmd                 # Original DFM notebook (reference)
│   └── transformations.qmd     # Original transformations notebook (reference)
├── nowcasting_GDP_Q_by_Elad_Benjo.pdf  # Full project report
└── README.md
```

---

## Data

Mixed-frequency Israeli macroeconomic indicators including:

- Monthly tax revenues, industrial production, trade, and employment data
- Quarterly national accounts (CBS Israel)
- FX rates, capital markets, and real estate indicators

> Data sourced from public Israeli government databases. Not included in repo due to licensing.

---

## Tech Stack

![R](https://img.shields.io/badge/R-276DC3?style=flat&logo=r&logoColor=white)
![Shiny](https://img.shields.io/badge/Shiny-blue?style=flat&logo=rstudio&logoColor=white)

**R packages:** `dfms`, `xgboost`, `seasonal` (X-13), `vars`, `tseries`, `shiny`, `bslib`, `plotly`

---

## Report

The full methodology, data sources, and results are documented in [`nowcasting_GDP_Q_by_Elad_Benjo.pdf`](./nowcasting_GDP_Q_by_Elad_Benjo.pdf).

---

## Author

**Elad Benjo** — [LinkedIn](https://www.linkedin.com/in/eladbenjo/) · [GitHub](https://github.com/EladBenjo)
