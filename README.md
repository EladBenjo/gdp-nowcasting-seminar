# GDP Nowcasting — Israeli Ministry of Finance

Real-time quarterly GDP growth forecasting using mixed-frequency macroeconomic data.  
Developed in collaboration with the Chief Economist's office at the Israeli Ministry of Finance.

---

## Overview

GDP figures are published with a significant lag — policymakers need estimates *before* the official release. This project builds a nowcasting pipeline that ingests high-frequency indicators (monthly, weekly) and produces quarterly GDP growth estimates in real time.

The core challenge: combining data that arrives at different frequencies, with missing values, under economic constraints — and doing it in a way that's actually useful to analysts.

---

## Models

| Model | Description |
|---|---|
| **DFM** (Dynamic Factor Model) | Core nowcasting model; extracts latent factors from mixed-frequency indicators |
| **ARIMA / SARIMA** | Univariate baseline and component forecasting |
| **XGBoost / Random Forest** | ML benchmark for feature-based nowcasting |

---

## Results

> _Update with your actual numbers below_

| Model | RMSE | MAE | vs. Baseline |
|---|---|---|---|
| DFM | — | — | — |
| XGBoost | — | — | — |
| ARIMA (baseline) | — | — | — |

---

## Repository Structure

```
gdp-nowcasting-seminar/
├── Notebooks/          # Exploratory analysis and model development
├── R/                  # R-based econometric models (DFM, ARIMA)
├── src/                # Python utilities and pipeline components
├── nowcasting_GDP_Q_by_Elad_Benjo.pdf  # Full project report
└── README.md
```

---

## Data

Mixed-frequency Israeli macroeconomic indicators including:
- Monthly industrial production, trade, and employment data
- Quarterly national accounts (CBS Israel)
- Additional high-frequency proxies

> Data sourced from public Israeli government databases. Not included in repo due to licensing.

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![R](https://img.shields.io/badge/R-276DC3?style=flat&logo=r&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white)

**Python:** pandas, scikit-learn, statsmodels, XGBoost  
**R:** dynfactoR / nowcasting package (DFM)

---

## Report

The full methodology, data sources, and results are documented in [`nowcasting_GDP_Q_by_Elad_Benjo.pdf`](./nowcasting_GDP_Q_by_Elad_Benjo.pdf).

---

## Author

**Elad Benjo** — [LinkedIn](https://www.linkedin.com/in/eladbenjo/) · [GitHub](https://github.com/EladBenjo)
