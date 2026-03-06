# GDP Nowcasting — Israeli Ministry of Finance

Real-time quarterly GDP growth forecasting using mixed-frequency macroeconomic data.  
Developed in collaboration with the Chief Economist's office at the Israeli Ministry of Finance.

---

## Overview

GDP figures are published with a significant lag — policymakers need estimates *before* the official release. This project builds a nowcasting pipeline that ingests high-frequency indicators (monthly, weekly) and produces quarterly GDP growth estimates in real time.

The core challenge: combining data that arrives at different frequencies, with missing values, under economic constraints — and doing it in a way that's actually useful to analysts.

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
