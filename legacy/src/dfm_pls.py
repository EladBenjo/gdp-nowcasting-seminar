"""
PLS-DFM (BoI-style) Nowcasting Pipeline — Python Skeleton
Author: ChatGPT (for Elad)

Overview
========
This repo-style skeleton implements a practical PLS-guided factor + bridge-nowcast workflow,
with hooks to extend into a full state-space (Kalman) variant.

Design goals:
- Input: monthly_df (monthly panel, DatetimeIndex at month start), quarterly_df (GDP QoQ growth),
         proxy series name (e.g., 'salaried_jobs_total_sa').
- Transformations: assumed already seasonally adjusted & stationary. We only StandardScale on train.
- Factor extraction: PLSRegression guided by monthly proxy to obtain 1–2 supervised monthly factors.
- Collapse to quarterly: simple average (default) or Mariano–Murasawa-like option (for log-level logic).
- Bridge regression: map quarterly factors (+ AR terms) to GDP QoQ growth; supports ragged edge
  via partial aggregation of available months.
- Rolling real-time backtest: expanding or rolling re-estimation each quarter.

TODO (next iteration):
- Full state-space (monthly latent GDP with Kalman and quarterly measurement).
- Publication-lag vintages by series.
- Automatic hyperparameter selection for #components, AR order, lag structure.

Conventions
===========
- All docstrings and code comments are in English (per user preference).
- GDP target is already QoQ change (stationary). RMSE can be scaled by 4 to annualized units.
- This is a skeleton: focus is correctness, clarity, and extendability.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

from sklearn.preprocessing import StandardScaler
from sklearn.impute import IterativeImputer
from sklearn.cross_decomposition import PLSRegression
import statsmodels.api as sm

# =========================
# Config and Data Contracts
# =========================

@dataclass
class PLSDFMConfig:
    proxy_name: str = "salaried_jobs_total_sa"
    n_components: int = 1                 # PLS components (start with 1; tune later)
    monthly_lags: int = 0                 # optional lags of monthly proxy in PLS target
    scaler_with_mean: bool = True
    scaler_with_std: bool = True
    imputation_max_iter: int = 10
    collapse_method: str = "avg"          # "avg" or "mm" (Mariano–Murasawa-like)
    bridge_ar_lags: int = 1               # AR lags for quarterly bridge regression
    expanding: bool = True                # expanding window (True) or fixed-length rolling (False)
    min_train_quarters: int = 50          # initial training quarters (e.g., 15 years)

# =========================
# Utility helpers
# =========================

def assert_time_index(df: pd.DataFrame, freq: str):
    """Ensure df has DatetimeIndex with a specific frequency label in values (month-start or quarter-end).
    We do not enforce .freq attribute because many real-world frames have irregular freq metadata.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex.")
    if freq == "M":
        # Expect day==1 for month start representation
        if not np.all(df.index.day == 1):
            raise ValueError("Monthly index must be at month start (day==1).")
    elif freq == "Q":
        # Accept quarter end stamps (last day of quarter). User said quarterly is like 'YYYY-12-31'.
        pass

# =========================
# Preprocessing (scaling & imputation)
# =========================

@dataclass
class PreprocessArtifacts:
    scaler_X: StandardScaler
    scaler_y: Optional[StandardScaler]
    imputer: IterativeImputer
    cols_X: List[str]


def fit_preprocess_X(
    X_train: pd.DataFrame,
    with_mean: bool = True,
    with_std: bool = True,
    imputation_max_iter: int = 10,
) -> PreprocessArtifacts:
    """Fit scaler & imputer on TRAIN-only, return transform artifacts.

    Note: y (proxy) will be standardized separately in PLS fit if desired. Here we focus on X.
    """
    cols_X = list(X_train.columns)

    scaler_X = StandardScaler(with_mean=with_mean, with_std=with_std)
    Xs = scaler_X.fit_transform(X_train.values)

    imputer = IterativeImputer(max_iter=imputation_max_iter, random_state=0, sample_posterior=False)
    X_imp = imputer.fit_transform(Xs)

    # Store only artifacts; transformed matrices returned by transform_preprocess_X when needed
    return PreprocessArtifacts(scaler_X=scaler_X, scaler_y=None, imputer=imputer, cols_X=cols_X)


def transform_preprocess_X(X: pd.DataFrame, art: PreprocessArtifacts) -> pd.DataFrame:
    """Apply scaler and imputer to any split using TRAIN-fitted artifacts."""
    X = X[art.cols_X].copy()
    Xs = art.scaler_X.transform(X.values)
    X_imp = art.imputer.transform(Xs)
    return pd.DataFrame(X_imp, index=X.index, columns=art.cols_X)

# =========================
# PLS factor extraction (monthly)
# =========================

@dataclass
class PLSArtifacts:
    pls: PLSRegression
    X_cols: List[str]
    y_name: str


def build_monthly_pls_factor(
    monthly_df: pd.DataFrame,
    proxy_name: str,
    X_cols: Optional[List[str]],
    n_components: int = 1,
) -> Tuple[pd.Series, PLSArtifacts]:
    """Fit a PLSRegression on monthly data to extract supervised factors guided by the proxy.

    Parameters
    ----------
    monthly_df : monthly panel after scaling/imputation
    proxy_name : name of the monthly proxy series (must exist in monthly_df)
    X_cols     : which columns to use as predictors (exclude proxy itself if you prefer)
    n_components : number of PLS components

    Returns
    -------
    factor : pd.Series of the first PLS component scores (monthly index)
    arts   : PLSArtifacts for later transform on new windows
    """
    if proxy_name not in monthly_df.columns:
        raise KeyError(f"Proxy '{proxy_name}' not found in monthly_df columns.")

    if X_cols is None:
        X_cols = [c for c in monthly_df.columns if c != proxy_name]

    y = monthly_df[[proxy_name]].values
    X = monthly_df[X_cols].values

    pls = PLSRegression(n_components=n_components)
    pls.fit(X, y)
    # Scores (X-scores) — first component as supervised factor
    T_scores = pls.x_scores_[:, 0]

    factor = pd.Series(T_scores, index=monthly_df.index, name=f"pls_factor_{n_components}c")
    arts = PLSArtifacts(pls=pls, X_cols=X_cols, y_name=proxy_name)
    return factor, arts


def transform_monthly_pls_factor(
    monthly_df: pd.DataFrame,
    arts: PLSArtifacts,
) -> pd.Series:
    """Project new monthly data into the trained PLS X-scores space and return the first factor."""
    X = monthly_df[arts.X_cols].values
    # Recompute scores using trained PLS (consistent with sklearn's transform)
    T = arts.pls.transform(X)
    factor = pd.Series(T[:, 0], index=monthly_df.index, name=f"pls_factor_{arts.pls.n_components}c")
    return factor

# =========================
# Monthly → Quarterly collapse (bridge inputs)
# =========================

MM_WEIGHTS = np.array([1, 2, 3, 2, 1]) / 9.0  # A common 5-month symmetric filter used in M-M contexts


def collapse_monthly_to_quarterly(
    monthly_series: pd.Series,
    method: str = "avg",
    q_anchor: str = "QE-DEC",  # עוגן רבעוני: סיום שנה בדצמבר (קלנדרי רגיל)
) -> pd.Series:
    """
    Collapse monthly factor to quarterly frequency.

    method:
      - 'avg' : ממוצע 3 חודשי הרבעון
      - 'mm'  : פילטר סימטרי (בקירוב Mariano–Murasawa), ואז לקחת את ערך סוף הרבעון
    q_anchor:
      - 'Q-DEC' לרבעונים קלנדריים (דצמבר סוף שנה)
      - אם צריך עוגן אחר: 'Q-MAR', 'Q-JUN', 'Q-SEP' וכו'.
    """
    s = monthly_series.copy()
    if not isinstance(s.index, pd.DatetimeIndex):
        raise TypeError("monthly_series must have a DatetimeIndex")
    s = s.sort_index().asfreq("MS")  # לוודא רצף חודשי בתחילת חודש

    if method == "avg":
        # ממוצע 3 חודשי הרבעון
        q = s.resample(q_anchor).mean()
        return q

    elif method == "mm":
        MM_WEIGHTS = np.array([1, 2, 3, 2, 1]) / 9.0
        filt = np.convolve(s.values, MM_WEIGHTS, mode="same")
        sf = pd.Series(filt, index=s.index)
        # לבחור את ערך החודש האחרון בכל רבעון לאחר סינון
        q = sf.resample(q_anchor).last()
        return q

    else:
        raise ValueError("method must be 'avg' or 'mm'.")

# =========================
# Bridge regression (quarterly)
# =========================

@dataclass
class BridgeModel:
    params: pd.Series
    ar_lags: int
    exog_names: List[str]


def make_bridge_design(
    q_factor: pd.Series,
    y_qoq: pd.Series,
    ar_lags: int = 1,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Construct a quarterly OLS design: y_t = const + beta * q_factor_t + AR terms of y.

    Both series must be aligned to quarterly end dates.
    """
    df = pd.concat({"y": y_qoq, "x": q_factor}, axis=1).dropna()
    # Add AR lags of y
    for L in range(1, ar_lags + 1):
        df[f"y_lag{L}"] = df["y"].shift(L)
    df = df.dropna()

    y = df["y"]
    X = sm.add_constant(pd.concat([df[["x"]], df[[c for c in df.columns if c.startswith("y_lag")]]], axis=1))
    return X, y


def fit_bridge(
    q_factor: pd.Series,
    y_qoq: pd.Series,
    ar_lags: int = 1,
) -> BridgeModel:
    X, y = make_bridge_design(q_factor, y_qoq, ar_lags=ar_lags)
    #res = sm.OLS(y, X).fit()
    # GLS with AR(p) errors (iterate to refine rho)
    res = sm.tsa.SARIMAX(y, exog=sm.add_constant(X), order=(1,0,2)).fit(disp=False) 


    return BridgeModel(params=res.params, ar_lags=ar_lags, exog_names=list(X.columns))


def predict_bridge(
    model: BridgeModel,
    q_factor: pd.Series,
    y_qoq_hist: pd.Series,
) -> pd.Series:
    """One-shot static prediction on a provided sample using fitted params.
    Uses actual lagged y where available; for genuine nowcast, ensure you only use info up to t-1.
    """
    X, _ = make_bridge_design(q_factor, y_qoq_hist, ar_lags=model.ar_lags)
    # Recreate design with available lags (uses actual y lags from y_qoq_hist)
    y_hat = pd.Series(X.values @ model.params.values, index=X.index, name="y_hat")
    return y_hat

# =========================
# Rolling real-time backtest
# =========================

@dataclass
class BacktestResult:
    y_true: pd.Series
    y_pred: pd.Series
    rmse: float
    rmse_annualized: float


def rolling_nowcast(
    monthly_df: pd.DataFrame,
    quarterly_df: pd.Series,
    cfg: PLSDFMConfig,
) -> BacktestResult:
    """Expanding-window (or rolling) real-time backtest.

    Steps per cutoff quarter t:
      1) Define train window up to t-1 (or initial min window).
      2) Fit preprocess (scaler, imputer) on monthly TRAIN slice only.
      3) Transform monthly TRAIN → PLS factor.
      4) Collapse factor to quarterly and fit Bridge OLS to quarterly TRAIN.
      5) Build quarterly factor for quarter t from available monthly data (ragged-safe via partial average).
      6) Predict y_t.

    Notes:
      - Ragged edge: for quarter t, aggregate only available months in that quarter from monthly_df.
        This function assumes monthly_df is complete historically; in a real-time setting you'd mask
        future months. Here we mimic that logic by only using months <= quarter end of t.
    """
    assert_time_index(monthly_df, freq="M")
    assert_time_index(quarterly_df.to_frame(name="y"), freq="Q")

    y_q = quarterly_df.sort_index()

    # Determine start/end
    q_list = y_q.index
    start_idx = cfg.min_train_quarters
    y_true_all = []
    y_pred_all = []

    for i in range(start_idx, len(q_list)):
        t = q_list[i]            # quarter to nowcast
        train_q_end = q_list[i-1]

        # TRAIN slices
        y_train = y_q.iloc[:i]   # up to t-1

        # Monthly TRAIN aligned to train_q_end (use all months up to end of train_q_end)
        last_train_month = pd.Period(train_q_end, freq='Q').end_time
        m_train = monthly_df.loc[monthly_df.index <= last_train_month]

        # 1) Preprocess on TRAIN
        # Separate X from proxy to avoid leakage (common practice). Include proxy in X only if desired.
        X_cols = [c for c in m_train.columns if c != cfg.proxy_name]
        art = fit_preprocess_X(m_train[X_cols], with_mean=cfg.scaler_with_mean,
                               with_std=cfg.scaler_with_std, imputation_max_iter=cfg.imputation_max_iter)

        m_train_X = transform_preprocess_X(m_train[X_cols], art)
        # Add proxy back (unscaled) as y for PLS; we can StandardScale proxy internally if needed.
        m_train_pls = m_train_X.copy()
        m_train_pls[cfg.proxy_name] = m_train[cfg.proxy_name]

        # 2) PLS factor on TRAIN
        factor_m_train, pls_art = build_monthly_pls_factor(
            m_train_pls, proxy_name=cfg.proxy_name, X_cols=list(m_train_X.columns), n_components=cfg.n_components
        )

        # 3) Collapse TRAIN factor to quarterly
        q_factor_train = collapse_monthly_to_quarterly(factor_m_train, method=cfg.collapse_method)
        q_factor_train = q_factor_train.loc[y_train.index]

        # 4) Fit bridge on TRAIN (quarterly)
        bridge = fit_bridge(q_factor_train, y_train, ar_lags=cfg.bridge_ar_lags)

        # ===== NOWCAST quarter t =====
        # Monthly data available up to end of quarter t (mimic ragged by slicing)
        last_nowcast_month = pd.Period(t, freq='Q').end_time
        m_all = monthly_df.loc[monthly_df.index <= last_nowcast_month]

        # Transform monthly X using TRAIN artifacts
        m_all_X = transform_preprocess_X(m_all[X_cols], art)
        m_all_pls = m_all_X.copy()
        m_all_pls[cfg.proxy_name] = m_all[cfg.proxy_name]

        # Project PLS factor for all months seen so far
        factor_m_all = transform_monthly_pls_factor(m_all_pls, pls_art)
        # Collapse to quarterly and take the current quarter's factor (partial avg handles ragged)
        q_factor_all = collapse_monthly_to_quarterly(factor_m_all, method=cfg.collapse_method)
        x_t = q_factor_all.loc[[t]]  # quarter t only (will be NaN if truly no month exists)

        # Predict using bridge; need y history for AR terms
        y_hat_all = predict_bridge(bridge, q_factor_all, y_q)
        y_hat_t = y_hat_all.loc[t]

        # Store
        y_true_all.append(y_q.loc[t])
        y_pred_all.append(y_hat_t)

    y_true = pd.Series(y_true_all, index=q_list[start_idx:], name="y_true")
    y_pred = pd.Series(y_pred_all, index=q_list[start_idx:], name="y_pred")

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    rmse_annualized = float(4 * rmse)  # per user note

    return BacktestResult(y_true=y_true, y_pred=y_pred, rmse=rmse, rmse_annualized=rmse_annualized)
