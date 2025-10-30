import numpy as np
import pandas as pd
import itertools
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit

# ------------------------------Random-Walk------------------------------
def random_walk_forecast(y: pd.Series, horizon: int = 1, drift: bool = False,
                         drift_window: int | None = None) -> pd.Series:
    """
    Make an h-step-ahead Random Walk forecast.
    """
    y = y.dropna()
    if y.empty:
        raise ValueError("y has no non-NaN observations.")

    last_val = y.iloc[-1]

    # Estimate drift as the mean of first differences over the chosen window
    if drift:
        diffs = y.diff().dropna()
        if diffs.empty:
            delta_hat = 0.0
        else:
            if drift_window is not None:
                diffs = diffs.iloc[-drift_window:]
            delta_hat = float(diffs.mean())
    else:
        delta_hat = 0.0

    # Build future index; will be overwritten during backtest to align to true index
    if getattr(y.index, "freq", None) is not None:
        future_index = pd.date_range(start=y.index[-1] + y.index.freq, periods=horizon, freq=y.index.freq)
    else:
        future_index = pd.RangeIndex(1, horizon + 1, name="h")

    steps = np.arange(1, horizon + 1, dtype=float)
    y_hat = last_val + steps * delta_hat
    return pd.Series(y_hat, index=future_index, name="RW_forecast")


def rolling_origin_backtest(y: pd.Series, horizon: int = 1, drift: bool = False,
                            start: str | pd.Timestamp | int | None = None,
                            min_train: int = 24, drift_window: int | None = None,
                            metric: str = "rmse",
                            return_all_horizons: bool = False) -> tuple[pd.Series, pd.Series, dict]:
    """
    Expanding-window backtest for RW / RW+drift.

    Returns:
      - y_pred: pd.Series of forecasts aligned to the true y index (great for plotting)
      - y_true: pd.Series of actual values aligned to y_pred
      - metrics: dict with overall RMSE/MAE over the aligned points

    Notes:
      * For horizon > 1:
          - If return_all_horizons=False (default): returns only the 1-step-ahead forecasts.
          - If return_all_horizons=True: returns a multi-index Series (index: [timestamp, h]).
    """
    y = y.dropna()
    if isinstance(start, (str, pd.Timestamp)):
        start_idx = y.index.get_loc(pd.to_datetime(start))
    elif isinstance(start, int):
        start_idx = start
    else:
        start_idx = min_train

    pred_pieces = []
    true_pieces = []

    # For multi-horizon optional return
    mh_pred_records = []
    mh_true_records = []

    for t in range(start_idx, len(y) - horizon + 1):
        y_train = y.iloc[:t]                 # expanding train up to t-1
        y_true_h = y.iloc[t:t + horizon]     # true path from t .. t+h-1

        # h-step RW forecast from the information available at time t-1
        y_fcst = random_walk_forecast(y_train, horizon=horizon, drift=drift, drift_window=drift_window)

        # Align forecast index to the true horizon's timestamps (prevents index mismatch)
        y_fcst.index = y_true_h.index

        if return_all_horizons:
            # Store all horizons with an extra 'h' level
            for i, (ts, val) in enumerate(y_fcst.items(), start=1):
                mh_pred_records.append(((ts, i), val))
                mh_true_records.append(((ts, i), y_true_h.loc[ts]))
        else:
            # Keep only the first step-ahead (common choice for evaluation/plotting)
            pred_pieces.append(y_fcst.iloc[:1])
            true_pieces.append(y_true_h.iloc[:1])

    if return_all_horizons:
        if not mh_pred_records:
            raise ValueError("Not enough data to run backtest at the chosen horizon.")
        # Build multi-index Series: index = (timestamp, h)
        idx_pred, vals_pred = zip(*mh_pred_records)
        idx_true, vals_true = zip(*mh_true_records)
        y_pred = pd.Series(vals_pred, index=pd.MultiIndex.from_tuples(idx_pred, names=["time", "h"]))
        y_true = pd.Series(vals_true, index=pd.MultiIndex.from_tuples(idx_true, names=["time", "h"]))
    else:
        if not pred_pieces:
            raise ValueError("Not enough data to run backtest at the chosen horizon.")
        # Concatenate first-step-ahead forecasts across all folds
        y_pred = pd.concat(pred_pieces).sort_index()
        y_true = pd.concat(true_pieces).sort_index()

    # Compute metrics on the aligned points
    aligned = y_true.index.intersection(y_pred.index)
    err = (y_pred.loc[aligned].values - y_true.loc[aligned].values)
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))

    metrics = {"rmse": rmse, "mae": mae, "n_points": int(len(aligned))}

    return y_pred, y_true, metrics


# ------------------------------Arima------------------------------
def fit_sarimax(y_train: pd.Series, exog: pd.Series | None = None, order=(1,0,1), seasonal_order=(0,0,0,0), trend='c'):
    """
    Fit a SARIMAX model on a stationary target (e.g., log-diff GDP).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            y_train,
            exog=exog,
            order=order,
            seasonal_order=seasonal_order,
            trend=trend,
            enforce_stationarity=True,
            enforce_invertibility=True,
        )
        res = model.fit(disp=False)
    return res

def select_order_aic(y_train: pd.Series,
                     exog: pd.Series | None = None,
                     p_max=3, q_max=3,
                     seasonal_order=(0,0,0,0),
                     trend='c'):
    """
    Small AIC grid-search over (p,q) with d=0 (already differenced).
    """
    best = None
    for p in range(p_max+1):
        for q in range(q_max+1):
            if p==0 and q==0:
                continue  # avoid white-noise unless you want it as a baseline
            try:
                res = fit_sarimax(y_train, exog=exog, order=(p,0,q),
                                  seasonal_order=seasonal_order, trend=trend)
                aic = res.aic
                if (best is None) or (aic < best["aic"]):
                    best = {"order": (p,0,q), "aic": aic}
            except Exception:
                continue
    # Fallback if all failed
    return best["order"] if best else (1,0,0)

def sarimax_rolling_backtest(y: pd.Series,
                             exog: pd.Series | pd.DataFrame | None = None,
                             horizon: int = 1,
                             start: int | str | pd.Timestamp | None = None,
                             min_train: int = 24,
                             p_max=3, q_max=3,
                             seasonal_order=(0,0,0,0),
                             trend='c'):
    """
    Expanding-window backtest for SARIMAX on a stationary series (e.g., log-diff GDP).
    If exog is provided, it is sliced per-fold for both training and h-step forecast.

    Returns:
        y_pred: pd.Series of 1-step-ahead forecasts aligned to true timestamps
        y_true: pd.Series of realized values (same index as y_pred)
        metrics: dict with RMSE, MAE, n_points
    """
    # --- Basic sanitation ---
    y = y.dropna()
    if exog is not None:
        # Ensure DataFrame shape and align index universe to y
        if isinstance(exog, pd.Series):
            exog = exog.to_frame(name=getattr(exog, "name", "exog"))
        exog = exog.copy()
        # Reindex exog to y's index to avoid accidental misalignment
        exog = exog.reindex(y.index)

    # --- Resolve start index ---
    if isinstance(start, (str, pd.Timestamp)):
        start_idx = y.index.get_loc(pd.to_datetime(start))
    elif isinstance(start, int):
        start_idx = start
    else:
        start_idx = min_train

    preds, trues = [], []

    # Optional: cache best order after first fold to speed up
    cached_order = None

    for t in range(start_idx, len(y) - horizon + 1):
        # Train window up to (but not including) t
        y_train = y.iloc[:t]
        # Forecast target window [t, t+horizon)
        y_true_h = y.iloc[t:t+horizon]

        # --- Slice exog for this fold (train + forecast), if provided ---
        if exog is not None:
            exog_train = exog.loc[y_train.index]
            exog_fcst  = exog.loc[y_true_h.index]

            # If any missing exog in train/forecast, skip this fold to avoid statsmodels errors.
            if exog_train.isnull().any().any() or exog_fcst.isnull().any().any():
                # You may switch this to an imputation strategy if preferred.
                continue
        else:
            exog_train = None
            exog_fcst  = None

        # --- Order selection (AIC) ---
        if cached_order is None:
            order = select_order_aic(y_train,
                                     exog_train,
                                     p_max=p_max, q_max=q_max,
                                     seasonal_order=seasonal_order,
                                     trend=trend)
            cached_order = order
        else:
            order = cached_order

        # --- Fit on expanding train window ---
        res = fit_sarimax(y_train,
                          exog_train,
                          order=order,
                          seasonal_order=seasonal_order,
                          trend=trend)

        # --- h-step-ahead forecast ---
        # IMPORTANT: supply exog for the *forecast horizon* when the model has exog
        fcst = res.get_forecast(steps=horizon, exog=exog_fcst).predicted_mean
        fcst.index = y_true_h.index  # align timestamps

        # Keep only 1-step-ahead error for scoring (like RW baseline)
        preds.append(fcst.iloc[:1])
        trues.append(y_true_h.iloc[:1])

    if not preds:
        raise ValueError("Not enough usable folds (likely due to exog gaps). Consider imputing exog or adjusting start/min_train.")

    y_pred = pd.concat(preds).sort_index()
    y_true = pd.concat(trues).sort_index()

    err = (y_pred.values - y_true.values)
    rmse = float(np.sqrt(np.mean(err**2)))
    mae  = float(np.mean(np.abs(err)))
    metrics = {"rmse": rmse, "mae": mae, "n_points": int(len(y_pred))}

    return y_pred, y_true, metrics


# ------------------------------XGB------------------------------
def build_quarterly_features(y_q: pd.Series,
                             F_q: pd.Series | pd.DataFrame | None = None,
                             lags_y=(1,2,4),
                             lags_F=(0,1,2)):
    """
    Build lagged features for quarterly nowcast.
    - y_q: quarterly target (stationary transform if used).
    - F_q: quarterly exogenous factor(s) already bridged from monthly (ragged-edge aware).
           If DataFrame: multiple exog columns.
    Returns X, y aligned with valid rows (no NA in required lags).
    """
    df = pd.DataFrame({"y": y_q}).copy()
    # y lags
    for L in lags_y:
        df[f"y_lag{L}"] = df["y"].shift(L)

    # exog factor(s)
    if F_q is not None:
        if isinstance(F_q, pd.Series):
            F_q = F_q.to_frame("F")
        for col in F_q.columns:
            df[col] = F_q[col]
            for L in lags_F:
                if L == 0:
                    df[f"{col}_t"] = F_q[col]  # contemporaneous quarterly factor (built only from months available)
                else:
                    df[f"{col}_lag{L}"] = F_q[col].shift(L)

    # optional seasonal dummies (quarters)
    df["qtr"] = df.index.quarter
    X = pd.get_dummies(df.drop(columns=["y"]), columns=["qtr"], drop_first=True)

    # drop rows with any NA in features or target
    valid = X.notna().all(axis=1) & df["y"].notna()
    return X.loc[valid], df["y"].loc[valid]

def xgb_rolling_backtest(y_q: pd.Series,
                         F_q: pd.Series | pd.DataFrame | None = None,
                         start: int | str | pd.Timestamp | None = None,
                         min_train: int = 24,
                         horizon: int = 1):
    """
    Walk-forward 1-step-ahead backtest for XGBoost on quarterly target.
    Assumes F_q already respects real-time availability (no leakage).
    """
    y_q = y_q.dropna()
    if isinstance(start, (str, pd.Timestamp)):
        start_idx = y_q.index.get_loc(pd.to_datetime(start))
    elif isinstance(start, int):
        start_idx = start
    else:
        start_idx = min_train

    # Prebuild feature matrix (lags cause initial NA rows; real-time handled upstream in F_q)
    X_all, y_all = build_quarterly_features(y_q, F_q)

    preds, trues = [], []
    # Hyperparams: fix once (optionally tuned via early time-split)
    model_params = dict(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42
    )

    # Align start_idx to first valid row in X_all
    start_idx = max(start_idx, X_all.index.get_loc(X_all.index[0]))

    for t in range(start_idx, len(y_q)):
        # Predict y_t -> train uses rows strictly before t
        cutoff_time = y_q.index[t]  # timestamp for y_t

        # Train set: indices < cutoff_time and present in X_all
        train_mask = X_all.index < cutoff_time
        test_mask  = X_all.index == cutoff_time

        if not train_mask.any() or not test_mask.any():
            continue

        X_train, y_train = X_all.loc[train_mask], y_all.loc[train_mask]
        X_test,  y_true  = X_all.loc[test_mask], y_all.loc[test_mask]

        # Skip fold if any NA slipped in (e.g., missing F at ragged edge)
        if X_train.isnull().any().any() or X_test.isnull().any().any():
            continue

        model = XGBRegressor(**model_params)
        model.fit(X_train, y_train)

        y_hat = pd.Series(model.predict(X_test), index=X_test.index)
        preds.append(y_hat)
        trues.append(y_true)

    if not preds:
        raise ValueError("No usable folds (check feature availability / start index).")

    y_pred = pd.concat(preds).sort_index()
    y_true = pd.concat(trues).sort_index()

    err = (y_pred.values - y_true.values)
    rmse = float(np.sqrt(np.mean(err**2)))
    mae  = float(np.mean(np.abs(err)))
    metrics = {"rmse": rmse, "mae": mae, "n_points": int(len(y_pred))}

    return y_pred, y_true, metrics

# ------------------------------MIDAS------------------------------
import numpy as np
import pandas as pd
from typing import List, Dict, Sequence, Tuple, Optional
from dataclasses import dataclass

# Assumes you have midas_beta.BetaMIDASRegressor in your path
from midas_beta import BetaMIDASRegressor


# =========================
# Helper: monthly → quarterly lag-matrix builder
# =========================
def build_monthly_lag_matrix_for_quarterly_target(
    y_q: pd.Series,
    x_m: pd.Series,
    K: int,
    agg: str = "end_inclusive"
) -> Tuple[np.ndarray, pd.Index]:
    """
    Build (T, K) lag matrix of monthly regressor aligned to quarterly target.
    For each quarter t, take the last K months up to the quarter's end (inclusive).
    Rows with insufficient history are dropped.

    Args:
        y_q: Quarterly target with DatetimeIndex at quarter-end (e.g., '2024-12-31').
        x_m: Monthly regressor with month-end DatetimeIndex.
        K:   Number of monthly lags to include (1..K), where lag=1 is the most recent month in the quarter.
        agg: Placeholder for future variants; currently only 'end_inclusive'.

    Returns:
        X_mat: np.ndarray shape (T_valid, K)
        idx_valid: pd.Index of quarters kept (aligned to X_mat rows).
    """
    # Ensure proper frequency alignment (quarter-end and month-end)
    y_q = y_q.copy()
    x_m = x_m.copy()

    # Make sure indices are sorted
    y_q = y_q.sort_index()
    x_m = x_m.sort_index()

    rows = []
    keep_idx = []

    for q_end in y_q.index:
        # Identify the three months in this quarter
        q = pd.Period(q_end, freq="Q")
        months_in_q = [pd.Period(q.start_time, freq="M"),
                       pd.Period(q.start_time + pd.offsets.MonthEnd(1), freq="M"),
                       pd.Period(q.end_time, freq="M")]

        # The most recent month inside the quarter is the quarter-end month
        last_month = months_in_q[-1].to_timestamp(how="end")

        # Collect K months ending at last_month: [last_month, last_month-1M, ...]
        month_ends = pd.date_range(end=last_month, periods=K, freq="ME")
        vals = x_m.reindex(month_ends)

        if vals.isna().any():
            # Not enough history or gaps; skip this quarter
            continue

        # Arrange as lag order: lag1 = most recent month, lagK = oldest
        rows.append(vals.values[::-1][::-1])  # explicitly keep order [m1,...,mK]
        keep_idx.append(q_end)

    if not rows:
        return np.empty((0, K)), pd.Index([])

    X_mat = np.vstack(rows)  # (T_valid, K)
    idx_valid = pd.Index(keep_idx)

    return X_mat, idx_valid


# =========================
# Rolling backtest for Beta MIDAS
# =========================
@dataclass
class MIDASBacktestResult:
    y_pred: pd.Series
    y_true: pd.Series
    metrics: Dict[str, float]


def midas_beta_rolling_backtest(
    y_q: pd.Series,
    monthly_regressors: Dict[str, pd.Series],
    K_list: Dict[str, int],
    *,
    start: int | str | pd.Timestamp | None = None,
    min_train: int = 24,
    ridge: float = 0.0,
    add_intercept: bool = True,
    random_state: int = 0
) -> MIDASBacktestResult:
    """
    Expanding-window 1-step-ahead backtest for Beta-MIDAS.

    Args:
        y_q: Quarterly target (stationary transform if used). DatetimeIndex at quarter-end.
        monthly_regressors: dict {name: monthly Series}, month-end DatetimeIndex.
        K_list: dict {name: K} number of monthly lags for each regressor.
        start: first forecast index (pos or date). If None -> min_train.
        min_train: minimal number of quarterly observations to start.
        ridge: L2 penalty on final linear coefficients (not on Beta-weights).
        add_intercept: include intercept term.
        random_state: optimizer init seed.

    Returns:
        MIDASBacktestResult with y_pred, y_true (aligned), and metrics.
    """
    y_q = y_q.dropna().sort_index()

    # Resolve start index
    if isinstance(start, (str, pd.Timestamp)):
        start_idx = y_q.index.get_loc(pd.to_datetime(start))
    elif isinstance(start, int):
        start_idx = start
    else:
        start_idx = min_train

    # Pre-build lag matrices per regressor over the whole sample
    X_mats: Dict[str, np.ndarray] = {}
    idx_common: Optional[pd.Index] = None

    for name, xm in monthly_regressors.items():
        K = int(K_list[name])
        Xi, idx_i = build_monthly_lag_matrix_for_quarterly_target(y_q, xm, K=K)

        # Align by intersecting indices across all regressors
        if idx_common is None:
            idx_common = idx_i
        else:
            idx_common = idx_common.intersection(idx_i)

        X_mats[name] = (Xi, idx_i)

    if idx_common is None or len(idx_common) == 0:
        raise ValueError("No valid quarters with complete lag windows across regressors.")

    # Align y and all Xi to the common index
    y_aligned = y_q.reindex(idx_common).dropna()
    # Re-check in case target has gaps
    idx_common = y_aligned.index

    # Build final aligned X_list with shapes (T, K_i)
    X_list_full: List[np.ndarray] = []
    for name, (Xi_raw, idx_i) in X_mats.items():
        Xi = pd.DataFrame(Xi_raw, index=idx_i).reindex(idx_common).values
        if np.isnan(Xi).any():
            raise ValueError(f"Regressor '{name}' has missing rows after alignment.")
        X_list_full.append(Xi)

    # Rolling origin
    preds, trues = [], []

    # Ensure start not before first valid row
    start_idx = max(start_idx, 1)  # need at least one obs before first forecast
    for t in range(start_idx, len(idx_common)):
        cutoff_time = idx_common[t]  # forecast y at this quarter
        # Train uses rows strictly before cutoff
        y_train = y_aligned.iloc[:t].values
        X_train_list = [Xi[:t, :] for Xi in X_list_full]

        # Skip if too short
        if len(y_train) < min_train:
            continue

        # Fit MIDAS on training window
        model = BetaMIDASRegressor(add_intercept=add_intercept, ridge=ridge, random_state=random_state)
        model.fit(y_train, X_train_list)

        # Prepare one-step-ahead (contemporaneous) design at cutoff_time
        X_test_list = [Xi[t:t+1, :] for Xi in X_list_full]  # shape (1, K_i) per regressor

        # Predict y_t
        y_hat_t = float(model.predict(X_test_list).ravel()[0])

        preds.append(pd.Series([y_hat_t], index=[cutoff_time]))
        trues.append(pd.Series([y_aligned.loc[cutoff_time]], index=[cutoff_time]))

    if not preds:
        raise ValueError("No usable folds — check start/min_train/K_list and data coverage.")

    y_pred = pd.concat(preds).sort_index()
    y_true = pd.concat(trues).sort_index()

    err = y_pred.values - y_true.values
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    metrics = {"rmse": rmse, "mae": mae, "n_points": int(len(y_pred))}

    return MIDASBacktestResult(y_pred=y_pred, y_true=y_true, metrics=metrics)

