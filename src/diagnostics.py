import numpy as np
import pandas as pd
import warnings
from typing import Optional, Union, Dict, Any
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.seasonal import STL
from statsmodels.tools.sm_exceptions import InterpolationWarning


def _dropna_series(s: pd.Series, mode: str = "any") -> pd.Series:
    """
    Drop missing values from a Series according to the chosen mode.

    Args:
        s (pd.Series): Input series.
        mode (str): 'any' -> drop all NaNs; 'all' -> drop only if all are NaN; 'none' -> do nothing.

    Returns:
        pd.Series: Cleaned series.
    """
    if mode == "any":
        return s.dropna()
    elif mode == "all":
        return s if s.notna().any() else s.dropna()
    elif mode == "none":
        return s
    else:
        raise ValueError("dropna must be one of {'any','all','none'}.")


def _is_constant(s: pd.Series) -> bool:
    """Return True if the series has (near) zero variance."""
    s_clean = s.dropna()
    if len(s_clean) <= 1:
        return True
    return np.isclose(np.nanvar(s_clean.values), 0.0)


def _seasonality_strength_stl(s: pd.Series, period: int) -> float:
    """
    Estimate seasonality strength using STL decomposition.

    Definition (Hyndman style):
        strength = max(0, 1 - Var(resid) / Var(seasonal + resid))

    Args:
        s (pd.Series): Input series (no NaNs).
        period (int): Seasonal period (e.g., 12 for monthly, 4 for quarterly).

    Returns:
        float: Seasonality strength in [0, 1] (NaN if not enough data).
    """
    if period is None or period <= 1 or len(s) < period * 2:
        return np.nan
    try:
        stl = STL(s.values, period=period, robust=True)
        res = stl.fit()
        denom = np.var(res.seasonal + res.resid)
        if np.isclose(denom, 0.0):
            return 0.0
        strength = 1.0 - (np.var(res.resid) / denom)
        return float(np.clip(strength, 0.0, 1.0))
    except Exception:
        return np.nan


def _suggest_transform(
    s: pd.Series,
    adf_p: Optional[float],
    kpss_p: Optional[float],
    alpha_adf: float,
    alpha_kpss: float,
    season_m: Optional[int],
    log_positive_only: bool,
    seasonality_strength: Optional[float],
    seasonality_threshold: float
) -> str:
    """
    Heuristic suggestion for transformation based on ADF/KPSS and seasonality strength.
    """
    # Handle missing results
    if adf_p is None and kpss_p is None:
        return "none"

    adf_stationary = (adf_p is not None) and (adf_p < alpha_adf)
    kpss_stationary = (kpss_p is not None) and (kpss_p > alpha_kpss)

    # Positive series?
    s_clean = s.dropna()
    positive_only = (s_clean > 0).all() if len(s_clean) else False

    # Seasonality flag
    has_seasonality = (seasonality_strength is not None) and (seasonality_strength >= seasonality_threshold)
    seasonal_tag = f"seasonal_diff({season_m})" if has_seasonality and season_m and season_m > 1 else None

    # Base decision (trend/unit-root)
    if adf_stationary and kpss_stationary:
        base = "none"
    elif (not adf_stationary) and (not kpss_stationary):
        base = "diff"
    elif (not adf_stationary) and kpss_stationary:
        base = "diff"
    elif adf_stationary and (not kpss_stationary):
        base = "detrend"
    else:
        base = "diff" if not adf_stationary else "none"

    # Prefer logdiff if strictly positive and we would otherwise diff
    if base == "diff" and positive_only and log_positive_only:
        base = "logdiff"

    # Add seasonal differencing if strong seasonality detected
    if seasonal_tag:
        if base == "none":
            return seasonal_tag
        else:
            return f"{base}+{seasonal_tag}"
    return base


def run_stationarity_panel(
    df: pd.DataFrame,
    regression_adf: str = "c",
    autolag_adf: Optional[str] = "AIC",
    kpss_regression: str = "c",
    nlags_kpss: Union[str, int] = "auto",
    min_obs: int = 20,
    dropna: str = "any",
    season_m: Optional[int] = None,
    seasonality_threshold: float = 0.30,
    log_positive_only: bool = True,
    alpha_adf: float = 0.05,
    alpha_kpss: float = 0.05
) -> pd.DataFrame:
    """
    Run ADF and KPSS tests on each numeric column, plus seasonality detection via STL.

    Adds:
        - 'seasonality_strength' in [0,1] (NaN if undetermined)
        - 'has_seasonality' boolean flag based on 'seasonality_threshold'
        - 'suggestion' incorporates seasonal differencing when warranted

    Args:
        df (pd.DataFrame): Wide DataFrame of time series (columns = variables).
        regression_adf (str): Trend option for ADF: {'c','ct','ctt','nc'}.
        autolag_adf (Optional[str]): {'AIC','BIC','t-stat',None}.
        kpss_regression (str): Trend option for KPSS: {'c','ct'}.
        nlags_kpss (Union[str,int]): 'auto' or integer.
        min_obs (int): Minimum observations to run tests.
        dropna (str): {'any','all','none'} NA handling per column.
        season_m (Optional[int]): Seasonal period (e.g., 12 monthly, 4 quarterly).
        seasonality_threshold (float): Threshold to flag seasonality strength.
        log_positive_only (bool): Suggest log-based diffs only when data > 0.
        alpha_adf (float): Significance threshold for ADF.
        alpha_kpss (float): Significance threshold for KPSS.

    Returns:
        pd.DataFrame: Summary with stats, decisions, seasonality, and suggestions.
    """
    num_cols = df.select_dtypes(include="number").columns.tolist()
    results: Dict[str, Dict[str, Any]] = {}

    for col in num_cols:
        s_raw = df[col]
        s = _dropna_series(s_raw, dropna)

        rec: Dict[str, Any] = {
            "n_eff": int(s.notna().sum()),
            "adf_stat": None, "adf_p": None, "adf_lags": None,
            "adf_crit_1": None, "adf_crit_5": None, "adf_crit_10": None,
            "kpss_stat": None, "kpss_p": None, "kpss_lags": None,
            "kpss_crit_1": None, "kpss_crit_5": None, "kpss_crit_10": None,
            "seasonality_strength": np.nan, "has_seasonality": None,
            "decision": None, "suggestion": None, "notes": ""
        }

        if rec["n_eff"] < min_obs:
            rec["decision"] = "too_short"
            rec["suggestion"] = "none"
            results[col] = rec
            continue

        if _is_constant(s):
            rec["decision"] = "constant"
            rec["suggestion"] = "none"
            results[col] = rec
            continue

        # --- Seasonality (STL) ---
        seas_strength = _seasonality_strength_stl(s, season_m) if season_m else np.nan
        rec["seasonality_strength"] = seas_strength
        rec["has_seasonality"] = bool(seas_strength >= seasonality_threshold) if np.isfinite(seas_strength) else False
        if np.isfinite(seas_strength):
            rec["notes"] += f"Seasonality strength={seas_strength:.2f}. "

        # --- ADF ---
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=InterpolationWarning)
                adf_out = adfuller(
                    s.values.astype(float),
                    regression=regression_adf,
                    autolag=autolag_adf
                )
            rec["adf_stat"] = float(adf_out[0])
            rec["adf_p"] = float(adf_out[1])
            rec["adf_lags"] = int(adf_out[2])
            crit = adf_out[4]
            rec["adf_crit_1"] = float(crit.get("1%", np.nan))
            rec["adf_crit_5"] = float(crit.get("5%", np.nan))
            rec["adf_crit_10"] = float(crit.get("10%", np.nan))
        except Exception as e:
            rec["notes"] += f"ADF error: {e}. "

        # --- KPSS ---
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=InterpolationWarning)
                kpss_out = kpss(
                    s.values.astype(float),
                    regression=kpss_regression,
                    nlags=nlags_kpss
                )
            rec["kpss_stat"] = float(kpss_out[0])
            rec["kpss_p"] = float(kpss_out[1])
            rec["kpss_lags"] = int(kpss_out[2])
            crit = kpss_out[3]
            rec["kpss_crit_1"] = float(crit.get("1%", np.nan))
            rec["kpss_crit_5"] = float(crit.get("5%", np.nan))
            rec["kpss_crit_10"] = float(crit.get("10%", np.nan))
        except Exception as e:
            rec["notes"] += f"KPSS error: {e}. "

        # --- Decision logic ---
        adf_p = rec["adf_p"]
        kpss_p = rec["kpss_p"]
        if (adf_p is None) and (kpss_p is None):
            rec["decision"] = "inconclusive"
            base_suggestion = "diff"
        else:
            adf_stationary = (adf_p is not None) and (adf_p < alpha_adf)
            kpss_stationary = (kpss_p is not None) and (kpss_p > alpha_kpss)

            if adf_stationary and kpss_stationary:
                rec["decision"] = "stationary"
            elif (not adf_stationary) and (not kpss_stationary):
                rec["decision"] = "nonstationary"
            else:
                rec["decision"] = "inconclusive"

            base_suggestion = None  # will be filled by _suggest_transform

        rec["suggestion"] = _suggest_transform(
            s=s,
            adf_p=adf_p,
            kpss_p=kpss_p,
            alpha_adf=alpha_adf,
            alpha_kpss=alpha_kpss,
            season_m=season_m,
            log_positive_only=log_positive_only,
            seasonality_strength=rec["seasonality_strength"],
            seasonality_threshold=seasonality_threshold
        )

        results[col] = rec

    out = pd.DataFrame.from_dict(results, orient="index").reset_index().rename(columns={"index": "variable"})
    cols_order = [
        "variable", "n_eff",
        "adf_stat", "adf_p", "adf_lags", "adf_crit_1", "adf_crit_5", "adf_crit_10",
        "kpss_stat", "kpss_p", "kpss_lags", "kpss_crit_1", "kpss_crit_5", "kpss_crit_10",
        "seasonality_strength", "has_seasonality",
        "decision", "suggestion", "notes"
    ]
    out = out[cols_order]
    return out

def _significant_lags(confint: np.ndarray) -> np.ndarray:
    """
    Return boolean mask for lags whose confidence interval excludes zero.

    Args:
        confint (np.ndarray): Confidence intervals array of shape (nlags+1, 2)
                              as returned by statsmodels (includes lag 0).
    Returns:
        np.ndarray: Boolean array (nlags+1,) where True indicates significance.
    """
    # A lag is significant if 0 is not inside [lower, upper]
    lower = confint[:, 0]
    upper = confint[:, 1]
    return (lower > 0) | (upper < 0)


def _suggest_ar_from_pacf(sig_pacf_mask: np.ndarray, max_order: int = 2) -> int:
    """
    Heuristic AR order suggestion from PACF significance pattern.
    - If PACF(1) significant and PACF(2) not -> AR(1)
    - If PACF(1) and PACF(2) significant -> AR(2)
    - Else -> AR(0)

    Args:
        sig_pacf_mask (np.ndarray): Boolean significance mask including lag 0.
        max_order (int): Maximum AR order to consider (1 or 2).

    Returns:
        int: Suggested AR order in {0, 1, 2}.
    """
    # Ensure we have at least lags 1..max_order
    L1 = bool(len(sig_pacf_mask) > 1 and sig_pacf_mask[1])
    L2 = bool(len(sig_pacf_mask) > 2 and sig_pacf_mask[2])

    if max_order >= 2 and L1 and L2:
        return 2
    if L1:
        return 1
    return 0


def _has_seasonal_peaks(sig_acf_mask: np.ndarray, season_m: Optional[int]) -> bool:
    """
    Detect seasonal peaks in ACF at multiples of season_m.

    Args:
        sig_acf_mask (np.ndarray): Boolean significance mask including lag 0.
        season_m (Optional[int]): Seasonal period (e.g., 12 for monthly, 5/7 for daily market).

    Returns:
        bool: True if a seasonal peak is detected at lag m or 2m (when available).
    """
    if not season_m or season_m <= 1:
        return False
    flags = []
    for k in (season_m, 2 * season_m):
        if k < len(sig_acf_mask):
            flags.append(bool(sig_acf_mask[k]))
    return any(flags)


def analyze_autocorrelation_panel(
    df: pd.DataFrame,
    max_lag: int = 12,
    alpha: float = 0.05,
    season_m: Optional[int] = None,
    pacf_method: str = "ywm",
    acf_fft: bool = True
) -> pd.DataFrame:
    """
    Analyze ACF/PACF per column and suggest idiosyncratic AR(order) for DFM.

    Pipeline assumptions:
    - Run this AFTER your stationarity/seasonal transforms (e.g., log-diff, seasonal diff).
    - Input must be numeric columns; NaNs are dropped per column (pairwise).

    Args:
        df (pd.DataFrame): Wide DataFrame; each column is a (transformed) series.
        max_lag (int): Max lag for ACF/PACF computation.
        alpha (float): Significance level for confidence intervals.
        season_m (Optional[int]): Seasonal period for seasonal-peak detection.
        pacf_method (str): PACF method ('ywm', 'ywunbiased', 'ols', etc.); 'ywm' is robust.
        acf_fft (bool): Whether to use FFT in ACF computation.

    Returns:
        pd.DataFrame: One row per variable with:
            - n_eff
            - acf1, pacf1
            - first_sig_acf_lag, first_sig_pacf_lag
            - has_seasonal_peaks (bool)
            - suggest_ar (0/1/2)
    """
    results: Dict[str, Dict[str, Any]] = {}
    for col in df.select_dtypes(include="number").columns:
        s = df[col].dropna()
        rec: Dict[str, Any] = {
            "variable": col,
            "n_eff": int(s.shape[0]),
            "acf1": np.nan,
            "pacf1": np.nan,
            "first_sig_acf_lag": np.nan,
            "first_sig_pacf_lag": np.nan,
            "has_seasonal_peaks": False,
            "suggest_ar": 0,
        }
        if rec["n_eff"] <= 3:
            results[col] = rec
            continue

        # ACF with confints
        acf_vals, acf_conf = acf(
            s.values,
            nlags=max_lag,
            fft=acf_fft,
            alpha=alpha
        )
        acf_sig = _significant_lags(acf_conf)
        rec["acf1"] = float(acf_vals[1]) if len(acf_vals) > 1 else np.nan
        # First significant lag > 0
        sig_acf_lags = np.where(acf_sig)[0]
        sig_acf_lags = sig_acf_lags[sig_acf_lags > 0]
        rec["first_sig_acf_lag"] = int(sig_acf_lags[0]) if len(sig_acf_lags) else np.nan

        # PACF with confints
        pacf_vals, pacf_conf = pacf(
            s.values,
            nlags=max_lag,
            method=pacf_method,
            alpha=alpha
        )
        pacf_sig = _significant_lags(pacf_conf)
        rec["pacf1"] = float(pacf_vals[1]) if len(pacf_vals) > 1 else np.nan
        sig_pacf_lags = np.where(pacf_sig)[0]
        sig_pacf_lags = sig_pacf_lags[sig_pacf_lags > 0]
        rec["first_sig_pacf_lag"] = int(sig_pacf_lags[0]) if len(sig_pacf_lags) else np.nan

        # Seasonal peaks
        rec["has_seasonal_peaks"] = _has_seasonal_peaks(acf_sig, season_m)

        # AR order suggestion from PACF cut-off pattern (up to AR(2))
        rec["suggest_ar"] = int(_suggest_ar_from_pacf(pacf_sig, max_order=2))

        results[col] = rec

    out = pd.DataFrame(results).T.reset_index(drop=True)
    # Nice ordering
    cols = [
        "variable", "n_eff",
        "acf1", "pacf1",
        "first_sig_acf_lag", "first_sig_pacf_lag",
        "has_seasonal_peaks", "suggest_ar"
    ]
    return out[cols]


def summarize_dfm_orders(panel_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Aggregate per-series AR suggestions into global DFM choices.

    Heuristics:
    - error_order: choose the mode of suggest_ar; if tie, take the smaller order.
    - factor_order: if the majority acf1 > 0.2 (or first_sig_acf_lag == 1 often), set 1; else 0.

    Args:
        panel_df (pd.DataFrame): Output of analyze_autocorrelation_panel.

    Returns:
        Dict[str, Any]: {"error_order": int, "factor_order": int, "distribution": dict}
    """
    # Distribution of suggested AR
    dist = panel_df["suggest_ar"].value_counts().sort_index()
    # Mode (tie -> smallest)
    error_order = int(dist.idxmax()) if not dist.empty else 0

    # Factor order heuristic from acf1 across series
    acf1_vals = panel_df["acf1"].dropna().values
    if len(acf1_vals) == 0:
        factor_order = 0
    else:
        # If many series still show positive lag-1 autocorrelation, use AR(1) factor
        fraction_pos_strong = np.mean(acf1_vals > 0.2)
        factor_order = 1 if fraction_pos_strong >= 0.5 else 0

    return {
        "error_order": error_order,
        "factor_order": factor_order,
        "distribution": dist.to_dict()
    }
