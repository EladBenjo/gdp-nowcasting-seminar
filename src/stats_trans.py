import numpy as np
import pandas as pd
from typing import Iterable, Optional, Union

def apply_transform_inplace(
    df: pd.DataFrame,
    col_indices: Iterable[Union[int, str]],
    transform: str,
    *,
    season_m: Optional[int] = None,
    detrend_degree: int = 1,
    log_require_positive: bool = True
) -> None:
    """
    Apply a single transformation to selected columns of a DataFrame (in-place),
    using previous-valid logic for differencing to handle irregular gaps.

    Supported transforms:
        - 'log'            : natural log (NaNs preserved)
        - 'diff'           : previous-valid first difference (uses last observed value)
        - 'seasonal_diff'  : previous-valid seasonal difference at lag m
        - 'detrend'        : polynomial detrending over *continuous time* if DatetimeIndex

    Notes:
    - The DataFrame is modified in place. Returns None.
    - NaNs are preserved. Differencing produces NaN when there is no previous valid value.
    - 'log' requires strictly positive values by default. Set `log_require_positive=False`
      only if you have handled non-positives safely upstream.

    Args:
        df: Input DataFrame with numeric columns.
        col_indices: Column *indices or names* to transform.
        transform: One of {'log', 'diff', 'seasonal_diff', 'detrend'}.
        season_m: Seasonal period (e.g., 12 for monthly) for 'seasonal_diff'.
        detrend_degree: Polynomial degree for detrending (>=1).
        log_require_positive: If True, raise on non-positive values for 'log'.

    Raises:
        ValueError: On invalid arguments or unsupported transform.
        IndexError: If a numeric column index is out of bounds.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame.")
    if transform not in {"log", "diff", "seasonal_diff", "detrend"}:
        raise ValueError("transform must be one of {'log','diff','seasonal_diff','detrend'}.")
    if transform == "seasonal_diff" and (season_m is None or season_m <= 1):
        raise ValueError("For 'seasonal_diff', you must provide season_m >= 2.")
    if transform == "detrend" and detrend_degree < 1:
        raise ValueError("detrend_degree must be >= 1 for 'detrend'.")

    # Resolve column names from indices or names; validate bounds
    cols = []
    ncols = df.shape[1]
    for idx in col_indices:
        if isinstance(idx, (int, np.integer)):
            if idx < 0 or idx >= ncols:
                raise IndexError(f"Column index {idx} out of bounds for {ncols} columns.")
            cols.append(df.columns[int(idx)])
        elif isinstance(idx, str):
            if idx not in df.columns:
                raise IndexError(f"Column name '{idx}' not found in DataFrame.")
            cols.append(idx)
        else:
            raise ValueError("col_indices must contain integers (positions) or strings (column names).")

    # Prepare continuous time for detrending if the index is datetime-like
    # We measure time in days since the first timestamp to avoid numeric scaling issues.
    if transform == "detrend":
        if isinstance(df.index, pd.DatetimeIndex):
            # Use elapsed days as a continuous time variable
            t_full = (df.index - df.index[0]).days.astype(float)
        else:
            # Fallback: step index (0..n-1)
            t_full = np.arange(len(df), dtype=float)

    for col in cols:
        s = df[col]

        if transform == "log":
            # Log transform (no differencing)
            if log_require_positive:
                # Ignore NaN when checking positivity
                non_pos_mask = s.notna() & (s <= 0)
                if non_pos_mask.any():
                    bad_idx = s.index[non_pos_mask].tolist()[:3]
                    raise ValueError(
                        f"Column '{col}' contains non-positive values; log undefined. "
                        f"Examples at indices: {bad_idx} (showing up to 3). "
                        f"Handle zeros/negatives or set log_require_positive=False."
                    )
            df[col] = np.log(s)

        elif transform == "diff":
            # Previous-valid first difference:
            # 1) forward-fill to get the last observed value, 2) shift by 1, 3) subtract,
            # 4) re-mask original NaNs so we don't create invented values.
            prev_valid = s.ffill().shift(1)
            out = s - prev_valid
            # Preserve NaNs where either current or previous-valid is missing
            out[s.isna() | prev_valid.isna()] = np.nan
            df[col] = out

        elif transform == "seasonal_diff":
            # Previous-valid seasonal difference at lag m:
            # forward-fill to last observed, then shift by m, then subtract, then re-mask.
            prev_valid_m = s.ffill().shift(season_m)
            out = s - prev_valid_m
            out[s.isna() | prev_valid_m.isna()] = np.nan
            df[col] = out

        elif transform == "detrend":
            # Polynomial detrending over continuous time
            mask = s.notna()
            if mask.sum() <= detrend_degree:
                # Not enough points to fit polynomial; leave as-is
                continue

            # Fit polynomial on observed points only
            t_obs = t_full[mask.values] if isinstance(t_full, np.ndarray) else t_full[mask]
            y_obs = s[mask].astype(float).values

            try:
                coeffs = np.polyfit(t_obs, y_obs, deg=detrend_degree)
                trend_full = np.polyval(coeffs, t_full)
                out = s.astype(float).copy()
                out[mask] = y_obs - trend_full[mask.values if isinstance(t_full, np.ndarray) else mask]
                df[col] = out
            except np.linalg.LinAlgError:
                # Ill-conditioned fit; skip gracefully
                continue

    # In-place; explicit None for clarity
    return None
